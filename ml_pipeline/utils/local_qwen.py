"""Cached Transformers Qwen client with OpenAI-compatible completions."""

import base64
import copy
import gc
import io
import json
import re
import threading
import time
import uuid
from functools import lru_cache
from types import SimpleNamespace

_MODEL_LOCK = threading.RLock()
_ACTIVE_CLIENT = None


def parse_tool_calls(text, tools):
    schemas = {tool["function"]["name"]: tool["function"].get("parameters", {}) for tool in tools}
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if text.count("<tool_call>") != len(blocks):
        raise ValueError("Incomplete local tool call")
    calls = []
    for block in blocks:
        function = re.fullmatch(r"\s*<function=([^>]+)>(.*?)</function>\s*", block, re.DOTALL)
        if not function or function[1] not in schemas:
            raise ValueError("Unknown or malformed local tool call")
        arguments = {}
        properties = schemas[function[1]].get("properties", {})
        parameters = re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", function[2], re.DOTALL)
        if re.sub(r"<parameter=[^>]+>.*?</parameter>", "", function[2], flags=re.DOTALL).strip():
            raise ValueError("Malformed local tool parameters")
        for name, value in parameters:
            if name not in properties or name in arguments:
                raise ValueError("Unknown or repeated local tool parameter")
            value = value.strip()
            if properties[name].get("type") != "string":
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    if properties[name].get("type") in ("number", "integer", "boolean", "array", "object"):
                        raise ValueError(f"Invalid local tool parameter: {name}")
            arguments[name] = value
        if set(schemas[function[1]].get("required", [])) - arguments.keys():
            raise ValueError("Missing required local tool parameter")
        calls.append({"id": f"call_{uuid.uuid4().hex}", "type": "function",
                      "function": {"name": function[1], "arguments": json.dumps(arguments)}})
    return re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip(), calls


class LocalQwenClient:
    def __init__(self, model_id, revision, device=None, quantization="none"):
        if quantization not in ("none", "4bit"):
            raise ValueError("Local quantization must be none or 4bit")
        self.model_id, self.revision = model_id, revision
        self.device, self.quantization = device, quantization
        self.model = self.processor = None
        completions = SimpleNamespace(create=self.create)
        completions.with_raw_response = SimpleNamespace(create=self.create_raw_response)
        self.chat = SimpleNamespace(completions=completions)

    def create_raw_response(self, **kwargs):
        response = self.create(**kwargs)
        return SimpleNamespace(parse=lambda: response, headers={})

    def load(self):
        with _MODEL_LOCK:
            self._load()

    def _load(self):
        global _ACTIVE_CLIENT
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if _ACTIVE_CLIENT is not None and _ACTIVE_CLIENT is not self:
            _ACTIVE_CLIENT.model = _ACTIVE_CLIENT.processor = None
            _ACTIVE_CLIENT = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        settings = {"revision": self.revision, "dtype": torch.bfloat16,
                    "device_map": device, "attn_implementation": "sdpa"}
        if self.quantization == "4bit":
            if not device.startswith("cuda"):
                raise ValueError("Local 4-bit inference requires a CUDA GPU and bitsandbytes")
            from transformers import BitsAndBytesConfig
            settings["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **settings).eval()
        _ACTIVE_CLIENT = self

    def create(self, *, messages, model=None, max_tokens=8192, max_completion_tokens=None,
               extra_body=None, tools=None, stream=False, **kwargs):
        if stream:
            raise ValueError("Local streaming is not supported")
        if model and model != self.model_id:
            raise ValueError("Local request model differs from loaded model")
        from PIL import Image
        import torch
        from openai.types.chat import ChatCompletion

        converted = copy.deepcopy(messages)
        for message in converted:
            for call in message.get("tool_calls", []):
                if isinstance(call["function"].get("arguments"), str):
                    call["function"]["arguments"] = json.loads(call["function"]["arguments"])
            if not isinstance(message.get("content"), list):
                continue
            for index, item in enumerate(message["content"]):
                if item.get("type") != "image_url":
                    continue
                url = item["image_url"]["url"]
                if not url.startswith("data:image/") or ";base64," not in url:
                    raise ValueError("Local images must be supplied as base64 data URLs")
                with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1], validate=True))) as image:
                    pixels = image.convert("RGB")
                    pixels.thumbnail((1024, 1024))
                message["content"][index] = {"type": "image", "image": pixels}
        template = {"enable_thinking": False, **(extra_body or {}).get("chat_template_kwargs", {})}
        with _MODEL_LOCK:
            self.load()
            inputs = self.processor.apply_chat_template(
                converted, tools=tools, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", **template).to(self.model.device)
            input_length = inputs["input_ids"].shape[-1]
            limit = max_completion_tokens or max_tokens
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=limit, do_sample=False)
            tokens = output[0, input_length:]
            raw = self.processor.decode(tokens, skip_special_tokens=True)
        answer = raw.rsplit("</think>", 1)[-1].strip()
        if template["enable_thinking"] and "</think>" not in raw:
            raise ValueError("Local reasoning exhausted its token budget before producing an answer")
        content, calls = parse_tool_calls(answer, tools or [])
        finish = "length" if len(tokens) >= limit else "tool_calls" if calls else "stop"
        if calls and finish == "length":
            raise ValueError("Truncated local tool response")
        return ChatCompletion.model_validate({
            "id": f"local_{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
            "model": self.model_id, "choices": [{"index": 0, "finish_reason": finish,
                "message": {"role": "assistant", "content": content, "tool_calls": calls or None}}],
            "usage": {"prompt_tokens": input_length, "completion_tokens": len(tokens),
                      "total_tokens": input_length + len(tokens)}})


@lru_cache(maxsize=1)
def _cached_local_client(model_id, revision, device=None, quantization="none"):
    return LocalQwenClient(model_id, revision, device, quantization)


def get_local_client(model_id, revision, device=None, quantization="none"):
    with _MODEL_LOCK:
        return _cached_local_client(model_id, revision, device, quantization)
