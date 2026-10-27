"""Capacity-preserving defaults and provider loader contracts, without downloads."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn


def test_gemini_lite_successor_omits_deprecated_sampling(monkeypatch):
    from ml_pipeline.data_augmentation.text_aug import LLMTextAugmenter, TextAugmentationConfig

    google, genai = ModuleType("google"), ModuleType("google.genai")
    genai.Client = lambda **kwargs: object()
    genai.types = SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    current = LLMTextAugmenter(api_key="test")
    assert current.model_name == "gemini-3.5-flash-lite"
    assert set(current._gemini_gen_config) == {"max_output_tokens"}
    legacy = LLMTextAugmenter(api_key="test", config=TextAugmentationConfig(model_name="gemini-3.1-flash-lite"))
    assert legacy._gemini_gen_config["temperature"] == legacy.config.temperature


def test_openrouter_free_router_and_explicit_model_overrides(monkeypatch):
    from ml_pipeline.data_augmentation import text_aug

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: object()))
    current = text_aug.LLMTextAugmenter(api_key="test", config=text_aug.TextAugmentationConfig(backend="openrouter"))
    assert current.model_name == "openrouter/free"
    factory = Mock()
    monkeypatch.setattr(text_aug, "LLMTextAugmenter", factory)
    text_aug.llm_paraphrase_augmentation(["A recipe"], api_key="test", backend="openrouter",
                                      model_name="chosen/provider-size")
    assert factory.call_args.kwargs["config"].model_name == "chosen/provider-size"


@pytest.fixture
def vision(monkeypatch):
    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace())
    from ml_pipeline.pipelines_torch import vision_models

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(3, 4)
            self.config = SimpleNamespace(hidden_size=4)

        def forward(self, pixel_values):
            return SimpleNamespace(last_hidden_state=self.layer(pixel_values.mean((2, 3)))[:, None])

    class Multimodal(nn.Module):
        device = torch.device("cpu")

        def forward(self, **kwargs):
            return SimpleNamespace(logits=torch.ones(1, 2, 5))

    class Batch(dict):
        def to(self, device):
            return self

    processing = Mock()
    processing.tokenizer = SimpleNamespace(padding_side="right")
    processing.apply_chat_template.return_value = Batch(pixel_values=torch.ones(1, 3, 4, 4))
    processing.return_value = Batch(pixel_values=torch.ones(1, 3, 4, 4))
    processor = Mock(return_value=processing)
    model_loader = Mock(return_value=Multimodal())
    backbone_loader = Mock(side_effect=lambda *args, **kwargs: Backbone())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=processor),
        AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=model_loader),
        AutoModel=SimpleNamespace(from_pretrained=backbone_loader), AutoImageProcessor=object,
        BitsAndBytesConfig=lambda **kwargs: kwargs,
    ))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        LoraConfig=lambda **kwargs: kwargs, get_peft_model=lambda model, config: model,
    ))
    return vision_models, processor, model_loader, backbone_loader


def test_qwen_keeps_two_billion_class_and_pins_both_processor_and_model(vision):
    module, processor, loader, _ = vision
    model = module.Qwen2VLQLoRA()
    assert loader.call_args.args == ("Qwen/Qwen3.5-2B",)
    assert loader.call_args.kwargs["revision"] == processor.call_args.kwargs["revision"]
    assert loader.call_args.kwargs["revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc"
    assert model(torch.ones(1, 3, 4, 4), ["classify"]).shape == (1, 2)
    assert model.processor.tokenizer.padding_side == "left"
    assert model.processor.apply_chat_template.call_args.kwargs["enable_thinking"] is False
    with pytest.raises(ValueError, match="one prompt"):
        model(torch.ones(2, 3, 4, 4), ["classify"])
    module.Qwen2VLQLoRA(model_name="Qwen/Qwen2-VL-2B-Instruct")
    assert loader.call_args.kwargs["revision"] is None


@pytest.mark.parametrize("size,backbone", [
    ("small", "facebook/dinov3-vits16-pretrain-lvd1689m"),
    ("base", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
])
def test_meta_successors_preserve_backbone_size_and_freezing(vision, size, backbone):
    module, _, _, loader = vision
    model = module.get_model(f"dinov3_{size}_classifier", num_classes=3)
    assert loader.call_args.args == (backbone,)
    assert loader.call_args.kwargs["revision"] == module.DINOV3_REVISIONS[backbone]
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert model(torch.ones(1, 3, 16, 16)).shape == (1, 3)
