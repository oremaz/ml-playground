"""OpenAI-compatible provider settings, shared with the NutriCoach integration."""

import os

OPENROUTER_MODEL_ID = "openrouter/free"
OPENROUTER_MAX_OUTPUT_TOKENS = 8192

LOCAL_MODEL_ID = "Qwen/Qwen3.8-27B"
LOCAL_MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PROVIDERS = ("openrouter", "groq", "local")
PROVIDER_URLS = {"openrouter": "https://openrouter.ai/api/v1",
                 "groq": "https://api.groq.com/openai/v1"}
DEFAULT_MODELS = {"openrouter": OPENROUTER_MODEL_ID, "groq": "qwen/qwen3.8-27b", "local": LOCAL_MODEL_ID}
REASONING_EFFORTS = {"openrouter": ("none", "low", "medium", "high", "xhigh", "max"),
                     "groq": ("none", "low", "medium", "high"),
                     "local": ("none", "low", "medium", "xhigh")}


def default_reasoning(provider):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}")
    return "xhigh" if provider == "local" else "high"


def default_model(provider):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}")
    return os.getenv(f"{provider.upper()}_MODEL_ID", "").strip() or DEFAULT_MODELS[provider]


def request_settings(provider, reasoning_effort):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}")
    if reasoning_effort is None:
        return {}
    if reasoning_effort not in REASONING_EFFORTS[provider]:
        raise ValueError(f"Unsupported reasoning effort for {provider}: {reasoning_effort}")
    if provider == "openrouter":
        return {"extra_body": {"reasoning": {"effort": reasoning_effort}}}
    if provider == "groq":
        return {"reasoning_effort": reasoning_effort, "extra_body": {"reasoning_format": "hidden"}}
    return {"extra_body": {"chat_template_kwargs": {
        "enable_thinking": reasoning_effort != "none",
        "reasoning_effort": reasoning_effort if reasoning_effort != "none" else "xhigh"}}}


def create_client(provider="openrouter", api_key=None, *, model=None, device=None,
                  quantization="none", revision=LOCAL_MODEL_REVISION, **kwargs):
    default_model(provider)
    if provider == "local":
        from .local_qwen import get_local_client
        return get_local_client(model or default_model(provider), revision, device, quantization)
    from openai import OpenAI
    key = api_key or os.getenv(f"{provider.upper()}_API_KEY", "")
    if not key:
        raise ValueError(f"{provider.upper()}_API_KEY not set")
    return OpenAI(api_key=key, base_url=PROVIDER_URLS[provider], **kwargs)


def chat_settings(provider, api_key, model, reasoning_effort, *, device=None, quantization="none"):
    settings = {"model": model, "api_key": api_key or os.getenv(f"{provider.upper()}_API_KEY", ""),
                "base_url": PROVIDER_URLS.get(provider, "http://localhost/v1"),
                "max_tokens": OPENROUTER_MAX_OUTPUT_TOKENS, **request_settings(provider, reasoning_effort)}
    if provider == "local":
        settings["api_key"] = "local"
        settings["client"] = create_client(provider, model=model, device=device,
                                           quantization=quantization).chat.completions
    return settings
