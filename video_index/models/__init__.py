"""Model roles and providers. See :mod:`video_index.models.base` for the interface and
``configs/models.example.yaml`` for the configuration."""
from __future__ import annotations

from .base import ChatModel, ContextLimitError, ModelError, ModelSpec, RefusalError, Reply, RequestError, encode_image

PROVIDERS: dict[str, str] = {
    "openai_compatible": "video_index.models.openai_compat:OpenAICompatible",
    "openai": "video_index.models.openai_compat:OpenAICompatible",
    "vllm": "video_index.models.openai_compat:OpenAICompatible",
    "anthropic": "video_index.models.anthropic:Anthropic",
    "gemini": "video_index.models.gemini:Gemini",
}

DEFAULT_BASE_URL = {"openai": "https://api.openai.com/v1"}


def register_provider(name: str, target: str) -> None:
    """Add a provider: ``target`` is ``"package.module:ClassName"`` of a :class:`ChatModel` subclass."""
    PROVIDERS[name] = target


def build_model(spec: ModelSpec | dict) -> ChatModel:
    if isinstance(spec, dict):
        known = {f for f in ModelSpec.__dataclass_fields__}
        extra = {k: v for k, v in spec.items() if k not in known}
        spec = ModelSpec(**{k: v for k, v in spec.items() if k in known})
        spec.extra = {**extra.get("extra", {}), **{k: v for k, v in extra.items() if k != "extra"}, **spec.extra}
    if spec.provider not in PROVIDERS:
        raise ModelError(f"unknown provider '{spec.provider}' (known: {sorted(PROVIDERS)}); add one with register_provider")
    if not spec.base_url and spec.provider in DEFAULT_BASE_URL:
        spec.base_url = DEFAULT_BASE_URL[spec.provider]
    mod, cls = PROVIDERS[spec.provider].split(":")
    import importlib
    return getattr(importlib.import_module(mod), cls)(spec)


__all__ = ["ChatModel", "ContextLimitError", "ModelError", "ModelSpec", "RefusalError", "Reply", "RequestError", "encode_image",
           "build_model", "register_provider", "PROVIDERS"]
