"""Models for the evaluation.

Two ways to plug a model in:

1. A model behind an endpoint: any role of ``configs/models.yaml`` or an inline specification
   (``video_index.models``). Open-weight models are served with an OpenAI-compatible server and an API key.

2. A model that runs in this process: a class with one method

       class MyModel:
           threaded = False                       # True when answer() may be called from several threads
           def __init__(self, **kwargs): ...
           def answer(self, frames, timestamps, prompt) -> str: ...

   ``frames`` is a list of PIL RGB images in temporal order (empty for the blind protocol), ``timestamps`` their
   times in seconds, ``prompt`` the full text. The method returns the reply text. When the frames do not fit, raise
   ``video_index.models.ContextLimitError`` (or any exception whose message names the context or the memory): the
   runner halves the frame count and calls again. Pass it as ``--local package.module:ClassName``.
"""
from __future__ import annotations

import importlib
import json

from ..models import ChatModel, ContextLimitError, ModelError, RefusalError, build_model

FALLBACK_PATTERNS = ("out of memory", "maximum context", "context length", "too long", "exceed", "max_position", "too many",
                     "longer than", "cuda error", "sequence length", "index out of range in self")


class EndpointModel:
    """A :class:`video_index.models.ChatModel` behind the ``answer`` interface."""
    threaded = True

    def __init__(self, model: ChatModel, max_tokens: int | None = None):
        self.model, self.max_tokens = model, max_tokens
        self.name = model.name

    def answer(self, frames, timestamps, prompt) -> str:
        try:
            return self.model.generate(prompt, list(frames or []), max_tokens=self.max_tokens).text
        except RefusalError:
            return "[refusal]"


def is_size_error(exc: Exception) -> bool:
    return isinstance(exc, ContextLimitError) or any(k in str(exc).lower() for k in FALLBACK_PATTERNS)


def load_local(target: str, kwargs: dict | None = None):
    mod, _, cls = target.partition(":")
    if not cls:
        raise ModelError(f"--local expects package.module:ClassName, got '{target}'")
    m = getattr(importlib.import_module(mod), cls)(**(kwargs or {}))
    if not hasattr(m, "answer"):
        raise ModelError(f"{target} has no answer(frames, timestamps, prompt) method")
    if not hasattr(m, "name"):
        m.name = cls
    return m


def load_model(role: str | None = None, model_config: str | None = None, local: str | None = None,
               local_kwargs: str | None = None, max_tokens: int | None = None, name: str | None = None):
    """``role`` is a role of the model configuration, or an inline specification: a JSON object or
    ``key=value,key=value`` (provider, model, base_url, api_key_env, max_frames, ...)."""
    if local:
        m = load_local(local, json.loads(local_kwargs) if local_kwargs else None)
    else:
        if not role:
            raise ModelError("give --role (a role of the model configuration or an inline specification) or --local")
        spec = None
        r = role.strip()
        if r.startswith("{"):
            spec = json.loads(r)
        elif "=" in r:
            spec = dict(kv.split("=", 1) for kv in r.split(","))
            for k in ("max_frames", "max_tokens", "timeout", "max_retries", "image_long_side", "jpeg_quality"):
                if k in spec:
                    spec[k] = int(spec[k])
            if "temperature" in spec:
                spec["temperature"] = float(spec["temperature"])
        if spec is not None:
            spec.setdefault("name", spec.get("model"))
            chat = build_model(spec)
        else:
            from ..config import Models
            chat = Models.from_file(model_config or "configs/models.yaml")[r]
        m = EndpointModel(chat, max_tokens)
    if name:
        m.name = name
    return m
