"""Model interface shared by every role of the pipelines.

A *role* is what a model does in a pipeline (reference, attacker, captioner, judge, labeler, verifier);
a *provider* is how it is reached. Every role is filled from the same interface, so any model can take any
role: an API model, or an open-weight model behind an OpenAI-compatible server (vLLM, SGLang, Ollama,
LM Studio, TGI). Open-weight servers also take an API key: the server is started with one (for example
``vllm serve ... --api-key $KEY``) and the client sends it, so credentials are handled the same way for
every provider. Keys are read from environment variables named in the model specification; they are
never written to configuration files, logs or result files.
"""
from __future__ import annotations

import abc
import base64
import io
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Sequence


class ModelError(RuntimeError):
    """A request failed after every retry."""


class ContextLimitError(ModelError):
    """The request does not fit the model's context or memory. Callers that send frames catch this,
    halve the frame count and retry (the frames actually sent are recorded per item)."""


class RefusalError(ModelError):
    """The provider returned a refusal instead of an answer."""


@dataclass
class ModelSpec:
    """One model as written in ``configs/models.yaml``."""
    name: str                                  # label used in result files and tables
    provider: str                              # openai_compatible | anthropic | gemini | <registered provider>
    model: str                                 # model identifier sent to the provider
    api_key_env: str | None = None             # name of the environment variable that holds the key
    base_url: str | None = None                # endpoint; required for openai_compatible
    max_frames: int | None = None              # largest number of images one request may carry
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout: int = 300
    max_retries: int = 6
    image_long_side: int | None = None         # resize before sending (None = send as given)
    jpeg_quality: int = 85
    extra: dict[str, Any] = field(default_factory=dict)   # provider-specific request fields

    def api_key(self) -> str:
        if not self.api_key_env:
            return ""
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ModelError(f"model '{self.name}': environment variable {self.api_key_env} is not set")
        return key


@dataclass
class Reply:
    text: str
    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    n_images: int = 0
    raw: dict | None = None


def encode_image(img, long_side: int | None = None, quality: int = 85) -> str:
    """PIL image, numpy array, bytes or path -> base64 JPEG string."""
    from PIL import Image
    if isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img))
    elif isinstance(img, (str, os.PathLike)):
        img = Image.open(img)
    elif not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    if long_side and max(img.size) > long_side:
        s = long_side / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


CONTEXT_PATTERNS = ("maximum context", "context length", "context window", "too long", "too many images", "too many tokens",
                    "max_position", "exceeds the", "request too large", "out of memory", "prompt is too long")


class ChatModel(abc.ABC):
    """Text + images in, text out. Subclasses implement :meth:`_request` for one provider."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @abc.abstractmethod
    def _request(self, prompt: str, images_b64: list[str], system: str | None, max_tokens: int,
                 temperature: float) -> Reply:
        """One request. Raise :class:`ContextLimitError` / :class:`RefusalError` where they apply, any other
        exception for a transient failure (the caller retries with backoff)."""

    def retryable(self, exc: Exception) -> bool:
        return not isinstance(exc, (ContextLimitError, RefusalError))

    def generate(self, prompt: str, images: Sequence = (), system: str | None = None,
                 max_tokens: int | None = None, temperature: float | None = None) -> Reply:
        spec = self.spec
        if spec.max_frames is not None and len(images) > spec.max_frames:
            raise ContextLimitError(f"{spec.name}: {len(images)} images exceed max_frames={spec.max_frames}")
        b64 = [encode_image(im, spec.image_long_side, spec.jpeg_quality) for im in images]
        last = None
        for attempt in range(spec.max_retries + 1):
            try:
                r = self._request(prompt, b64, system, max_tokens or spec.max_tokens,
                                  spec.temperature if temperature is None else temperature)
                r.n_images = len(b64)
                return r
            except Exception as e:  # noqa: BLE001
                last = e
                msg = str(e).lower()
                if not isinstance(e, ModelError) and any(p in msg for p in CONTEXT_PATTERNS):
                    raise ContextLimitError(str(e)[:300]) from e
                if not self.retryable(e) or attempt == spec.max_retries:
                    break
                time.sleep(min(90.0, 2.0 ** attempt + random.random()))
        if isinstance(last, ModelError):
            raise last
        raise ModelError(f"{spec.name}: {type(last).__name__}: {str(last)[:300]}") from last
