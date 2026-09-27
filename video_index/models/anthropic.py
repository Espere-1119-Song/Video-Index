"""Anthropic Messages API."""
from __future__ import annotations

import requests

from .base import ChatModel, ContextLimitError, ModelError, RefusalError, Reply, CONTEXT_PATTERNS


class Anthropic(ChatModel):
    URL = "https://api.anthropic.com/v1/messages"

    def _request(self, prompt, images_b64, system, max_tokens, temperature) -> Reply:
        spec = self.spec
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}} for b in images_b64]
        content.append({"type": "text", "text": prompt})
        body = {"model": spec.model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
        if temperature is not None and "thinking" not in spec.extra:
            body["temperature"] = temperature
        if system:
            body["system"] = system
        body.update(spec.extra)
        r = requests.post(spec.base_url or self.URL, json=body, timeout=spec.timeout,
                          headers={"x-api-key": spec.api_key(), "anthropic-version": "2023-06-01", "content-type": "application/json"})
        if r.status_code in (400, 413):
            if any(p in r.text.lower() for p in CONTEXT_PATTERNS):
                raise ContextLimitError(r.text[:300])
            raise ModelError(f"{spec.name}: HTTP {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
        d = r.json()
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text").strip()
        if d.get("stop_reason") == "refusal":
            raise RefusalError(f"{spec.name}: refusal")
        u = d.get("usage") or {}
        return Reply(text=text, stop_reason=d.get("stop_reason"), input_tokens=u.get("input_tokens"),
                     output_tokens=u.get("output_tokens"), raw=d)
