"""OpenAI-compatible chat completions: OpenAI, and every open-weight model served by vLLM, SGLang, Ollama,
LM Studio, TGI or a hosted gateway. ``base_url`` ends in ``/v1``."""
from __future__ import annotations

import requests

from .base import ChatModel, ContextLimitError, ModelError, Reply


class OpenAICompatible(ChatModel):
    def _request(self, prompt, images_b64, system, max_tokens, temperature) -> Reply:
        spec = self.spec
        if not spec.base_url:
            raise ModelError(f"{spec.name}: base_url is required for provider openai_compatible")
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in images_b64]
        content.append({"type": "text", "text": prompt})
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        body = {"model": spec.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        body.update(spec.extra)
        headers = {"Content-Type": "application/json"}
        key = spec.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = requests.post(spec.base_url.rstrip("/") + "/chat/completions", json=body, headers=headers, timeout=spec.timeout)
        if r.status_code == 400 or r.status_code == 413:
            raise ContextLimitError(r.text[:300]) if _is_context(r.text) else ModelError(f"{spec.name}: HTTP {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
        d = r.json()
        ch = d["choices"][0]
        text = ch["message"].get("content") or ""
        if isinstance(text, list):                       # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        u = d.get("usage") or {}
        return Reply(text=text.strip(), stop_reason=ch.get("finish_reason"), input_tokens=u.get("prompt_tokens"),
                     output_tokens=u.get("completion_tokens"), raw=d)


def _is_context(text: str) -> bool:
    from .base import CONTEXT_PATTERNS
    t = text.lower()
    return any(p in t for p in CONTEXT_PATTERNS)
