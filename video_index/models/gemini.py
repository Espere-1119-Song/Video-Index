"""Google Gemini generateContent API."""
from __future__ import annotations

import requests

from .base import ChatModel, ContextLimitError, ModelError, Reply, CONTEXT_PATTERNS


class Gemini(ChatModel):
    URL = "https://generativelanguage.googleapis.com/v1beta"

    def _request(self, prompt, images_b64, system, max_tokens, temperature) -> Reply:
        spec = self.spec
        parts = [{"inline_data": {"mime_type": "image/jpeg", "data": b}} for b in images_b64] + [{"text": prompt}]
        body = {"contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        for k, v in spec.extra.items():
            if k == "generationConfig":
                body["generationConfig"].update(v)
            else:
                body[k] = v
        url = f"{(spec.base_url or self.URL).rstrip('/')}/models/{spec.model}:generateContent"
        r = requests.post(url, json=body, timeout=spec.timeout,
                          headers={"x-goog-api-key": spec.api_key(), "Content-Type": "application/json"})
        if r.status_code in (400, 413):
            if any(p in r.text.lower() for p in CONTEXT_PATTERNS):
                raise ContextLimitError(r.text[:300])
            raise ModelError(f"{spec.name}: HTTP {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
        d = r.json()
        cand = (d.get("candidates") or [{}])[0]
        # thinking models flag some parts as thoughts: skip them, join the rest
        text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought")).strip()
        u = d.get("usageMetadata") or {}
        return Reply(text=text, stop_reason=cand.get("finishReason"), input_tokens=u.get("promptTokenCount"),
                     output_tokens=u.get("candidatesTokenCount"), raw=d)
