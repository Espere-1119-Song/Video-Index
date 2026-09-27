"""Option-letter scores of the pool screen.

The screen reads the log-probabilities of the option letters at the first generated position:
``max_tokens=1, logprobs=true, top_logprobs=20``, and the assistant turn prefilled with "Answer:" so that the next
token is the letter. Token variants ("A", " A", "(A", "A.") are merged by log-sum-exp; a letter absent from the
top 20 gets log(1e-6).

This needs an OpenAI-compatible endpoint that returns log-probabilities (vLLM, SGLang, OpenAI). For any other
provider the model is asked for the letter as text (`text_scores`): the chosen letter is known, the margin is not,
and the removal rules fall back to correctness alone (see docs/pool.md).
"""
from __future__ import annotations

import math
import random
import re
import time

import requests

from ..models import ContextLimitError, ModelError, encode_image
from ..models.base import CONTEXT_PATTERNS
from ..scoring import mcq_letter
from .schema import LETTERS

SEED = 42
N_PERM = 4
LOG2 = math.log(2.0)
LP_FLOOR = math.log(1e-6)
TOP_LOGPROBS = 20
ASSISTANT_PREFIX = "Answer:"
LOGPROB_PROVIDERS = ("openai_compatible", "openai", "vllm")
_TOKEN_LETTER = re.compile(r"^\s*[\(\[]?([A-Z])[\)\]\.:]?\s*$")


def permutations_for(item_id, k, n_perm=N_PERM, seed=SEED):
    """n_perm option orders per item; perm[j] is the index of the original option shown at position j."""
    rng = random.Random(f"{seed}|{item_id}")
    return [rng.sample(range(k), k) for _ in range(n_perm)]


def letter_logprobs(top_logprobs, k):
    """top_logprobs: [{token, logprob}] of the first generated position -> (log-probabilities of the k letters,
    letters that are missing)."""
    acc = {}
    for e in top_logprobs or []:
        m = _TOKEN_LETTER.match(e.get("token") or "")
        if not m:
            continue
        letter, v = m.group(1), e["logprob"]
        acc[letter] = v if letter not in acc else max(acc[letter], v) + math.log1p(math.exp(-abs(acc[letter] - v)))
    lp = [float(acc.get(LETTERS[i], LP_FLOOR)) for i in range(k)]
    return lp, [LETTERS[i] for i in range(k) if LETTERS[i] not in acc]


def margin_of(lp, correct_pos):
    """log p(correct) minus the mean log p of the other options."""
    others = [v for i, v in enumerate(lp) if i != correct_pos]
    return lp[correct_pos] - (sum(others) / len(others) if others else 0.0)


def argmax_pos(lp):
    return lp.index(max(lp))


def confidence(lp):
    """(index of the best letter, its probability among the letters present); (None, 0.0) when none is present."""
    present = [v for v in lp if v > LP_FLOOR]
    if not present:
        return None, 0.0
    mx = max(lp)
    return lp.index(mx), round(1.0 / sum(math.exp(v - mx) for v in present), 4)


def supports_logprobs(model, mode="auto"):
    if mode in (True, "true", "on"):
        return True
    if mode in (False, "false", "off"):
        return False
    return model.spec.provider in LOGPROB_PROVIDERS


def logprob_request(model, prompt, images=(), prefill=True):
    """One forward through the endpoint of `model`. -> {top: [{token, logprob}], top1_token, prompt_tokens}."""
    spec = model.spec
    if not spec.base_url:
        raise ModelError(f"{spec.name}: base_url is required for log-probability requests")
    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                                                   + encode_image(im, spec.image_long_side, spec.jpeg_quality)}}
               for im in images]
    content.append({"type": "text", "text": prompt})
    body = {"model": spec.model, "temperature": 0.0, "max_tokens": 1, "logprobs": True,
            "top_logprobs": TOP_LOGPROBS, "messages": [{"role": "user", "content": content}]}
    if prefill:
        body["messages"].append({"role": "assistant", "content": ASSISTANT_PREFIX})
        body.update(continue_final_message=True, add_generation_prompt=False)
    headers = {"Content-Type": "application/json"}
    key = spec.api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    last = None
    for attempt in range(spec.max_retries + 1):
        try:
            r = requests.post(spec.base_url.rstrip("/") + "/chat/completions", json=body, headers=headers,
                              timeout=spec.timeout)
            if r.status_code in (400, 413):
                if any(p in r.text.lower() for p in CONTEXT_PATTERNS + ("longer than",)):
                    raise ContextLimitError(r.text[:200])
                raise ModelError(f"{spec.name}: HTTP {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
            d = r.json()
            lpc = (d["choices"][0].get("logprobs") or {}).get("content") or []
            if not lpc:
                raise ModelError(f"{spec.name}: the endpoint returned no log-probabilities")
            return dict(top=[dict(token=e.get("token"), logprob=e.get("logprob"))
                             for e in (lpc[0].get("top_logprobs") or [])],
                        top1_token=lpc[0].get("token"), prompt_tokens=(d.get("usage") or {}).get("prompt_tokens", 0))
        except ModelError:
            raise
        except Exception as e:  # noqa: BLE001  connection, 5xx, timeout
            last = e
            time.sleep(min(20.0, 2.0 ** attempt + random.random()))
    raise ModelError(f"{spec.name}: {type(last).__name__}: {str(last)[:200]}")


def letter_scores(model, prompt, images, texts, mode="auto", prefill=True):
    """Scores of the options `texts` as presented (letters A, B, ... in this order).
    -> dict(lp, argmax, missing, top1_token, prompt_tokens, scored). scored = "logprobs" | "text".
    In text mode lp is None and argmax is the option the reply names (None when it names none)."""
    k = len(texts)
    if supports_logprobs(model, mode):
        r = logprob_request(model, prompt, images, prefill)
        lp, missing = letter_logprobs(r["top"], k)
        am = argmax_pos(lp)
        return dict(lp=lp, argmax=am if lp[am] > LP_FLOOR else None, missing=missing, top1_token=r["top1_token"],
                    prompt_tokens=r["prompt_tokens"], scored="logprobs")
    reply = model.generate(prompt, list(images), max_tokens=16)
    letter = mcq_letter(reply.text, [f"{LETTERS[i]}. {t}" for i, t in enumerate(texts)])
    pos = ord(letter) - 65 if letter and ord(letter) - 65 < k else None
    return dict(lp=None, argmax=pos, missing=[], top1_token=reply.text[:8], prompt_tokens=reply.input_tokens or 0,
                scored="text")
