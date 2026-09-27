"""Shared execution helpers: resumable parallel maps over items, and model calls that halve the frame count when a
request does not fit."""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable

from .data.frames import subsample
from .models import ChatModel, ContextLimitError, ModelError, RefusalError


def log(msg: str, tag: str = "video-index") -> None:
    print(f"[{tag} {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def ask(model: ChatModel, prompt: str, frames: list | None = None, timestamps: list | None = None,
        min_frames: int = 8, **kw) -> dict:
    """Call the model; on a context or memory error halve the frames (uniformly) and retry.
    Returns {"pred", "frames", "fallback", "error", "stop_reason"}; never raises."""
    frames = list(frames or [])
    ts = list(timestamps or [0.0] * len(frames))
    fallback = 0
    while True:
        try:
            r = model.generate(prompt, frames, **kw)
            return dict(pred=r.text, frames=len(frames), fallback=fallback, error=None, stop_reason=r.stop_reason)
        except ContextLimitError as e:
            if len(frames) > min_frames:
                frames, ts = subsample(frames, ts, max(min_frames, len(frames) // 2))
                fallback += 1
                continue
            return dict(pred=None, frames=len(frames), fallback=fallback, error=f"context: {str(e)[:200]}", stop_reason=None)
        except RefusalError as e:
            return dict(pred=None, frames=len(frames), fallback=fallback, error="refusal", stop_reason="refusal")
        except ModelError as e:
            return dict(pred=None, frames=len(frames), fallback=fallback, error=str(e)[:300], stop_reason=None)


def pmap(fn: Callable[[dict], dict | None], rows: Iterable[dict], workers: int = 8, desc: str = "", every: int = 50):
    """Run fn over rows in a thread pool, yielding results as they finish. fn handles its own errors."""
    rows = list(rows)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(fn, r) for r in rows]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                out = f.result()
            except Exception as e:  # noqa: BLE001
                out = None
                log(f"{desc}: worker failure {type(e).__name__}: {str(e)[:160]}")
            if out is not None:
                yield out
            if every and k % every == 0:
                log(f"{desc}: {k}/{len(rows)} ({(time.time() - t0) / 60:.1f} min)")
