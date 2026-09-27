"""The two protocols of Video-Index and their prompts.

VIDEO   one frame per second of the timeline, at most ``frame_cap`` frames (512; uniform thinning beyond), every frame
        resized to a short side of 224 pixels. When the model rejects the request for its size, the frame count is
        halved (uniformly) and the request repeated; the number of frames actually sent is stored with the answer.
BLIND   the question and the options only, no frames. Every item is asked under four option permutations
        (``permutations_for``); the blind score is the mean over the permutations.

Gain = VIDEO accuracy minus BLIND accuracy on the items that have both.
"""
from __future__ import annotations

import random

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ANSWER_INSTR = "Reply with ONLY the option letter (or the exact short answer if no options)."
PROMPT = "{intro}\n\nQuestion: {q}\n{opts}\n" + ANSWER_INSTR
INTRO_VISUAL = "You are given {n} frame(s) sampled from a video. Answer the question based on these frames."
INTRO_BLIND = "You are given NO frames from the video. Answer the question from the text alone."
N_PERM, PERM_SEED = 4, 42
SHORT_SIDE = 224
FRAME_CAP = 512
FPS = 1.0
GROUPS = [("perception", "Perception"), ("temporal", "Temporal"), ("spatial_physical", "Spatial"),
          ("reasoning_knowledge", "Reasoning")]


def permutations_for(item_id: str, k: int, n_perm: int = N_PERM, seed: int = PERM_SEED) -> list[list[int]]:
    """The option orders of the blind protocol: ``perm[j]`` is the index of the original option shown at position j."""
    rng = random.Random(f"{seed}|{item_id}")
    return [rng.sample(range(k), k) for _ in range(n_perm)]


def render(texts: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(texts))


def video_prompt(item: dict, n_frames: int, perm: list[int] | None = None) -> str:
    texts = [item["options"][j] for j in (perm or range(len(item["options"])))]
    return PROMPT.format(intro=INTRO_VISUAL.format(n=n_frames), q=item["question"], opts=render(texts))


def blind_prompt(item: dict, perm: list[int]) -> str:
    return PROMPT.format(intro=INTRO_BLIND, q=item["question"], opts=render([item["options"][j] for j in perm]))


def gold_letter(item: dict, perm: list[int]) -> str:
    return LETTERS[list(perm).index(int(item["answer_idx"]))]


def score_reply(pred: str | None, item: dict, perm: list[int]):
    """True / False by the rule scorer (first option letter, stated answer, or the option text the reply repeats)."""
    from ..scoring import score
    if pred is None:
        return None
    texts = [item["options"][j] for j in perm]
    return score(pred, gold_letter(item, perm), [f"{LETTERS[i]}. {t}" for i, t in enumerate(texts)], "list")


def one_fps_indices(n_frames: int, native_fps: float, duration_s: float | None, rate: float = FPS,
                    cap: int | None = FRAME_CAP) -> list[int]:
    """Indices of the stored frames that the VIDEO protocol sends: the stored frame nearest to every multiple of
    1/rate seconds of the timeline [0, duration); every stored frame when the video is stored below the rate (long
    videos are stored with at most 1,024 frames); uniform thinning to ``cap`` frames. ``duration_s`` is the duration
    of the source video given with the item (the stored file can be a fraction of a second shorter)."""
    import numpy as np
    if n_frames <= 0:
        return []
    native = native_fps or 2.0
    ts = np.arange(n_frames, dtype=np.float64) / native
    dur = float(duration_s) if duration_s else float(ts[-1] + 0.5)
    if n_frames <= dur * rate + 1:
        idx = list(range(n_frames))
    else:
        targets = np.arange(0.0, max(dur, 0.5 / rate), 1.0 / rate)
        idx = sorted(set(int(np.abs(ts - t).argmin()) for t in targets))
    if cap and len(idx) > cap:
        idx = [idx[i] for i in sorted(set(int(round(x)) for x in np.linspace(0, len(idx) - 1, cap)))]
    return idx
