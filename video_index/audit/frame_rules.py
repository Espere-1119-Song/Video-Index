"""Frame selection of the audit conditions. Two rule families, one per role:

reference   an index grid over the stored frames, long side <= frame_long_side (768)
              single     the middle frame, round((n - 1) / 2)
              uniform    round(i (n - 1) / (k - 1)), i = 0..k-1
              shuffle    the uniform frames in one fixed random order (seed), the same order for every item
              window     k frames inside one contiguous tenth of the video; the start is drawn per item with the seed
                         "<seed>|<qid>", uniformly in [0, 0.9]
attacker    a time grid over the frame timestamps of meta.json (the stored frame nearest to each target time), stored
            resolution unless a short side is configured
              single     frame 16 of the 32-frame grid
              uniform    k times from the first to the last timestamp, duplicates removed
              shuffle    the 32 frames in a random order drawn per item with the seed "<seed>|<qid>"
              window     k times inside the first tenth of the timeline
Every function returns (indices, extra): indices in the order the frames are sent, extra = fields recorded in the row.
"""
from __future__ import annotations

import bisect
import math
import random

from ..data import frames as F


def _dedup(idx):
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


# ------------------------------------------------------------------ reference: index grid
def _grid(n, k, lo=0.0, hi=None):
    hi = (n - 1) if hi is None else hi
    if k <= 1:
        return [int(round(lo))]
    return [int(round(lo + i * (hi - lo) / (k - 1))) for i in range(k)]


def _clamp(idx, n):
    return _dedup(max(0, min(n - 1, i)) for i in idx)


def reference_indices(kind, n, k, qid, seed=42):
    if n <= 0:
        return [], {}
    if kind == "single":
        return [int(round((n - 1) / 2))], {}
    if kind == "uniform":
        return _clamp(_grid(n, k), n), {}
    if kind == "shuffle":
        idx = _clamp(_grid(n, k), n)
        random.Random(seed).shuffle(idx)
        return idx, {}
    if kind == "window":
        start = random.Random(f"{seed}|{qid}").uniform(0.0, 0.9)
        idx = _clamp(_grid(n, k, start * (n - 1), (start + 0.1) * (n - 1)), n)
        return idx, dict(window_start=round(start, 4))
    raise ValueError(kind)


# ------------------------------------------------------------------ attacker: time grid
def uniform_times(t0, t1, k):
    if k <= 1:
        return [(t0 + t1) / 2.0]
    return [t0 + j * (t1 - t0) / (k - 1) for j in range(k)]


def nearest_indices(ts, targets):
    """Index of the stored frame nearest to each target time (ties go to the earlier frame)."""
    out = []
    for t in targets:
        j = bisect.bisect_left(ts, t)
        if j <= 0:
            out.append(0)
        elif j >= len(ts):
            out.append(len(ts) - 1)
        else:
            out.append(j if (ts[j] - t) < (t - ts[j - 1]) else j - 1)
    return out


def attacker_indices(kind, ts, k, qid, seed=42):
    n = len(ts)
    if n <= 0:
        return [], {}
    if kind == "single":
        grid = nearest_indices(ts, uniform_times(ts[0], ts[-1], 32))
        return [grid[16]], {}
    if kind == "uniform":
        if n < k:
            return list(range(n)), dict(capped=True)
        return _dedup(nearest_indices(ts, uniform_times(ts[0], ts[-1], k))), {}
    if kind == "shuffle":
        idx = _dedup(nearest_indices(ts, uniform_times(ts[0], ts[-1], k)))
        random.Random(f"{seed}|{qid}").shuffle(idx)
        return idx, {}
    if kind == "window":
        t0, t1 = ts[0], ts[0] + (ts[-1] - ts[0]) / 10.0
        return _dedup(nearest_indices(ts, uniform_times(t0, t1, k))), dict(window_start=0.0)
    raise ValueError(kind)


# ------------------------------------------------------------------ decoding
def timestamps_of(video, meta):
    """(timestamps of the stored frames, frame count). Both come from meta.json; a video without it is probed."""
    ts = list((meta or {}).get("frame_timestamps") or [])
    n = int((meta or {}).get("n_frames") or len(ts))
    if n <= 0:
        p = F.probe(video)
        n, rate = p["n_frames"], p["fps"] or 1.0
        ts = [i / rate for i in range(n)]
    if len(ts) != n:
        ts = [float(i) for i in range(n)]
    return ts, n


def decode(video, indices, short_side=None, long_side=None, pad=False):
    """Frames at the given indices, in the given order. Indices that cannot be decoded are dropped, or with pad=True
    filled with the last decoded frame."""
    import cv2
    from PIL import Image
    cap, n, _ = F._open(video)
    want = set(int(i) for i in indices)
    got, i, last = {}, 0, max(want) if want else -1
    while i <= last:
        if not cap.grab():
            break
        if i in want:
            ok, fr = cap.retrieve()
            if ok:
                got[i] = F._resize(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), short_side, long_side)
        i += 1
    cap.release()
    if pad and got:                           # frame count of the container above the decodable frames
        last = got[max(got)]
        for i in want:
            got.setdefault(i, last)
    idx = [int(i) for i in indices if int(i) in got]
    return [got[i] for i in idx], idx


def fit_token_budget(frames, budget, patch=32, per_frame_extra=6):
    """Scale every frame by sqrt(budget / estimate) when the estimated visual tokens exceed the budget of the server
    (tokens per frame = ceil(w / patch) * ceil(h / patch) + per_frame_extra). Returns (frames, short side or None)."""
    if not budget or not frames:
        return frames, None
    est = sum(math.ceil(f.width / patch) * math.ceil(f.height / patch) + per_frame_extra for f in frames)
    if est <= budget:
        return frames, None
    s = math.sqrt(budget / est) * 0.98
    from PIL import Image
    out = [f.resize((max(patch, int(f.width * s)), max(patch, int(f.height * s))), Image.LANCZOS) for f in frames]
    return out, min(min(f.size) for f in out)


def select(role_family, kind, video, meta, k, qid, ctx):
    """Frames of one item for one condition. role_family: "reference" | "attacker".
    Returns (frames, timestamps, extra)."""
    ts, n = timestamps_of(video, meta)
    seed = ctx.get("seed", 42)
    if role_family == "reference":
        idx, extra = reference_indices(kind, n, k, qid, seed)
        frames, idx = decode(video, idx, long_side=ctx.get("frame_long_side", 768), pad=True)
    else:
        idx, extra = attacker_indices(kind, ts, k, qid, seed)
        frames, idx = decode(video, idx, short_side=ctx.get("audit_attacker_short_side"))
        frames, fit = fit_token_budget(frames, ctx.get("attacker_token_budget"), int(ctx.get("attacker_patch", 32)))
        if fit:
            extra = dict(extra, fit_short_side=fit)
    return frames, [ts[i] for i in idx], extra
