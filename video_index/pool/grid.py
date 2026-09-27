"""Videos and frame grids of the pool stages.

The screen and the labels use the attacker grid of the audit (video_index.audit.frame_rules): 32 times spread
uniformly from the first to the last frame timestamp, the stored frame nearest to each time, duplicates removed;
the single frame is position 16 of that grid. Frames are resized to a long side of 448, never enlarged.
"""
from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict

from ..audit import frame_rules

FRAME_LONG_SIDE = 448
N_FRAMES = 32
MID_POS = 16


def video_file(ctx, video_id):
    """Path of the normalized video, or None when it is not in the workspace."""
    if not video_id or str(video_id).startswith("k:"):
        return None
    p = os.path.join(ctx.work_dir, "videos", str(video_id), "video.mp4")
    return p if os.path.exists(p) else None


def video_meta(video):
    p = os.path.join(os.path.dirname(video), "meta.json")
    try:
        return json.load(open(p)) if os.path.exists(p) else {}
    except Exception:  # noqa: BLE001
        return {}


def grid_indices(video, meta=None, k=N_FRAMES):
    """(frame indices of the grid without duplicates, index of the single frame)."""
    ts, _ = frame_rules.timestamps_of(video, video_meta(video) if meta is None else meta)
    if not ts:
        return [], None
    grid = frame_rules.nearest_indices(ts, frame_rules.uniform_times(ts[0], ts[-1], k))
    return frame_rules._dedup(grid), grid[min(MID_POS, len(grid) - 1)]


def grid_frames(video, meta=None, k=N_FRAMES, long_side=FRAME_LONG_SIDE):
    """(frames of the grid, position of the single frame in that list)."""
    idx, mid = grid_indices(video, meta, k)
    frames, got = frame_rules.decode(video, idx, long_side=long_side)
    if not frames:
        return [], None
    return frames, got.index(mid) if mid in got else len(frames) // 2


def spread(frames, n):
    """n of the given frames, spread uniformly over the list (fewer when the list is shorter)."""
    if len(frames) <= n:
        return list(frames)
    pick = sorted({min(len(frames) - 1, round(i * (len(frames) - 1) / (n - 1))) for i in range(n)})
    return [frames[i] for i in pick]


class FrameCache:
    """The grid of the most recent videos (items of one video follow each other)."""

    def __init__(self, size=64):
        self.size, self.d, self.lock = size, OrderedDict(), threading.Lock()

    def get(self, video):
        with self.lock:
            if video in self.d:
                self.d.move_to_end(video)
                return self.d[video]
        ent = grid_frames(video)
        with self.lock:
            self.d[video] = ent
            while len(self.d) > self.size:
                self.d.popitem(last=False)
        return ent
