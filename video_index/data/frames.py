"""Reading frames from the stored videos with OpenCV.

Frames are decoded in stream order (grab every frame up to the last wanted index and retrieve the wanted ones), since
random access returns a neighbouring frame on some videos; the stored videos hold at most 1,024 frames. Frames are
PIL images in RGB, scaled down (never up) with bicubic resampling so that the short side is at most ``short_side``
and the long side at most ``long_side``. The timestamp of stored frame i is i / fps."""
from __future__ import annotations


def _open(video: str):
    """(cv2.VideoCapture, stored frame count, frames per second); raises when the file cannot be opened."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise IOError(f"cannot open video: {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rate = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return cap, n, rate


def _resize(img, short_side: int | None = None, long_side: int | None = None):
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    s = 1.0
    if short_side and min(w, h) > short_side:
        s = short_side / min(w, h)
    if long_side and max(w, h) * s > long_side:
        s = long_side / max(w, h)
    if s < 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    return img


def probe(video: str) -> dict:
    """{"n_frames", "fps", "width", "height", "duration_s"} of a stored video."""
    import cv2
    cap, n, rate = _open(video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return dict(n_frames=n, fps=rate, width=w, height=h, duration_s=(n / rate) if rate else None)


def read_indices(video: str, indices, short_side: int | None = None, long_side: int | None = None):
    """(frames, timestamps) at the given stored-frame indices, in the given order. Indices past the last decodable
    frame are dropped."""
    import cv2
    from PIL import Image
    want = [int(i) for i in indices]
    if not want:
        return [], []
    cap, _, rate = _open(video)
    wanted, got = set(want), {}
    try:
        for i in range(max(wanted) + 1):
            if not cap.grab():
                break
            if i in wanted:
                ok, fr = cap.retrieve()
                if ok:
                    got[i] = _resize(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), short_side, long_side)
    finally:
        cap.release()
    rate = rate or 1.0
    idx = [i for i in want if i in got]
    return [got[i] for i in idx], [i / rate for i in idx]


def subsample(frames: list, timestamps: list, n: int):
    """Thin frames and their timestamps uniformly to at most n, keeping the first and the last (the thinning of the
    frame rule in :func:`video_index.evaluate.protocol.one_fps_indices`)."""
    import numpy as np
    frames, timestamps = list(frames), list(timestamps)
    if n <= 0 or not frames:
        return [], []
    if len(frames) <= n:
        return frames, timestamps
    keep = sorted(set(int(round(x)) for x in np.linspace(0, len(frames) - 1, n)))
    return [frames[i] for i in keep], [timestamps[i] for i in keep]


def fps(video: str, rate: float = 1.0, cap: int | None = None, short_side: int | None = None,
        long_side: int | None = None, duration_s: float | None = None):
    """(frames, timestamps) of the frame rule of the VIDEO protocol: the stored frame nearest to every multiple of
    1/rate seconds, every stored frame when the video is stored below the rate, uniform thinning to cap frames.
    Without ``duration_s`` the duration is the stored frame count over the frame rate."""
    from ..evaluate.protocol import one_fps_indices
    info = probe(video)
    idx = one_fps_indices(info["n_frames"], info["fps"], duration_s or info["duration_s"], rate, cap)
    return read_indices(video, idx, short_side, long_side)
