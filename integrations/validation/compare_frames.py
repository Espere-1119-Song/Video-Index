"""Frames of a slice as read by vi-eval (OpenCV, stream order) and by the toolkit integrations (decord):
frame indices, sizes and pixel differences, for decord in stream order (the integrations) and with random access.

    python compare_frames.py <dataset directory> <start> <stop> <frame cap> [<VLMEvalKit frame directory>]

Needs opencv, decord and pillow, and `video_index` on the Python path.
"""
import glob
import importlib.util
import json
import os
import sys

import decord
import numpy as np
from PIL import Image

from video_index.data import frames as F
from video_index.evaluate import protocol as P


def _helper():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vlmevalkit", "vlmeval", "dataset", "utils",
                        "video_index.py")
    spec = importlib.util.spec_from_file_location("vi_integration_helper", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resize(frame, short_side=224):
    im = Image.fromarray(frame).convert("RGB")
    w, h = im.size
    if min(w, h) > short_side:
        s = short_side / min(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    return im


def _diff(a, b):
    if len(a) != len(b) or any(x.size != y.size for x, y in zip(a, b)):
        return "size mismatch"
    return f"{float(np.mean([np.abs(np.asarray(x, dtype=np.int16) - np.asarray(y, dtype=np.int16)).mean() for x, y in zip(a, b)])):.2f}"


def main():
    root, lo, hi, cap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    frame_dir = sys.argv[5] if len(sys.argv) > 5 else None
    helper = _helper()
    items = [json.loads(line) for line in open(os.path.join(root, "items/meta_benchmark.jsonl"))][lo:hi]
    print("Mean absolute pixel difference to the frames of vi-eval, scale 0-255.\n")
    print("| Item | Frames vi-eval / toolkit | Same indices | Frame size | decord, stream order | decord, random access | VLMEvalKit JPEG files |")
    print("|---|---|---|---|---|---|---|")
    for it in items:
        video = os.path.join(root, it["video"])
        info = F.probe(video)
        idx = P.one_fps_indices(info["n_frames"], info["fps"], it.get("duration_s"), 1.0, cap)
        ours, _ = F.read_indices(video, idx, 224)
        reader = decord.VideoReader(video, ctx=decord.cpu(0), num_threads=1)
        idx2 = helper.one_fps_indices(len(reader), reader.get_avg_fps(), it.get("duration_s"), rate=1.0, cap=cap)
        stream = [_resize(f) for f in helper.read_frames(reader, idx2)]
        reader = decord.VideoReader(video, ctx=decord.cpu(0), num_threads=1)
        direct = [_resize(reader[i].asnumpy()) for i in idx2]
        files = []
        if frame_dir:
            files = sorted(glob.glob(os.path.join(frame_dir, it["video_id"], f"frame-*-of-{len(idx2)}-1fps-cap{cap}-s224.jpg")),
                           key=lambda p: int(os.path.basename(p).split("-")[1]))
        jpeg = _diff(ours, [Image.open(f).convert("RGB") for f in files]) if files else "not given"
        print(f"| {it['item_id']} | {len(ours)} / {len(stream)} | {'yes' if list(idx) == list(idx2) else 'no'} | "
              f"{ours[0].size[0]}x{ours[0].size[1]} | {_diff(ours, stream)} | {_diff(ours, direct)} | {jpeg} |")


if __name__ == "__main__":
    main()
