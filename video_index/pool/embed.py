"""Embeddings of the pool (optional dependency group ``embeddings``).

text    BAAI/bge-large-en-v1.5 on question + options: CLS pooling, L2 norm, at most 512 tokens
        (video_index.audit.embeddings, the encoder of the audit's near-duplicate check)
visual  google/siglip2-so400m-patch14-384: pooled image features of the 32 uniform frames of the reference grid
        (long side 512), mean over the frames; the vectors are normalized when they are compared

Both caches are incremental: pool/embeddings/text/<benchmark>.(ids.json|npy), pool/embeddings/visual.(ids.json|npy).
The model names and the device are read from ``pool`` in the run configuration (text_embedding_model,
visual_embedding_model, device). `set_encoders(ctx, text=..., video=...)` replaces the encoders of one run context.
The module keeps no state between runs: the encoders are resolved from the context on every call.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..audit import frame_rules
from ..runner import log

TEXT_MODEL = "BAAI/bge-large-en-v1.5"
VISUAL_MODEL = "google/siglip2-so400m-patch14-384"
N_FRAMES = 32
LONG_SIDE = 512


def _encoders(ctx):
    if getattr(ctx, "_pool_encoders", None) is None:
        ctx._pool_encoders = {}
    return ctx._pool_encoders


def set_encoders(ctx, text=None, video=None):
    """Replace the encoders of this run context: text(list of str) -> array [n, d]; video(path) -> array [d]."""
    enc = _encoders(ctx)
    if text is not None:
        enc["text"] = text
    if video is not None:
        enc["video"] = video


def encoder(ctx, kind):
    """The encoder of this context; built from the run configuration on first use."""
    enc = _encoders(ctx)
    if kind not in enc:
        if kind == "text":
            enc[kind] = text_encoder(_cfg(ctx, "text_embedding_model", TEXT_MODEL), _cfg(ctx, "device", None))
        else:
            enc[kind] = VideoEncoder(_cfg(ctx, "visual_embedding_model", VISUAL_MODEL), _device(ctx))
    return enc[kind]


def _cfg(ctx, key, default):
    return (ctx.get("pool") or {}).get(key, default)


def _device(ctx):
    dev = _cfg(ctx, "device", None)
    if dev:
        return dev
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def text_encoder(name=TEXT_MODEL, device=None):
    from ..audit.embeddings import embed
    return lambda texts: embed([t or " " for t in texts], name, device)


class VideoEncoder:
    def __init__(self, name=VISUAL_MODEL, device="cpu"):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.device = torch, device
        self.dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.proc = AutoProcessor.from_pretrained(name, use_fast=True).image_processor
        self.model = AutoModel.from_pretrained(name, torch_dtype=self.dtype).to(device).eval()

    def __call__(self, video):
        n = frame_rules.timestamps_of(video, None)[1]
        idx = frame_rules.reference_indices("uniform", n, N_FRAMES, None)[0]
        images, _ = frame_rules.decode(video, idx, long_side=LONG_SIDE)
        if not images:
            raise RuntimeError("no frames decoded")
        with self.torch.no_grad():
            px = self.proc(images=images, return_tensors="pt")["pixel_values"].to(self.device, dtype=self.dtype)
            v = self.model.get_image_features(pixel_values=px)
        return v.float().cpu().numpy().mean(0).astype(np.float32)


def _load(stem):
    ip, vp = stem + ".ids.json", stem + ".npy"
    if os.path.exists(ip) and os.path.exists(vp):
        ids, vecs = json.load(open(ip)), np.load(vp)
        if len(ids) == len(vecs):
            return ids, vecs
    return [], None


def _save(stem, ids, vecs):
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    np.save(stem + ".tmp.npy", vecs)
    os.replace(stem + ".tmp.npy", stem + ".npy")
    json.dump(ids, open(stem + ".ids.json.tmp", "w"))
    os.replace(stem + ".ids.json.tmp", stem + ".ids.json")


def text_embeddings(ctx, bench, pairs):
    """pairs: [(item_id, text)] -> ({item_id: row}, matrix); missing texts are embedded and added to the cache."""
    stem = os.path.join(ctx.work_dir, "pool", "embeddings", "text", bench)
    ids, vecs = _load(stem)
    have = set(ids)
    todo = [(i, t) for i, t in pairs if i not in have]
    if todo:
        new = np.asarray(encoder(ctx, "text")([t for _, t in todo]), dtype=np.float32)
        ids = ids + [i for i, _ in todo]
        vecs = new if vecs is None else np.concatenate([vecs, new])
        _save(stem, ids, vecs)
        log(f"embeddings/{bench}: {len(todo)} texts added", "pool")
    if vecs is None:
        return {}, np.zeros((0, 1), np.float32)
    return {k: n for n, k in enumerate(ids)}, vecs.astype(np.float32)


def video_embeddings(ctx, videos):
    """videos: {video_id: path} -> ({video_id: row}, matrix of mean frame vectors, not normalized)."""
    stem = os.path.join(ctx.work_dir, "pool", "embeddings", "visual")
    ids, vecs = _load(stem)
    have = set(ids)
    todo = [(v, p) for v, p in sorted(videos.items()) if v not in have and p]
    new_ids, new = [], []
    for k, (vid, path) in enumerate(todo, 1):
        try:
            new.append(np.asarray(encoder(ctx, "video")(path), dtype=np.float32))
            new_ids.append(vid)
        except Exception as e:  # noqa: BLE001
            log(f"embeddings/visual: {vid}: {type(e).__name__}: {str(e)[:120]}", "pool")
        if k % 500 == 0 or k == len(todo):
            if new:
                ids = ids + new_ids
                vecs = np.stack(new) if vecs is None else np.concatenate([vecs, np.stack(new)])
                _save(stem, ids, vecs)
                new_ids, new = [], []
            log(f"embeddings/visual: {k}/{len(todo)} videos", "pool")
    if vecs is None:
        return {}, np.zeros((0, 1), np.float32)
    return {k: n for n, k in enumerate(ids)}, vecs.astype(np.float32)
