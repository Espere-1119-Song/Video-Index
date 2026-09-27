"""Text embeddings (sentence-transformers) for the near-duplicate check and the learned attacker.
Default model BAAI/bge-large-en-v1.5 (CLS pooling, L2-normalized, at most 512 tokens); `embedding_model` changes it."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
_ENCODERS = {}


def encoder(name=DEFAULT_MODEL, device=None):
    if (name, device) not in _ENCODERS:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("this stage needs the embeddings extra: pip install 'video-index[embeddings]'") from e
        m = SentenceTransformer(name, device=device)
        m.max_seq_length = min(512, m.max_seq_length or 512)
        _ENCODERS[(name, device)] = m
    return _ENCODERS[(name, device)]


def embed(texts, name=DEFAULT_MODEL, device=None, batch_size=64):
    """L2-normalized float32 embeddings, one row per text."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    v = encoder(name, device).encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                     show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(v, dtype=np.float32)


def embed_cached(ctx, bench, kind, ids, texts):
    """Embeddings of one benchmark, cached under <work_dir>/embeddings/<kind>/; the cache key covers the model name,
    the ids and the texts, so a changed sample is embedded again."""
    name = ctx.get("embedding_model", DEFAULT_MODEL)
    key = hashlib.sha256(json.dumps([name, list(ids), list(texts)], ensure_ascii=False).encode()).hexdigest()[:16]
    d = os.path.join(ctx.work_dir, "embeddings", kind)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{bench}__{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    v = embed(texts, name, ctx.get("embedding_device"), int(ctx.get("embedding_batch", 64)))
    np.save(path, v)
    return v
