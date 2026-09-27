"""Visual scene clusters and question templates, the two inputs of the selection caps.

Scene clusters
    Video vector = mean SigLIP2 vector of 32 uniform frames. Videos are compared only inside their duration group
    (<10 s, 10-30 s, 30-60 s, 1-3 min, 3-10 min, 10-30 min, >30 min), because the mean vector of a long video lies
    closer to the global centroid than that of a short clip. Threshold per group: q = share of random pairs of the
    shortest group whose cosine is below T0 = 0.90; every group uses the q-quantile of its own random-pair cosines,
    which gives the same false-match rate in every group. Clustering: greedy leader clustering in a fixed random
    order (seed 0); a video joins the most similar leader when the cosine reaches the threshold.

Question templates
    spaCy (en_core_web_sm): entities -> <ENT>, numbers -> <NUM>, nouns and proper nouns -> <N>, everything else
    lower-cased; runs of one placeholder are collapsed. Without spaCy a regular-expression rule replaces numbers
    and quoted strings only (coarser templates; a warning is logged).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from ..runner import log

EDGES = [0, 10, 30, 60, 180, 600, 1800, 1e12]
GROUPS = ["<10s", "10-30s", "30-60s", "1-3min", "3-10min", "10-30min", ">30min"]
T0 = 0.90
N_PAIRS = 300000


def duration_group(seconds):
    for lo, hi, g in zip(EDGES, EDGES[1:], GROUPS):
        if lo <= seconds < hi:
            return g
    return None


def video_duration(ctx, video_id, fallback=None):
    """Duration in seconds from videos/<video_id>/meta.json, else the fallback (the item's duration)."""
    p = os.path.join(ctx.work_dir, "videos", str(video_id), "meta.json")
    if os.path.exists(p):
        try:
            m = json.load(open(p))
            for k in ("original_duration", "duration_s", "duration"):
                if m.get(k) is not None:
                    return float(m[k])
            if m.get("frame_timestamps"):
                return float(m["frame_timestamps"][-1]) + 0.5
        except Exception:  # noqa: BLE001
            pass
    return float(fallback) if fallback is not None and fallback == fallback else float("nan")


def leader(X, threshold, order):
    """Greedy leader clustering -> (cluster index per row, number of clusters)."""
    n, d = X.shape
    leaders = np.zeros((n, d), dtype=np.float32)
    k = 0
    assign = np.empty(n, dtype=np.int64)
    for i in order:
        if k:
            s = leaders[:k] @ X[i]
            j = int(np.argmax(s))
            if s[j] >= threshold:
                assign[i] = j
                continue
        leaders[k] = X[i]
        assign[i] = k
        k += 1
    return assign, k


def random_pair_cos(X, n_pairs, rng):
    n = len(X)
    if n < 2:
        return np.array([])
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    m = i != j
    return np.einsum("ij,ij->i", X[i[m]], X[j[m]])


def scene_clusters(groups, X, t0=T0, n_pairs=N_PAIRS, seed=0):
    """groups: duration group per row of X (mean frame vectors). -> (cluster_cal, cluster_fixed, table rows).
    Cluster ids are unique over all groups."""
    groups = np.asarray(groups)
    X = np.asarray(X, dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(seed)
    rc0 = random_pair_cos(X[groups == GROUPS[0]], n_pairs, rng)
    q = float((rc0 < t0).mean()) if len(rc0) else 1.0
    cal = np.full(len(X), -1, dtype=np.int64)
    fix = np.full(len(X), -1, dtype=np.int64)
    rows, off_c, off_f = [], 0, 0
    for g in GROUPS:
        gi = np.where(groups == g)[0]
        if len(gi) == 0:
            continue
        rc = random_pair_cos(X[gi], n_pairs, rng)
        tg = float(np.quantile(rc, q)) if len(rc) else t0
        order = rng.permutation(len(gi))
        a_c, k_c = leader(X[gi], tg, order)
        a_f, k_f = leader(X[gi], t0, order)
        cal[gi], fix[gi] = a_c + off_c, a_f + off_f
        off_c, off_f = off_c + k_c, off_f + k_f
        sizes = np.bincount(a_c)
        rows.append(dict(group=g, videos=len(gi), rand_pair_cos_median=float(np.median(rc)) if len(rc) else None,
                         threshold_cal=tg, clusters_cal=k_c, singletons_cal=int((sizes == 1).sum()),
                         largest_cal=int(sizes.max()), videos_sharing_cal=int(sizes[sizes > 1].sum()),
                         clusters_fixed=k_f, false_match_rate=1 - q))
    return cal, fix, rows


# ------------------------------------------------------------------ templates
_NUM = re.compile(r"\d+(?:[\.,:]\d+)*")
_QUOTED = re.compile(r"\"[^\"]+\"|'[^']+'")


def _collapse(toks):
    out = []
    for t in toks:
        if not out or out[-1] != t or not t.startswith("<"):
            out.append(t)
    return " ".join(out)


def template_regex(question):
    q = _QUOTED.sub(" <ENT> ", str(question or "")[:800])
    return _collapse([t if t in ("<NUM>", "<ENT>") else t.lower() for t in _NUM.sub(" <NUM> ", q).split()])


def templates_for(ctx, items, procs=1):
    """items: [(item_id, question)] -> {item_id: template}; cached in pool/templates.json."""
    path = ctx.path("pool", name="templates.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    missing = [(i, q) for i, q in items if i not in cache]
    if missing:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except Exception:  # noqa: BLE001
            nlp = None
            log("templates: spaCy model en_core_web_sm not available; regular-expression templates are used", "pool")
        if nlp is None:
            cache.update({i: template_regex(q) for i, q in missing})
        else:
            docs = nlp.pipe([(q or "")[:800] for _, q in missing], batch_size=256, n_process=procs)
            for (i, _), doc in zip(missing, docs):
                cache[i] = _collapse(["<ENT>" if t.ent_type_ else "<NUM>" if (t.like_num or t.pos_ == "NUM")
                                      else "<N>" if t.pos_ in ("NOUN", "PROPN") else t.lower_ for t in doc])
        json.dump(cache, open(path, "w"))
    return {i: cache[i] for i, _ in items}
