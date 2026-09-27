"""Screen stage: item-level checks of the sample. No gold answer is changed.

duplicate   near-duplicates within a benchmark: same video AND cosine >= 0.90 between the question embeddings AND
            identical normalized option set and answer. Connected components keep the lexicographically smallest qid;
            the other members are removed (action = remove).
scoring     option-count repair: for a benchmark with declared option counts, an item whose option list has another
            length is recorded (action = repair_num_options, detail num_options:<declared>-><actual>); its chance is
            1 / (actual count), which the chance stage already uses.
Outputs     tables/screen_items.csv   benchmark, item_id, screen_type, action, repair_detail
            tables/screen_counts.csv  benchmark, n_sampled, n_duplicate_removed, n_option_count_repaired, n_after_screen
"""
from __future__ import annotations

import csv
import os
import re
from collections import defaultdict

import numpy as np

from ..runner import log
from ..scoring import coerce_options, norm_text
from . import chance as chance_mod
from .items import bench_meta, load_samples, select, video_key

COS_DUP = 0.90
ITEM_COLS = ["benchmark", "item_id", "screen_type", "action", "repair_detail"]
COUNT_COLS = ["benchmark", "n_sampled", "n_duplicate_removed", "n_option_count_repaired", "n_after_screen"]


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def q_text(s):
    t = str(s.get("question") or "").strip()
    if not t:
        o = s.get("options")
        t = " ".join(str(x) for x in o) if isinstance(o, list) else str(o or "")
    return " ".join(t.split())


def opt_sig(s):
    o = coerce_options(s.get("options"))
    if not isinstance(o, list):
        return None
    return tuple(sorted(norm_text(re.sub(r"^\s*[\(\[]?[A-Pa-p][\)\]\.,:]\s+", "", str(x))) for x in o))


def identity(s):
    """Video identity of an item: the content hash when the video is normalized, else the source key."""
    return s.get("video_id") or ("vk:" + video_key(s))


def guard(s):
    return identity(s), opt_sig(s), norm_text(s.get("answer"))


def find_duplicates(samples, embed, cos_dup=COS_DUP):
    """samples {qid: row}; embed(texts) -> normalized vectors. -> {removed qid: (kept qid, cosine or None)}.
    Only items that share video, option set and answer with another item are embedded."""
    groups = defaultdict(list)
    for qid, s in samples.items():
        groups[guard(s)].append(qid)
    dsu, pair_cos = DSU(), {}
    for qids in groups.values():
        if len(qids) < 2:
            continue
        M = np.asarray(embed([q_text(samples[q]) for q in qids]), dtype=np.float32)
        S = M @ M.T
        for a in range(len(qids)):
            for b in range(a + 1, len(qids)):
                if float(S[a, b]) >= cos_dup:
                    dsu.union(qids[a], qids[b])
                    pair_cos[(qids[a], qids[b])] = pair_cos[(qids[b], qids[a])] = float(S[a, b])
    comps = defaultdict(list)
    for q in list(dsu.p):
        comps[dsu.find(q)].append(q)
    out = {}
    for members in comps.values():
        if len(members) < 2:
            continue
        members.sort()
        for q in members[1:]:
            out[q] = (members[0], pair_cos.get((q, members[0])))
    return out


def option_count_repairs(samples, meta):
    decl = chance_mod.declared_counts(meta or {})
    out = {}
    if not decl:
        return out
    for qid, s in samples.items():
        o = coerce_options(s.get("options"))
        n = len(o) if isinstance(o, list) else 0
        if n >= 2 and n not in decl:
            out[qid] = f"num_options:{'|'.join(map(str, decl))}->{n}"
    return out


def embedder(ctx):
    from .embeddings import DEFAULT_MODEL, embed
    return lambda texts: embed(texts, ctx.get("embedding_model", DEFAULT_MODEL), ctx.get("embedding_device"),
                               int(ctx.get("embedding_batch", 64)))


def duplicate_checker(ctx):
    """is_duplicate(candidate, kept rows) for the refill rule."""
    emb, cos_dup, cache = embedder(ctx), float(ctx.get("duplicate_cos", COS_DUP)), {}

    def vec(r):
        if r["qid"] not in cache:
            cache[r["qid"]] = emb([q_text(r)])[0]
        return cache[r["qid"]]

    def is_duplicate(r, kept):
        g = guard(r)
        return any(float(vec(r) @ vec(k)) >= cos_dup for k in kept if guard(k) == g)
    return is_duplicate


def _merge(path, cols, rows, benches):
    old = []
    if os.path.exists(path):
        old = [r for r in csv.DictReader(open(path)) if r["benchmark"] not in benches]
    rows = sorted(old + rows, key=lambda r: (r["benchmark"], r.get("item_id", ""), r.get("screen_type", "")))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run(ctx, benchmarks=None, limit=None):
    items, counts, benches = [], [], set()
    emb = embedder(ctx)
    for bench in select(ctx, benchmarks):
        samples = load_samples(ctx, bench, limit)
        if not samples:
            continue
        benches.add(bench)
        dups = find_duplicates(samples, emb, float(ctx.get("duplicate_cos", COS_DUP)))
        reps = option_count_repairs(samples, bench_meta(ctx, bench))
        for q, (kept, c) in sorted(dups.items()):
            items.append(dict(benchmark=bench, item_id=q, screen_type="duplicate", action="remove",
                              repair_detail=f"kept={kept};cos={c:.3f}" if c is not None else f"kept={kept};cos=transitive"))
        for q, d in sorted(reps.items()):
            items.append(dict(benchmark=bench, item_id=q, screen_type="scoring", action="repair_num_options",
                              repair_detail=d))
        counts.append(dict(benchmark=bench, n_sampled=len(samples), n_duplicate_removed=len(dups),
                           n_option_count_repaired=len(reps), n_after_screen=len(samples) - len(dups)))
        log(f"screen: {bench}: {len(dups)} near-duplicates removed, {len(reps)} option counts repaired", "audit")
    _merge(ctx.path("table", name="screen_items.csv"), ITEM_COLS, items, benches)
    _merge(ctx.path("table", name="screen_counts.csv"), COUNT_COLS, counts, benches)
    return counts
