"""Duplicate and option-format screens of the pool (no model call; the duplicate screens need text embeddings).

within benchmark   two multiple-choice items of one benchmark are duplicates when they sit on the same video and the
                   cosine of their text embeddings exceeds 0.90; connected components; the smallest item_id is kept
across benchmarks  same rule between items of different benchmarks (same video, cosine >= 0.90), applied last, among
                   the items that are still in the pool; the copy of the benchmark with the earliest release year is
                   kept, ties by item_id
option format      items with a meta option ("all of the above", "both A and B", "cannot be determined", ...),
                   a repeated option text or an empty correct option
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

COS_DUP = 0.90

META_OPTION = re.compile(
    r"^\s*(?:\(?[a-p][\.\)]\s*)?(?:"
    r"(?:all|none|both|neither|any|either)\s+of\s+(?:the\s+)?(?:above|these|them|the\s+options|the\s+choices)"
    r"|all\s+(?:of\s+)?the\s+above|none\s+of\s+the\s+above|all\s+the\s+above"
    r"|both\s+[a-p]\s+and\s+[a-p]|[a-p]\s+and\s+[a-p]\s*(?:are\s+correct)?|[a-p]\s*(?:,|&|and)\s*[a-p]\s*(?:,|&|and)\s*[a-p]"
    r"|(?:cannot|can\s*not|can't)\s+be\s+determined\b.*|not\s+(?:enough|sufficient)\s+information\b.*"
    r"|insufficient\s+information\b.*|not\s+mentioned\b.*|not\s+shown\s+in\s+the\s+video\b.*|none|all|other|others"
    r")\s*[\.\)]?\s*$", re.I)


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

    def components(self):
        comps = defaultdict(list)
        for x in list(self.p):
            comps[self.find(x)].append(x)
        return [sorted(m) for m in comps.values() if len(m) > 1]


def video_groups(item_ids, video_ids):
    groups = defaultdict(list)
    for i, v in zip(item_ids, video_ids):
        if v:
            groups[v].append(i)
    return {v: ids for v, ids in groups.items() if len(ids) >= 2}


def within_benchmark(item_ids, video_ids, index, vecs, cos=COS_DUP):
    """-> ({removed item_id: (kept item_id, cosine to it or None)}, number of groups, item_ids without a vector).
    index: {item_id: row of vecs}; vecs are L2-normalized."""
    dsu, pair_cos, unembedded = DSU(), {}, set()
    for ids in video_groups(item_ids, video_ids).values():
        unembedded.update(i for i in ids if i not in index)
        ids = [i for i in ids if i in index]
        if len(ids) < 2:
            continue
        X = vecs[[index[i] for i in ids]]
        S = X @ X.T
        for p, q in zip(*np.triu_indices(len(ids), 1)):
            if S[p, q] > cos:
                dsu.union(ids[p], ids[q])
                pair_cos[(ids[p], ids[q])] = pair_cos[(ids[q], ids[p])] = float(S[p, q])
    removed, comps = {}, dsu.components()
    for members in comps:
        for q in members[1:]:
            removed[q] = (members[0], pair_cos.get((q, members[0])))
    return removed, len(comps), unembedded


def cross_benchmark_pairs(item_ids, benchmarks, video_ids, index, vecs, cos=COS_DUP):
    """Pairs (a, b, cosine) of items of different benchmarks on the same video with cosine >= cos."""
    bench_of = dict(zip(item_ids, benchmarks))
    pairs = []
    for ids in video_groups(item_ids, video_ids).values():
        ids = [i for i in ids if i in index]
        if len({bench_of[i] for i in ids}) < 2:
            continue
        X = vecs[[index[i] for i in ids]]
        S = X @ X.T
        for p, q in zip(*np.triu_indices(len(ids), 1)):
            if S[p, q] >= cos and bench_of[ids[p]] != bench_of[ids[q]]:
                pairs.append((ids[p], ids[q], round(float(S[p, q]), 4)))
    return pairs


def cross_benchmark_drop(pairs, alive, bench_of, year_of):
    """{removed item_id: kept item_id} among the items in `alive`."""
    dsu = DSU()
    for a, b, *_ in pairs:
        if a in alive and b in alive:
            dsu.union(a, b)
    drop = {}
    for members in dsu.components():
        keep = min(members, key=lambda x: (year_of.get(bench_of[x], 9999), x))
        drop.update({x: keep for x in members if x != keep})
    return drop


def _norm(t):
    return re.sub(r"\s+", " ", re.sub(r"^\s*[\(\[]?[A-Pa-p][\)\]\.:]\s*", "", str(t))).strip().lower()


def option_format_reasons(options, answer_idx):
    """[(reason, detail)] of one item; empty when the option list passes."""
    o = [str(x) for x in (options if options is not None else [])]
    texts = [_norm(x) for x in o]
    reasons = []
    metas = [x for x in o if META_OPTION.match(x)]
    if metas:
        reasons.append(("meta_option", metas[0][:60]))
    if len(set(texts)) < len(texts):
        reasons.append(("repeated_option_text", [t for t in texts if texts.count(t) > 1][0][:60]))
    try:
        ai = int(answer_idx)
    except Exception:  # noqa: BLE001
        ai = -1
    if 0 <= ai < len(texts) and not texts[ai]:
        reasons.append(("empty_gold_option", ""))
    return reasons


def gold_position_share(answer_idx):
    """Largest share of one answer position among the items of a benchmark (None without items)."""
    pos = Counter(int(a) for a in answer_idx if a is not None and int(a) >= 0)
    n = sum(pos.values())
    return max(pos.values()) / n if n else None
