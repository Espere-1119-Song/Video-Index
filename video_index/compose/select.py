"""Stage compose_select: the item list of Video-Index.

Per capability group (perception, temporal, spatial_physical, reasoning_knowledge) the verified candidates are
taken in the order of the worst-case attacker percentile (hardest first, ties by item_id) until the quota
(budget / 4 = 210), one item per video over the whole list. No model's accuracy constrains the selection.

Options
    compose.reference_cap       share of a group that the reference model may answer correctly (the released list
                                uses no cap). With a cap, the correctness of the reference model on the candidates
                                is read from pool/compose/reference_correct.json ({item_id: true | false});
                                candidates without an entry are not eligible; a group that stays short under the
                                cap is filled without it.
    compose.fill_short_groups   a group without enough verified candidates passes its shortfall to the groups with
                                the largest supply (default: off, the shortfall is reported).

Output pool/compose/video_index.json (the item list) and pool/compose/video_index_summary.csv.
"""
from __future__ import annotations

import csv
import json
import os
import statistics
from collections import Counter

from ..runner import log
from . import CAPS, cfg, path
from .items import load, save

FIELDS = ["item_id", "benchmark", "question", "options", "answer_idx", "video_id", "chance", "attack_pct", "task_class",
          "capability", "verdict", "keep_reason", "source"]


def compose(verified, budget=840, caps=CAPS, reference_cap=None, correct=None, fill_short_groups=False):
    """verified: candidate records that passed the verification. -> (selected records, report).
    Groups are filled in the order of `caps`; a video is used once."""
    quota = budget // len(caps)
    cap_n = None if reference_cap is None or reference_cap >= 1 else int(reference_cap * quota)
    pool, seen_items = [], set()
    for it in verified:
        if it.get("keep", True) and it["item_id"] not in seen_items and (cap_n is None or it["item_id"] in correct):
            seen_items.add(it["item_id"])
            pool.append(it)
    cand = {c: sorted([it for it in pool if it.get("capability") == c], key=lambda r: (r["attack_pct"], r["item_id"]))
            for c in caps}
    taken, seen_videos, relaxed = {c: [] for c in caps}, set(), {}

    def fill(c, n, capped):
        n_right = sum(1 for it in taken[c] if cap_n is not None and correct.get(it["item_id"]))
        have = {it["item_id"] for it in taken[c]}
        for it in cand[c]:
            if len(taken[c]) >= n:
                break
            if it["item_id"] in have or it["video_id"] in seen_videos:
                continue
            if capped and correct.get(it["item_id"]):
                if n_right >= cap_n:
                    continue
                n_right += 1
            seen_videos.add(it["video_id"])
            taken[c].append(it)
        return n_right

    for c in caps:
        fill(c, quota, cap_n is not None)
        if cap_n is not None and len(taken[c]) < quota:
            relaxed[c] = fill(c, quota, False)
    short = budget - sum(len(v) for v in taken.values())
    if fill_short_groups and short > 0:
        for c in sorted(caps, key=lambda c: -len(cand[c])):
            fill(c, len(taken[c]) + short, False)
            short = budget - sum(len(v) for v in taken.values())
            if short <= 0:
                break
    items = [{k: it.get(k) for k in FIELDS} for c in caps for it in taken[c]]
    report = dict(n_items=len(items), n_verified=len(pool), quota=quota, per_group={c: len(taken[c]) for c in caps},
                  supply={c: len(cand[c]) for c in caps}, shortfall=budget - len(items), cap_relaxed=relaxed,
                  n_benchmarks=len({it["benchmark"] for it in items}),
                  n_videos=len({it["video_id"] for it in items}),
                  reasons=dict(Counter(it["keep_reason"] for it in items)))
    if items:
        report.update(attack_pct_median=statistics.median(it["attack_pct"] for it in items),
                      attack_pct_max=max(it["attack_pct"] for it in items),
                      mean_chance=statistics.mean(it["chance"] for it in items))
    return items, report


def run(ctx, benchmarks=None, limit=None):
    verified = load(path(ctx, "verified.json"))
    if benchmarks:
        verified = [it for it in verified if it["benchmark"] in set(benchmarks)]
    cap, correct = cfg(ctx, "reference_cap"), None
    if cap is not None and cap < 1:
        p = path(ctx, "reference_correct.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"compose.reference_cap needs {p} ({{item_id: true | false}})")
        correct = json.load(open(p))
    items, report = compose(verified, limit or int(cfg(ctx, "budget")), CAPS, cap, correct,
                            bool(cfg(ctx, "fill_short_groups")))
    save(path(ctx, "video_index.json"), items)
    by_b = Counter(it["benchmark"] for it in items)
    with open(path(ctx, "video_index_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "n", "attack_pct_mean", "chance"])
        for b, n in by_b.most_common():
            rows = [it for it in items if it["benchmark"] == b]
            w.writerow([b, n, f"{statistics.mean(r['attack_pct'] for r in rows):.4f}",
                        f"{statistics.mean(r['chance'] for r in rows):.4f}"])
    json.dump(report, open(path(ctx, "video_index_report.json"), "w"), indent=1)
    log(f"compose_select: {report}", "compose")
    return report
