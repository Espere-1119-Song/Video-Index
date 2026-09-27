"""Error matrices and benchmark similarity (called by attribution_tables; no model call).

A row is resolved when it has a group: agreement on the category (including rule rows and dense-recheck results)
or on the group. Disputed rows only enter attribution_disputed_rates.csv.

    attribution_matrix.csv          capability groups x benchmarks, every column sums to 1 (data_protocol excluded)
    attribution_matrix_counts.csv   the same with counts
    attribution_matrix_fine.csv     fine categories x benchmarks, rows with agreement on the category
    attribution_protocol_share.csv  share of the data_protocol group among the resolved failures
    attribution_disputed_rates.csv  fine / group-only / disputed rates over the judged rows
    attribution_jaccard.csv         weighted Jaccard between the benchmark columns of attribution_matrix.csv;
                                    benchmarks with fewer than 100 capability failures are marked low_sample
"""
from __future__ import annotations

import csv
from collections import Counter

from ...data.schema import read_jsonl
from . import common as C
from .taxonomy import CAPABILITY_GROUPS, CATEGORIES

LOW_SAMPLE_N = 100
FINE_LABELS = [c for c in CATEGORIES + ["sampling_gap", "evidence_not_in_video", "language_prior_dominated"]
               if c != "evidence_not_in_input_frames"] + ["evidence_not_in_input_frames"]


def load_rows(ctx, benchmarks=None):
    by_key = {}
    for bench in (b for b in C.traced_benchmarks(ctx, "judge") if not benchmarks or b in benchmarks):
        for r in read_jsonl(ctx.path("attribution", stage="judge", bench=bench)):
            r["benchmark"] = bench
            by_key[(bench, r.get("model", "?"), str(r["qid"]))] = r          # last row wins
    return list(by_key.values())


def fine_category(r):
    return r["category"] if r.get("agreed_level") in ("fine", "rule") and r.get("category") else None


def jaccard(a, b):
    mx = sum(max(x, y) for x, y in zip(a, b))
    return sum(min(x, y) for x, y in zip(a, b)) / mx if mx else 0.0


def write_matrix(path, labels, benches, counts, normalize=True):
    extra = sorted({l for b in benches for l in counts[b]} - set(labels))
    labels = list(labels) + extra
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label"] + benches)
        tot = {b: sum(counts[b].get(l, 0) for l in labels) or 1 for b in benches}
        for l in labels:
            w.writerow([l] + [f"{counts[b].get(l, 0) / tot[b]:.4f}" if normalize else counts[b].get(l, 0)
                              for b in benches])
        w.writerow(["n_total"] + [sum(counts[b].get(l, 0) for l in labels) for b in benches])


def run(ctx, benchmarks=None):
    rows = load_rows(ctx, benchmarks)
    if not rows:
        return None
    benches = sorted({r["benchmark"] for r in rows})
    grp = {b: Counter() for b in benches}
    proto = {b: Counter() for b in benches}
    fine = {b: Counter() for b in benches}
    for r in rows:
        b, g, fc = r["benchmark"], r.get("group"), fine_category(r)
        if g in CAPABILITY_GROUPS:
            grp[b][g] += 1
        elif g == "data_protocol":
            proto[b][fc or "protocol_group_only"] += 1
        if fc:
            fine[b][fc] += 1
    t = lambda name: ctx.path("table", name=name)  # noqa: E731
    write_matrix(t("attribution_matrix.csv"), CAPABILITY_GROUPS, benches, grp)
    write_matrix(t("attribution_matrix_counts.csv"), CAPABILITY_GROUPS, benches, grp, normalize=False)
    write_matrix(t("attribution_matrix_fine.csv"), FINE_LABELS, benches, fine)

    cats = sorted({c for pc in proto.values() for c in pc})
    with open(t("attribution_protocol_share.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "n_resolved", "n_data_protocol", "protocol_share"] + cats)
        for b in benches:
            nprot = sum(proto[b].values())
            tot = sum(grp[b].values()) + nprot
            w.writerow([b, tot, nprot, f"{(nprot / tot if tot else 0):.4f}"] + [proto[b].get(c, 0) for c in cats])

    with open(t("attribution_disputed_rates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "n_judged", "fine_rate", "group_only_rate", "disputed_rate", "n_rule_direct",
                    "n_total_attributed"])
        for b in benches:
            br = [r for r in rows if r["benchmark"] == b]
            judged = [r for r in br if r.get("agreed_level") != "rule"]
            n = len(judged) or 1
            lv = Counter(r.get("agreed_level") for r in judged)
            w.writerow([b, len(judged), f"{lv.get('fine', 0) / n:.4f}", f"{lv.get('group', 0) / n:.4f}",
                        f"{lv.get('disputed', 0) / n:.4f}", len(br) - len(judged), len(br)])

    n_cap = {b: sum(grp[b].values()) for b in benches}
    prop = {b: [grp[b].get(g, 0) / (n_cap[b] or 1) for g in CAPABILITY_GROUPS] for b in benches}
    low = {b: n_cap[b] < LOW_SAMPLE_N for b in benches}
    with open(t("attribution_jaccard.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "n_failures", "n_capability", "low_sample"] + benches)
        for b1 in benches:
            w.writerow([b1, sum(1 for r in rows if r["benchmark"] == b1), n_cap[b1], str(low[b1]).lower()]
                       + [f"{jaccard(prop[b1], prop[b2]):.4f}" for b2 in benches])
    vals = [jaccard(prop[b1], prop[b2]) for i, b1 in enumerate(benches) for b2 in benches[i + 1:]
            if not low[b1] and not low[b2]]
    return dict(mean_jaccard=(sum(vals) / len(vals)) if vals else None, n_pairs=len(vals),
                low_sample=[b for b in benches if low[b]])
