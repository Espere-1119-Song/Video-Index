"""Stage attribution_tables: how the attributed failures distribute over annotation, coverage and capability.

Every attributed instance (model, qid) gets one status:

    unstable    turned correct on the moved frame grid (attribution_checks)
    disputed    the judges disagree, or agree only on the data_protocol group with categories of different groups
    unverified  evidence_not_in_input_frames without a conclusive dense recheck (none, skipped, or disputed)
    other       category ``other``
    ok          annotation | coverage | capability (with the capability subgroup)

Precedence: unstable > disputed > (unverified | other | ok). The denominator of every share is the number of ``ok``
instances. Multiple-choice items only (video_index.audit.chance.is_mcq_item). The shares describe the failures; no
item is removed or corrected by them.

Outputs (tables/):
    attribution_shares.csv           per benchmark
    attribution_shares_pooled.csv    all benchmarks, instance-weighted and item-weighted
    attribution_fine.csv             per benchmark x fine category within the capability group
    attribution_fine_pooled.csv
    attribution_row_labels.csv       one row per instance: status, group, subgroup, the rule that applied
    attribution_matrix.csv, attribution_matrix_counts.csv, attribution_matrix_fine.csv,
    attribution_protocol_share.csv, attribution_disputed_rates.csv, attribution_jaccard.csv (matrix.py)
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict

from ...data.schema import read_jsonl
from ...runner import log
from ..chance import mcq_items
from ..items import bench_meta, load_samples
from . import common as C
from . import matrix
from .checks import unstable_keys
from .taxonomy import (CAPABILITY_GROUPS, DENSE_FINAL_GROUP, FINE_GROUP, GROUP_OF_SUB, SUBGROUPS, TIE_ORDER)

COUNT_COLS = ["n_errors", "n_unstable", "n_disputed", "n_disputed_group_split", "n_unverified",
              "n_unverified_skipped", "n_unverified_dense_disputed", "n_other", "n_denominator", "n_annotation_group",
              "n_annotation", "n_scoring", "n_coverage", "n_capability", "n_perception", "n_temporal", "n_spatial",
              "n_reasoning"]
SHARE_COLS = ["annotation_group_share", "annotation_share", "scoring_share", "coverage_share", "capability_share",
              "perception_share", "temporal_share", "spatial_share", "reasoning_share"]
SUB_TO_COL = {"annotation": "n_annotation", "scoring": "n_scoring", "coverage": "n_coverage",
              "perception": "n_perception", "temporal": "n_temporal", "spatial_physical": "n_spatial",
              "reasoning_knowledge": "n_reasoning"}
STATUS_TIE = ["disputed", "unstable", "unverified", "other"]
LABEL_COLS = ["benchmark", "qid", "model", "agreed_level", "category", "judge_group", "dense_state", "status", "group",
              "subgroup", "basis", "judge_categories"]
UNDET = "undetermined"
FINE_SUB = {c: s for c, (g, s) in FINE_GROUP.items() if g == "capability"}
FINE_BY_SUB = {s: [c for c, ss in FINE_SUB.items() if ss == s] for s in CAPABILITY_GROUPS}
FINE_COLS = ["benchmark", "subgroup", "fine_category", "count", "share", "share_of_capability", "n_capability",
             "n_fine_determined", "n_undetermined"]
FINE_POOLED_COLS = ["subgroup", "fine_category", "count", "share", "share_of_capability", "share_in_subgroup",
                    "n_benchmarks_present", "n_capability", "n_fine_determined", "n_undetermined",
                    "n_subgroup_capability", "n_subgroup_fine_determined"]


# ------------------------------------------------------------------ row classification
def dense_state(d):
    """none | skipped | dense_disputed | final:<resolution>"""
    if not d:
        return "none"
    if d.get("skipped"):
        return "skipped"
    if d.get("disputed"):
        return "dense_disputed"
    if d.get("final"):
        return f"final:{d['final']}"
    return "none"


def classify_fine(category, dense):
    """(status, group, subgroup, basis) of one fine category and its dense-recheck record."""
    if category is None:
        return ("disputed", "", "", "no_category")
    if category == "other":
        return ("other", "", "", "fine:other")
    st = dense_state(dense)
    if st.startswith("final:"):
        res = st.split(":", 1)[1]
        if res in DENSE_FINAL_GROUP and category in ("evidence_not_in_input_frames", res):
            g, s = DENSE_FINAL_GROUP[res]
            return ("ok", g, s, f"dense_{st}")
    if category == "evidence_not_in_input_frames":
        return ("unverified", "", "", f"dense:{st}")
    if category in FINE_GROUP:
        g, s = FINE_GROUP[category]
        return ("ok", g, s, f"fine:{category}")
    raise KeyError(f"category without a group: {category!r}")


def classify_row(row, unstable, bench):
    by = row.get("by") or {}
    jcats = "|".join(f"{j}:{v.get('category')}" for j, v in sorted(by.items()))
    if (bench, row.get("model"), str(row["qid"])) in unstable:
        return dict(status="unstable", group="", subgroup="", basis="check:flipped_correct", judge_categories=jcats)
    if row.get("agreed_level") == "disputed":
        return dict(status="disputed", group="", subgroup="", basis="agreed_level:disputed", judge_categories=jcats)
    cat = row.get("category")
    if cat is not None:
        st, g, s, basis = classify_fine(cat, row.get("dense_recheck"))
        return dict(status=st, group=g, subgroup=s, basis=basis, judge_categories=jcats)
    grp = row.get("group")                         # agreement at group level only
    if grp in CAPABILITY_GROUPS:
        return dict(status="ok", group="capability", subgroup=grp, basis=f"group:{grp}", judge_categories=jcats)
    cats = "|".join(sorted(v.get("category") or "" for v in by.values()))
    if grp == "data_protocol":
        labs = [classify_fine(v.get("category"), None) for v in by.values()]
        if len(labs) >= 2 and len({l[:3] for l in labs}) == 1:
            st, g, s, _ = labs[0]
            return dict(status=st, group=g, subgroup=s, basis="group_both:" + cats, judge_categories=jcats)
        return dict(status="disputed", group="", subgroup="", basis="group_split:" + cats, judge_categories=jcats)
    return dict(status="disputed", group="", subgroup="", basis=f"unmapped_group:{grp}", judge_categories=jcats)


# ------------------------------------------------------------------ counts and shares
def count_labels(labels):
    c = Counter({k: 0 for k in COUNT_COLS})
    for l in labels:
        c["n_errors"] += 1
        st = l["status"]
        if st == "unstable":
            c["n_unstable"] += 1
        elif st == "disputed":
            c["n_disputed"] += 1
            c["n_disputed_group_split"] += l["basis"].startswith("group_split")
        elif st == "unverified":
            c["n_unverified"] += 1
            c["n_unverified_skipped"] += l["basis"] == "dense:skipped"
            c["n_unverified_dense_disputed"] += l["basis"] == "dense:dense_disputed"
        elif st == "other":
            c["n_other"] += 1
        else:
            c["n_denominator"] += 1
            c[SUB_TO_COL[l["subgroup"]]] += 1
    c["n_annotation_group"] = c["n_annotation"] + c["n_scoring"]
    c["n_capability"] = c["n_perception"] + c["n_temporal"] + c["n_spatial"] + c["n_reasoning"]
    return c


def fmt_share(num, den):
    return "NA" if not den else f"{num / den:.4f}"


def shares(c):
    d = c["n_denominator"]
    return {k: fmt_share(c["n_" + k[:-len("_share")]], d) for k in SHARE_COLS}


def item_label(rows):
    """One label per (benchmark, qid) from its instances: the most frequent subgroup of the ok instances (ties:
    annotation, coverage, capability); without an ok instance the most frequent status."""
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        cnt = Counter(r["subgroup"] for r in ok)
        top = max(cnt.values())
        s = sorted([s for s, n in cnt.items() if n == top],
                   key=lambda s: (TIE_ORDER.index(GROUP_OF_SUB[s]), SUBGROUPS.index(s)))[0]
        return dict(status="ok", group=GROUP_OF_SUB[s], subgroup=s, basis=f"item_majority:{cnt[s]}/{len(rows)}")
    cnt = Counter(r["status"] for r in rows)
    top = max(cnt.values())
    st = sorted([s for s, n in cnt.items() if n == top], key=STATUS_TIE.index)[0]
    same = [r["basis"] for r in rows if r["status"] == st]
    if st == "disputed" and all(b.startswith("group_split") for b in same):
        basis = "group_split"
    elif st == "unverified" and len(set(same)) == 1 and same[0] in ("dense:skipped", "dense:dense_disputed"):
        basis = same[0]
    else:
        basis = f"item_majority_status:{st}"
    return dict(status=st, group="", subgroup="", basis=basis)


# ------------------------------------------------------------------ fine categories
def fine_of(lab):
    sub = lab["subgroup"]
    if lab["basis"].startswith("fine:"):
        c = lab["basis"][5:]
        if FINE_SUB.get(c) != sub:
            raise ValueError(f"fine category {c!r} is not in subgroup {sub!r}")
        return sub, c
    return sub, f"{sub}/{UNDET}"


def fine_totals(c, sub=None):
    n_cap = sum(v for (s, f), v in c.items() if sub in (None, s))
    n_undet = sum(v for (s, f), v in c.items() if sub in (None, s) and f.endswith("/" + UNDET))
    return n_cap, n_cap - n_undet, n_undet


def fine_tables(labels):
    """(per-benchmark rows, pooled rows) of the capability instances."""
    benches = sorted({l["benchmark"] for l in labels})
    per = {b: Counter() for b in benches}
    for l in labels:
        if l["status"] == "ok" and l["group"] == "capability":
            per[l["benchmark"]][fine_of(l)] += 1
    pooled = sum(per.values(), Counter())
    n_cap, n_fd, n_undet = fine_totals(pooled)

    def order(sub):
        return sorted(FINE_BY_SUB[sub], key=lambda f: (-pooled.get((sub, f), 0), f)) + [f"{sub}/{UNDET}"]

    rows_b, rows_p = [], []
    for b in benches:
        bc, bfd, bu = fine_totals(per[b])
        for sub in CAPABILITY_GROUPS:
            for f in order(sub):
                n = per[b].get((sub, f), 0)
                rows_b.append(dict(benchmark=b, subgroup=sub, fine_category=f, count=n,
                                   share="NA" if f.endswith(UNDET) else fmt_share(n, bfd),
                                   share_of_capability=fmt_share(n, bc), n_capability=bc, n_fine_determined=bfd,
                                   n_undetermined=bu))
    for sub in CAPABILITY_GROUPS:
        sc, sfd, _ = fine_totals(pooled, sub)
        for f in order(sub):
            n = pooled.get((sub, f), 0)
            undet = f.endswith(UNDET)
            rows_p.append(dict(subgroup=sub, fine_category=f, count=n, share="NA" if undet else fmt_share(n, n_fd),
                               share_of_capability=fmt_share(n, n_cap),
                               share_in_subgroup="NA" if undet else fmt_share(n, sfd),
                               n_benchmarks_present=sum(1 for c in per.values() if c.get((sub, f), 0) > 0),
                               n_capability=n_cap, n_fine_determined=n_fd, n_undetermined=n_undet,
                               n_subgroup_capability=sc, n_subgroup_fine_determined=sfd))
    return rows_b, rows_p


# ------------------------------------------------------------------ stage
def write_csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def collect(ctx, benchmarks=None):
    """(instance labels, per-benchmark rows, number of instances dropped as not multiple-choice)."""
    unstable = unstable_keys(ctx)
    labels, per_bench, n_dropped = [], [], 0
    for bench in (b for b in C.traced_benchmarks(ctx, "judge") if not benchmarks or b in benchmarks):
        rows = {(r.get("model"), str(r["qid"])): r
                for r in read_jsonl(ctx.path("attribution", stage="judge", bench=bench))}
        samples = load_samples(ctx, bench)
        keep = set(mcq_items(samples, bench_meta(ctx, bench))) if samples else None
        if keep is None:
            log(f"tables/{bench}: sample file not found; multiple-choice filter not applied")
        kept = [r for r in rows.values() if keep is None or str(r["qid"]) in keep]
        dropped = len(rows) - len(kept)
        n_dropped += dropped
        if not kept:
            continue
        blabels = [dict(benchmark=bench, qid=r["qid"], model=r.get("model"), agreed_level=r.get("agreed_level"),
                        category=r.get("category") or "", judge_group=r.get("group") or "",
                        dense_state=dense_state(r.get("dense_recheck")), **classify_row(r, unstable, bench))
                   for r in kept]
        labels += blabels
        c = count_labels(blabels)
        per_bench.append(dict(benchmark=bench, n_excluded_non_mcq=dropped, **c, **shares(c)))
    return labels, per_bench, n_dropped


def run(ctx, benchmarks=None, limit=None):
    labels, per_bench, n_dropped = collect(ctx, benchmarks)
    write_csv(ctx.path("table", name="attribution_row_labels.csv"), LABEL_COLS, labels)
    write_csv(ctx.path("table", name="attribution_shares.csv"),
              ["benchmark", "n_excluded_non_mcq"] + COUNT_COLS + SHARE_COLS, per_bench)
    inst = count_labels(labels)
    by_item = defaultdict(list)
    for l in labels:
        by_item[(l["benchmark"], l["qid"])].append(l)
    item = count_labels([item_label(rs) for rs in by_item.values()])
    head = dict(n_benchmarks=len(per_bench), n_models=len({l["model"] for l in labels}), n_items=len(by_item),
                n_excluded_non_mcq=n_dropped)
    write_csv(ctx.path("table", name="attribution_shares_pooled.csv"),
              ["weighting"] + list(head) + COUNT_COLS + SHARE_COLS,
              [dict(weighting="instance_weighted", **head, **inst, **shares(inst)),
               dict(weighting="item_weighted", **head, **item, **shares(item))])
    rows_b, rows_p = fine_tables(labels)
    write_csv(ctx.path("table", name="attribution_fine.csv"), FINE_COLS, rows_b)
    write_csv(ctx.path("table", name="attribution_fine_pooled.csv"), FINE_POOLED_COLS, rows_p)
    matrix.run(ctx, benchmarks)
    log(f"tables: {inst['n_errors']} instances in {len(per_bench)} benchmarks, denominator {inst['n_denominator']}; "
        + ", ".join(f"{k}={v}" for k, v in shares(inst).items()))
    return dict(n_instances=inst["n_errors"], n_denominator=inst["n_denominator"], n_excluded_non_mcq=n_dropped,
                **shares(inst))
