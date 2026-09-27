"""Stage pool_select: reduce the screened pool to the coverage-first selection (10,000 items).

Eligible items: still in the pool, screening complete, video with a duration group and a scene cluster, scene type
labelled, question scorable as single choice.

Selection = round-robin with caps:
    outer loop   the seven duration groups in turn; a group without an admissible item is skipped, so the budget
                 moves to the other groups
    inner loop   scene types in turn inside the group, then benchmarks in turn inside the scene type, then the
                 best admissible item of that benchmark
    item order   inside a benchmark: smaller template group first, then smaller worst-case attacker percentile,
                 then item_id
    caps         K items per video (1); J videos per visual scene cluster (1); M items per (benchmark, template)
                 (5); SHARE of the budget per benchmark (5 %, the per-source cap)
    coverage     a first pass gives every (benchmark, task class) pair one item, rarest pairs first, from the
                 duration group with the fewest selected items; only the per-video and per-benchmark caps apply
    scarce cells (group, scene type) cells whose supply is below their equal share: up to K_SCARCE items per video
                 (3), no cluster or template cap
    relaxation   a group below budget / 7 raises its cluster cap step by step up to J_MAX (4), then doubles the
                 template cap, then is exempt from the per-benchmark cap

Worst-case attacker percentile: each of the five screening margins is ranked among the eligible items
(percentile); the item's largest percentile is used. A small value means that no attacker came close.

Parameters: ``pool`` in the run configuration (budget, k, j, j_max, m, k_scarce, share, max_pct).

Outputs (pool/):
    scene_groups_videos.csv, scene_groups_table.csv       scenes.py
    video_index_selection.csv                             selected items in the order of selection
    video_index_items.jsonl                               the selected items, ready for an evaluation run
    video_index_sources.csv, video_index_composition.csv, video_index_summary.json
"""
from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict, deque

import numpy as np

from ..runner import log
from . import scenes as SC
from . import schema as S
from .grid import video_file
from .screen import MARGINS

GROUPS = SC.GROUPS
DEFAULTS = dict(budget=10000, k=1, j=1, j_max=4, m=5, k_scarce=3, share=0.05, max_pct=None)
UNLABELLED = "(unlabelled)"


def attack_percentile(df, margins=MARGINS):
    """Adds <margin>_pct, attack_margin and attack_pct. A missing margin takes the largest value of its column; a
    column without any value (screen without log-probabilities) does not enter the maximum."""
    df = df.copy()
    pct = []
    for c in margins:
        col = df[c].astype(float)
        if col.notna().any():
            df[c] = col.fillna(col.max())
            df[c + "_pct"] = df[c].rank(pct=True)
            pct.append(c + "_pct")
    df["attack_margin"] = df[[c for c in margins if c + "_pct" in df]].max(axis=1) if pct else np.nan
    df["attack_pct"] = df[pct].max(axis=1) if pct else 0.0
    return df


def eligible(pool, groups, templates=None, margins=MARGINS, max_pct=None):
    """pool: item table; groups: video table (video_id, group, cluster_cal, duration_s). -> (eligible items sorted
    in item order, counts of the exclusions)."""
    rem = pool[pool["removed_by"] == "none"]
    n_pending = int((rem["pending_steps"].fillna("") != "").sum())
    rem = rem[rem["pending_steps"].fillna("") == ""].copy()
    rem = rem.merge(groups[["video_id", "group", "cluster_cal", "duration_s"]], on="video_id", how="left")
    rem["scene_type"] = rem["scene_type"].fillna("").replace("", UNLABELLED)
    n_nogroup = int(rem["group"].isna().sum())
    n_unlab = int(((rem["scene_type"] == UNLABELLED) & rem["group"].notna()).sum())
    el = rem[rem["group"].notna() & (rem["scene_type"] != UNLABELLED)].copy()
    unscorable = el["question"].astype(str).str.contains(S.ORDERING_RE, case=False, regex=True, na=False)
    el = el[~unscorable].copy()
    el["template"] = el["item_id"].map(templates) if templates is not None else el["question"].map(SC.template_regex)
    el["tgroup"] = el.groupby(["benchmark", "template"])["item_id"].transform("size")
    el = attack_percentile(el, margins)
    if max_pct is not None:
        el = el[el["attack_pct"] <= max_pct].copy()
    el["task_class"] = el["task_class"].fillna("").replace("", "_none")
    el = el.sort_values(["tgroup", "attack_pct", "item_id"]).reset_index(drop=True)
    return el, dict(n_pending=n_pending, n_no_group=n_nogroup, n_unlabelled=n_unlab,
                    n_unscorable=int(unscorable.sum()))


def select(el, budget=10000, k=1, j=1, j_max=4, m=5, k_scarce=3, share=0.05, verbose=True):
    """The round-robin selection on the eligible items (sorted as returned by `eligible`).
    -> (list of (row index, order of selection), state for the report)."""
    rows = list(el.itertuples())                         # rows[i].Index == i
    queues = defaultdict(lambda: defaultdict(deque))     # (group, scene) -> benchmark -> row indices
    for i, r in enumerate(rows):
        queues[(r.group, r.scene_type)][r.benchmark].append(i)
    scenes_in = {g: deque(sorted({s for (gg, s) in queues if gg == g})) for g in GROUPS}
    bench_rot = {key: deque(sorted(bd)) for key, bd in queues.items()}

    per_video, per_tmpl, per_bench = defaultdict(int), defaultdict(int), defaultdict(int)
    bench_cap = int(np.floor(budget * share))
    videos_in_cluster = defaultdict(set)
    chosen, order_no = [], 0
    j_of = {g: j for g in GROUPS}
    m_of = {g: m for g in GROUPS}
    n_group = defaultdict(int)
    target = budget / len(GROUPS)
    cell_size = el.groupby(["group", "scene_type"]).size()
    scarce = set()
    for g in GROUPS:
        n_sc = sum(1 for (gg, _) in cell_size.index if gg == g)
        cell_target = target / max(n_sc, 1)
        scarce |= {(gg, sc) for (gg, sc), n in cell_size.items() if gg == g and n < cell_target}
    bench_exempt = set()
    n_exempt = defaultdict(int)
    blocked = defaultdict(deque)                         # held back by the cluster or template cap
    stage = defaultdict(int)

    def admissible(r, coverage=False):
        sc = (r.group, r.scene_type) in scarce
        if per_video[r.video_id] >= (k_scarce if sc else k):
            return False
        if not (coverage or sc) and r.video_id not in videos_in_cluster[r.cluster_cal] \
                and len(videos_in_cluster[r.cluster_cal]) >= j_of[r.group]:
            return False
        if not (coverage or sc) and per_tmpl[(r.benchmark, r.template)] >= m_of[r.group]:
            return False
        if per_bench[r.benchmark] >= bench_cap and r.group not in bench_exempt:
            return False
        return True

    def take(r):
        nonlocal order_no
        per_video[r.video_id] += 1
        videos_in_cluster[r.cluster_cal].add(r.video_id)
        per_tmpl[(r.benchmark, r.template)] += 1
        per_bench[r.benchmark] += 1
        if per_bench[r.benchmark] > bench_cap:
            n_exempt[r.group] += 1
        n_group[r.group] += 1
        order_no += 1
        chosen.append((r.Index, order_no))

    def pick_from(key):
        rot = bench_rot[key]
        for _ in range(len(rot)):
            b = rot[0]
            rot.rotate(-1)
            q = queues[key][b]
            if per_bench[b] >= bench_cap and key[0] not in bench_exempt:
                continue                                  # benchmark cap: keep the queue, try the next benchmark
            while q:
                row = rows[q[0]]
                if admissible(row):
                    q.popleft()
                    take(row)
                    return True
                if per_video[row.video_id] >= (k_scarce if key in scarce else k):
                    q.popleft()                           # video cap: permanent
                    continue
                blocked[key].append(q.popleft())          # cluster or template cap: may return after a relaxation
            rot.remove(b)
            if not rot:
                break
        return False

    # coverage pass
    cov_taken = 0
    by_pair = defaultdict(list)
    for i, r in enumerate(rows):
        by_pair[(r.benchmark, r.task_class)].append(i)
    for pair in sorted(by_pair, key=lambda pr: (len(by_pair[pr]), pr)):
        for i in sorted(by_pair[pair], key=lambda i: (n_group[rows[i].group], rows[i].attack_pct, i)):
            row = rows[i]
            if admissible(row, coverage=True):
                take(row)
                queues[(row.group, row.scene_type)][row.benchmark].remove(i)
                cov_taken += 1
                break

    def relax(g):
        if j_of[g] < j_max:
            j_of[g] += 1
        elif m_of[g] == m:
            m_of[g] = 2 * m
        elif g not in bench_exempt:
            bench_exempt.add(g)
        else:
            return False
        stage[g] += 1
        for key in [kk for kk in blocked if kk[0] == g]:
            for i in blocked[key]:
                queues[key][rows[i].benchmark].append(i)
                if rows[i].benchmark not in bench_rot[key]:
                    bench_rot[key].append(rows[i].benchmark)
                if key[1] not in scenes_in[g]:
                    scenes_in[g].append(key[1])
            blocked[key].clear()
        return True

    active = deque(GROUPS)
    while len(chosen) < budget and active:
        g = active[0]
        active.rotate(-1)
        got = False
        sc = scenes_in[g]
        tries = len(sc)
        while sc and tries > 0 and not got:
            s = sc[0]
            sc.rotate(-1)
            tries -= 1
            if pick_from((g, s)):
                got = True
            elif not bench_rot[(g, s)]:
                sc.remove(s)
        if not got and not (n_group[g] < target and relax(g)):
            active.remove(g)
    state = dict(coverage_items=cov_taken, n_pairs=len(by_pair), bench_cap=bench_cap, scarce=sorted(scarce),
                 cluster_cap=dict(j_of), template_cap=dict(m_of), bench_cap_lifted=sorted(bench_exempt),
                 items_above_bench_cap=dict(n_exempt), relaxation_steps=dict(stage), per_group=dict(n_group))
    if verbose:
        log(f"pool_select: {len(chosen)} items; coverage pass {cov_taken} of {len(by_pair)} pairs; "
            f"scarce cells {len(scarce)}; relaxations {dict(stage)}", "pool")
    return chosen, state


# ------------------------------------------------------------------ stage
def build_scene_groups(ctx, pool):
    """Video table of the remaining items with duration group and scene cluster -> DataFrame (and the csv files)."""
    import pandas as pd
    from .embed import video_embeddings
    r = pool[pool["removed_by"] == "none"]
    v = r.drop_duplicates("video_id")[["video_id", "benchmark", "scene_type", "duration"]].copy().reset_index(drop=True)
    v["n_items"] = v["video_id"].map(r.groupby("video_id").size()).astype(int)
    v["duration_s"] = [SC.video_duration(ctx, h, d) for h, d in zip(v["video_id"], v["duration"])]
    index, V = video_embeddings(ctx, {h: video_file(ctx, h) for h in v["video_id"]})
    cov = v[v["video_id"].isin(index) & v["duration_s"].notna()].reset_index(drop=True)
    cov["group"] = [SC.duration_group(d) for d in cov["duration_s"]]
    cfg = ctx.get("pool") or {}
    if len(cov):
        cal, fix, rows = SC.scene_clusters(cov["group"].values, V[[index[h] for h in cov["video_id"]]],
                                           cfg.get("scene_t0", SC.T0), cfg.get("scene_pairs", SC.N_PAIRS))
    else:
        cal, fix, rows = [], [], []
    cov["cluster_cal"], cov["cluster_fixed"] = cal, fix
    cov.drop(columns=["duration"]).to_csv(ctx.path("pool", name="scene_groups_videos.csv"), index=False)
    pd.DataFrame(rows).to_csv(ctx.path("pool", name="scene_groups_table.csv"), index=False)
    log(f"pool_select: {len(v)} videos remaining, {len(cov)} with a vector and a duration, "
        f"{sum(x['clusters_cal'] for x in rows)} scene clusters", "pool")
    return cov


def with_labels(ctx, pool):
    """scene_type and capability from the labelling stage (pool/scene_labels.csv, pool/item_capability.csv)."""
    pool = pool.copy()
    for name, key, cols in (("scene_labels.csv", "video_id", {"scene_type": "scene_type"}),
                            ("item_capability.csv", "item_id", {"group": "capability", "fine": "fine_category"})):
        p = ctx.path("pool", name=name)
        rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
        for src, dst in cols.items():
            lab = {r[key]: r[src] for r in rows if r.get(src)}
            hit = pool[key].map(lab)
            if dst not in pool:
                pool[dst] = None
            pool.loc[hit.notna(), dst] = hit[hit.notna()]
    return pool


def export(ctx, pool, el, sel):
    """The selected items and their provenance tables."""
    out = sel.copy()
    out["license"] = out["license"].fillna("").replace("", "unknown")
    out["answer"] = [S.LETTERS[int(a)] for a in out["answer_idx"]]
    out = out.rename(columns={"group": "duration_group", "cluster_cal": "scene_cluster"})
    cols = ["selection_order", "item_id", "benchmark", "video_id", "duration_s", "duration_group", "scene_type",
            "scene_cluster", "task_class", "capability", "fine_category", "question", "options", "answer",
            "answer_idx", "n_options", "license", "template", "tgroup", "attack_pct", "attack_margin"] + MARGINS
    out = out[[c for c in cols if c in out]]
    with open(ctx.path("pool", name="video_index_items.jsonl"), "w") as f:
        for r in out.to_dict(orient="records"):
            r = {k: (None if isinstance(v, float) and np.isnan(v) else v.item() if isinstance(v, np.generic) else v)
                 for k, v in r.items()}
            r["options"] = list(r["options"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    src = (out.groupby("benchmark").agg(selected=("item_id", "size"), videos_selected=("video_id", "nunique"))
           .join(el.groupby("benchmark").size().rename("eligible"))
           .join(pool.groupby("benchmark").size().rename("pool_total")))
    for g in GROUPS:
        src[g] = out[out["duration_group"] == g].groupby("benchmark").size()
    src = src.fillna(0).astype(int)
    src["share_pct"] = (src["selected"] / max(len(out), 1) * 100).round(2)
    src.sort_values("selected", ascending=False).to_csv(ctx.path("pool", name="video_index_sources.csv"))
    comp = out.pivot_table(index="scene_type", columns="duration_group", values="item_id", aggfunc="size",
                           fill_value=0).reindex(columns=GROUPS).fillna(0).astype(int)
    comp["total"] = comp.sum(axis=1)
    comp.sort_values("total", ascending=False).to_csv(ctx.path("pool", name="video_index_composition.csv"))
    summ = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), items=int(len(out)),
                videos=int(out["video_id"].nunique()), benchmarks=int(out["benchmark"].nunique()),
                scene_clusters=int(out["scene_cluster"].nunique()), scene_types=int(out["scene_type"].nunique()),
                task_class_pairs=int(out.groupby(["benchmark", "task_class"]).ngroups),
                eligible_items=int(len(el)), eligible_videos=int(el["video_id"].nunique()),
                per_group={g: int((out["duration_group"] == g).sum()) for g in GROUPS},
                per_scene=out["scene_type"].value_counts().to_dict(),
                attack_pct_median=float(out["attack_pct"].median()) if len(out) else None)
    json.dump(summ, open(ctx.path("pool", name="video_index_summary.json"), "w"), indent=1, ensure_ascii=False)
    return summ


def run(ctx, benchmarks=None, limit=None):
    cfg = {**DEFAULTS, **{k: v for k, v in (ctx.get("pool") or {}).items() if k in DEFAULTS}}
    if limit:
        cfg["budget"] = min(cfg["budget"], limit)
    pool = with_labels(ctx, S.read_table(ctx, "pool_items"))
    if benchmarks:
        pool = pool[pool["benchmark"].isin(set(benchmarks))]
    groups = build_scene_groups(ctx, pool)
    pre = pool[(pool["removed_by"] == "none") & pool["video_id"].isin(set(groups["video_id"]))]
    qtext = pre["question_raw"].where(pre["question_raw"].fillna("") != "", pre["question"])
    templates = SC.templates_for(ctx, list(zip(pre["item_id"], qtext)))
    el, excluded = eligible(pool, groups, templates, max_pct=cfg.pop("max_pct"))
    chosen, state = select(el, **cfg)
    sel = el.loc[[i for i, _ in chosen]].copy()
    sel["selection_order"] = [o for _, o in chosen]
    sel = sel.sort_values("selection_order")
    sel_cols = ["selection_order", "benchmark", "item_id", "video_id", "group", "duration_s", "scene_type",
                "cluster_cal", "task_class", "template", "tgroup", "attack_pct", "attack_margin"] + MARGINS
    sel[[c for c in sel_cols if c in sel]].to_csv(ctx.path("pool", name="video_index_selection.csv"), index=False)
    summ = export(ctx, pool, el, sel)
    summ.update(excluded=excluded, parameters=cfg, selection=state)
    json.dump(summ, open(ctx.path("pool", name="video_index_summary.json"), "w"), indent=1, ensure_ascii=False,
              default=str)
    log(f"pool_select: {summ['items']} of {summ['eligible_items']} eligible items selected from "
        f"{summ['benchmarks']} benchmarks; excluded before selection: {excluded}", "pool")
    return summ
