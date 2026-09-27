"""Stage pool_screen: the sequential funnel of the pool, in pyramid order.

    1 duplicate        same video and text cosine > 0.90 inside a benchmark (dedup.py)
    2 option_format    meta options, repeated option texts, empty correct option
    3 opt_only         ``attacker``, options only, four option orders
    4 blind            ``attacker``, question and options, four option orders
    5 single_frame     ``attacker``, frame 16 of the 32-frame grid
    6 v32              ``attacker_small``, 32 uniform frames
    7 v32_shuffle      ``attacker_small``, the same 32 frames in a seeded random order
    8 xbench_duplicate same video and text cosine >= 0.90 across benchmarks

Every step runs on the items the earlier steps left, so each removed item has exactly one reason. Frames: the
attacker grid of the audit (grid.py), long side 448. Option orders: ``random.Random(f"42|{item_id}")``. Removal rules (margin = log p(correct) minus the mean log p
of the other options, from the log-probabilities of the first generated token):

    opt_only, blind     mean margin over the four orders > log 2 AND at least three of the four orders correct
    visual steps        correct AND margin > log 2

Without log-probabilities (provider other than OpenAI-compatible, or ``pool.logprobs: false``) the margin is not
available; the rules are then "at least three of four orders correct" and "correct".

Outputs (pool/):
    steps/<step>/<benchmark>.jsonl   one row per item, append-only; failed requests in steps/<step>/errors.jsonl
    dedup.jsonl, option_format.jsonl, xbench_pairs.jsonl
    pool_items                       the item table with the screening columns, removed_by and pending_steps
    funnel.csv                       stage, n_entering, n_removed, n_after
    pool_summary.csv                 the same counts per benchmark
"""
from __future__ import annotations

import csv
import json
import os
import random
import threading
from collections import Counter

from ..audit.items import bench_meta
from ..data.schema import JsonlWriter, read_jsonl, write_jsonl
from ..runner import log, pmap
from . import dedup as D
from . import logprobs as L
from . import schema as S
from .embed import text_embeddings
from .grid import FrameCache, video_file

INTRO_VISUAL = ("You are given {n} frame(s) sampled from a video. "
                "Answer the question based on these frames.")
INTRO_BLIND = ("You are given NO frames from the video. "
               "Answer the question from the text alone.")
INTRO_OPT_ONLY = ("You are given NO frames from the video and NO question text. "
                  "Using only the answer options below, choose the option most likely to be correct.")
PROMPT = "{intro}\n\nQuestion: {q}\nOptions:\n{opts}\nAnswer with the option letter only."
PROMPT_OPT_ONLY = "{intro}\n\nOptions:\n{opts}\nAnswer with the option letter only."

# step, role, column prefix, needs the video
STEPS = [("opt_only", "attacker", "opt", False), ("blind", "attacker", "blind", False),
         ("single_frame", "attacker", "sf", True), ("v32", "attacker_small", "v32", True),
         ("v32_shuffle", "attacker_small", "v32_shuffle", True)]
STEP_NAMES = [s[0] for s in STEPS]
MARGINS = ["opt_margin", "blind_margin", "sf_margin", "v32_margin", "v32_shuffle_margin"]
FORMAT_REASONS = ["open", "format_yesno", "duplicate", "option_format"]
REMOVAL_KEYS = FORMAT_REASONS + STEP_NAMES + ["xbench_duplicate"]
FUNNEL_LABELS = {"duplicate": "duplicates within a benchmark", "option_format": "option format",
                 "opt_only": "options only", "blind": "question and options, no frames",
                 "single_frame": "single frame", "v32": "32 frames, small model",
                 "v32_shuffle": "32 shuffled frames, small model",
                 "xbench_duplicate": "duplicates across benchmarks"}
SUMMARY_COLS = ["benchmark", "n_total", "n_open", "n_yesno", "n_mcq", "n_duplicate", "n_option_format", "n_opt_only",
                "n_blind", "n_single_frame", "n_v32", "n_v32_shuffle", "n_xbench_duplicate", "n_remaining",
                "n_pending", "n_videos"]


def _pool_cfg(ctx, key, default=None):
    return (ctx.get("pool") or {}).get(key, default)


# ------------------------------------------------------------------ removal rules
def text_step_result(perms, scores, answer_idx, prefix):
    """Options-only and blind: one score record per option order -> the step columns of the item."""
    recs, hits, margins = [], 0, []
    for perm, sc in zip(perms, scores):
        cpos = perm.index(answer_idx)
        hit = int(sc["argmax"] == cpos)
        hits += hit
        rec = dict(order=perm, correct_pos=cpos, argmax=sc["argmax"], hit=hit, missing=sc["missing"],
                   top1_token=sc["top1_token"])
        if sc["lp"] is not None:
            m = L.margin_of(sc["lp"], cpos)
            margins.append(m)
            rec.update(lp=[round(x, 4) for x in sc["lp"]], margin=round(m, 4))
        recs.append(rec)
    margin = sum(margins) / len(margins) if margins else None
    enough = hits >= len(perms) - 1
    return {"perms": recs, "remove": bool(enough and (margin is None or margin > L.LOG2)),
            f"{prefix}_acc_4perm": round(hits / len(perms), 4),
            f"{prefix}_margin": None if margin is None else round(margin, 4), "n_forwards": len(perms)}


def visual_step_result(sc, answer_idx, prefix):
    correct = bool(sc["argmax"] == answer_idx)
    margin = None if sc["lp"] is None else L.margin_of(sc["lp"], answer_idx)
    return {"correct_pos": answer_idx, "lp": None if sc["lp"] is None else [round(x, 4) for x in sc["lp"]],
            "argmax": sc["argmax"], "missing": sc["missing"], "top1_token": sc["top1_token"],
            "remove": bool(correct and (margin is None or margin > L.LOG2)), f"{prefix}_correct": correct,
            f"{prefix}_margin": None if margin is None else round(margin, 4), "n_forwards": 1}


# ------------------------------------------------------------------ attack steps
def run_item(ctx, step, model, row, cache):
    name, _, prefix, visual = step
    iid, opts, aidx = row["item_id"], list(row["options"]), int(row["answer_idx"])
    mode, prefill = _pool_cfg(ctx, "logprobs", "auto"), _pool_cfg(ctx, "assistant_prefill", True)
    base = dict(item_id=iid, benchmark=row["benchmark"], step=name, model=model.name, n_options=len(opts))
    if not visual:
        perms = L.permutations_for(iid, len(opts))
        scores = []
        for perm in perms:
            texts = [opts[i] for i in perm]
            prompt = (PROMPT_OPT_ONLY.format(intro=INTRO_OPT_ONLY, opts=S.render_options(texts)) if name == "opt_only"
                      else PROMPT.format(intro=INTRO_BLIND, q=row["question"], opts=S.render_options(texts)))
            scores.append(L.letter_scores(model, prompt, (), texts, mode, prefill))
        out = text_step_result(perms, scores, aidx, prefix)
        out["scored"] = scores[0]["scored"]
    else:
        images, mid = cache.get(row["video_file"])
        if not images:
            raise RuntimeError("no frames decoded")
        if name == "single_frame":
            images = [images[mid]]
        elif name == "v32_shuffle":
            order = list(range(len(images)))
            random.Random(f"{L.SEED}|{iid}").shuffle(order)
            images = [images[i] for i in order]
            base["perm_seed"] = f"{L.SEED}|{iid}"
        prompt = PROMPT.format(intro=INTRO_VISUAL.format(n=len(images)), q=row["question"],
                               opts=S.render_options(opts))
        sc = L.letter_scores(model, prompt, images, opts, mode, prefill)
        out = visual_step_result(sc, aidx, prefix)
        out.update(video_id=row["video_id"], n_frames=len(images), scored=sc["scored"])
    return dict(base, **out, status="ok")


def step_rows(ctx, name, benchmarks=None):
    """{item_id: row} of a step over all benchmarks (the last row of an item wins)."""
    d = os.path.join(ctx.work_dir, "pool", "steps", name)
    out = {}
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".jsonl") and f != "errors.jsonl" and (not benchmarks or f[:-6] in benchmarks):
                for r in read_jsonl(os.path.join(d, f)):
                    if r.get("status", "ok") == "ok" and "item_id" in r:
                        out[str(r["item_id"])] = r
    return out


def run_step(ctx, step, df, limit=None):
    """Run one step on the rows of `df` that have no result yet. Returns the number of new rows."""
    name, role, _, visual = step
    model = ctx.models[role]
    cache, n_new, lock = FrameCache(), 0, threading.Lock()
    err_path = ctx.path("pool", name=f"steps/{name}/errors.jsonl")
    for bench, g in df.groupby("benchmark", sort=True):
        with JsonlWriter(ctx.path("pool", name=f"steps/{name}/{bench}.jsonl"), key="item_id") as w:
            rows = [r for r in g.to_dict("records") if not w.has(r)]
            if visual:
                rows = [r for r in rows if r.get("video_file")]
                rows.sort(key=lambda r: (r["video_id"], r["item_id"]))
            if limit:
                rows = rows[:limit]

            def one(row):
                try:
                    out = run_item(ctx, step, model, row, cache)
                except Exception as e:  # noqa: BLE001
                    with lock, open(err_path, "a") as f:
                        f.write(json.dumps(dict(item_id=row["item_id"], benchmark=bench, step=name,
                                                error=f"{type(e).__name__}: {str(e)[:300]}")) + "\n")
                    return None
                w.write(out)
                return out

            n_new += sum(1 for _ in pmap(one, rows, _pool_cfg(ctx, "workers", ctx.get("workers", 8)),
                                         desc=f"pool/{name}/{bench}", every=500))
    return n_new


# ------------------------------------------------------------------ duplicate and format screens
def run_dedup(ctx, df):
    """Within-benchmark duplicates of the multiple-choice items -> {item_id: (kept item_id, cosine)}."""
    removed, records = {}, []
    mcq = df[df["format"] == "mcq"]
    for bench, g in mcq.groupby("benchmark", sort=True):
        groups = D.video_groups(g["item_id"], g["video_id"])
        need = {i for ids in groups.values() for i in ids}
        if not need:
            continue
        sub = g[g["item_id"].isin(need)].sort_values("item_id")
        index, vecs = text_embeddings(ctx, bench, [(i, S.item_text(q, o)) for i, q, o in
                                                   zip(sub["item_id"], sub["question"], sub["options"])])
        rem, n_groups, unemb = D.within_benchmark(list(g["item_id"]), list(g["video_id"]), index, vecs)
        removed.update(rem)
        records += [dict(benchmark=bench, item_id=q, kept=k, cos_to_kept=None if c is None else round(c, 4))
                    for q, (k, c) in sorted(rem.items())]
        log(f"pool/dedup/{bench}: {len(rem)} duplicates in {n_groups} groups"
            + (f", {len(unemb)} items without a vector" if unemb else ""), "pool")
    write_jsonl(ctx.path("pool", name="dedup.jsonl"), records)
    return removed


def run_option_format(ctx, df):
    flags, records = {}, []
    for iid, bench, opts, aidx in zip(df["item_id"], df["benchmark"], df["options"], df["answer_idx"]):
        for reason, detail in D.option_format_reasons(opts, aidx):
            flags.setdefault(iid, reason)
            records.append(dict(item_id=iid, benchmark=bench, reason=reason, detail=detail))
    write_jsonl(ctx.path("pool", name="option_format.jsonl"), records)
    return flags


def run_cross_benchmark(ctx, alive_df):
    """{removed item_id: kept item_id} among the items still in the pool."""
    ids, benches, vids = list(alive_df["item_id"]), list(alive_df["benchmark"]), list(alive_df["video_id"])
    bench_of = dict(zip(ids, benches))
    shared = {i for g in D.video_groups(ids, vids).values() if len({bench_of[i] for i in g}) > 1 for i in g}
    if not shared:
        write_jsonl(ctx.path("pool", name="xbench_pairs.jsonl"), [])
        return {}
    sub = alive_df[alive_df["item_id"].isin(shared)]
    index, mats, offset = {}, [], 0
    import numpy as np
    for bench, g in sub.groupby("benchmark", sort=True):
        g = g.sort_values("item_id")
        ix, vecs = text_embeddings(ctx, bench, [(i, S.item_text(q, o)) for i, q, o in
                                                zip(g["item_id"], g["question"], g["options"])])
        rows = [ix[i] for i in g["item_id"]]
        index.update({i: offset + k for k, i in enumerate(g["item_id"])})
        mats.append(vecs[rows])
        offset += len(rows)
    pairs = D.cross_benchmark_pairs(list(sub["item_id"]), list(sub["benchmark"]), list(sub["video_id"]), index,
                                    np.concatenate(mats))
    write_jsonl(ctx.path("pool", name="xbench_pairs.jsonl"), [dict(a=a, b=b, cos=c) for a, b, c in pairs])
    year = {b: int(bench_meta(ctx, b).get("year") or 9999) for b in set(benches)}
    return D.cross_benchmark_drop(pairs, set(ids), bench_of, year)


# ------------------------------------------------------------------ assembly and accounting
def assemble(df, dup, flags, steps, xdrop):
    """Item table with the screening columns. dup: {item_id: (kept, cos)}; flags: {item_id: reason};
    steps: {step: {item_id: row}}; xdrop: {item_id: kept item_id}. One removal reason per item, in funnel order."""
    cols = {k: [] for k in ["removed_by", "dup_of", "pending_steps"]}
    step_cols = {}
    for name, _, prefix, visual in STEPS:
        for c in ([f"{prefix}_correct"] if visual else [f"{prefix}_acc_4perm"]) + [f"{prefix}_margin"]:
            step_cols[c] = (name, [])
    for iid, fmt in zip(df["item_id"], df["format"]):
        removed, dup_of, pending = "none", None, []
        if fmt == "open":
            removed = "open"
        elif fmt == "yesno":
            removed = "format_yesno"
        elif iid in dup:
            removed, dup_of = "duplicate", dup[iid][0]
        elif iid in flags:
            removed = "option_format"
        else:
            for name, _, _, visual in STEPS:
                r = steps.get(name, {}).get(iid)
                if r is None:
                    pending = STEP_NAMES[STEP_NAMES.index(name):]
                    break
                if r.get("remove"):
                    removed = name
                    break
            if removed == "none" and not pending and iid in xdrop:
                removed, dup_of = "xbench_duplicate", xdrop[iid]
        for c, (name, vals) in step_cols.items():
            r = steps.get(name, {}).get(iid)
            v = r.get(c) if r else None
            vals.append(bool(v) if c.endswith("_correct") and v is not None else v)
        cols["removed_by"].append(removed)
        cols["dup_of"].append(dup_of)
        cols["pending_steps"].append(",".join(pending))
    out = df.copy()
    for c, (_, vals) in step_cols.items():
        out[c] = vals
    for c, vals in cols.items():
        out[c] = vals
    out["task_class"] = out["declared_task"]
    out["scene_type"] = out["declared_scene"]
    return out


def funnel(removed_by):
    """Funnel rows from the removal reasons of all items: all items, multiple choice, then one row per screen.
    n_after of a row is n_entering of the next."""
    c = Counter(removed_by)
    n = sum(c.values())
    rows = [dict(stage="all_items", label="all items", n_entering=n, n_removed=0, n_after=n)]
    n_mcq = n - c["open"] - c["format_yesno"]
    rows.append(dict(stage="multiple_choice", label="multiple choice (open and yes/no items excluded)", n_entering=n,
                     n_removed=c["open"] + c["format_yesno"], n_after=n_mcq))
    entering = n_mcq
    for key in ["duplicate", "option_format"] + STEP_NAMES + ["xbench_duplicate"]:
        rows.append(dict(stage=key, label=FUNNEL_LABELS[key], n_entering=entering, n_removed=c[key],
                         n_after=entering - c[key]))
        entering -= c[key]
    rows.append(dict(stage="remaining", label="remaining", n_entering=entering, n_removed=0, n_after=c["none"]))
    return rows


def summary(df):
    rows = []
    for bench, g in df.groupby("benchmark", sort=True):
        c = Counter(g["removed_by"])
        rem = g[g["removed_by"] == "none"]
        rows.append(dict(benchmark=bench, n_total=len(g), n_open=c["open"], n_yesno=c["format_yesno"],
                         n_mcq=len(g) - c["open"] - c["format_yesno"], n_duplicate=c["duplicate"],
                         n_option_format=c["option_format"], **{f"n_{s}": c[s] for s in STEP_NAMES},
                         n_xbench_duplicate=c["xbench_duplicate"], n_remaining=len(rem),
                         n_pending=int((rem["pending_steps"] != "").sum()), n_videos=int(g["video_id"].nunique())))
    return rows


def _write_csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def run(ctx, benchmarks=None, limit=None):
    df = S.read_table(ctx, "pool_items_base")
    df["video_file"] = [video_file(ctx, v) for v in df["video_id"]]
    dup = run_dedup(ctx, df) if _pool_cfg(ctx, "dedup", True) else {}
    flags = run_option_format(ctx, df[(df["format"] == "mcq") & ~df["item_id"].isin(set(dup))])
    alive = df[(df["format"] == "mcq") & ~df["item_id"].isin(set(dup)) & ~df["item_id"].isin(set(flags))]
    steps = {}
    for step in STEPS:
        name, visual = step[0], step[3]
        todo = alive[alive["benchmark"].isin(set(benchmarks))] if benchmarks else alive
        n_new = run_step(ctx, step, todo, limit)
        steps[name] = step_rows(ctx, name)
        done = alive["item_id"].isin(set(steps[name]))
        removed = {i for i, r in steps[name].items() if r.get("remove")}
        log(f"pool/{name}: {n_new} new rows; {int(done.sum())} of {len(alive)} items done, "
            f"{len(removed & set(alive['item_id']))} removed", "pool")
        alive = alive[done & ~alive["item_id"].isin(removed)]      # items without a result wait for the next run
    xdrop = run_cross_benchmark(ctx, alive) if _pool_cfg(ctx, "cross_benchmark", True) else {}
    out = assemble(df, dup, flags, steps, xdrop)
    out["video_path"] = out.pop("video_file")
    path = S.write_table(ctx, "pool_items", out)
    rows = funnel(out["removed_by"])
    _write_csv(ctx.path("pool", name="funnel.csv"), ["stage", "label", "n_entering", "n_removed", "n_after"], rows)
    _write_csv(ctx.path("pool", name="pool_summary.csv"), SUMMARY_COLS, summary(out))
    rem = out[out["removed_by"] == "none"]
    log(f"pool_screen: {len(out)} items, {len(rem)} remaining ({int((rem['pending_steps'] != '').sum())} with "
        f"pending steps) -> {path}", "pool")
    return dict(n_items=len(out), n_remaining=len(rem), n_pending=int((rem["pending_steps"] != "").sum()),
                funnel=rows)
