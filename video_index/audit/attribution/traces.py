"""Stage attribution_traces: reasoning traces of the reference errors.

Every item the reference model answers wrongly from 32 frames is asked once more with the same 32 uniform frames
and a prompt that requires written reasoning before the final answer.

    rerun correct      recovered; not attributed, counted in tables/attribution_recovered.csv
    rerun still wrong  the written reasoning is the trace read by the judges
    refusal            counted (n_refused); the item is neither recovered nor attributed

Output attribution/traces/<benchmark>.jsonl, one row per (model, qid):
    qid, model, orig_pred, pred_cot, still_wrong (true | false | null = not scorable by rule or refused),
    refused, trace, frame_check, n_frames, usage
Requests that fail are listed in attribution/traces_failures.jsonl and repeated by the next run.
"""
from __future__ import annotations

import csv
import json
import os
import threading

from ...data.schema import JsonlWriter
from ...runner import log, pmap
from ...scoring import score
from ..items import item_options, load_samples, question_format, select, video_of
from . import common as C

RECOVERED_COLS = ["benchmark", "model", "n_rerun", "n_recovered_by_cot", "n_scoring_na", "n_refused", "recovery_rate"]


def trace_item(ctx, model, label, bench, qid, srow, fmt, orig_pred):
    """(row, status); status is ok | refusal | no_video | frame_err | api_err."""
    video, meta = video_of(ctx, srow)
    if not video:
        return None, "no_video"
    k = int(ctx.get("reference_frames", 32))
    n = C.n_frames_of(video, meta)
    try:
        images = C.uniform_frames(ctx, video, meta, k)
    except Exception as e:  # noqa: BLE001
        log(f"traces/{bench}/{qid}: frames: {str(e)[:150]}")
        return None, "frame_err"
    if not images:
        return None, "frame_err"
    base = dict(qid=qid, model=label, orig_pred=str(orig_pred)[:300],
                frame_check=C.frame_check(srow["video_id"], n, k), n_frames=n)
    prompt = C.COT_PROMPT.format(n=len(images), q=srow["question"], opts=C.options_block(ctx, srow))
    text, usage, err = C.call(model, prompt, images, C.budget(ctx, "trace"))
    if err == "refusal":
        return dict(base, pred_cot=None, still_wrong=None, refused=True, trace="", usage=usage), "refusal"
    if text is None:
        log(f"traces/{bench}/{qid}: {err}")
        return None, "api_err"
    opts = item_options(srow)[0]
    pred_cot = C.extract_final_answer(text)
    ok = score(pred_cot, srow.get("answer", ""), opts, fmt)
    if ok is False and C.gold_contained(pred_cot, srow.get("answer", ""), opts):
        ok = True
    return dict(base, pred_cot=pred_cot, still_wrong=None if ok is None else not ok, refused=False, trace=text,
                usage=usage), "ok"


def run_bench(ctx, bench, limit=None):
    model, label = ctx.models["reference"], C.erring_label(ctx)
    samples = load_samples(ctx, bench)
    fmt = question_format(ctx, bench)
    wrong, n_total = C.wrong_qids(ctx, bench, samples, fmt, label)
    if wrong is None:
        log(f"traces/{bench}: no reference results of {label}; skipped")
        return dict(skipped=1)
    st = dict(wrong=len(wrong), reference_rows=n_total, new=0, refused=0)
    lock = threading.Lock()
    fail_path = os.path.join(ctx.work_dir, "attribution", "traces_failures.jsonl")
    with JsonlWriter(ctx.path("attribution", stage="traces", bench=bench), key=("model", "qid")) as w:
        todo = [q for q in sorted(wrong) if not w.has(dict(model=label, qid=q))]
        if limit:
            todo = todo[:limit]

        def one(qid):
            row, status = trace_item(ctx, model, label, bench, qid, samples[qid], fmt, wrong[qid])
            if row is None:
                with lock:
                    C.count_failure(st, status)
                    with open(fail_path, "a") as f:
                        f.write(json.dumps(dict(benchmark=bench, model=label, qid=qid, kind=status)) + "\n")
                return None
            w.write(row)
            with lock:
                st["new"] += 1
                st["refused"] += status == "refusal"
            return row

        for _ in pmap(one, todo, ctx.get("workers", 8), desc=f"traces/{bench}"):
            pass
    return st


def write_recovered(ctx):
    """tables/attribution_recovered.csv, recomputed from every trace file."""
    rows = []
    for bench in C.traced_benchmarks(ctx):
        per = {}
        for r in C.load_traces(ctx, bench):
            d = per.setdefault(r.get("model", "?"), dict(rerun=0, rec=0, na=0, refused=0))
            d["rerun"] += 1
            if r.get("refused"):
                d["refused"] += 1
            elif r.get("still_wrong") is None:
                d["na"] += 1
            elif r["still_wrong"] is False:
                d["rec"] += 1
        for m, d in sorted(per.items()):
            den = d["rerun"] - d["na"] - d["refused"]
            rows.append(dict(benchmark=bench, model=m, n_rerun=d["rerun"], n_recovered_by_cot=d["rec"],
                             n_scoring_na=d["na"], n_refused=d["refused"],
                             recovery_rate=round(d["rec"] / den, 4) if den else 0))
    with open(ctx.path("table", name="attribution_recovered.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RECOVERED_COLS)
        w.writeheader()
        w.writerows(rows)
    return rows


def run(ctx, benchmarks=None, limit=None):
    out = {}
    for bench in select(ctx, benchmarks):
        out[bench] = run_bench(ctx, bench, limit)
        log(f"traces/{bench}: {out[bench]}")
    rows = write_recovered(ctx)
    n_ref = sum(r["n_refused"] for r in rows)
    log(f"traces: {sum(r['n_rerun'] for r in rows)} reruns, {sum(r['n_recovered_by_cot'] for r in rows)} recovered, "
        f"{n_ref} refusals")
    return out
