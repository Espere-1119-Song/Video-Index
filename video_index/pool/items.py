"""Stage pool_items: one table with every item of every benchmark.

Input   items/<benchmark>.jsonl (all items of the benchmark after its adapter, not only the audited sample)
Output  pool/pool_items_base.(parquet|jsonl)   columns in video_index/pool/schema.py
        pool/pool_items_manifest.csv           per benchmark: items, multiple-choice, yes/no, open, videos, reasons

Options written inside the question text ("A. ..", "(A) ..", "Option A: ..") are split out when the item has no
option field. Licences come from the item (field ``license``) or from ``licenses`` in the run configuration
(mapping benchmark -> licence).
"""
from __future__ import annotations

import csv
import os
from collections import Counter

from ..data.schema import read_jsonl
from ..runner import log
from . import schema as S

MANIFEST_COLS = ["benchmark", "n_items", "n_mcq", "n_yesno", "n_open", "n_options_from_question", "n_videos",
                 "n_videos_available", "reasons"]


def benchmarks_with_items(ctx):
    d = os.path.join(ctx.work_dir, "items")
    return sorted(f[:-6] for f in os.listdir(d) if f.endswith(".jsonl")) if os.path.isdir(d) else []


def build_bench(ctx, bench, limit=None):
    licences = ctx.get("licenses") or {}
    rows, seen = [], set()
    for it in read_jsonl(ctx.path("items", bench=bench)):
        qid = str(it["qid"])
        if qid in seen:
            continue
        seen.add(qid)
        it.setdefault("benchmark", bench)
        vp = os.path.join(ctx.work_dir, "videos", str(it["video_id"]), "video.mp4") if it.get("video_id") else None
        rows.append(S.item_row(it, licences.get(bench, ""), vp if vp and os.path.exists(vp) else None))
        if limit and len(rows) >= limit:
            break
    fmt = Counter(r["format"] for r in rows)
    reasons = Counter(r["format_reason"] for r in rows if r["format_reason"])
    vids = {r["video_id"] for r in rows if r["video_id"]}
    return rows, dict(benchmark=bench, n_items=len(rows), n_mcq=fmt["mcq"], n_yesno=fmt["yesno"], n_open=fmt["open"],
                      n_options_from_question=sum(r["options_source"] == "inline_question" for r in rows),
                      n_videos=len(vids), n_videos_available=len({r["video_id"] for r in rows if r["video_path"]}),
                      reasons="; ".join(f"{k}={v}" for k, v in sorted(reasons.items())))


def run(ctx, benchmarks=None, limit=None):
    import pandas as pd
    rows, manifest = [], []
    for bench in benchmarks or benchmarks_with_items(ctx):
        r, m = build_bench(ctx, bench, limit)
        rows += r
        manifest.append(m)
        log(f"pool_items/{bench}: {m['n_items']} items, {m['n_mcq']} multiple-choice, {m['n_open']} open, "
            f"{m['n_yesno']} yes/no", "pool")
    if benchmarks:                                  # a run on some benchmarks keeps the rows of the others
        try:
            old = S.read_table(ctx, "pool_items_base")
            rows = old[~old.benchmark.isin(set(benchmarks))].to_dict("records") + rows
        except FileNotFoundError:
            pass
    df = pd.DataFrame(rows, columns=S.BASE_COLS)
    path = S.write_table(ctx, "pool_items_base", df)
    with open(ctx.path("pool", name="pool_items_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(manifest)
    log(f"pool_items: {len(df)} items ({int((df.format == 'mcq').sum())} multiple-choice) -> {path}", "pool")
    return dict(n_items=len(df), n_mcq=int((df.format == "mcq").sum()), path=path)
