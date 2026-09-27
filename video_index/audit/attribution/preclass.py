"""Stage attribution_preclass: rule-based pre-classification of the still-wrong items. No model call.

The text-only run (condition ``blind``) of the erring model is compared with the 32-frame error:

    rule 1  text-only answer wrong AND the same wrong answer as with frames -> language_prior_dominated;
            the item is recorded and not sent to the judges
    rule 2  text-only answer correct, 32-frame answer wrong -> vision_introduced_error = true; still judged

The 32-frame side is the answer of the reasoning rerun; when that answer names no option, the original reference
answer is used. Both rules need the text-only run of the erring model itself; a text-only run of another model is
shown to the judges as context but triggers neither rule.

Output attribution/preclass/<benchmark>.jsonl (rewritten in full by every run):
    qid, model, preclass, to_judge, vision_introduced_error, blind_correct, blind_pred, mf_pred_used, letter_source
and tables/attribution_preclass.csv.
"""
from __future__ import annotations

import csv

from ...data.schema import read_jsonl, write_jsonl
from ...runner import log
from ...scoring import mcq_letter, score
from ..items import item_options, load_samples, question_format
from . import common as C

SUMMARY_COLS = ["benchmark", "n_still_wrong", "n_language_prior_dominated", "language_prior_rate",
                "n_vision_introduced", "vision_introduced_rate", "n_to_judge_unflagged", "n_blind_missing"]


def preclassify(trace, srow, fmt, blind):
    """One still-wrong trace row -> pre-classification record, or None when the rerun answer is not scorable by rule.
    blind: the text-only result row of the erring model, or None."""
    opts = item_options(srow)[0]
    if score(trace.get("pred_cot", ""), srow.get("answer", ""), opts, fmt) is None:
        return None
    pred_used, letter_source = trace.get("pred_cot", ""), "cot"
    if opts and not mcq_letter(pred_used, opts):
        if mcq_letter(trace.get("orig_pred", ""), opts):
            pred_used, letter_source = trace.get("orig_pred", ""), "orig"
        else:
            letter_source = "text"
    rec = dict(qid=trace["qid"], model=trace.get("model", "?"), preclass="none", to_judge=True,
               vision_introduced_error=False, blind_correct=None, blind_pred=None,
               mf_pred_used=str(pred_used)[:200], letter_source=letter_source)
    if blind is None:
        return rec
    bc = score(blind.get("pred"), blind.get("answer") or srow.get("answer"), opts, fmt)
    rec["blind_correct"] = None if bc is None else bool(bc)
    rec["blind_pred"] = str(blind.get("pred", ""))[:200]
    if bc:
        rec["vision_introduced_error"] = True
    elif bc is False and C.same_wrong_answer(blind.get("pred", ""), pred_used, opts):
        rec["preclass"] = "language_prior_dominated"
        rec["to_judge"] = False
    return rec


def run_bench(ctx, bench):
    samples = load_samples(ctx, bench)
    fmt = question_format(ctx, bench)
    out, st = [], dict(n=0, language_prior=0, vision_introduced=0, blind_missing=0, missing_sample=0, scoring_na=0)
    for tr in C.load_traces(ctx, bench, only_still_wrong=True):
        srow = samples.get(str(tr["qid"]))
        if not srow:
            st["missing_sample"] += 1
            continue
        blind, same = C.blind_row(ctx, bench, str(tr["qid"]), tr.get("model", "?"))
        rec = preclassify(tr, srow, fmt, blind if same else None)
        if rec is None:
            st["scoring_na"] += 1
            continue
        st["n"] += 1
        st["blind_missing"] += rec["blind_pred"] is None
        st["language_prior"] += rec["preclass"] == "language_prior_dominated"
        st["vision_introduced"] += rec["vision_introduced_error"]
        out.append(rec)
    write_jsonl(ctx.path("attribution", stage="preclass", bench=bench), out)
    return st


def write_summary(ctx):
    rows = []
    for bench in C.traced_benchmarks(ctx, "preclass"):
        rs = read_jsonl(ctx.path("attribution", stage="preclass", bench=bench))
        n = len(rs)
        lang = sum(r.get("preclass") == "language_prior_dominated" for r in rs)
        vis = sum(bool(r.get("vision_introduced_error")) for r in rs)
        rows.append(dict(benchmark=bench, n_still_wrong=n, n_language_prior_dominated=lang,
                         language_prior_rate=round(lang / (n or 1), 4), n_vision_introduced=vis,
                         vision_introduced_rate=round(vis / (n or 1), 4), n_to_judge_unflagged=n - lang - vis,
                         n_blind_missing=sum(r.get("blind_pred") is None for r in rs)))
    with open(ctx.path("table", name="attribution_preclass.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(rows)
    return rows


def run(ctx, benchmarks=None, limit=None):
    traced = C.traced_benchmarks(ctx)
    out = {}
    for bench in (b for b in traced if not benchmarks or b in benchmarks):
        out[bench] = run_bench(ctx, bench)
        log(f"preclass/{bench}: {out[bench]}")
    write_summary(ctx)
    return out
