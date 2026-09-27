"""Stage attribution_checks: two checks of the attribution protocol.

Unstable failures
    A seeded sample of the still-wrong items (10 %, at least one per benchmark; ``attribution.check_n`` sets a fixed
    number) is asked again with the reasoning prompt on a frame grid moved by half an inter-frame step. Only the
    frames change. The share that turns correct is the unstable-failure rate; these items are excluded from the
    share tables. Videos whose moved grid equals the uniform grid are recorded (identical_frames) and excluded from
    the rate.
    The moved grid is the window [0.016, 1.016] x (n - 1) of the frame indices, clamped to the video.
    Output attribution/checks/<benchmark>.jsonl and tables/attribution_unstable.csv.

Judge self-attribution
    Chi-square test between the category distribution a judge assigns to failures of its own model and to failures
    of other models. With a single erring model the status is not_applicable.
    Output tables/attribution_self_bias.json.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter

from ...data.schema import JsonlWriter, read_jsonl
from ...runner import log, pmap
from ...scoring import score
from ..items import item_options, load_samples, question_format, video_of
from . import common as C

FRAC = 0.10
UNSTABLE_COLS = ["benchmark", "model", "n_failures", "n_requested", "n_completed", "n_scored", "n_flipped_correct",
                 "n_refused", "unstable_failure_rate"]


def pick_sample(rows, frac=FRAC, n=None, seed=C.SEED):
    """Seeded sample of the still-wrong rows (sorted by model and qid before sampling)."""
    rows = sorted(rows, key=lambda t: (t.get("model", ""), str(t["qid"])))
    if not rows:
        return []
    n = n if n else max(1, round(frac * len(rows)))
    return random.Random(seed).sample(rows, min(n, len(rows)))


def rerun_item(ctx, bench, tr, srow, fmt):
    qid, label = str(tr["qid"]), tr.get("model", "?")
    video, meta = video_of(ctx, srow)
    if not video:
        return None
    k = int(ctx.get("reference_frames", 32))
    n_fr = C.n_frames_of(video, meta)
    offset = dict(mode="window", start=C.OFFSET_START, end=C.OFFSET_END, k=k)
    idx = C.offset_indices(n_fr, k)
    if n_fr and idx == C.uniform_indices(n_fr, k):
        return dict(qid=qid, model=label, pred_offset=None, flipped_correct=None, identical_frames=True,
                    offset=offset, usage=None)
    try:
        images = C.decode(ctx, video, idx)
    except Exception as e:  # noqa: BLE001
        log(f"checks/{bench}/{qid}: frames: {str(e)[:120]}")
        return None
    prompt = C.COT_PROMPT.format(n=len(images), q=srow["question"], opts=C.options_block(ctx, srow))
    text, usage, err = C.call(ctx.models["reference"], prompt, images, C.budget(ctx, "trace"))
    if err == "refusal":
        return dict(qid=qid, model=label, pred_offset=None, flipped_correct=None, refused=True, offset=offset,
                    usage=usage)
    if text is None:
        log(f"checks/{bench}/{qid}: {err}")
        return None
    pred = C.extract_final_answer(text)
    ok = score(pred, srow.get("answer", ""), item_options(srow)[0], fmt)
    return dict(qid=qid, model=label, pred_offset=pred, flipped_correct=None if ok is None else bool(ok),
                offset=offset, usage=usage)


def aggregate(bench, rows, picked, detail):
    """Rate rows of one benchmark, restricted to the current sample."""
    det = {(d.get("model", "?"), str(d["qid"])): d for d in detail}
    keys = {(t.get("model", "?"), str(t["qid"])) for t in picked}
    n_fail = Counter(t.get("model", "?") for t in rows)
    n_req = Counter(t.get("model", "?") for t in picked)
    per = {}
    for k, d in det.items():
        if k not in keys:
            continue
        s = per.setdefault(k[0], dict(n=0, flip=0, excluded=0, refused=0))
        if d.get("flipped_correct") is None:
            s["excluded"] += 1
            s["refused"] += bool(d.get("refused"))
            continue
        s["n"] += 1
        s["flip"] += bool(d["flipped_correct"])
    out = []
    for m in sorted(set(per) | set(n_req)):
        s = per.get(m, dict(n=0, flip=0, excluded=0, refused=0))
        out.append(dict(benchmark=bench, model=m, n_failures=n_fail.get(m, 0), n_requested=n_req.get(m, 0),
                        n_completed=s["n"] + s["excluded"], n_scored=s["n"], n_flipped_correct=s["flip"],
                        n_refused=s["refused"], unstable_failure_rate=round(s["flip"] / s["n"], 4) if s["n"] else 0))
    return out


def _sample_args(ctx):
    a = ctx.get("attribution") or {}
    return a.get("check_frac", FRAC), a.get("check_n")


def unstable_bench(ctx, bench, limit=None):
    samples = load_samples(ctx, bench)
    fmt = question_format(ctx, bench)
    rows = [t for t in C.load_traces(ctx, bench, only_still_wrong=True) if str(t["qid"]) in samples]
    frac, n = _sample_args(ctx)
    picked = pick_sample(rows, frac, limit or n)
    with JsonlWriter(ctx.path("attribution", stage="checks", bench=bench), key=("model", "qid")) as w:
        todo = [t for t in picked if not w.has(dict(model=t.get("model", "?"), qid=str(t["qid"])))]

        def one(tr):
            row = rerun_item(ctx, bench, tr, samples[str(tr["qid"])], fmt)
            if row:
                w.write(row)
            return row

        for _ in pmap(one, todo, ctx.get("workers", 8), desc=f"checks/{bench}"):
            pass
    return aggregate(bench, rows, picked, read_jsonl(ctx.path("attribution", stage="checks", bench=bench)))


def write_unstable(ctx, limit=None):
    frac, n = _sample_args(ctx)
    rows = []
    for bench in C.traced_benchmarks(ctx, "checks"):
        samples = load_samples(ctx, bench)
        tr = [t for t in C.load_traces(ctx, bench, only_still_wrong=True) if str(t["qid"]) in samples]
        rows += aggregate(bench, tr, pick_sample(tr, frac, limit or n),
                          read_jsonl(ctx.path("attribution", stage="checks", bench=bench)))
    with open(ctx.path("table", name="attribution_unstable.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=UNSTABLE_COLS)
        w.writeheader()
        w.writerows(rows)
    return rows


def unstable_keys(ctx):
    """{(benchmark, model, qid)} of the items that turned correct on the moved grid."""
    keys = set()
    for bench in C.traced_benchmarks(ctx, "checks"):
        for r in read_jsonl(ctx.path("attribution", stage="checks", bench=bench)):
            if r.get("flipped_correct") is True:
                keys.add((bench, r.get("model"), str(r["qid"])))
    return keys


def self_attribution(ctx, role="judge_a"):
    dist, judge = {}, None
    for bench in C.traced_benchmarks(ctx, "judge"):
        for r in read_jsonl(ctx.path("attribution", stage="judge", bench=bench)):
            v = (r.get("by") or {}).get(role)
            if not v or not v.get("category"):
                continue
            judge = v.get("model")
            dist.setdefault(r.get("model", "?"), Counter())[v["category"]] += 1
    own = [m for m in dist if m == judge]
    other = [m for m in dist if m != judge]
    out = dict(role=role, judge=judge, models_seen={m: sum(c.values()) for m, c in dist.items()},
               distributions={m: dict(c) for m, c in dist.items()})
    if not own or not other:
        out["status"] = "not_applicable"
        out["note"] = "needs attributed failures of the judge's own model and of another model"
    else:
        cs, co = dist[own[0]], sum((dist[m] for m in other), Counter())
        cats = sorted(set(cs) | set(co))
        table = [[cs.get(c, 0) for c in cats], [co.get(c, 0) for c in cats]]
        try:
            from scipy.stats import chi2_contingency
            stat, p, dof, _ = chi2_contingency(table)
            out.update(status="computed", chi2=float(stat), p_value=float(p), dof=int(dof), categories=cats,
                       table=table)
        except Exception as e:  # noqa: BLE001
            out.update(status="not_computed", categories=cats, table=table, error=str(e)[:200])
    with open(ctx.path("table", name="attribution_self_bias.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def run(ctx, benchmarks=None, limit=None):
    out = {}
    for bench in (b for b in C.traced_benchmarks(ctx) if not benchmarks or b in benchmarks):
        out[bench] = unstable_bench(ctx, bench, limit)
        for r in out[bench]:
            log(f"checks/{bench}: requested {r['n_requested']}, scored {r['n_scored']}, "
                f"turned correct {r['n_flipped_correct']}, refused {r['n_refused']}")
    write_unstable(ctx, limit)
    out["self_attribution"] = self_attribution(ctx)["status"]
    return out
