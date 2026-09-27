"""One runner for every model condition of the pyramid.

A condition is (name, role, prompt builder, frame rule). Rows go to results/<condition>/<bench>__<model>.jsonl, one row
per item and role: qid, role, pred, answer, correct, frames, fallback, error (+ condition fields). A rerun skips the
rows on disk that hold a prediction; rows with an error are tried again.
"""
from __future__ import annotations

from ..data.frames import subsample
from ..data.schema import JsonlWriter, read_jsonl
from ..runner import ask, log, pmap
from ..scoring import score
from . import frame_rules, prompts
from .items import bench_meta, fmt_opts, item_options, lettered, load_samples, select, video_of

PRED_CHARS = 300
FAMILY = {"reference": "reference", "reference_long": "reference"}          # every other role uses the attacker rules


def family(role):
    return FAMILY.get(role, "attacker")


def ask_frames(model, make_prompt, frames, timestamps=None, min_frames=8, **kw):
    """runner.ask for prompts that state the number of frames: when the request does not fit, the frames are halved
    (uniformly) and the prompt is written again for the frames that are sent."""
    frames = list(frames or [])
    ts = list(timestamps or [0.0] * len(frames))
    fallback = 0
    while True:
        res = ask(model, make_prompt(len(frames)), frames, ts, min_frames=len(frames), **kw)
        if str(res.get("error") or "").startswith("context:") and len(frames) > min_frames:
            frames, ts = subsample(frames, ts, max(min_frames, len(frames) // 2))
            fallback += 1
            continue
        res["fallback"] = fallback
        return res


def finished(path, role):
    """qids of the rows of this role that hold a prediction."""
    return {str(r["qid"]) for r in read_jsonl(path)
            if r.get("role", role) == role and r.get("pred") is not None and not r.get("error")}


def make_row(row, role, fmt, res, opts=None, **extra):
    pred = res.get("pred")
    out = dict(qid=row["qid"], role=role, pred=None if pred is None else str(pred)[:PRED_CHARS], answer=row.get("answer"))
    sc = None if pred is None else score(pred, row.get("answer"), item_options(row)[0] if opts is None else opts, fmt)
    out["correct"] = None if sc is None else bool(sc)
    out.update(frames=res.get("frames", 0), fallback=res.get("fallback", 0), error=res.get("error"))
    if res.get("stop_reason") not in (None, "end_turn", "stop"):
        out["stop_reason"] = res["stop_reason"]
    out.update(extra)
    return out


def run_condition(ctx, condition, role, build, benchmarks=None, limit=None, max_tokens=None):
    """build(row, bench) -> None (item not applicable) or dict(prompt, frames=[], timestamps=[], extra={}, opts=None)."""
    if role not in ctx.models:
        log(f"{condition}: no model configured for the role {role}; skipped", "audit")
        return []
    model = ctx.models[role]
    label = ctx.models.label(role)
    summary = []
    for bench in select(ctx, benchmarks):
        samples = load_samples(ctx, bench, limit)
        if not samples:
            log(f"{condition}: {bench}: no sample", "audit")
            continue
        fmt = bench_meta(ctx, bench).get("question_format", "")
        path = ctx.path("result", condition=condition, bench=bench, model=label)
        done = finished(path, role)
        todo = [r for q, r in samples.items() if q not in done]
        n_ok = n_na = 0

        def work(row):
            try:
                b = build(row, bench)
            except Exception as e:  # noqa: BLE001
                return make_row(row, role, fmt, dict(error=f"frames: {type(e).__name__}: {str(e)[:160]}"))
            if b is None:
                return dict(qid=row["qid"], role=role, skipped=True)
            kw = dict(max_tokens=max_tokens) if max_tokens else {}
            if callable(b["prompt"]):
                res = ask_frames(model, b["prompt"], b.get("frames"), b.get("timestamps"), **kw)
            else:
                res = ask(model, b["prompt"], b.get("frames"), b.get("timestamps"), **kw)
            return make_row(row, role, fmt, res, b.get("opts"), **(b.get("extra") or {}))

        if todo:
            with JsonlWriter(path, key=("qid", "role")) as w:
                for out in pmap(work, todo, workers=int(ctx.get("workers", 8)), desc=f"{condition} {bench}"):
                    if out.get("skipped"):
                        n_na += 1
                        continue
                    w.write(out)
                    n_ok += out.get("pred") is not None
        rows = {str(r["qid"]): r for r in read_jsonl(path) if r.get("role", role) == role and r.get("pred") is not None}
        rows = {q: r for q, r in rows.items() if q in samples}
        sc = [r["correct"] for r in rows.values() if r.get("correct") is not None]
        acc = sum(sc) / len(sc) if sc else None
        log(f"{condition} [{role} = {label}] {bench}: {len(rows)}/{len(samples)} items answered"
            f"{f', {n_na} not applicable' if n_na else ''}, accuracy "
            f"{'n/a' if acc is None else f'{acc:.3f}'} over {len(sc)} scored", "audit")
        summary.append(dict(benchmark=bench, condition=condition, role=role, model=label, n=len(sc), acc=acc))
    return summary


# ------------------------------------------------------------------ builders
def visual_builder(ctx, role, kind, k, require_frames=None):
    """Prompt with frames: the question, the options as stored, k frames chosen by the rule of the role."""
    fam = family(role)

    def build(row, bench):
        video, meta = video_of(ctx, row)
        if video is None:
            raise FileNotFoundError(f"no normalized video for {row['qid']} (run the videos stage)")
        frames, ts, extra = frame_rules.select(fam, kind, video, meta, k, row["qid"], ctx)
        if not frames:
            raise RuntimeError("no frame decoded")
        q, o = row.get("question", ""), fmt_opts(row.get("options"), ctx.get("letter_options", True))
        return dict(prompt=lambda n: prompts.VISUAL.format(n=n, q=q, opts=o), frames=frames, timestamps=ts, extra=extra)
    return build


def blind_builder(ctx):
    def build(row, bench):
        return dict(prompt=prompts.BLIND.format(q=row.get("question", ""),
                                                opts=fmt_opts(row.get("options"), ctx.get("letter_options", True))))
    return build


def options_only_builder(ctx):
    def build(row, bench):
        opts, src = item_options(row)
        if not opts:
            return None
        return dict(prompt=prompts.OPTIONS_ONLY.format(opts=lettered(opts)), opts=opts,
                    extra=dict(opts_source=src, n_options=len(opts)))
    return build
