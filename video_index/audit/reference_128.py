"""128-frame reference run (role reference_long), used by the rescreen rule of the levels stage: a benchmark whose
32-frame reference accuracy is within delta of chance is assigned its level from the 128-frame accuracy when that
accuracy exceeds chance by more than delta. The stage runs on those benchmarks only unless rescreen_all is set."""
from ..runner import log
from . import chance as chance_mod
from .conditions import run_condition, visual_builder
from .items import bench_meta, load_samples, score_results, select

ROLE = "reference_long"


def near_chance(ctx, bench, limit=None):
    samples = load_samples(ctx, bench, limit)
    m = bench_meta(ctx, bench)
    d = chance_mod.bench_chance(samples, m)
    mcq = {q: samples[q] for q in d["mcq_qids"]}
    ref = score_results(ctx.path("result", condition="reference", bench=bench, model=ctx.models.label("reference")),
                        mcq, m.get("question_format", ""), "reference")
    if not ref:
        return None
    c = sum(d["cvec"][q] for q in ref) / len(ref)
    return sum(ref.values()) / len(ref) - c <= float(ctx.get("delta", 0.05))


def run(ctx, benchmarks=None, limit=None):
    k = int(ctx.get("reference_long_frames", 128))
    out = []
    for bench in select(ctx, benchmarks):
        if not ctx.get("rescreen_all", False):
            nc = near_chance(ctx, bench, limit)
            if not nc:
                log(f"reference_128: {bench}: " + ("no reference run" if nc is None else
                                                   "reference accuracy exceeds chance by more than delta; not needed"), "audit")
                continue
        out += run_condition(ctx, "reference_128", ROLE, visual_builder(ctx, ROLE, "uniform", k), [bench], limit)
    return out
