"""``vi-audit``: the attack pyramid.

    vi-audit run    --config configs/audit.yaml --models configs/models.yaml [--work-dir DIR]
                    [--stages sample,videos,...] [--bench "A|B"] [--limit N] [--workers N]
    vi-audit status --config configs/audit.yaml [--models configs/models.yaml] [--work-dir DIR] [--bench "A|B"]

Python API: ``run_stages(ctx, stages, benchmarks=None, limit=None)``.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

from ..config import Models
from ..data.schema import read_jsonl
from ..runner import log
from .context import DEFAULTS, Context

# stage name -> (module, function), in pipeline order
STAGES = {
    "sample": ("sample", "run"),
    "videos": ("videos", "run"),
    "chance": ("chance", "run"),
    "screen": ("screen", "run"),
    "refill": ("sample", "run_refill"),
    "options_only": ("options_only", "run"),
    "blind": ("blind", "run"),
    "options_only_perm": ("options_only_perm", "run"),
    "pool_attack": ("pool_attack", "run"),
    "single_frame": ("single_frame", "run"),
    "reference": ("reference", "run"),
    "reference_128": ("reference_128", "run"),
    "shuffle": ("shuffle", "run"),
    "window": ("window", "run"),
    "captions": ("captions", "run"),
    "levels": ("levels", "run"),
    "report": ("report", "run"),
}
PYRAMID = [s for s in STAGES if s != "refill"]                 # refill changes the sample; it runs on request


def _register(package: str) -> list[str]:
    """Add the stages of a later part of the pipeline (modules named by absolute path)."""
    mod = importlib.import_module(package)
    for name, target in mod.STAGES.items():
        STAGES[name] = (target, "run")
    return list(getattr(mod, "ORDER", mod.STAGES))


ATTRIBUTION = _register("video_index.audit.attribution")
POOL = _register("video_index.pool")
COMPOSE = _register("video_index.compose")
# named groups accepted by --stages, besides single stage names
GROUPS = {"pyramid": PYRAMID, "attribution": ATTRIBUTION, "pool": POOL, "compose": COMPOSE,
          "all": PYRAMID + ATTRIBUTION + POOL + COMPOSE}
DEFAULT_STAGES = PYRAMID
NO_MODEL = {"sample", "videos", "chance", "screen", "refill", "pool_attack", "levels", "report", "attribution_preclass",
            "attribution_tables", "pool_items", "pool_select", "compose_select", "compose_export"}


def expand(stages) -> list[str]:
    """Stage names and group names -> stage names, keeping the pipeline order."""
    out = []
    for s in stages:
        out += GROUPS.get(s, [s])
    return out
CONDITIONS = ["options_only", "blind", "options_only_perm", "single_frame", "reference", "reference_128", "shuffle",
              "window", "captions", "captions_video"]


def run_stages(ctx, stages, benchmarks=None, limit=None):
    """Run the stages in pipeline order; returns {stage: summary rows}."""
    stages = expand(stages)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; known: {', '.join(STAGES)}; groups: {', '.join(GROUPS)}")
    out = {}
    for s in [s for s in STAGES if s in stages]:
        mod, fn = STAGES[s]
        log(f"stage {s}", "audit")
        module = importlib.import_module(mod if mod.startswith("video_index.") else f"{__package__}.{mod}")
        out[s] = getattr(module, fn)(ctx, benchmarks, limit)
    if "refill" in stages and out.get("refill"):
        b = [r["benchmark"] for r in out["refill"]]
        log(f"stage videos (items added by the refill: {', '.join(b)})", "audit")
        out["videos_refill"] = importlib.import_module(f"{__package__}.videos").run(ctx, b, limit)
    return out


def context(a, need_models=True):
    models = a.models
    if not need_models and not (models and os.path.exists(models)):
        from ..config import load_yaml
        cfg = {**DEFAULTS, **(load_yaml(a.config) if a.config else {})}
        cfg["normalize"] = {**DEFAULTS["normalize"], **(cfg.get("normalize") or {})}
        mf = cfg.get("models_file")
        ctx = Context(os.path.abspath(a.work_dir or cfg.get("work_dir") or "work"),
                      Models.from_file(mf) if mf and os.path.exists(mf) else Models({}), cfg)
    else:
        ctx = Context.from_files(a.config, models, a.work_dir)
    if getattr(a, "workers", None):
        ctx.cfg["workers"] = a.workers
    return ctx


def _benches(a):
    return [b.strip() for b in a.bench.split("|") if b.strip()] if a.bench else None


def cmd_run(a):
    stages = expand([s.strip() for s in a.stages.split(",") if s.strip()]) if a.stages else list(DEFAULT_STAGES)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s) {unknown}; known: {', '.join(STAGES)}")
    ctx = context(a, need_models=any(s not in NO_MODEL for s in stages))
    out = run_stages(ctx, stages, _benches(a), a.limit)
    for s, rows in out.items():
        for r in rows or []:
            if isinstance(r, dict) and r.get("acc") is not None and "condition" in r:
                print(f"{s:<18} {r['benchmark']:<28} {r['role']:<14} {r['model']:<20} n={r['n']:<4} acc={r['acc']:.3f}")
    return 0


def cmd_status(a):
    from .items import select
    ctx = context(a, need_models=False)
    benches = _benches(a) or select(ctx)
    print(f"workspace {ctx.work_dir}: {len(benches)} benchmark(s)")
    for b in benches:
        rows = read_jsonl(ctx.path("samples", bench=b))
        n_vid = sum(1 for r in rows if r.get("video_id") and os.path.exists(ctx.path("video", video_id=r["video_id"])))
        print(f"\n{b}: {len(rows)} sampled items, {n_vid} with a normalized video")
        for cond in CONDITIONS:
            d = os.path.join(ctx.work_dir, "results", cond)
            for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                if not (f.startswith(b + "__") and f.endswith(".jsonl")):
                    continue
                per = {}
                for r in read_jsonl(os.path.join(d, f)):
                    k = r.get("role", "")
                    done = per.setdefault(k, [set(), 0, 0])
                    if r.get("pred") is not None:
                        key = (r["qid"], r.get("permutation_id"))
                        if key not in done[0]:
                            done[0].add(key)
                            done[1] += r.get("correct") is not None
                            done[2] += bool(r.get("correct"))
                for role, (seen, n_sc, n_ok) in sorted(per.items()):
                    acc = f"{n_ok / n_sc:.3f}" if n_sc else "n/a"
                    print(f"  {cond:<18} {f[len(b) + 2:-6]:<22} {role:<14} answered {len(seen):<5} scored {n_sc:<5} accuracy {acc}")
    t = os.path.join(ctx.work_dir, "tables")
    if os.path.isdir(t):
        print("\ntables: " + ", ".join(sorted(f for f in os.listdir(t) if os.path.isfile(os.path.join(t, f)))))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vi-audit", description="Attack-pyramid audit of video benchmarks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("run", cmd_run), ("status", cmd_status)):
        p = sub.add_parser(name)
        p.add_argument("--config", default=None, help="run configuration (configs/audit.example.yaml)")
        p.add_argument("--models", default=None, help="model roles (configs/models.example.yaml)")
        p.add_argument("--work-dir", default=None)
        p.add_argument("--bench", default=None, help="benchmark names separated by |")
        if name == "run":
            p.add_argument("--stages", default=None, help="comma list of stages or groups (" + ", ".join(GROUPS) + "); default: pyramid = " + ",".join(DEFAULT_STAGES))
            p.add_argument("--limit", type=int, default=None, help="first N items of every sample")
            p.add_argument("--workers", type=int, default=None)
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
