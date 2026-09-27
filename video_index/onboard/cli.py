"""``vi-onboard``: add one benchmark to the audit.

    vi-onboard check --name NAME --repo SOURCE [--adapter spec.yaml]      format check only, nothing is audited
    vi-onboard add   --name NAME --repo SOURCE [--adapter spec.yaml] [--video-repo SOURCE] [--sample-n 300]
                     [--stages sample,fetch,videos,...] [--limit N] [--force]
    vi-onboard list                                                        the registry
    vi-onboard watch [--keywords ...]                                      candidate benchmarks (manual approval)

SOURCE is a Hugging Face dataset id (``owner/name``), a local directory, or ``hf::`` / ``local::`` / ``urls::`` followed
by the target. ``add`` runs: annotations -> adapter -> items -> format check and eligibility -> the audit stages for
this benchmark -> registry and levels table -> report card.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from ..data.schema import read_jsonl, write_jsonl
from ..runner import log as _log
from . import formatcheck, registry, sources
from .adapters import AdapterError, AdapterSpec, build_items

STAGES = ["sample", "fetch", "videos", "chance", "screen", "options_only", "blind", "options_only_perm", "pool_attack",
          "single_frame", "reference", "reference_128", "shuffle", "window", "captions", "levels", "report"]
LOCAL_STAGES = {"fetch"}                 # run by this package; every other stage belongs to video_index.audit


def log(msg: str) -> None:
    _log(msg, "onboard")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def safe_name(name: str) -> str:
    return name.replace("/", "_")


def find_spec(name: str, adapter: str | None, adapters_dir: str = "configs/adapters") -> AdapterSpec:
    if adapter:
        spec = AdapterSpec.from_yaml(adapter)
    else:
        cand = [os.path.join(adapters_dir, f) for f in (f"{slug(name)}.yaml", f"{name}.yaml")]
        hit = next((c for c in cand if os.path.exists(c)), None)
        if hit is None and os.path.isdir(adapters_dir):
            for f in sorted(os.listdir(adapters_dir)):
                if f.endswith((".yaml", ".yml")):
                    s = AdapterSpec.from_yaml(os.path.join(adapters_dir, f))
                    if s.name == name:
                        hit = os.path.join(adapters_dir, f)
                        break
        spec = AdapterSpec.from_yaml(hit) if hit else AdapterSpec(name=name)
        if hit:
            log(f"adapter spec: {hit}")
    spec.name = name
    return spec


def context(a, need_models: bool):
    from ..audit.context import DEFAULTS, Context
    from ..config import Models, load_yaml
    cfg = dict(DEFAULTS)
    if a.config and os.path.exists(a.config):
        cfg.update(load_yaml(a.config))
    elif a.config and need_models:
        sys.exit(f"run configuration not found: {a.config}")
    work = os.path.abspath(a.work_dir or cfg.get("work_dir") or "work")
    models_file = a.models or cfg.get("models_file") or "configs/models.yaml"
    if os.path.exists(models_file):
        models = Models.from_file(models_file)
    elif need_models:
        sys.exit(f"model configuration not found: {models_file} (copy configs/models.example.yaml and edit it)")
    else:
        models = Models({})
    if getattr(a, "sample_n", None):
        cfg["sample_n"] = a.sample_n
    return Context(work, models, cfg)


def resolve_sources(a, spec: AdapterSpec) -> dict:
    ann = sources.parse_source(a.repo) or spec.source.get("annotations")
    vid = sources.parse_source(a.video_repo) or spec.source.get("videos") or (ann if a.repo else None)
    if a.repo and isinstance(ann, dict) and spec.source.get("annotations", {}).get("include") and "include" not in ann:
        ann["include"] = spec.source["annotations"]["include"]
    if ann is None:
        sys.exit("no annotation source: pass --repo or name source.annotations in the adapter spec")
    return dict(annotations=ann, videos=vid)


def prepare(ctx, a, spec: AdapterSpec, resolve_videos: bool) -> tuple[list[dict], dict, dict, dict]:
    """annotations -> items; returns (items of every format, adapter info, format report, sources)."""
    name = spec.name
    src = resolve_sources(a, spec)
    ann_dir = ctx.path("annotations", bench=safe_name(name))
    os.makedirs(ann_dir, exist_ok=True)
    files = sources.fetch_annotations(src["annotations"], ann_dir)
    log(f"{name}: {len(files)} annotation file(s) from {sources.label(src['annotations'])}")
    resolver = None
    if resolve_videos and src["videos"] is not None:
        index = sources.VideoIndex(src["videos"], os.path.join(ctx.work_dir, "cache", "video_index")).build()
        for s in index.skipped:
            log(f"{name}: video index skipped {s}")
        if src["videos"]["kind"] in ("hf", "local", "urls") and not index.idx:
            raise sources.SourceError(f"no video file or video-bearing archive in {sources.label(src['videos'])}")
        resolver = index.resolve
    items, info = build_items(name, ann_dir, spec, resolver)
    if resolver is not None and info["resolution_rate"] < 0.5:
        raise AdapterError(f"{name}: only {info['n_items']} of {info['n_records'] - info['n_filtered']} video references "
                           f"resolve in {sources.label(src['videos'])}; name the video field or template in the adapter spec")
    if info["n_unresolved"]:
        log(f"{name}: {info['n_unresolved']} items left out, their video is not in the source")
    rep = formatcheck.format_report(items, spec.requires_audio, spec.language)
    rep.update(adapter=info["via"], columns=info["columns"], n_rows=info["n_rows"], n_unresolved=info["n_unresolved"],
               videos_resolved=resolver is not None)
    return items, info, rep, src


def write_items(ctx, name: str, items: list[dict], rep: dict) -> str:
    mcq = [i for i in items if i["format"] == "mcq"]
    path = ctx.path("items", bench=safe_name(name))
    write_jsonl(path, mcq)
    write_jsonl(ctx.path("table", name=f"format/{safe_name(name)}_excluded.jsonl"), [i for i in items if i["format"] != "mcq"])
    flat = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in rep.items()
            if k not in ("checks", "subtasks")}
    flat["failed_rules"] = ";".join(c["rule"] for c in rep["checks"] if not c["passed"])
    registry.upsert(ctx.path("table", name="format_check.csv"), dict(name=name, **flat))
    return path


def fetch_stage(ctx, name: str, src: dict, limit: int | None = None) -> None:
    """Fetch the videos of the sample into raw_videos/<benchmark>/ and record each row's ``video_path``."""
    sp = ctx.path("samples", bench=safe_name(name))
    rows = read_jsonl(sp)
    if not rows:
        sys.exit(f"{name}: no sample at {sp}; run the sample stage first")
    if src.get("videos") is None:
        sys.exit(f"{name}: no video source; pass --video-repo")
    raw_dir = os.path.join(ctx.work_dir, "raw_videos", safe_name(name))

    def dest_of(ref: dict) -> str:
        ext = os.path.splitext(ref["member"])[1].lower()
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", (ref.get("archive") or "") + "__" + ref["member"]).strip("_")
        return os.path.join(raw_dir, base if ext in sources.VIDEO_EXT else base + ".mp4")

    refs, seen = [], set()
    for r in rows[:limit] if limit else rows:
        ref = r["video_ref"] if isinstance(r["video_ref"], dict) else dict(source="", archive="", member=str(r["video_ref"]))
        if sources.ref_key(ref) not in seen:
            seen.add(sources.ref_key(ref))
            refs.append(ref)
    log(f"{name}: fetching {len(refs)} videos from {sources.label(src['videos'])}")
    done, failed = sources.fetch_videos(src["videos"], refs, dest_of, log)
    for r in rows:
        ref = r["video_ref"] if isinstance(r["video_ref"], dict) else dict(source="", archive="", member=str(r["video_ref"]))
        p = done.get(sources.ref_key(ref))
        r["video_path"] = os.path.relpath(p, ctx.work_dir) if p else None
    write_jsonl(sp, rows)
    if failed:
        write_jsonl(ctx.path("log", name=f"fetch_failed_{safe_name(name)}.jsonl"), [dict(video=k, error=v) for k, v in failed.items()])
    log(f"{name}: {len(done)} videos fetched, {len(failed)} failed"
        + (f" (logs/fetch_failed_{safe_name(name)}.jsonl)" if failed else ""))


def run_audit_stages(ctx, name: str, stages: list[str], limit: int | None) -> bool:
    try:
        from ..audit.cli import run_stages
    except ImportError as e:
        if stages == ["sample"]:
            items = read_jsonl(ctx.path("items", bench=safe_name(name)))
            sample, full = registry.stratified_sample(items, int(ctx.get("sample_n", 300)), int(ctx.get("seed", 42)))
            write_jsonl(ctx.path("samples", bench=safe_name(name)), sample)
            log(f"{name}: sample of {len(sample)} items" + (" (all items)" if full else ""))
            return True
        log(f"the audit stages are not installed ({e}); stopped before: {', '.join(stages)}")
        return False
    run_stages(ctx, stages, benchmarks=[safe_name(name)], limit=limit)
    return True


def cmd_check(a) -> int:
    spec = find_spec(a.name, a.adapter)
    ctx = context(a, need_models=False)
    try:
        items, info, rep, src = prepare(ctx, a, spec, resolve_videos=bool(a.resolve_videos))
    except (AdapterError, sources.SourceError) as e:
        print(f"check failed: {e}", file=sys.stderr)
        return 2
    print(formatcheck.render_report(a.name, rep, int(ctx.get("sample_n", 300))))
    print(f"  columns           {info['columns']}" if info["via"] == "generic" else "  adapter           Python hook")
    if not rep["videos_resolved"]:
        print("  videos            not resolved against the video source (pass --resolve-videos)")
    else:
        print(f"  video references  {info['n_items']} of {info['n_records'] - info['n_filtered']} resolve in "
              f"{sources.label(src['videos'])} (rate {info['resolution_rate']})")
    if a.write:
        print(f"  items written to  {write_items(ctx, a.name, items, rep)}")
    if a.show:
        for it in [i for i in items if i["format"] == "mcq"][:a.show]:
            print(json.dumps({k: it[k] for k in ("qid", "question", "options", "answer", "subtask", "video_ref")}, ensure_ascii=False))
    return 0 if rep["eligible"] else 1


def cmd_add(a) -> int:
    spec = find_spec(a.name, a.adapter)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()] if a.stages else list(STAGES)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s) {unknown}; known: {', '.join(STAGES)}")
    ctx = context(a, need_models=any(s not in ("sample", "fetch", "videos", "chance", "screen", "levels", "report") for s in stages))
    try:
        items, info, rep, src = prepare(ctx, a, spec, resolve_videos=not a.no_resolve)
    except (AdapterError, sources.SourceError) as e:
        print(f"onboarding stopped: {e}", file=sys.stderr)
        return 2
    print(formatcheck.render_report(a.name, rep, int(ctx.get("sample_n", 300))))
    if not rep["eligible"] and not a.force:
        print("onboarding stopped: the benchmark fails the eligibility rules above (--force overrides)", file=sys.stderr)
        registry.register(ctx, a.name, spec, src, rep, status="not_eligible")
        return 1
    write_items(ctx, a.name, items, rep)
    registry.register(ctx, a.name, spec, src, rep, status="items_ready")
    block: list[str] = []
    ok = True

    def flush():
        nonlocal block, ok
        if block and ok:
            ok = run_audit_stages(ctx, a.name, block, a.limit)
        block = []

    for s in stages:
        if s in LOCAL_STAGES:
            flush()
            if ok:
                fetch_stage(ctx, a.name, src, a.limit)
        else:
            if s == "sample" and os.path.exists(ctx.path("samples", bench=safe_name(a.name))) and not a.resample:
                log(f"{a.name}: the sample exists and is kept (--resample draws it again)")
                continue
            block.append(s)
            if s == "sample":
                flush()
    flush()
    n_sampled = len(read_jsonl(ctx.path("samples", bench=safe_name(a.name))))
    row = registry.register(ctx, a.name, spec, src, rep, n_sampled=n_sampled, status="onboarded" if ok else "items_ready")
    print(f"registry: {a.name} status {row['status']}, sample {n_sampled}, break level {row['break_level'] or 'not computed'}")
    return 0 if ok else 3


def cmd_list(a) -> int:
    ctx = context(a, need_models=False)
    rows = registry.read_csv(ctx.path("table", name="registry.csv"))
    if not rows:
        print("no benchmark onboarded yet")
        return 0
    cols = ["name", "status", "n_mcq", "n_sampled", "chance", "break_level", "capability_group", "source"]
    width = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(width[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(width[c]) for c in cols))
    return 0


def cmd_watch(a) -> int:
    from . import watch
    ctx = context(a, need_models=False)
    return watch.run(ctx, keywords=a.keywords, days=a.days, use_labeler=a.label, limit=a.max)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vi-onboard", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, named=True):
        p.add_argument("--config", default="configs/onboard.yaml", help="run configuration (configs/onboard.example.yaml)")
        p.add_argument("--models", default=None, help="model roles (default: models_file of the run configuration)")
        p.add_argument("--work-dir", default=None)
        if named:
            p.add_argument("--name", required=True)
            p.add_argument("--repo", default=None, help="annotation source: HF dataset id, directory, or kind::target")
            p.add_argument("--video-repo", default=None, help="video source when it differs from --repo")
            p.add_argument("--adapter", default=None, help="adapter spec (YAML); default: configs/adapters/<name>.yaml or detection")
            p.add_argument("--sample-n", type=int, default=None)

    p = sub.add_parser("check", help="format check and eligibility, nothing is audited")
    common(p)
    p.add_argument("--resolve-videos", action="store_true", help="also resolve every video reference in the video source")
    p.add_argument("--write", action="store_true", help="write items/<name>.jsonl")
    p.add_argument("--show", type=int, default=0, help="print the first N items")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("add", help="onboard the benchmark")
    common(p)
    p.add_argument("--stages", default=None, help="comma list; default: " + ",".join(STAGES))
    p.add_argument("--limit", type=int, default=None, help="items per stage (smoke test)")
    p.add_argument("--force", action="store_true", help="continue although an eligibility rule fails")
    p.add_argument("--no-resolve", action="store_true", help="do not resolve video references in the video source")
    p.add_argument("--resample", action="store_true", help="draw the sample again")
    p.set_defaults(fn=cmd_add)
    p = sub.add_parser("list", help="the registry")
    common(p, named=False)
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("watch", help="list candidate benchmarks (approval stays manual)")
    common(p, named=False)
    p.add_argument("--keywords", nargs="*", default=None)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--max", type=int, default=100)
    p.add_argument("--label", action="store_true", help="ask the labeler role for a suggestion per candidate")
    p.set_defaults(fn=cmd_watch)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
