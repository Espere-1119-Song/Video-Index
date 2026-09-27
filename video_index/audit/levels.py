"""Levels stage: chance c, reference accuracy s*, the largest attacker margin per level, and the breaking level.

Items      the multiple-choice items of the sample (chance.is_mcq_item) that the screen did not remove.
s*         accuracy of the reference role at 32 frames on its scored items (at least 30, else no level).
margin     accuracy of an attacker - mean item chance, on the items scored for both the reference and the attacker;
           attackers with fewer than 30 common items are dropped.
eps_l      largest margin among the attackers of level l, then the running maximum over the levels
           option < text < pool < frame < order; a level without an attacker value is not measured.
level      the lowest level l with eps_l >= s* - c - delta (delta = 0.05; also reported at 0.03 and 0.08), else
           "unbroken".
128-frame rescreen  when s* - c <= delta and the 128-frame reference accuracy exceeds c by more than delta, the 128-frame
           accuracy is used as s*; when it does not, the level is the first measured level and
           reference_near_chance = 1.
threshold  2.326 sqrt(c (1 - c) / n_ref).

Attackers
  option   fixed_position (answers the modal gold position of the sample); options_only (file must cover >= 95 % of
           the items that carry options)
  text     blind, one attacker per model with a result file
  pool     every attacker of tables/pool_attack.csv (margin re-based: eps_last20 + chance_export - c)
  frame    single_frame of the roles in `frame_level_roles` (default: reference); captions (per-frame captions; the
           file must hold >= 95 % of the reference rows)
  order    shuffle and window of the roles reference and attacker; measured when one shuffle run and one window run exist
Second reference: the attacker role at 32 frames; the profile is recomputed on its items.

Outputs    tables/levels.csv, tables/level_attackers.csv, tables/position_shares.csv, tables/audited_benchmarks.txt
           (audited set = benchmarks with a level: >= 30 multiple-choice items with a reference score)
"""
from __future__ import annotations

import csv
import math
import os
from collections import Counter

from ..runner import log
from . import chance as chance_mod
from .items import bench_meta, item_options, load_samples, read_results, result_files, score_results, select
from .sample import removed_items

Z = 2.326
DELTA = 0.05
DELTAS_ALT = {"delta3": 0.03, "delta8": 0.08}
LEVELS = ["option", "text", "pool", "frame", "order"]
EPS_COL = {"option": "eps_opt", "text": "eps_text", "pool": "eps_pool", "frame": "eps_frame", "order": "eps_order"}
MIN_N = 30
PROBE_MIN_COVERAGE = 0.95
CAPTIONS_MIN_COVERAGE = 0.95
LEVEL_OF = {"fixed_position": "option", "options_only": "option", "blind": "text", "pool": "pool", "single_frame": "frame",
            "captions": "frame", "shuffle": "order", "window": "order"}


def kind_of(attacker):
    """Attacker names are "<kind>" or "<kind>:<model or role>"."""
    return attacker.split(":", 1)[0]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def profile(ref, attackers, cvec, pool, level_of=None, min_n=MIN_N):
    """ref / attackers[name]: {qid: bool}; cvec: {qid: item chance}; pool: {name: (eps, chance_export, n)}.
    level_of(name) -> level; default: by the kind of the attacker name."""
    level_of = level_of or (lambda a: LEVEL_OF.get(kind_of(a)))
    I = set(ref)
    n_ref = len(I)
    if not n_ref:
        return None
    s_star = mean(ref[q] for q in I)
    c = mean(cvec[q] for q in I)
    thr = Z * math.sqrt(max(c * (1 - c), 0.0) / n_ref)
    res, dropped = {}, []
    for a, sc in attackers.items():
        J = I & set(sc)
        if len(J) < min_n:
            if J:
                dropped.append(f"{a} n={len(J)}<{min_n}")
            continue
        acc = mean(sc[q] for q in J)
        res[a] = dict(acc=acc, margin=acc - mean(cvec[q] for q in J), n=len(J))
    for a, (eps, chance_export, n) in pool.items():
        acc = eps + chance_export
        res[a] = dict(acc=acc, margin=acc - c, n=n, eps_last20=eps, chance_export=chance_export)
    eps, run, measured = {}, None, []
    for l in LEVELS:
        names = [a for a in res if level_of(a) == l]
        vals = [res[a]["margin"] for a in names]
        if l == "order" and not (any(kind_of(a).endswith("shuffle") for a in names)
                                 and any(kind_of(a).endswith("window") for a in names)):
            vals = []
        if vals:
            measured.append(l)
            run = max(vals) if run is None else max(run, max(vals))
        eps[l] = run
    return dict(n_ref=n_ref, s_star=s_star, c=c, thr=thr, res=res, eps=eps, measured=measured, dropped=dropped)


def assign_level(p, delta=DELTA, s128=None):
    """-> dict(level, near, rescreened, s_used)."""
    s_used, near, resc = p["s_star"], p["s_star"] - p["c"] <= delta, 0
    if near and s128 is not None and s128 - p["c"] > delta:
        s_used, resc = s128, 1
    if near and not resc:
        return dict(level=p["measured"][0] if p["measured"] else "unbroken", near=1, rescreened=0, s_used=s_used)
    for l in LEVELS:
        e = p["eps"][l]
        if e is not None and e >= s_used - p["c"] - delta:
            return dict(level=l, near=int(near), rescreened=resc, s_used=s_used)
    return dict(level="unbroken", near=int(near), rescreened=resc, s_used=s_used)


def fixed_position(pos):
    """pos {qid: gold letter} -> (modal letter, {qid: bool}, counts)."""
    cnt = Counter(pos.values())
    if not cnt:
        return None, {}, cnt
    modal = max(sorted(cnt), key=cnt.get)
    return modal, {q: (L == modal) for q, L in pos.items()}, cnt


def r4(v):
    return "" if v is None else f"{v:.4f}"


def load_pool(ctx):
    out = {}
    p = os.path.join(ctx.work_dir, "tables", "pool_attack.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            try:
                out.setdefault(r["benchmark"], {})[f"pool:{r['attacker']}"] = (
                    float(r["eps_last20"]), float(r["chance_export"]), int(float(r.get("n20") or r["n_items"])))
            except (TypeError, ValueError, KeyError):
                continue
    return out


def gather(ctx, bench, samples, fmt, ref, notes):
    """Attacker scores of one benchmark, {name: {qid: bool}}."""
    att = {}
    n_with_opts = sum(1 for r in samples.values() if item_options(r)[0])
    for label, path in result_files(ctx, "options_only", bench).items():
        n = len([q for q in read_results(path) if q in samples])
        if n and n_with_opts and n >= PROBE_MIN_COVERAGE * n_with_opts:
            att[f"options_only:{label}"] = score_results(path, samples, fmt)
        elif n:
            notes.append(f"options_only ({label}) covers {n}/{n_with_opts} items with options; not used")
    for label, path in result_files(ctx, "blind", bench).items():
        att[f"blind:{label}"] = score_results(path, samples, fmt)
    for role in ctx.get("frame_level_roles", ["reference"]):
        if role in ctx.models:
            p = ctx.path("result", condition="single_frame", bench=bench, model=ctx.models.label(role))
            att[f"single_frame:{role}"] = score_results(p, samples, fmt, role)
    base = len(ref) if ref else len(samples)
    for label, path in result_files(ctx, "captions", bench).items():
        n = len([q for q in read_results(path) if q in samples])
        if n >= CAPTIONS_MIN_COVERAGE * max(1, base):
            att[f"captions:{label}"] = score_results(path, samples, fmt)
        elif n:
            notes.append(f"captions ({label}) cover {n}/{base} reference rows; not used")
    for cond in ("shuffle", "window"):
        for role in ctx.get("order_roles", ["reference", "attacker"]):
            if role in ctx.models:
                p = ctx.path("result", condition=cond, bench=bench, model=ctx.models.label(role))
                att[f"{cond}:{role}"] = score_results(p, samples, fmt, role)
    return {k: v for k, v in att.items() if v}


def audit_benchmark(ctx, bench, limit=None, pool=None):
    """-> (row of levels.csv, attacker rows, position rows)."""
    delta = float(ctx.get("delta", DELTA))
    min_n = int(ctx.get("min_items", MIN_N))
    m = bench_meta(ctx, bench)
    fmt = m.get("question_format", "")
    removed = removed_items(ctx, bench)
    samples_all = {q: r for q, r in load_samples(ctx, bench, limit).items() if q not in removed}
    bc = chance_mod.bench_chance(samples_all, m)
    mcq = set(bc["mcq_qids"])
    samples = {q: r for q, r in samples_all.items() if q in mcq}
    notes = []
    row = dict(benchmark=bench, n_items=len(samples_all), n_mcq=len(mcq), n_excluded=len(samples_all) - len(mcq),
               low_n_mcq=int(0 < len(mcq) < min_n), c_source=bc["c_source"], n_ref=0)
    if not mcq:
        row["notes"] = "no multiple-choice item"
        return row, [], []
    pos = {q: L for q, L in bc["pos"].items() if q in mcq}
    modal, fixed, cnt = fixed_position(pos)
    shares = [dict(benchmark=bench, position=L, n_gold=cnt[L], share=r4(cnt[L] / len(pos)), n_positioned=len(pos),
                   n_items=len(samples_all)) for L in sorted(cnt)]
    row.update(modal_position=modal or "", n_positioned=len(pos))
    ref_label = ctx.models.label("reference") if "reference" in ctx.models else ""
    ref = score_results(ctx.path("result", condition="reference", bench=bench, model=ref_label), samples, fmt,
                        "reference") if ref_label else {}
    row["reference_model"] = ref_label if ref else ""
    if not ref:
        notes.append("no reference run")
    elif len(ref) < min_n:
        notes.append(f"reference n={len(ref)}<{min_n}, no level")
        ref = {}
    att = gather(ctx, bench, samples, fmt, ref, notes)
    if modal:
        att["fixed_position"] = fixed
    pool_b = (pool if pool is not None else load_pool(ctx)).get(bench, {})
    p = profile(ref, att, bc["cvec"], pool_b, min_n=min_n) if ref else None
    if p is None:
        row["notes"] = "; ".join(notes)
        return row, [], shares
    notes += p["dropped"]
    if not any(kind_of(a) == "captions" for a in p["res"]):
        notes.append("no per-frame caption attacker; frame level = single frame only")
    if "order" not in p["measured"] and any(kind_of(a) in ("shuffle", "window") for a in p["res"]):
        notes.append("order level not measured: one shuffle run and one window run are needed")
    s128 = None
    if "reference_long" in ctx.models:
        sc = score_results(ctx.path("result", condition="reference_128", bench=bench, model=ctx.models.label("reference_long")),
                           samples, fmt, "reference_long")
        if len(sc) >= min_n:
            s128 = mean(sc.values())
    al = assign_level(p, delta, s128)
    row.update(n_ref=p["n_ref"], c=r4(p["c"]), s_star=r4(p["s_star"]), threshold=r4(p["thr"]),
               **{EPS_COL[l]: r4(p["eps"][l]) if l in p["measured"] else "" for l in LEVELS},
               levels_measured=";".join(p["measured"]),
               eps_opt_provisional=int(not any(kind_of(a) == "options_only" for a in p["res"])),
               frame_pending=int(not any(kind_of(a) == "captions" for a in p["res"])),
               break_level=al["level"], reference_near_chance=al["near"], rescreened_128=al["rescreened"],
               s_star_used=r4(al["s_used"]), s_star_128=r4(s128), delta=delta)
    for k, d in DELTAS_ALT.items():
        row[f"break_level_{k}"] = assign_level(p, d, s128)["level"]
    if "attacker" in ctx.models:
        ref2 = score_results(ctx.path("result", condition="reference", bench=bench, model=ctx.models.label("attacker")),
                             samples, fmt, "attacker")
        ref2 = ref2 if len(ref2) >= min_n else {}
        p2 = profile(ref2, att, bc["cvec"], pool_b, min_n=min_n) if ref2 else None
        if p2:
            lv2 = assign_level(p2, delta)["level"]
            row.update(sr_model=ctx.models.label("attacker"), sr_n_ref=p2["n_ref"], sr_s_star=r4(p2["s_star"]),
                       sr_c=r4(p2["c"]), second_reference_break_level=lv2, sr_agree=int(lv2 == al["level"]))
    row["notes"] = "; ".join(notes)
    arows = [dict(benchmark=bench, level=LEVEL_OF.get(kind_of(a), ""), attacker=a, acc=r4(d["acc"]), margin=r4(d["margin"]),
                  n=d["n"]) for a, d in sorted(p["res"].items(), key=lambda x: (LEVELS.index(LEVEL_OF[kind_of(x[0])]), x[0]))]
    for label, path in result_files(ctx, "captions_video", bench).items():        # reported only
        sc = score_results(path, samples, fmt)
        J = set(sc) & set(ref)
        if len(J) >= min_n:
            acc = mean(sc[q] for q in J)
            arows.append(dict(benchmark=bench, level="(reported)", attacker=f"captions_video:{label}", acc=r4(acc),
                              margin=r4(acc - mean(bc["cvec"][q] for q in J)), n=len(J)))
    return row, arows, shares


COLS = ["benchmark", "n_items", "n_mcq", "n_excluded", "low_n_mcq", "n_ref", "c", "c_source", "s_star", "reference_model",
        "threshold", "modal_position", "n_positioned", "eps_opt", "eps_text", "eps_pool", "eps_frame", "eps_order",
        "levels_measured", "eps_opt_provisional", "frame_pending", "delta", "break_level", "reference_near_chance",
        "rescreened_128", "s_star_used", "s_star_128", "break_level_delta3", "break_level_delta8", "sr_model", "sr_n_ref",
        "sr_s_star", "sr_c", "second_reference_break_level", "sr_agree", "notes"]
ATT_COLS = ["benchmark", "level", "attacker", "acc", "margin", "n"]
POS_COLS = ["benchmark", "position", "n_gold", "share", "n_positioned", "n_items"]


def _merge(path, cols, rows, benches):
    old = [r for r in csv.DictReader(open(path)) if r["benchmark"] not in benches] if os.path.exists(path) else []
    rows = sorted(old, key=lambda r: r["benchmark"]) + rows
    rows.sort(key=lambda r: r["benchmark"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


def run(ctx, benchmarks=None, limit=None):
    pool = load_pool(ctx)
    rows, arows, shares, benches = [], [], [], set()
    for bench in select(ctx, benchmarks):
        if not os.path.exists(ctx.path("samples", bench=bench)):
            continue
        r, a, s = audit_benchmark(ctx, bench, limit, pool)
        benches.add(bench)
        rows.append(r)
        arows += a
        shares += s
        log(f"levels: {bench}: " + (f"c {r['c']}, s* {r['s_star']} (n = {r['n_ref']}), level {r['break_level']}"
                                    if r.get("break_level") else f"no level ({r.get('notes', '')})"), "audit")
    allrows = _merge(ctx.path("table", name="levels.csv"), COLS, rows, benches)
    _merge(ctx.path("table", name="level_attackers.csv"), ATT_COLS, arows, benches)
    _merge(ctx.path("table", name="position_shares.csv"), POS_COLS, shares, benches)
    with open(ctx.path("table", name="audited_benchmarks.txt"), "w") as f:
        f.write("".join(f"{r['benchmark']}\n" for r in allrows if r.get("break_level")))
    return rows
