"""Option-permutation audit (role attacker): every multiple-choice item is answered under the original option order
(permutation 0) and under eight seeded permutations of its options.

Input     `permutation_input: frames` (default, the protocol of the study): the 32-frame attacker condition with the
          permuted options; permutation 0 is the attacker row of results/reference/ when it exists.
          `permutation_input: options_only`: no frames and no question, the options only.
Items     2 to 16 options, no empty option, letter prefixes absent or in A, B, C order, options that are not bare
          letters, a question text without a lettered option list, a gold answer that resolves to an option.
Seeds     permutation k of an item: random.Random(f"{seed}:{qid}:{k}").shuffle(range(n)); perm[i] = original index of
          the option at new position i.
Outputs   results/options_only_perm/<bench>__<model>.jsonl   one row per (qid, permutation_id)
          tables/permutation_runs.csv            benchmark, permutation_id, acc, n_scored, n_rows
          tables/permutation_by_benchmark.csv    width = acc_max - acc_min over the permutations on the common items,
                                                 threshold = 2.326 sqrt(c (1 - c) / n), item_consistency = share of
                                                 items whose predicted option is the same original option in every
                                                 permutation
"""
from __future__ import annotations

import csv
import math
import os
import random
import re

from ..data.schema import JsonlWriter, read_jsonl
from ..runner import ask, log, pmap
from ..scoring import _MULTI, _letter_set, coerce_options, mcq_letter, score
from . import frame_rules, prompts
from .conditions import PRED_CHARS, ask_frames
from .items import bench_meta, fmt_opts, load_samples, select, video_of
from .sample import removed_items

Z = 2.326
MAX_OPTIONS = 16
_PFX = re.compile(r"^\s*[\(\[]?([A-Pa-p])[\)\]\.,:]\s*")
_Q_MARKER = re.compile(r"(?m)^\s*[\(\[]?[A-P][\)\]\.:]\s")
_LETTER_ONLY = re.compile(r"^\s*[A-Pa-p]\s*$")
ROLE = "attacker"


def letter(i):
    return chr(65 + i)


def permutation(qid, k, n, seed=42):
    perm = list(range(n))
    if k:
        random.Random(f"{seed}:{qid}:{k}").shuffle(perm)
    return perm


def options_in_question(question, bodies):
    q = str(question or "")
    if len(_Q_MARKER.findall(q)) >= 2:
        return True
    hits = 0
    for b in bodies:
        b = b.strip()
        if len(b) >= 2 and re.search(r"[\(\[]?[A-P][\)\]\.:]\s*" + re.escape(b[:60]), q):
            hits += 1
    return hits >= 2


def resolve_answer(answer, co):
    """-> ("single", [index]) | ("multi", [indices]) | (None, None)."""
    if answer is None:
        return None, None
    n, a = len(co), str(answer)
    if _MULTI.match(a):
        idx = sorted({ord(c.upper()) - 65 for c in re.findall(r"[A-Pa-p]", a)})
        return ("multi", idx) if idx and all(0 <= i < n for i in idx) else (None, None)
    L = mcq_letter(a, co)
    if L is None or ord(L) - 65 >= n:
        return None, None
    return "single", [ord(L) - 65]


def option_shape(question, options, answer):
    """-> (shape, None) or (None, reason the item is left out)."""
    co = coerce_options(options)
    if not co:
        return None, "no_options"
    if len(co) < 2:
        return None, "lt2_options"
    if len(co) > MAX_OPTIONS:
        return None, "too_many_options"
    if any(not isinstance(x, str) for x in co):
        return None, "option_not_string"
    if any(not t.strip() for t in co):
        return None, "empty_option"
    seq = [bool(_PFX.match(t)) and _PFX.match(t).group(1).upper() == letter(i) for i, t in enumerate(co)]
    if all(seq):
        labeled = True
    elif not any(seq):
        labeled = False
    else:
        return None, "partial_labels"
    bodies = [t[_PFX.match(t).end():] if labeled else t for t in co]
    if all(_LETTER_ONLY.match(b) for b in bodies):
        return None, "options_letter_only"
    if options_in_question(question, bodies):
        return None, "options_in_question"
    kind, gold = resolve_answer(answer, co)
    if kind is None:
        return None, "answer_unresolved"
    return dict(n=len(co), texts=list(co), labeled=labeled, kind=kind, gold_idx=gold), None


def render(shape, perm):
    """Option list in the permuted order; letter prefixes are re-assigned to the new positions."""
    out = []
    for i, oi in enumerate(perm):
        t = shape["texts"][oi]
        if shape["labeled"]:
            m = _PFX.match(t)
            t = t[:m.start(1)] + (letter(i) if m.group(1).isupper() else letter(i).lower()) + t[m.end(1):]
        out.append(t)
    return out


def new_answer(shape, perm):
    pos = {oi: i for i, oi in enumerate(perm)}
    return (",".join(letter(i) for i in shape["gold_idx"]), ",".join(sorted(letter(pos[i]) for i in shape["gold_idx"])))


def mapping_of(perm):
    """new letter -> original letter."""
    return {letter(i): letter(oi) for i, oi in enumerate(perm)}


def pred_identity(pred, options, kind, mapping=None):
    """Original option chosen by a prediction, or None when it does not resolve."""
    if pred is None:
        return None
    n = len(options) if options else 0
    if kind == "multi":
        s = _letter_set(pred, wide=n > 10)
        if not s:
            return None
        out = []
        for L in s:
            if mapping is not None:
                if L not in mapping:
                    return None
                L = mapping[L]
            elif ord(L) - 65 >= n:
                return None
            out.append(L)
        return tuple(sorted(out))
    L = mcq_letter(pred, options)
    if L is None:
        return None
    if mapping is not None:
        return mapping.get(L)
    return L if ord(L) - 65 < n else None


def plan(samples, n_perms=8, seed=42):
    """{(qid, k): plan row} for k = 0..n_perms, and the counts of the items left out by reason."""
    out, skipped = {}, {}
    for qid, r in samples.items():
        shape, why = option_shape(r.get("question"), r.get("options"), r.get("answer"))
        if shape is None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        for k in range(n_perms + 1):
            perm = permutation(qid, k, shape["n"], seed)
            a0, a1 = new_answer(shape, perm)
            out[(qid, k)] = dict(qid=qid, permutation_id=k, perm=perm, options_new=render(shape, perm), answer_orig=a0,
                                 answer_new=a1, mapping=mapping_of(perm), answer_kind=shape["kind"], n_options=shape["n"])
    return out, skipped


def bench_stats(bench, plan_rows, rows, fmt, n_perms=8):
    """rows: {(qid, k): result row}. -> (runs, summary)."""
    items = sorted({q for q, _ in plan_rows})
    res = {k: {} for k in range(n_perms + 1)}
    for (q, k), pr in plan_rows.items():
        r = rows.get((q, k))
        if not r or r.get("pred") is None:
            continue
        sc = score(r["pred"], pr["answer_new"], pr["options_new"], fmt)
        if sc is None:
            continue
        res[k][q] = (bool(sc), pred_identity(r["pred"], pr["options_new"], pr["answer_kind"], pr["mapping"]))
    present = [k for k in res if res[k]]
    complete = [k for k in res if len(res[k]) == len(items)]
    common = [q for q in items if all(q in res[k] for k in present)]
    n = len(common)
    accs = {k: sum(res[k][q][0] for q in common) / n for k in present} if n else {}
    runs = [dict(benchmark=bench, permutation_id=k, acc="" if k not in accs else f"{accs[k]:.4f}",
                 n_scored=n if k in present else 0, n_rows=len(res[k])) for k in res]
    out = dict(benchmark=bench, n=n, n_planned=len(items), n_perms_present=len(present), n_perms_complete=len(complete),
               status="complete" if len(complete) == n_perms + 1 else "partial")
    if n and accs:
        ks = [plan_rows[(q, 0)]["n_options"] for q in common]
        c = sum(1.0 / k for k in ks) / n
        width, thr = max(accs.values()) - min(accs.values()), Z * math.sqrt(c * (1 - c) / n)
        n_cons = n_unres = 0
        for q in common:
            ids = [res[k][q][1] for k in present]
            if any(i is None for i in ids):
                n_unres += 1
            elif len(set(ids)) == 1:
                n_cons += 1
        out.update(k_mean=f"{sum(ks) / n:.4f}", c=f"{c:.4f}", acc_min=f"{min(accs.values()):.4f}",
                   acc_max=f"{max(accs.values()):.4f}", width=f"{width:.4f}", threshold=f"{thr:.4f}",
                   exceeds_threshold=width > thr, item_consistency=f"{n_cons / n:.4f}", n_items_consistent=n_cons,
                   n_items_unresolved=n_unres)
    return runs, out


RUN_COLS = ["benchmark", "permutation_id", "acc", "n_scored", "n_rows"]
BENCH_COLS = ["benchmark", "input", "model", "n", "n_planned", "k_mean", "c", "acc_min", "acc_max", "width", "threshold",
              "exceeds_threshold", "item_consistency", "n_items_consistent", "n_items_unresolved", "n_perms_present",
              "n_perms_complete", "status"]


def _merge(path, cols, rows, benches):
    old = [r for r in csv.DictReader(open(path)) if r["benchmark"] not in benches] if os.path.exists(path) else []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(old + rows, key=lambda r: (r["benchmark"], int(r.get("permutation_id") or 0))))


def run(ctx, benchmarks=None, limit=None):
    mode = ctx.get("permutation_input", "frames")
    n_perms, seed, k_frames = int(ctx.get("permutations", 8)), int(ctx.get("seed", 42)), int(ctx.get("reference_frames", 32))
    model, label = ctx.models[ROLE], ctx.models.label(ROLE)
    letter_opts = ctx.get("letter_options", True)
    all_runs, all_stats, benches = [], [], set()
    for bench in select(ctx, benchmarks):
        samples = load_samples(ctx, bench, limit)
        removed = removed_items(ctx, bench)
        samples = {q: r for q, r in samples.items() if q not in removed}
        if not samples:
            continue
        benches.add(bench)
        fmt = bench_meta(ctx, bench).get("question_format", "")
        plan_rows, skipped = plan(samples, n_perms, seed)
        path = ctx.path("result", condition="options_only_perm", bench=bench, model=label)
        rows = {(str(r["qid"]), int(r["permutation_id"])): r for r in read_jsonl(path)
                if r.get("input", mode) == mode and r.get("pred") is not None}
        if mode == "frames":                     # permutation 0 = the attacker's 32-frame run, when it exists
            ref = ctx.path("result", condition="reference", bench=bench, model=label)
            for r in read_jsonl(ref):
                key = (str(r["qid"]), 0)
                if r.get("role") == ROLE and r.get("pred") is not None and key in plan_rows and key not in rows:
                    rows[key] = r
        todo = {}
        for key, pr in plan_rows.items():
            if key not in rows:
                todo.setdefault(pr["qid"], []).append(pr)

        def work(prs):
            """Every pending permutation of one item; the frames are decoded once."""
            row = samples[prs[0]["qid"]]
            frames, ts, err = [], [], None
            if mode == "frames":
                try:
                    video, meta = video_of(ctx, row)
                    if video is None:
                        raise FileNotFoundError("no normalized video")
                    frames, ts, _ = frame_rules.select("attacker", "uniform", video, meta, k_frames, row["qid"], ctx)
                except Exception as e:  # noqa: BLE001
                    err = f"frames: {type(e).__name__}: {str(e)[:160]}"
            out = []
            for pr in prs:
                base = dict(qid=pr["qid"], permutation_id=pr["permutation_id"], role=ROLE, input=mode, perm=pr["perm"],
                            options_new=pr["options_new"], answer_orig=pr["answer_orig"], answer_new=pr["answer_new"],
                            mapping=pr["mapping"], answer=pr["answer_new"])
                if err:
                    out.append(dict(base, pred=None, correct=None, frames=0, fallback=0, error=err))
                    continue
                opts = fmt_opts(pr["options_new"], letter_opts)
                if mode == "frames":
                    res = ask_frames(model, lambda n, o=opts: prompts.VISUAL.format(n=n, q=row.get("question", ""), opts=o),
                                     frames, ts)
                else:
                    res = ask(model, prompts.ATTACKER_OPTIONS_ONLY.format(opts=opts))
                pred = res["pred"]
                sc = None if pred is None else score(pred, pr["answer_new"], pr["options_new"], fmt)
                out.append(dict(base, pred=None if pred is None else str(pred)[:PRED_CHARS],
                                correct=None if sc is None else bool(sc), frames=res["frames"], fallback=res["fallback"],
                                error=res["error"]))
            return out

        if todo:
            with JsonlWriter(path, key=("qid", "permutation_id")) as w:
                for outs in pmap(work, list(todo.values()), workers=int(ctx.get("workers", 8)),
                                 desc=f"options_only_perm {bench}"):
                    for out in outs:
                        w.write(out)
                        if out.get("pred") is not None:
                            rows[(str(out["qid"]), int(out["permutation_id"]))] = out
        runs, st = bench_stats(bench, plan_rows, rows, fmt, n_perms)
        st.update(input=mode, model=label)
        all_runs += runs
        all_stats.append(st)
        log(f"options_only_perm [{mode}, {label}] {bench}: {st['n']} common items of {st['n_planned']} planned, "
            f"width {st.get('width', 'n/a')}, threshold {st.get('threshold', 'n/a')}, {st['status']}"
            + (f"; left out: {skipped}" if skipped else ""), "audit")
    _merge(ctx.path("table", name="permutation_runs.csv"), RUN_COLS, all_runs, benches)
    _merge(ctx.path("table", name="permutation_by_benchmark.csv"), BENCH_COLS, all_stats, benches)
    return all_stats
