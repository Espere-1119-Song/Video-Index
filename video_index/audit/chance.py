"""Chance level: one definition for every table.

item_chance(row, meta) -> (c, source), in this precedence:
    list      1/k, k = length of the item's option list (lists with k >= 2)
    question  1/k, k = number of lettered option lines embedded in the question text
    declared  1/num_options of the benchmark metadata, only for benchmarks with question_format MCQ;
              several integers ("4 or 6") give the mean of 1/n
    inferred  1/k, k inferred from the letter-form golds of the items without an option list: the smallest number of
              leading positions that holds >= 95 % of those golds, at least 2
    yesno     0.5 for golds yes / no / true / false
    open      0 (free text, numbers, intervals)
Every statistic uses the multiple-choice items only (is_mcq_item): sources list / question / declared / inferred, and
options that are not a bare yes/no pair. The benchmark chance c is the mean item chance over those items.

Stage output: tables/chance.csv (one row per benchmark).
"""
from __future__ import annotations

import csv
import math
import re
from collections import Counter

from ..scoring import _MULTI, _pre, mcq_letter
from .items import bench_meta, item_options, load_samples, select

SOURCES = ["list", "question", "declared", "inferred", "yesno", "open"]
_LETTER_GOLD = re.compile(r"^\s*[\(\[]?([A-P])(?:[\)\]\.:]\s*(?:\S.*)?|\s*)$")
_YESNO = {"yes", "no", "true", "false"}
INFER_SHARE = 0.95
MCQ_SOURCES = ("list", "question", "declared", "inferred")


def declared_chance(m):
    if (m.get("question_format") or "").strip().upper() != "MCQ":
        return None
    ns = [int(x) for x in re.findall(r"\d+", str(m.get("num_options") or "")) if int(x) > 1]
    if not ns:
        return None
    return sum(1.0 / n for n in ns) / len(ns)


def declared_counts(m):
    """Sorted integers of num_options (any format), for the option-count check."""
    return sorted({int(x) for x in re.findall(r"\d+", str(m.get("num_options") or "")) if int(x) > 1})


def is_yesno(ans):
    return str(ans if ans is not None else "").strip().lower() in _YESNO


def gold_position(row, opts, fmt, digit_index=False):
    """Letter of the gold option, or None (multi-select, open, unresolvable).
    digit_index: the benchmark stores the gold as a bare option index."""
    ans = row.get("answer")
    if ans is None:
        return None
    if _MULTI.match(_pre(ans)):
        return None
    if opts:
        return mcq_letter(ans, opts)
    fmt = (fmt or "").strip().upper()
    if fmt not in ("MCQ", "MIXED"):
        return None
    m = _LETTER_GOLD.match(_pre(ans))
    if m:
        return m.group(1).upper()
    if fmt == "MCQ":
        L = mcq_letter(ans)
        if L:
            return L
        s = str(ans).strip()
        if digit_index and re.fullmatch(r"\d", s):
            return chr(65 + int(s))
    return None


def item_chance(row, m, k_inferred=None, digit_index=False):
    opts, src = item_options(row)
    if opts:
        return 1.0 / len(opts), src
    dec = declared_chance(m)
    if dec:
        return dec, "declared"
    if k_inferred and gold_position(row, None, m.get("question_format", ""), digit_index):
        return 1.0 / k_inferred, "inferred"
    if is_yesno(row.get("answer")):
        return 0.5, "yesno"
    return 0.0, "open"


def infer_k(samples, m, opts_of=None):
    """(k_inferred, pos, digit_index); pos maps every item to its gold letter when resolvable."""
    fmt = m.get("question_format", "")
    is_mcq = (fmt or "").strip().upper() == "MCQ"
    if opts_of is None:
        opts_of = {qid: item_options(r)[0] for qid, r in samples.items()}
    unc_golds = [str(r.get("answer", "")).strip() for qid, r in samples.items() if not opts_of[qid]]
    digit_index = bool(is_mcq and unc_golds and all(re.fullmatch(r"\d", g) for g in unc_golds if g))
    pos = {}
    for qid, r in samples.items():
        L = gold_position(r, opts_of[qid], fmt, digit_index)
        if L:
            pos[qid] = L
    unc_pos = sorted(ord(L) - 64 for q, L in pos.items() if not opts_of[q])
    k_inf = 0
    if unc_pos:
        k_inf = max(2, unc_pos[min(len(unc_pos) - 1, int(math.ceil(INFER_SHARE * len(unc_pos))) - 1)])
    return k_inf, pos, digit_index


def _yes_no_options(opts):
    texts = set()
    for o in opts or []:
        t = re.sub(r"^\s*[\(\[]?[A-Za-z][\)\]\.:]\s*", "", str(o)).strip().lower().rstrip(".")
        texts.add(t)
    return bool(texts) and texts <= _YESNO


def is_mcq_item(row, m, k_inferred=None, digit_index=False):
    c, src = item_chance(row, m, k_inferred, digit_index)
    if src not in MCQ_SOURCES or c <= 0:
        return False
    opts, _ = item_options(row)
    if opts and _yes_no_options(opts):
        return False
    if not opts and is_yesno(row.get("answer")):
        return False
    return True


def mcq_items(samples, m):
    m = m or {}
    opts_of = {qid: item_options(r)[0] for qid, r in samples.items()}
    k_inf, _, digit_index = infer_k(samples, m, opts_of)
    return {qid: r for qid, r in samples.items() if is_mcq_item(r, m, k_inf, digit_index)}


def source_string(csrc):
    return ";".join(f"{k}:{v}" for k, v in sorted(csrc.items()))


def bench_chance(samples, m):
    """samples = {qid: row}; m = benchmark metadata or {}. Returns c (multiple-choice items only), cvec {qid: c_item},
    csrc, c_source, k_inf, n_items, n_mcq, n_excluded, mcq_qids, n_open, n_yesno, digit_index, pos {qid: gold letter},
    c_all (mean over every item)."""
    m = m or {}
    opts_of = {qid: item_options(r)[0] for qid, r in samples.items()}
    k_inf, pos, digit_index = infer_k(samples, m, opts_of)
    cvec, csrc, mcq = {}, Counter(), []
    for qid, r in samples.items():
        c, src = item_chance(r, m, k_inf, digit_index)
        cvec[qid], csrc[src] = c, csrc[src] + 1
        if is_mcq_item(r, m, k_inf, digit_index):
            mcq.append(qid)
    n = len(cvec)
    c_mcq = (sum(cvec[q] for q in mcq) / len(mcq)) if mcq else None
    return dict(c=c_mcq, cvec=cvec, csrc=csrc, c_source=source_string(csrc), k_inf=k_inf, n_items=n,
                n_mcq=len(mcq), n_excluded=n - len(mcq), mcq_qids=mcq,
                n_open=csrc.get("open", 0), n_yesno=csrc.get("yesno", 0), digit_index=digit_index, pos=pos,
                c_all=(sum(cvec.values()) / n) if n else None)


COLS = ["benchmark", "n_items", "n_mcq", "n_excluded", "n_open", "n_yesno", "c", "c_all", "c_source", "k_inferred",
        "declared_num_options"]


def run(ctx, benchmarks=None, limit=None):
    rows = []
    for bench in select(ctx, benchmarks):
        samples = load_samples(ctx, bench, limit)
        if not samples:
            continue
        m = bench_meta(ctx, bench)
        d = bench_chance(samples, m)
        rows.append(dict(benchmark=bench, n_items=d["n_items"], n_mcq=d["n_mcq"], n_excluded=d["n_excluded"],
                         n_open=d["n_open"], n_yesno=d["n_yesno"], c="" if d["c"] is None else f"{d['c']:.4f}",
                         c_all="" if d["c_all"] is None else f"{d['c_all']:.4f}", c_source=d["c_source"],
                         k_inferred=d["k_inf"], declared_num_options=m.get("num_options", "")))
    out = ctx.path("table", name="chance.csv")
    old = {}
    try:
        old = {r["benchmark"]: r for r in csv.DictReader(open(out))}
    except FileNotFoundError:
        pass
    for r in rows:
        old[r["benchmark"]] = r
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(old[k] for k in sorted(old))
    return rows
