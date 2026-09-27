"""Item helpers shared by the stages: option lists, prompts text, samples, videos, benchmark metadata."""
from __future__ import annotations

import csv
import json
import os
import re

from ..data.schema import read_jsonl
from ..scoring import coerce_options, score

# lettered option line: "(A) text" / "A. text" / "A: text" / "A) text" / "[A] text"
_OPT_LINE = re.compile(r"^\s*[\(\[]?([A-P])[\)\]\.:]\s*(\S.*)?$")
# inline forms: "... Option A: text Option B: text" and "... A. text B. text C. text."
_INLINE_OPTION_WORD = re.compile(r"\bOption\s+([A-P])\s*[:\.]\s*")
_INLINE_LETTER = re.compile(r"(?:^|(?<=\s))\(?([A-P])[\.\)]\s+(?=\S)")


def _inline_options(text):
    """Options written inline in one line; letters must run consecutively from A, >= 2 options, else None."""
    for rx in (_INLINE_OPTION_WORD, _INLINE_LETTER):
        ms = list(rx.finditer(text))
        letters = [m.group(1) for m in ms]
        if "A" not in letters:
            continue
        i0 = letters.index("A")
        run = [ms[i0]]
        for m in ms[i0 + 1:]:
            if m.group(1) == chr(65 + len(run)):
                run.append(m)
            else:
                break
        if len(run) < 2:
            continue
        if rx is _INLINE_LETTER:
            # bare letters need a question mark or colon right before the first marker
            if not text[:run[0].start()].rstrip().endswith(("?", ":")):
                continue
        out = []
        for j, m in enumerate(run):
            end = run[j + 1].start() if j + 1 < len(run) else len(text)
            body = text[m.end():end].strip().rstrip(".;,")
            if not body:
                break
            out.append(f"{m.group(1)}. {body}")
        if len(out) == len(run):
            return out
    return None


def embedded_options(question):
    """Option lines embedded in the question text, in order, or None (>= 2 lines, letters consecutive from A)."""
    out = []
    for line in str(question or "").splitlines():
        m = _OPT_LINE.match(line)
        if m and m.group(1) == chr(65 + len(out)):
            out.append(line.strip())
        elif m and out:
            break
    if len(out) >= 2:
        return out
    return _inline_options(str(question or ""))


def item_options(row):
    """(option list, source): source "list" (the options field) or "question" (embedded); (None, None) without options."""
    opts = coerce_options(row.get("options"))
    if opts and len(opts) >= 2:
        return opts, "list"
    emb = embedded_options(row.get("question"))
    if emb:
        return emb, "question"
    return None, None


def lettered(opts):
    """One option per line with a letter prefix; existing prefixes are kept."""
    if all(_OPT_LINE.match(str(o)) for o in opts):
        return "\n".join(str(o).strip() for o in opts)
    return "\n".join(f"{chr(65 + i)}. {str(o).strip()}" for i, o in enumerate(opts))


def fmt_opts(o, letter=False):
    """Options as they appear in the question prompts: the field as stored, one entry per line. letter=True adds
    "A. ", "B. ", ... to a list whose entries carry no letter prefix (the item schema stores options without them)."""
    if not o:
        return ""
    if isinstance(o, str):
        return o
    if letter:
        return lettered(o)
    return "\n".join(str(x) for x in o)


def subtask(row):
    return row.get("subtask") or row.get("subcategory") or ""


def item_text(question, options):
    """Text embedded for the near-duplicate check and the learned attacker: question + options."""
    q = " ".join(str(question or "").split())
    if isinstance(options, list) and options:
        return q + "\n" + "\n".join(str(o) for o in options)
    return q


# ------------------------------------------------------------------ samples and results
def load_samples(ctx, bench, limit=None):
    """{qid: row} of samples/<bench>.jsonl, in file order."""
    rows = read_jsonl(ctx.path("samples", bench=bench))
    if limit is not None:
        rows = rows[:limit]
    return {str(r["qid"]): r for r in rows}


def read_results(path, role=None):
    """{qid: row}; the last row per qid wins; role filters rows that carry a role field."""
    out = {}
    for r in read_jsonl(path):
        if "qid" not in r:
            continue
        if role is not None and r.get("role", role) != role:
            continue
        out[str(r["qid"])] = r
    return out


def score_results(path, samples, fmt, role=None):
    """Result file -> {qid: bool}. Gold = the row's answer, options = the sample's list (else the options embedded in
    the question); rows without a prediction and rows the rule scorer cannot score are dropped."""
    out = {}
    for qid, r in read_results(path, role).items():
        srow = samples.get(qid)
        if srow is None or r.get("pred") is None:
            continue
        sc = score(r.get("pred"), r.get("answer", srow.get("answer")), item_options(srow)[0], fmt)
        if sc is not None:
            out[qid] = bool(sc)
    return out


def result_files(ctx, condition, bench):
    """{model label: path} of the result files of one condition and benchmark."""
    d = os.path.join(ctx.work_dir, "results", condition)
    out = {}
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.startswith(bench + "__") and f.endswith(".jsonl"):
                out[f[len(bench) + 2:-6]] = os.path.join(d, f)
    return out


def model_tag(name):
    return str(name).replace("/", "_")


# ------------------------------------------------------------------ videos
def video_key(row):
    """Stable key of the source video of an item (string form of video_ref)."""
    ref = row.get("video_ref")
    if isinstance(ref, dict):
        return f"{ref.get('repo', '')}::{ref.get('zip_path', '') or ''}::{ref.get('member') or ref.get('path') or ''}"
    return str(ref or "")


def video_of(ctx, row):
    """(video path, meta dict) of the normalized video of an item, or (None, None)."""
    vid = row.get("video_id")
    if not vid:
        return None, None
    vp, mp = ctx.path("video", video_id=vid), ctx.path("video_meta", video_id=vid)
    if not os.path.exists(vp):
        return None, None
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    return vp, meta


# ------------------------------------------------------------------ benchmark metadata
def benchmark_table(ctx):
    """{name: metadata}. Sources, in this order: `benchmarks` of the run configuration (list of mappings or mapping
    name -> mapping) and `benchmarks_file` (CSV with a name column). Fields used by the stages: question_format
    (MCQ | mixed | open), num_options (declared), text_heavy, source."""
    if getattr(ctx, "_bench_table", None) is not None:
        return ctx._bench_table
    out = {}
    path = ctx.get("benchmarks_file")
    if path and os.path.exists(path):
        for r in csv.DictReader(open(path)):
            out[r["name"]] = dict(r)
    b = ctx.get("benchmarks") or []
    if isinstance(b, dict):
        b = [dict(name=k, **(v or {})) for k, v in b.items()]
    for r in b:
        if isinstance(r, str):
            r = dict(name=r)
        out.setdefault(r["name"], {}).update(r)
    ctx._bench_table = out
    return out


def bench_meta(ctx, bench):
    t = benchmark_table(ctx)
    return t.get(bench) or t.get(bench.replace("_", "/")) or {}


def question_format(ctx, bench):
    return bench_meta(ctx, bench).get("question_format", "") or ""


def select(ctx, benchmarks=None, kind="samples"):
    """Benchmarks a stage runs on: the argument, else the configured list, else every file of `kind`."""
    if benchmarks:
        return list(benchmarks)
    names = list(benchmark_table(ctx))
    d = os.path.join(ctx.work_dir, kind)
    have = sorted(f[:-6] for f in os.listdir(d) if f.endswith(".jsonl")) if os.path.isdir(d) else []
    if names:
        return [n for n in names if n in have or n.replace("/", "_") in have] or names
    return have
