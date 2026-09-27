"""Item table of the pool: option handling, the multiple-choice rule, and table files.

Columns of pool/pool_items_base: benchmark, item_id, video_id, video_path, question, question_raw, options (texts
without the letter prefix), options_source (field | inline_question), option_prefix_stripped, n_options, answer,
answer_idx (-1 when the answer is not an option), duration, declared_task, declared_scene, license,
format (mcq | yesno | open), format_reason.

format = open when: fewer than two options; more than 26 options; a multi-select answer ("A,B"); an answer that
resolves to no option; a letter beyond the option count. format = yesno when the options are a bare yes/no
(true/false) pair. Only format = mcq items enter the screen.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

from ..scoring import _MULTI, _MULTI_X, _pre, _wide, coerce_options, mcq_letter

MAX_OPTIONS = 26
LETTERS = [chr(65 + i) for i in range(MAX_OPTIONS)]
_BARE_LETTER = re.compile(r"^\s*[\(\[]?([A-Za-z])[\)\]\.]?\s*$")
_SEQ_PREFIX = re.compile(r"^\s*[\(\[]?([A-Za-z])[\)\]\.:]\s*(.*)$", re.S)
_YESNO = {"yes", "no", "true", "false"}
# markers of options inside the question text: at a line start "A. x" / "A) x" / "(A) x" / "A: x"; after white space
# only with the word Option or Caption ("Option A: x")
_INLINE = re.compile(r"(?:(?:(?<=^)|(?<=\n))\s*(?:(?:Option|Caption)\s+)?|(?<=\s)(?:Option|Caption)\s+)"
                     r"(?:\(([A-J])\)|([A-J])[\.\):])\s+", re.S | re.I)
_TRAIL_INSTR = re.compile(r"\n\s*(answer|please|respond|only |output|reply|select|choose)", re.I)
ORDERING_RE = r"select the correct order of the following options"

BASE_COLS = ["benchmark", "item_id", "video_id", "video_path", "question", "question_raw", "options",
             "options_source", "option_prefix_stripped", "n_options", "answer", "answer_idx", "duration",
             "declared_task", "declared_scene", "license", "format", "format_reason"]


def strip_sequential_prefix(opts):
    """['A. x', 'B. y'] -> (['x', 'y'], True) when every option carries the letter of its position."""
    texts = []
    for i, o in enumerate(opts):
        m = _SEQ_PREFIX.match(str(o))
        if not m or m.group(1).upper() != chr(65 + i):
            return [str(o) for o in opts], False
        texts.append(m.group(2).strip())
    return texts, True


def render_options(texts):
    return "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(texts))


def item_text(question, options):
    """Text embedded for the near-duplicate checks: question and options."""
    q = " ".join(str(question or "").split())
    if isinstance(options, (list, tuple)) and len(options):
        return q + "\n" + "\n".join(str(o) for o in options)
    return q


def yes_no_options(opts):
    """True when the option texts are only yes/no or true/false, e.g. ['A. Yes', 'B. No']."""
    texts = {re.sub(r"^\s*[\(\[]?[A-Za-z][\)\]\.:]\s*", "", str(o)).strip().lower().rstrip(".") for o in opts or []}
    return bool(texts) and texts <= _YESNO


def render_option(x):
    """An option stored as a mapping {'label': 'A', 'text': ...} is reduced to its text."""
    if isinstance(x, dict) and "text" in x:
        return str(x["text"]).strip()
    s = str(x)
    if s.startswith("{") and ("'text'" in s or '"text"' in s):
        try:
            import ast
            d = ast.literal_eval(s) if s.startswith("{'") else json.loads(s)
            if isinstance(d, dict) and "text" in d:
                return str(d["text"]).strip()
        except Exception:  # noqa: BLE001
            pass
    return s


def resolve_answer(answer, options, inline=False):
    """-> (option texts, prefix_stripped, n_options, answer_idx, format, reason); format is mcq or open.
    inline: the options came from the question text; a bare number is then not accepted as an option index."""
    opts = coerce_options(options)
    if isinstance(opts, (list, tuple)):
        opts = [render_option(o) for o in opts]
    if inline and re.fullmatch(r"\s*\d{1,2}\s*", "" if answer is None else str(answer)):
        return [str(o) for o in opts], False, len(opts), -1, "open", "answer_not_in_options"
    if not isinstance(opts, list) or len(opts) < 2:
        return ([str(o) for o in opts] if isinstance(opts, list) else []), False, \
            (len(opts) if isinstance(opts, list) else 0), -1, "open", "no_options"
    texts, stripped = strip_sequential_prefix(opts)
    n = len(opts)
    if n > MAX_OPTIONS:
        return texts, stripped, n, -1, "open", "too_many_options"
    a = "" if answer is None else str(answer)
    if (_MULTI_X if _wide(opts) else _MULTI).match(_pre(a)):
        return texts, stripped, n, -1, "open", "multi_select"
    idx = None
    m = _BARE_LETTER.match(a)
    if m and 0 <= ord(m.group(1).upper()) - 65 < n:
        idx = ord(m.group(1).upper()) - 65
    if idx is None:
        letter = mcq_letter(a, opts)
        if letter is None:
            na = " ".join(a.lower().split())                  # the answer may be the option text itself
            hits = [i for i, t in enumerate(texts) if na and " ".join(t.lower().split()) == na]
            if len(hits) != 1:
                return texts, stripped, n, -1, "open", "answer_not_in_options"
            idx = hits[0]
        else:
            idx = ord(letter) - 65
            if idx >= n:
                return texts, stripped, n, -1, "open", "letter_out_of_range"
    return texts, stripped, n, idx, "mcq", ""


def split_inline_options(question):
    """Options written inside the question text -> (stem, [option texts]) or None. The markers must run A, B, C, ...
    from A, every option text and the stem must be non-empty."""
    q = str(question or "")
    ms = list(_INLINE.finditer(q))
    if len(ms) < 2:
        return None
    seq, want = [], "A"
    for m in ms:
        letter = (m.group(1) or m.group(2)).upper()
        if letter == want and (not seq or m.start() >= seq[-1].end()):
            seq.append(m)
            want = chr(ord(want) + 1)
    if len(seq) < 2:
        return None
    stem = q[:seq[0].start()].strip()
    if not stem:
        return None
    opts = []
    for i, m in enumerate(seq):
        end = seq[i + 1].start() if i + 1 < len(seq) else len(q)
        t = q[m.end():end].strip()
        if i + 1 == len(seq):                                 # instruction after the last option
            mt = _TRAIL_INSTR.search(t)
            if mt:
                t = t[:mt.start()].strip()
        if not t:
            return None
        opts.append(t)
    return stem, opts


def video_id_of(item):
    """Content hash of the normalized video; before the video is normalized "k:" + hash of its source reference,
    so that items of one source video still share an id."""
    if item.get("video_id"):
        return str(item["video_id"])
    ref = item.get("video_ref")
    if not ref:
        return ""
    key = json.dumps(ref, sort_keys=True, default=str) if isinstance(ref, dict) else str(ref)
    return "k:" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def item_row(item, license="", video_path=None):
    """One item of items/<benchmark>.jsonl -> one row of the item table."""
    question = str(item.get("question") if item.get("question") is not None else "")
    field = coerce_options(item.get("options"))
    source, question_raw = "", None
    if isinstance(field, (list, tuple)) and len(field) >= 2:
        source = "field"
        texts, stripped, n, aidx, fmt, reason = resolve_answer(item.get("answer"), field)
    else:
        texts, stripped, n, aidx, fmt, reason = [], False, 0, -1, "open", "no_options"
        got = split_inline_options(question)
        if got:
            res = resolve_answer(item.get("answer"), got[1], inline=True)
            if res[4] == "mcq":
                question_raw, question, source = question, got[0], "inline_question"
                texts, stripped, n, aidx, fmt, reason = res
    if fmt == "mcq" and yes_no_options(texts):
        fmt, reason = "yesno", "yes_no_options"
    dur = item.get("duration_s")
    return dict(benchmark=item["benchmark"], item_id=str(item["qid"]), video_id=video_id_of(item),
                video_path=video_path, question=question, question_raw=question_raw, options=list(texts),
                options_source=source, option_prefix_stripped=bool(stripped), n_options=int(n),
                answer="" if item.get("answer") is None else str(item.get("answer")), answer_idx=int(aidx),
                duration=float(dur) if dur is not None else None,
                declared_task=str(item.get("subtask") or item.get("subcategory") or "_none"),
                declared_scene=str(item.get("scene") or ""), license=item.get("license") or license or "",
                format=fmt, format_reason=reason)


def is_mcq_item(item):
    """Multiple-choice rule of the pool: an option list (or options inside the question text) with at least two
    and at most 26 entries, a single answer that resolves to one of them, and options that are not a bare yes/no
    pair. Yes/no and open items never enter the screen."""
    return item_row(dict(benchmark="", qid="", **{k: item.get(k) for k in ("question", "options", "answer")})
                    )["format"] == "mcq"


# ------------------------------------------------------------------ table files
def table_path(ctx, name):
    """Path of a pool table without extension -> the existing file, else the preferred format."""
    stem = ctx.path("pool", name=name)
    for ext in (".parquet", ".jsonl"):
        if os.path.exists(stem + ext):
            return stem + ext
    try:
        import pyarrow  # noqa: F401
        return stem + ".parquet"
    except ImportError:
        return stem + ".jsonl"


def write_table(ctx, name, df):
    """Parquet when pyarrow is installed, JSON Lines otherwise."""
    stem = ctx.path("pool", name=name)
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(stem + ".parquet.tmp", index=False)
        os.replace(stem + ".parquet.tmp", stem + ".parquet")
        return stem + ".parquet"
    except ImportError:
        df.to_json(stem + ".jsonl.tmp", orient="records", lines=True, force_ascii=False)
        os.replace(stem + ".jsonl.tmp", stem + ".jsonl")
        return stem + ".jsonl"


def read_table(ctx, name, columns=None):
    import pandas as pd
    p = table_path(ctx, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"pool table '{name}' not found below {os.path.dirname(p)}")
    if p.endswith(".parquet"):
        df = pd.read_parquet(p, columns=columns)
    else:
        df = pd.read_json(p, orient="records", lines=True, dtype={"item_id": str, "video_id": str})
        df = df[columns] if columns else df
    if "options" in df:
        df["options"] = df["options"].map(lambda o: [] if o is None else [str(x) for x in list(o)])
    return df
