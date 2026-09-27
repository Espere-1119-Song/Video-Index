"""From a raw annotation record to an item of the shared schema.

A *raw record* is what an adapter yields for one question:

    v     video reference: a path / file name / id, or a pinned ``(archive, member)`` pair
    q     question text
    a     gold answer as the source gives it (letter, index, option text, free text)
    opts  options as the source gives them (list, dict, string, or None)
    sub   the benchmark's own category

:func:`to_item` resolves the record into the schema of :mod:`video_index.data.schema` and decides its format:

    mcq    an option list with at least two entries and a gold that resolves to one option
    yesno  the options are a bare yes/no (true/false) pair, or there are no options and the gold is yes/no
    open   everything else (free text, numbers, intervals, multi-select, unresolved gold)

Only ``mcq`` items enter the audit.
"""
from __future__ import annotations

import re
from typing import Any

from ...scoring import _MULTI, _MULTI_X, _pre, _wide, coerce_options, mcq_letter

MAX_OPTIONS = 26
LETTERS = [chr(65 + i) for i in range(MAX_OPTIONS)]
_SEQ_PREFIX = re.compile(r"^\s*[\(\[]?([A-Za-z])[\)\]\.:]\s*(.*)$", re.S)
_BARE_LETTER = re.compile(r"^\s*[\(\[]?([A-Za-z])[\)\]\.]?\s*$")
_YN = {"yes", "no", "true", "false"}
# option markers inside the question text: at a line start "A. x" / "A) x" / "(A) x" / "A: x"; after whitespace only
# together with the word Option / Caption ("Option A: x"). A bare "(A)" inside prose is not a marker.
_INLINE = re.compile(r"(?:(?:(?<=^)|(?<=\n))\s*(?:(?:Option|Caption)\s+)?|(?<=\s)(?:Option|Caption)\s+)"
                     r"(?:\(([A-J])\)|([A-J])[\.\):])\s+", re.S | re.I)
_TRAIL_INSTR = re.compile(r"\n\s*(answer|please|respond|only |output|reply|select|choose)", re.I)


def strip_sequential_prefix(opts: list) -> tuple[list[str], bool]:
    """['A. x', 'B. y'] -> (['x', 'y'], True) when every option carries the letter of its position."""
    texts = []
    for i, o in enumerate(opts):
        m = _SEQ_PREFIX.match(str(o))
        if not m or m.group(1).upper() != chr(65 + i):
            return [str(o) for o in opts], False
        texts.append(m.group(2).strip())
    return texts, True


def options_from_dict(d: dict) -> list[str] | None:
    """Dict-typed options -> list of option texts in option order.

    Keys that are option letters ({'A': .., 'B': ..}) are read in key order; other keys (option_0, choice_a, 1, 2)
    are ordered by their trailing number or letter. Entries whose value is None or empty are placeholders and are
    dropped; a dict of placeholders means no options. (Taking ``list(d)`` keeps only the keys: the defect that
    left several samples with the bare letters as options.)"""
    items = [(str(k), v) for k, v in d.items() if v is not None and str(v).strip() != ""]
    if not items:
        return None
    if all(len(k.strip()) == 1 and k.strip().isalpha() for k, _ in items):
        return [str(v).strip() for _, v in items]

    def order(kv):
        tail = re.split(r"[_\-\s\.]", kv[0])[-1]
        if tail.isdigit():
            return (0, int(tail), kv[0])
        if len(tail) == 1 and tail.isalpha():
            return (0, ord(tail.lower()) - 97, kv[0])
        return (1, 0, kv[0])
    return [str(v).strip() for _, v in sorted(items, key=order)]


def normalize_options(opts: Any) -> list[str] | None:
    """Options as given by a source -> list of strings (prefixes kept), or None."""
    if opts is None:
        return None
    if isinstance(opts, dict):
        return options_from_dict(opts)
    if not isinstance(opts, (list, str)):
        try:
            opts = list(opts)                       # numpy arrays, tuples
        except TypeError:
            opts = str(opts)
    got = coerce_options(opts)
    if isinstance(got, dict):
        return options_from_dict(got)
    if isinstance(got, list):
        out = [str(o) for o in got if o is not None]
        return out or None
    return None


def yes_no_options(opts: list | None) -> bool:
    """The option texts are only yes/no (or true/false), e.g. ['A. Yes', 'B. No']."""
    texts = {re.sub(r"^\s*[\(\[]?[A-Za-z][\)\]\.:]\s*", "", str(o)).strip().lower().rstrip(".") for o in opts or []}
    return bool(texts) and texts <= _YN


def is_yes_no(answer: Any) -> bool:
    return str(answer if answer is not None else "").strip().lower().rstrip(".") in _YN


def split_inline_options(question: Any) -> tuple[str, list[str]] | None:
    """Options embedded in the question text -> (stem, option texts), or None.

    Conservative: the markers run A, B, C, ... from A, every option text is non-empty, and a non-empty stem precedes
    the first marker. A trailing instruction after the last option ("Answer with ...") is cut off."""
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
        if i + 1 == len(seq):
            mt = _TRAIL_INSTR.search(t)
            if mt:
                t = t[:mt.start()].strip()
        if not t:
            return None
        opts.append(t)
    return stem, opts


def resolve_answer(answer: Any, options: list[str] | None, inline: bool = False):
    """-> (option_texts, n_options, answer_idx, format, reason).

    The gold may be a letter ("B", "(B)", "B."), a letter with text ("B. dog"), the option text, or a numeric index
    (accepted only when the options come from an options field, not from the question text)."""
    opts = options
    a = "" if answer is None else str(answer)
    if inline and re.fullmatch(r"\s*\d{1,2}\s*", a):
        return list(opts or []), len(opts or []), -1, "open", "answer_not_in_options"
    if not isinstance(opts, list) or len(opts) < 2:
        return list(opts or []), len(opts or []), -1, "open", "no_options"
    texts, _ = strip_sequential_prefix(opts)
    n = len(opts)
    if n > MAX_OPTIONS:
        return texts, n, -1, "open", "too_many_options"
    if (_MULTI_X if _wide(opts) else _MULTI).match(_pre(a)):
        return texts, n, -1, "open", "multi_select"
    idx = None
    m = _BARE_LETTER.match(a)
    if m:
        i = ord(m.group(1).upper()) - 65
        if 0 <= i < n:
            idx = i
    if idx is None:
        letter = mcq_letter(a, opts)
        if letter is None:
            na = " ".join(a.lower().split())
            hits = [i for i, t in enumerate(texts) if na and " ".join(t.lower().split()) == na]
            if len(hits) != 1:
                return texts, n, -1, "open", "answer_not_in_options"
            idx = hits[0]
        else:
            idx = ord(letter) - 65
            if idx >= n:
                return texts, n, -1, "open", "letter_out_of_range"
    return texts, n, idx, "mcq", ""


def to_item(benchmark: str, index: int, raw: dict) -> dict:
    """Raw record (v, q, a, opts, sub) -> item of the shared schema, with its format decided."""
    question = "" if raw.get("q") is None else str(raw["q"])[:2000]
    answer_raw = "" if raw.get("a") is None else str(raw["a"])[:2000]
    field_opts = normalize_options(raw.get("opts"))
    source = ""
    if isinstance(field_opts, list) and len(field_opts) >= 2:
        source = "field"
        texts, n, idx, fmt, reason = resolve_answer(answer_raw, field_opts)
    else:
        texts, n, idx, fmt, reason = [], 0, -1, "open", "no_options"
        got = split_inline_options(question)
        if got:
            t2, n2, i2, f2, r2 = resolve_answer(answer_raw, got[1], inline=True)
            if f2 == "mcq":
                question, source = got[0], "question"
                texts, n, idx, fmt, reason = t2, n2, i2, f2, r2
    if fmt == "mcq" and yes_no_options(texts):
        fmt, reason = "yesno", "yes_no_options"
    elif fmt == "open" and not texts and is_yes_no(answer_raw):
        fmt, reason = "yesno", "yes_no_answer"
    texts = [str(t).strip() for t in texts]
    sub = raw.get("sub")
    return dict(qid=f"{benchmark}_{index}", benchmark=benchmark, question=question,
                options=texts if len(texts) >= 2 else None,
                answer=LETTERS[idx] if fmt == "mcq" else answer_raw, answer_idx=idx, answer_raw=answer_raw,
                subtask=str(sub) if sub not in (None, "") else "_none", video_ref=raw.get("v"),
                format=fmt, format_reason=reason, options_source=source, n_options=n)
