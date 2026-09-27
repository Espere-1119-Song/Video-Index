"""Item records of the composition (the format of the candidate, verified and released lists).

    item_id, benchmark, question, options (texts without letter prefix), answer_idx, video_id, chance (1 / number of
    options), attack_pct, task_class, capability
"""
from __future__ import annotations

import json
import re

from ..data.schema import read_jsonl
from ..pool.schema import LETTERS, render_option

_PREFIX = re.compile(r"^\s*\(?([A-Za-z])[\.\):]\s*(.*)$", re.S)


def strip_prefix(opts):
    """['A. x', 'B. y'] -> ['x', 'y'] when every option carries the letter of its position."""
    out = []
    for i, o in enumerate(opts):
        m = _PREFIX.match(str(o))
        if not m or m.group(1).upper() != LETTERS[i]:
            return [str(o).strip() for o in opts]
        out.append(m.group(2).strip())
    return out


def clean_stem(q):
    q = re.sub(r"^\s*Question\s*:\s*", "", str(q or "").strip(), flags=re.I)
    return re.sub(r"\s*(Options|Choices|Answer choices)\s*:?\s*$", "", q, flags=re.I).strip()


def from_pool_row(r, capability=None):
    """r: one row of the pool table (mapping) with attack_pct."""
    opts = [render_option(x) for x in list(r["options"])]
    return dict(item_id=r["item_id"], benchmark=r["benchmark"], question=clean_stem(r["question"]),
                options=strip_prefix(opts), answer_idx=int(r["answer_idx"]), video_id=r["video_id"],
                chance=1.0 / len(opts), attack_pct=round(float(r["attack_pct"]), 4), task_class=r.get("task_class"),
                capability=capability)


def from_samples(path, benchmark=None):
    """Items of a sample file (samples/<benchmark>.jsonl) in the record format above; items without options or
    without an answer that names an option are dropped. Used to label or verify items that did not come through
    the pool."""
    items = []
    for r in read_jsonl(path):
        opts = r.get("options") or []
        if not opts or not str(r.get("answer", "")).strip():
            continue
        pref = [re.match(r"^\(?([A-Z])[\.\):]\s*(.*)$", str(o), re.S) for o in opts]
        if all(pref):
            letters, opts = [m.group(1) for m in pref], [m.group(2).strip() for m in pref]
        else:
            letters, opts = list(LETTERS[:len(opts)]), [str(o) for o in opts]
        ans = str(r["answer"]).strip()
        if ans in letters:
            idx = letters.index(ans)
        elif ans in opts:
            idx = opts.index(ans)
        else:
            m = re.match(r"^\(?([A-Z])[\.\):]?", ans)
            if not (m and m.group(1) in letters):
                continue
            idx = letters.index(m.group(1))
        items.append(dict(item_id=str(r["qid"]), benchmark=benchmark or r.get("benchmark"), question=r["question"],
                          options=opts, answer_idx=idx, video_id=str(r.get("video_id") or ""),
                          chance=1.0 / len(opts), task_class=r.get("subtask") or r.get("subcategory") or ""))
    return items


def load(path):
    with open(path) as f:
        return json.load(f)


def save(path, items):
    with open(path, "w") as f:
        json.dump(items, f, ensure_ascii=False)
