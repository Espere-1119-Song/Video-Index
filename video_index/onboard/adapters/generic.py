"""Generic adapter: annotation files -> raw records, driven by column detection and an optional declarative spec.

Without a spec the adapter detects the question / answer / options / video / subtask columns from their names. A spec
(:class:`AdapterSpec`, usually a YAML file under ``configs/adapters/``) names what detection cannot find:

    name: NExT-QA
    files: ["MC/*"]                       # glob(s) on the path relative to the annotation directory
    records: data.*                       # where the records sit inside a nested JSON file (optional)
    fields:
      question: question                  # column name, or a template such as "{caption} Answer with 'start end'."
      answer: answer
      options: null                       # column holding the options (list, dict or string) ...
      option_fields: [a0, a1, a2, a3, a4] # ... or one column per option
      subtask: type
      video: "{video}.mp4"                # column name or template
    answer_type: index0                   # auto | letter | index0 | index1 | text
    filters:                              # rows that fail a filter are dropped before items are numbered
      - {field: metadata.source_dataset, in: [ht100m, coin]}
    keep: mcq                             # items that fail `keep` get no item but keep their number
    dedupe: true                          # one row per (video, question)
    hook: my_adapters.py:load             # Python function for irregular sources (see docs/onboard.md)

Templates: ``{field}``, nested ``{a.b}``, and the converters ``{x|int}``, ``{x|sec}``, ``{x|basename}``, ``{x|stem}``,
``{x|first}``, ``{x|strip:./}``, ``{__file|dir0}`` (``__file`` is the annotation file of the row).
"""
from __future__ import annotations

import fnmatch
import glob
import importlib
import importlib.util
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

VIDEO_EXT = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")
ANNOTATION_EXT = (".json", ".jsonl", ".csv", ".tsv", ".parquet")
VID_COLS = ["video", "video_path", "video_name", "video_id", "videoid", "video_file", "videoname", "vid_name", "vid",
            "video_url", "path", "file_name", "filename", "clip", "media_path", "video_link", "video_idx", "videoidx",
            "visual", "id"]
Q_COLS = ["question", "q", "query", "question_text", "instruction", "prompt", "text", "problem", "input", "caption",
          "sentence", "expression"]
A_COLS = ["answer", "answer_text", "gt", "ground_truth", "label", "correct_answer", "correct_choice", "gt_answer",
          "response", "a", "golden_answer", "timestamp", "segment"]
OPT_COLS = ["options", "candidates", "choices", "option", "multi_choice", "answer_choices", "index2ans"]
SUB_COLS = ["category", "task_type", "question_type", "type", "subtask", "task", "domain", "dimension", "subcategory",
            "sub_category", "question_category", "capability", "skill"]
_MCQ_LETTER = re.compile(r"^[\(\[]?[A-Za-z][\)\]\.]?$")


class AdapterError(RuntimeError):
    """The annotations cannot be turned into items; the message says what is missing."""


# ------------------------------------------------------------------ annotation files
def load_annotations(annotation_dir: str, files: list[str] | None = None, records: str | None = None) -> list[dict]:
    """Every record of every annotation file below ``annotation_dir`` as a dict; ``__file`` holds the relative path."""
    rows: list[dict] = []
    for p in sorted(glob.glob(os.path.join(annotation_dir, "**", "*"), recursive=True)):
        if not os.path.isfile(p) or not p.lower().endswith(ANNOTATION_EXT):
            continue
        rel = os.path.relpath(p, annotation_dir)
        if "/." in "/" + rel:                                   # caches of the download tools
            continue
        if files and not any(_file_match(rel, pat) for pat in files):
            continue
        try:
            rs = _read_file(p, records)
        except Exception as e:  # noqa: BLE001
            raise AdapterError(f"cannot parse {rel}: {type(e).__name__}: {str(e)[:160]}") from e
        for r in rs:
            r["__file"] = rel
        rows.extend(rs)
    return rows


def _file_match(rel: str, pat: str) -> bool:
    return fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(os.path.basename(rel), pat) or pat in rel


def _read_file(path: str, records: str | None) -> list[dict]:
    low = path.lower()
    if low.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        for c in list(df.columns):                              # embedded media columns
            first = df[c].iloc[0] if len(df) else None
            if isinstance(first, (bytes, bytearray)) or (isinstance(first, dict) and "bytes" in first):
                df = df.drop(columns=[c])
        return df.to_dict("records")
    if low.endswith(".jsonl"):
        return [r for r in (json.loads(l) for l in open(path, encoding="utf-8", errors="ignore") if l.strip())
                if isinstance(r, dict)]
    if low.endswith(".json"):
        d = json.load(open(path, encoding="utf-8", errors="ignore"))
        if records:
            return [r for r in _walk(d, records.split(".")) if isinstance(r, dict)]
        if isinstance(d, dict):
            for v in d.values():                                # the first list of records inside the object
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    d = v
                    break
            else:
                d = [d]
        return [r for r in d if isinstance(r, dict)] if isinstance(d, list) else []
    import pandas as pd
    return pd.read_csv(path, sep="\t" if low.endswith(".tsv") else ",", on_bad_lines="skip").to_dict("records")


def _walk(node: Any, parts: list[str], ctx: dict | None = None):
    """Records at a path such as ``data.*`` or ``*.questions.*``; fields of the enclosing objects are inherited
    under their own names (the record's own fields win)."""
    ctx = ctx or {}
    if not parts:
        if isinstance(node, dict):
            yield {**ctx, **node}
        return
    head, rest = parts[0], parts[1:]
    if head == "*":
        if isinstance(node, dict):
            for k, v in node.items():
                yield from _walk(v, rest, {**ctx, "__key": k})
        elif isinstance(node, list):
            for v in node:
                yield from _walk(v, rest, ctx)
    elif isinstance(node, dict) and head in node:
        inherit = {k: v for k, v in node.items() if not isinstance(v, (list, dict))}
        yield from _walk(node[head], rest, {**ctx, **inherit})


def norm_key(k: str) -> str:
    return str(k).lower().replace(" ", "_").replace("-", "_")


def detect(rows: list[dict], candidates: list[str]) -> str | None:
    """The first candidate column name present in the first 50 rows (case, spaces and dashes ignored)."""
    keymap: dict[str, str] = {}
    for r in rows[:50]:
        for k in r.keys():
            keymap.setdefault(norm_key(k), k)
    for c in candidates:
        if norm_key(c) in keymap:
            return keymap[norm_key(c)]
    return None


# ------------------------------------------------------------------ templates
_TPL = re.compile(r"\{([^{}|]+)(?:\|([^{}]+))?\}")


def _sec(x) -> str:
    x = float(x)
    return str(int(x)) if x == int(x) else f"{x:.2f}".rstrip("0").rstrip(".")


def _first(x):
    if x is None or isinstance(x, str):
        return x or None
    try:
        seq = list(x)
        return seq[0] if seq else None
    except TypeError:
        return x


CONVERTERS: dict[str, Callable] = {
    "int": lambda v, a: int(float(v)), "sec": lambda v, a: _sec(v), "str": lambda v, a: str(v),
    "basename": lambda v, a: os.path.basename(str(v)), "stem": lambda v, a: os.path.splitext(os.path.basename(str(v)))[0],
    "first": lambda v, a: _first(v), "strip": lambda v, a: str(v).lstrip(a or " "),
    "dir0": lambda v, a: str(v).split("/")[0], "lower": lambda v, a: str(v).lower(),
    "letter0": lambda v, a: chr(65 + int(v)), "letter1": lambda v, a: chr(64 + int(v)),
}


def get_field(row: dict, name: str):
    """``a.b.c`` looks into nested dicts; a missing field is None."""
    if name in row:
        return row[name]
    cur: Any = row
    for part in name.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def render(spec_value: str | None, row: dict):
    """A column name returns the column's value (any type); a template returns a string, or None when a field it
    names is missing."""
    if spec_value is None:
        return None
    if "{" not in spec_value:
        return get_field(row, spec_value)
    missing = False

    def sub(m):
        nonlocal missing
        v = get_field(row, m.group(1).strip())
        for conv in (m.group(2) or "").split("|"):
            conv = conv.strip()
            if not conv or v is None:
                continue
            name, _, arg = conv.partition(":")
            v = CONVERTERS[name](v, arg)
        if v is None or (isinstance(v, float) and v != v):
            missing = True
            return ""
        return str(v)
    out = _TPL.sub(sub, spec_value)
    return None if missing else out


# ------------------------------------------------------------------ spec
@dataclass
class AdapterSpec:
    name: str
    files: list[str] = field(default_factory=list)
    records: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)      # question, answer, options, option_fields, subtask, video
    answer_type: str = "auto"                                  # auto | letter | index0 | index1 | text
    filters: list[dict] = field(default_factory=list)
    keep: str | None = None                                    # mcq: only items with options and a letter gold
    dedupe: bool = False
    hook: str | None = None                                    # "module_or_file.py:function"
    pinned: bool = False                                       # video references are (archive, member) pairs
    source: dict = field(default_factory=dict)                 # annotations / videos (video_index.onboard.sources)
    meta: dict = field(default_factory=dict)                   # year, citation_key, licence, capability_group, ...
    requires_audio: bool = False
    language: str = "en"

    @classmethod
    def from_dict(cls, d: dict) -> "AdapterSpec":
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise AdapterError(f"adapter spec: unknown keys {sorted(unknown)} (known: {sorted(known)})")
        d = dict(d)
        if isinstance(d.get("files"), str):
            d["files"] = [d["files"]]
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> "AdapterSpec":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        spec = cls.from_dict(d)
        if spec.hook and ":" in spec.hook and not os.path.isabs(spec.hook.split(":")[0]):
            cand = os.path.join(os.path.dirname(os.path.abspath(path)), spec.hook.split(":")[0])
            if os.path.exists(cand):
                spec.hook = cand + ":" + spec.hook.split(":", 1)[1]
        return spec


def _passes(row: dict, flt: dict) -> bool:
    v = get_field(row, flt["field"])
    if "equals" in flt:
        return v == flt["equals"] or str(v) == str(flt["equals"])
    if "in" in flt:
        return str(v).lower() in {str(x).lower() for x in flt["in"]}
    if "not_in" in flt:
        return str(v).lower() not in {str(x).lower() for x in flt["not_in"]}
    if "truthy" in flt:
        return bool(v) == bool(flt["truthy"])
    if "matches" in flt:
        return re.search(flt["matches"], str(v if v is not None else "")) is not None
    raise AdapterError(f"filter {flt}: expected one of equals / in / not_in / truthy / matches")


def keep_mcq(raw: dict) -> bool:
    """The record has an option list (at least two entries) and a single-letter gold."""
    return isinstance(raw.get("opts"), (list, dict)) and len(raw["opts"]) >= 2 \
        and bool(_MCQ_LETTER.match(str(raw.get("a", "")).strip()))


HOOKS: dict[str, Callable[[str], list[dict]]] = {}


def register_adapter(name: str):
    """Decorator: ``@register_adapter("MyBench") def load(annotation_dir) -> list[dict(v, q, a, opts, sub)]``."""
    def deco(fn):
        HOOKS[name] = fn
        return fn
    return deco


def load_hook(target: str) -> Callable[[str], list[dict]]:
    """``package.module:function`` or ``path/to/file.py:function``."""
    mod, _, fn = target.partition(":")
    if mod.endswith(".py"):
        s = importlib.util.spec_from_file_location("vi_adapter_" + os.path.basename(mod)[:-3], mod)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, fn or "load")


# ------------------------------------------------------------------ adapter
class GenericAdapter:
    """``records(annotation_dir)`` -> (raw records, info). Raw records are numbered in list order; that number is the
    item's qid suffix, so rerunning the adapter on the same files gives the same ids."""

    def __init__(self, spec: AdapterSpec | None = None, name: str | None = None):
        self.spec = spec or AdapterSpec(name=name or "benchmark")

    def records(self, annotation_dir: str) -> tuple[list[dict], dict]:
        spec = self.spec
        hook = load_hook(spec.hook) if spec.hook else HOOKS.get(spec.name)
        if hook is not None:
            recs = list(hook(annotation_dir))
            if not recs:
                raise AdapterError(f"{spec.name}: the adapter hook returned no records")
            return recs, dict(via="hook", n_rows=len(recs), n_dropped=0, columns={})
        rows = load_annotations(annotation_dir, spec.files or None, spec.records)
        if not rows:
            raise AdapterError(f"{spec.name}: no parseable annotation file in {annotation_dir}"
                               + (f" matching {spec.files}" if spec.files else ""))
        n_rows = len(rows)
        rows = [r for r in rows if all(_passes(r, f) for f in spec.filters)]
        f = spec.fields
        vcol = f.get("video") or detect(rows, VID_COLS)
        qcol = f.get("question") or detect(rows, Q_COLS)
        if not vcol or not qcol:
            raise AdapterError(f"{spec.name}: field detection failed (video={vcol}, question={qcol}); columns: "
                               f"{[k for k in rows[0] if k != '__file'][:14]}. Name the fields in an adapter spec.")
        acol = f.get("answer") or detect(rows, A_COLS)
        ocol = f.get("options") or (None if f.get("option_fields") else detect(rows, OPT_COLS))
        scol = f.get("subtask") or detect(rows, SUB_COLS)
        out, dropped = [], 0
        for r in rows:
            try:
                v = render(vcol, r)
                q = render(qcol, r)
            except Exception:  # noqa: BLE001
                v = q = None
            if v is None or q is None or (isinstance(v, str) and not v.strip()):
                dropped += 1                                    # a row without a video or question is not an item
                continue
            if f.get("option_fields"):
                opts = [get_field(r, c) for c in f["option_fields"]]
                opts = [str(o).strip() for o in opts if o is not None and str(o).strip() not in ("", "nan")] or None
            else:
                opts = render(ocol, r) if ocol else None
            try:
                a = self._answer(render(acol, r) if acol else "")
            except (TypeError, ValueError):
                dropped += 1                                    # the row lacks a usable answer field
                continue
            sub = render(scol, r) if scol else None
            if sub is not None and not isinstance(sub, str):
                sub = _first(sub)
            if spec.pinned and isinstance(v, str) and "::" in v:
                v = tuple(v.split("::", 1))
            out.append(dict(v=v, q=q, a=a, opts=opts, sub=str(sub) if sub is not None else "_none"))
        if spec.dedupe:
            seen, dd = set(), []
            for it in out:
                k = (str(it["v"]), str(it["q"])[:200])
                if k not in seen:
                    seen.add(k)
                    dd.append(it)
            out = dd
        if not out:
            raise AdapterError(f"{spec.name}: every row was dropped (filters, missing video or answer fields)")
        return out, dict(via="generic", n_rows=n_rows, n_dropped=dropped,
                         columns=dict(video=vcol, question=qcol, answer=acol, options=ocol or f.get("option_fields"),
                                      subtask=scol))

    def _answer(self, a):
        t = self.spec.answer_type
        if t == "index0":
            return chr(65 + int(float(a)))
        if t == "index1":
            return chr(64 + int(float(a)))
        if t == "letter":
            return str(a).strip().upper()
        return "" if a is None else a

    def keep(self, raw: dict) -> bool:
        if self.spec.keep in (None, "", "all"):
            return True
        if self.spec.keep == "mcq":
            return keep_mcq(raw)
        raise AdapterError(f"{self.spec.name}: unknown keep rule '{self.spec.keep}'")
