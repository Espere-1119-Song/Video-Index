"""Scores of a run: the row of the paper's Table 1 (Video, Blind, Gain, accuracy per capability group)."""
from __future__ import annotations

import csv
import glob
import json
import os

from . import protocol as P

COLUMNS = ["model", "n", "video", "blind", "gain"] + [k for k, _ in P.GROUPS] + [f"{k}_n" for k, _ in P.GROUPS] + \
          ["n_blind", "n_total", "partial", "mean_frames", "fallback_items", "unanswered"]


def load_rows(run_dir: str) -> list[dict]:
    """Scored rows of a run directory. A row whose ``correct`` is null (request failed, no video) is not scored; of
    several rows for the same (model, protocol, item, permutation) the first scored one is kept."""
    rows, seen = [], set()
    for p in sorted(glob.glob(os.path.join(run_dir, "*__*.jsonl"))):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("correct") is None:
                continue
            perm = tuple(r.get("perm") or []) if r.get("protocol") == "blind" else ()
            k = (r["model"], r["protocol"], r["item_id"], perm)
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    return rows


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def table(rows: list[dict], groups: dict[str, str] | None = None, n_total: int | None = None,
          video_protocol: str = "video1fps") -> list[dict]:
    """One dict per model. ``groups`` maps item_id -> capability group (taken from the rows when omitted)."""
    out = []
    for m in sorted({r["model"] for r in rows}):
        vid = [r for r in rows if r["model"] == m and r["protocol"] == video_protocol]
        bl = [r for r in rows if r["model"] == m and r["protocol"] == "blind"]
        per_item_blind: dict[str, list[float]] = {}
        for r in bl:
            per_item_blind.setdefault(r["item_id"], []).append(float(r["correct"]))
        d = dict(model=m, n=len(vid), video=100 * _mean(float(r["correct"]) for r in vid) if vid else float("nan"))
        if vid:
            common = [r for r in vid if r["item_id"] in per_item_blind]
            if common:
                d["blind"] = 100 * _mean(_mean(per_item_blind[r["item_id"]]) for r in common)
                d["gain"] = 100 * _mean(float(r["correct"]) for r in common) - d["blind"]
                d["n_blind"] = len(common)
        elif per_item_blind:
            d["blind"] = 100 * _mean(_mean(v) for v in per_item_blind.values())
            d["n_blind"] = len(per_item_blind)
        for k, _ in P.GROUPS:
            gg = [r for r in vid if (groups or {}).get(r["item_id"], r.get("capability_group")) == k]
            d[k] = 100 * _mean(float(r["correct"]) for r in gg) if gg else float("nan")
            d[f"{k}_n"] = len(gg)
        d["n_total"] = n_total
        d["partial"] = bool(n_total and len(vid) < n_total)
        d["mean_frames"] = _mean(r.get("frames") or 0 for r in vid) if vid else float("nan")
        d["fallback_items"] = sum(1 for r in vid if r.get("fallback"))
        d["unanswered"] = sum(1 for r in vid if not str(r.get("pred") or "").strip())
        out.append(d)
    out.sort(key=lambda d: -(d["video"] if d["video"] == d["video"] else -1))
    return out


def fmt(v, d=1):
    if v is None or (isinstance(v, float) and v != v):
        return "--"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def render(tab: list[dict]) -> str:
    head = ["Model", "n", "Video", "Blind", "Gain"] + [lab for _, lab in P.GROUPS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for d in tab:
        n = f"{d['n']}" + (f" of {d['n_total']}" if d.get("partial") else "")
        lines.append("| " + " | ".join([d["model"], n, fmt(d.get("video")), fmt(d.get("blind")), fmt(d.get("gain"))]
                                       + [fmt(d.get(k)) for k, _ in P.GROUPS]) + " |")
    return "\n".join(lines)


def write_csv(tab: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for d in tab:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()})
