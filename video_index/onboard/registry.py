"""tables/registry.csv (one row per onboarded benchmark) and small CSV helpers."""
from __future__ import annotations

import csv
import os
import time

REGISTRY_COLS = ["name", "source", "video_source", "year", "citation_key", "licence", "capability_group", "adapter",
                 "n_items", "n_mcq", "n_sampled", "chance", "break_level", "status", "added"]


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def upsert(path: str, row: dict, key: str = "name", cols: list[str] | None = None) -> None:
    """Insert the row or update the row with the same key; columns that the row does not carry keep their value."""
    rows = read_csv(path)
    cols = list(cols or [])
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    for c in row:
        if c not in cols:
            cols.append(c)
    for r in rows:
        if r.get(key) == str(row[key]):
            r.update({k: v for k, v in row.items() if v is not None})
            break
    else:
        rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    os.replace(tmp, path)


def register(ctx, name: str, spec, sources: dict, report: dict | None = None, n_sampled: int | None = None,
             status: str = "onboarded") -> dict:
    from .sources import label
    path = ctx.path("table", name="registry.csv")
    meta = spec.meta or {}
    level = next((r for r in read_csv(ctx.path("table", name="levels.csv"))
                  if name in (r.get("benchmark"), r.get("bench"), r.get("name"))), {})
    row = dict(name=name, source=label(sources["annotations"]) if sources.get("annotations") else "",
               video_source=label(sources["videos"]) if sources.get("videos") else "",
               year=meta.get("year", ""), citation_key=meta.get("citation_key", ""),
               licence=meta.get("licence", meta.get("license", "")), capability_group=meta.get("capability_group", ""),
               adapter=("hook" if spec.hook else "spec" if (spec.fields or spec.files) else "detected"),
               n_items=(report or {}).get("n_items"), n_mcq=(report or {}).get("n_mcq"), n_sampled=n_sampled,
               chance=(report or {}).get("chance"), break_level=level.get("break_level"), status=status,
               added=time.strftime("%Y-%m-%d"))
    upsert(path, row, cols=REGISTRY_COLS)
    return row


def stratified_sample(items: list[dict], n: int, seed: int = 42, min_per_stratum: int = 10) -> tuple[list[dict], bool]:
    """The sample of the audit: all items when there are at most n, else n items allocated to the subtasks in
    proportion to their size (at least ``min_per_stratum`` each), drawn with a seeded shuffle of the qid-sorted items.
    Returns (sample, full_coverage). Used when the audit package's own ``sample`` stage is not installed."""
    import random
    from collections import defaultdict
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in items:
        strata[r.get("subtask") or "_none"].append(r)
    if len(items) <= n:
        return sorted(items, key=lambda r: r["qid"]), True
    if len(strata) == 1:
        pool = sorted(items, key=lambda r: r["qid"])
        rng.shuffle(pool)
        return pool[:n], False
    alloc = {k: min(len(v), max(min_per_stratum, round(n * len(v) / len(items)))) for k, v in strata.items()}
    while sum(alloc.values()) > n:
        k = max(alloc, key=lambda k: (alloc[k], len(strata[k])))
        if alloc[k] <= min_per_stratum:
            break
        alloc[k] -= 1
    out = []
    for k in sorted(strata):
        pool = sorted(strata[k], key=lambda r: r["qid"])
        rng.shuffle(pool)
        out.extend(pool[:alloc[k]])
    return out, False
