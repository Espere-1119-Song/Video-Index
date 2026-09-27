"""Stage compose_candidates: the hardest items of the screened pool.

Difficulty of an item = worst-case attacker percentile: each of the five screening margins (options only, blind,
single frame, 32 frames, 32 shuffled frames) is ranked among the items that passed the whole screen, and the
item's largest percentile is used. A small value means that no attacker came close to the correct option.

Candidates: items with at least two options and a valid answer index, ordering questions excluded (their options
are the events themselves; they cannot be scored as single choice); sorted by (percentile, item_id); one item per
video; the first ``compose.candidates`` (3,000). Every candidate gets a capability label through the ``labeler``
role (18 fine categories, four groups); candidates without a label in the four groups are dropped.

Output pool/compose/candidates.json and pool/compose/candidates_summary.csv.
"""
from __future__ import annotations

import csv
from collections import Counter

from ..pool import schema as S
from ..pool.labels import label_capabilities
from ..pool.select import attack_percentile
from ..runner import log
from . import CAPS, cfg, path
from .items import from_pool_row, save


def hardest(pool, n, margins=None):
    """pool: item table after the screen. -> candidate rows (mappings with attack_pct), hardest first."""
    el = pool[(pool["removed_by"] == "none") & (pool["pending_steps"].fillna("") == "")]
    el = attack_percentile(el, margins) if margins else attack_percentile(el)
    el = el[(el["options"].map(len) >= 2) & el["answer_idx"].notna()]
    el = el[~el["question"].astype(str).str.contains(S.ORDERING_RE, case=False, regex=True, na=False)]
    el = el[[0 <= int(a) < len(o) for a, o in zip(el["answer_idx"], el["options"])]]
    el = el.sort_values(["attack_pct", "item_id"])
    seen, out = set(), []
    for r in el.to_dict("records"):
        if r["video_id"] in seen:
            continue
        seen.add(r["video_id"])
        out.append(r)
        if n and len(out) >= n:
            break
    return out


def run(ctx, benchmarks=None, limit=None):
    pool = S.read_table(ctx, "pool_items")
    if benchmarks:
        pool = pool[pool["benchmark"].isin(set(benchmarks))]
    rows = hardest(pool, limit or cfg(ctx, "candidates"))
    labels = label_capabilities(ctx, rows, desc="compose/labels")
    items = [from_pool_row(r, (labels.get(r["item_id"]) or {}).get("group")) for r in rows]
    n_all = len(items)
    items = [it for it in items if it["capability"] in CAPS]
    save(path(ctx, "candidates.json"), items)
    by = Counter((it["benchmark"], it["capability"]) for it in items)
    with open(path(ctx, "candidates_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "capability", "n"])
        w.writerows([b, c, n] for (b, c), n in sorted(by.items()))
    per = dict(Counter(it["capability"] for it in items))
    log(f"compose_candidates: {n_all} hardest items, {len(items)} with a capability group: {per}", "compose")
    return dict(n_candidates=len(items), n_without_label=n_all - len(items), per_group=per)
