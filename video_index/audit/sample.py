"""Sample stage: items/<bench>.jsonl -> samples/<bench>.jsonl.

Draw    all items when the benchmark has at most n (= 300); else n items allocated to the subtasks in proportion to
        their size, at least `sample_min_per_stratum` (= 10) per subtask, and drawn with one seeded generator: subtasks in
        sorted order, the qid-sorted items of each subtask shuffled, the allocated prefix taken.
Refill  after the screen stage has removed items, or when a sample is below n: the draw is reproduced and continued.
        (a) every removed item is replaced from its own subtask, in the shuffled order after the allocated prefix;
        (b) a remaining shortfall is filled across the other subtasks in proportion to the original allocation
            (largest remainder), skipping exhausted subtasks;
        (c) when every subtask is exhausted the sample keeps the shortfall (full coverage).
        Not eligible as a replacement: items of the original draw, items that are not multiple choice, option lists
        whose length is not among the declared option counts, near-duplicates of a kept or accepted item.
"""
from __future__ import annotations

import csv
import os
import random
from collections import defaultdict

from ..data.schema import read_jsonl, write_jsonl
from ..runner import log
from . import chance as chance_mod
from .items import bench_meta, item_options, select, subtask, video_key


def _strata(items):
    strata = defaultdict(list)
    for r in items:
        strata[subtask(r) or "_none"].append(r)
    return strata


def draw_order(items, n_target=300, seed=42, min_per_stratum=10):
    """-> (alloc, orders, full). orders[k] = the shuffled qid-sorted items of subtask k; the sample is
    orders[k][:alloc[k]] over the subtasks in sorted order."""
    rng = random.Random(seed)
    strata = _strata(items)
    if len(items) <= n_target:
        return ({k: len(v) for k, v in strata.items()},
                {k: sorted(v, key=lambda r: r["qid"]) for k, v in strata.items()}, True)
    if len(strata) == 1:
        k = next(iter(strata))
        pool = sorted(strata[k], key=lambda r: r["qid"])
        rng.shuffle(pool)
        return {k: n_target}, {k: pool}, False
    total = len(items)
    alloc = {}
    for k, v in strata.items():
        alloc[k] = min(max(min_per_stratum, round(n_target * len(v) / total)), len(v))
    while sum(alloc.values()) > n_target:
        k = max(alloc, key=lambda k: (alloc[k], len(strata[k])))
        if alloc[k] <= min_per_stratum:
            break
        alloc[k] -= 1
    orders = {}
    for k in sorted(strata):
        pool = sorted(strata[k], key=lambda r: r["qid"])
        rng.shuffle(pool)
        orders[k] = pool
    return alloc, orders, False


def stratified_sample(items, n_target=300, seed=42, min_per_stratum=10):
    """-> (sample, full_coverage)."""
    alloc, orders, full = draw_order(items, n_target, seed, min_per_stratum)
    if full:
        return sorted(items, key=lambda r: r["qid"]), True
    out = []
    for k in sorted(orders):
        out.extend(orders[k][:alloc[k]])
    return out, False


def largest_remainder(weights, n):
    """Integer allocation of n over the keys, proportional to the weights."""
    tot = sum(weights.values())
    if n <= 0 or tot <= 0:
        return {k: 0 for k in weights}
    raw = {k: n * w / tot for k, w in weights.items()}
    base = {k: int(v) for k, v in raw.items()}
    rest = n - sum(base.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - base[k]), k))[:rest]:
        base[k] += 1
    return base


def refill(items, sample, removed, meta=None, n_target=300, seed=42, min_per_stratum=10, is_duplicate=None):
    """-> (new sample, info). sample: current rows; removed: qids taken out by the screen.
    is_duplicate(candidate, kept_rows) -> bool is the near-duplicate check (None = not checked)."""
    meta = meta or {}
    samples = {str(r["qid"]): r for r in sample}
    kept = [r for q, r in samples.items() if q not in removed]
    n_needed = n_target - len(kept)
    info = dict(n_sampled=len(samples), n_removed=len(set(removed) & set(samples)), n_needed=max(0, n_needed), n_added=0,
                excl_sampled=0, excl_non_mcq=0, excl_option_count=0, excl_near_dup=0, draw_method="", full_coverage=False)
    if n_needed <= 0:
        return kept, info
    alloc, orders, _ = draw_order(items, n_target, seed, min_per_stratum)
    drawn = {str(r["qid"]) for k, o in orders.items() for r in o[:alloc[k]]}
    if set(samples) <= drawn:
        info["draw_method"] = "continuation"
        cont = {k: list(o[alloc[k]:]) for k, o in orders.items()}
        weights = {k: alloc[k] for k in orders}
    else:                                     # the stored sample is not the reproduced draw
        info["draw_method"] = "seed_fallback"
        n_sub = defaultdict(int)
        for r in samples.values():
            n_sub[subtask(r) or "_none"] += 1
        cont = {}
        for k, o in orders.items():
            pool = sorted([r for r in o if str(r["qid"]) not in samples], key=lambda r: r["qid"])
            random.Random(seed).shuffle(pool)
            cont[k] = pool
        weights = {k: n_sub.get(k, 0) or alloc.get(k, 0) for k in orders}
    bc = chance_mod.bench_chance(samples, meta)
    decl = chance_mod.declared_counts(meta)
    excluded = set(samples) | drawn
    current = list(kept)

    def eligible(r):
        if str(r["qid"]) in excluded:
            info["excl_sampled"] += 1
            return False
        c, src = chance_mod.item_chance(r, meta, bc["k_inf"], bc["digit_index"])
        if src == "open" or c <= 0:
            info["excl_non_mcq"] += 1
            return False
        opts = item_options(r)[0]
        if decl and opts and len(opts) not in decl:
            info["excl_option_count"] += 1
            return False
        if is_duplicate is not None and is_duplicate(r, current):
            info["excl_near_dup"] += 1
            return False
        return True

    def take(k):
        while cont.get(k):
            r = cont[k].pop(0)
            if eligible(r):
                current.append(r)
                return r
        return None

    added = []
    for rq in sorted(set(removed) & set(samples)):                    # (a) same subtask
        if len(added) >= n_needed:
            break
        r = take(subtask(samples[rq]) or "_none")
        if r is not None:
            added.append(dict(r, replaces=rq))
    need = n_needed - len(added)
    while need > 0:                                                   # (b) proportional fill
        live = {k: w for k, w in weights.items() if cont.get(k)}
        if not live:
            break
        if sum(live.values()) <= 0:
            live = {k: 1 for k in live}
        share, got = largest_remainder(live, need), 0
        for k in sorted(share):
            for _ in range(share[k]):
                r = take(k)
                if r is None:
                    break
                added.append(r)
                got += 1
        if got == 0:
            break
        need = n_needed - len(added)
    info["n_added"] = len(added)
    info["full_coverage"] = len(added) < n_needed                     # (c)
    return kept + added, info


def removed_items(ctx, bench):
    """qids with action = remove in tables/screen_items.csv."""
    p = os.path.join(ctx.work_dir, "tables", "screen_items.csv")
    if not os.path.exists(p):
        return set()
    return {r["item_id"] for r in csv.DictReader(open(p)) if r["benchmark"] == bench and r["action"] == "remove"}


def _params(ctx):
    return dict(n_target=int(ctx.get("sample_n", 300)), seed=int(ctx.get("seed", 42)),
                min_per_stratum=int(ctx.get("sample_min_per_stratum", 10)))


def run(ctx, benchmarks=None, limit=None):
    """Draw the sample of every benchmark that has items and no sample (an existing sample is kept)."""
    out = []
    for bench in select(ctx, benchmarks, kind="items"):
        sp = ctx.path("samples", bench=bench)
        if os.path.exists(sp) and not ctx.get("resample", False):
            log(f"sample: {bench}: kept ({len(read_jsonl(sp))} items)", "audit")
            continue
        items = read_jsonl(ctx.path("items", bench=bench))
        if not items:
            log(f"sample: {bench}: no items file", "audit")
            continue
        sample, full = stratified_sample(items, **_params(ctx))
        write_jsonl(sp, sample)
        log(f"sample: {bench}: {len(sample)} of {len(items)} items" + (" (all items)" if full else ""), "audit")
        out.append(dict(benchmark=bench, n_items=len(items), n_sampled=len(sample), full_coverage=full))
    return out


def run_refill(ctx, benchmarks=None, limit=None):
    """Stage `refill`: replace the items removed by the screen and fill samples below n."""
    from . import screen
    out = []
    for bench in select(ctx, benchmarks):
        sp, ip = ctx.path("samples", bench=bench), ctx.path("items", bench=bench)
        sample, items = read_jsonl(sp), read_jsonl(ip)
        removed = removed_items(ctx, bench)
        n = int(ctx.get("sample_n", 300))
        if not items or (not (removed & {str(r["qid"]) for r in sample}) and len(sample) >= min(n, len(items))):
            continue
        new, info = refill(items, sample, removed, bench_meta(ctx, bench), is_duplicate=screen.duplicate_checker(ctx),
                           **_params(ctx))
        write_jsonl(ctx.path("log", name=f"sample_before_refill_{bench}.jsonl"), sample)
        write_jsonl(sp, new)
        log(f"refill: {bench}: {info['n_removed']} removed, {info['n_added']} added, sample = {len(new)}"
            + (" (every subtask exhausted)" if info["full_coverage"] else ""), "audit")
        out.append(dict(benchmark=bench, **info))
    if out:
        p = ctx.path("table", name="refill.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
    return out
