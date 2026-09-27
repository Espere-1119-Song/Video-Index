"""Sampling: the stratified draw, its continuation (refill), and the comparison with the research code."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_audit_common import research_module, run_all  # noqa: E402

from video_index.audit import sample  # noqa: E402


def make_items(sizes, seed=0):
    rng = random.Random(seed)
    items = []
    for sub, n in sizes.items():
        for _ in range(n):
            i = len(items)
            items.append(dict(qid=f"B_{i}", question=f"question {i}?", options=["a", "b", "c", "d"], answer="ABCD"[rng.randrange(4)],
                              subtask=sub, subcategory=sub, video_ref=f"v{i}.mp4"))
    rng.shuffle(items)
    return items


CASES = [dict(a=700, b=200, c=90, d=9, e=3), dict(only=1200), dict(x=120, y=130), dict(a=400, b=5, c=5, d=5),
         {f"s{i:02d}": 40 + i for i in range(40)}, {"": 500, "z": 100}]


def test_sizes_and_strata():
    for sizes in CASES:
        items = make_items(sizes)
        out, full = sample.stratified_sample(items, 300, 42, 10)
        assert full == (len(items) <= 300)
        assert len({r["qid"] for r in out}) == len(out)
        if full:
            assert len(out) == len(items)
            continue
        alloc, _, _ = sample.draw_order(items, 300, 42, 10)
        for k, n in alloc.items():
            assert n == sum(1 for r in out if (r["subtask"] or "_none") == k)
            assert n >= min(10, sizes[k if k != "_none" else ""])
        if all(n >= 10 for n in sizes.values()) and len(sizes) <= 30:
            assert len(out) == 300


def test_deterministic():
    items = make_items(CASES[0])
    a, _ = sample.stratified_sample(items, 300, 42, 10)
    b, _ = sample.stratified_sample(list(reversed(items)), 300, 42, 10)
    assert [r["qid"] for r in a] == [r["qid"] for r in b]
    c, _ = sample.stratified_sample(items, 300, 43, 10)
    assert [r["qid"] for r in a] != [r["qid"] for r in c]


def test_refill_replaces_from_the_same_subtask():
    items = make_items(CASES[0])
    drawn, _ = sample.stratified_sample(items, 300, 42, 10)
    removed = {r["qid"] for r in drawn if r["subtask"] == "b"}
    removed = set(sorted(removed)[:7])
    new, info = sample.refill(items, drawn, removed, n_target=300, seed=42, min_per_stratum=10)
    assert len(new) == 300 and info["n_added"] == 7 and info["draw_method"] == "continuation"
    assert not ({r["qid"] for r in new} & removed)
    added = [r for r in new if r.get("replaces")]
    assert len(added) == 7 and all(r["subtask"] == "b" for r in added)
    alloc, orders, _ = sample.draw_order(items, 300, 42, 10)
    assert [r["qid"] for r in added] == [r["qid"] for r in orders["b"][alloc["b"]:alloc["b"] + 7]]


def test_refill_exhausted_subtask_and_full_coverage():
    items = make_items(dict(a=400, b=12))
    drawn, _ = sample.stratified_sample(items, 300, 42, 10)
    removed = {r["qid"] for r in drawn if r["subtask"] == "b"}
    new, info = sample.refill(items, drawn, removed, n_target=300, seed=42, min_per_stratum=10)
    n_b_left = 12 - len(removed)
    assert len(new) == 300 and sum(1 for r in new if r.get("replaces")) == n_b_left
    small = make_items(dict(a=20, b=15))
    drawn, full = sample.stratified_sample(small, 300, 42, 10)
    new, info = sample.refill(small, drawn, {drawn[0]["qid"]}, n_target=300, seed=42, min_per_stratum=10)
    assert full and len(new) == 34 and info["full_coverage"]


def test_refill_skips_non_multiple_choice_and_duplicates():
    items = make_items(dict(a=500, b=100))
    drawn, _ = sample.stratified_sample(items, 300, 42, 10)
    alloc, orders, _ = sample.draw_order(items, 300, 42, 10)
    nxt = orders["a"][alloc["a"]:alloc["a"] + 3]
    nxt[0]["options"], nxt[0]["answer"] = None, "a free-text answer"          # open item
    first_a = next(r for r in drawn if r["subtask"] == "a")
    new, info = sample.refill(items, drawn, {first_a["qid"]}, n_target=300, seed=42, min_per_stratum=10,
                              is_duplicate=lambda r, kept: r["qid"] == nxt[1]["qid"])
    assert info["excl_non_mcq"] == 1 and info["excl_near_dup"] == 1
    assert [r["qid"] for r in new if r.get("replaces")] == [nxt[2]["qid"]]


def test_largest_remainder():
    assert sample.largest_remainder(dict(a=1, b=1, c=1), 4) == dict(a=2, b=1, c=1)
    assert sum(sample.largest_remainder(dict(a=7, b=2, c=1), 13).values()) == 13


def test_same_draw_as_the_research_code():
    ref = research_module("stage_b")
    if ref is None:
        print("skipped: research code not available")
        return
    for sizes in CASES:
        items = make_items(sizes)
        a, fa = sample.stratified_sample(items, 300, ref.SEED, ref.MIN_STRATUM)
        b, fb = ref.stratified_sample(items, 300)
        assert fa == fb and [r["qid"] for r in a] == [r["qid"] for r in b], sizes


if __name__ == "__main__":
    run_all(globals())
