"""Coverage-first selection of the pool: the caps and the coverage pass on a synthetic table, and the reproduction of
the released 10,000-item selection from the research files when VIDEO_INDEX_RESEARCH_CODE is set."""
import json
import os
import random

import pandas as pd

from video_index.pool import scenes as SC
from video_index.pool import select as SEL
from video_index.pool.screen import MARGINS


def run_all(namespace):
    """Run every test_* function of a module without pytest."""
    n = 0
    for k, f in sorted(namespace.items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
            print(f"ok  {k}")
    print(f"{n} tests passed")


SCENES = ["cooking_food", "sports_exercise", "animals_wildlife", "film_tv_narrative"]
WORDS = "what why when where who which how does is the a person object colour moves before after left right".split()


def synthetic(n_bench=8, n_videos=900, seed=5):
    """(item table, video table): 8 benchmarks, 2 to 4 items per video, templated questions in one benchmark."""
    rng = random.Random(seed)
    items, videos = [], []
    for v in range(n_videos):
        bench = f"B{v % n_bench}"
        dur = rng.choice([5, 20, 45, 100, 400, 1000, 3000])
        videos.append(dict(video_id=f"v{v:04d}", group=SC.duration_group(dur), duration_s=float(dur),
                           cluster_cal=v if v % 5 else v - v % 50, scene=SCENES[v % len(SCENES)]))
        for j in range(rng.choice([2, 3, 4])):
            q = (f"How many times does the person jump in clip {v}?" if bench == "B0"
                 else " ".join(rng.choice(WORDS) for _ in range(6)) + "?")
            items.append(dict(benchmark=bench, item_id=f"{bench}_{v:04d}_{j}", video_id=f"v{v:04d}", question=q,
                              question_raw=None, removed_by="none" if rng.random() < 0.9 else "blind",
                              pending_steps="" if rng.random() < 0.97 else "v32",
                              task_class=f"task{(v + j) % 5}" if bench != "B7" else "", scene_type=videos[-1]["scene"],
                              **{m: rng.gauss(0, 1) for m in MARGINS}))
    pool, groups = pd.DataFrame(items), pd.DataFrame(videos)
    pool.loc[pool.index[:40], "scene_type"] = ""                                   # videos without a scene label
    pool.loc[pool.index[-3:], "question"] = "Select the correct order of the following options"
    return pool, groups


def test_eligible():
    pool, groups = synthetic()
    el, exc = SEL.eligible(pool, groups)
    assert exc["n_unscorable"] == int(((pool.removed_by == "none") & (pool.pending_steps == "")
                                       & pool.question.str.startswith("Select the correct")).sum())
    assert exc["n_pending"] == int(((pool.removed_by == "none") & (pool.pending_steps != "")).sum())
    assert exc["n_unlabelled"] > 0 and exc["n_no_group"] == 0
    assert set(el.removed_by) == {"none"} and set(el.pending_steps) == {""} and "" not in set(el.scene_type)
    assert list(el.index) == list(range(len(el)))
    key = list(zip(el.tgroup, el.attack_pct, el.item_id))
    assert key == sorted(key)                                                       # item order of the selection
    assert el.attack_pct.between(0, 1).all() and (el.task_class != "").all()
    b0 = el[el.benchmark == "B0"]
    assert b0.template.nunique() == 1 and (b0.tgroup == len(b0)).all()              # one template for B0


def test_selection_rules():
    pool, groups = synthetic()
    el, _ = SEL.eligible(pool, groups)
    budget = 300
    chosen, state = SEL.select(el, budget=budget, verbose=False)
    sel = el.loc[[i for i, _ in chosen]]
    assert len(sel) == budget and [o for _, o in chosen] == list(range(1, budget + 1))
    assert sel.item_id.is_unique
    # per-source cap: 5 % of the budget per benchmark, unless the cap was lifted for a group
    over = sel.benchmark.value_counts() - state["bench_cap"]
    assert state["bench_cap"] == 15 and int(over.clip(lower=0).sum()) == sum(state["items_above_bench_cap"].values())
    # per-video cap: one item per video, up to three in scarce cells
    scarce = set(map(tuple, state["scarce"]))
    per_video = sel.groupby("video_id").size()
    cell = {v: (g, s) for v, g, s in zip(sel.video_id, sel.group, sel.scene_type)}
    assert all(n <= (3 if cell[v] in scarce else 1) for v, n in per_video.items())
    # coverage pass: every (benchmark, task class) pair with an eligible item is represented
    pairs = set(map(tuple, el[["benchmark", "task_class"]].drop_duplicates().values))
    got = set(map(tuple, sel[["benchmark", "task_class"]].drop_duplicates().values))
    assert state["n_pairs"] == len(pairs) and state["coverage_items"] <= len(pairs)
    assert len(pairs - got) == len(pairs) - state["coverage_items"]
    # every scene type and every duration group with supply is represented; the groups are balanced
    assert set(sel.scene_type) == set(el.scene_type) and set(sel.group) == set(el.group)
    counts = sel.group.value_counts()
    assert counts.max() - counts.min() <= 0.5 * budget / 7
    # deterministic
    again, _ = SEL.select(el, budget=budget, verbose=False)
    assert again == chosen
    # a budget above the supply stops when nothing is admissible
    small, _ = SEL.select(el.iloc[:50].reset_index(drop=True), budget=1000, verbose=False)
    assert 0 < len(small) <= 50


def test_template_cap():
    pool, groups = synthetic()
    pool = pool[pool.benchmark.isin(["B0", "B1"])].copy()
    el, _ = SEL.eligible(pool, groups)
    chosen, state = SEL.select(el, budget=70, share=1.0, verbose=False)
    sel = el.loc[[i for i, _ in chosen]]
    scarce = set(map(tuple, state["scarce"]))
    normal = sel[[(g, s) not in scarce for g, s in zip(sel.group, sel.scene_type)]]
    n_tmpl = normal[normal.benchmark == "B0"].groupby(["group"]).size()
    caps = state["template_cap"]
    # B0 has one template: outside scarce cells a group takes at most its template cap, plus the coverage items
    assert all(n <= caps[g] + state["coverage_items"] for g, n in n_tmpl.items())


def test_reproduces_released_selection():
    root = os.environ.get("VIDEO_INDEX_RESEARCH_CODE")
    r = os.path.join(root or "", "results", "pool")
    if not root or not os.path.exists(os.path.join(r, "video_index_selection.csv")):
        print("    skipped: VIDEO_INDEX_RESEARCH_CODE not set")
        return
    marg = ["opt_margin", "blind_margin", "sf_margin", "v32_2b_margin", "v32_2b_shuffle_margin"]
    cols = ["benchmark", "item_id", "video_id", "question", "question_raw", "removed_by", "pending_steps", "task_class",
            "scene_type"] + marg
    pool = pd.read_parquet(os.path.join(r, "pool_items.parquet"), columns=cols)
    groups = pd.read_csv(os.path.join(r, "scene_groups_videos.csv"),
                         usecols=["video_id", "group", "cluster_cal", "duration_s"])
    templates = json.load(open(os.path.join(r, "screen_templates.json")))
    el, _ = SEL.eligible(pool, groups, templates, margins=marg)
    chosen, _ = SEL.select(el, verbose=False)
    ref = pd.read_csv(os.path.join(r, "video_index_selection.csv"))
    mine = list(el.loc[[i for i, _ in chosen], "item_id"])
    assert mine == list(ref.item_id), f"{len(set(mine) & set(ref.item_id))} of {len(ref)} items shared"
    print(f"    {len(mine)} items, identical to the released selection in the same order")


def test_reproduces_scene_clusters():
    """About ten minutes on 38,835 videos: runs only with VIDEO_INDEX_SLOW_TESTS=1."""
    import numpy as np
    root = os.environ.get("VIDEO_INDEX_RESEARCH_CODE")
    e = os.path.join(root or "", "data", "embeddings")
    if not root or os.environ.get("VIDEO_INDEX_SLOW_TESTS") != "1" or not os.path.exists(os.path.join(e, "visual_siglip2.npy")):
        print("    skipped: needs VIDEO_INDEX_RESEARCH_CODE and VIDEO_INDEX_SLOW_TESTS=1")
        return
    ref = pd.read_csv(os.path.join(root, "results", "pool", "scene_groups_videos.csv"))
    row = {h: i for i, h in enumerate(json.load(open(os.path.join(e, "visual_hashes.json"))))}
    V = np.load(os.path.join(e, "visual_siglip2.npy"), mmap_mode="r")
    X = np.asarray(V[[row[h] for h in ref.video_id]], dtype=np.float32)
    cal, fix, _ = SC.scene_clusters(ref.group.values.astype(str), X)
    assert (cal == ref.cluster_cal.values).all() and (fix == ref.cluster_fixed.values).all()
    print(f"    {len(ref)} videos, cluster assignment identical")


if __name__ == "__main__":
    run_all(globals())
