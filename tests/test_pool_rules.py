"""Rules of the pool: the multiple-choice rule, option handling, letter scores, removal rules, duplicate screens,
funnel accounting, scene clusters, worst-case attacker percentile. Comparison with the research code on the same
inputs when VIDEO_INDEX_RESEARCH_CODE is set."""
import math
import os
import random
import sys

import numpy as np
import pandas as pd

from video_index.pool import STAGES, dedup as D, logprobs as L, scenes as SC, schema as S, screen
from video_index.pool.select import attack_percentile

ITEMS = [
    (dict(question="Q", options=["cat", "dog", "bird"], answer="B"), "mcq", 1),
    (dict(question="Q", options=["A. cat", "B. dog"], answer="(A)"), "mcq", 0),
    (dict(question="Q", options=["cat", "dog"], answer="dog"), "mcq", 1),
    (dict(question="Q", options=["A. Yes", "B. No"], answer="A"), "yesno", 0),
    (dict(question="Q", options=["True", "False"], answer="B"), "yesno", 1),
    (dict(question="Q", options=None, answer="a man opens a door"), "open", -1),
    (dict(question="Q", options=["only one"], answer="A"), "open", -1),
    (dict(question="Q", options=["a", "b", "c"], answer="A,B"), "open", -1),
    (dict(question="Q", options=["a", "b", "c"], answer="E"), "open", -1),
    (dict(question="Q", options=[f"o{i}" for i in range(27)], answer="A"), "open", -1),
    (dict(question="What is it?\nA. cat\nB. dog\nC. bird\nAnswer with the letter.", options=None, answer="C"), "mcq", 2),
    (dict(question="What is it? Option A: cat Option B: dog", options=None, answer="B"), "mcq", 1),
    (dict(question="What is it?\nA. cat\nB. dog", options=None, answer="1"), "open", -1),
    (dict(question="Q", options=[{"label": "A", "text": "cat"}, {"label": "B", "text": "dog"}], answer="B"), "mcq", 1),
]


def research_module(name, *subdirs):
    """Module of the research code of the study (VIDEO_INDEX_RESEARCH_CODE = the directory that holds its
    `pipeline/` and `analysis/`); None when it is not available."""
    import importlib
    root = os.environ.get("VIDEO_INDEX_RESEARCH_CODE")
    if not root or not os.path.isdir(root):
        return None
    for d in ("pipeline",) + subdirs:
        if os.path.join(root, d) not in sys.path:
            sys.path.insert(0, os.path.join(root, d))
    try:
        return importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        print(f"    research module {name} not importable: {type(e).__name__}: {e}")
        return None


def run_all(namespace):
    """Run every test_* function of a module without pytest."""
    n = 0
    for k, f in sorted(namespace.items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
            print(f"ok  {k}")
    print(f"{n} tests passed")


def test_stage_registry():
    import importlib
    for name, mod in STAGES.items():
        assert callable(importlib.import_module(mod).run), name


def test_is_mcq_item():
    for it, fmt, idx in ITEMS:
        row = S.item_row(dict(benchmark="B", qid="q", **it))
        assert (row["format"], row["answer_idx"]) == (fmt, idx if fmt != "open" else -1), it
        assert S.is_mcq_item(it) == (fmt == "mcq"), it
    row = S.item_row(dict(benchmark="B", qid="q", **ITEMS[10][0]))
    assert row["options"] == ["cat", "dog", "bird"] and row["question"] == "What is it?"
    assert row["options_source"] == "inline_question" and row["question_raw"].startswith("What is it?\nA.")
    assert S.item_row(dict(benchmark="B", qid="q", **ITEMS[1][0]))["option_prefix_stripped"] is True


def test_resolve_answer_reasons():
    reasons = [S.item_row(dict(benchmark="B", qid="q", **it))["format_reason"] for it, _, _ in ITEMS]
    assert reasons[3:10] == ["yes_no_options", "yes_no_options", "no_options", "no_options", "multi_select",
                             "letter_out_of_range", "too_many_options"]
    assert S.resolve_answer("a fish", ["a cat", "a dog", "a bird"])[5] == "answer_not_in_options"
    assert S.resolve_answer("2", ["x", "y"], inline=True)[5] == "answer_not_in_options"


def test_video_id():
    assert S.video_id_of(dict(video_id="abc")) == "abc"
    a = S.video_id_of(dict(video_ref=dict(repo="r", member="x.mp4")))
    assert a.startswith("k:") and a == S.video_id_of(dict(video_ref=dict(member="x.mp4", repo="r")))
    assert S.video_id_of({}) == ""


def test_letter_logprobs():
    top = [dict(token="B", logprob=-0.2), dict(token=" B", logprob=-2.0), dict(token="A", logprob=-3.0),
           dict(token="The", logprob=-4.0), dict(token="(C", logprob=-5.0)]
    lp, missing = L.letter_logprobs(top, 4)
    assert missing == ["D"] and lp[3] == L.LP_FLOOR
    assert abs(lp[1] - math.log(math.exp(-0.2) + math.exp(-2.0))) < 1e-9 and lp[2] == -5.0
    assert L.argmax_pos(lp) == 1
    assert abs(L.margin_of(lp, 1) - (lp[1] - (lp[0] + lp[2] + lp[3]) / 3)) < 1e-12
    i, conf = L.confidence(lp)
    assert i == 1 and 0.9 < conf < 1.0 and L.confidence([L.LP_FLOOR] * 3) == (None, 0.0)
    perms = L.permutations_for("B_1", 4)
    assert perms == L.permutations_for("B_1", 4) and len(perms) == 4 and all(sorted(p) == [0, 1, 2, 3] for p in perms)
    assert perms != L.permutations_for("B_2", 4)


def _score(lp, k=3):
    am = L.argmax_pos(lp)
    return dict(lp=lp, argmax=am if lp[am] > L.LP_FLOOR else None, missing=[], top1_token="A", scored="logprobs")


def test_removal_rules():
    perms = [[0, 1, 2]] * 4
    strong, weak, wrong = [-0.1, -3.0, -3.0], [-0.9, -1.2, -1.3], [-3.0, -0.1, -3.0]
    r = screen.text_step_result(perms, [_score(strong)] * 4, 0, "blind")
    assert r["remove"] and r["blind_acc_4perm"] == 1.0 and r["blind_margin"] > L.LOG2
    r = screen.text_step_result(perms, [_score(strong)] * 3 + [_score(wrong)], 0, "blind")
    assert r["remove"] and r["blind_acc_4perm"] == 0.75                     # three of four, mean margin > log 2
    r = screen.text_step_result(perms, [_score(strong)] * 2 + [_score(wrong)] * 2, 0, "opt")
    assert not r["remove"] and r["opt_acc_4perm"] == 0.5
    r = screen.text_step_result(perms, [_score(weak)] * 4, 0, "opt")
    assert not r["remove"] and r["opt_acc_4perm"] == 1.0                    # correct, margin below log 2
    r = screen.text_step_result([[2, 0, 1]] * 4, [_score(wrong)] * 4, 0, "blind")
    assert r["remove"] and r["perms"][0]["correct_pos"] == 1               # the answer moved to position 1
    assert screen.visual_step_result(_score(strong), 0, "sf")["remove"]
    assert not screen.visual_step_result(_score(weak), 0, "sf")["remove"]
    assert not screen.visual_step_result(_score(wrong), 0, "v32")["v32_correct"]
    floor = _score([L.LP_FLOOR] * 3)
    assert floor["argmax"] is None and not screen.visual_step_result(floor, 0, "sf")["sf_correct"]
    text = dict(lp=None, argmax=0, missing=[], top1_token="A", scored="text")
    assert screen.visual_step_result(text, 0, "sf")["remove"] and screen.visual_step_result(text, 0, "sf")["sf_margin"] is None
    assert screen.text_step_result(perms, [text] * 3 + [dict(text, argmax=1)], 0, "opt")["remove"]


def test_option_format():
    assert D.option_format_reasons(["cat", "dog", "All of the above"], 0)[0][0] == "meta_option"
    assert D.option_format_reasons(["cat", "Both A and B", "dog"], 0)[0][0] == "meta_option"
    assert D.option_format_reasons(["cat", "Cannot be determined from the video"], 0)[0][0] == "meta_option"
    assert D.option_format_reasons(["cat", "Cat ", "dog"], 0)[0][0] == "repeated_option_text"
    assert D.option_format_reasons(["", "dog"], 0) == [("empty_gold_option", "")]
    assert D.option_format_reasons(["a cat on all fours", "a dog and a bird"], 0) == []
    assert D.gold_position_share([0, 0, 0, 1]) == 0.75 and D.gold_position_share([]) is None


def _unit(rows):
    v = np.asarray(rows, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_duplicates():
    ids, vids = ["b", "a", "c", "d", "e"], ["v1", "v1", "v1", "v2", "v2"]
    vecs = _unit([[1, 0, 0], [1, 0.01, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
    removed, n_groups, unemb = D.within_benchmark(ids, vids, {k: i for i, k in enumerate(ids)}, vecs)
    assert set(removed) == {"b"} and removed["b"][0] == "a" and n_groups == 1 and not unemb     # smallest id kept
    removed, _, unemb = D.within_benchmark(ids, vids, {"a": 1, "c": 2}, vecs)
    assert removed == {} and unemb == {"b", "d", "e"}
    # across benchmarks: same video and cosine >= 0.90; the benchmark with the earliest year keeps its copy
    ids, ben, vids = ["x1", "y1", "y2", "z1"], ["X", "Y", "Y", "Z"], ["v", "v", "v", "w"]
    vecs = _unit([[1, 0], [1, 0.02], [0, 1], [1, 0]])
    pairs = D.cross_benchmark_pairs(ids, ben, vids, {k: i for i, k in enumerate(ids)}, vecs)
    assert [(a, b) for a, b, _ in pairs] == [("x1", "y1")]
    bench_of = dict(zip(ids, ben))
    assert D.cross_benchmark_drop(pairs, set(ids), bench_of, dict(X=2024, Y=2021)) == {"x1": "y1"}
    assert D.cross_benchmark_drop(pairs, set(ids), bench_of, {}) == {"y1": "x1"}                 # tie: smaller id
    assert D.cross_benchmark_drop(pairs, {"x1", "z1"}, bench_of, {}) == {}                       # y1 left the pool


def _table():
    rng = random.Random(3)
    rows = []
    for i in range(400):
        fmt = "open" if i % 10 == 0 else "yesno" if i % 10 == 1 else "mcq"
        rows.append(dict(benchmark=f"B{i % 4}", item_id=f"i{i:03d}", video_id=f"v{i // 2}", format=fmt,
                         declared_task="t", declared_scene="", video_file="x" if i % 7 else None))
    df = pd.DataFrame(rows)
    mcq = [r["item_id"] for r in rows if r["format"] == "mcq"]
    dup = {i: ("i000", 0.95) for i in mcq[:10]}
    flags = {i: "meta_option" for i in mcq[8:20]}                # two of them are duplicates as well
    steps, alive = {}, [i for i in mcq if i not in dup and i not in flags]
    for name, _, prefix, visual in screen.STEPS:
        steps[name] = {}
        for i in alive:
            if visual and i in ("i399", "i398"):
                continue                                         # no video: the item waits at this step
            steps[name][i] = {"remove": rng.random() < 0.3 and i not in ("i399", "i398"), f"{prefix}_margin": rng.random(),
                              f"{prefix}_correct" if visual else f"{prefix}_acc_4perm": 1}
        alive = [i for i in alive if i in steps[name] and not steps[name][i]["remove"]]
    xdrop = {alive[0]: alive[1], "i399": "i001"}                  # i399 has pending steps: not removed
    return df, dup, flags, steps, xdrop


def test_funnel_accounting():
    df, dup, flags, steps, xdrop = _table()
    out = screen.assemble(df, dup, flags, steps, xdrop)
    rows = screen.funnel(out["removed_by"])
    assert rows[0]["n_after"] == len(df) == 400 and rows[1]["n_removed"] == 80 and rows[1]["n_after"] == 320
    for a, b in zip(rows[1:], rows[2:]):
        assert a["n_after"] == b["n_entering"]
    for r in rows:
        assert r["n_entering"] - r["n_removed"] == r["n_after"], r
    by = {r["stage"]: r for r in rows}
    assert by["duplicate"]["n_removed"] == 10 and by["option_format"]["n_removed"] == 10
    assert by["xbench_duplicate"]["n_removed"] == 1
    assert by["remaining"]["n_after"] == int((out["removed_by"] == "none").sum())
    assert sum(r["n_removed"] for r in rows) + by["remaining"]["n_after"] == 400
    # one reason per item, the first screen in funnel order; items after a removal carry no later reason
    assert set(out["removed_by"]) <= set(screen.REMOVAL_KEYS) | {"none"}
    first = out.set_index("item_id")
    for name in screen.STEP_NAMES:
        for iid, r in steps[name].items():
            if r["remove"]:
                assert first.loc[iid, "removed_by"] == name
    pend = out[(out["removed_by"] == "none") & (out["pending_steps"] != "")]
    assert set(pend["item_id"]) == {"i398", "i399"} and set(pend["pending_steps"]) == {"single_frame,v32,v32_shuffle"}
    per = screen.summary(out)
    assert sum(r["n_remaining"] for r in per) == by["remaining"]["n_after"] and sum(r["n_total"] for r in per) == 400


def test_attack_percentile():
    df = pd.DataFrame(dict(item_id=list("abcd"), opt_margin=[0.1, 0.5, 0.2, None], blind_margin=[0.9, 0.1, 0.2, 0.3],
                           sf_margin=[0.0, 0.1, 0.2, 0.3], v32_margin=[0.0, 0.1, 0.2, 0.3],
                           v32_shuffle_margin=[None] * 4))
    out = attack_percentile(df)
    assert "v32_shuffle_margin_pct" not in out                                  # column without any value
    assert list(out["opt_margin_pct"]) == [0.25, 0.875, 0.5, 0.875]             # missing -> column maximum, ties share
    assert list(out["attack_pct"]) == [1.0, 0.875, 0.75, 1.0]
    assert list(out["attack_margin"]) == [0.9, 0.5, 0.2, 0.5]


def test_scene_clusters():
    rng = np.random.default_rng(1)
    centres = rng.normal(size=(6, 32))
    X = np.concatenate([c + 0.01 * rng.normal(size=(10, 32)) for c in centres])
    groups = np.array(["<10s"] * 30 + ["1-3min"] * 30)
    cal, fix, rows = SC.scene_clusters(groups, X, n_pairs=2000)
    assert len(set(fix[:30])) == 3 and len(set(fix[30:])) == 3 and not set(fix[:30]) & set(fix[30:])
    assert all(len(set(fix[i:i + 10])) == 1 for i in range(0, 60, 10))
    assert [r["group"] for r in rows] == ["<10s", "1-3min"] and rows[0]["clusters_fixed"] == 3
    assert [SC.duration_group(s) for s in (0, 9.9, 10, 59, 60, 600, 1799, 1800, 5000)] == \
        ["<10s", "<10s", "10-30s", "30-60s", "1-3min", "10-30min", "10-30min", ">30min", ">30min"]
    assert SC.template_regex('How many "red" cars pass in 12.5 seconds?') == "how many <ENT> cars pass in <NUM> seconds?"


# ------------------------------------------------------------------ comparison with the research code
def test_matches_research_code():
    ps = research_module("pool_schema", "analysis/pool")
    pc = research_module("pool_common", "analysis/pool")
    ch = research_module("chance")
    px = research_module("pool_screen_extra", "analysis/pool")
    if ps is None or pc is None or ch is None:
        print("    skipped: VIDEO_INDEX_RESEARCH_CODE not set")
        return
    rng = random.Random(0)
    texts = ["cat", "dog", "a red car", "Yes", "No", "All of the above", "12", "bird", "True", "False"]
    golds = ["A", "B", "(C)", "D.", "dog", "A,B", "E", "b", "Yes", "2", "C. bird", None]
    n = 0
    for _ in range(3000):
        k = rng.choice([0, 1, 2, 3, 4, 5])
        opts = rng.sample(texts, k)
        if rng.random() < 0.4:
            opts = [f"{chr(65 + i)}. {o}" for i, o in enumerate(opts)]
        ans = rng.choice(golds)
        assert S.resolve_answer(ans, opts or None) == ps.resolve_answer(ans, opts or None), (ans, opts)
        row = dict(qid="q", question="What is shown?", options=opts or None, answer=ans)
        mine = S.is_mcq_item(row)
        if opts and len(opts) >= 2:          # with an option list the two rules coincide when the answer resolves
            fmt = ps.resolve_answer(ans, opts)[4]
            assert mine == (fmt == "mcq" and ch.is_mcq_item(row, {})), row
        n += 1
    qs = ["What?\nA. x\nB. y\nC. z", "Pick one. (A) x (B) y", "What? Option A: x Option B: y\nAnswer with the letter",
          "A. x\nB. y", "no options here", "What?\nA. x\nC. y", "What?\nA) x\nB) y\nPlease answer."]
    for q in qs:
        assert S.split_inline_options(q) == ps.split_inline_options(q), q
    for _ in range(500):
        top = [dict(token=rng.choice(["A", " A", "B", "(B", "C.", "D", "x", " the", "E:"]), logprob=-rng.random() * 8)
               for _ in range(rng.randint(0, 12))]
        k = rng.randint(2, 6)
        lp, missing = L.letter_logprobs(top, k)
        lp2, missing2, _ = pc.letter_logprobs(top, k)
        assert lp == lp2 and missing == missing2
        assert L.margin_of(lp, 0) == pc.margin_of(lp, 0) and L.argmax_pos(lp) == pc.argmax_pos(lp)
        iid = f"B_{rng.randint(0, 999)}"
        assert L.permutations_for(iid, k) == pc.permutations_for(iid, k)
    assert (L.LOG2, L.LP_FLOOR, L.TOP_LOGPROBS, L.ASSISTANT_PREFIX) == (pc.LOG2, pc.LP_FLOOR, pc.TOP_LOGPROBS, pc.ASSISTANT_PREFIX)
    assert screen.PROMPT == pc.PROMPT and screen.PROMPT_OPT_ONLY == pc.PROMPT_OPT_ONLY
    pp = research_module("pool_prompts", "analysis/pool")
    assert (screen.INTRO_VISUAL, screen.INTRO_BLIND, screen.INTRO_OPT_ONLY) == (pp.INTRO_VISUAL, pp.INTRO_BLIND, pp.INTRO_OPT_ONLY)
    if px is not None:
        assert D.META_OPTION.pattern == px.META_OPTION.pattern and D.COS_DUP == px.COS_DUP
    print(f"    compared {n} option lists, 500 log-probability lists, {len(qs)} questions")


if __name__ == "__main__":
    run_all(globals())
