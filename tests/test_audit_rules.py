"""Frame rules, option permutations and the learned attacker; comparison with the research code."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_audit_common import research_module, run_all  # noqa: E402

from video_index.audit import frame_rules as fr  # noqa: E402
from video_index.audit import options_only_perm as perm  # noqa: E402


def test_reference_rules():
    assert fr.reference_indices("single", 101, 1, "q")[0] == [50]
    idx, _ = fr.reference_indices("uniform", 1000, 32, "q")
    assert len(idx) == 32 and idx[0] == 0 and idx[-1] == 999 and idx == sorted(idx)
    assert fr.reference_indices("uniform", 10, 32, "q")[0] == list(range(10))          # duplicates removed
    a, _ = fr.reference_indices("shuffle", 1000, 32, "q1")
    b, _ = fr.reference_indices("shuffle", 1000, 32, "q2")
    assert a == b and sorted(a) == idx and a != idx                                     # one order for every item
    w, extra = fr.reference_indices("window", 1000, 32, "q1")
    start = random.Random("42|q1").uniform(0.0, 0.9)
    assert extra == dict(window_start=round(start, 4))
    assert w[0] == round(start * 999) and w[-1] == round((start + 0.1) * 999) and len(w) == 32
    assert fr.reference_indices("window", 1000, 32, "q2")[0] != w


def test_attacker_rules():
    ts = [round(i / 2.0, 3) for i in range(400)]
    idx, _ = fr.attacker_indices("uniform", ts, 32, "q")
    assert len(idx) == 32 and idx[0] == 0 and idx[-1] == 399
    assert fr.attacker_indices("single", ts, 1, "q")[0] == [idx[16]]
    sh, _ = fr.attacker_indices("shuffle", ts, 32, "q1")
    assert sorted(sh) == idx and sh != fr.attacker_indices("shuffle", ts, 32, "q2")[0]
    w, extra = fr.attacker_indices("window", ts, 32, "q")
    assert w[0] == 0 and ts[w[-1]] <= ts[-1] / 10 + 0.5 and extra == dict(window_start=0.0)
    assert fr.attacker_indices("uniform", ts[:20], 128, "q") == (list(range(20)), dict(capped=True))
    assert fr.nearest_indices([0.0, 1.0, 2.0], [0.5, 1.5, 1.51, -1, 9]) == [0, 1, 2, 0, 2]   # ties to the earlier frame


def test_permutations():
    row = dict(question="What is shown?", options=["A. cat", "B. dog", "C. bird", "D. fish"], answer="C")
    shape, why = perm.option_shape(row["question"], row["options"], row["answer"])
    assert why is None and shape["labeled"] and shape["gold_idx"] == [2]
    assert perm.permutation("q", 0, 4) == [0, 1, 2, 3]
    p = perm.permutation("q", 3, 4)
    x = list(range(4))
    random.Random("42:q:3").shuffle(x)
    assert p == x
    new = perm.render(shape, p)
    assert [o[:3] for o in new] == ["A. ", "B. ", "C. ", "D. "] and sorted(o[3:] for o in new) == ["bird", "cat", "dog", "fish"]
    a0, a1 = perm.new_answer(shape, p)
    assert a0 == "C" and new[ord(a1) - 65].endswith("bird")
    assert perm.pred_identity(a1, new, "single", perm.mapping_of(p)) == "C"
    assert perm.option_shape("Q?\nA. x\nB. y", ["x", "y"], "A")[1] == "options_in_question"
    assert perm.option_shape("Q?", ["A", "B"], "A")[1] == "options_letter_only"
    assert perm.option_shape("Q?", ["A. x", "y"], "A")[1] == "partial_labels"
    plan, skipped = perm.plan({"q": dict(qid="q", **row), "r": dict(qid="r", question="Q?", options=None, answer="x")})
    assert len(plan) == 9 and skipped == dict(no_options=1)


def test_permutation_statistics():
    samples = {f"q{i}": dict(qid=f"q{i}", question=f"Question {i}?", options=["cat", "dog", "bird", "fish"], answer="ABCD"[i % 4])
               for i in range(40)}
    plan, _ = perm.plan(samples)
    always_gold = {k: dict(pred=pr["answer_new"]) for k, pr in plan.items()}
    runs, st = perm.bench_stats("B", plan, always_gold, "MCQ")
    assert st["status"] == "complete" and st["n"] == 40 and st["width"] == "0.0000" and st["item_consistency"] == "1.0000"
    always_a = {k: dict(pred="A") for k in plan}                        # position bias: the chosen option changes
    runs, st = perm.bench_stats("B", plan, always_a, "MCQ")
    assert float(st["item_consistency"]) < 0.1 and st["c"] == "0.2500"


def test_same_rules_as_the_research_code():
    p1 = research_module("stage_p1_runner")
    ef = research_module("extract_frames")
    if p1 is None or ef is None:
        print("skipped: research code not available")
        return
    rng = random.Random(0)
    for case in range(200):
        n = rng.choice([5, 20, 31, 32, 33, 100, 640, 1024])
        rate = rng.choice([2.0, 1.0, 0.37, 1024 / 3000.0])
        ts = [round(i / rate, 3) for i in range(n)]
        meta, qid = dict(n_frames=n, frame_timestamps=ts), f"item_{case}"
        for kind, cond, k in (("single", "v_1", 1), ("uniform", "v_32", 32), ("uniform", "v_128", 128),
                              ("shuffle", "v_32_shuffle", 32), ("window", "v_32_win10", 32)):
            assert fr.attacker_indices(kind, ts, k, qid)[0] == p1.cond_indices(cond, meta, qid)[0], (case, cond)
        assert fr.reference_indices("uniform", n, 32, qid)[0] == list(dict.fromkeys(ef.sample_indices(n, 32)))
        assert fr.reference_indices("single", n, 1, qid)[0] == ef.sample_indices(n, 1)
    n3 = research_module("n3_common", os.path.join("analysis", "n3"))
    if n3 is not None:
        for case in range(200):
            k = rng.choice([2, 3, 4, 5, 8])
            labeled = rng.random() < 0.5
            opts = [(f"{chr(65 + i)}. " if labeled else "") + f"option text {case} {i}" for i in range(k)]
            ans = chr(65 + rng.randrange(k)) if rng.random() < 0.5 else opts[rng.randrange(k)]
            a, wa = perm.option_shape("What is shown?", opts, ans)
            b, wb = n3.option_shape("What is shown?", opts, ans)
            assert wa == wb and (a is None) == (b is None)
            if a is None:
                continue
            for j in range(9):
                pa, pb = perm.permutation(f"q{case}", j, k), n3.permutation(f"q{case}", j, k)
                assert pa == pb and perm.render(a, pa) == n3.render(b, pb, opts)["options_new"]
                assert perm.new_answer(a, pa) == n3.new_answer(b, pb) and perm.mapping_of(pa) == n3.mapping_of(pb)


def test_same_learner_as_the_research_code():
    ref = research_module("l4_learned", "t3")
    if ref is None:
        print("skipped: research code not available")
        return
    import numpy as np
    from video_index.audit import pool_attack as pa
    rng = np.random.default_rng(0)
    n, dim = 120, 64
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ks = [4] * n
    ys = [int(e[0] > 0) + 2 * int(e[1] > 0) for e in emb]               # gold letter depends on the embedding
    ks[5], ys[5] = None, None
    for seed in (1, 2, 3):
        assert (pa.run_permutation(emb, ks, ys, seed) == ref.run_permutation(emb, ks, ys, seed)).all()
    raw = np.stack([pa.run_permutation(emb, ks, ys, s) for s in (1, 2, 3)]).mean(0)
    a, b = pa.curve_metrics(raw, 0.25), ref.curve_metrics(raw, 0.25)
    assert a["eps_last20"] == b["eps_last20"] and a["ALC"] == b["ALC"] and a["steps"] == b["steps"]
    d = pa.prequential(emb, ks, ys, [1, 2, 3])
    assert d["eps_last20"] == a["eps_last20"] and d["n20"] == 24 and d["ci_lo"] <= d["ci_hi"]


if __name__ == "__main__":
    run_all(globals())
