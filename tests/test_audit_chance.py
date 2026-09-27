"""Chance level and the multiple-choice item rule; comparison with the research code on the same samples."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_audit_common import research_module, run_all  # noqa: E402

from video_index.audit import chance, items  # noqa: E402


def samples_mixed(seed=0, n=120):
    rng = random.Random(seed)
    out = {}
    for i in range(n):
        kind = i % 6
        q = f"What happens in clip {i}?"
        if kind == 0:
            r = dict(question=q, options=["red", "green", "blue", "black"], answer="ABCD"[rng.randrange(4)])
        elif kind == 1:
            r = dict(question=q, options=["A. one", "B. two", "C. three"], answer=rng.choice(["A", "B. two", "(C)"]))
        elif kind == 2:
            r = dict(question=q + "\nA. up\nB. down\nC. left\nD. right\nE. none", options=None, answer="ABCDE"[rng.randrange(5)])
        elif kind == 3:
            r = dict(question=q, options=["Yes", "No"], answer=rng.choice(["Yes", "No"]))
        elif kind == 4:
            r = dict(question=q, options=None, answer=rng.choice(["yes", "no"]))
        else:
            r = dict(question=q, options=None, answer=f"a person opens door {i}")
        out[f"S_{i}"] = dict(qid=f"S_{i}", **r)
    return out


def samples_letter_gold(seed=1, n=90):
    rng = random.Random(seed)
    return {f"L_{i}": dict(qid=f"L_{i}", question=f"Which option fits clip {i}?", options=None,
                           answer=rng.choice(["A", "B", "C", "D", "(B)", "C. text"])) for i in range(n)}


METAS = [{}, dict(question_format="MCQ", num_options="4"), dict(question_format="MCQ", num_options="4 or 5"),
         dict(question_format="mixed", num_options=""), dict(question_format="open", num_options="")]


def test_item_sources():
    s = samples_mixed()
    d = chance.bench_chance(s, {})
    assert d["csrc"] == {"list": 60, "question": 20, "yesno": 20, "open": 20}
    assert d["n_mcq"] == 60 and d["n_excluded"] == 60           # list items with a yes/no pair are not multiple choice
    assert abs(d["c"] - ((20 * 0.25 + 20 / 3 + 20 * 0.2) / 60)) < 1e-12
    assert d["cvec"]["S_3"] == 0.5 and d["cvec"]["S_5"] == 0.0


def test_declared_and_inferred():
    s = samples_letter_gold()
    d = chance.bench_chance(s, dict(question_format="MCQ", num_options="4 or 5"))
    assert set(d["csrc"]) == {"declared"} and abs(d["c"] - (0.25 + 0.2) / 2) < 1e-12
    d = chance.bench_chance(s, dict(question_format="mixed", num_options=""))
    assert d["k_inf"] == 4 and set(d["csrc"]) == {"inferred"} and abs(d["c"] - 0.25) < 1e-12 and d["n_mcq"] == 90
    assert chance.declared_counts(dict(num_options="4 or 6")) == [4, 6]


def test_embedded_options():
    assert items.embedded_options("Pick one.\n(A) cat\n(B) dog\n(C) bird") == ["(A) cat", "(B) dog", "(C) bird"]
    assert items.embedded_options("Which is it? A. cat B. dog C. bird.") == ["A. cat", "B. dog", "C. bird"]
    assert items.embedded_options("Choose: Option A: left. Option B: right.") == ["A. left", "B. right"]
    assert items.embedded_options("A. cat B. dog") is None          # no question mark or colon before the first marker
    assert items.embedded_options("What colour is the car?") is None
    assert items.lettered(["cat", "dog"]) == "A. cat\nB. dog"
    assert items.lettered(["A. cat", "B. dog"]) == "A. cat\nB. dog"


def test_same_values_as_the_research_code():
    ref = research_module("chance")
    if ref is None:
        print("skipped: research code not available")
        return
    sets = [samples_mixed(0), samples_mixed(7, 61), samples_letter_gold(), {**samples_mixed(3), **samples_letter_gold(4)}]
    d = os.environ.get("VIDEO_INDEX_RESEARCH_SAMPLES")
    names = []
    if d and os.path.isdir(d):
        import json
        for f in sorted(os.listdir(d))[:60]:
            if f.endswith(".jsonl"):
                rows = [json.loads(l) for l in open(os.path.join(d, f)) if l.strip()]
                sets.append({r["qid"]: r for r in rows})
                names.append(f[:-6])
    table = {}
    if names and hasattr(ref, "load_bench_meta"):
        table = ref.load_bench_meta()
    n = 0
    for i, s in enumerate(sets):
        metas = METAS if i < 4 else [ref.bench_meta(table, names[i - 4]) if table else {}]
        for m in metas:
            a, b = chance.bench_chance(s, m), ref.bench_chance(s, m)
            for k in ("c", "cvec", "c_source", "k_inf", "n_items", "n_mcq", "n_excluded", "n_open", "n_yesno", "digit_index",
                      "pos", "c_all"):
                assert a[k] == b[k], (i, m, k)
            assert a["mcq_qids"] == b["mcq_qids"]
            n += 1
    print(f"    compared {n} (sample, metadata) pairs")


if __name__ == "__main__":
    run_all(globals())
