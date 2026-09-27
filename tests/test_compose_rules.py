"""Rules of the composition: candidates, verification keep rule, hardest-first selection. Reproduction of the released
item list from the research files when VIDEO_INDEX_RESEARCH_CODE is set."""
import json
import os
import random

import pandas as pd

from video_index.compose import CAPS, STAGES
from video_index.compose import candidates as CA
from video_index.compose import items as IT
from video_index.compose import select as SE
from video_index.compose import verify as VE
from video_index.data.schema import write_jsonl
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


def test_stage_registry():
    import importlib
    for name, mod in STAGES.items():
        assert callable(importlib.import_module(mod).run), name


def test_item_records():
    assert IT.strip_prefix(["A. cat", "(B) dog", "C: bird"]) == ["cat", "dog", "bird"]
    assert IT.strip_prefix(["A. cat", "C. dog"]) == ["A. cat", "C. dog"]          # letters out of order: unchanged
    assert IT.clean_stem("Question: What is shown?  Options:") == "What is shown?"
    r = dict(item_id="i", benchmark="B", question="Question: Q?", options=["A. x", "B. y", "C. z", "D. w"], answer_idx=2,
             video_id="v", attack_pct=0.123456, task_class="t")
    rec = IT.from_pool_row(r, "temporal")
    assert rec["options"] == ["x", "y", "z", "w"] and rec["chance"] == 0.25 and rec["attack_pct"] == 0.1235
    assert rec["question"] == "Q?" and rec["capability"] == "temporal"


def test_from_samples():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "B.jsonl")
        write_jsonl(p, [dict(qid="B_0", question="Q", options=["A. x", "B. y"], answer="B", video_id="v0"),
                        dict(qid="B_1", question="Q", options=["x", "y", "z"], answer="z", video_id="v1"),
                        dict(qid="B_2", question="Q", options=["x", "y"], answer="(A) x", video_id="v2"),
                        dict(qid="B_3", question="Q", options=None, answer="free text"),
                        dict(qid="B_4", question="Q", options=["x", "y"], answer="E")])
        items = IT.from_samples(p, "B")
    assert [(i["item_id"], i["answer_idx"], i["options"]) for i in items] == \
        [("B_0", 1, ["x", "y"]), ("B_1", 2, ["x", "y", "z"]), ("B_2", 0, ["x", "y"])]


def _pool(n=200, seed=2):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        rows.append(dict(benchmark=f"B{i % 5}", item_id=f"i{i:03d}", video_id=f"v{i // 2:03d}", question=f"q {i}",
                         options=["a", "b", "c"], answer_idx=i % 3, removed_by="none" if i % 4 else "blind",
                         pending_steps="" if i % 9 else "v32", task_class="t",
                         **{m: rng.gauss(0, 1) for m in MARGINS}))
    rows[1]["question"] = "Select the correct order of the following options"
    rows[2]["options"] = ["only"]
    rows[3]["answer_idx"] = 7
    return pd.DataFrame(rows)


def test_candidates_percentile_ranking():
    pool = _pool()
    rows = CA.hardest(pool, 30)
    ids = [r["item_id"] for r in rows]
    assert len(rows) == 30 and len({r["video_id"] for r in rows}) == 30        # one item per video
    assert not {"i001", "i002", "i003"} & set(ids)
    el = pool[(pool.removed_by == "none") & (pool.pending_steps == "")]
    assert set(ids) <= set(el.item_id)
    # percentile = rank of each margin among the items that passed the screen; the largest of the five is used
    pct = pd.concat([el[m].rank(pct=True) for m in MARGINS], axis=1).max(axis=1)
    want = dict(zip(el.item_id, pct))
    assert all(abs(r["attack_pct"] - want[r["item_id"]]) < 1e-12 for r in rows)
    key = [(r["attack_pct"], r["item_id"]) for r in rows]
    assert key == sorted(key)                                                   # hardest first
    # the hardest eligible item of every video precedes the items that were skipped for their video
    best = el[~el.item_id.isin(["i001", "i002", "i003"])].assign(p=pct).sort_values(["p", "item_id"])
    assert ids == list(best.drop_duplicates("video_id").item_id[:30])


def test_keep_rule():
    k = VE.keep_rule
    assert k("supported", False, 5.0) == (True, "supported")
    assert k("contradicted", True, 5.0) == (True, "reversed_clip")             # reversed clip: the verdict is not used
    assert k("not_verifiable", False, 60.0) == (True, "not_verifiable_long")
    assert k("not_verifiable", False, 59.9) == (False, "not_verifiable_short")
    assert k("not_verifiable", False, None) == (False, "not_verifiable_short")
    assert k("contradicted", False, 500.0) == (False, "contradicted")
    assert k("definition_dependent", False, 500.0) == (False, "definition_dependent")
    assert k(None, False, 500.0) == (False, "no_verdict")
    first = {"a": dict(verdict="not_verifiable", frames=32), "b": dict(verdict="supported", frames=32)}
    again = {"a": dict(verdict="supported", frames=512)}
    assert VE.final_verdict("a", first, again)[0] == "supported" and VE.final_verdict("b", first, again)[1]["frames"] == 32
    ann = VE.annotate([dict(item_id="a", video_id="v"), dict(item_id="c", video_id="w", reversed_clip=True)], first,
                      again, set(), {})
    assert [(x["keep"], x["keep_reason"], x["verdict_frames"]) for x in ann] == \
        [(True, "supported", 512), (True, "reversed_clip", None)]


def test_parse_verdict():
    assert VE.parse('{"verdict": "supported", "visible_answer": "b", "reason": "r"}') == ("supported", "B", "r")
    assert VE.parse('text {"verdict": "Not_Verifiable", "visible_answer": null, "reason": "r"}')[:2] == ("not_verifiable", None)
    assert VE.parse('{"verdict": "contradicted", "visible_answer": "C", "reason": "the fra')[:2] == ("contradicted", "C")
    assert VE.parse('{"verdict": "maybe"}')[0] is None and VE.parse("")[0] is None
    p = VE.build_prompt(dict(question="Q?", options=["x", "y"], answer_idx=1), [0.0, 1.0, 2.5])
    assert "(3 frames; timestamps in seconds):\nt=0.0s, t=1.0s, t=2.5s" in p and "Marked answer: B" in p


def _verified(n=400, seed=4):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append(dict(item_id=f"i{i:03d}", benchmark=f"B{i % 7}", question="q", options=["a", "b"], answer_idx=0,
                        video_id=f"v{i if i % 10 else i + 1:03d}", chance=0.5, attack_pct=round(rng.random(), 4),
                        task_class="t", capability=CAPS[i % 4] if i % 13 else "other", verdict="supported", keep=True,
                        keep_reason="supported", source="candidates"))
    return out


def test_selection_rules():
    ver = _verified()
    items, rep = SE.compose(ver, budget=80)
    assert len(items) == 80 and rep["per_group"] == {c: 20 for c in CAPS} and rep["shortfall"] == 0
    assert len({it["video_id"] for it in items}) == 80                          # one item per video
    assert [it["capability"] for it in items] == [c for c in CAPS for _ in range(20)]
    for c in CAPS:
        got = [it for it in items if it["capability"] == c]
        key = [(it["attack_pct"], it["item_id"]) for it in got]
        assert key == sorted(key)                                               # hardest first inside the group
        # no verified item of the group that was left out is harder than the easiest selected one, unless its video
        # is already used
        used = {it["video_id"] for it in items}
        left = [v for v in ver if v["capability"] == c and v["item_id"] not in {it["item_id"] for it in got}]
        assert all((v["attack_pct"], v["item_id"]) > key[-1] or v["video_id"] in used for v in left)
    assert set(items[0]) == set(SE.FIELDS)
    # items dropped by the verification never enter
    ver2 = [dict(v, keep=(i % 2 == 0)) for i, v in enumerate(ver)]
    items2, _ = SE.compose(ver2, budget=80)
    assert {it["item_id"] for it in items2} <= {v["item_id"] for v in ver2 if v["keep"]}
    # short supply: reported, or passed to the other groups on request
    few = [v for v in ver if v["capability"] != "temporal"] + [v for v in ver if v["capability"] == "temporal"][:5]
    items3, rep3 = SE.compose(few, budget=80)
    assert rep3["per_group"]["temporal"] == 5 and rep3["shortfall"] == 15 and len(items3) == 65
    items4, rep4 = SE.compose(few, budget=80, fill_short_groups=True)
    assert len(items4) == 80 and rep4["per_group"]["temporal"] == 5
    # reference cap: at most 45 % of a group answered correctly by the reference model
    correct = {v["item_id"]: (i % 3 != 0) for i, v in enumerate(ver)}
    items5, rep5 = SE.compose(ver, budget=80, reference_cap=0.45, correct=correct)
    for c in CAPS:
        assert sum(correct[it["item_id"]] for it in items5 if it["capability"] == c) <= 9
    assert len(items5) == 80 and not rep5["cap_relaxed"]


def test_reproduces_released_items():
    root = os.environ.get("VIDEO_INDEX_RESEARCH_CODE")
    m = os.path.join(root or "", "results", "meta")
    if not root or not os.path.exists(os.path.join(m, "items_hard_verified.json")):
        print("    skipped: VIDEO_INDEX_RESEARCH_CODE not set")
        return
    load = lambda name: json.load(open(os.path.join(m, name)))  # noqa: E731
    verified, released = load("items_hard_verified.json"), load("items_census_hard.json")
    items, rep = SE.compose(verified)
    assert items == released and len(items) == 840 and rep["n_benchmarks"] == 76
    # the keep rule on the stored verdicts gives the verified list
    first = {**VE.verdicts(os.path.join(m, "gold_audit_census_hard.jsonl")),
             **VE.verdicts(os.path.join(m, "gold_audit_hard_refill.jsonl"))}
    again = {**VE.verdicts(os.path.join(m, "gold_audit_census_hard_f512.jsonl")),
             **VE.verdicts(os.path.join(m, "gold_audit_hard_refill_f512.jsonl"))}
    rev, dur = set(load("reversed_clip_items.json")), load("video_durations_hard.json")
    ann = (VE.annotate(load("items_census_hard_v4.json"), first, again, rev, dur, source="census_v4")
           + VE.annotate(load("items_hard_refill.json"), first, again, rev, dur, source="refill"))
    assert [a for a in ann if a["keep"]] == verified
    # second pass of the study: the items whose 32-frame verdict was not_verifiable or contradicted
    scope = {i for i, r in first.items() if r["verdict"] in VE.RECHECK}
    assert len(set(again) - scope) == 0 and len(set(again) & scope) / len(again) == 1.0
    print(f"    {len(items)} items identical to the released list; {len(verified)} verified candidates reproduced")


if __name__ == "__main__":
    run_all(globals())
