"""Rules of the error attribution: answer extraction, pre-classification, agreement, row classification, shares.
Comparison with the research code on the same inputs when VIDEO_INDEX_RESEARCH_CODE is set."""
import glob
import json
import os
import random
import sys

from video_index.audit.attribution import STAGES, checks, common as C, judge, preclass, tables, taxonomy as T


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


def test_taxonomy():
    assert len(T.CATEGORIES) == 30 and len(set(T.CATEGORIES)) == 30
    assert all(c in T.GROUP for c in T.CATEGORIES)
    assert set(T.GROUP.values()) == set(T.CAPABILITY_GROUPS) | {"data_protocol"}
    assert "language_prior_dominated" not in T.CATEGORIES and T.GROUP["language_prior_dominated"] == "reasoning_knowledge"
    for c in T.CATEGORIES:
        if c not in ("evidence_not_in_input_frames", "other"):
            assert c in T.FINE_GROUP, c


def test_prompt_wording():
    # the wording that does not trigger refusals: no request for a "visible reply", no "do not keep ... hidden"
    assert "visible reply" not in C.COT_PROMPT and "hidden" not in C.COT_PROMPT
    assert C.COT_PROMPT.rstrip().endswith("Final answer: <option letter, or the exact short answer if no options>")
    p = judge.MAIN_PROMPT.format(n=32, q="Q", opts="A. x\nB. y", ans="A", pred="B", trace="t", aux="(none available)",
                                 vie=judge.VIE_NOTE, cats=judge.cat_lines())
    assert p.count("\n- ") == 30 and "vision-introduced error" in p
    assert "{" not in judge.DENSE_PROMPT.format(n=128, q="Q", opts="", ans="A", pred="B").replace('{"evidence', "")


def test_extract_final_answer():
    assert C.extract_final_answer("reasoning...\nFinal answer: B") == "B"
    assert C.extract_final_answer("x\n**Final answer:**\n\n**C. a cat**\n") == "C. a cat"
    assert C.extract_final_answer("Final answer: A\nmore\nfinal answer - D") == "D"
    assert C.extract_final_answer("no marker here\nlast line") == "last line"
    assert C.extract_final_answer("") == "" and C.extract_final_answer(None) == ""


def test_same_wrong_answer():
    opts = ["A. cat", "B. dog", "C. bird"]
    assert C.same_wrong_answer("B", "(B) dog", opts)
    assert not C.same_wrong_answer("B", "C", opts)
    assert not C.same_wrong_answer("A man walks", "A dog runs", None)     # no letter path without options
    assert C.same_wrong_answer("3", "3.0", None) and not C.same_wrong_answer("3", "4", None)
    assert C.same_wrong_answer("The Cat", "the cat", None)


def test_gold_contained():
    assert C.gold_contained("it is a red car", "red car", None)
    assert not C.gold_contained("it is a red car", "red car", ["A. x", "B. y"])
    assert not C.gold_contained("a b c d e", "a b c d e", None)           # gold longer than four tokens


def test_parse_json_obj():
    assert C.parse_json_obj('```json\n{"category": "other"}\n```') == {"category": "other"}
    assert C.parse_json_obj('Here: {"a": 1} done') == {"a": 1}
    assert C.parse_json_obj("[1, 2]") is None and C.parse_json_obj("") is None


def test_judge_parsers_and_consensus():
    assert judge.parse_main('{"category": "ocr_text", "reason": "r", "free_desc": "f"}')["category"] == "ocr_text"
    assert judge.parse_main('{"category": "not_a_category"}') is None
    assert judge.parse_dense('{"evidence_visible": true, "resolution": "sampling_gap", "reason": "r"}')["evidence_visible"]
    assert judge.parse_dense('{"resolution": "ocr_text"}') is None
    assert judge.consensus(["ocr_text", "ocr_text"]) == ("ocr_text", "perception", "fine")
    assert judge.consensus(["ocr_text", "perception_attribute"]) == (None, "perception", "group")
    assert judge.consensus(["ocr_text", "temporal_order"]) == (None, None, "disputed")
    assert judge.dense_consensus(["sampling_gap", "sampling_gap"], 128, {})["final"] == "sampling_gap"
    d = judge.dense_consensus(["sampling_gap", "audio_needed"], 128, {})
    assert d["disputed"] and d["final"] is None


def test_preclassify():
    srow = dict(qid="q", question="Q", options=["cat", "dog", "bird"], answer="A")
    tr = dict(qid="q", model="m", pred_cot="B", orig_pred="B", still_wrong=True)
    r1 = preclass.preclassify(tr, srow, "MCQ", dict(pred="B. dog", answer="A"))
    assert r1["preclass"] == "language_prior_dominated" and r1["to_judge"] is False and r1["blind_correct"] is False
    r2 = preclass.preclassify(tr, srow, "MCQ", dict(pred="A", answer="A"))
    assert r2["vision_introduced_error"] and r2["to_judge"] and r2["preclass"] == "none"
    r3 = preclass.preclassify(tr, srow, "MCQ", dict(pred="C", answer="A"))
    assert r3["to_judge"] and not r3["vision_introduced_error"] and r3["preclass"] == "none"
    r4 = preclass.preclassify(tr, srow, "MCQ", None)
    assert r4["blind_pred"] is None and r4["to_judge"]
    r5 = preclass.preclassify(dict(tr, pred_cot="I am not sure", orig_pred="C"), srow, "MCQ", dict(pred="C", answer="A"))
    assert r5["letter_source"] == "orig" and r5["preclass"] == "language_prior_dominated"


def _row(qid, level, category=None, group=None, dense=None, cats=("ocr_text", "ocr_text")):
    return dict(qid=qid, model="m", agreed_level=level, category=category, group=group, dense_recheck=dense,
                by={f"judge_{x}": dict(category=c) for x, c in zip("ab", cats)})


ROWS = [
    _row("1", "fine", "ocr_text", "perception"),
    _row("2", "rule", "language_prior_dominated", "reasoning_knowledge", cats=()),
    _row("3", "group", None, "temporal", cats=("temporal_order", "speed_duration")),
    _row("4", "disputed", None, None, cats=("ocr_text", "temporal_order")),
    _row("5", "fine", "other", "data_protocol", cats=("other", "other")),
    _row("6", "fine", "sampling_gap", "data_protocol", dict(triggered=True, disputed=False, final="sampling_gap"),
         cats=("evidence_not_in_input_frames",) * 2),
    _row("7", "fine", "evidence_not_in_input_frames", "data_protocol", dict(triggered=True, disputed=True, final=None),
         cats=("evidence_not_in_input_frames",) * 2),
    _row("8", "fine", "evidence_not_in_input_frames", "data_protocol", dict(triggered=False, skipped="short"),
         cats=("evidence_not_in_input_frames",) * 2),
    _row("9", "group", None, "data_protocol", cats=("annotation_suspect", "question_ambiguous")),
    _row("10", "group", None, "data_protocol", cats=("annotation_suspect", "timestamp_frame_anchoring")),
    _row("11", "fine", "annotation_suspect", "data_protocol", cats=("annotation_suspect",) * 2),
    _row("12", "fine", "spatial_relation", "spatial_physical", cats=("spatial_relation",) * 2),
]


def test_classify_and_shares():
    unstable = {("B", "m", "12")}
    labs = [tables.classify_row(r, unstable, "B") for r in ROWS]
    got = [(l["status"], l["group"], l["subgroup"]) for l in labs]
    assert got == [("ok", "capability", "perception"), ("ok", "capability", "reasoning_knowledge"),
                   ("ok", "capability", "temporal"), ("disputed", "", ""), ("other", "", ""),
                   ("ok", "coverage", "coverage"), ("unverified", "", ""), ("unverified", "", ""),
                   ("ok", "annotation", "annotation"), ("disputed", "", ""), ("ok", "annotation", "annotation"),
                   ("unstable", "", "")]
    assert labs[9]["basis"].startswith("group_split") and labs[8]["basis"].startswith("group_both")
    c = tables.count_labels(labs)
    assert c["n_errors"] == 12 and c["n_unstable"] == 1 and c["n_disputed"] == 2 and c["n_disputed_group_split"] == 1
    assert c["n_unverified"] == 2 and c["n_unverified_skipped"] == 1 and c["n_unverified_dense_disputed"] == 1
    assert c["n_other"] == 1
    assert c["n_denominator"] == c["n_errors"] - c["n_unstable"] - c["n_disputed"] - c["n_unverified"] - c["n_other"] == 6
    assert c["n_denominator"] == c["n_annotation_group"] + c["n_coverage"] + c["n_capability"]
    s = tables.shares(c)
    assert s["capability_share"] == "0.5000" and s["annotation_share"] == "0.3333" and s["coverage_share"] == "0.1667"
    assert tables.shares(tables.count_labels([]))["capability_share"] == "NA"


def test_fine_tables_reconcile():
    labs = [dict(benchmark="B", qid=r["qid"], model="m", agreed_level=r["agreed_level"],
                 **tables.classify_row(r, set(), "B")) for r in ROWS]
    rows_b, rows_p = tables.fine_tables(labs)
    c = tables.count_labels(labs)
    assert sum(r["count"] for r in rows_p) == c["n_capability"] == sum(r["count"] for r in rows_b)
    undet = [r for r in rows_p if r["fine_category"].endswith("/undetermined")]
    assert len(undet) == 4 and sum(r["count"] for r in undet) == 1          # the group-level temporal row
    one = next(r for r in rows_p if r["fine_category"] == "ocr_text")
    assert one["count"] == 1 and one["share_in_subgroup"] == "1.0000"


def test_item_label():
    a = dict(status="ok", group="capability", subgroup="perception", basis="fine:ocr_text")
    b = dict(status="ok", group="annotation", subgroup="annotation", basis="fine:annotation_suspect")
    d = dict(status="disputed", group="", subgroup="", basis="agreed_level:disputed")
    assert tables.item_label([a, b])["subgroup"] == "annotation"            # tie: annotation before capability
    assert tables.item_label([a, a, b])["subgroup"] == "perception"
    assert tables.item_label([d, d])["status"] == "disputed"


def test_sample_and_frame_rules():
    rows = [dict(model="m", qid=str(i)) for i in range(40)]
    a, b = checks.pick_sample(rows), checks.pick_sample(list(reversed(rows)))
    assert [r["qid"] for r in a] == [r["qid"] for r in b] and len(a) == 4
    assert len(checks.pick_sample(rows[:3])) == 1 and len(checks.pick_sample(rows, n=10)) == 10
    assert C.uniform_indices(1024, 32)[0] == 0 and C.uniform_indices(1024, 32)[-1] == 1023
    off = C.offset_indices(1024, 32)
    assert off[0] == 16 and off[-1] == 1023 and len(off) == 32 and off != C.uniform_indices(1024, 32)
    assert C.offset_indices(5, 32) == C.uniform_indices(5, 32) == [0, 1, 2, 3, 4]
    assert C.frame_check("h", 1024) == C.frame_check("h", 1024) != C.frame_check("h", 1023)
    assert C.frame_check("abcdef", 0).startswith("nframes_unknown:")


# ------------------------------------------------------------------ comparison with the research code
def _dedup(idx):
    out = []
    for i in idx:
        if i not in out:
            out.append(i)
    return out


def test_matches_research_code():
    qc = research_module("q_common")
    if qc is None:
        print("    skipped: VIDEO_INDEX_RESEARCH_CODE not set")
        return
    assert T.CATEGORIES == qc.CATEGORIES_V2 and T.GROUP == qc.GROUP_V2
    assert T.DENSE_RESOLUTIONS == qc.DENSE_RESOLUTIONS and C.COT_PROMPT == qc.COT_PROMPT
    q3, q7 = research_module("q3_judge"), research_module("q7_checks")
    assert (judge.MAIN_PROMPT, judge.DENSE_PROMPT, judge.VIE_NOTE) == (q3.MAIN_PROMPT, q3.DENSE_PROMPT, q3.VIE_NOTE)
    rng = random.Random(0)
    texts = ["Final answer: B", "**Final answer:** C. x", "x\ny\nz", "final answer - 12", "", "Final Answer:\n\n`A`"]
    for t in texts:
        assert C.extract_final_answer(t) == qc.extract_final_answer(t)
    opts = ["A. cat", "B. dog", "C. bird", "D. fish"]
    preds = ["A", "(B)", "C. bird", "the dog", "3", "3.0", "a man", "A man", "fish", ""]
    for _ in range(300):
        a, b, o = rng.choice(preds), rng.choice(preds), rng.choice([opts, None])
        assert C.same_wrong_answer(a, b, o) == qc.same_wrong_answer(a, b, o), (a, b, o)
    for n in [1, 5, 31, 32, 33, 64, 100, 777, 1024]:
        assert C.uniform_indices(n, 32) == _dedup(qc.uniform_indices(n, 32)), n
        assert C.offset_indices(n, 32) == _dedup(q7.window_indices(n)), n
    assert [t["qid"] for t in checks.pick_sample([dict(model="m", qid=f"q{i}") for i in range(57)])] == \
        [t["qid"] for t in rng.__class__(qc.SEED).sample(sorted([dict(model="m", qid=f"q{i}") for i in range(57)],
                                                                key=lambda t: (t["model"], t["qid"])), 6)]


def test_row_classification_matches_research_code():
    sc = research_module("screen_common", "analysis/screen")
    a1 = research_module("a1_shares", "analysis/screen")
    root = os.environ.get("VIDEO_INDEX_RESEARCH_CODE")
    if sc is None or a1 is None:
        print("    skipped: VIDEO_INDEX_RESEARCH_CODE not set")
        return
    files = sorted(glob.glob(os.path.join(root, "results", "q_attrib", "*.jsonl")))[:40]
    n, mine, theirs = 0, [], []
    for p in files:
        bench = os.path.basename(p)[:-6]
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            a, b = tables.classify_row(r, set(), bench), sc.classify_row(r, set(), bench)
            assert (a["status"], a["group"], a["subgroup"]) == (b["status"], b["group"], b["subgroup"]), (bench, r["qid"])
            assert a["basis"] == b["basis"] or a["basis"].startswith("check:")
            mine.append(a)
            theirs.append(b)
            n += 1
    ca, cb = tables.count_labels(mine), a1.count_labels(theirs)
    for k in cb:
        assert ca[k.replace("n_screen", "n_annotation_group")] == cb[k], k
    print(f"    compared {n} attributed rows of {len(files)} benchmarks")


if __name__ == "__main__":
    run_all(globals())
