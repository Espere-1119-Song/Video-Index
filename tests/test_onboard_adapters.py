"""Adapter tests on small annotation excerpts: every shape the generic adapter has to read."""
import json
import os

import pytest

from video_index.onboard.adapters import (AdapterError, AdapterSpec, build_items, normalize_options, resolve_answer,
                                          split_inline_options, to_item, yes_no_options)
from video_index.onboard.adapters.generic import render


def write(tmp_path, name, rows, nested=None):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if name.endswith(".jsonl"):
        p.write_text("\n".join(json.dumps(r) for r in rows))
    else:
        p.write_text(json.dumps(nested if nested is not None else rows))
    return str(tmp_path)


def test_dict_options_keep_their_texts():
    # list(dict) keeps only the keys: the options must come out as texts, in key order
    assert normalize_options({"A": "cat", "B": "dog", "C": None, "D": ""}) == ["cat", "dog"]
    assert normalize_options({"option_1": "second", "option_0": "first"}) == ["first", "second"]
    assert normalize_options({"choice_b": "y", "choice_a": "x"}) == ["x", "y"]
    assert normalize_options({"A": "", "B": None}) is None
    it = to_item("B", 0, dict(v="v.mp4", q="What?", a="B", opts={"A": "cat", "B": "dog"}, sub="s"))
    assert it["options"] == ["cat", "dog"] and it["answer"] == "B" and it["answer_idx"] == 1 and it["format"] == "mcq"


def test_string_options_are_parsed():
    assert normalize_options("['Hold', 'Read', 'Tap', 'Move']") == ["Hold", "Read", "Tap", "Move"]
    assert normalize_options('["a", "b"]') == ["a", "b"]


def test_letter_prefixes_are_stripped_only_when_sequential():
    it = to_item("B", 1, dict(v="v", q="Q?", a="(B)", opts=["A. one", "B. two  ", "C. three"], sub=None))
    assert it["options"] == ["one", "two", "three"] and it["answer"] == "B" and it["subtask"] == "_none"
    it = to_item("B", 2, dict(v="v", q="Q?", a="B", opts=["B. one", "A. two"], sub=None))
    assert it["options"] == ["B. one", "A. two"] and it["answer_idx"] == 1


@pytest.mark.parametrize("gold,idx", [("C", 2), ("c.", 2), ("C. bird", 2), ("bird", 2), ("2", 2), ("(A)", 0)])
def test_gold_as_letter_text_or_index(gold, idx):
    texts, n, got, fmt, reason = resolve_answer(gold, ["cat", "dog", "bird"])
    assert (got, fmt, n) == (idx, "mcq", 3), reason


@pytest.mark.parametrize("gold,reason", [("fish", "answer_not_in_options"), ("E", "letter_out_of_range"), ("A,C", "multi_select")])
def test_gold_that_does_not_resolve_is_open(gold, reason):
    *_, fmt, why = resolve_answer(gold, ["cat", "dog", "bird"])
    assert fmt == "open" and why == reason


def test_options_inside_the_question():
    q = "What does the man pick up?\nA. a cup\nB. a book\nC. a phone\nAnswer with the option's letter."
    stem, opts = split_inline_options(q)
    assert stem == "What does the man pick up?" and opts == ["a cup", "a book", "a phone"]
    it = to_item("B", 3, dict(v="v", q=q, a="B", opts=None, sub="x"))
    assert it["format"] == "mcq" and it["options_source"] == "question" and it["question"] == stem and it["answer"] == "B"
    assert split_inline_options("Is option (A) or (B) better here?") is None           # markers inside prose
    assert split_inline_options("A. starts with a marker\nB. no stem") is None
    # a numeric gold is not read as an index when the options come from the question text
    assert to_item("B", 4, dict(v="v", q=q, a="1", opts=None, sub="x"))["format"] == "open"


def test_yes_no_items_are_not_multiple_choice():
    assert yes_no_options(["A. Yes", "B. No"]) and yes_no_options(["true", "false"]) and not yes_no_options(["yes", "maybe"])
    it = to_item("B", 5, dict(v="v", q="Is the video tampered?", a="No", opts=["Yes", "No"], sub="x"))
    assert it["format"] == "yesno" and it["format_reason"] == "yes_no_options"
    it = to_item("B", 6, dict(v="v", q="Is the door open?", a="yes", opts=None, sub="x"))
    assert it["format"] == "yesno" and it["options"] is None
    it = to_item("B", 7, dict(v="v", q="What happens next?", a="He leaves the room.", opts=None, sub="x"))
    assert it["format"] == "open"


def test_templates():
    row = {"id": 7.0, "metadata": {"vid": "abc", "start": 12.0, "end": 47.5}, "path": "./clips/x.mp4", "refs": ["a.mp4", "b.mp4"],
           "__file": "scene1/ann.json"}
    assert render("{id|int}.mp4", row) == "7.mp4"
    assert render("{metadata.vid}|{metadata.start|sec}-{metadata.end|sec}", row) == "abc|12-47.5"
    assert render("{path|strip:./}", row) == "clips/x.mp4" and render("{path|stem}", row) == "x"
    assert render("{refs|first}", row) == "a.mp4" and render("{__file|dir0}/{id|int}.mp4", row) == "scene1/7.mp4"
    assert render("{missing}.mp4", row) is None and render("metadata.vid", row) == "abc"


def test_detection_without_a_spec(tmp_path):
    d = write(tmp_path, "test.jsonl", [
        {"video_path": "a.mp4", "Question": "Q1?", "candidates": ["x", "y", "z"], "answer": "y", "task_type": "count"},
        {"video_path": "b.mp4", "Question": "Q2?", "candidates": ["x", "y", "z"], "answer": "A", "task_type": "order"},
        {"video_path": None, "Question": "no video", "candidates": ["x", "y"], "answer": "A", "task_type": "order"}])
    items, info = build_items("Tiny", d)
    assert [i["qid"] for i in items] == ["Tiny_0", "Tiny_1"] and info["n_dropped"] == 1
    assert info["columns"] == dict(video="video_path", question="Question", answer="answer", options="candidates", subtask="task_type")
    assert [i["answer"] for i in items] == ["B", "A"] and items[0]["subtask"] == "count"


def test_option_columns_and_index_answers(tmp_path):
    d = write(tmp_path, "MC/val.jsonl", [{"video": "1", "question": "q", "a0": "p", "a1": "q", "a2": "r", "a3": "s", "a4": "t",
                                          "answer": 3, "type": "CW"}])
    write(tmp_path, "OE/val.jsonl", [{"video": "1", "question": "open q", "answer": "free text", "type": "CW"}])
    spec = AdapterSpec.from_dict(dict(name="N", files=["MC/*"], answer_type="index0",
                                      fields=dict(option_fields=["a0", "a1", "a2", "a3", "a4"], subtask="type", video="{video}.mp4")))
    items, _ = build_items("N", d, spec)
    assert len(items) == 1 and items[0]["answer"] == "D" and items[0]["options"] == list("pqrst") and items[0]["video_ref"] == "1.mp4"


def test_filters_keep_and_dedupe_preserve_numbering(tmp_path):
    rows = [{"video": "a.mp4", "question": "q1", "choices": {"A": "x", "B": "y"}, "answer": "A", "src": "yt"},
            {"video": "b.mp4", "question": "q2", "choices": {"A": "x", "B": "y"}, "answer": "B", "src": "licensed"},
            {"video": "c.mp4", "question": "q3", "choices": {"A": "", "B": ""}, "answer": "free text", "src": "yt"},
            {"video": "d.mp4", "question": "q4", "choices": {"A": "x", "B": "y"}, "answer": "B", "src": "yt"},
            {"video": "d.mp4", "question": "q4", "choices": {"A": "x", "B": "y"}, "answer": "B", "src": "yt"}]
    d = write(tmp_path, "ann.json", rows)
    spec = AdapterSpec.from_dict(dict(name="F", fields=dict(options="choices"), keep="mcq", dedupe=True,
                                      filters=[dict(field="src", **{"in": ["yt"]})]))
    items, info = build_items("F", d, spec)
    # the filtered row is dropped before numbering, the open row keeps its number but gets no item
    assert [i["qid"] for i in items] == ["F_0", "F_2"] and info["n_filtered"] == 1 and info["n_records"] == 3


def test_nested_records(tmp_path):
    nested = {"data": {"counting": [{"video": "a.mp4", "question": "q", "choices": ["1", "2"], "answer": "B"}],
                       "order": [{"video": "b.mp4", "question": "q", "choices": ["x", "y"], "answer": "A"}]}}
    d = write(tmp_path, "qa.json", None, nested=nested)
    spec = AdapterSpec.from_dict(dict(name="V", records="data.*.*", fields=dict(options="choices", subtask="__key")))
    items, _ = build_items("V", d, spec)
    assert [(i["subtask"], i["answer"]) for i in items] == [("counting", "B"), ("order", "A")]


def test_python_hook(tmp_path):
    hook = tmp_path / "my_adapter.py"
    hook.write_text("def load(d):\n    return [dict(v=('videos.zip', 'clips/a.mp4'), q='Q?', a='A', opts=['x', 'y'], sub='s')]\n")
    spec = AdapterSpec.from_dict(dict(name="H", hook=f"{hook}:load"))
    items, info = build_items("H", str(tmp_path), spec)
    assert info["via"] == "hook" and items[0]["video_ref"] == dict(archive="videos.zip", member="clips/a.mp4")


def test_errors_say_what_is_missing(tmp_path):
    d = write(tmp_path, "x.jsonl", [{"foo": 1, "bar": 2}])
    with pytest.raises(AdapterError, match="field detection failed"):
        build_items("E", d)
    with pytest.raises(AdapterError, match="unknown keys"):
        AdapterSpec.from_dict(dict(name="E", colums={}))
    with pytest.raises(AdapterError, match="no parseable annotation"):
        build_items("E", str(tmp_path / "empty"))


def test_shipped_specs_load():
    root = os.path.join(os.path.dirname(__file__), "..", "configs", "adapters")
    names = [AdapterSpec.from_yaml(os.path.join(root, f)).name for f in sorted(os.listdir(root)) if f.endswith(".yaml")]
    assert {"EgoPlan-Bench", "CG-Bench", "PLM-VideoBench", "NExT-QA", "LongVideoBench", "Lemonade"} <= set(names)
