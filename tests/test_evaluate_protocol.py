"""Protocol of the evaluation: prompts, option permutations, frame rule, and the agreement of the scorer shipped with
the toolkit integrations with ``video_index.scoring``."""
import importlib.util
import os
import random

import pytest

from video_index.evaluate import protocol as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEM = {"item_id": "Demo_1", "question": "What does the person pick up?", "options": ["a red cup", "a knife", "the phone"],
        "answer_idx": 2}


def _integration_helper():
    path = os.path.join(ROOT, "integrations", "vlmevalkit", "vlmeval", "dataset", "utils", "video_index.py")
    if not os.path.exists(path):
        pytest.skip("integration files not present")
    spec = importlib.util.spec_from_file_location("vi_integration_helper", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prompts():
    assert P.video_prompt(ITEM, 12) == (
        "You are given 12 frame(s) sampled from a video. Answer the question based on these frames.\n\n"
        "Question: What does the person pick up?\nA. a red cup\nB. a knife\nC. the phone\n"
        "Reply with ONLY the option letter (or the exact short answer if no options).")
    assert P.blind_prompt(ITEM, [2, 0, 1]).startswith("You are given NO frames from the video.")
    assert "A. the phone\nB. a red cup\nC. a knife" in P.blind_prompt(ITEM, [2, 0, 1])


def test_permutations_are_fixed_per_item():
    perms = P.permutations_for("Demo_1", 4)
    rng = random.Random("42|Demo_1")
    assert perms == [rng.sample(range(4), 4) for _ in range(4)]
    assert perms == P.permutations_for("Demo_1", 4)
    assert all(sorted(p) == [0, 1, 2, 3] for p in perms)
    assert P.permutations_for("Demo_2", 4) != perms


def test_gold_letter_follows_the_permutation():
    assert P.gold_letter(ITEM, [0, 1, 2]) == "C"
    assert P.gold_letter(ITEM, [2, 0, 1]) == "A"
    assert P.score_reply("A", ITEM, [2, 0, 1]) is True
    assert P.score_reply("C", ITEM, [2, 0, 1]) is False
    assert P.score_reply(None, ITEM, [2, 0, 1]) is None


def test_one_fps_indices():
    # stored at 2 fps, 20 s: one frame per second
    assert P.one_fps_indices(40, 2.0, 20.0) == list(range(0, 40, 2))
    # stored below the rate: every stored frame
    assert P.one_fps_indices(10, 0.5, 20.0) == list(range(10))
    # cap: uniform thinning, first and last kept
    idx = P.one_fps_indices(2000, 2.0, 1000.0, cap=512)
    assert len(idx) == 512 and idx[0] == 0 and idx[-1] == 1998
    idx = P.one_fps_indices(40, 2.0, 20.0, cap=8)
    assert len(idx) == 8 and idx == sorted(set(idx))
    # the source duration decides the count when the stored file is slightly shorter
    assert len(P.one_fps_indices(39, 2.0, 20.0)) == 20


REPLIES = ["A", "b", "(C)", "B.", "C) the phone", "[a]", "The answer is B", "Answer: C.", "option a", "the phone",
           "A red cup", "a knife", "I think it is the phone because the person calls someone", "A person", "I cannot tell",
           "", "  ", "**B**", "答案是C", "（B）", "2", "7", "B and C", "The answer is A. Wait, the answer is C", "D", "phone",
           "It is a red cup or a knife", "c: the phone", "None of the above"]


@pytest.mark.parametrize("perm", [[0, 1, 2], [2, 0, 1], [1, 2, 0]])
def test_integration_scorer_agrees_with_the_library(perm):
    helper = _integration_helper()
    texts = [ITEM["options"][j] for j in perm]
    gold = P.gold_letter(ITEM, perm)
    for reply in REPLIES:
        assert bool(helper.score_reply(reply, gold, texts)) == bool(P.score_reply(reply, ITEM, perm)), (reply, perm)


def test_integration_helper_matches_the_protocol():
    helper = _integration_helper()
    assert helper.PROMPT == P.PROMPT and helper.INTRO_FRAMES == P.INTRO_VISUAL and helper.INTRO_BLIND == P.INTRO_BLIND
    assert helper.permutations_for("Demo_1", 5) == P.permutations_for("Demo_1", 5)
    for args in [(40, 2.0, 20.0), (1024, 2.0, 2801.7), (39, 2.0, 20.0), (7, 29.97, 0.2), (300, 30.0, 10.0)]:
        for cap in (8, 32, 512):
            assert helper.one_fps_indices(*args, rate=1.0, cap=cap) == P.one_fps_indices(*args, 1.0, cap)
