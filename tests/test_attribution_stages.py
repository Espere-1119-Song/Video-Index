"""The five attribution stages on a synthetic workspace with scripted models (no network)."""
import csv
import json
import os
import tempfile

import numpy as np

from video_index.audit.attribution import ORDER, STAGES
from video_index.audit.context import Context
from video_index.config import Models
from video_index.data.schema import read_jsonl, write_jsonl
from video_index.models import ChatModel, RefusalError, Reply, register_provider


def run_all(namespace):
    """Run every test_* function of a module without pytest."""
    n = 0
    for k, f in sorted(namespace.items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
            print(f"ok  {k}")
    print(f"{n} tests passed")


class Scripted(ChatModel):
    """Replies chosen from the prompt: the question text carries the case name. Every model records its own
    requests (name, number of frames) in `self.calls`."""

    def __init__(self, spec):
        super().__init__(spec)
        self.calls = []

    def _request(self, prompt, images_b64, system, max_tokens, temperature):
        self.calls.append((self.name, len(images_b64)))
        if "reason step by step" in prompt:
            if "case-refuse" in prompt:
                raise RefusalError("refusal")
            letter = "A" if "case-recover" in prompt else "B"
            return Reply(text=f"Frame 3 shows the scene.\nFinal answer: {letter}", input_tokens=10, output_tokens=5)
        if "Diagnose the PRIMARY failure cause" in prompt:
            assert "the model" in prompt and "scripted-ref" not in prompt          # identity withheld
            cat = "ocr_text"
            if "case-evidence" in prompt:
                cat = "evidence_not_in_input_frames"
            elif "case-disagree" in prompt and self.name == "judge-b":
                cat = "temporal_order"
            return Reply(text=json.dumps(dict(category=cat, reason="r", free_desc="f")), input_tokens=20, output_tokens=8)
        if "denser frame set" in prompt:
            return Reply(text='{"evidence_visible": true, "resolution": "sampling_gap", "reason": "r"}')
        raise AssertionError("unexpected prompt")


register_provider("scripted_attr", f"{__name__}:Scripted")


def workspace(root):
    import cv2
    vid = "v" * 8
    os.makedirs(os.path.join(root, "videos", vid))
    w = cv2.VideoWriter(os.path.join(root, "videos", vid, "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(200):
        w.write(np.full((48, 64, 3), i, dtype=np.uint8))
    w.release()
    json.dump(dict(n_frames=200, duration=20.0, frame_timestamps=[i / 10 for i in range(200)]),
              open(os.path.join(root, "videos", vid, "meta.json"), "w"))
    cases = ["case-plain", "case-recover", "case-refuse", "case-evidence", "case-disagree", "case-prior", "case-right"]
    samples = [dict(qid=f"B_{i}", benchmark="B", question=f"What is shown ({c})?", options=["cat", "dog", "bird"],
                    answer="A", video_id=vid, subtask="s") for i, c in enumerate(cases)]
    write_jsonl(os.path.join(root, "samples", "B.jsonl"), samples)
    ref = [dict(qid=s["qid"], role="reference", pred="A" if "right" in s["question"] else "B", answer="A",
                correct="right" in s["question"], frames=32) for s in samples]
    write_jsonl(os.path.join(root, "results", "reference", "B__scripted-ref.jsonl"), ref)
    blind = [dict(qid=s["qid"], role="text_attacker", answer="A",
                  pred="B" if "prior" in s["question"] else "A" if "plain" in s["question"] else "C") for s in samples]
    write_jsonl(os.path.join(root, "results", "blind", "B__scripted-ref.jsonl"), blind)
    spec = lambda name: dict(name=name, provider="scripted_attr", model=name, max_retries=0)  # noqa: E731
    models = Models(dict(reference=spec("scripted-ref"), judge_a=spec("judge-a"), judge_b=spec("judge-b")))
    return Context(root, models, dict(seed=42, reference_frames=32, frame_long_side=768, workers=2,
                                      benchmarks=[dict(name="B", question_format="MCQ")],
                                      attribution=dict(check_n=3)))


def test_stages_end_to_end():
    import importlib
    with tempfile.TemporaryDirectory() as root:
        ctx = workspace(root)
        out = {s: importlib.import_module(STAGES[s]).run(ctx, ["B"]) for s in ORDER}

        traces = {r["qid"]: r for r in read_jsonl(ctx.path("attribution", stage="traces", bench="B"))}
        assert set(traces) == {f"B_{i}" for i in range(6)}                      # B_6 was answered correctly
        assert traces["B_1"]["still_wrong"] is False and traces["B_0"]["still_wrong"] is True
        assert traces["B_2"]["refused"] and traces["B_2"]["still_wrong"] is None
        assert out["attribution_traces"]["B"]["refused"] == 1
        assert len({r["frame_check"] for r in traces.values()}) == 1 and traces["B_0"]["n_frames"] == 200
        rec = list(csv.DictReader(open(ctx.path("table", name="attribution_recovered.csv"))))[0]
        assert (rec["n_rerun"], rec["n_recovered_by_cot"], rec["n_refused"], rec["recovery_rate"]) == ("6", "1", "1", "0.2")

        pre = {r["qid"]: r for r in read_jsonl(ctx.path("attribution", stage="preclass", bench="B"))}
        assert set(pre) == {"B_0", "B_3", "B_4", "B_5"}
        assert pre["B_5"]["preclass"] == "language_prior_dominated" and not pre["B_5"]["to_judge"]
        assert pre["B_0"]["vision_introduced_error"] and not pre["B_3"]["vision_introduced_error"]

        rows = {r["qid"]: r for r in read_jsonl(ctx.path("attribution", stage="judge", bench="B"))}
        assert rows["B_5"]["agreed_level"] == "rule" and rows["B_5"]["by"] == {}
        assert rows["B_0"]["agreed_level"] == "fine" and rows["B_0"]["category"] == "ocr_text"
        assert set(rows["B_0"]["by"]) == {"judge_a", "judge_b"} and rows["B_0"]["by"]["judge_b"]["model"] == "judge-b"
        assert rows["B_4"]["agreed_level"] == "disputed" and rows["B_4"]["category"] is None
        d = rows["B_3"]["dense_recheck"]
        assert d["triggered"] and d["final"] == "sampling_gap" and d["n_frames_dense"] == 128
        assert rows["B_3"]["category"] == "sampling_gap" and rows["B_3"]["frame_check"] == traces["B_3"]["frame_check"]
        calls = lambda: [c for r in ("reference", "judge_a", "judge_b") for c in ctx.models[r].calls]  # noqa: E731
        assert ("judge-a", 32) in calls() and ("judge-b", 128) in calls()

        chk = read_jsonl(ctx.path("attribution", stage="checks", bench="B"))
        assert len(chk) == 3 and all(r["offset"]["mode"] == "window" for r in chk)
        assert all(r["flipped_correct"] is False for r in chk)

        shares = list(csv.DictReader(open(ctx.path("table", name="attribution_shares.csv"))))[0]
        assert shares["n_errors"] == "4" and shares["n_disputed"] == "1" and shares["n_denominator"] == "3"
        assert shares["n_coverage"] == "1" and shares["n_perception"] == "1" and shares["n_reasoning"] == "1"
        assert shares["capability_share"] == "0.6667"
        fine = list(csv.DictReader(open(ctx.path("table", name="attribution_fine.csv"))))
        assert sum(int(r["count"]) for r in fine) == 2
        for name in ("attribution_shares_pooled.csv", "attribution_fine_pooled.csv", "attribution_row_labels.csv",
                     "attribution_matrix.csv", "attribution_disputed_rates.csv", "attribution_unstable.csv"):
            assert os.path.getsize(ctx.path("table", name=name)) > 0, name

        n = len(calls())                                 # a second run sends no request
        for s in ORDER:
            importlib.import_module(STAGES[s]).run(ctx, ["B"])
        assert len(calls()) == n


if __name__ == "__main__":
    run_all(globals())
