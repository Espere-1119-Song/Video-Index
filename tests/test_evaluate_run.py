"""A run of ``vi-eval`` with an in-process model on a small local copy of the dataset layout."""
import csv
import json
import os

import numpy as np
import pytest

from video_index.evaluate import cli
from video_index.evaluate import protocol as P

cv2 = pytest.importorskip("cv2")

GROUPS = [k for k, _ in P.GROUPS]


class FirstOption:
    """Replies "A" and records what it was given."""
    calls = []

    def __init__(self, reject_above=None):
        self.reject_above = reject_above

    def answer(self, frames, timestamps, prompt):
        if self.reject_above and len(frames) > self.reject_above:
            raise RuntimeError("the request exceeds the maximum context length")
        FirstOption.calls.append((len(frames), prompt))
        return "A"


def _video(path, n_frames, fps=2.0):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n_frames):
        w.write(np.full((48, 64, 3), (i * 5) % 255, dtype=np.uint8))
    w.release()


@pytest.fixture()
def data_dir(tmp_path):
    root = tmp_path / "data"
    (root / "items").mkdir(parents=True)
    (root / "videos").mkdir()
    with open(root / "items" / "meta_benchmark.jsonl", "w") as f:
        for i in range(8):
            _video(str(root / "videos" / f"v{i}.mp4"), 40)
            f.write(json.dumps({"item_id": f"Demo_{i}", "benchmark": "Demo", "capability_group": GROUPS[i % 4],
                                "question": f"Question {i}?", "options": ["yes", "no", "maybe"], "answer": "AB"[i % 2],
                                "answer_idx": i % 2, "video": f"videos/v{i}.mp4", "video_id": f"v{i}",
                                "duration_s": 20.0}) + "\n")
    return str(root)


def _rows(path):
    return [json.loads(line) for line in open(path)]


def test_run_score_and_resume(data_dir, tmp_path, capsys):
    out = str(tmp_path / "run")
    FirstOption.calls = []
    argv = ["run", "--data", data_dir, "--local", f"{__name__}:FirstOption", "--name", "first", "--protocol", "both",
            "--out", out, "--frame-cap", "8"]
    assert cli.main(argv) == 0
    video = _rows(os.path.join(out, "first__video1fps.jsonl"))
    blind = _rows(os.path.join(out, "first__blind.jsonl"))
    assert len(video) == 8 and len(blind) == 32
    assert all(r["frames"] == 8 for r in video)
    assert sum(r["correct"] for r in video) == 4          # the answer is A on the even items
    assert "You are given 8 frame(s)" in FirstOption.calls[0][1]
    table = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
    assert len(table) == 1 and float(table[0]["video"]) == pytest.approx(50.0)
    assert "first" in capsys.readouterr().out

    n_calls = len(FirstOption.calls)
    assert cli.main(argv) == 0                            # nothing left to do
    assert len(FirstOption.calls) == n_calls
    assert len(_rows(os.path.join(out, "first__video1fps.jsonl"))) == 8


def test_window_and_ids(data_dir, tmp_path):
    out = str(tmp_path / "run")
    base = ["run", "--data", data_dir, "--local", f"{__name__}:FirstOption", "--name", "first", "--protocol", "video",
            "--frame-cap", "8"]
    assert cli.main(base + ["--out", out + "_w", "--window", "2:5"]) == 0
    assert [r["item_id"] for r in _rows(out + "_w/first__video1fps.jsonl")] == ["Demo_2", "Demo_3", "Demo_4"]
    assert cli.main(base + ["--out", out + "_i", "--ids", "Demo_7,Demo_1"]) == 0
    assert sorted(r["item_id"] for r in _rows(out + "_i/first__video1fps.jsonl")) == ["Demo_1", "Demo_7"]


def test_frames_are_halved_when_the_model_rejects_the_size(data_dir, tmp_path):
    out = str(tmp_path / "run")
    FirstOption.calls = []
    argv = ["run", "--data", data_dir, "--local", f"{__name__}:FirstOption", "--local-kwargs", '{"reject_above": 10}',
            "--name", "first", "--protocol", "video", "--out", out, "--window", "0:2", "--min-frames", "4"]
    assert cli.main(argv) == 0
    rows = _rows(os.path.join(out, "first__video1fps.jsonl"))
    assert [r["frames"] for r in rows] == [10, 10] and all(r["fallback"] == 1 for r in rows)
    assert all("You are given 10 frame(s)" in prompt for _, prompt in FirstOption.calls)
