"""video_index.data: JSON Lines records and frame reading."""
import os
import tempfile
import threading

import numpy as np
import pytest

from video_index.data import frames as F
from video_index.data.schema import JsonlWriter, read_jsonl, write_jsonl


def test_jsonl_roundtrip_and_resume():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sub", "rows.jsonl")
        assert read_jsonl(p) == []
        write_jsonl(p, [dict(qid="a", role="r", pred="A"), dict(qid="b", role="r", pred="B")])
        with open(p, "a") as f:
            f.write('{"qid": "c", "role"')                       # a line cut off by an interrupted write
        assert [r["qid"] for r in read_jsonl(p)] == ["a", "b"]
        write_jsonl(p, read_jsonl(p))
        with JsonlWriter(p, key=("qid", "role")) as w:
            assert w.has(dict(qid="a", role="r")) and not w.has(dict(qid="a", role="other"))
            threads = [threading.Thread(target=w.write, args=(dict(qid=f"t{i}", role="r"),)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert w.has(dict(qid="t7", role="r"))
        assert len(read_jsonl(p)) == 22


def test_subsample_keeps_ends_and_matches_the_frame_rule():
    frames, ts = list(range(100)), [i / 2 for i in range(100)]
    f2, t2 = F.subsample(frames, ts, 10)
    assert len(f2) == 10 and f2[0] == 0 and f2[-1] == 99 and t2 == [i / 2 for i in f2]
    assert F.subsample(frames[:5], ts[:5], 10) == (frames[:5], ts[:5])
    assert F.subsample(frames, ts, 0) == ([], [])


def test_read_indices_in_stream_order_and_resize():
    cv2 = pytest.importorskip("cv2")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "v.avi")
        out = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (640, 360))
        for i in range(20):
            out.write(np.full((360, 640, 3), i * 12, dtype=np.uint8))
        out.release()
        info = F.probe(p)
        assert info["n_frames"] == 20 and info["fps"] == pytest.approx(2.0) and (info["width"], info["height"]) == (640, 360)
        frames, ts = F.read_indices(p, [15, 3, 99], short_side=224)
        assert ts == [7.5, 1.5] and [f.size for f in frames] == [(398, 224)] * 2
        assert abs(int(np.asarray(frames[0]).mean()) - 180) <= 3 and abs(int(np.asarray(frames[1]).mean()) - 36) <= 3
        frames, ts = F.fps(p, 1.0, cap=4, short_side=224)
        assert len(frames) == 4 and ts[0] == 0.0
