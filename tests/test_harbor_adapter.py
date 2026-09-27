"""Harbor adapter: task layout, permutations and answer letters on a two-item local dataset."""
import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "harbor" / "src"))
from video_index_harbor.adapter import VideoIndexAdapter, permutations_for  # noqa: E402

ITEMS = [dict(item_id="AoTBench_1", benchmark="AoTBench", capability_group="temporal", fine_category="temporal_order",
              question="Is the clip reversed?", options=["yes", "no"], answer="B", answer_idx=1, chance=0.5,
              video="videos/a.mp4", video_id="a", duration_s=4.0),
         dict(item_id="X_2", benchmark='Name "quoted"', capability_group="perception", fine_category="ocr_text",
              question="Which word?", options=["cat", "dog", "cow", "hen"], answer="C", answer_idx=2, chance=0.25,
              video="videos/b.mp4", video_id="b", duration_s=9.0)]


def _dataset(tmp_path):
    (tmp_path / "items").mkdir(parents=True)
    (tmp_path / "videos").mkdir()
    (tmp_path / "items" / "test.jsonl").write_text("\n".join(json.dumps(r) for r in ITEMS))
    for v in ("a", "b"):
        (tmp_path / "videos" / f"{v}.mp4").write_bytes(b"\x00")
    return tmp_path


def test_video_tasks(tmp_path):
    src = _dataset(tmp_path / "src")
    out = tmp_path / "out"
    ad = VideoIndexAdapter(out, str(src), revision="abc123", protocol="video")
    assert ad.generate() == 2
    t = out / "x-2"
    for f in ("task.toml", "instruction.md", "environment/Dockerfile", "solution/solve.sh", "tests/test.sh"):
        assert (t / f).exists(), f
    cfg = tomllib.loads((t / "task.toml").read_text())
    assert cfg["task"]["name"] == "video-index/x-2" and cfg["agent"]["network_mode"] == "no-network"
    assert cfg["metadata"]["source_benchmark"] == "Name 'quoted'"
    assert 'expected="C"' in (t / "tests/test.sh").read_text() and '"C"' in (t / "solution/solve.sh").read_text()
    assert "C. cow" in (t / "instruction.md").read_text() and "resolve/abc123/videos/b.mp4" in (t / "environment/Dockerfile").read_text()
    assert ad.generate() == 0 and ad.generate(overwrite=True) == 2


def test_blind_permutations(tmp_path):
    src = _dataset(tmp_path / "src")
    out = tmp_path / "out"
    ad = VideoIndexAdapter(out, str(src), revision="abc123", protocol="blind")
    assert ad.generate() == 8
    perms = permutations_for("X_2", 4)
    for k, perm in enumerate(perms):
        letter = "ABCD"[perm.index(2)]
        assert f'expected="{letter}"' in (out / f"x-2-p{k}" / "tests/test.sh").read_text()
    assert "video.mp4" not in (out / "x-2-p0" / "environment/Dockerfile").read_text()


def test_embedded_video(tmp_path):
    src = _dataset(tmp_path / "src")
    out = tmp_path / "out"
    VideoIndexAdapter(out, str(src), revision="abc123", embed_videos=True).generate()
    assert (out / "aotbench-1" / "environment" / "video.mp4").exists()
    assert "COPY video.mp4 /app/video.mp4" in (out / "aotbench-1" / "environment/Dockerfile").read_text()
