"""The whole audit on a synthetic benchmark, offline: four generated videos, 48 items, a stub model that answers
by rule, and hash-based vectors in place of the text encoder. Needs ffmpeg and ffprobe on PATH."""
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_audit_common import run_all  # noqa: E402

import numpy as np  # noqa: E402

from video_index.audit import cli, embeddings  # noqa: E402
from video_index.audit.context import DEFAULTS, Context  # noqa: E402
from video_index.config import Models  # noqa: E402
from video_index.data.schema import read_jsonl, write_jsonl  # noqa: E402
from video_index.models import ChatModel, Reply, register_provider  # noqa: E402


class StubModel(ChatModel):
    """Answers by rule. GOLD maps a question text to (gold letter in the original option order, option texts).
    With frames: the gold option with probability `p_frames`; question without frames: `p_text`; options only: the
    first option; caption requests: a fixed sentence. The draw depends on the question text only."""
    GOLD = {}
    CALLS = []

    def _request(self, prompt, images_b64, system, max_tokens, temperature):
        StubModel.CALLS.append((self.spec.name, len(images_b64)))
        if prompt.startswith(("Describe this image", "Describe this video")):
            return Reply(text="A person stands in a room.", stop_reason="end_turn")
        m = re.search(r"Question: (.*?)\n", prompt)
        if not m or m.group(1) not in self.GOLD:
            return Reply(text="A", stop_reason="end_turn")
        gold_text = self.GOLD[m.group(1)]
        shown = dict((t.strip(), L) for L, t in re.findall(r"(?m)^([A-P])\. (.*)$", prompt))
        u = int(hashlib.sha256(m.group(1).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        p = float(self.spec.extra.get("p_frames", 0.8)) if images_b64 else float(self.spec.extra.get("p_text", 0.3))
        if "captions of" in prompt:
            p = float(self.spec.extra.get("p_captions", 0.4))
        wrong = sorted(t for t in shown if t != gold_text)                # the same wrong option under every order
        letter = shown.get(gold_text, "A") if u < p or not wrong else shown[wrong[0]]
        return Reply(text=letter, stop_reason="end_turn")


register_provider("stub", f"{__name__}:StubModel")
MODELS = dict(reference=dict(name="stub-reference", provider="stub", model="stub", p_frames=0.8, p_text=0.3),
              attacker=dict(name="stub-attacker", provider="stub", model="stub", p_frames=0.6, p_text=0.3))


def fake_embed(texts, name=None, device=None, batch_size=64):
    out = []
    for t in texts:
        v = np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).normal(size=32)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, dtype=np.float32)


def build_workspace(root):
    raw = os.path.join(root, "raw")
    os.makedirs(raw)
    for i, dur in enumerate((6, 20, 40, 70)):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=duration={dur}:size=640x360:rate=10",
                        "-pix_fmt", "yuv420p", os.path.join(raw, f"v{i}.mp4")], check=True)
    items = []
    for i in range(48):
        opts = [f"object {i} {j}" for j in range(4)]
        q = f"Which object appears in scene {i}?"
        items.append(dict(qid=f"Synth_{i:03d}", benchmark="Synth", question=q, options=opts, answer="ABCD"[i % 4],
                          subtask=f"task{i % 4}", video_ref=f"v{i % 4}.mp4"))
        StubModel.GOLD[q] = opts[i % 4]
    items.append(dict(items[0], qid="Synth_900"))                       # near-duplicate of Synth_000
    write_jsonl(os.path.join(root, "items", "Synth.jsonl"), items)
    cfg = {**DEFAULTS, "video_root": raw, "sample_n": 40, "sample_min_per_stratum": 5, "min_items": 30, "workers": 4,
           "caption_modes": ["frames", "video"], "pool_attack_permutations": 3, "rescreen_all": True,
           "benchmarks": [dict(name="Synth", question_format="MCQ", num_options=4)]}
    return Context(root, Models(MODELS), cfg)


def test_pipeline_offline():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        print("skipped: ffmpeg not available")
        return
    root = tempfile.mkdtemp(prefix="vi_audit_")
    real_embed = embeddings.embed
    embeddings.embed = fake_embed
    try:
        ctx = build_workspace(root)
        cli.run_stages(ctx, ["sample"])
        sample = read_jsonl(ctx.path("samples", bench="Synth"))
        assert len(sample) == 40 and len({r["subtask"] for r in sample}) == 4
        pair = ("Synth_000", "Synth_900")                                # the duplicate pair enters the sample
        sample = [r for r in sample if r["qid"] not in pair][:38] + \
                 [r for r in read_jsonl(ctx.path("items", bench="Synth")) if r["qid"] in pair]
        write_jsonl(ctx.path("samples", bench="Synth"), sample)
        out = cli.run_stages(ctx, list(cli.DEFAULT_STAGES))
        sample = read_jsonl(ctx.path("samples", bench="Synth"))
        assert all(r.get("video_id") and len(r["video_id"]) == 64 for r in sample)
        meta = json.load(open(ctx.path("video_meta", video_id=sample[0]["video_id"])))
        assert meta["normalized_fps"] == 2.0 and meta["n_frames"] == len(meta["frame_timestamps"])
        assert meta["normalized_short_side"] == 360                      # never enlarged
        scr = list(csv.DictReader(open(ctx.path("table", name="screen_items.csv"))))
        assert [(r["item_id"], r["action"]) for r in scr] == [("Synth_900", "remove")]
        for cond in ("options_only", "blind", "single_frame", "reference", "reference_128", "shuffle", "window", "captions",
                     "captions_video", "options_only_perm"):
            d = os.path.join(root, "results", cond)
            assert os.path.isdir(d) and os.listdir(d), cond
        ref = [r for r in read_jsonl(ctx.path("result", condition="reference", bench="Synth", model="stub-reference"))]
        assert len(ref) == 40 and all(set(r) >= {"qid", "pred", "answer", "correct", "frames", "fallback", "error"} for r in ref)
        assert max(r["frames"] for r in ref) == 32 and min(r["frames"] for r in ref) == 12     # the 6 s video holds 12 frames
        lv = list(csv.DictReader(open(ctx.path("table", name="levels.csv"))))[0]
        assert lv["n_items"] == "39" and lv["n_mcq"] == "39" and lv["n_ref"] == "39" and lv["c"] == "0.2500"
        assert lv["levels_measured"] == "option;text;pool;frame;order" and lv["break_level"] in ("frame", "order")
        assert open(ctx.path("table", name="audited_benchmarks.txt")).read() == "Synth\n"
        pm = list(csv.DictReader(open(ctx.path("table", name="permutation_by_benchmark.csv"))))[0]
        assert pm["status"] == "complete" and pm["n"] == "39" and pm["item_consistency"] == "1.0000"
        assert os.path.exists(ctx.path("table", name="report/Synth.md"))
        n_calls = len(StubModel.CALLS)
        cli.run_stages(ctx, list(cli.DEFAULT_STAGES))                    # rerun: every row is on disk
        assert len(StubModel.CALLS) == n_calls
        out = cli.run_stages(ctx, ["refill"])                            # the removed item is replaced from its subtask
        new = read_jsonl(ctx.path("samples", bench="Synth"))
        assert len(new) == 40 and not any(r["qid"] == "Synth_900" for r in new)
        assert out["refill"][0]["n_added"] == 1 and out["refill"][0]["draw_method"] == "seed_fallback"
        assert all(r.get("video_id") for r in new)
        print(f"    {n_calls} stub calls; level {lv['break_level']}; s* {lv['s_star']}")
    finally:
        embeddings.embed = real_embed
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    run_all(globals())
