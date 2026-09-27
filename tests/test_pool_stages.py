"""The four pool stages on a synthetic workspace with scripted models and encoders (no network, no GPU)."""
import base64
import csv
import hashlib
import io
import json
import os
import re
import tempfile

import numpy as np

from video_index.audit.context import Context
from video_index.config import Models
from video_index.data.schema import read_jsonl, write_jsonl
from video_index.models import ChatModel, Reply, register_provider
from video_index.pool import ORDER, STAGES, embed, schema as S

TAGS = "obsfhnnnnn"          # solved by: options only, blind, single frame, 32 frames, shuffled frames, nobody
FINE = [1, 10, 14, 17]       # perception, temporal, spatial_physical, reasoning_knowledge
VERBS = ["happens", "moves", "falls", "appears", "changes", "breaks"]
ADVERBS = ["first", "slowly", "twice", "again", "suddenly"]


def run_all(namespace):
    """Run every test_* function of a module without pytest."""
    n = 0
    for k, f in sorted(namespace.items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
            print(f"ok  {k}")
    print(f"{n} tests passed")


def in_temporal_order(images_b64):
    """The frames of the test videos get brighter with time: a frame list is in temporal order when its mean
    brightness rises."""
    from PIL import Image
    level = [float(np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("L")).mean()) for b in images_b64]
    return sum(a > b + 1.0 for a, b in zip(level, level[1:])) <= 1


class Scripted(ChatModel):
    """The correct option of an item reads "right-<tag>"; the tag names the screen step that solves the item.
    The reply depends on the request alone (prompt and frames), not on earlier requests."""

    def _request(self, prompt, images_b64, system, max_tokens, temperature):
        n = len(images_b64)
        if "Which label best describes" in prompt:
            return Reply(text="E")
        if "Where does the video take place" in prompt:
            return Reply(text="A.")
        if "Which capability does answering it" in prompt:
            k = int(re.search(r"item (\d+)", prompt).group(1))
            return Reply(text=json.dumps(dict(category=FINE[k % 4], first_person=False, absence_question=False)))
        if "Marked answer" in prompt:
            k = int(re.search(r"item (\d+)", prompt).group(1))
            verdict = "contradicted" if k % 10 == 9 else "not_verifiable" if k % 10 == 8 and n <= 32 else "supported"
            return Reply(text=json.dumps(dict(verdict=verdict, visible_answer=None, reason="r")))
        if n > 1:
            step = "f" if in_temporal_order(images_b64) else "h"
        else:
            step = "s" if n == 1 else "o" if "NO question text" in prompt else "b"
        lines = {m.group(2): m.group(1) for m in re.finditer(r"^([A-Z])\. (.+)$", prompt, re.M)}
        right = next((letter for text, letter in lines.items() if text.startswith("right-")), None)
        solved = any(text.startswith(f"right-{step}") for text in lines)
        wrong = next(letter for text, letter in lines.items() if not text.startswith("right-"))
        return Reply(text=right if solved else wrong)


register_provider("scripted_pool", f"{__name__}:Scripted")


def text_encoder(texts):
    out = []
    for t in texts:
        rng = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16))
        v = rng.normal(size=16)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, dtype=np.float32)


def video_encoder(path):
    return np.random.default_rng(int(hashlib.sha1(path.encode()).hexdigest()[:8], 16)).normal(size=16)


def make_video(root, vid, n=60, fps=10):
    import cv2
    d = os.path.join(root, "videos", vid)
    os.makedirs(d, exist_ok=True)
    w = cv2.VideoWriter(os.path.join(d, "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), 10 + int(235 * i / n), dtype=np.uint8))
    w.release()
    json.dump(dict(n_frames=n, duration=n / fps, frame_timestamps=[i / fps for i in range(n)]),
              open(os.path.join(d, "meta.json"), "w"))


def workspace(root, n_videos=40):
    """Two benchmarks; item k sits on video k // 2 (benchmark X) or on its own video (benchmark Y)."""
    items = {"X": [], "Y": []}
    for v in range(n_videos):
        make_video(root, f"vid{v:03d}", n=60 if v % 3 else 400)
    for k in range(60):
        bench = "X" if k < 40 else "Y"
        vid = f"vid{k // 2:03d}" if bench == "X" else f"vid{k - 20:03d}"
        tag = TAGS[k % 10]
        opts = [f"right-{tag} {k}", f"decoy one {k}", f"decoy two {k}"]
        opts = opts[k % 3:] + opts[:k % 3]
        items[bench].append(dict(qid=f"{bench}_{k}", benchmark=bench,
                                 question=f"What {VERBS[k % 6]} {ADVERBS[k % 5]} in item {k}?",
                                 options=opts, answer="ABC"[[o.startswith("right-") for o in opts].index(True)],
                                 video_id=vid, subtask=f"task{k % 3}", duration_s=6.0, license="CC-BY-4.0"))
    x = items["X"]
    x.append(dict(x[5], qid="X_dup"))                                                       # duplicate of X_5
    x.append(dict(x[0], qid="X_open", options=None, answer="a person walks"))
    x.append(dict(x[0], qid="X_yesno", options=["Yes", "No"], answer="A"))
    x.append(dict(x[0], qid="X_meta", question="Which is it (item 7)?", options=["right-n", "cat", "All of the above"], answer="A"))
    x.append(dict(x[0], qid="X_novideo", question="What is it (item 3)?", video_id="missing", options=["right-n a", "b"], answer="A"))
    items["Y"].append(dict(x[7], qid="Y_copy", benchmark="Y"))                              # copy of X_7 in benchmark Y
    for b, rows in items.items():
        write_jsonl(os.path.join(root, "items", f"{b}.jsonl"), rows)
    spec = lambda name: dict(name=name, provider="scripted_pool", model=name, max_retries=0)  # noqa: E731
    models = Models(dict(attacker=spec("big"), attacker_small=spec("small"), labeler=spec("labeler"),
                         verifier=spec("verifier"), reference=spec("big")))
    ctx = Context(root, models, dict(seed=42, workers=1, benchmarks=[dict(name="X", year=2024), dict(name="Y", year=2022)],
                                     pool=dict(budget=20, share=0.6, j=4, logprobs=False, scene_pairs=500),
                                     compose=dict(candidates=30, budget=8, tag="test")))
    embed.set_encoders(ctx, text=text_encoder, video=video_encoder)
    return ctx


def run_pool(ctx):
    import importlib
    return {s: importlib.import_module(STAGES[s]).run(ctx) for s in ORDER}


def test_pool_end_to_end():
    with tempfile.TemporaryDirectory() as root:
        ctx = workspace(root)
        out = run_pool(ctx)
        base = S.read_table(ctx, "pool_items_base")
        assert len(base) == 66 and dict(base.format.value_counts()) == dict(mcq=64, open=1, yesno=1)
        pool = S.read_table(ctx, "pool_items").set_index("item_id")
        by = pool.removed_by.to_dict()
        assert by["X_open"] == "open" and by["X_yesno"] == "format_yesno" and by["X_meta"] == "option_format"
        assert by["X_dup"] == "duplicate" and pool.loc["X_dup", "dup_of"] == "X_5"
        assert by["X_novideo"] == "none" and pool.loc["X_novideo", "pending_steps"] == "single_frame,v32,v32_shuffle"
        assert by["Y_copy"] == "none" and by["X_7"] == "xbench_duplicate" and pool.loc["X_7", "dup_of"] == "Y_copy"
        expect = dict(o="opt_only", b="blind", s="single_frame", f="v32", h="v32_shuffle", n="none")
        for k in range(60):
            if k != 7:
                assert by[f"{'X' if k < 40 else 'Y'}_{k}"] == expect[TAGS[k % 10]], k
        rows = {r["stage"]: r for r in out["pool_screen"]["funnel"]}
        assert rows["multiple_choice"]["n_after"] == 64 and rows["duplicate"]["n_removed"] == 1
        assert [rows[s]["n_removed"] for s in ("opt_only", "blind", "single_frame", "v32", "v32_shuffle")] == [6] * 5
        assert rows["xbench_duplicate"]["n_removed"] == 1 and rows["remaining"]["n_after"] == 31
        assert list(csv.DictReader(open(ctx.path("pool", name="funnel.csv"))))[-1]["n_after"] == "31"
        # steps run in funnel order: an item removed by a step has no row in the later steps
        assert "X_0" not in {r["item_id"] for r in read_jsonl(ctx.path("pool", name="steps/blind/X.jsonl"))}
        assert pool.loc["X_5", "opt_acc_4perm"] == 0.0 and pool.loc["X_1", "blind_acc_4perm"] == 1.0

        assert out["pool_labels"]["n_videos_labelled"] == 22 and out["pool_labels"]["n_items_labelled"] == 31
        scenes = list(csv.DictReader(open(ctx.path("pool", name="scene_labels.csv"))))
        assert {r["scene_type"] for r in scenes} == {"cooking_food"} and {r["setting"] for r in scenes} == {"indoor"}
        caps = {r["item_id"]: r for r in csv.DictReader(open(ctx.path("pool", name="item_capability.csv")))}
        assert caps["X_5"]["fine"] == "temporal_order" and caps["X_5"]["group"] == "temporal"

        sel = out["pool_select"]
        items = read_jsonl(ctx.path("pool", name="video_index_items.jsonl"))
        assert sel["items"] == len(items) == 20 and sel["eligible_items"] == 30 and sel["excluded"]["n_pending"] == 1
        assert len({r["video_id"] for r in items}) == 20                       # one item per video
        assert all(r["options"][r["answer_idx"]].startswith("right-n") and r["answer"] == "ABC"[r["answer_idx"]]
                   for r in items)
        assert {r["duration_group"] for r in items} == {"<10s", "30-60s"} and items[0]["capability"]
        n_calls = sum(len(read_jsonl(ctx.path("pool", name=f"steps/{s}/{b}.jsonl")))
                      for s in ("opt_only", "blind", "single_frame", "v32", "v32_shuffle") for b in "XY")
        run_pool(ctx)                                                          # resumable: nothing is asked again
        assert n_calls == sum(len(read_jsonl(ctx.path("pool", name=f"steps/{s}/{b}.jsonl")))
                              for s in ("opt_only", "blind", "single_frame", "v32", "v32_shuffle") for b in "XY")


if __name__ == "__main__":
    run_all(globals())
