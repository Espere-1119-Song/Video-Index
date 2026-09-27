"""The four composition stages on the synthetic workspace of the pool test (scripted models, no network), and the
released layout read back by the evaluation loader."""
import importlib
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pool_stages as P  # noqa: E402

from video_index.compose import CAPS, ORDER, STAGES, path  # noqa: E402
from video_index.compose.verify import RECHECK  # noqa: E402
from video_index.data.schema import read_jsonl  # noqa: E402


def number(item):
    return int(re.search(r"item (\d+)", item["question"]).group(1))


def test_compose_end_to_end():
    with tempfile.TemporaryDirectory() as root:
        ctx = P.workspace(root)
        P.run_pool(ctx)
        out = {s: importlib.import_module(STAGES[s]).run(ctx) for s in ORDER}

        cand = json.load(open(path(ctx, "candidates.json")))
        assert out["compose_candidates"]["n_candidates"] == len(cand) == 22
        assert len({c["video_id"] for c in cand}) == len(cand)                    # one candidate per video
        assert all(c["capability"] == CAPS[number(c) % 4] for c in cand)
        assert all(c["options"][c["answer_idx"]].startswith("right-n") and c["chance"] == 1 / 3 for c in cand)

        first = {r["item_id"]: r for r in read_jsonl(path(ctx, "verify_f32.jsonl"))}
        again = {r["item_id"]: r for r in read_jsonl(path(ctx, "verify_f512.jsonl"))}
        assert set(first) == {c["item_id"] for c in cand}
        assert set(again) == {i for i, r in first.items() if r["verdict"] in RECHECK}
        assert all(r["frames"] <= 32 for r in first.values()) and max(r["frames"] for r in again.values()) == 40
        verified = json.load(open(path(ctx, "verified.json")))
        by_id = {c["item_id"]: c for c in cand}
        for v in verified:
            assert v["keep"] and v["verdict"] == "supported" and number(v) % 10 != 9
        kept_again = [v for v in verified if v["verdict_frames"] == 40]
        assert kept_again and all(number(v) % 10 == 8 for v in kept_again)         # confirmed by the second pass
        dropped = set(by_id) - {v["item_id"] for v in verified}
        assert all(number(by_id[i]) % 10 in (8, 9) for i in dropped)
        summ = out["compose_verify"]
        assert summ["n_kept"] == len(verified) and summ["n_asked_recheck"] == len(again) and summ["n_refused"] == 0

        rep = out["compose_select"]
        items = json.load(open(path(ctx, "video_index.json")))
        assert rep["n_items"] == len(items) <= 8 and all(n <= 2 for n in rep["per_group"].values())
        assert len({it["video_id"] for it in items}) == len(items)
        assert {it["item_id"] for it in items} <= {v["item_id"] for v in verified}

        rel = out["compose_export"]["out_dir"]
        rows = read_jsonl(os.path.join(rel, "items", "meta_benchmark.jsonl"))
        assert [r["item_id"] for r in rows] == [it["item_id"] for it in items]
        r = rows[0]
        assert r["answer"] == "ABC"[r["answer_idx"]] and r["video"] == f"videos/{r['video_id']}.mp4"
        assert r["verification"] == dict(verdict="supported", keep_reason="supported") and r["license"] == "CC-BY-4.0"
        assert r["fine_category"] and r["capability_group"] in CAPS and r["duration_s"] in (6.0, 40.0)
        assert all(os.path.exists(os.path.join(rel, x["video"])) for x in rows)
        readme = open(os.path.join(rel, "README.md")).read()
        assert readme.startswith("---\n") and f"{len(rows)} multiple-choice video questions" in readme
        assert "`labeler`" in readme and "`verifier`" in readme
        assert json.load(open(os.path.join(rel, "items", "summary.json")))["n_items"] == len(rows)
        assert os.path.exists(os.path.join(rel, "code", "prompt.md"))

        from video_index.evaluate.dataset import VideoIndex
        data = VideoIndex(rel)
        assert len(data.items) == len(rows) and os.path.exists(data.video_path(data.items[0]))

        n = len(first) + len(again)                                               # a second run asks nothing again
        importlib.import_module(STAGES["compose_verify"]).run(ctx)
        assert n == len(read_jsonl(path(ctx, "verify_f32.jsonl"))) + len(read_jsonl(path(ctx, "verify_f512.jsonl")))


if __name__ == "__main__":
    P.run_all(globals())
