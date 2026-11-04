"""Mage-VL on Video-Index through its codec-native video input.

This is the benchmark's native-video setting (the `video_index` task of lmms-eval, VIDEO_LLM models
of VLMEvalKit): the model receives the video file and samples it itself, and the prompt starts with
"You are given a video. Answer the question based on this video.". The video goes through Mage-VL's
HEVC codec processor, which packs up to target_canvas / 4 * 32 frames into target_canvas canvases.
Options, scoring and result rows follow vi-eval. A video the codec cannot read falls back to the
1 FPS frames of the official protocol and is flagged with "codec_fallback".

    PYTHONPATH=integrations/mage_vl python integrations/mage_vl/codec_run.py --out runs/mage-vl
    PYTHONPATH=integrations/mage_vl python integrations/mage_vl/codec_run.py --out runs/mage-vl --score
"""
import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from mage_vl import MageVL
from video_index.data import frames as F
from video_index.evaluate import protocol as P
from video_index.evaluate import score as S
from video_index.evaluate.dataset import HUB_ID, VideoIndex

INTRO_VIDEO = "You are given a video. Answer the question based on this video."


def run(args, protocol, out_p):
    done = set()
    if os.path.exists(out_p):
        done = {r["item_id"] for r in map(json.loads, open(out_p)) if r.get("correct") is not None}
    data = VideoIndex(args.data)
    items = [it for it in data.select(shard=args.shard) if it["item_id"] not in done]
    print(f"[codec] {protocol}: {len(items)} items to do, {len(done)} done", flush=True)
    model = MageVL(max_new_tokens=args.max_tokens)
    pad_id = model.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    tmp_root = Path(tempfile.mkdtemp(prefix="vi_codec_"))

    with open(out_p, "a") as f:
        for k, it in enumerate(items, 1):
            perm = list(range(len(it["options"])))
            row = dict(model=model.name, protocol=protocol, item_id=it["item_id"], benchmark=it["benchmark"],
                       capability_group=it.get("capability_group"), perm=perm)
            t0 = time.time()
            video = data.video_path(it)
            try:
                prompt = P.PROMPT.format(intro=INTRO_VIDEO, q=it["question"], opts=P.render(it["options"]))
                work = Path(tempfile.mkdtemp(dir=tmp_root))
                try:
                    inputs = model.processor(
                        text=[model.chat_text(prompt, video=True)], videos=[video], video_backend="codec",
                        max_pixels=args.max_pixels, return_tensors="pt",
                        codec_config={"patch": 16, "target_canvas": args.target_canvas, "cache_root": work})
                finally:
                    shutil.rmtree(work, ignore_errors=True)
                row["canvases"] = int(inputs["image_grid_thw"].shape[0])
                row["visual_tokens"] = int((inputs["input_ids"] == pad_id).sum())
                pred = model.generate(inputs)
            except Exception as e:  # noqa: BLE001 - fall back to the official 1 FPS frames
                row["codec_fallback"] = f"{type(e).__name__}: {str(e)[:200]}"
                info = F.probe(video)
                idx = P.one_fps_indices(info["n_frames"], info["fps"], it.get("duration_s"))
                frames, ts = F.read_indices(video, idx, P.SHORT_SIDE)
                row["frames"] = len(frames)
                pred = model.answer(frames, ts, P.video_prompt(it, len(frames), perm))
            row.update(correct=P.score_reply(pred, it, perm), pred=pred[:200], gold=P.gold_letter(it, perm),
                       ms=int((time.time() - t0) * 1000))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if k % 25 == 0:
                print(f"[codec] {k}/{len(items)} done", flush=True)
    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"[codec] DONE -> {out_p}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=HUB_ID)
    ap.add_argument("--out", required=True, help="run directory (put the vi-eval blind rows here too)")
    ap.add_argument("--target-canvas", type=int, default=128)
    ap.add_argument("--max-pixels", type=int, default=150000)
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--score", action="store_true", help="score the run directory instead of running")
    args = ap.parse_args()

    protocol = f"video_codec_tc{args.target_canvas}"
    os.makedirs(args.out, exist_ok=True)
    if args.score:
        rows = S.load_rows(args.out)
        vid = [r for r in rows if r["protocol"] == protocol]
        tokens = [r["visual_tokens"] for r in vid if "visual_tokens" in r]
        print(S.render(S.table(rows, video_protocol=protocol)))
        print(f"{protocol}: {len(vid)} items, mean visual tokens {sum(tokens) / max(len(tokens), 1):.0f}, "
              f"{sum(1 for r in vid if r.get('codec_fallback'))} codec fallbacks")
        return
    run(args, protocol, os.path.join(args.out, f"mage-vl-4b__{protocol}.jsonl"))


if __name__ == "__main__":
    main()
