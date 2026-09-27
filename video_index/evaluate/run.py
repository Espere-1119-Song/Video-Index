"""Running a model on Video-Index. Result files are append-only JSON Lines, one row per (item, permutation):

    runs/<name>/<model>__video1fps.jsonl
    runs/<name>/<model>__blind.jsonl

A rerun of the same command continues after the last finished row."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..data import frames as F
from ..runner import log
from . import protocol as P
from .dataset import VideoIndex
from .local import is_size_error

PROTOCOL_NAME = {"video": "video1fps", "blind": "blind"}


def safe(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def _done(path: str) -> set:
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("correct") is not None or r.get("error") == "no_video":
                done.add((r["item_id"], tuple(r.get("perm") or [])))
    return done


def run_protocol(model, data: VideoIndex, protocol: str, out_dir: str, limit: int | None = None, shard: str | None = None,
                 workers: int = 8, frame_cap: int = P.FRAME_CAP, fps: float = P.FPS, short_side: int = P.SHORT_SIDE,
                 min_frames: int = 8, max_minutes: float | None = None, ids: str | None = None,
                 window: str | None = None) -> str:
    assert protocol in PROTOCOL_NAME, protocol
    pname = PROTOCOL_NAME[protocol] if (protocol == "blind" or (fps == P.FPS)) else f"video{fps:g}fps"
    os.makedirs(out_dir, exist_ok=True)
    out_p = os.path.join(out_dir, f"{safe(model.name)}__{pname}.jsonl")
    items = data.select(limit, shard, ids=ids, window=window)
    done = _done(out_p)
    perms_of = (lambda it: P.permutations_for(it["item_id"], len(it["options"]))) if protocol == "blind" else \
               (lambda it: [list(range(len(it["options"])))])
    todo = [(it, p) for it in items for p in perms_of(it) if (it["item_id"], tuple(p)) not in done]
    log(f"{model.name}/{pname}: {len(items)} items, {len(todo)} (item, permutation) to do, {len(done)} done", "vi-eval")
    if not todo:
        return out_p

    def run_one(it, perm):
        row = dict(model=model.name, protocol=pname, item_id=it["item_id"], benchmark=it["benchmark"],
                   capability_group=it.get("capability_group"), perm=perm, frames=0, short_side=short_side)
        t0 = time.time()
        try:
            frames, ts = [], []
            if protocol == "video":
                try:
                    video = data.video_path(it)
                except Exception as e:  # noqa: BLE001
                    row.update(correct=None, pred=None, error="no_video", detail=str(e)[:200])
                    return row
                info = F.probe(video)
                idx = P.one_fps_indices(info["n_frames"], info["fps"], it.get("duration_s"), fps, frame_cap)
                frames, ts = F.read_indices(video, idx, short_side)
                if not frames:
                    row.update(correct=None, pred=None, error="no_frames")
                    return row
            while True:
                row["frames"] = len(frames)
                prompt = P.video_prompt(it, len(frames), perm) if protocol == "video" else P.blind_prompt(it, perm)
                try:
                    pred = model.answer(frames, ts, prompt)
                    break
                except Exception as e:  # noqa: BLE001
                    if frames and len(frames) > min_frames and is_size_error(e):
                        frames, ts = F.subsample(frames, ts, max(min_frames, len(frames) // 2))
                        row["fallback"] = row.get("fallback", 0) + 1
                        continue
                    raise
            pred = pred or ""
            row.update(correct=P.score_reply(pred, it, perm), pred=pred[:200], gold=P.gold_letter(it, perm),
                       ms=int((time.time() - t0) * 1000))
        except Exception as e:  # noqa: BLE001
            row.update(correct=None, pred=None, error=f"{type(e).__name__}: {str(e)[:300]}", ms=int((time.time() - t0) * 1000))
        return row

    t_start = time.time()
    n_ok = n_err = 0
    with open(out_p, "a") as f:
        def record(row):
            nonlocal n_ok, n_err
            if row.get("error"):
                n_err += 1
                if n_err <= 3 or n_err % 50 == 0:
                    log(f"error on {row['benchmark']}::{row['item_id']}: {row['error']}", "vi-eval")
            else:
                n_ok += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

        if getattr(model, "threaded", False) and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(run_one, it, perm) for it, perm in todo]
                for k, fu in enumerate(as_completed(futs), 1):
                    record(fu.result())
                    if k % 50 == 0:
                        log(f"{k}/{len(todo)} done ({n_ok} ok, {n_err} errors)", "vi-eval")
                    if max_minutes and (time.time() - t_start) / 60 > max_minutes:
                        log("time limit reached; stopping (rerun the command to continue)", "vi-eval")
                        ex.shutdown(cancel_futures=True)
                        break
        else:
            for k, (it, perm) in enumerate(todo, 1):
                if max_minutes and (time.time() - t_start) / 60 > max_minutes:
                    log("time limit reached; stopping (rerun the command to continue)", "vi-eval")
                    break
                record(run_one(it, perm))
                if n_err >= 20 and n_ok == 0:
                    log("20 errors and no success: stopping this protocol", "vi-eval")
                    break
                if k % 50 == 0:
                    log(f"{k}/{len(todo)} done ({n_ok} ok, {n_err} errors)", "vi-eval")
    log(f"DONE {model.name}/{pname}: {n_ok} ok, {n_err} errors, {(time.time() - t_start) / 60:.1f} min -> {out_p}", "vi-eval")
    return out_p
