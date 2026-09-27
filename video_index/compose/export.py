"""Stage compose_export: the release layout of Video-Index.

    <work_dir>/release/
        README.md                     dataset card: construction, protocols, fields, licences
        items/meta_benchmark.jsonl    one line per item: item_id, benchmark, capability_group, fine_category,
                                      question, options, answer (letter), answer_idx, chance, video, video_id,
                                      duration_s, attack_pct, verification {verdict, keep_reason}, source_version,
                                      license
        items/summary.json            counts per group, benchmark and verification reason
        videos/<video_id>.mp4         the normalized videos (links; copies with ``compose.copy_videos: true``)
        code/prompt.md                prompts, frame sampling and scoring of the two evaluation protocols

`video_index.evaluate` reads this layout from a local directory or from the Hugging Face hub. Upload is a separate,
explicit step: ``compose.upload: true`` with ``compose.hf_repo`` (the token is read from the environment by
huggingface_hub); the repository is created private unless ``compose.public: true``.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import statistics
import time
from collections import Counter

from ..pool import schema as S
from ..pool.grid import video_file
from ..pool.scenes import video_duration
from ..runner import log
from . import cfg, path
from .items import load

GROUP_LABEL = {"perception": "Perception", "temporal": "Temporal", "spatial_physical": "Spatial / physical",
               "reasoning_knowledge": "Reasoning / knowledge"}

PROMPT_MD = """# Protocols

## VIDEO
Frames: one frame per second of the timeline, at most 512 frames per item (uniform subsample beyond), each frame
resized to short side 224. When a model's context or memory rejects the request the frame count is halved and the
request retried; the frames actually sent are recorded per item.

Prompt (frames first, then the text):
    You are given {n} frame(s) sampled from a video. Answer the question based on these frames.

    Question: {question}
    A. ...
    B. ...
    Reply with ONLY the option letter (or the exact short answer if no options).

## BLIND
Question and options only, no frames; four option permutations per item (seed 42,
`video_index.evaluate.protocol.permutations_for`); the blind score is the mean over the four permutations.

## Scoring
The rule scorer (`video_index.scoring.score`): the first option letter in the reply, or the option whose text the
reply repeats; a reply that names no option is wrong. Gain = VIDEO - BLIND on the same items.
"""

README = """---
license: other
license_name: mixed-source-licenses
task_categories:
- visual-question-answering
language:
- en
pretty_name: Video-Index ({tag})
size_categories:
- n<1K
configs:
- config_name: meta_benchmark
  data_files: items/meta_benchmark.jsonl
  default: true
---
# Video-Index ({tag})

{n_items} multiple-choice video questions from {n_benchmarks} public video benchmarks, one item per video, four
capability groups (Perception, Temporal, Spatial / physical, Reasoning / knowledge). The items are the hardest items
of a screened pool: no attacker of the pool screen (options only, question and options without frames, single frame,
32 frames with a small model, 32 shuffled frames) solved them; they are ranked by the worst-case attacker percentile
and their marked answers are verified against the frames. No model's accuracy constrains the selection.

## Construction
1. Candidates: the screened pool; the {n_candidates} items with the smallest worst-case attacker percentile, one per
   video.
2. Capability labels: `{labeler}` assigns one of 18 fine categories from the question, the options and the marked
   answer; the categories map to the four groups.
3. Verification of the marked answer: `{verifier}` reads the frames ({verify_frames}, then {recheck_frames} for
   unconfirmed items), the question, the options and the marked answer and returns supported / contradicted /
   not verifiable / definition dependent. Kept: supported; reversed clips; not verifiable on videos of at least
   {short_s:.0f} s. Dropped: contradicted; definition dependent; not verifiable on shorter videos. Ordering questions
   whose options are the events themselves are excluded.
4. Selection: per group the verified items in the order of the worst-case attacker percentile (hardest first) until
   {quota}, one item per video.
5. Protocols: VIDEO (1 fps, at most 512 frames, short side 224) and BLIND (question and options, 4 permutations);
   see `code/prompt.md`.

## Files
- `items/meta_benchmark.jsonl`: one line per item (fields: item_id, benchmark, capability_group, fine_category,
  question, options, answer, answer_idx, chance, video, video_id, duration_s, attack_pct, verification,
  source_version, license).
- `items/summary.json`: counts per group, benchmark and verification reason.
- `videos/<video_id>.mp4`: normalized videos ({n_videos} files).
- `code/prompt.md`: prompts, frame sampling and scoring.

## Composition
Groups: {per_group}. Largest sources: {largest}.
Verification reasons: {verification}. Median duration {duration_median:.0f} s; {n_long} videos of at least 300 s;
mean chance {mean_chance:.1f}%.

## Licences
Every item keeps the licence of its source benchmark (field `license`; {licenses}). Items are redistributed for
research evaluation only; check the terms of the source benchmark before any other use.
"""


def _lookup(ctx):
    """(fine category per item, licence per benchmark) from the pool files, where they exist."""
    fine, lic = {}, dict(ctx.get("licenses") or {})
    p = ctx.path("pool", name="item_capability.csv")
    if not os.path.exists(p):
        from ..data.schema import read_jsonl
        fine = {r["item_id"]: r["fine"] for r in read_jsonl(ctx.path("pool", name="labels/capability.jsonl"))
                if r.get("fine")}
    else:
        fine = {r["item_id"]: r["fine"] for r in csv.DictReader(open(p))}
    try:
        pool = S.read_table(ctx, "pool_items", columns=["benchmark", "license"])
        pool = pool[pool["license"].fillna("") != ""].drop_duplicates("benchmark")
        lic = {**dict(zip(pool["benchmark"], pool["license"])), **lic}
    except Exception:  # noqa: BLE001
        pass
    return fine, lic


def build_rows(ctx, items):
    fine, lic = _lookup(ctx)
    rows = []
    for it in items:
        vid = it.get("video_id")
        if not vid:
            continue
        d = video_duration(ctx, vid)
        rows.append(dict(item_id=it["item_id"], benchmark=it["benchmark"], capability_group=it.get("capability"),
                         capability_group_label=GROUP_LABEL.get(it.get("capability"), it.get("capability")),
                         fine_category=fine.get(it["item_id"]), question=it["question"], options=it["options"],
                         answer=S.LETTERS[int(it["answer_idx"])], answer_idx=int(it["answer_idx"]),
                         chance=it.get("chance"), video=f"videos/{vid}.mp4", video_id=vid,
                         duration_s=None if d != d else d, attack_pct=it.get("attack_pct"),
                         verification=dict(verdict=it.get("verdict"), keep_reason=it.get("keep_reason")),
                         source_version=it.get("source"), license=lic.get(it["benchmark"], "unknown")))
    return rows


def summary(rows, tag):
    durs = [r["duration_s"] for r in rows if r["duration_s"] is not None]
    chances = [r["chance"] for r in rows if r["chance"] is not None]
    return dict(tag=tag, generated=time.strftime("%Y-%m-%d %H:%M:%S"), n_items=len(rows),
                n_videos=len({r["video_id"] for r in rows}), n_benchmarks=len({r["benchmark"] for r in rows}),
                per_group=dict(Counter(r["capability_group"] for r in rows)),
                per_benchmark=dict(Counter(r["benchmark"] for r in rows).most_common()),
                verification=dict(Counter(r["verification"]["keep_reason"] for r in rows)),
                duration_median_s=statistics.median(durs) if durs else None,
                n_long_video_ge_300s=sum(1 for d in durs if d >= 300),
                mean_chance=statistics.mean(chances) if chances else None,
                licenses=dict(Counter(r["license"] for r in rows)))


def _label(ctx, role):
    try:
        return ctx.models.label(role)
    except Exception:  # noqa: BLE001
        return f"the {role} model"


def upload(out, repo, tag, private=True):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    have = set(api.list_repo_files(repo, repo_type="dataset"))
    for sub in ["README.md", "items", "code"]:
        p = os.path.join(out, sub)
        if os.path.isdir(p):
            api.upload_folder(folder_path=p, path_in_repo=sub, repo_id=repo, repo_type="dataset",
                              commit_message=f"{tag}: {sub}")
        else:
            api.upload_file(path_or_fileobj=p, path_in_repo=sub, repo_id=repo, repo_type="dataset",
                            commit_message=f"{tag}: {sub}")
    todo = sorted(f for f in os.listdir(os.path.join(out, "videos")) if f"videos/{f}" not in have)
    for i in range(0, len(todo), 40):
        api.upload_folder(folder_path=os.path.join(out, "videos"), path_in_repo="videos", repo_id=repo,
                          repo_type="dataset", allow_patterns=todo[i:i + 40],
                          commit_message=f"{tag}: videos {i // 40 + 1}")
    return len(todo)


def run(ctx, benchmarks=None, limit=None):
    items = load(path(ctx, "video_index.json"))
    if limit:
        items = items[:limit]
    tag = cfg(ctx, "tag") or "v1"
    out = os.path.join(ctx.work_dir, "release")
    for sub in ("items", "code", "videos"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    rows = build_rows(ctx, items)
    n_video, n_missing = 0, 0
    for r in rows:
        src, dst = video_file(ctx, r["video_id"]), os.path.join(out, r["video"])
        if not src:
            n_missing += 1
        elif not os.path.lexists(dst):
            if cfg(ctx, "copy_videos"):
                shutil.copyfile(src, dst)
            else:
                os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
            n_video += 1
    with open(os.path.join(out, "items", "meta_benchmark.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = summary(rows, tag)
    json.dump(summ, open(os.path.join(out, "items", "summary.json"), "w"), indent=1)
    open(os.path.join(out, "code", "prompt.md"), "w").write(PROMPT_MD)
    n_groups = max(len(summ["per_group"]), 1)
    open(os.path.join(out, "README.md"), "w").write(README.format(
        tag=tag, n_items=summ["n_items"], n_benchmarks=summ["n_benchmarks"], n_videos=summ["n_videos"],
        n_candidates=f"{int(cfg(ctx, 'candidates')):,}", labeler=_label(ctx, "labeler"),
        verifier=_label(ctx, "verifier"), verify_frames=cfg(ctx, "verify_frames"),
        recheck_frames=cfg(ctx, "recheck_frames"), short_s=float(cfg(ctx, "short_video_s")),
        quota=int(cfg(ctx, "budget")) // n_groups if not limit else len(rows) // n_groups,
        per_group=summ["per_group"], largest=dict(list(summ["per_benchmark"].items())[:8]),
        verification=summ["verification"], duration_median=summ["duration_median_s"] or 0,
        n_long=summ["n_long_video_ge_300s"], mean_chance=100 * (summ["mean_chance"] or 0),
        licenses=summ["licenses"]))
    log(f"compose_export: {len(rows)} items, {n_video} new videos, {n_missing} videos not in the workspace -> {out}",
        "compose")
    res = dict(out_dir=out, n_items=len(rows), n_videos_missing=n_missing)
    if cfg(ctx, "upload") and cfg(ctx, "hf_repo"):
        res["n_videos_uploaded"] = upload(out, cfg(ctx, "hf_repo"), tag, private=not cfg(ctx, "public"))
    return res
