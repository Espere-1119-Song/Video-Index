"""Caption attack: the captioner describes the video, the caption reader answers from the text alone.

frames (enters the frame level)
    The captioner describes each of 32 frames in one sentence, one frame per call, without the question and without
    the other frames (the frames of the attacker's 32-frame grid). The reader receives the captions in frame order,
    without timestamps.  captions/<bench>__<captioner>.jsonl (kind = frame), results/captions/<bench>__<reader>.jsonl
video (reported, does not enter a level)
    The captioner writes one description of the video from 32 frames; the reader answers from it.
    captions/<bench>__<captioner>.jsonl (kind = video), results/captions_video/<bench>__<reader>.jsonl
`caption_modes` selects the modes (default: frames).
"""
from __future__ import annotations

from ..data.schema import JsonlWriter, read_jsonl
from ..runner import ask, log, pmap
from . import frame_rules, prompts
from .conditions import run_condition
from .items import fmt_opts, load_samples, select, video_of

CAPTIONER, READER = "captioner", "caption_reader"


def _videos(ctx, samples):
    out = {}
    for r in samples.values():
        if r.get("video_id") and r["video_id"] not in out and video_of(ctx, r)[0]:
            out[r["video_id"]] = r
    return out


def write_captions(ctx, bench, samples, mode):
    model, label = ctx.models[CAPTIONER], ctx.models.label(CAPTIONER)
    path = ctx.path("captions", bench=bench, model=label)
    k = int(ctx.get("reference_frames", 32))
    have = {(r["video_id"], r["kind"], r.get("index")) for r in read_jsonl(path) if r.get("caption")}
    jobs = []
    for vid, row in _videos(ctx, samples).items():
        video, meta = video_of(ctx, row)
        ts, _ = frame_rules.timestamps_of(video, meta)
        idx, _ = frame_rules.attacker_indices("uniform", ts, k, row["qid"], ctx.get("seed", 42))
        if mode == "frames":
            jobs += [dict(video_id=vid, video=video, kind="frame", index=i, t=ts[i]) for i in idx if (vid, "frame", i) not in have]
        elif (vid, "video", None) not in have:
            jobs.append(dict(video_id=vid, video=video, kind="video", index=None, indices=idx))
    side = ctx.get("caption_short_side")

    def work(j):
        if j["kind"] == "frame":
            frames, _ = frame_rules.decode(j["video"], [j["index"]], short_side=side)
            res = ask(model, prompts.CAPTION_FRAME, frames, max_tokens=int(ctx.get("frame_caption_max_tokens", 128)))
            return dict(video_id=j["video_id"], kind="frame", index=j["index"], t=j["t"], caption=res["pred"], error=res["error"])
        frames, idx = frame_rules.decode(j["video"], j["indices"], long_side=ctx.get("frame_long_side", 768))
        res = ask(model, prompts.CAPTION_VIDEO, frames, max_tokens=int(ctx.get("video_caption_max_tokens", 1024)))
        return dict(video_id=j["video_id"], kind="video", index=None, caption=res["pred"], frames=res["frames"], error=res["error"])

    if jobs:
        with JsonlWriter(path, key=("video_id", "kind", "index")) as w:
            for out in pmap(work, jobs, workers=int(ctx.get("workers", 8)), desc=f"captions {bench}", every=200):
                if out.get("caption"):
                    w.write(out)
    rows = [r for r in read_jsonl(path) if r.get("caption")]
    log(f"captions [{mode}, {label}] {bench}: {sum(r['kind'] == ('frame' if mode == 'frames' else 'video') for r in rows)} "
        f"captions on disk, {len(jobs)} requested in this run", "audit")
    return path, label


def run(ctx, benchmarks=None, limit=None):
    out = []
    letter = ctx.get("letter_options", True)
    for mode in ctx.get("caption_modes", ["frames"]):
        for bench in select(ctx, benchmarks):
            samples = load_samples(ctx, bench, limit)
            if not samples:
                continue
            path, captioner = write_captions(ctx, bench, samples, mode)
            caps = {}
            for r in read_jsonl(path):
                if r.get("caption"):
                    caps.setdefault((r["video_id"], r["kind"]), {})[r.get("index")] = r["caption"]

            def build(row, bench, mode=mode, caps=caps, captioner=captioner):
                q, o = row.get("question", ""), fmt_opts(row.get("options"), letter)
                if mode == "frames":
                    c = caps.get((row.get("video_id"), "frame"))
                    if not c:
                        return None
                    lines = [f"- {' '.join(str(c[i]).split())}" for i in sorted(c)]
                    return dict(prompt=prompts.READ_FRAME_CAPTIONS.format(n=len(lines), caps="\n".join(lines), q=q, opts=o),
                                extra=dict(n_captions=len(lines), captioner=captioner))
                c = caps.get((row.get("video_id"), "video"))
                if not c:
                    return None
                return dict(prompt=prompts.READ_VIDEO_CAPTION.format(cap=c[None], q=q, opts=o), extra=dict(captioner=captioner))

            out += run_condition(ctx, "captions" if mode == "frames" else "captions_video", READER, build, [bench], limit)
    return out
