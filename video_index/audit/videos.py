"""Video stage: normalize the videos of the sample with ffmpeg and record `video_id` in the sample rows.

Rule      frame rate = min(max_fps, max_frames / duration) (2 fps, at most 1,024 frames); short side 720, and 480 for
          videos longer than 1,800 s unless the benchmark is text-heavy; never enlarged; no audio track.
Identity  video_id = SHA-256 of the source file; one directory per video: videos/<video_id>/video.mp4 and meta.json
          (durations, resolutions, frame count, frame timestamps).
Sources   per sample row, the first that applies:
            video_path   local file (absolute, or relative to the workspace); written by `vi-onboard`'s fetch stage
            video_ref    a path (absolute, or relative to `video_root` of the benchmark entry or the configuration), or
                         a mapping {repo, member | path, archive | zip_path, repo_type}: a file of a Hugging Face
                         repository, or a member of a zip / tar archive in it (the archive is downloaded once)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tarfile
import zipfile

from ..data.schema import read_jsonl, write_jsonl
from ..runner import log, pmap
from .items import bench_meta, select, video_key

ENCODERS = {"libx265": ["-c:v", "libx265", "-crf", "28", "-preset", "medium", "-g", "10", "-pix_fmt", "yuv420p"],
            "libx264": ["-c:v", "libx264", "-crf", "23", "-preset", "medium", "-g", "10", "-pix_fmt", "yuv420p"]}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def ffprobe(path, ffprobe_bin="ffprobe"):
    out = subprocess.run([ffprobe_bin, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    v = next(s for s in d["streams"] if s.get("codec_type") == "video")
    num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    rate = float(num) / float(den) if float(den or 0) else 0.0
    dur = float(d.get("format", {}).get("duration") or v.get("duration") or 0.0)
    return dict(duration=dur, fps=rate, width=int(v["width"]), height=int(v["height"]),
                has_audio=any(s.get("codec_type") == "audio" for s in d["streams"]))


def count_frames(path, ffprobe_bin="ffprobe"):
    out = subprocess.run([ffprobe_bin, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    try:
        return int(out.strip().split(",")[0])
    except ValueError:
        return 0


def plan(duration, width, height, text_heavy=False, max_fps=2.0, max_frames=1024, short_side=720, short_side_long=480,
         long_s=1800):
    """-> (output frame rate, short side, ffmpeg scale filter)."""
    out_fps = min(max_fps, max_frames / duration) if duration > 0 else max_fps
    short = short_side if (duration <= long_s or text_heavy) else short_side_long
    if min(width, height) <= short:
        scale = "scale=ceil(iw/2)*2:ceil(ih/2)*2"                 # never enlarged; even dimensions
    elif width <= height:
        scale = f"scale={short}:-2"
    else:
        scale = f"scale=-2:{short}"
    return out_fps, short, scale


def normalize(src, dest_dir, text_heavy=False, params=None, ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe",
              encoders=("libx265", "libx264"), source_key=""):
    """Normalize one source file into dest_dir (video.mp4 + meta.json); returns the meta dict."""
    p = ffprobe(src, ffprobe_bin)
    if p["duration"] <= 0:
        raise RuntimeError("source duration unknown")
    out_fps, short, scale = plan(p["duration"], p["width"], p["height"], text_heavy, **(params or {}))
    os.makedirs(dest_dir, exist_ok=True)
    tmp, used, err = os.path.join(dest_dir, "video.tmp.mp4"), None, ""
    for enc in encoders:
        r = subprocess.run([ffmpeg_bin, "-y", "-v", "error", "-i", src, "-vf", f"fps={out_fps:.6f},{scale}",
                            *ENCODERS[enc], "-an", tmp], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            used = enc
            break
        err = r.stderr[-300:]
    if used is None:
        raise RuntimeError(f"ffmpeg failed: {err}")
    n = count_frames(tmp, ffprobe_bin)
    expected = int(math.floor(p["duration"] * out_fps)) + 1
    note = ""
    if abs(n - expected) > max(3, expected * 0.02):
        if n <= 0:
            os.remove(tmp)
            raise RuntimeError("normalized file holds no frame")
        note = f"frame count {n} differs from the expected {expected} (container duration of the source)"
    os.replace(tmp, os.path.join(dest_dir, "video.mp4"))
    q = ffprobe(os.path.join(dest_dir, "video.mp4"), ffprobe_bin)
    meta = dict(content_hash=os.path.basename(dest_dir), source_key=source_key, original_duration=p["duration"],
                original_fps=p["fps"], original_resolution=[p["width"], p["height"]], normalized_fps=out_fps,
                normalized_short_side=min(q["width"], q["height"]), normalized_resolution=[q["width"], q["height"]],
                n_frames=n, frame_timestamps=[round(i / out_fps, 3) for i in range(n)], text_heavy=bool(text_heavy),
                too_short=n < 32, has_audio=p["has_audio"], encoder=used,
                norm_bytes=os.path.getsize(os.path.join(dest_dir, "video.mp4")))
    if note:
        meta["note"] = note
    with open(os.path.join(dest_dir, "meta.json.tmp"), "w") as f:
        json.dump(meta, f)
    os.replace(os.path.join(dest_dir, "meta.json.tmp"), os.path.join(dest_dir, "meta.json"))
    return meta


# ------------------------------------------------------------------ sources
def _hf_file(repo, filename, repo_type, cache_dir):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=filename, repo_type=repo_type or "dataset", local_dir=cache_dir,
                           token=os.environ.get("HF_TOKEN") or None)


def _member(archive, member, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z, z.open(member) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
    else:
        with tarfile.open(archive) as t, t.extractfile(member) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
    return dest


def locate(ctx, bench, row):
    """Local path of the source video of a sample row (downloads from the Hugging Face hub when needed)."""
    vp = row.get("video_path")
    if vp:
        p = vp if os.path.isabs(vp) else os.path.join(ctx.work_dir, vp)
        if os.path.exists(p):
            return p
    ref = row.get("video_ref")
    root = bench_meta(ctx, bench).get("video_root") or ctx.get("video_root") or ""
    if isinstance(ref, str) and ref:
        for p in (ref, os.path.join(root, ref), os.path.join(root, bench, ref)):
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"video not found: {ref} (video_root = {root or 'not set'})")
    if isinstance(ref, dict):
        repo = ref.get("repo") or ref.get("source") or bench_meta(ctx, bench).get("video_repo")
        member = ref.get("member") or ref.get("path")
        archive = ref.get("archive") or ref.get("zip_path")
        if root and member and os.path.exists(os.path.join(root, member)):
            return os.path.join(root, member)
        if not repo or not member:
            raise FileNotFoundError(f"video_ref without repository or member: {ref}")
        cache = os.path.join(ctx.work_dir, "raw_videos", bench)
        if not archive:
            return _hf_file(repo, member, ref.get("repo_type"), cache)
        dest = os.path.join(cache, "members", hashlib.sha256(video_key(row).encode()).hexdigest()[:16]
                            + os.path.splitext(member)[1])
        return dest if os.path.exists(dest) else _member(_hf_file(repo, archive, ref.get("repo_type"), cache), member, dest)
    raise FileNotFoundError("the row carries neither video_path nor video_ref")


def run(ctx, benchmarks=None, limit=None):
    params = dict(ctx.get("normalize") or {})
    bins = dict(ffmpeg_bin=ctx.get("ffmpeg", "ffmpeg"), ffprobe_bin=ctx.get("ffprobe", "ffprobe"))
    encoders = tuple(ctx.get("encoders", ("libx265", "libx264")))
    import shutil
    missing = [b for b in bins.values() if shutil.which(b) is None]
    if missing:
        raise SystemExit(f"videos: {', '.join(missing)} not found on PATH; install ffmpeg or set `ffmpeg:` / `ffprobe:` "
                         "in the run configuration to the full paths")
    summary = []
    for bench in select(ctx, benchmarks):
        sp = ctx.path("samples", bench=bench)
        rows = read_jsonl(sp)
        text_heavy = str(bench_meta(ctx, bench).get("text_heavy", "")).lower() in ("1", "true", "yes")
        todo, seen = [], set()
        for r in rows[:limit] if limit else rows:
            if r.get("video_id") and os.path.exists(ctx.path("video_meta", video_id=r["video_id"])):
                continue
            k = r.get("video_path") or video_key(r)
            if k not in seen:
                seen.add(k)
                todo.append(r)

        def work(r):
            k = r.get("video_path") or video_key(r)
            try:
                src = locate(ctx, bench, r)
                vid = sha256_file(src)
                d = os.path.dirname(ctx.path("video", video_id=vid))
                mp = os.path.join(d, "meta.json")
                meta = json.load(open(mp)) if os.path.exists(mp) else normalize(
                    src, d, text_heavy, params, encoders=encoders, source_key=video_key(r), **bins)
                return dict(key=k, video_id=vid, duration_s=meta["original_duration"], error=None)
            except Exception as e:  # noqa: BLE001
                return dict(key=k, video_id=None, error=f"{type(e).__name__}: {str(e)[:200]}")

        done = {o["key"]: o for o in pmap(work, todo, workers=int(ctx.get("video_workers", 4)), desc=f"videos {bench}",
                                          every=20)}
        failed = [o for o in done.values() if o["error"]]
        for r in rows:
            o = done.get(r.get("video_path") or video_key(r))
            if o and o["video_id"]:
                r["video_id"], r["duration_s"] = o["video_id"], o["duration_s"]
        if done:
            write_jsonl(sp, rows)
        if failed:
            write_jsonl(ctx.path("log", name=f"videos_failed_{bench}.jsonl"), failed)
        n_ok = sum(1 for r in rows if r.get("video_id"))
        log(f"videos: {bench}: {n_ok}/{len(rows)} items with a normalized video, {len(failed)} videos failed", "audit")
        summary.append(dict(benchmark=bench, n_items=len(rows), n_with_video=n_ok, n_failed=len(failed)))
    return summary
