"""CLI: materialize Video-Index questions as Harbor tasks.

    python -m video_index_harbor.main --output-dir datasets/video-index [--protocol video|blind]
        [--source Video-Index/Video-Index | <local dir>] [--revision SHA] [--embed-videos]
        [--task-ids ID ...] [--limit N] [--overwrite]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from video_index_harbor.adapter import REPO, VideoIndexAdapter
else:
    from .adapter import REPO, VideoIndexAdapter


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Convert Video-Index questions into Harbor task directories.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--protocol", choices=["video", "blind"], default="video")
    ap.add_argument("--source", default=REPO, help="dataset repository id, or a local directory with items/ and videos/")
    ap.add_argument("--revision", default=None, help="dataset revision to pin (default: the current commit)")
    ap.add_argument("--embed-videos", action="store_true", help="copy each mp4 into its task instead of downloading at build")
    ap.add_argument("--task-ids", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    ad = VideoIndexAdapter(a.output_dir, a.source, a.revision, a.protocol, a.embed_videos)
    n = ad.generate(a.task_ids, a.limit, a.overwrite)
    print(f"wrote {n} task(s) to {a.output_dir} (protocol {a.protocol}, dataset revision {ad.revision})")


if __name__ == "__main__":
    main()
