"""vi-eval: evaluate a model on Video-Index.

    vi-eval run   --model-config configs/models.yaml --role reference --protocol both --out runs/my-model
    vi-eval run   --role provider=openai_compatible,model=Qwen/Qwen3-VL-8B-Instruct,base_url=http://127.0.0.1:8000/v1,api_key_env=VLLM_API_KEY,max_frames=128 --out runs/qwen
    vi-eval run   --local my_package.my_module:MyModel --protocol video --out runs/local
    vi-eval score --run runs/my-model
    vi-eval leaderboard --runs runs/a runs/b --out leaderboard.csv
"""
from __future__ import annotations

import argparse
import os
import sys

from . import protocol as P
from . import score as S
from .dataset import HUB_ID, VideoIndex
from .local import load_model
from .run import run_protocol


def _data(a) -> VideoIndex:
    return VideoIndex(a.data, revision=getattr(a, "revision", None), cache_dir=getattr(a, "cache_dir", None))


def cmd_run(a) -> int:
    model = load_model(a.role, a.model_config, a.local, a.local_kwargs, a.max_tokens, a.name)
    data = _data(a)
    if a.fetch_all and a.protocol != "blind":
        data.fetch_all_videos()
    for p in (["video", "blind"] if a.protocol == "both" else [a.protocol]):
        run_protocol(model, data, p, a.out, a.limit, a.shard, a.workers, a.frame_cap, a.fps, a.short_side, a.min_frames,
                     a.max_minutes, a.ids, a.window)
    return cmd_score(argparse.Namespace(run=a.out, data=a.data, revision=a.revision, cache_dir=a.cache_dir, quiet=False))


def cmd_score(a) -> int:
    rows = S.load_rows(a.run)
    if not rows:
        print(f"no scored rows in {a.run}", file=sys.stderr)
        return 1
    n_total, groups = None, None
    try:
        data = _data(a)
        n_total, groups = len(data.items), {r["item_id"]: r.get("capability_group") for r in data.items}
    except Exception as e:  # noqa: BLE001
        print(f"[vi-eval] item file not available ({type(e).__name__}); groups taken from the result rows", file=sys.stderr)
    vp = next((r["protocol"] for r in rows if r["protocol"] != "blind"), "video1fps")
    tab = S.table(rows, groups, n_total, vp)
    S.write_csv(tab, os.path.join(a.run, "results.csv"))
    if not getattr(a, "quiet", False):
        print(S.render(tab))
    return 0


def cmd_leaderboard(a) -> int:
    tab = []
    n_total, groups = None, None
    try:
        data = _data(a)
        n_total, groups = len(data.items), {r["item_id"]: r.get("capability_group") for r in data.items}
    except Exception:  # noqa: BLE001
        pass
    for run in a.runs:
        rows = S.load_rows(run)
        if rows:
            vp = next((r["protocol"] for r in rows if r["protocol"] != "blind"), "video1fps")
            tab += S.table(rows, groups, n_total, vp)
    tab.sort(key=lambda d: -(d["video"] if d["video"] == d["video"] else -1))
    if a.out:
        S.write_csv(tab, a.out)
    print(S.render(tab))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vi-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data", default=HUB_ID, help="hub dataset id or a local directory with items/ and videos/")
        p.add_argument("--revision", default=None)
        p.add_argument("--cache-dir", default=None)

    r = sub.add_parser("run", help="run a model (resumable) and print its scores")
    common(r)
    r.add_argument("--model-config", default="configs/models.yaml")
    r.add_argument("--role", default=None, help="a role of the model configuration, or an inline specification")
    r.add_argument("--local", default=None, help="package.module:ClassName of an in-process model")
    r.add_argument("--local-kwargs", default=None, help="JSON object passed to the in-process model's constructor")
    r.add_argument("--name", default=None, help="label of the model in the result files")
    r.add_argument("--protocol", choices=["video", "blind", "both"], default="both")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=None, help="seeded random subset of this size")
    r.add_argument("--ids", default=None, help="item ids, comma separated, or a file with one id per line")
    r.add_argument("--window", default=None, help="start:stop, consecutive items in the order of the item file")
    r.add_argument("--shard", default=None, help="i/n")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--max-tokens", type=int, default=None)
    r.add_argument("--frame-cap", type=int, default=P.FRAME_CAP)
    r.add_argument("--fps", type=float, default=P.FPS)
    r.add_argument("--short-side", type=int, default=P.SHORT_SIDE)
    r.add_argument("--min-frames", type=int, default=8)
    r.add_argument("--max-minutes", type=float, default=None)
    r.add_argument("--fetch-all", action="store_true", help="download every video before the run")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("score", help="score a run directory")
    common(s)
    s.add_argument("--run", required=True)
    s.set_defaults(fn=cmd_score)

    lb = sub.add_parser("leaderboard", help="one table over several runs")
    common(lb)
    lb.add_argument("--runs", nargs="+", required=True)
    lb.add_argument("--out", default=None)
    lb.set_defaults(fn=cmd_leaderboard)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
