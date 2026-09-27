"""Per-item comparison of the three harnesses on the validation slice (same model, same items, same frame rule).

    python compare_harnesses.py --items LMUData/Video-Index/items/meta_benchmark.jsonl --vi-eval runs/vi_eval_<model> \
        --vlmevalkit runs/vlmevalkit/<model> runs/vlmevalkit_blind --lmms-eval runs/lmms_eval --out parity_report.md
"""
import argparse
import glob
import json
import os
import random

import pandas as pd


def vi_eval(run):
    out = {}
    for path in glob.glob(os.path.join(run, "*__*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            proto = "blind" if r["protocol"] == "blind" else "video"
            out[(proto, r["item_id"], tuple(r["perm"]))] = dict(reply=r["pred"], correct=float(bool(r["correct"])), frames=r["frames"])
    return out


def vlmevalkit(runs):
    out = {}
    for path in [p for run in runs for p in glob.glob(os.path.join(run, "*_judge.xlsx"))]:
        proto = "blind" if "Blind" in os.path.basename(path) else "video"
        for _, r in pd.read_excel(path).iterrows():
            out[(proto, r["item_id"], tuple(json.loads(r["perm"])))] = dict(reply=str(r["prediction"]), correct=float(r["score"]))
    return out


def permutations_for(item_id, k, n_perm=4, seed=42):
    rng = random.Random(f"{seed}|{item_id}")
    return [rng.sample(range(k), k) for _ in range(n_perm)]


def lmms_eval(run, video_task, items):
    out = {}
    for path in glob.glob(os.path.join(run, "**", "*samples_*.jsonl"), recursive=True):
        task = os.path.basename(path).split("samples_")[1].rsplit(".jsonl", 1)[0]
        if task not in (video_task, "video_index_blind"):
            continue
        proto = "blind" if task.endswith("blind") else "video"
        for line in open(path):
            r = json.loads(line)
            rec = r["video_index_acc"]
            k = len(items[rec["item_id"]]["options"])
            perm = permutations_for(rec["item_id"], k)[rec["perm_idx"]] if proto == "blind" else list(range(k))
            reply = r.get("filtered_resps") or r.get("resps")
            while isinstance(reply, list) and reply:
                reply = reply[0]
            out[(proto, rec["item_id"], tuple(perm))] = dict(reply=str(reply), correct=float(rec["score"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vi-eval", required=True)
    ap.add_argument("--vlmevalkit", required=True, nargs="+")
    ap.add_argument("--items", required=True, help="items/meta_benchmark.jsonl")
    ap.add_argument("--lmms-eval", required=True)
    ap.add_argument("--lmms-eval-png", default=None, help="a run of the frame task with the default PNG encoding (extra column)")
    ap.add_argument("--lmms-video-task", default="video_index_8frame")
    ap.add_argument("--out", default="parity_report.md")
    a = ap.parse_args()
    items = {r["item_id"]: r for r in map(json.loads, open(a.items))}
    runs = {"vi-eval": vi_eval(a.vi_eval), "VLMEvalKit": vlmevalkit(a.vlmevalkit), "lmms-eval": lmms_eval(a.lmms_eval, a.lmms_video_task, items)}
    extra = {}
    if a.lmms_eval_png:
        extra = {k: v for k, v in lmms_eval(a.lmms_eval_png, a.lmms_video_task, items).items() if k[0] == "video"}
    keys = sorted(set().union(*[set(v) for v in runs.values()]), key=lambda k: (k[0] != "video", k[1], k[2]))
    lines = ["| Protocol | Item | Option order | " + " | ".join(f"{n} reply" for n in runs) + " | " + " | ".join(f"{n} correct" for n in runs)
             + " | Same correctness |" + (" lmms-eval, PNG frames: reply | correct |" if extra else ""),
             "|---|---|---|" + "---|" * (2 * len(runs) + 1) + ("---|---|" if extra else "")]
    stats = {}
    for k in keys:
        cells = [runs[n].get(k) for n in runs]
        correct = [c["correct"] if c else None for c in cells]
        same = None not in correct and len(set(correct)) == 1
        s = stats.setdefault(k[0], dict(rows=0, complete=0, same=0, same_reply=0))
        s["rows"] += 1
        s["complete"] += None not in correct
        s["same"] += same
        s["same_reply"] += None not in correct and len({c["reply"].strip()[:40] for c in cells}) == 1
        lines.append(f"| {k[0]} | {k[1]} | {''.join(map(str, k[2]))} | "
                     + " | ".join((c["reply"].strip().replace("|", "/").replace("\n", " ")[:40] if c else "missing") for c in cells) + " | "
                     + " | ".join("missing" if c is None else str(int(c)) for c in correct) + f" | {'yes' if same else 'no'} |"
                     + ((f" {extra[k]['reply'].strip()[:40]} | {int(extra[k]['correct'])} |" if k in extra else "  |  |") if extra else ""))
    head = ["# Per-item comparison of the three harnesses", ""]
    for proto, s in stats.items():
        head.append(f"- {proto}: {s['rows']} rows, {s['complete']} present in all three harnesses, "
                    f"{s['same']} with the same correctness, {s['same_reply']} with the same reply text")
    for n, v in runs.items():
        for proto in stats:
            vals = {}
            for k, c in v.items():
                if k[0] == proto:
                    vals.setdefault(k[1], []).append(c["correct"])
            if vals:
                acc = 100 * sum(sum(x) / len(x) for x in vals.values()) / len(vals)
                head.append(f"- {n}, {proto}: accuracy {acc:.2f} over {len(vals)} items")
    if extra:
        ref = runs["vi-eval"]
        common = [k for k in extra if k in ref]
        head.append(f"- lmms-eval with the default PNG encoding of the frames, video: {sum(extra[k]['correct'] == ref[k]['correct'] for k in common)} of "
                    f"{len(common)} rows with the correctness of vi-eval, "
                    f"{sum(extra[k]['reply'].strip()[:40] == ref[k]['reply'].strip()[:40] for k in common)} with the same reply text")
    text = "\n".join(head + [""] + lines) + "\n"
    open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
