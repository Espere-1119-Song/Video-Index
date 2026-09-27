"""Stage attribution_discover (optional): failure patterns the taxonomy does not capture.

The free-form descriptions of the rows attributed to ``other`` or disputed between the judges are clustered by the
``judge_a`` model, in windows of 20 benchmarks. A cluster larger than 0.5 % of all attributed failures is written
as a proposal for a new category. The taxonomy itself is never changed by this stage.

Output attribution/discover/proposals.json; replies are cached in attribution/discover/cache/.
"""
from __future__ import annotations

import hashlib
import json
import os

from ...data.schema import read_jsonl
from ...runner import log
from ..items import load_samples
from . import common as C
from .taxonomy import CATEGORIES

WINDOW = 20
THRESHOLD = 0.005
TOKENS = 6144

PROMPT = (
    "You are auditing an error-attribution taxonomy for video-QA failures. "
    "The fixed taxonomy is:\n{cats}\n\n"
    "Below are failures the taxonomy did NOT capture well — attributed to "
    "'other' or DISPUTED between two judges. Each item shows the question, "
    "the judges' category votes, and their free-form failure descriptions.\n\n"
    "{items}\n\n"
    "Cluster these into coherent RECURRING failure patterns that are NOT "
    "expressible as one existing category (singletons may stay unclustered — "
    "omit them). For each cluster give: a name, a description, the item ids, "
    "a proposed_category_slug (snake_case, taxonomy style), the capability "
    "group it belongs to, and a rationale for why no existing category "
    "covers it.\n\n"
    'Reply as JSON: {{"clusters": [{{"name": "...", "description": "...", '
    '"item_ids": [0, 1], "proposed_category_slug": "...", '
    '"proposed_group": "...", "rationale": "..."}}]}}')


def load_items(ctx):
    """(number of attributed failures, rows attributed to `other` or disputed)."""
    total, items = 0, []
    for bench in C.traced_benchmarks(ctx, "judge"):
        for r in read_jsonl(ctx.path("attribution", stage="judge", bench=bench)):
            total += 1
            if r.get("category") == "other" or r.get("agreed_level") == "disputed":
                votes = r.get("by", {})
                items.append(dict(benchmark=bench, qid=str(r["qid"]), model=r.get("model"),
                                  votes={j: v.get("category") for j, v in votes.items()},
                                  free_descs=[v["free_desc"] for v in votes.values() if v.get("free_desc")],
                                  reasons=[v.get("reason", "") for v in votes.values()]))
    return total, items


def fmt_item(i, it, question):
    votes = {f"judge_{k + 1}": v for k, (_, v) in enumerate(sorted(it["votes"].items()))}
    fd = " | ".join(it["free_descs"]) or " | ".join(it["reasons"])
    return f"[{i}] bench={it['benchmark']} votes={votes}\n    Q: {str(question)[:180]}\n    free_desc: {fd[:400]}"


def clusters_of(reply, n_items, total):
    out = []
    for cl in reply.get("clusters", []):
        ids = sorted({i for i in cl.get("item_ids", []) if isinstance(i, int) and 0 <= i < n_items})
        out.append(dict(name=cl.get("name"), description=cl.get("description"),
                        proposed_category_slug=cl.get("proposed_category_slug"),
                        proposed_group=cl.get("proposed_group"), rationale=cl.get("rationale"), size=len(ids),
                        share_of_all_failures=round(len(ids) / total, 5), item_ids=ids))
    return out


def run(ctx, benchmarks=None, limit=None):
    cfg = ctx.get("attribution") or {}
    window = int(cfg.get("discover_window", WINDOW))
    total, items = load_items(ctx)
    if not items:
        log("discover: no rows attributed to `other` or disputed")
        return dict(proposals=0)
    benches = C.traced_benchmarks(ctx, "judge")
    windows = [benches[i:i + window] for i in range(0, len(benches), window)]
    if len(windows[-1]) < window and not cfg.get("discover_partial", True):
        windows.pop()
    threshold = THRESHOLD * total
    out_dir = os.path.join(ctx.work_dir, "attribution", "discover")
    os.makedirs(os.path.join(out_dir, "cache"), exist_ok=True)
    samples, proposals, records = {}, [], []
    for wi, wb in enumerate(windows):
        witems = [it for it in items if it["benchmark"] in set(wb)]
        if not witems:
            continue
        key = hashlib.sha1(json.dumps([sorted(wb), [(it["benchmark"], it["qid"], sorted(it["votes"].items()),
                                                     it["free_descs"]) for it in witems]],
                                      sort_keys=True, default=str).encode()).hexdigest()[:10]
        cache = os.path.join(out_dir, "cache", f"window{wi:02d}_{key}.json")
        if os.path.exists(cache):
            reply = json.load(open(cache))
        else:
            lines = []
            for i, it in enumerate(witems):
                if it["benchmark"] not in samples:
                    samples[it["benchmark"]] = load_samples(ctx, it["benchmark"])
                lines.append(fmt_item(i, it, samples[it["benchmark"]].get(it["qid"], {}).get("question", "")))
            text, usage, err = C.call(ctx.models["judge_a"], PROMPT.format(
                cats="\n".join(f"- {c}" for c in CATEGORIES), items="\n".join(lines)), (), TOKENS)
            reply = C.parse_json_obj(text)
            if not reply or "clusters" not in reply:        # never cached: the window is repeated by the next run
                log(f"discover: window {wi}: {err or 'reply not parseable'}")
                continue
            reply["_usage"] = usage
            json.dump(reply, open(cache, "w"), ensure_ascii=False, indent=1)
        cl = clusters_of(reply, len(witems), total)
        for c in cl:
            c["members"] = [dict(benchmark=witems[i]["benchmark"], qid=witems[i]["qid"]) for i in c.pop("item_ids")]
        records.append(dict(window=wi, benchmarks=wb, n_items=len(witems), clusters=cl))
        proposals += [c for c in cl if c["size"] > threshold]
    json.dump(dict(n_failures_total=total, n_other_disputed=len(items), threshold_items=threshold, windows=records,
                   proposals=proposals), open(os.path.join(out_dir, "proposals.json"), "w"),
              ensure_ascii=False, indent=1)
    log(f"discover: {len(proposals)} proposal(s) above {threshold:.1f} items")
    return dict(proposals=len(proposals))
