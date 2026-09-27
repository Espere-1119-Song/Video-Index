"""Stage compose_verify: is the marked answer supported by the frames?

The ``verifier`` role sees frames at 1 fps (at most ``compose.verify_frames`` = 32, thinned uniformly; short side
224) with their timestamps, the question, the options and the marked answer, and returns one verdict:

    supported             the frames show the marked answer is correct and the alternatives wrong
    contradicted          the frames show a different option is correct
    not_verifiable        the frames neither confirm nor rule out the marked answer
    definition_dependent  the correct option hinges on a benchmark-specific definition or a subjective judgement

Candidates whose verdict is not_verifiable or contradicted are asked again with up to ``compose.recheck_frames`` =
512 frames; the second verdict replaces the first. When a request does not fit the model the frame count is
halved; the number of frames sent is stored.

Keep rule
    keep   supported; reversed clips (the marked answer describes the reversed action by construction; the
           verdict is not used); not_verifiable on videos of at least 60 s
    drop   contradicted; definition_dependent; not_verifiable on videos shorter than 60 s; no verdict

Reversed clips are named by ``compose.reversed_clip_items`` (JSON list of item ids) or by the field
``reversed_clip`` of a candidate.

Outputs (pool/compose/): verify_f32.jsonl, verify_f512.jsonl (append-only), verified.json (the kept candidates
with verdict, verdict_frames, keep_reason), verify_summary.json
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

from ..data import frames as F
from ..data.schema import JsonlWriter, read_jsonl
from ..models import ContextLimitError, ModelError, RefusalError
from ..pool.grid import video_file
from ..pool.scenes import video_duration
from ..pool.schema import LETTERS, ORDERING_RE
from ..runner import log, pmap
from . import cfg, path
from .items import load, save

VERDICTS = ["supported", "contradicted", "not_verifiable", "definition_dependent"]
RECHECK = ("not_verifiable", "contradicted")
SHORT_SIDE = 224
TOKENS = 600

INTRO = ("Frames sampled from one video at about 1 fps (uniformly thinned when the video is longer than the frame budget), "
         "in order ({n} frames; timestamps in seconds):")
QUESTION = (
    "A benchmark asks the multiple-choice question below about this video and marks one option as correct.\n\n"
    "Question: {q}\n{opts}\nMarked answer: {gold}\n\n"
    "Judge whether the marked answer is right given what is visible in the frames. Use exactly one of these verdicts:\n"
    "- supported: the frames show the marked answer is correct and the alternatives are wrong\n"
    "- contradicted: the frames show a different option is correct\n"
    "- not_verifiable: the frames neither confirm nor rule out the marked answer (the evidence is not visible, would need "
    "audio or text outside the frames, or the options differ only in details the frames cannot settle)\n"
    "- definition_dependent: which option is correct hinges on a benchmark-specific definition or a subjective judgement "
    "(for example what counts as anomalous) rather than on visible evidence\n"
    "Also give the option letter that the frames support best (null if none).\n"
    "Reply with exactly one line of JSON: {{\"verdict\": \"...\", \"visible_answer\": \"A\"|null, \"reason\": \"<one sentence>\"}}")


def build_prompt(item, timestamps):
    """The frames are sent before the text; their timestamps are listed in frame order."""
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"]))
    return (INTRO.format(n=len(timestamps)) + "\n" + ", ".join(f"t={t:.1f}s" for t in timestamps) + "\n\n"
            + QUESTION.format(q=item["question"], opts=opts, gold=LETTERS[int(item["answer_idx"])]))


def parse(text):
    """-> (verdict or None, visible answer letter or None, reason)"""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except Exception:  # noqa: BLE001
        d = {}
    if not d.get("verdict"):                               # reply cut off: read the fields one by one
        mv = re.search(r'"verdict"\s*:\s*"(\w+)"', text or "")
        ma = re.search(r'"visible_answer"\s*:\s*("([A-Za-z])"|null)', text or "")
        mr = re.search(r'"reason"\s*:\s*"([^"]*)', text or "")
        d = dict(verdict=mv.group(1) if mv else None, visible_answer=ma.group(2) if ma and ma.group(2) else None,
                 reason=mr.group(1) if mr else "")
    v = str(d.get("verdict") or "").strip().lower()
    va = d.get("visible_answer")
    return (v if v in VERDICTS else None, va.strip().upper()[:1] if isinstance(va, str) and va.strip() else None,
            str(d.get("reason", ""))[:300])


def keep_rule(verdict, reversed_clip, duration, short_s=60.0):
    """-> (keep, reason)"""
    if reversed_clip:
        return True, "reversed_clip"
    if verdict == "supported":
        return True, "supported"
    if verdict == "not_verifiable":
        if duration is not None and duration == duration and duration >= short_s:
            return True, "not_verifiable_long"
        return False, "not_verifiable_short"
    if verdict in ("contradicted", "definition_dependent"):
        return False, verdict
    return False, "no_verdict"


def final_verdict(item_id, first, recheck):
    r = recheck.get(item_id) or first.get(item_id)
    return (r["verdict"], r) if r else (None, None)


def annotate(items, first, recheck, reversed_clips, durations, short_s=60.0, source="candidates"):
    out = []
    for it in items:
        verdict, r = final_verdict(it["item_id"], first, recheck)
        keep, why = keep_rule(verdict, it["item_id"] in reversed_clips or bool(it.get("reversed_clip")),
                              durations.get(it["video_id"]), short_s)
        out.append(dict(it, verdict=verdict, verdict_frames=(r or {}).get("frames"), keep=keep, keep_reason=why,
                        source=it.get("source") or source))
    return out


def audit_item(ctx, model, item, cap):
    video = video_file(ctx, item["video_id"])
    if not video:
        return dict(item_id=item["item_id"], benchmark=item["benchmark"], error="no_video")
    frames, ts = F.fps(video, 1.0, cap, short_side=SHORT_SIDE)
    while True:
        try:
            r = model.generate(build_prompt(item, ts), frames, max_tokens=TOKENS)
            if r.stop_reason in ("max_tokens", "length") and not re.search(r'"verdict"\s*:\s*"\w+"', r.text or ""):
                r = model.generate(build_prompt(item, ts), frames, max_tokens=2000)
            break
        except ContextLimitError as e:
            if len(frames) <= 8:
                return dict(item_id=item["item_id"], benchmark=item["benchmark"], error=f"context: {str(e)[:160]}")
            frames, ts = F.subsample(frames, ts, max(8, len(frames) // 2))
        except RefusalError:
            return dict(item_id=item["item_id"], benchmark=item["benchmark"], error="refusal")
        except ModelError as e:
            return dict(item_id=item["item_id"], benchmark=item["benchmark"], error=str(e)[:200])
    v, va, reason = parse(r.text)
    return dict(item_id=item["item_id"], benchmark=item["benchmark"], verdict=v, visible_answer=va,
                gold=LETTERS[int(item["answer_idx"])], frames=len(frames), reason=reason, raw=(r.text or "")[:300])


def verdicts(p):
    return {r["item_id"]: r for r in read_jsonl(p) if r.get("verdict")}


def run_pass(ctx, items, cap, name):
    model = ctx.models["verifier"]
    out_path = path(ctx, name)
    done = verdicts(out_path)
    todo = [it for it in items if it["item_id"] not in done and it.get("video_id")
            and not re.search(ORDERING_RE, it["question"], re.I)]
    n_err = Counter()
    with JsonlWriter(out_path, key="item_id") as w:
        def one(it):
            row = audit_item(ctx, model, it, cap)
            if row.get("verdict"):
                w.write(row)
            return row

        for row in pmap(one, todo, ctx.get("workers", 8), desc=f"compose/{name}"):
            if not row.get("verdict"):
                n_err[row.get("error") or "reply not parseable"] += 1
    if n_err:
        log(f"compose_verify/{name}: without a verdict: {dict(n_err)}", "compose")
    return verdicts(out_path), len(todo), n_err


def run(ctx, benchmarks=None, limit=None):
    items = load(path(ctx, "candidates.json"))
    if benchmarks:
        items = [it for it in items if it["benchmark"] in set(benchmarks)]
    if limit:
        items = items[:limit]
    rev_p = cfg(ctx, "reversed_clip_items")
    reversed_clips = set(json.load(open(rev_p))) if rev_p and os.path.exists(rev_p) else set()
    is_rev = lambda it: it["item_id"] in reversed_clips or bool(it.get("reversed_clip"))  # noqa: E731
    first, n1, e1 = run_pass(ctx, items, int(cfg(ctx, "verify_frames")), "verify_f32.jsonl")
    again = [it for it in items if (first.get(it["item_id"]) or {}).get("verdict") in RECHECK and not is_rev(it)]
    recheck, n2, e2 = run_pass(ctx, again, int(cfg(ctx, "recheck_frames")), "verify_f512.jsonl")
    durations = {it["video_id"]: video_duration(ctx, it["video_id"]) for it in items}
    ann = annotate(items, first, recheck, reversed_clips, durations, float(cfg(ctx, "short_video_s")))
    kept = [it for it in ann if it["keep"]]
    save(path(ctx, "verified.json"), kept)
    summ = dict(n_candidates=len(items), n_asked_first=n1, n_asked_recheck=n2, n_recheck_scope=len(again),
                n_refused=e1["refusal"] + e2["refusal"], n_kept=len(kept),
                reasons=dict(Counter(it["keep_reason"] for it in ann)),
                kept_per_group=dict(Counter(it["capability"] for it in kept)))
    json.dump(summ, open(path(ctx, "verify_summary.json"), "w"), indent=1)
    log(f"compose_verify: {summ}", "compose")
    return summ
