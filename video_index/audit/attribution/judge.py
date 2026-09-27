"""Stage attribution_judge: two independent judges name the primary cause of every still-wrong item.

Each judge (roles ``judge_a`` and ``judge_b``) receives the same 32 frames the erring model saw, the question, the
options, the correct answer, the wrong answer, the full written reasoning, the auxiliary signals of the other
conditions and the vision-introduced flag. The erring model is only called "the model".

Agreement    fine (same category) | group (same group of the taxonomy) | disputed
Rule rows    items pre-classified as language_prior_dominated are written with agreed_level "rule", without a call
Dense recheck  when both judges choose evidence_not_in_input_frames, both see min(n_frames, 128) uniform frames and
             choose sampling_gap | evidence_not_in_video | audio_needed | annotation_suspect. Agreement replaces the
             category; disagreement keeps it with dense_recheck.disputed = true. Videos with at most 32 frames skip
             the recheck.
Frame check  the certificate of the trace row is recomputed; a mismatch skips the item.
Failures     a judge reply that cannot be parsed, a refusal or a failed request leaves the item unwritten; the
             next run repeats it.

Output attribution/judge/<benchmark>.jsonl, one row per (model, qid):
    qid, model, category, group, agreed_level, dense_recheck, by {role: {model, category, reason, free_desc, usage}},
    vision_introduced_error, frame_check
"""
from __future__ import annotations

import threading

from ...data.schema import JsonlWriter, read_jsonl
from ...runner import log, pmap
from ..items import load_samples, question_format, video_of
from . import common as C
from .taxonomy import CATEGORIES, DENSE_RESOLUTIONS, GROUP

JUDGE_ROLES = ("judge_a", "judge_b")

MAIN_PROMPT = (
    "A multimodal model (refer to it ONLY as \"the model\"; its identity is "
    "withheld) answered this video question INCORRECTLY. You see the SAME "
    "{n} frames the model saw (uniformly sampled, temporal order), the "
    "question, the correct answer, the model's wrong answer, and the model's "
    "full written reasoning.\n\n"
    "Question: {q}\n{opts}\n"
    "Correct answer: {ans}\n"
    "Model's wrong answer: {pred}\n\n"
    "Model's reasoning (verbatim):\n---\n{trace}\n---\n\n"
    "Auxiliary signals from controlled ablations of THIS question (context "
    "only, not ground truth):\n{aux}\n{vie}\n"
    "Diagnose the PRIMARY failure cause. Pick exactly one category:\n"
    "{cats}\n\n"
    "Note: choose evidence_not_in_input_frames when the decisive visual "
    "evidence is NOT visible in these {n} frames (fell between samples, or "
    "the frame budget cannot cover the needed span).\n\n"
    "Also produce free_desc: ONE free-form sentence describing the failure "
    "in your own words, NOT constrained to the category list — name any "
    "pattern you notice even if no category fits it.\n\n"
    'Reply as JSON: {{"category": "...", "reason": "<one sentence>", '
    '"free_desc": "<one sentence>"}}')

VIE_NOTE = ("NOTE: WITHOUT any visual input the model answered this question "
            "CORRECTLY, but with the 32 frames it answered wrongly "
            "(vision-introduced error).\n")

DENSE_PROMPT = (
    "Earlier, a failure on this video question was attributed to the "
    "decisive evidence not being visible in 32 uniformly sampled frames. "
    "You now see {n} frames from the SAME video (denser uniform sampling).\n\n"
    "Question: {q}\n{opts}\n"
    "Correct answer: {ans}\n"
    "The model's wrong answer was: {pred}\n\n"
    "With this denser frame set, decide whether the visual evidence required "
    "to derive the correct answer is actually present:\n"
    "- evidence IS visible now                          -> resolution=sampling_gap\n"
    "- evidence appears in no frame and is likely absent from the video "
    "entirely                                           -> resolution=evidence_not_in_video\n"
    "- the question most likely requires the AUDIO track -> resolution=audio_needed\n"
    "- the labeled correct answer itself looks wrong or inconsistent with "
    "the video                                          -> resolution=annotation_suspect\n\n"
    'Reply as JSON: {{"evidence_visible": true|false, "resolution": "...", '
    '"reason": "<one sentence>"}}')


def cat_lines():
    return "\n".join(f"- {c}" for c in CATEGORIES)


def parse_main(text):
    d = C.parse_json_obj(text)
    if not d or d.get("category") not in CATEGORIES:
        return None
    return dict(category=d["category"], reason=str(d.get("reason", ""))[:400],
                free_desc=str(d.get("free_desc", ""))[:400])


def parse_dense(text):
    d = C.parse_json_obj(text)
    if not d or d.get("resolution") not in DENSE_RESOLUTIONS:
        return None
    return dict(evidence_visible=bool(d.get("evidence_visible")), resolution=d["resolution"],
                reason=str(d.get("reason", ""))[:400])


def consensus(cats):
    """(category, group, agreed_level) of the judges' categories."""
    fine = len(set(cats)) == 1
    groups = {GROUP.get(c) for c in cats}
    grp_ok = len(groups) == 1 and None not in groups
    return (cats[0] if fine else None, groups.pop() if grp_ok else None,
            "fine" if fine else ("group" if grp_ok else "disputed"))


def dense_consensus(resolutions, n_frames_dense, by):
    ok = len(set(resolutions)) == 1
    return dict(triggered=True, n_frames_dense=n_frames_dense, disputed=not ok,
                final=resolutions[0] if ok else None, by=by)


def _ask(judges, prompt, images, parse, who, max_tokens=C.JUDGE_TOKENS):
    """Both judges on one prompt -> ({role: parsed}, None), or (None, failure kind)."""
    by = {}
    for role, model in judges:
        text, usage, err = C.call(model, prompt, images, max_tokens)
        parsed = parse(text) if text else None
        if parsed is None:
            log(f"{who}: {role} {'refused' if err == 'refusal' else (err or 'reply not parseable')}; item deferred")
            return None, "judge_refusal" if err == "refusal" else "judge_err"
        by[role] = dict(model=model.name, **parsed, usage=usage)
    return by, None


def judge_item(ctx, judges, bench, tr, pre, srow):
    """(row, status); row is None when the item cannot be finished in this run."""
    qid, erring = str(tr["qid"]), tr.get("model", "?")
    base = dict(qid=qid, model=erring, vision_introduced_error=bool(pre.get("vision_introduced_error")),
                frame_check=tr.get("frame_check"))
    if not pre.get("to_judge", True):
        base.update(category="language_prior_dominated", group=GROUP["language_prior_dominated"],
                    agreed_level="rule", dense_recheck=None, by={})
        return base, "rule"
    video, meta = video_of(ctx, srow)
    if not video:
        return None, "no_video"
    k = int(ctx.get("reference_frames", 32))
    n_fr = C.n_frames_of(video, meta)
    fc = C.frame_check(srow["video_id"], n_fr, k)
    if tr.get("frame_check") and fc != tr["frame_check"]:
        log(f"judge/{bench}/{qid}: frame check differs from the trace; skipped")
        return None, "frame_check_mismatch"
    try:
        images = C.uniform_frames(ctx, video, meta, k)
    except Exception as e:  # noqa: BLE001
        log(f"judge/{bench}/{qid}: frames: {str(e)[:150]}")
        return None, "frame_err"
    if not images:
        return None, "frame_err"
    _, aux = C.aux_features(ctx, bench, qid, srow, question_format(ctx, bench), erring)
    pred = pre.get("mf_pred_used") or tr.get("pred_cot", "")
    opts = C.options_block(ctx, srow)
    prompt = MAIN_PROMPT.format(n=len(images), q=srow["question"], opts=opts, ans=srow.get("answer", ""), pred=pred,
                                trace=str(tr.get("trace", ""))[:8000], aux=aux,
                                vie=VIE_NOTE if pre.get("vision_introduced_error") else "", cats=cat_lines())
    tokens = C.budget(ctx, "judge")
    by, fail = _ask(judges, prompt, images, parse_main, f"judge/{bench}/{qid}", tokens)
    if by is None:
        return None, fail
    category, group, level = consensus([by[r]["category"] for r, _ in judges])

    dense = None
    if level == "fine" and category == "evidence_not_in_input_frames":
        if n_fr and n_fr <= k:
            dense = dict(triggered=False, skipped=f"video has <={k} frames; the uniform frames already covered it")
        else:
            try:
                dimages = C.uniform_frames(ctx, video, meta, min(n_fr, C.DENSE_MAX) if n_fr else C.DENSE_MAX)
            except Exception as e:  # noqa: BLE001
                dimages, dense = None, dict(triggered=False, skipped=f"dense frames failed: {str(e)[:120]}")
            if dimages:
                dprompt = DENSE_PROMPT.format(n=len(dimages), q=srow["question"], opts=opts,
                                              ans=srow.get("answer", ""), pred=pred)
                dby, fail = _ask(judges, dprompt, dimages, parse_dense, f"judge-dense/{bench}/{qid}", tokens)
                if dby is None:
                    return None, "dense_" + fail
                dense = dense_consensus([dby[r]["resolution"] for r, _ in judges], len(dimages), dby)
                if dense["final"]:
                    category, group = dense["final"], GROUP[dense["final"]]
    base.update(category=category, group=group, agreed_level=level, dense_recheck=dense, by=by, frame_check=fc)
    return base, "ok"


def run_bench(ctx, judges, bench, limit=None):
    samples = load_samples(ctx, bench)
    traces = {(t.get("model", "?"), str(t["qid"])): t for t in C.load_traces(ctx, bench, only_still_wrong=True)}
    pre = {(r.get("model", "?"), str(r["qid"])): r
           for r in read_jsonl(ctx.path("attribution", stage="preclass", bench=bench))}
    if not pre:
        log(f"judge/{bench}: no pre-classification rows; run attribution_preclass first")
        return dict(skipped=1)
    st = dict(new=0, rule=0, dense=0, refused=0)
    lock = threading.Lock()
    with JsonlWriter(ctx.path("attribution", stage="judge", bench=bench), key=("model", "qid")) as w:
        todo = [k for k in sorted(traces) if k in pre and k[1] in samples and not w.has(dict(model=k[0], qid=k[1]))]
        st["missing_preclass"] = sum(1 for k in traces if k not in pre)
        if limit:
            todo = todo[:limit]

        def one(key):
            row, status = judge_item(ctx, judges, bench, traces[key], pre[key], samples[key[1]])
            with lock:
                if row is None:
                    C.count_failure(st, status)
                    st["refused"] += status.endswith("judge_refusal")
                    return None
                st["new"] += 1
                st["rule"] += status == "rule"
                st["dense"] += bool(row.get("dense_recheck") and row["dense_recheck"].get("triggered"))
            w.write(row)
            return row

        for _ in pmap(one, todo, ctx.get("workers", 8), desc=f"judge/{bench}"):
            pass
    return st


def run(ctx, benchmarks=None, limit=None):
    judges = [(r, ctx.models[r]) for r in JUDGE_ROLES]
    if judges[0][1].name == judges[1][1].name:
        log("judge: judge_a and judge_b are the same model; agreement levels then measure repeatability only")
    out = {}
    for bench in (b for b in C.traced_benchmarks(ctx) if not benchmarks or b in benchmarks):
        out[bench] = run_bench(ctx, judges, bench, limit)
        log(f"judge/{bench}: {out[bench]}")
    return out
