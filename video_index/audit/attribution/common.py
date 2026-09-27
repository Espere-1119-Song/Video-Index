"""Shared pieces of the attribution stages: the reasoning prompt, answer extraction, the frame certificate, the
auxiliary signals shown to the judges, and the model call that records token usage and refusals."""
from __future__ import annotations

import hashlib
import json
import os
import re

from ...data.schema import read_jsonl
from ...models import ContextLimitError, ModelError, RefusalError
from ...runner import log
from ...scoring import coerce_options, mcq_letter, norm_text, num_seq, score
from .. import frame_rules
from ..items import fmt_opts, read_results
from ..items import item_options  # noqa: F401  (used by the stages through this module)

SEED = 42
DENSE_MAX = 128            # frames of the dense recheck: min(n_frames, DENSE_MAX)
TRACE_TOKENS = 8192        # token budget of a reasoning trace (attribution.trace_tokens)
JUDGE_TOKENS = 1536        # token budget of a judge reply (attribution.judge_tokens)
REFERENCE_CONDITION = "reference"
# conditions that may hold the 32-frame answers of the diagnostic open-weight model, first match wins
DIAGNOSTIC_CONDITIONS = ("reference",)
OFFSET_START, OFFSET_END = 0.016, 1.016      # shifted grid of the unstable-failure check, in units of (n - 1)

# Wording of 2026-09-22. The earlier instruction ("In your visible reply, describe ... Do not keep the reasoning
# hidden.") made one API model return refusals instead of answers; only that sentence differs.
COT_PROMPT = (
    "You are given {n} frames sampled uniformly in temporal order from a "
    "video. Answer the question.\n\n"
    "Question: {q}\n{opts}\n\n"
    "IMPORTANT: reason step by step in writing before answering: describe "
    "the relevant visual evidence you see (refer to specific frames), weigh "
    "the options, and only then decide.\n"
    "End with the final answer on its own last line, exactly in the form:\n"
    "Final answer: <option letter, or the exact short answer if no options>")

_FINAL = re.compile(r"final\s*answer\s*[:\-]*\s*", re.IGNORECASE)


def _clean_answer_line(line):
    return line.strip().strip("*#_` ").strip()


def extract_final_answer(text):
    """First non-empty line after the last 'Final answer' marker (markdown emphasis stripped); without a marker the
    last non-empty line."""
    if not text:
        return ""
    ms = list(_FINAL.finditer(text))
    if ms:
        for line in text[ms[-1].end():].splitlines():
            line = _clean_answer_line(line)
            if line:
                return line[:300]
    for line in reversed(text.splitlines()):
        line = _clean_answer_line(line)
        if line:
            return line[:300]
    return ""


def same_wrong_answer(pred_a, pred_b, options):
    """True when two wrong predictions pick the same answer. Letters are compared only with a real option list
    (>= 2 options); open answers compare by number sequence, else by normalized text."""
    options = coerce_options(options)
    if isinstance(options, (list, tuple)) and len(options) >= 2:
        la, lb = mcq_letter(pred_a, options), mcq_letter(pred_b, options)
        if la and lb:
            return la == lb
    na_, nb_ = num_seq(pred_a), num_seq(pred_b)
    if na_ is not None and nb_ is not None:
        return len(na_) == len(nb_) and all(abs(a - b) <= 1e-6 for a, b in zip(na_, nb_))
    na, nb = norm_text(pred_a), norm_text(pred_b)
    return bool(na) and na == nb


def gold_contained(pred_cot, answer, options):
    """Recovery check for questions without options: a short gold (<= 4 tokens) that appears as a token
    subsequence of the final-answer line counts as recovered."""
    if options:
        return False
    gt = norm_text(answer).split()
    if not gt or len(gt) > 4:
        return False
    pt = norm_text(pred_cot).split()
    return any(pt[i:i + len(gt)] == gt for i in range(len(pt) - len(gt) + 1))


def parse_json_obj(text):
    """JSON object of a reply; code fences and surrounding prose are tolerated. None when there is none."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        d = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
        except Exception:
            return None
    return d if isinstance(d, dict) else None


# ------------------------------------------------------------------ frames
def n_frames_of(video, meta=None):
    """Frame count used by the reference frame rule (video_index.audit.frame_rules.select)."""
    try:
        ts, n = frame_rules.timestamps_of(video, meta)
        return min(n, len(ts)) if n else len(ts)
    except Exception:  # noqa: BLE001
        return int((meta or {}).get("n_frames") or 0)


def uniform_indices(n_frames, k=32):
    """The uniform grid of the reference condition: round(i (n - 1) / (k - 1)), duplicates removed."""
    return frame_rules.reference_indices("uniform", n_frames, k, None)[0]


def offset_indices(n_frames, k=32, start=OFFSET_START, end=OFFSET_END):
    """The uniform grid moved by half an inter-frame step: the window [start, end] x (n - 1), clamped to the
    video. Every index except the last one moves."""
    if n_frames <= 0:
        return []
    return frame_rules._clamp(frame_rules._grid(n_frames, k, start * (n_frames - 1), end * (n_frames - 1)), n_frames)


def frame_check(video_id, n_frames, k=32):
    """sha1 of the uniform frame index list (the k grid positions, repeated indices included) and the video hash.
    The trace stage and the judge stage compute it from the same inputs; equality certifies that the judge sees the
    frames the erring model saw."""
    if not n_frames:
        return "nframes_unknown:" + str(video_id)[:16]
    idx = [min(max(0, i), n_frames - 1) for i in frame_rules._grid(n_frames, k)]
    return hashlib.sha1((",".join(map(str, idx)) + "|" + str(video_id)).encode()).hexdigest()


def decode(ctx, video, indices):
    """Frames at the given indices at the resolution of the reference condition."""
    return frame_rules.decode(video, indices, long_side=ctx.get("frame_long_side", 768))[0]


def uniform_frames(ctx, video, meta, k):
    return decode(ctx, video, uniform_indices(n_frames_of(video, meta), k))


def options_block(ctx, srow):
    """The options as the reference prompt shows them."""
    return fmt_opts(srow.get("options"), ctx.get("letter_options", True))


# ------------------------------------------------------------------ model calls
def budget(ctx, kind):
    """Token budget of a request: kind = "trace" | "judge". Models that spend output tokens on hidden reasoning
    need a larger judge budget (the study used 4,096 for such a judge)."""
    default = TRACE_TOKENS if kind == "trace" else JUDGE_TOKENS
    return int((ctx.get("attribution") or {}).get(f"{kind}_tokens", default))


def new_usage():
    return dict(in_tokens=0, out_tokens=0, calls=0)


def call(model, prompt, images=(), max_tokens=None):
    """One request. Returns (text or None, usage, error); error is None, "refusal", "context" or a message.
    An empty reply is repeated once with twice the token budget. The frames are never thinned: the stages that
    use this function need the exact frame set."""
    usage = new_usage()
    budget = max_tokens or model.spec.max_tokens
    for attempt in range(2):
        try:
            r = model.generate(prompt, list(images), max_tokens=budget)
        except RefusalError:
            usage["calls"] += 1
            return None, usage, "refusal"
        except ContextLimitError as e:
            return None, usage, "context: " + str(e)[:200]
        except ModelError as e:
            return None, usage, str(e)[:300]
        usage["in_tokens"] += int(r.input_tokens or 0)
        usage["out_tokens"] += int(r.output_tokens or 0)
        usage["calls"] += 1
        if r.text:
            return r.text, usage, None
        budget *= 2
    return None, usage, "empty reply"


# ------------------------------------------------------------------ inputs
def erring_label(ctx):
    return ctx.models.label("reference")


def reference_results(ctx, bench, label=None):
    return read_results_cached(ctx, REFERENCE_CONDITION, bench, label or erring_label(ctx), "reference")


def wrong_qids(ctx, bench, samples, fmt, label=None):
    """({qid: wrong prediction}, number of reference rows). Rows that the rule scorer cannot score are neither wrong
    nor right. (None, None) when the reference run of the benchmark is missing."""
    preds = reference_results(ctx, bench, label)
    if not preds:
        return None, None
    wrong = {}
    for qid, r in preds.items():
        s = samples.get(qid)
        if s and score(r.get("pred"), s.get("answer") or r.get("answer"), item_options(s)[0], fmt) is False:
            wrong[qid] = r.get("pred", "")
    return wrong, len(preds)


def load_traces(ctx, bench, only_still_wrong=False):
    rows = {}
    for r in read_jsonl(ctx.path("attribution", stage="traces", bench=bench)):
        if "qid" in r:
            rows[(r.get("model", "?"), str(r["qid"]))] = r          # last row wins
    out = list(rows.values())
    return [r for r in out if r.get("still_wrong")] if only_still_wrong else out


def traced_benchmarks(ctx, stage="traces"):
    d = os.path.join(ctx.work_dir, "attribution", stage)
    return sorted(f[:-6] for f in os.listdir(d) if f.endswith(".jsonl")) if os.path.isdir(d) else []


def _label(ctx, role):
    try:
        return ctx.models.label(role)
    except ModelError:
        return None


def blind_row(ctx, bench, qid, erring):
    """(row, same_model): the text-only answer of the erring model when that run exists, else the one of the
    text_attacker role."""
    row = read_results_cached(ctx, "blind", bench, erring).get(qid)
    if row is not None:
        return row, True
    other = _label(ctx, "text_attacker")
    if other and other != erring:
        row = read_results_cached(ctx, "blind", bench, other).get(qid)
        if row is not None:
            return row, False
    return None, False


def read_results_cached(ctx, condition, bench, label, role=None):
    """{qid: row} of one result file; role keeps the rows of that role (a file holds one model, possibly in two
    roles). Rows without a prediction are dropped. The rows are kept on the run context and read again when the
    file has changed (size or modification time)."""
    if getattr(ctx, "_attribution_results", None) is None:
        ctx._attribution_results = {}
    p = os.path.join(ctx.work_dir, "results", condition, f"{bench}__{str(label).replace('/', '_')}.jsonl")
    try:
        st = os.stat(p)
        sig = (st.st_size, st.st_mtime_ns)
    except FileNotFoundError:
        sig = None
    hit = ctx._attribution_results.get((p, role))
    if hit is None or hit[0] != sig:
        rows = read_results(p, role) if sig is not None else {}
        hit = (sig, {q: r for q, r in rows.items() if r.get("pred") is not None})
        ctx._attribution_results[(p, role)] = hit
    return hit[1]


def aux_features(ctx, bench, qid, srow, fmt, erring):
    """Correctness of the same question under the other conditions, as context for the judges. Model names are
    never shown. Returns (list of records, text block)."""
    specs = []
    brow, same = blind_row(ctx, bench, qid, erring)
    if brow is not None:
        specs.append((brow, "blind", "no visual input (question+options only), same model" if same else
                      "no visual input (question+options only), answered by a DIFFERENT strong model",
                      "frontier" if same else "frontier_other"))
    srow1 = read_results_cached(ctx, "single_frame", bench, erring, "reference").get(qid)
    if srow1 is not None:
        specs.append((srow1, "single_frame", "a single middle frame, same model", "frontier"))
    reader = _label(ctx, "caption_reader")
    if reader:
        crow = read_results_cached(ctx, "captions", bench, reader).get(qid)
        if crow is not None:
            same_c = reader == erring
            specs.append((crow, "captions", "a text caption instead of the video, same model" if same_c else
                          "a text caption instead of the video, answered by a DIFFERENT strong model",
                          "frontier" if same_c else "frontier_other"))
    diag = _label(ctx, "attacker")
    if diag and diag != erring:
        for cond in DIAGNOSTIC_CONDITIONS:
            drow = read_results_cached(ctx, cond, bench, diag, "attacker").get(qid)
            if drow is not None:
                specs.append((drow, cond, "the same 32 frames, answered by a weaker open-source diagnostic model",
                              "diagnostic_model"))
                break
    feats, lines = [], []
    opts = item_options(srow)[0]
    for row, cond, desc, source in specs:
        ok = score(row.get("pred"), row.get("answer") or srow.get("answer"), opts, fmt)
        verdict = "not rule-scorable" if ok is None else ("CORRECT" if ok else "WRONG")
        feats.append(dict(condition=cond, source=source, correct=None if ok is None else bool(ok),
                          pred=str(row.get("pred", ""))[:200]))
        lines.append(f"- {desc} [source={source}]: {verdict} (answered: {str(row.get('pred', ''))[:80]})")
    return feats, "\n".join(lines) if lines else "(none available)"


def count_failure(stats, kind):
    stats["err"] = stats.get("err", 0) + 1
    stats[kind] = stats.get(kind, 0) + 1


__all__ = ["COT_PROMPT", "DENSE_MAX", "JUDGE_TOKENS", "SEED", "TRACE_TOKENS", "aux_features", "blind_row", "budget",
           "call",
           "count_failure", "erring_label", "extract_final_answer", "frame_check", "gold_contained", "load_traces",
           "log", "n_frames_of", "offset_indices", "options_block", "parse_json_obj", "same_wrong_answer",
           "traced_benchmarks", "uniform_frames", "uniform_indices", "wrong_qids"]
