"""Format check of a benchmark's items and the eligibility rules of the audit.

Eligibility (the audit covers English multiple-choice benchmarks that do not require audio):

    gold_available    the items carry gold answers (a test split without answers cannot be audited)
    multiple_choice   the benchmark has multiple-choice items; open and yes/no items are left out
    min_items         at least 30 multiple-choice items
    english           the questions are in English
    no_audio          the questions can be answered without the audio track
"""
from __future__ import annotations

import math
import re
from collections import Counter

MIN_MCQ_ITEMS = 30
ENGLISH_SHARE = 0.95          # share of items whose text is English
AUDIO_SHARE = 0.20            # share of items that mention sound before the benchmark is flagged
_AUDIO = re.compile(r"\b(hear|heard|hears|sound|sounds|audio|listen|voice|voices|spoken|speech|says?|said|music|song|"
                    r"noise|narrat\w+|dialogue|conversation)\b", re.I)


def is_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    return sum(c.isascii() for c in letters) / len(letters) >= 0.9


def position_entropy(positions: list[int], n_options: int) -> float:
    """Entropy of the gold positions divided by log(n_options): 1 = uniform, 0 = always the same position."""
    if not positions or n_options < 2:
        return float("nan")
    n = len(positions)
    h = -sum(c / n * math.log(c / n) for c in Counter(positions).values())
    return h / math.log(n_options)


def format_report(items: list[dict], requires_audio: bool = False, language: str = "en") -> dict:
    """Counts, distributions and eligibility verdicts for the items of one benchmark (every format included)."""
    mcq = [i for i in items if i.get("format") == "mcq"]
    n = len(items)
    fm = Counter(i.get("format") for i in items)
    k_dist = Counter(len(i["options"]) for i in mcq)
    modal_k = k_dist.most_common(1)[0][0] if k_dist else 0
    pos = Counter(i["answer_idx"] for i in mcq)
    text = lambda i: f"{i.get('question', '')} " + " ".join(i.get("options") or [])
    eng = sum(is_english(text(i)) for i in mcq) / len(mcq) if mcq else 0.0
    audio = sum(bool(_AUDIO.search(str(i.get("question", "")))) for i in mcq) / len(mcq) if mcq else 0.0
    no_gold = sum(1 for i in items if not str(i.get("answer_raw", i.get("answer", ""))).strip()
                  or str(i.get("answer_raw")).strip().lower() in ("none", "nan", "-1"))
    rep = dict(
        n_items=n, n_mcq=fm.get("mcq", 0), n_open=fm.get("open", 0), n_yesno=fm.get("yesno", 0),
        open_reasons=dict(Counter(i.get("format_reason") for i in items if i.get("format") != "mcq")),
        options_from_question=sum(1 for i in mcq if i.get("options_source") == "question"),
        option_counts=dict(sorted(k_dist.items())), modal_options=modal_k,
        answer_positions={chr(65 + k): v for k, v in sorted(pos.items())},
        position_entropy=round(position_entropy([i["answer_idx"] for i in mcq if len(i["options"]) == modal_k], modal_k), 4)
        if mcq else float("nan"),
        modal_position_share=round(max(pos.values()) / len(mcq), 4) if mcq else float("nan"),
        chance=round(sum(1.0 / len(i["options"]) for i in mcq) / len(mcq), 4) if mcq else float("nan"),
        n_videos=len({str(i.get("video_ref")) for i in mcq}), n_subtasks=len({i.get("subtask") for i in mcq}),
        subtasks=dict(Counter(i.get("subtask") for i in mcq).most_common(12)),
        english_share=round(eng, 4), audio_mention_share=round(audio, 4), n_without_gold=no_gold)
    checks = [
        ("gold_available", n > 0 and no_gold < 0.5 * n,
         f"{no_gold} of {n} items carry no gold answer" + ("" if no_gold < 0.5 * max(n, 1) else
                                                          ": this looks like a test split without answers; use the split that ships them")),
        ("multiple_choice", rep["n_mcq"] > 0,
         f"{rep['n_mcq']} multiple-choice items, {rep['n_open']} open, {rep['n_yesno']} yes/no (only multiple-choice items are audited)"),
        ("min_items", rep["n_mcq"] >= MIN_MCQ_ITEMS,
         f"{rep['n_mcq']} multiple-choice items; the audit needs at least {MIN_MCQ_ITEMS}"),
        ("english", language.lower().startswith("en") and eng >= ENGLISH_SHARE,
         f"{100 * eng:.1f}% of the multiple-choice items are English text (declared language: {language}; "
         f"required: {100 * ENGLISH_SHARE:.0f}%)"),
        ("no_audio", not requires_audio and audio <= AUDIO_SHARE,
         ("the adapter spec declares requires_audio: true" if requires_audio else
          f"{100 * audio:.1f}% of the questions mention sound or speech (flagged above {100 * AUDIO_SHARE:.0f}%; "
          "set requires_audio in the spec after reading a sample of them)")),
    ]
    rep["checks"] = [dict(rule=r, passed=bool(ok), message=m) for r, ok, m in checks]
    rep["eligible"] = all(c["passed"] for c in rep["checks"])
    return rep


def render_report(name: str, rep: dict, sample_n: int = 300) -> str:
    L = [f"Format check: {name}",
         f"  items             {rep['n_items']:,}  (multiple choice {rep['n_mcq']:,}, open {rep['n_open']:,}, yes/no {rep['n_yesno']:,})"]
    if rep["open_reasons"]:
        L.append("  not multiple choice: " + ", ".join(f"{k} {v:,}" for k, v in sorted(rep["open_reasons"].items(), key=lambda x: -x[1])))
    if rep["options_from_question"]:
        L.append(f"  options read from the question text: {rep['options_from_question']:,} items")
    L += [f"  option counts     " + ", ".join(f"{k} options: {v:,}" for k, v in rep["option_counts"].items()),
          f"  gold positions    " + ", ".join(f"{k} {v:,}" for k, v in rep["answer_positions"].items()),
          f"  position entropy  {rep['position_entropy']} (normalized, items with {rep['modal_options']} options); "
          f"modal position share {rep['modal_position_share']}",
          f"  chance            {rep['chance']}",
          f"  videos            {rep['n_videos']:,}; subtasks {rep['n_subtasks']}",
          f"  audited sample    {min(sample_n, rep['n_mcq']):,} items" + (" (all multiple-choice items)" if rep["n_mcq"] <= sample_n else
                                                                        f" of {rep['n_mcq']:,}, stratified by subtask"),
          "  eligibility"]
    for c in rep["checks"]:
        L.append(f"    [{'pass' if c['passed'] else 'FAIL'}] {c['rule']}: {c['message']}")
    L.append(f"  => {'eligible' if rep['eligible'] else 'not eligible'}")
    return "\n".join(L)
