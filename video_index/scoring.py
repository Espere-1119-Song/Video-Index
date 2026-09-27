"""Scoring rules shared by stages D/E/F.

score(pred, answer, options, question_format) -> bool | None
  * MCQ (question_format == "MCQ", or options is a list with >= 2 entries):
    extract the leading option letter from pred (regex ^\\s*\\(?([A-J])[).\\s:]
    plus bare-letter / "Answer: X" fallbacks); normalize the gold answer the
    same way (it may be "A", "(A)", "A. text", or the full option text -> the
    letter comes from its index in options). Compare letters; if either side
    yields no letter, fall back to normalized-text comparison against the
    resolved option text. A bare numeric answer ("0") resolves as an index
    into options (text match has priority). A comma-separated multi-select
    gold ("A,B,C") is scored as letter-set equality with the prediction.
    str-typed options are coerced to a list when the format is recognizable
    (JSON dict/list, python list repr, "A. ..." lines).
  * temporal grounding (gold parses as EXACTLY two floats, e.g. "[ 0.  12.6]"
    or "21 39"): extract the first two non-negative floats from pred and score
    by interval IoU >= 0.5 (industry-standard R@0.5). Pred without two floats
    falls back to normalized exact match.
  * counting / numeric (gold parses as ONE number): if pred is not itself a
    pure number sequence, extract the number from pred ("is/are/answer: N"
    pattern preferred, else the last standalone number) and compare.
  * free text (non-MCQ gold with >= FREETEXT_MIN_WORDS words): NOT rule-
    scorable -> returns None. Callers must treat None as NA (exclude from
    accuracy), never as wrong.
  * other non-MCQ: normalized exact match (lowercase, punctuation stripped,
    whitespace collapsed). If BOTH sides parse entirely as number sequences,
    compare numerically ("5" == "5.0").

chance_level(num_options, has_chance_level) -> float | None
  None unless has_chance_level is truthy ("True"/"true"/True/1). num_options
  may be junk from the CSV ('', 'NA', '4', '4 or 6'); every integer found
  contributes 1/n and the mean is returned ("4 or 6" -> (1/4+1/6)/2).

Run `python scoring.py` for the self-test.
"""
import ast
import json
import re

# leading option letter: "B", "(B)", "B.", "B) text", "[b]", "b: text"
_LEAD = re.compile(r"^\s*[\(\[]?([A-Ja-j])[\)\]\.,:]?(?:\s|$)")
# comma-separated multi-select gold: "A,B,C" / "a, c"
_MULTI = re.compile(r"^\s*[A-Ja-j](\s*,\s*[A-Ja-j])+\s*$")
# "answer is (B)" / "Answer: B." / "option B"  (the LAST occurrence wins:
# predictions often discuss wrong options before stating the answer)
_STATED = re.compile(
    r"(?:answer|option|choice)\s*(?:is|would\s+be)?\s*[:\-]?\s*[\(\[]?([A-Ja-j])[\)\]\.,:]?(?:\s|$)",
    re.IGNORECASE)
_STATED_CN = re.compile(r"答案\s*(?:是|为)?\s*[:：]?\s*[\(\[]?([A-Ja-j])(?![A-Za-z])")
# extended alphabet K-P for benches with MORE than 10 options (N1 intake:
# UCF101-AD 11, iSafetyBench 16). Selected only when len(options) > 10, so
# every bench with <=10 options keeps the exact A-J behaviour above.
_LEAD_X = re.compile(r"^\s*[\(\[]?([A-Pa-p])[\)\]\.,:]?(?:\s|$)")
_MULTI_X = re.compile(r"^\s*[A-Pa-p](\s*,\s*[A-Pa-p])+\s*$")
_STATED_X = re.compile(
    r"(?:answer|option|choice)\s*(?:is|would\s+be)?\s*[:\-]?\s*[\(\[]?([A-Pa-p])[\)\]\.,:]?(?:\s|$)",
    re.IGNORECASE)


def _wide(options):
    return isinstance(options, (list, tuple)) and len(options) > 10
# non-negative decimal number (timestamps/counts); '-' is treated as a
# separator ("10-20"), not a sign
_NUM = re.compile(r"\d+(?:\.\d+)?")
# "is/are/answer(ed)/total/count ... N" — preferred counting extraction
_NUM_STATED = re.compile(
    r"(?:\bis\b|\bare\b|\banswer\b|\banswered\b|\btotal\b|\bcount\b)"
    r"[^0-9\n]{0,20}?(\d+(?:\.\d+)?)", re.IGNORECASE)
# standalone number (not part of a word/identifier like "f16" or "1st")
_NUM_ALONE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])")

FREETEXT_MIN_WORDS = 10  # gold this long cannot be scored by exact match
IOU_THRESHOLD = 0.5      # temporal grounding R@0.5


def _pre(s):
    """Light normalization before letter extraction: full-width punctuation
    and markdown emphasis defeat the regexes ('（C）', '**C**', 'C。')."""
    s = str(s)
    s = s.replace("（", "(").replace("）", ")").replace("：", ":")
    s = s.replace("。", ".").replace("，", ",")
    return s.replace("*", "").replace("#", "")


def norm_text(s):
    s = str(s).strip().lower()
    s = re.sub(r"[‘’“”`]", "'", s)
    s = re.sub(r"[^\w\s.\-]", " ", s)     # keep word chars, '.', '-' (numbers)
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(".").strip()


def num_seq(s):
    """Parse s as a pure sequence of numbers, else None."""
    toks = re.sub(r"[\[\](){},;:]", " ", str(s)).split()
    if not toks:
        return None
    out = []
    for t in toks:
        try:
            out.append(float(t))
        except ValueError:
            return None
    return out


def coerce_options(options):
    """Normalize an options field to a list (or None). The data contract
    allows list|str|null; live str shapes: JSON dict '{"A": "Yes", ...}',
    python-list repr, and newline 'A. ...' blocks."""
    if options is None:
        return None
    if isinstance(options, (list, tuple)):
        return list(options)
    s = str(options).strip()
    if not s:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            v = parser(s)
        except Exception:
            continue
        if isinstance(v, dict) and v:
            # {'A': 'Yes', 'B': 'No'} -> values ordered by (letter) key
            return [f"{k}. {v[k]}" for k in sorted(v, key=lambda x: str(x))]
        if isinstance(v, (list, tuple)) and v:
            return [str(x) for x in v]
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    if len(lines) >= 2 and sum(
            bool(re.match(r"^\(?[A-J][).:\.]\s*", l)) for l in lines) >= 2:
        return lines
    return None


def _strip_letter_prefix(opt):
    return re.sub(r"^\s*[\(\[]?[A-Ja-j][\)\]\.,:]\s+", "", str(opt)).strip()


def mcq_letter(s, options=None):
    """Extract an option letter from a prediction / gold answer."""
    if s is None:
        return None
    s = _pre(s).strip()
    options = coerce_options(options)
    wide = _wide(options)
    m = (_LEAD_X if wide else _LEAD).match(s)
    if m:
        letter = m.group(1).upper()
        # bare 'A '/'I ' followed by more words is usually the article/pronoun
        # ('A man walks...', 'I think...'), not the option letter: only accept
        # it with an explicit delimiter or when the pred is the letter alone.
        has_delim = any(ch in m.group(0) for ch in "()[].,:")
        if has_delim or letter not in ("A", "I") or len(s.split()) == 1:
            return letter
    ms = (list((_STATED_X if wide else _STATED).finditer(s))
          or list(_STATED_CN.finditer(s)))
    if ms:
        return ms[-1].group(1).upper()
    if options:
        ns = norm_text(s)
        for i, opt in enumerate(options):
            if ns and ns in (norm_text(opt), norm_text(_strip_letter_prefix(opt))):
                return chr(65 + i)
        # option text stated followed by explanation: containment, but only
        # when EXACTLY ONE option matches (ambiguous containment stays None)
        flat = lambda t: re.sub(r"\s+", " ", re.sub(r"[.\-]", " ", t)).strip()
        nsf = flat(ns)
        hits = []
        for i, opt in enumerate(options):
            ot = flat(norm_text(_strip_letter_prefix(opt)))
            if ot and len(ot) >= 3 and f" {ot} " in f" {nsf} ":
                hits.append(i)
        if len(hits) == 1:
            return chr(65 + hits[0])
        # bare numeric index into options ("0" -> A); text match takes priority
        if re.fullmatch(r"\d{1,2}", s) and int(s) < len(options):
            return chr(65 + int(s))
    return None


def _letter_set(s, wide=False):
    """Letters of a multi-select answer. Strict comma form first; otherwise
    uppercase standalone letters ("The artifacts are A and C" -> {A, C}).
    In the loose fallback 'I' is excluded (almost always the pronoun).
    wide=True widens the alphabet to A-P (benches with >10 options)."""
    s = _pre(s)
    if (_MULTI_X if wide else _MULTI).match(s):
        return {c.upper() for c in re.findall(r"[A-Pa-p]" if wide else r"[A-Ja-j]", s)}
    toks = [t for t in re.findall(r"\b([A-P])\b" if wide else r"\b([A-J])\b", s)
            if t != "I"]
    return set(toks) if toks else None


def parse_interval(s):
    """(start, end) when s parses as exactly two numbers, else None."""
    ns = num_seq(s)
    if ns is not None and len(ns) == 2:
        return (ns[0], ns[1])
    return None


def extract_pred_interval(pred):
    """First two non-negative floats in pred ('from 10.5 to 20s' -> (10.5, 20))."""
    nums = _NUM.findall(str(pred))
    if len(nums) < 2:
        return None
    return (float(nums[0]), float(nums[1]))


def interval_iou(a, b):
    """Temporal IoU of two (start, end) intervals (order-normalized)."""
    a = (min(a), max(a))
    b = (min(b), max(b))
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    if union <= 0:  # both degenerate points
        return 1.0 if abs(a[0] - b[0]) <= 1e-6 else 0.0
    return inter / union


def extract_pred_number(pred):
    """Number stated in a verbose prediction; 'is/are/answer: N' pattern
    preferred (last such match), else the last standalone number."""
    s = str(pred).replace(",", "")   # '1,234' -> '1234'
    ms = _NUM_STATED.findall(s)
    if ms:
        return float(ms[-1])
    ms = _NUM_ALONE.findall(s)
    if ms:
        return float(ms[-1])
    return None


def score(pred, answer, options=None, question_format=None):
    """True/False = rule-scored; None = NOT rule-scorable (free-text gold).
    Callers must treat None as NA — excluded from accuracy, never 'wrong'."""
    if pred is None or answer is None:
        return False
    options = coerce_options(options)
    wide = _wide(options)
    if (_MULTI_X if wide else _MULTI).match(_pre(answer)):  # multi-select gold "A,B,C"
        return _letter_set(pred, wide) == _letter_set(answer, wide)
    is_mcq = ((question_format or "").strip().upper() == "MCQ"
              or (isinstance(options, (list, tuple)) and len(options) >= 2))
    if is_mcq and isinstance(options, (list, tuple)) and options:
        pl = mcq_letter(pred, options)
        al = mcq_letter(answer, options)
        if pl and al:
            return pl == al
        # one side unresolvable -> compare texts (resolve letters to option text)
        ptxt = (norm_text(_strip_letter_prefix(options[ord(pl) - 65]))
                if pl and ord(pl) - 65 < len(options) else norm_text(pred))
        atxt = (norm_text(_strip_letter_prefix(options[ord(al) - 65]))
                if al and ord(al) - 65 < len(options) else norm_text(answer))
        return bool(ptxt) and ptxt == atxt
    if is_mcq:  # MCQ without an options list: letter-vs-letter if possible
        pl, al = mcq_letter(pred), mcq_letter(answer)
        if pl and al:
            return pl == al
    # ---- temporal grounding: gold is exactly two floats -> IoU >= 0.5
    gold_iv = parse_interval(answer)
    if gold_iv is not None:
        pred_iv = (parse_interval(pred) if num_seq(pred) is not None
                   else extract_pred_interval(pred))
        if pred_iv is not None:
            return interval_iou(pred_iv, gold_iv) >= IOU_THRESHOLD
        return bool(norm_text(pred)) and norm_text(pred) == norm_text(answer)
    pn, an = num_seq(pred), num_seq(answer)
    if pn is not None and an is not None:
        return len(pn) == len(an) and all(abs(a - b) <= 1e-6 for a, b in zip(pn, an))
    # ---- counting/numeric gold with a verbose prediction: extract the number
    if an is not None and len(an) == 1:
        pv = extract_pred_number(pred)
        if pv is not None:
            return abs(pv - an[0]) <= 1e-6
        return bool(norm_text(pred)) and norm_text(pred) == norm_text(answer)
    # ---- long free-text gold: not rule-scorable -> NA
    if len(norm_text(answer).split()) >= FREETEXT_MIN_WORDS:
        return None
    return bool(norm_text(pred)) and norm_text(pred) == norm_text(answer)


def chance_level(num_options, has_chance_level):
    flag = str(has_chance_level).strip().lower()
    if flag not in ("true", "1", "yes"):
        return None
    ns = [int(x) for x in re.findall(r"\d+", str(num_options or "")) if int(x) > 0]
    if not ns:
        return None
    return sum(1.0 / n for n in ns) / len(ns)


# ---------------------------------------------------------------- self-test
def _selftest():
    OPTS = ["Holding something", "Releasing something", "Not sure"]
    LOPTS = ["A. Turn right and walk.", "B. Turn left and walk.", "C. Stay put."]
    cases = [
        # (pred, answer, options, fmt, expected)
        ("B", "B", LOPTS, "MCQ", True),                       # bare letter
        ("(B)", "B", LOPTS, "MCQ", True),                     # parenthesized
        ("b) Turn left and walk.", "B", LOPTS, "MCQ", True),  # lowercase + text
        ("B.", "C", LOPTS, "MCQ", False),                     # wrong letter
        ("The answer is (C)", "C", LOPTS, "MCQ", True),       # stated answer
        ("Answer: A", "A. Turn right and walk.", LOPTS, "MCQ", True),  # gold has text
        ("Turn left and walk.", "B", LOPTS, "MCQ", True),     # pred is option text
        ("Releasing something", "Releasing something", OPTS, "mixed", True),  # unlabeled opts
        ("A", "Holding something", OPTS, "MCQ", True),        # letter vs option text
        ("B", "Holding something", OPTS, "MCQ", False),
        ("As shown, people run.", "C", LOPTS, "MCQ", False),  # 'As' is not letter A
        ("Not sure.", "Not sure", OPTS, "MCQ", True),         # trailing period
        # numeric index answers ("0" -> options[0]); text match has priority
        ("A", "0", OPTS, "MCQ", True),                        # gold is index
        ("Holding something", "0", OPTS, "MCQ", True),        # pred text vs gold index
        ("B", "0", OPTS, "MCQ", False),
        ("1", "B", ["cat", "dog", "fox"], "MCQ", True),       # pred is index
        ("3", "5", ["5", "3", "1"], "MCQ", False),            # "3" is option text (B), not index
        # multi-select "A,B,C": normalized letter-set equality
        ("A,C", "C, A", LOPTS, "MCQ", True),
        ("B, A and C", "A,B,C", None, "MCQ", True),
        ("The artifacts are A and C.", "A,C", None, "MCQ", True),
        ("A", "A,B", None, "MCQ", False),
        ("A,B,C", "A,B", None, "MCQ", False),
        # non-MCQ text
        ("  Yes. ", "yes", None, "open", True),
        ("A dog", "a  dog!", None, "open", True),
        ("dog", "cat", None, "open", False),
        ("", "cat", None, "open", False),                     # empty pred never correct
        # numeric equivalence
        ("5", "5.0", None, "numeric", True),
        ("5", "6", None, "numeric", False),
        # ---- letter-extraction robustness (review fixes) -----------------
        ("I think the answer is B", "B", LOPTS, "MCQ", True),   # 'I' not letter I
        ("A man walks by; the answer is C", "C", LOPTS, "MCQ", True),
        ("A man walks in the park.", "A", LOPTS, "MCQ", False),  # article, no signal
        ("**C**", "C", LOPTS, "MCQ", True),                   # markdown bold
        ("（C）", "C", LOPTS, "MCQ", True),                    # full-width parens
        ("C。", "C", LOPTS, "MCQ", True),                      # full-width period
        ("答案是C", "C", LOPTS, "MCQ", True),                  # Chinese stated
        ("Option A is wrong. The answer is B.", "B", LOPTS, "MCQ", True),  # last stated wins
        ("Turn left and walk. That matches what happens.", "B", LOPTS, "MCQ", True),  # opt text + explanation
        ("The artifacts I see are A and C.", "A,C", None, "MCQ", True),   # 'I' in multi-select prose
        # str-typed options are coerced
        ("A", "A", '{"A": "Yes", "B": "No"}', "MCQ", True),
        ("Yes", "A", '{"A": "Yes", "B": "No"}', "MCQ", True),
        ("B", "A", "['Yes', 'No']", None, False),
        # ---- temporal grounding: IoU >= 0.5 (R@0.5) ----------------------
        ("0, 12.6", "[ 0.  12.6]", None, "grounding", True),   # exact interval
        ("0.0 12.0", "[ 0.  12.6]", None, "grounding", True),  # IoU 0.95
        ("From 10.2 to 20.5 seconds.", "[10, 21]", None, "grounding", True),
        ("5 - 8", "[20, 30]", None, "grounding", False),       # IoU 0
        ("10-20", "[12, 22]", None, "grounding", True),        # '-' as separator, IoU 0.67
        ("around 14", "[10, 20]", None, "grounding", False),   # one float -> exact fallback
        ("2702 2715", "[2702. 2715.]", None, "grounding", True),  # numpy repr gold
        ("[21, 39]", "[21, 39]", None, "open", True),
        ("22 38", "[21, 39]", None, "open", True),             # IoU 16/18
        # ---- counting: extract the number from a sentence ----------------
        ("There are 13 repetitions.", "13", None, "numeric", True),
        ("The person does 12 push-ups in total, so the answer is 12.", "12", None, "numeric", True),
        ("I counted 5 pull-ups and then 7 squats.", "7", None, "numeric", True),  # last standalone
        ("The count is 27.", "27", None, "open", True),
        ("approximately 14.1 meters", "14.10", None, "open", True),
        ("There are 5 people.", "6", None, "numeric", False),
        ("many repetitions", "13", None, "numeric", False),    # no number -> exact fallback
        ("13", "13", None, "numeric", True),
        # ---- long free-text gold -> None (NA, not rule-scorable) ---------
        ("some answer", "A woman wearing a white coat and black pants attacks "
         "a public trash can with her feet on the street", None, "open", None),
        ("the man opens the door", "the man opens the door", None, "open", True),
    ]
    for i, (p, a, o, f, want) in enumerate(cases):
        got = score(p, a, o, f)
        assert got is want if want is None else got == want, \
            f"case {i}: score({p!r}, {a!r}) = {got}, want {want}"
    assert chance_level("4", "True") == 0.25
    assert chance_level("4 or 6", "True") == (0.25 + 1 / 6) / 2
    assert chance_level("NA", "True") is None
    assert chance_level("4", "False") is None
    assert chance_level("", True) is None
    assert abs(chance_level(5, True) - 0.2) < 1e-9
    # coerce_options shapes
    assert coerce_options('{"A": "Yes", "B": "No"}') == ["A. Yes", "B. No"]
    assert coerce_options("['x', 'y']") == ["x", "y"]
    assert coerce_options("A. foo\nB. bar\nC. baz") == ["A. foo", "B. bar", "C. baz"]
    assert coerce_options("free text blob") is None
    assert coerce_options(None) is None
    assert coerce_options(["a", "b"]) == ["a", "b"]
    # interval helpers
    assert parse_interval("[ 0.  12.6]") == (0.0, 12.6)
    assert parse_interval("12") is None
    assert extract_pred_interval("from 3.5s to 9s") == (3.5, 9.0)
    assert abs(interval_iou((0, 10), (5, 15)) - 1 / 3) < 1e-9
    assert extract_pred_number("the answer is 42.") == 42.0
    assert extract_pred_number("no digits here") is None
    print(f"scoring selftest: {len(cases)} score cases + option/interval/"
          f"number helper cases OK")


if __name__ == "__main__":
    _selftest()
