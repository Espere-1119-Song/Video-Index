"""Breaking levels on small synthetic tables; comparison with the research code on the same input."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_audit_common import research_module, run_all  # noqa: E402

from video_index.audit import levels  # noqa: E402

# attacker names of the research code -> names of this package
NAMES = {"fixed_position": "fixed_position", "options_only": "options_only:m", "gm_blind": "blind:a", "cl_blind": "blind:b",
         "gm_1f": "single_frame:reference", "cap_frames": "captions:m", "gm_shuffle": "shuffle:reference",
         "gm_window": "window:reference", "qwen_shuffle": "shuffle:attacker", "qwen_window": "window:attacker"}
POOL = {"l1_copy": "pool:copy", "l3_retrieval": "pool:retrieval", "l4_learned": "pool:learned"}


def bernoulli(rng, qids, p, cover=1.0):
    return {q: rng.random() < p for q in qids if rng.random() < cover}


def table(seed):
    rng = random.Random(seed)
    n = rng.choice([40, 120, 300])
    qids = [f"q{i}" for i in range(n)]
    cvec = {q: 1.0 / rng.choice([2, 4, 4, 5]) for q in qids}
    ref = bernoulli(rng, qids, rng.choice([0.22, 0.3, 0.55, 0.8]), rng.choice([1.0, 0.9]))
    att = {}
    for a in NAMES:
        if rng.random() < 0.75:
            att[a] = bernoulli(rng, qids, rng.uniform(0.15, 0.8), rng.choice([1.0, 0.7, 0.05]))
    pool = {a: (rng.uniform(-0.05, 0.4), rng.choice([0.25, 0.2]), n) for a in POOL if rng.random() < 0.6}
    s128 = rng.choice([None, rng.uniform(0.2, 0.7)])
    return ref, {k: v for k, v in att.items() if v}, cvec, pool, s128


def test_hand_computed():
    qids = [f"q{i}" for i in range(100)]
    cvec = {q: 0.25 for q in qids}
    ref = {q: i < 60 for i, q in enumerate(qids)}                       # s* = 0.60
    att = {"fixed_position": {q: i < 30 for i, q in enumerate(qids)},   # margin 0.05
           "blind:m": {q: i < 40 for i, q in enumerate(qids)},          # margin 0.15
           "single_frame:reference": {q: i < 56 for i, q in enumerate(qids)},     # margin 0.31
           "shuffle:reference": {q: i < 59 for i, q in enumerate(qids)}}          # no window run: order not measured
    p = levels.profile(ref, att, cvec, {})
    assert p["measured"] == ["option", "text", "frame"]
    assert abs(p["eps"]["option"] - 0.05) < 1e-12 and abs(p["eps"]["pool"] - 0.15) < 1e-12
    assert abs(p["eps"]["frame"] - 0.31) < 1e-12 and abs(p["thr"] - 2.326 * (0.25 * 0.75 / 100) ** 0.5) < 1e-12
    assert levels.assign_level(p, 0.05)["level"] == "frame"             # 0.31 >= 0.60 - 0.25 - 0.05
    assert levels.assign_level(p, 0.03)["level"] == "unbroken"
    att["window:attacker"] = {q: i < 50 for i, q in enumerate(qids)}
    p = levels.profile(ref, att, cvec, {"pool:learned": (0.12, 0.2, 100)})
    assert p["measured"] == levels.LEVELS and abs(p["res"]["pool:learned"]["margin"] - 0.07) < 1e-12
    assert abs(p["eps"]["order"] - 0.34) < 1e-12 and levels.assign_level(p, 0.03)["level"] == "order"


def test_near_chance_and_rescreen():
    qids = [f"q{i}" for i in range(100)]
    cvec = {q: 0.25 for q in qids}
    ref = {q: i < 28 for i, q in enumerate(qids)}                       # s* - c = 0.03 <= delta
    att = {"blind:m": {q: i < 26 for i, q in enumerate(qids)}}
    p = levels.profile(ref, att, cvec, {})
    a = levels.assign_level(p, 0.05)
    assert a == dict(level="text", near=1, rescreened=0, s_used=0.28)
    a = levels.assign_level(p, 0.05, s128=0.29)                         # 128-frame accuracy still within delta
    assert a["level"] == "text" and a["rescreened"] == 0
    a = levels.assign_level(p, 0.05, s128=0.50)
    assert a == dict(level="unbroken", near=1, rescreened=1, s_used=0.50)
    assert levels.profile({}, att, cvec, {}) is None


def test_minimum_common_items():
    qids = [f"q{i}" for i in range(60)]
    cvec = {q: 0.25 for q in qids}
    ref = {q: True for q in qids}
    p = levels.profile(ref, {"blind:m": {q: True for q in qids[:29]}, "blind:n": {q: False for q in qids[:30]}}, cvec, {})
    assert set(p["res"]) == {"blind:n"} and p["dropped"] == ["blind:m n=29<30"]


def test_fixed_position():
    modal, sc, cnt = levels.fixed_position({"a": "B", "b": "B", "c": "A", "d": "C", "e": "A"})
    assert modal == "A" and sc == {"a": False, "b": False, "c": True, "d": False, "e": True}


def test_same_levels_as_the_research_code():
    ref_mod = research_module("breaking_levels", os.path.join("analysis", "apex"))
    if ref_mod is None:
        print("skipped: research code not available")
        return
    level_of = {NAMES[a]: l for l, names in ref_mod.LEVEL_ATT.items() for a in names if a in NAMES}
    level_of.update({v: "pool" for v in POOL.values()})
    seen = {}
    for seed in range(400):
        ref, att, cvec, pool, s128 = table(seed)
        a = ref_mod.profile(ref, att, cvec, pool)
        b = levels.profile(ref, {NAMES[k]: v for k, v in att.items()}, cvec, {POOL[k]: v for k, v in pool.items()},
                           level_of=level_of.get)
        assert (a is None) == (b is None)
        if a is None:
            continue
        for k in ("n_ref", "s_star", "c", "thr", "eps", "measured"):
            assert a[k] == b[k], (seed, k, a[k], b[k])
        assert {NAMES.get(k, POOL.get(k)): v["margin"] for k, v in a["res"].items()} == {k: v["margin"] for k, v in b["res"].items()}
        for delta in (0.03, 0.05, 0.08):
            x, y = ref_mod.assign_level(a, delta, s128), levels.assign_level(b, delta, s128)
            assert x == y, (seed, delta, x, y)
            seen[x["level"]] = seen.get(x["level"], 0) + 1
    print(f"    levels compared: {seen}")
    assert len(seen) >= 5


if __name__ == "__main__":
    run_all(globals())
