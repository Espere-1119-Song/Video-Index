"""Stage pool_labels: scene labels per video and capability labels per item, through the ``labeler`` role.

Scene (one label per video; input: 8 uniform frames and the benchmark name, no question text)
    scene_type  one of 19 event or domain classes
    setting     indoor | outdoor | screen or synthetic | mixed
    With log-probabilities the confidence is the probability of the chosen letter among the letters present in the
    top 20; without them the letter of the text reply is used and the confidence is empty.

Capability (one label per multiple-choice item; input: question, options, correct answer; no frames)
    one of 18 fine categories of the attribution taxonomy, mapped to the four capability groups
    (perception, temporal, spatial_physical, reasoning_knowledge), plus two flags read from the text:
    first_person and absence_question.

Scope: the items that are still in the pool (removed_by = none). ``pool.capability_scope`` = remaining (default) |
none; the composition stage labels its own candidates when the pool was not labelled.

Outputs (pool/): labels/videos.jsonl, labels/capability.jsonl (append-only), scene_labels.csv, item_capability.csv
"""
from __future__ import annotations

import csv
import json
import re

from ..audit.attribution.taxonomy import GROUP
from ..data.schema import JsonlWriter, read_jsonl
from ..runner import log, pmap
from . import logprobs as L
from . import schema as S
from .grid import grid_frames, spread, video_file

N_FRAMES = 8
SCENE = [
    ("surveillance_security_anomaly", "surveillance, security, or anomaly (CCTV, smart-home camera, crime, safety violation)"),
    ("medical_clinical_first_aid", "medical, clinical, or first aid (surgery, endoscopy, emergency care, nursing)"),
    ("traffic_driving_accidents", "traffic, driving, or accidents (dashcam, intersections, crashes)"),
    ("sports_exercise", "sports and exercise"),
    ("cooking_food", "cooking and food"),
    ("household_daily_life", "household and daily life (first-person chores, shopping, commuting)"),
    ("education_lecture_tutorial", "education, lecture, or tutorial (classroom, whiteboard, teaching, math explanation)"),
    ("science_experiment_lab", "science experiment or laboratory"),
    ("industrial_construction_manufacturing", "industrial, construction, or manufacturing"),
    ("film_tv_narrative", "film, TV, or narrative (movies, series, staged story)"),
    ("news_documentary_interview", "news, documentary, or interview"),
    ("music_dance_performance", "music, dance, or performance"),
    ("animals_wildlife", "animals and wildlife (including pets)"),
    ("video_games_software_ui", "video games or software interfaces (game capture, GUI operation, screen text)"),
    ("aerial_drone_urban_navigation", "aerial, drone, or urban navigation footage"),
    ("ai_generated_synthetic", "AI-generated or synthetic video (generated, rendered, animated, simulated physics)"),
    ("humor_memes_advertising", "humor, memes, or advertising"),
    ("embodied_robot_manipulation", "embodied or robot manipulation (robot arms, simulated agents)"),
    ("other", "other"),
]
SETTING = [("indoor", "indoor"), ("outdoor", "outdoor"),
           ("screen_or_synthetic", "screen recording, synthetic, rendered, or animated"), ("mixed", "mixed")]
# capability categories of the taxonomy (data and protocol categories excluded), with the definition the labeler reads
FINE = [
    ("perception_object_misidentify", "identify an object, person, animal or scene (what / who / which)"),
    ("perception_fine_grained_action", "recognise a subtle action, gesture or manipulation"),
    ("perception_attribute", "a low-level attribute: colour, shape, size, texture, material"),
    ("perception_object_counting", "count objects or people present"),
    ("perception_localization", "where in the frame / which region or object is meant"),
    ("perception_comparison", "compare two or more things seen in the video"),
    ("perception_hallucination", "whether something is present at all, or whether the video is tampered / generated"),
    ("perception_low_visual_quality", "read a cue under blur, night or low resolution"),
    ("ocr_text", "read on-screen text, captions, signs or numbers"),
    ("temporal_order", "before / after, first / last, the order of events"),
    ("temporal_localization", "when something happens (a time, a segment, a moment)"),
    ("action_counting", "how many times an action is repeated"),
    ("speed_duration", "how fast or how long"),
    ("spatial_relation", "left / right, distance, layout, viewpoint, 3D relation, navigation"),
    ("physics_causality", "physical plausibility or a cause-effect in the physical world"),
    ("metric_scale_estimation", "estimate a metric distance, size or speed in units"),
    ("multi_hop_reasoning", "integrate several observations into an inference (why, what will happen, intent, plot)"),
    ("domain_knowledge", "subject knowledge beyond what is visible (science, culture, rules, expertise)"),
]
FINE_KEYS = [k for k, _ in FINE]
_LETTER = re.compile(r"^\s*(?:answer\s*:\s*)?[\(\[]?([A-Z])(?![A-Za-z])", re.I)


def options_block(vocab):
    return "\n".join(f"{S.LETTERS[i]}. {desc}" for i, (_, desc) in enumerate(vocab))


def prompt_scene(bench, n=N_FRAMES):
    return (f"These are {n} frames sampled uniformly from one video of the video benchmark \"{bench}\". "
            "Which label best describes the event or domain shown in the video? Pick the single best option.\n"
            f"{options_block(SCENE)}\nReply with the letter only.")


def prompt_setting(bench, n=N_FRAMES):
    return (f"These are {n} frames sampled uniformly from one video of the video benchmark \"{bench}\". "
            "Where does the video take place? Pick the single best option.\n"
            f"{options_block(SETTING)}\nReply with the letter only.")


def prompt_capability(question, options, answer_letter):
    cats = "\n".join(f"{i + 1}. {k}: {d}" for i, (k, d) in enumerate(FINE))
    return (
        "A multiple-choice question was asked about a video. Which capability does answering it from the video primarily "
        "require? Judge from the question, the options and the correct answer only; pick the single best category.\n\n"
        f"Question: {question}\n{S.render_options(options)}\nCorrect answer: {answer_letter}\n\nCategories:\n{cats}\n\n"
        "Also say whether the question is phrased from a first-person / egocentric viewpoint (the camera wearer: I, my, "
        "the person recording), and whether it asks about the presence, absence or authenticity of something.\n"
        "Reply with exactly one line of JSON: {\"category\": <number>, \"first_person\": true/false, \"absence_question\": true/false}")


def parse_capability(text):
    """-> (fine category or None, first_person, absence_question)"""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except Exception:  # noqa: BLE001
        d = {}
    c = d.get("category")
    if isinstance(c, str):
        c = FINE_KEYS.index(c) + 1 if c in FINE_KEYS else int(re.sub(r"\D", "", c) or 0)
    fine = FINE_KEYS[int(c) - 1] if c and 1 <= int(c) <= len(FINE_KEYS) else None
    return fine, bool(d.get("first_person")), bool(d.get("absence_question"))


def choose(model, prompt, images, vocab, mode="auto", prefill=True):
    """-> (index or None, confidence or None, the three best indices)"""
    k = len(vocab)
    if L.supports_logprobs(model, mode):
        lp, _ = L.letter_logprobs(L.logprob_request(model, prompt, images, prefill)["top"], k)
        i, conf = L.confidence(lp)
        return i, conf, sorted(range(k), key=lambda j: -lp[j])[:3]
    m = _LETTER.match(model.generate(prompt, list(images), max_tokens=16).text or "")
    i = ord(m.group(1).upper()) - 65 if m else None
    i = i if i is not None and i < k else None
    return i, None, [] if i is None else [i]


def scene_frames(video):
    """Eight frames spread over the 32-frame grid of the screen."""
    return spread(grid_frames(video)[0], N_FRAMES)


def label_video(ctx, model, video_id, bench):
    path = video_file(ctx, video_id)
    if not path:
        return dict(video_id=video_id, benchmark=bench, status="no_video")
    images = scene_frames(path)
    mode, prefill = (ctx.get("pool") or {}).get("logprobs", "auto"), (ctx.get("pool") or {}).get("assistant_prefill", True)
    out = dict(video_id=video_id, benchmark=bench, n_frames=len(images), status="ok", model=model.name)
    for name, vocab, prompt in (("scene", SCENE, prompt_scene), ("setting", SETTING, prompt_setting)):
        i, conf, top = choose(model, prompt(bench, len(images)), images, vocab, mode, prefill)
        out["scene_type" if name == "scene" else name] = vocab[i][0] if i is not None else None
        out[f"{name}_conf"] = conf
        out[f"{name}_top3"] = [vocab[j][0] for j in top]
    return out


def label_item(model, row):
    opts = list(row["options"])
    txt = model.generate(prompt_capability(row["question"], opts, S.LETTERS[int(row["answer_idx"])]), (),
                         max_tokens=256).text
    fine, first, absent = parse_capability(txt)
    return dict(item_id=row["item_id"], benchmark=row["benchmark"], fine=fine, group=GROUP.get(fine) if fine else None,
                first_person=first, absence_question=absent, raw=(txt or "")[:200])


def label_capabilities(ctx, rows, desc="pool/labels/capability"):
    """Label the item rows that have no capability label yet. Returns {item_id: label row} of all labelled items."""
    model = ctx.models["labeler"]
    path = ctx.path("pool", name="labels/capability.jsonl")
    done = {r["item_id"] for r in read_jsonl(path) if r.get("fine")}
    todo = [r for r in rows if r["item_id"] not in done]
    n_err = 0
    with open(path, "a") as f:
        import threading
        lock = threading.Lock()

        def one(row):
            try:
                out = label_item(model, row)
            except Exception as e:  # noqa: BLE001
                out = dict(item_id=row["item_id"], benchmark=row["benchmark"], fine=None,
                           error=f"{type(e).__name__}: {str(e)[:200]}")
            with lock:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
            return out

        for out in pmap(one, todo, ctx.get("workers", 8), desc=desc, every=200):
            n_err += out.get("fine") is None
    if todo:
        log(f"{desc}: {len(todo)} items asked, {n_err} without a label", "pool")
    return {r["item_id"]: r for r in read_jsonl(path) if r.get("fine")}


def export(ctx):
    vids = {r["video_id"]: r for r in read_jsonl(ctx.path("pool", name="labels/videos.jsonl"))
            if r.get("status") == "ok" and r.get("scene_type")}
    with open(ctx.path("pool", name="scene_labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "benchmark", "scene_type", "scene_conf", "setting", "setting_conf"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(vids[k] for k in sorted(vids))
    caps = {r["item_id"]: r for r in read_jsonl(ctx.path("pool", name="labels/capability.jsonl")) if r.get("fine")}
    with open(ctx.path("pool", name="item_capability.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "benchmark", "fine", "group", "first_person", "absence_question"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(caps[k] for k in sorted(caps))
    return len(vids), len(caps)


def run(ctx, benchmarks=None, limit=None):
    df = S.read_table(ctx, "pool_items")
    df = df[df["removed_by"] == "none"]
    if benchmarks:
        df = df[df["benchmark"].isin(set(benchmarks))]
    model = ctx.models["labeler"]
    # one benchmark name per video: the benchmark with most items on it (ties: alphabetical)
    vb = (df.groupby(["video_id", "benchmark"]).size().reset_index(name="n")
          .sort_values(["video_id", "n", "benchmark"], ascending=[True, False, True]).drop_duplicates("video_id"))
    with JsonlWriter(ctx.path("pool", name="labels/videos.jsonl"), key="video_id") as w:
        todo = [dict(video_id=v, benchmark=b) for v, b in zip(vb["video_id"], vb["benchmark"])
                if v and not w.has(dict(video_id=v)) and video_file(ctx, v)]
        if limit:
            todo = todo[:limit]

        def one(r):
            try:
                out = label_video(ctx, model, r["video_id"], r["benchmark"])
            except Exception as e:  # noqa: BLE001
                log(f"pool/labels/{r['video_id']}: {type(e).__name__}: {str(e)[:160]}", "pool")
                return None
            w.write(out)
            return out

        n_v = sum(1 for _ in pmap(one, todo, ctx.get("workers", 8), desc="pool/labels/videos", every=200))
    n_i = 0
    if (ctx.get("pool") or {}).get("capability_scope", "remaining") != "none":
        rows = df.to_dict("records")
        label_capabilities(ctx, rows[:limit] if limit else rows)
        n_i = len(rows[:limit] if limit else rows)
    n_vid, n_cap = export(ctx)
    log(f"pool_labels: {n_v} videos and {n_i} items in this run; {n_vid} videos and {n_cap} items labelled", "pool")
    return dict(n_videos_labelled=n_vid, n_items_labelled=n_cap)
