# Item pool (`video_index.pool`)

The pool holds every multiple-choice item of the audited benchmarks, removes the items that an attacker of the
pyramid solves, labels the rest, and selects 10,000 items with coverage first.

```
items/<benchmark>.jsonl ──pool_items──> pool_items_base ──pool_screen──> pool_items, funnel.csv
        ──pool_labels──> scene_labels.csv, item_capability.csv ──pool_select──> video_index_items.jsonl
```

## Running

```python
import importlib
from video_index.audit.context import Context
from video_index.pool import ORDER, STAGES

ctx = Context.from_files("configs/audit.yaml", "configs/models.yaml")
for stage in ORDER:                                   # pool_items, pool_screen, pool_labels, pool_select
    importlib.import_module(STAGES[stage]).run(ctx, benchmarks=None, limit=None)
```

Inputs: `items/<benchmark>.jsonl` with all items of a benchmark (the onboarding pipeline writes it) and the
normalized videos `videos/<video_id>/`. All outputs are below `pool/`. Tables are Parquet files when `pyarrow` is
installed, JSON Lines otherwise. The duplicate screens and the scene clusters need the optional dependency group
`embeddings`; the question templates use spaCy (`en_core_web_sm`) when it is installed.
`video_index.pool.embed.set_encoders(ctx, text=..., video=...)` replaces the two encoders of one run context.

## Models

| Role | Use | Study |
|---|---|---|
| `attacker` | options only, blind, single frame | open-weight model, 8B |
| `attacker_small` | 32 frames, 32 shuffled frames | open-weight model, 2B |
| `labeler` | scene type per video, capability per item | scene: 8B open-weight model; capability: API model |

The screen reads the log-probabilities of the option letters (`max_tokens=1, logprobs=true, top_logprobs=20`, the
assistant turn prefilled with `Answer:`). This needs an OpenAI-compatible endpoint that returns log-probabilities
(vLLM, SGLang, OpenAI); the request is built in `video_index/pool/logprobs.py` from the model specification, and
the API key comes from the environment variable named there. With another provider, or with
`pool.logprobs: false`, the model is asked for the letter as text: the margin is then not available, the removal
rules use correctness alone, and the worst-case attacker percentile cannot rank the items (it is 0 for every item).
Set `pool.assistant_prefill: false` for endpoints that do not continue a prefilled assistant turn.

Requests are sent one by one from a thread pool. A provider with a batch interface can be added with
`video_index.models.register_provider`.

## Stages

### `pool_items`

One row per item. Options written inside the question text are split out when the item has no option field.

| `format` | Rule |
|---|---|
| `mcq` | 2 to 26 options, a single answer that resolves to one option, options not a bare yes/no pair |
| `yesno` | the options are only yes/no or true/false |
| `open` | no options, more than 26 options, multi-select answer, answer not among the options |

`video_index.pool.schema.is_mcq_item(item)` is the rule for one item. Output `pool_items_base`,
`pool_items_manifest.csv`.

### `pool_screen`

Sequential funnel in pyramid order; every step runs on the items the earlier steps left, so a removed item has
exactly one reason (`removed_by`).

| Step | Role | Input | Removed when |
|---|---|---|---|
| `duplicate` | — | text embedding | same video and cosine > 0.90 inside a benchmark; the smallest item id is kept |
| `option_format` | — | options | meta option ("all of the above", ...), repeated option text, empty correct option |
| `opt_only` | `attacker` | options, 4 option orders | mean margin > log 2 and at least 3 of 4 orders correct |
| `blind` | `attacker` | question and options, 4 option orders | same rule |
| `single_frame` | `attacker` | frame 16 of the 32-frame grid | correct and margin > log 2 |
| `v32` | `attacker_small` | 32 frames | correct and margin > log 2 |
| `v32_shuffle` | `attacker_small` | the 32 frames in a seeded random order | correct and margin > log 2 |
| `xbench_duplicate` | — | text embedding | same video and cosine ≥ 0.90 across benchmarks; the benchmark with the earliest `year` keeps its copy |

Margin = log p(correct) − mean log p(other options). Option orders and the frame order use
`random.Random(f"42|{item_id}")`. Frames: long side 448. Items without a normalized video wait at the first visual
step (`pending_steps`); they count as remaining and are not eligible for the selection.

Outputs: `steps/<step>/<benchmark>.jsonl`, `dedup.jsonl`, `option_format.jsonl`, `xbench_pairs.jsonl`,
`pool_items` (item table with the screening columns), `funnel.csv` (`stage, n_entering, n_removed, n_after`),
`pool_summary.csv` (per benchmark).

### `pool_labels`

| Label | Unit | Input | Values |
|---|---|---|---|
| `scene_type` | video | 8 frames, benchmark name | 19 classes (`video_index/pool/labels.py`) |
| `setting` | video | 8 frames, benchmark name | indoor, outdoor, screen or synthetic, mixed |
| capability | item | question, options, correct answer | 18 fine categories → perception, temporal, spatial_physical, reasoning_knowledge |

Scope: items with `removed_by = none`. `pool.capability_scope: none` skips the capability labels; the composition
labels its own candidates. Outputs: `labels/videos.jsonl`, `labels/capability.jsonl`, `scene_labels.csv`,
`item_capability.csv`.

### `pool_select`

1. Scene clusters: mean SigLIP2 vector of 32 frames per video; clustering inside seven duration groups with a
   threshold calibrated to the same false-match rate in every group (reference threshold 0.90 in the group < 10 s).
2. Question templates per item.
3. Worst-case attacker percentile: each of the five margins is ranked among the eligible items; the largest
   percentile of the item is used.
4. Round-robin selection over duration groups, scene types and benchmarks, after a coverage pass that gives every
   (benchmark, task class) pair one item.

| Parameter (`pool.*`) | Default | Study | Meaning |
|---|---|---|---|
| `budget` | 10000 | 10,000 | selected items |
| `k` | 1 | 1 | items per video |
| `j`, `j_max` | 1, 4 | 1, 4 | videos per scene cluster; limit of the relaxation |
| `m` | 5 | 5 | items per (benchmark, template) |
| `share` | 0.05 | 0.05 | per-source cap: share of the budget per benchmark |
| `k_scarce` | 3 | 3 | items per video in cells whose supply is below their equal share |
| `max_pct` | unset | unset | keep only items with a worst-case percentile up to this value |
| `scene_t0`, `scene_pairs` | 0.90, 300000 | 0.90, 300,000 | reference threshold and random pairs of the calibration |
| `dedup`, `cross_benchmark` | true, true | true, true | duplicate screens |

A duration group that stays below budget / 7 raises its cluster cap step by step up to `j_max`, then doubles its
template cap, then is exempt from the per-benchmark cap; every step is recorded in `video_index_summary.json`.

Outputs: `scene_groups_videos.csv`, `scene_groups_table.csv`, `video_index_selection.csv`,
`video_index_items.jsonl`, `video_index_sources.csv`, `video_index_composition.csv`, `video_index_summary.json`.

## Counts of the study

| Quantity | Value |
|---|---|
| Selection | 10,000 of 61,976 eligible items |
| Items per duration group | 1,422 to 1,433 |
| Coverage pass | 735 of 736 (benchmark, task class) pairs |
| Per-source cap | 500 items |

`tests/test_pool_select.py` reproduces the 10,000 selected items in the same order from the files of the study
when `VIDEO_INDEX_RESEARCH_CODE` points to them.
