# Composition of Video-Index (`video_index.compose`)

Video-Index is the set of the hardest verified items of the screened pool: 840 items, 210 per capability group, one
item per video.

```
pool/pool_items ──candidates──> compose/candidates.json ──verify──> compose/verified.json
                                  ──select──> compose/video_index.json ──export──> release/
```

## Running

```python
import importlib
from video_index.audit.context import Context
from video_index.compose import ORDER, STAGES

ctx = Context.from_files("configs/audit.yaml", "configs/models.yaml")
for stage in ORDER:                 # compose_candidates, compose_verify, compose_select, compose_export
    importlib.import_module(STAGES[stage]).run(ctx, benchmarks=None, limit=None)
```

Input: `pool/pool_items` of the stage `pool_screen` and the normalized videos. Intermediate files are below
`pool/compose/`, the release below `release/`.

## Models

| Role | Use | Fallback |
|---|---|---|
| `labeler` | capability label of every candidate (question, options, correct answer; no frames) | `judge_a` |
| `verifier` | checks the marked answer against the frames | `reference` |

Requests are sent one by one from a thread pool (`workers`); a provider with a batch interface can be added with
`video_index.models.register_provider`. API keys are read from the environment variables named in
`configs/models.yaml`.

## Stages

### `compose_candidates`

| Step | Rule |
|---|---|
| Population | items with `removed_by = none` and no pending screening step |
| Difficulty | worst-case attacker percentile: rank of each of the five screening margins among the population, largest percentile of the item |
| Exclusions | fewer than two options, invalid answer index, ordering questions ("select the correct order of the following options") |
| Order | (percentile, item id), hardest first |
| Cap | one item per video, the first `compose.candidates` items |
| Labels | 18 fine categories → four groups; candidates without a group are dropped |

Output `candidates.json`, `candidates_summary.csv`.

### `compose_verify`

The verifier sees frames at 1 fps (short side 224) with their timestamps, the question, the options and the marked
answer.

| Verdict | Meaning |
|---|---|
| `supported` | the frames show the marked answer is correct and the alternatives wrong |
| `contradicted` | the frames show a different option is correct |
| `not_verifiable` | the frames neither confirm nor rule out the marked answer |
| `definition_dependent` | the correct option hinges on a benchmark-specific definition or a subjective judgement |

First pass: at most 32 frames. Second pass: candidates with `not_verifiable` or `contradicted`, at most 512 frames;
the second verdict replaces the first. A request that does not fit the model is repeated with half the frames; the
number of frames sent is stored.

| Keep | Drop |
|---|---|
| `supported` | `contradicted` |
| reversed clips (the verdict is not used) | `definition_dependent` |
| `not_verifiable` on videos of at least 60 s | `not_verifiable` on videos shorter than 60 s |
| | no verdict |

Reversed clips: item ids in the JSON file named by `compose.reversed_clip_items`, or candidates with
`reversed_clip: true`. Outputs: `verify_f32.jsonl`, `verify_f512.jsonl`, `verified.json`, `verify_summary.json`
(with the number of refusals).

### `compose_select`

Per group (perception, temporal, spatial_physical, reasoning_knowledge) the verified candidates in the order
(percentile, item id) until budget / 4, one item per video over the whole list. No model's accuracy constrains the
selection. Outputs: `video_index.json`, `video_index_summary.csv`, `video_index_report.json`.

### `compose_export`

```
release/
  README.md                     dataset card
  items/meta_benchmark.jsonl    one line per item
  items/summary.json
  videos/<video_id>.mp4         links to the normalized videos (copies with compose.copy_videos: true)
  code/prompt.md                prompts, frame sampling and scoring of the two protocols
```

Fields of an item: `item_id, benchmark, capability_group, capability_group_label, fine_category, question, options,
answer, answer_idx, chance, video, video_id, duration_s, attack_pct, verification {verdict, keep_reason},
source_version, license`. `video_index.evaluate` and the two toolkit integrations read this layout.

Upload to the Hugging Face hub runs only with `compose.upload: true` and `compose.hf_repo`; the repository is
created private unless `compose.public: true`. The token is read from the environment by `huggingface_hub`.

## Parameters

| Key (`compose.*`) | Default | Study |
|---|---|---|
| `candidates` | 3000 | 3,000 |
| `verify_frames` | 32 | 32 |
| `recheck_frames` | 512 | 512 |
| `short_video_s` | 60 | 60 |
| `budget` | 840 | 840 (210 per group) |
| `reference_cap` | unset | unset in the released list; an earlier list used 0.45 |
| `fill_short_groups` | false | false |
| `tag` | v1 | — |

## Counts of the study

| Quantity | Value |
|---|---|
| Candidates with a capability group | 2,935 |
| Verified candidates | 1,675 (supported 1,103; not verifiable on videos ≥ 60 s 446; reversed clips 126) |
| Supply per group | perception 565, temporal 423, spatial_physical 367, reasoning_knowledge 320 |
| Released items | 840 from 76 benchmarks |
| Worst-case attacker percentile of the released items | median 0.140, maximum 0.231 |

`tests/test_compose_rules.py` reproduces the verified candidates (1,675) and the released list (840 items, identical
records) from the files of the study when `VIDEO_INDEX_RESEARCH_CODE` points to them.

## Items that did not come through the pool

`video_index.compose.items.from_samples(path, benchmark)` converts a sample file into the record format of the
candidates, so that such items can be labelled and verified with the same functions.
