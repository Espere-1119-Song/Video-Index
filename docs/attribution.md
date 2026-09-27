# Error attribution (`video_index.audit.attribution`)

The attribution explains why the reference model fails on the items it answers wrongly from 32 frames. It runs after
the audit conditions `reference`, `blind`, `single_frame` and `captions` (see `docs/DESIGN.md`).

```
results/reference ──traces──> attribution/traces ──preclass──> attribution/preclass ──judge──> attribution/judge
                                     └──checks──> attribution/checks            tables <── judge + checks
```

## Running

```python
import importlib
from video_index.audit.context import Context
from video_index.audit.attribution import ORDER, STAGES

ctx = Context.from_files("configs/audit.yaml", "configs/models.yaml")
for stage in ORDER:                                   # traces, preclass, judge, checks, tables
    importlib.import_module(STAGES[stage]).run(ctx, benchmarks=None, limit=None)
```

`STAGES` maps the stage names to modules; the command line of the audit registers them under the same names.
Every stage is resumable: finished rows are on disk and are skipped by the next run.

## Models

| Role | Use | Study |
|---|---|---|
| `reference` | the erring model; writes the reasoning traces and answers the shifted-grid check | the model whose 32-frame errors are attributed |
| `judge_a`, `judge_b` | two independent judges | two different models |

Roles fall back as listed in `video_index/config.py` (`judge_b` → `judge_a` → `reference`). With one model in both
judge roles the agreement levels measure repeatability only; the stage logs this. The judges never see the name of
the erring model. API keys are read from the environment variables named in `configs/models.yaml`.

Requests are sent one by one from a thread pool (`workers`). A provider with a batch interface can be added with
`video_index.models.register_provider`; the stages only call `model.generate`.

## Stages

### `attribution_traces`

Every item that the reference model answers wrongly (rule scorer, `video_index.scoring.score`) is asked again with
the same 32 uniform frames and a prompt that requires written reasoning and a last line `Final answer: ...`.

| Outcome | Treatment |
|---|---|
| answer correct | recovered; not attributed |
| answer wrong | the written reasoning is the trace the judges read |
| answer not scorable by rule | neither recovered nor attributed |
| refusal | counted (`n_refused`), neither recovered nor attributed |

The prompt is the wording of 2026-09-22. The earlier instruction asked for the reasoning "in your visible reply" and
ended with "Do not keep the reasoning hidden."; with it one API model returned refusals instead of answers, and the
judges then labelled those rows `over_refusal`. Refusals are counted per benchmark in
`tables/attribution_recovered.csv`.

Output `attribution/traces/<benchmark>.jsonl`: `qid, model, orig_pred, pred_cot, still_wrong, refused, trace,
frame_check, n_frames, usage`. `frame_check` = SHA-1 of the frame index list and the video hash.
Failed requests: `attribution/traces_failures.jsonl`.

### `attribution_preclass`

Rules on the text-only run of the erring model (`results/blind`), no model call.

| Rule | Condition | Effect |
|---|---|---|
| 1 | text-only answer wrong and equal to the 32-frame wrong answer | `language_prior_dominated`, not sent to the judges |
| 2 | text-only answer correct, 32-frame answer wrong | `vision_introduced_error = true`, still judged |

The rules need the text-only run of the erring model itself. When only the `text_attacker` role (another model) ran
the text-only condition, its result is shown to the judges as context and the rules are not applied
(`n_blind_missing`). Output `attribution/preclass/<benchmark>.jsonl`, `tables/attribution_preclass.csv`.

### `attribution_judge`

Each judge receives the same 32 frames (the frame check of the trace row is recomputed and must match), the
question, the options, the correct answer, the wrong answer, the trace (first 8,000 characters), the results of the
same question under the other conditions, and the vision-introduced flag. It returns one of 30 categories, a reason,
and one free sentence that is not bound to the category list.

| Agreement | Meaning |
|---|---|
| `fine` | both judges name the same category |
| `group` | different categories of the same group |
| `disputed` | categories of different groups |
| `rule` | pre-classified by rule 1, no judge call |

Dense recheck: when both judges choose `evidence_not_in_input_frames`, both see min(n_frames, 128) uniform frames
and choose `sampling_gap`, `evidence_not_in_video`, `audio_needed` or `annotation_suspect`. Agreement replaces the
category. Videos with at most 32 frames skip the recheck.

A reply that cannot be parsed, a refusal or a failed request leaves the item unwritten; the next run repeats it.
Output `attribution/judge/<benchmark>.jsonl`: `qid, model, category, group, agreed_level, dense_recheck,
by {judge_a, judge_b: {model, category, reason, free_desc, usage}}, vision_introduced_error, frame_check`.

### `attribution_checks`

| Check | Rule | Output |
|---|---|---|
| Unstable failures | seeded sample of the still-wrong items (10 %, at least 1 per benchmark); same prompt, frame grid moved by half an inter-frame step (window [0.016, 1.016] × (n − 1)); share that turns correct | `attribution/checks/<benchmark>.jsonl`, `tables/attribution_unstable.csv` |
| Judge self-attribution | chi-square test between the categories `judge_a` assigns to failures of its own model and of other models | `tables/attribution_self_bias.json` |

Items that turn correct on the moved grid are excluded from the share tables.

### `attribution_tables`

Every attributed instance gets one status; precedence: unstable > disputed > (unverified | other | ok).

| Status | Rule |
|---|---|
| `unstable` | turned correct on the moved grid |
| `disputed` | `agreed_level = disputed`, or agreement on `data_protocol` with categories of different groups |
| `unverified` | `evidence_not_in_input_frames` without a conclusive dense recheck |
| `other` | category `other` |
| `ok` | annotation, coverage or capability (perception, temporal, spatial_physical, reasoning_knowledge) |

Denominator of every share = number of `ok` instances. Multiple-choice items only
(`video_index.audit.chance.is_mcq_item`).

| File (`tables/`) | Content |
|---|---|
| `attribution_shares.csv` | counts and shares per benchmark |
| `attribution_shares_pooled.csv` | all benchmarks; instance-weighted and item-weighted |
| `attribution_fine.csv`, `attribution_fine_pooled.csv` | fine categories inside the capability group |
| `attribution_row_labels.csv` | one row per instance with the rule that applied |
| `attribution_matrix.csv`, `_counts`, `_fine` | capability groups (fine categories) × benchmarks |
| `attribution_protocol_share.csv`, `attribution_disputed_rates.csv`, `attribution_jaccard.csv` | data-protocol share, agreement rates, similarity of benchmarks |

### `attribution_discover` (optional)

Clusters the free sentences of the rows attributed to `other` or disputed, in windows of 20 benchmarks, with the
`judge_a` model. A cluster above 0.5 % of all attributed failures is written to
`attribution/discover/proposals.json`. The taxonomy is not changed by this stage.

## Parameters

| Key (run configuration) | Default | Study |
|---|---|---|
| `reference_frames` | 32 | 32 |
| `frame_long_side` | 768 | 768 |
| `seed` | 42 | 42 |
| `workers` | 8 | 4 to 6 |
| `attribution.check_frac` | 0.10 | 0.10 |
| `attribution.check_n` | unset | unset (fixed sample size, used for validation runs) |
| `attribution.discover_window` | 20 | 20 |
| `attribution.trace_tokens` | 8192 | 8,192 |
| `attribution.judge_tokens` | 1536 | 1,536; 4,096 for a judge that spends output tokens on hidden reasoning |

Fixed in the code: dense recheck at most 128 frames; an empty reply is repeated once with twice the token budget;
`LOW_SAMPLE_N = 100` capability failures for the similarity table.

## Taxonomy

`video_index/audit/attribution/taxonomy.py` lists the 30 categories with one-line definitions and their groups:
perception (9), temporal (5), spatial_physical (3), reasoning_knowledge (3 judge-facing and
`language_prior_dominated`), data_protocol (10 judge-facing, `sampling_gap`, `evidence_not_in_video`).

## Requests per benchmark

With E reference errors, S still-wrong items after the rerun, R items pre-classified by rule 1 and D dense rechecks:
E (traces) + 2 (S − R) (judges) + 2 D (dense) + max(1, 0.1 S) (checks).
