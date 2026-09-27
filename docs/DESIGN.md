# Design

Three pipelines share one library (`video_index/`).

| Pipeline | Package | Command | Purpose |
|---|---|---|---|
| 1. Audit | `video_index/audit`, `video_index/pool`, `video_index/compose` | `vi-audit` | Rebuild the whole study: sample benchmarks, run the attack pyramid, assign breaking levels, attribute errors, screen the pool, compose Video-Index |
| 2. Onboard | `video_index/onboard` | `vi-onboard` | Add one new benchmark: adapter, sample, videos, the audit stages for that benchmark, updated tables |
| 3. Evaluate | `video_index/evaluate`, `integrations/` | `vi-eval` | Score a model on Video-Index (standalone, VLMEvalKit, lmms-eval) |

## Models

Every model is reached through `video_index.models.ChatModel.generate(prompt, images)` and configured as a *role* in
`configs/models.yaml` (`video_index/config.py` lists the roles and their fallbacks). No stage names a concrete model:
a stage asks for `models["reference"]`, `models["judge_a"]`, ... Providers: `openai_compatible` (OpenAI and every
open-weight server: vLLM, SGLang, Ollama, LM Studio, TGI), `anthropic`, `gemini`; `register_provider` adds more.
API keys come from environment variables named by `api_key_env`.

## Workspace

All stages read and write below one directory (`work_dir` of the run configuration):

    annotations/<benchmark>/            raw annotation files as downloaded
    items/<benchmark>.jsonl             every item of the benchmark after its adapter (schema: video_index/data/schema.py)
    samples/<benchmark>.jsonl           the audited sample (n = 300, stratified by subtask; all items when fewer)
    videos/<video_id>/video.mp4         normalized video, <video_id> = SHA-256 of the source file
    videos/<video_id>/meta.json         duration, frame timestamps, source resolution
    captions/<benchmark>__<model>.jsonl
    results/<condition>/<benchmark>__<model>.jsonl     one row per item: qid, pred, answer, correct, frames, fallback, error
    attribution/<stage>/<benchmark>.jsonl
    tables/*.csv                        chance.csv, screen_counts.csv, levels.csv, attribution_shares.csv, ...
    pool/                               pool_items.parquet, funnel.csv, selection files
    logs/

Result files are append-only and keyed by `qid` (plus the permutation index where a condition uses permutations);
a rerun skips finished rows (`video_index.data.schema.JsonlWriter`). `correct` is `true`, `false` or `null`
(`null` = not scorable by rule, excluded from accuracy).

## Conditions

| Condition | Role | Input |
|---|---|---|
| `options_only` | text_attacker | options without the question |
| `options_only_perm` | attacker | options-only under eight option permutations plus the original order |
| `blind` | text_attacker | question and options |
| `pool` | (learned) | logistic regression on question embeddings, five-fold held-out prediction within the benchmark |
| `single_frame` | reference, attacker | the middle frame |
| `captions` | caption_reader | captions written by the captioner |
| `reference` | reference | 32 uniform frames, long side at most 768 |
| `reference_128` | reference_long | 128 uniform frames (long-video benchmarks whose `reference` is within delta of chance) |
| `shuffle` | reference, attacker | the 32 frames in a seeded random order |
| `window` | reference, attacker | 32 frames from one tenth of the video |

## Stage interface

Every stage is a module with `run(ctx, benchmarks=None, limit=None)`, where `ctx` is `video_index.audit.context.Context`
(`ctx.work_dir`, `ctx.models`, `ctx.cfg`, `ctx.path(kind, ...)`). `video_index.audit.cli.run_stages(ctx, stages, benchmarks)`
runs a list of stages in order; the onboarding pipeline calls it for one benchmark.

## Conventions

- Seeds: every sampler takes an explicit seed; the default is 42, per-item seeds are `f"{seed}|{qid}"`.
- Scoring: `video_index.scoring.score` (rule scorer: option letter, stated answer, option text, in that order).
- No absolute paths, no cluster-specific code, no credentials in the repository.
