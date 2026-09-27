# Pipeline 1a: the attack pyramid (`vi-audit`)

The audit answers one question per benchmark: which is the weakest attacker whose accuracy comes within `delta` of the
reference model's accuracy? Attackers are ordered in five levels by the input they receive: **option** (the options
only), **text** (question and options), **pool** (a classifier trained on other items of the same benchmark),
**frame** (one frame, or captions of single frames), **order** (shuffled frames, or one tenth of the video). The level
at which a benchmark breaks is the lowest level whose largest attacker margin over chance reaches
`s* - c - delta`, where `s*` is the reference accuracy at 32 frames and `c` is chance.

Models are addressed by role (`configs/models.yaml`), never by name: `reference`, `reference_long`, `attacker`,
`text_attacker`, `captioner`, `caption_reader`. Any model can take any role, including an open-weight model behind an
OpenAI-compatible server; every provider reads its API key from the environment variable named in `api_key_env`.

## Quick start

    pip install -e ".[embeddings]"           # ffmpeg and ffprobe must be on PATH
    cp configs/models.example.yaml configs/models.yaml
    cp configs/audit.example.yaml  configs/audit.yaml
    export ANTHROPIC_API_KEY=...             # and the keys of the other providers in configs/models.yaml

    # items of each benchmark in <work_dir>/items/<benchmark>.jsonl (schema: video_index/data/schema.py);
    # `vi-onboard` writes this file from a benchmark's annotation files
    vi-audit run --config configs/audit.yaml --models configs/models.yaml            # every stage
    vi-audit run --config configs/audit.yaml --models configs/models.yaml \
                 --stages reference,shuffle,window,levels --bench "MyBenchmark|Other" --limit 20 --workers 4
    vi-audit status --config configs/audit.yaml

`--stages` takes any subset; stages run in pipeline order. `--limit N` restricts every stage to the first N items of
each sample. Every stage can be interrupted and started again: rows on disk are not requested again, rows that ended
with an error are. From Python:

    from video_index.audit.context import Context
    from video_index.audit.cli import run_stages
    ctx = Context.from_files("configs/audit.yaml", "configs/models.yaml")
    run_stages(ctx, ["blind", "reference", "levels"], benchmarks=["MyBenchmark"], limit=None)

## Stages

| Stage | Role | Reads | Writes |
|---|---|---|---|
| `sample` | - | `items/<b>.jsonl` | `samples/<b>.jsonl` |
| `videos` | - | sample, source videos | `videos/<id>/video.mp4`, `videos/<id>/meta.json`; `video_id`, `duration_s` in the sample |
| `chance` | - | sample | `tables/chance.csv` |
| `screen` | - | sample | `tables/screen_items.csv`, `tables/screen_counts.csv` |
| `refill` (on request) | - | items, sample, `screen_items.csv` | sample, `tables/refill.csv`, `logs/sample_before_refill_<b>.jsonl` |
| `options_only` | text_attacker | sample | `results/options_only/` |
| `blind` | text_attacker | sample | `results/blind/` |
| `options_only_perm` | attacker | sample, videos | `results/options_only_perm/`, `tables/permutation_runs.csv`, `tables/permutation_by_benchmark.csv` |
| `pool_attack` | (learned) | sample | `tables/pool_attack.csv`, `embeddings/text/` |
| `single_frame` | reference, attacker | sample, videos | `results/single_frame/` |
| `reference` | reference, attacker | sample, videos | `results/reference/` |
| `reference_128` | reference_long | sample, videos, `results/reference/` | `results/reference_128/` |
| `shuffle` | reference, attacker | sample, videos | `results/shuffle/` |
| `window` | reference, attacker | sample, videos | `results/window/` |
| `captions` | captioner, caption_reader | sample, videos | `captions/`, `results/captions/`, `results/captions_video/` |
| `levels` | - | every result file, `pool_attack.csv`, `screen_items.csv` | `tables/levels.csv`, `tables/level_attackers.csv`, `tables/position_shares.csv`, `tables/audited_benchmarks.txt` |
| `report` | - | tables | `tables/report/<b>.md`, `tables/report_cards.csv` |

Result files are `results/<condition>/<benchmark>__<model>.jsonl`, one row per item and role: `qid`, `role`, `pred`,
`answer`, `correct` (`true`, `false`, or `null` when the rule scorer cannot score the prediction), `frames` (frames
sent), `fallback` (times the frame count was halved because the request did not fit), `error`. Two roles that resolve
to the same model share a file; the `role` field separates their rows.

### sample

All items when the benchmark has at most `sample_n` (300). Otherwise each subtask receives
`round(300 * size / total)` items, at least `sample_min_per_stratum` (10) and at most its size; while the total exceeds
300 the largest allocation is reduced by one. One generator seeded with `seed` shuffles the qid-sorted items of each
subtask, subtasks in sorted order, and the allocated prefix is taken. An existing sample is kept (`resample: true`
draws again).

`refill` continues the same draw after the screen removed items: a removed item is replaced by the next item of its
own subtask in the shuffled order; a remaining shortfall is distributed over the other subtasks in proportion to the
original allocation (largest remainder); when every subtask is exhausted the sample stays below 300. Items of the
original draw, items that are not multiple choice, option lists whose length is not among the declared option counts
and near-duplicates of a kept item are not eligible. `refill` changes the sample, so it runs only when named in
`--stages`; the videos of the added items are normalized in the same call, and the model stages then answer the added
items on their next run.

### videos

ffmpeg writes one normalized file per source video: frame rate `min(2, 1024 / duration)`, short side 720 (480 for
videos longer than 1,800 s unless the benchmark is `text_heavy`), never enlarged, no audio, H.265 (H.264 when the
encoder is missing). `video_id` is the SHA-256 of the source file. `meta.json` holds the source duration, frame rate
and resolution, the normalized frame rate and resolution, the frame count, and the timestamp of every frame.

The source of a video is the first of: `video_path` of the sample row (written by the `fetch` stage of `vi-onboard`);
`video_ref` as a path, absolute or relative to `video_root`; `video_ref` as a mapping `{repo, member, archive}`, which
downloads the file, or the archive that holds it, from the Hugging Face hub (`HF_TOKEN` is used when set).

### chance, screen

`chance`: item chance is `1 / k` for `k` options (the option list, else lettered options inside the question text,
else the declared option count of a multiple-choice benchmark, else `k` inferred from letter-form gold answers),
0.5 for yes/no answers, 0 for open answers. Every statistic of the audit uses the multiple-choice items only; the
benchmark chance `c` is their mean item chance.

`screen`: two items are near-duplicates when they share the video, the normalized option set and the answer, and the
cosine between their question embeddings is at least 0.90; of each group the smallest qid is kept. For a benchmark
with declared option counts, items with another number of options are recorded (their chance already follows the
actual count). No gold answer is changed.

### options_only, blind

`blind`: the text attacker receives the question and the options. `options_only`: the options alone, for items that
carry options. The prompts are in `video_index/audit/prompts.py`.

### options_only_perm

Every multiple-choice item is answered under the original option order and under `permutations` (8) seeded
permutations, by the attacker role. `permutation_input: frames` (default) is the protocol of the study: the 32-frame
attacker condition with the permuted options; permutation 0 is taken from the attacker rows of `results/reference/`
when they exist. `permutation_input: options_only` sends the options without question and frames.
`tables/permutation_by_benchmark.csv` reports, on the items present in every permutation, the width
`acc_max - acc_min`, the threshold `2.326 sqrt(c (1 - c) / n)` and the share of items whose prediction is the same
original option in every permutation.

### pool_attack

Text embeddings of "question + options" (`embedding_model`, sentence-transformers). `prequential` (default, the
protocol of the study): the items are visited in 50 random orders; each item is predicted by an online logistic
regression trained on the items before it; the reported margin is the accuracy over the last 20 % of the positions
minus chance. `five_fold`: logistic regression with five-fold held-out prediction within the benchmark.

### single_frame, reference, reference_128, shuffle, window

| Condition | reference role | attacker role |
|---|---|---|
| frame selection | index grid over the stored frames | time grid over the frame timestamps (nearest stored frame) |
| resolution | long side at most 768 | stored resolution (`audit_attacker_short_side` sets a short side) |
| `single_frame` | the middle frame | frame 16 of the 32-frame grid |
| `reference` | 32 uniform frames | 32 uniform frames |
| `shuffle` | the 32 frames in one random order (seed), identical for every item | the 32 frames in an order drawn per item (`"<seed>|<qid>"`) |
| `window` | 32 frames in one tenth of the video; start drawn per item (`"<seed>|<qid>"`), uniform in [0, 0.9] | 32 frames in the first tenth |

The prompt is the same for the four conditions and does not mention the frame order or the window. A video with fewer
stored frames than requested contributes every frame once. `reference_128` (role `reference_long`, 128 frames) runs
only for benchmarks whose 32-frame reference accuracy is within `delta` of chance (`rescreen_all: true` runs it for
all).

### captions

`frames`: the captioner describes each of the 32 frames of the attacker grid in one sentence, one frame per request;
the caption reader answers from the 32 captions in frame order. This attacker enters the frame level. `video`: one
description of the video from 32 frames, read by the caption reader; reported in `level_attackers.csv`, outside the
levels.

### levels

For every benchmark: `c`, `s*`, the threshold `2.326 sqrt(c (1 - c) / n)`, the margin of every attacker on the items
scored for both the reference and the attacker (at least `min_items` = 30), the running maximum per level, and the
breaking level at `delta` = 0.05, 0.03 and 0.08. A level without an attacker value is not measured. The order level
needs one shuffle run and one window run. The option level also contains the fixed-position attacker (the modal gold
position of the sample), which needs no model.

128-frame rescreen: when `s* - c <= delta` and the 128-frame accuracy exceeds `c` by more than `delta`, the 128-frame
accuracy replaces `s*`. When it does not, the benchmark is assigned the first measured level and
`reference_near_chance = 1`.

Audited set (`tables/audited_benchmarks.txt`): the benchmarks with a level, that is with at least 30 multiple-choice
items scored in the reference run.

The columns `sr_*` repeat the assignment with the attacker role at 32 frames as the reference (second reference).

### report

`tables/report/<benchmark>.md`: sample size, chance, reference accuracy, breaking level at the three tolerances, margin
per level, every attacker with accuracy, margin and item count, screen counts and the permutation audit.

## Changing the reference model

Edit the `reference` entry of `configs/models.yaml`; nothing else changes. For an open-weight model:

    models:
      reference:
        name: qwen3-vl-32b                    # label in file names and tables
        provider: openai_compatible
        model: Qwen/Qwen3-VL-32B-Instruct
        base_url: ${REFERENCE_BASE_URL:-http://127.0.0.1:8000/v1}
        api_key_env: VLLM_API_KEY             # the key the server was started with
        max_frames: 128

    vi-audit run --config configs/audit.yaml --models configs/models.yaml \
                 --stages single_frame,reference,reference_128,shuffle,window,levels,report

Result files carry the model label, so the rows of the previous reference stay on disk and the new model writes new
files; `levels` reads the files of the model that currently holds the role. The text attackers, the captions and the
pool attacker do not depend on the reference and are not run again. To compare two references, keep one as
`reference` and configure the other as `attacker`: the `sr_*` columns of `tables/levels.csv` hold the second
assignment.

## Differences from the runs of the study

- Options stored without letter prefixes are written with letters in the prompts (`letter_options: true`). The study
  sent the options field as stored; `letter_options: false` restores that.
- The study used one fixed model per role; here every role is configurable, and every model with a `blind` or
  `options_only` file in the workspace counts as an attacker of its level.
- The video-caption condition uses the 32-frame grid of the attacker role for the captioner.
