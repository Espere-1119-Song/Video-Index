# Adding a benchmark (`vi-onboard`)

Pipeline 2 brings one new benchmark into the audit: it reads the annotations, converts them to items, checks the
format and the eligibility rules, draws the audited sample, fetches the videos of that sample, runs the audit stages
for this benchmark and writes its row into the registry and the tables.

```
annotations ──adapter──> items/<name>.jsonl ──format check──> samples/<name>.jsonl ──fetch──> raw_videos/<name>/
                                                                     └──> audit stages (video_index.audit) ──> tables/, reports
```

## 1. Setup

```bash
pip install -e .
cp configs/onboard.example.yaml configs/onboard.yaml
cp configs/models.example.yaml  configs/models.yaml      # model roles, see section 6
export HF_TOKEN=...                                      # only for gated or private repositories
```

API keys are read from the environment variables named in `configs/models.yaml`; no key is stored in a file.
All outputs go below `work_dir` (layout in `docs/DESIGN.md`).

## 2. Eligibility rules

`check` and `add` apply the inclusion rules of the paper and print one line per rule:

| Rule | Passes when |
|---|---|
| `gold_available` | fewer than half of the items lack a gold answer (a hidden test set fails) |
| `multiple_choice` | at least one item has an option list and a gold that resolves to one option |
| `min_items` | at least 30 multiple-choice items |
| `english` | at least 95% of the multiple-choice items are English text, and the spec declares `language: en` |
| `no_audio` | the spec does not set `requires_audio: true`, and at most 20% of the questions mention sound or speech |

Yes/no items and open-ended items are counted and left out of the audit; they are written to
`tables/format/<name>_excluded.jsonl` with the reason (`yes_no_options`, `multi_select`, `answer_not_in_options`,
`letter_out_of_range`, `no_options`, ...). `add` stops when a rule fails; `--force` continues.

The `no_audio` share is a keyword count. Read a sample of the flagged questions before deciding, and set
`requires_audio: true` in the spec when the questions cannot be answered without the audio track.

## 3. Worked example: CG-Bench

The annotation file `cgbench.json` is a list of records:

```json
{"qid": 26, "video_uid": "BV1s94y1G7RD", "question": "...", "choices": ["...", "...", "...", "...", "...", "..."],
 "right_answer": "D", "sub_category": "...", "domain": "...", "duration": 1320}
```

The videos are `<video_uid>.mp4` inside `video_chunk_01.zip` ... `video_chunk_45.zip`. The spec
(`configs/adapters/cg_bench.yaml`):

```yaml
name: CG-Bench
source:
  annotations: {kind: hf, repo: CG-Bench/CG-Bench, include: ["cgbench.json"]}
  videos: {kind: hf, repo: CG-Bench/CG-Bench, include: ["video_chunk_*.zip"]}
files: ["cgbench.json"]
fields:
  question: question
  options: choices
  answer: right_answer
  subtask: sub_category
  video: "{video_uid}.mp4"
answer_type: letter
```

Step 1, format check (downloads the annotation file only):

```bash
vi-onboard check --name CG-Bench --resolve-videos --show 2
```

```
Format check: CG-Bench
  items             12,129  (multiple choice 12,129, open 0, yes/no 0)
  option counts     5 options: 118, 6 options: 2,968, 7 options: 6,344, 8 options: 2,699
  gold positions    A 1,507, B 1,786, C 1,472, D 1,800, E 1,840, F 1,946, G 1,399, H 379
  position entropy  0.9973 (normalized, items with 7 options); modal position share 0.1604
  chance            0.1453
  videos            1,219; subtasks 12
  audited sample    300 items of 12,129, stratified by subtask
  eligibility
    [pass] gold_available: 0 of 12129 items carry no gold answer
    [pass] multiple_choice: 12129 multiple-choice items, 0 open, 0 yes/no (only multiple-choice items are audited)
    [pass] min_items: 12129 multiple-choice items; the audit needs at least 30
    [pass] english: 99.8% of the multiple-choice items are English text (declared language: en; required: 95%)
    [pass] no_audio: 7.4% of the questions mention sound or speech (flagged above 20%; ...)
  => eligible
```

With `--resolve-videos` every video reference is looked up in the video source (the central directory of each zip
is read through range requests; no video is downloaded) and the resolution rate is printed. A rate below 0.5 means
that the `video` field of the spec does not match the file names of the source.

Step 2, a smoke test on ten items, then the full run:

```bash
vi-onboard add --name CG-Bench --sample-n 300 --stages sample,fetch,videos,chance,reference --limit 10
vi-onboard add --name CG-Bench --sample-n 300
vi-onboard list
```

`add` without `--stages` runs, in order: `sample`, `fetch`, `videos`, `chance`, `screen`, `options_only`, `blind`,
`options_only_perm`, `pool_attack`, `single_frame`, `reference`, `reference_128`, `shuffle`, `window`, `captions`,
`levels`, `report`. `fetch` belongs to this package; every other stage is the stage of the same name in `vi-audit`,
run for this benchmark only. Every stage writes one row per item and skips finished rows on a rerun, so an
interrupted `add` is continued by running the same command again. The sample is drawn once and kept;
`--resample` draws it again.

Files written:

| Path below `work_dir` | Content |
|---|---|
| `annotations/<name>/` | the annotation files |
| `items/<name>.jsonl` | every multiple-choice item |
| `samples/<name>.jsonl` | the audited sample (seed 42, proportional to subtask size, at least 10 per subtask) |
| `raw_videos/<name>/` | the videos of the sample; each sample row gets `video_path` |
| `tables/format_check.csv`, `tables/format/<name>_excluded.jsonl` | format statistics; items left out |
| `tables/registry.csv` | one row per benchmark: source, counts, chance, status, breaking level |
| `logs/fetch_failed_<name>.jsonl` | videos that could not be fetched |

## 4. Writing an adapter spec

Without a spec the adapter detects the columns by name (`video`, `video_id`, `question`, `answer`, `options`,
`candidates`, `task_type`, ...). `check` prints the detected columns; write a spec when one of them is wrong.

| Key | Meaning |
|---|---|
| `files` | annotation files to read (glob patterns or path fragments); default: every json, jsonl, csv, tsv, parquet file |
| `records` | path to the record list inside a JSON file, `*` for every key: `data.*.*`; the last key is available as `__key` |
| `fields.question`, `fields.answer`, `fields.subtask`, `fields.video` | a column name (`metadata.question_type` looks into nested objects) or a template |
| `fields.options` | the column that holds the options: a list, a dict, or a string that contains a list |
| `fields.option_fields` | one column per option: `[a0, a1, a2, a3, a4]` |
| `answer_type` | `auto` (letter, option text or index, decided per item), `letter`, `index0`, `index1`, `text` |
| `filters` | row filters, applied before numbering: `{field: f, equals: x}`, `in`, `not_in`, `truthy`, `matches` (regular expression) |
| `keep: mcq` | keep only records with an option list and a single-letter gold (mixed-format benchmarks) |
| `dedupe` | drop records that repeat a (video, question) pair |
| `pinned` | the video field names archive and member: `videos_01.zip::clips/a.mp4` |
| `source.annotations`, `source.videos` | where the files come from (section 5) |
| `requires_audio`, `language` | declarations used by the eligibility rules |
| `meta` | year, citation key, licence, capability group (copied into the registry) |
| `hook` | a Python function for an irregular source (below) |

Templates: `"{video_uid}.mp4"`, `"{rearranged_id|int}.mp4"`, `"{metadata.id}|{metadata.start|sec}-{metadata.end|sec}"`.
Converters: `int`, `sec`, `str`, `basename`, `stem`, `first`, `strip:<prefix>`, `dir0`, `lower`, `letter0`, `letter1`.
`__file` is the path of the annotation file the row comes from.

Format rules applied to every record:

- Options given as a dict become the option texts in key order (`A`, `B`, ... or `option_0`, `option_1`, ...);
  empty values are dropped.
- Letter prefixes (`A. `, `(B) `) are removed from the option texts only when they run `A, B, C, ...` in order.
- Options written inside the question text are recovered when no option column exists and the gold resolves
  to one of them; the question keeps its stem.
- The gold may be a letter, an option text or an index. A gold that names several options, or none, makes the item
  open-ended.
- A `Yes`/`No` or `True`/`False` pair is a yes/no item.

The item id is `<name>_<index>`, where the index counts the records after the filters; rerunning the adapter on the
same files gives the same ids.

The shipped specs cover the common shapes:

| Spec | Shape |
|---|---|
| `cg_bench.yaml` | option list, letter gold, videos in zip chunks |
| `egoplan_bench.yaml` | one column per option, row filter, integer video id, parquet |
| `plm_videobench_fgqa.yaml` | option dict, 0-based index gold, nested fields, videos that are not redistributed |
| `next_qa.yaml` | one column per option, index gold, one of several annotation directories |
| `longvideobench.yaml` | option list, index gold, one file of several |
| `lemonade.yaml` | options as a string, column names with spaces, video name from a template |

### Irregular sources: a Python hook

When the annotations cannot be described by a spec (several files to join, options in a separate file, timestamps
to compute), write a function that returns the raw records and name it in the spec:

```python
# my_adapters/mybench.py
import json, os

def load(annotation_dir):
    questions = json.load(open(os.path.join(annotation_dir, "questions.json")))
    clips = json.load(open(os.path.join(annotation_dir, "clips.json")))
    for q in questions:
        yield dict(v=clips[q["clip"]]["file"],          # file name, path, or (archive, member)
                   q=q["text"],
                   a=q["gold"],                         # letter, option text or index
                   opts=q["candidates"],                # list or dict; None when there are no options
                   sub=q.get("category"))
```

```yaml
# configs/adapters/mybench.yaml
name: MyBench
hook: my_adapters/mybench.py:load        # a file path relative to the spec, or package.module:function
source:
  annotations: {kind: local, root: /data/mybench/annotations}
  videos: {kind: local, root: /data/mybench/videos}
```

The format rules, the numbering and the eligibility checks are the same as for a declarative spec. A package can
also register the function with `@register_adapter("MyBench")` (`video_index.onboard.adapters.generic`).

## 5. Sources

| Source | Command line | Spec |
|---|---|---|
| Hugging Face dataset repository | `--repo owner/name` | `{kind: hf, repo: owner/name, revision: <sha>, include: [...]}` |
| local directory | `--repo /path/to/dir` | `{kind: local, root: /path/to/dir}` |
| URL list | `--repo urls::files.tsv` | `{kind: urls, list: files.tsv}` (lines of `name<TAB>url`, or a JSON object) |

`--video-repo` names the video source when it differs from the annotation source. The command line overrides the
spec.

Videos inside archives are fetched member by member: a zip and a plain tar are read through range requests, a split
tar (`x.tar.part*`) is joined during reading, and a compressed tar is streamed once for all videos of the sample
(archives above 10 GB are skipped and listed). Only the videos of the sample are fetched.

### Videos that are not redistributed

Benchmarks that reference YouTube or another site ship no videos, and this package ships no downloader for them.
Two options:

1. Download the videos with your own tool and point `--video-repo` at the directory.
2. Register a fetcher for a custom source kind; the video reference of the item arrives unchanged in `ref["member"]`:

```python
from video_index.onboard.sources import register_fetcher

@register_fetcher("youtube")
def fetch(source, ref, dest):
    video_id, _, window = ref["member"].partition("|")     # e.g. "abc123|12-47.5" in PLM-VideoBench FGQA
    ...                                                     # write the segment to dest
    return dest
```

Import the module that holds the fetcher before calling `video_index.onboard.cli.main()`.

## 6. Model roles

`vi-onboard` has no model of its own. The stages it runs use the roles of `configs/models.yaml`:

| Stage | Role |
|---|---|
| `options_only`, `blind` | `text_attacker` |
| `options_only_perm` | `attacker` |
| `screen` | `attacker_small` |
| `single_frame`, `shuffle`, `window` | `reference`, `attacker` |
| `reference` | `reference` |
| `reference_128` | `reference_long` |
| `captions` | `captioner`, `caption_reader` |
| `watch --label` | `labeler` (optional) |

`sample`, `fetch`, `videos`, `chance`, `pool_attack`, `levels`, `report`, `check`, `list` and `watch` without
`--label` call no model. Any role can be an API model or an open-weight model behind an OpenAI-compatible server
(`provider: openai_compatible`, `base_url`, `api_key_env`). To compare a new benchmark with the audited ones, keep
the roles of the audit unchanged.

## 7. Watcher (optional)

```bash
vi-onboard watch --days 30                 # Hugging Face hub and arXiv, keywords from configs/onboard.yaml
vi-onboard watch --days 30 --label         # adds a one-line suggestion from the labeler role
```

New candidates are merged into `tables/candidates.csv` with `status = new`. The watcher onboards nothing: a person
reads the list, sets `status` to `approved` or `rejected` (with a `note`), and runs `vi-onboard add` for each approved
candidate. `video_index.onboard.watch.approved(ctx)` returns the approved rows for scripted use.

## 8. Messages

| Message | Action |
|---|---|
| `field detection failed (video=..., question=...)` | name the fields in a spec; the message lists the columns found |
| `no parseable annotation file` | check `files` and `source.annotations.include` |
| `every row was dropped` | a filter or a template names a field that the rows do not have |
| `only N of M video references resolve` | fewer than half resolve: the `video` field does not match the file names of the video source |
| `no fetcher for source kind` | register a fetcher (section 5) or use a local directory |
| `the audit stages are not installed` | `video_index.audit` is missing; items and sample are written, the registry status is `items_ready` |
