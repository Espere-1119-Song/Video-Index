# Evaluating a model on Video-Index (`vi-eval`)

Pipeline 3 scores a model on Video-Index and prints its row of Table 1: accuracy with the video, accuracy without
the video (blind), the gain, and the accuracy per capability group. Three harnesses implement the same protocol:

| Harness | Location | Use it when |
|---|---|---|
| `vi-eval` | `video_index/evaluate` | the model is behind an endpoint, or a Python class in this process |
| VLMEvalKit | `integrations/vlmevalkit` | the model already has a VLMEvalKit wrapper |
| lmms-eval | `integrations/lmms_eval` | the model already has an lmms-eval wrapper |

## 1. Data

[`GMLRVigil/Video-Index`](https://huggingface.co/datasets/GMLRVigil/Video-Index) holds 840 multiple-choice items
from 76 video benchmarks, one item per video, 210 items per capability group (`perception`, `temporal`,
`spatial_physical`, `reasoning_knowledge`). Items have 2 to 10 options; the mean chance accuracy is 29.7%.

| File | Content |
|---|---|
| `items/meta_benchmark.jsonl` | one item per line: `item_id`, `benchmark`, `capability_group`, `fine_category`, `question`, `options`, `answer`, `answer_idx`, `chance`, `video`, `video_id`, `duration_s`, `license` |
| `videos/<video_id>.mp4` | the video of the item, stored at no more than 2 frames per second and 1,024 frames |

`--data` takes the hub id (default) or a local directory with the same layout. From the hub, the item file is
downloaded first and every video when its item is reached, so a partial run downloads only the videos it uses;
`--fetch-all` downloads the 840 videos before the run.

## 2. Protocol

| | Video | Blind |
|---|---|---|
| Visual input | one frame per second of the timeline, at most 512 frames (uniform thinning beyond), short side 224 pixels, frames before the text | none |
| Option order | the order of the item file | four permutations per item, fixed by `random.Random("42|<item_id>")` |
| Requests per item | 1 | 4 |
| Item score | 1 when the reply names the marked option, else 0 | mean over the four permutations |

Frame rule. The stored frame nearest to every full second of the timeline `[0, duration_s)` is sent; when the
video is stored below 1 frame per second, every stored frame is sent. `duration_s` is the duration of the source
video given in the item file. `--frame-cap N` lowers the cap, `--fps` changes the rate.

When the model rejects a request for its size (`ContextLimitError`, or an error message that names the context
length or the memory), the frame count is halved by uniform thinning, down to `--min-frames`, and the request is
repeated with the prompt of the new frame count. The number of frames sent is stored with the answer.

Prompt (video; the blind prompt starts with `You are given NO frames from the video. Answer the question from the
text alone.`):

```text
You are given {n} frame(s) sampled from a video. Answer the question based on these frames.

Question: {question}
A. {option}
B. {option}
Reply with ONLY the option letter (or the exact short answer if no options).
```

Scoring is rule based (`video_index/scoring.py`); no judge model is called. The option named by a reply is, in
this order: the leading option letter (`B`, `(B)`, `B. text`), a stated answer (`the answer is B`; the last
statement counts), the single option whose text the reply repeats. A reply that names no option is scored as wrong.

Aggregation (`video_index/evaluate/score.py`):

| Column | Definition |
|---|---|
| Video | mean item score of the video protocol |
| Blind | mean item score of the blind protocol over the items that also have a video answer |
| Gain | Video on those common items minus Blind |
| Perception, Temporal, Spatial, Reasoning | Video accuracy within one capability group |

A row without a score (failed request, missing video) is left out and counted in `unanswered`; a run that covers
fewer than 840 items is marked `partial` and printed as `n of 840`.

## 3. Setup

```bash
pip install -e .
cp configs/models.example.yaml configs/models.yaml
export ANTHROPIC_API_KEY=...        # the variable named in api_key_env of the model
```

API keys are read from the environment variables named in the model specification; no key is stored in a file.

## 4. Commands

```bash
# a role of configs/models.yaml
vi-eval run --model-config configs/models.yaml --role reference --protocol both --out runs/claude-opus-5

# an inline specification: an open-weight model behind an OpenAI-compatible server
vllm serve Qwen/Qwen3-VL-8B-Instruct --api-key $VLLM_API_KEY --port 8000
vi-eval run --role provider=openai_compatible,model=Qwen/Qwen3-VL-8B-Instruct,base_url=http://127.0.0.1:8000/v1,api_key_env=VLLM_API_KEY,max_frames=128,max_tokens=16 \
    --protocol both --out runs/qwen3-vl-8b

# a model that runs in this process
vi-eval run --local my_package.my_module:MyModel --local-kwargs '{"checkpoint": "..."}' --workers 1 --out runs/my-model

vi-eval score --run runs/qwen3-vl-8b
vi-eval leaderboard --runs runs/claude-opus-5 runs/qwen3-vl-8b --out leaderboard.csv
```

Options of `vi-eval run`:

| Option | Default | Meaning |
|---|---|---|
| `--data` | `GMLRVigil/Video-Index` | hub id or local directory |
| `--revision`, `--cache-dir` | | hub revision and cache directory |
| `--model-config`, `--role` | `configs/models.yaml` | a configured role, a JSON object, or `key=value,key=value` (`provider`, `model`, `base_url`, `api_key_env`, `max_frames`, `max_tokens`, `temperature`, `timeout`, `max_retries`, `image_long_side`, `jpeg_quality`) |
| `--local`, `--local-kwargs` | | `package.module:ClassName` of an in-process model and the JSON object passed to its constructor |
| `--name` | model name | label of the model in the result files |
| `--protocol` | `both` | `video`, `blind` or `both` |
| `--out` | | run directory |
| `--limit` | | seeded random subset of this size (the same items for every model) |
| `--ids` | | item ids, comma separated, or a file with one id per line |
| `--window` | | `start:stop`, consecutive items in the order of the item file (the order in which the toolkit integrations count `offset` and `limit`) |
| `--shard` | | `i/n`, every n-th item starting at i |
| `--workers` | 8 | parallel requests (endpoint models, and in-process models with `threaded = True`) |
| `--max-tokens` | model specification | output token limit (16 in the paper for models that reply directly) |
| `--frame-cap`, `--fps`, `--short-side` | 512, 1, 224 | frame rule |
| `--min-frames` | 8 | lower bound of the halving |
| `--max-minutes` | | stop after this time; a rerun continues |
| `--fetch-all` | | download every video before the run |

## 5. Result files

```
runs/<name>/<model>__video1fps.jsonl
runs/<name>/<model>__blind.jsonl
runs/<name>/results.csv
```

The `.jsonl` files are append-only, one row per (item, permutation):

```json
{"model": "qwen3-vl-8b", "protocol": "video1fps", "item_id": "TempCompass_4224", "benchmark": "TempCompass",
 "capability_group": "spatial_physical", "perm": [0, 1, 2, 3], "frames": 3, "short_side": 224, "correct": false,
 "pred": "A", "gold": "C", "ms": 912}
```

`fallback` counts the halvings of the item; `error` is set and `correct` is null when the request failed. A rerun
of the same command skips the rows that have a score and repeats the failed ones. `results.csv` has the columns
`model, n, video, blind, gain, perception, temporal, spatial_physical, reasoning_knowledge, <group>_n, n_blind,
n_total, partial, mean_frames, fallback_items, unanswered`.

## 6. Python interface for in-process models

```python
from video_index.models import ContextLimitError


class MyModel:
    name = "my-model"
    threaded = False                 # True when answer() may be called from several threads

    def __init__(self, checkpoint: str, max_frames: int = 128):
        self.max_frames = max_frames
        ...

    def answer(self, frames, timestamps, prompt) -> str:
        """frames: PIL RGB images in temporal order (empty in the blind protocol);
        timestamps: their times in seconds; prompt: the full text. Returns the reply text."""
        if len(frames) > self.max_frames:
            raise ContextLimitError(f"{len(frames)} frames")   # the runner halves the frames and calls again
        ...
        return reply
```

The same interface from Python:

```python
from video_index.evaluate import VideoIndex
from video_index.evaluate.run import run_protocol
from video_index.evaluate import score

data = VideoIndex("GMLRVigil/Video-Index")
model = MyModel(checkpoint="...")
for protocol in ("video", "blind"):
    run_protocol(model, data, protocol, "runs/my-model", workers=1)
print(score.render(score.table(score.load_rows("runs/my-model"))))
```

## 7. VLMEvalKit

Files: `integrations/vlmevalkit` (copy them over a VLMEvalKit checkout, or use the branch of the pull request).

```bash
python run.py --data Video-Index_1fps Video-Index_Blind --model <model>
```

| Dataset name | Setting |
|---|---|
| `Video-Index_1fps` | paper protocol: 1 frame per second, at most 512 frames, short side 224 |
| `Video-Index_64frame`, `Video-Index_32frame`, `Video-Index_8frame` | the same frame rule with a cap of 64, 32 or 8 frames |
| `Video-Index_Blind` | question and options only, four permutations per item |

Models with `VIDEO_LLM = True` receive the video file and the first sentence `You are given a video. Answer the
question based on this video.`; all other models receive the frames. The score file holds `Overall`, the four
groups, `Items`, `Replies` and `Replies without an option`. The constructor arguments `limit` and `offset`
(`--config` file) select a slice in the order of the item file; only the videos of the slice are downloaded.

## 8. lmms-eval

Files: `integrations/lmms_eval` (copy `video_index/` to `lmms_eval/tasks/`, or use the branch of the pull request).

```bash
python -m lmms_eval --model <model> --tasks video_index,video_index_blind --batch_size 1
```

| Task | Setting |
|---|---|
| `video_index` | the video file; the model wrapper applies its own frame sampling |
| `video_index_1fps` | frames of the paper protocol, passed as images |
| `video_index_64frame`, `video_index_32frame`, `video_index_8frame` | the same frame rule with a cap of 64, 32 or 8 frames |
| `video_index_blind` | question and options only, four permutations per item |

Metrics: `video_index_acc`, `video_index_perception`, `video_index_temporal`, `video_index_spatial`,
`video_index_reasoning`, in percent. `VIDEO_INDEX_DIR` points the task to a local copy of the dataset.

## 9. Agreement of the three harnesses

The three harnesses share the prompt, the option permutations, the frame rule and the scorer.
`tests/test_evaluate_protocol.py` checks that the scorer and the frame rule shipped with the integrations return
the results of `video_index.scoring` and `video_index.evaluate.protocol`.

Check with a model behind an API (2026-09-26; `gemini-2.5-flash-lite`, 12 items, frame cap 8, output limit 16
tokens; blind protocol on 2 items, 8 requests):

| Comparison | Video (12 rows) | Blind (8 rows) |
|---|---|---|
| Same correctness in the three harnesses | 12 | 8 |
| Same reply text in the three harnesses | 11 | 8 |

Conditions of the agreement:

| Topic | Setting |
|---|---|
| Frame decoding | stream order. With random access, decord returned other frames than OpenCV on 7 of the 12 videos; the integrations decode in stream order and return the frames of `vi-eval` (pixel difference 0) |
| Frame encoding | JPEG, quality 85. lmms-eval wrappers that build OpenAI-style messages send PNG by default; with PNG, 9 of 12 replies and 11 of 12 scores equal those of `vi-eval`; `LMMS_IMAGE_ENCODE_FORMAT=JPEG` selects JPEG |
| Output token limit | the same value in the three harnesses |

Reports and scripts: `integrations/validation/`.

## 10. Tests

```bash
pytest tests/test_evaluate_protocol.py tests/test_evaluate_run.py
```
