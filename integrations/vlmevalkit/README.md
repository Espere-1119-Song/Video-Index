# Video-Index in VLMEvalKit

Dataset class `VideoIndex` for [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). The files mirror the
branch `add-video-index` (base: `open-compass/VLMEvalKit` main, commit `6f03707`).

| File | Content |
|---|---|
| `vlmeval/dataset/video_index.py` | dataset class: data preparation, frames, prompt, evaluation |
| `vlmeval/dataset/utils/video_index.py` | prompt, option permutations, frame rule, rule scorer |
| `tests/test_video_index.py` | unit tests |
| `registration.patch` | the edits of `vlmeval/dataset/__init__.py`, `vlmeval/dataset/video_dataset_config.py` and `tests/fixtures/predefined_video_shortcuts_manifest.json` |
| `add-video-index.patch` | the whole commit (`git am add-video-index.patch`) |
| `smoke_test_config.json` | `--config` file of the 12-item check (model, slice) |

## Install

```bash
git clone https://github.com/open-compass/VLMEvalKit && cd VLMEvalKit
git am /path/to/Video-Index/integrations/vlmevalkit/add-video-index.patch
pip install -e .
```

## Run

```bash
python run.py --data Video-Index_1fps Video-Index_Blind --model <model>
```

| Dataset name | Setting |
|---|---|
| `Video-Index_1fps` | paper protocol: 1 frame per second, at most 512 frames, short side 224 |
| `Video-Index_64frame`, `Video-Index_32frame`, `Video-Index_8frame` | the same frame rule with a cap of 64, 32 or 8 frames |
| `Video-Index_Blind` | question and options only, four option permutations per item |

The item file and the videos are downloaded from `GMLRVigil/Video-Index` into `$LMUData/Video-Index`. The score
file (`*_score.json`) holds `Overall`, `Perception`, `Temporal`, `Spatial`, `Reasoning` (percent), `Items`,
`Replies` and `Replies without an option`. Gain = `Overall` of a video dataset minus `Overall` of
`Video-Index_Blind`.

Constructor arguments for a `--config` file: `fps`, `max_frames`, `nframe`, `short_side`, `limit`, `offset`
(`limit` and `offset` count items in the order of the item file; only the videos of the slice are downloaded).

## Differences from `vi-eval`

| Topic | `vi-eval` | VLMEvalKit |
|---|---|---|
| Models that read video files (`VIDEO_LLM`) | not applicable | receive the file; the first sentence of the prompt is `You are given a video. Answer the question based on this video.` |
| Request rejected for its size | frame count halved, request repeated | handled by the model wrapper |
| Frames | sent from memory as JPEG (quality 85) | written as JPEG files (quality 85), then encoded by the model wrapper |
| Output token limit | `--max-tokens` | model configuration |
