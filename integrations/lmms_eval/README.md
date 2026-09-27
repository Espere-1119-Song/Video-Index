# Video-Index in lmms-eval

Task directory `video_index` for [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). The files mirror the
branch `add-video-index` (base: `EvolvingLMMs-Lab/lmms-eval` main, commit `42296f8`).

| File | Content |
|---|---|
| `lmms_eval/tasks/video_index/` | task files: six YAML tasks, `_default_template_yaml`, `utils.py`, `README.md` |
| `test/eval/test_video_index.py` | unit tests |
| `add-video-index.patch` | the whole commit, including the entry in `docs/advanced/current_tasks.md` (`git am add-video-index.patch`) |

## Install

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval && cd lmms-eval
git am /path/to/Video-Index/integrations/lmms_eval/add-video-index.patch
pip install -e .
```

## Run

```bash
python -m lmms_eval --model <model> --tasks video_index,video_index_blind --batch_size 1
```

| Task | Setting |
|---|---|
| `video_index` | the video file; the model wrapper applies its own frame sampling |
| `video_index_1fps` | frames of the paper protocol (1 frame per second, at most 512 frames, short side 224), passed as images |
| `video_index_64frame`, `video_index_32frame`, `video_index_8frame` | the same frame rule with a cap of 64, 32 or 8 frames |
| `video_index_blind` | question and options only, four option permutations per item |

Metrics (percent): `video_index_acc`, `video_index_perception`, `video_index_temporal`, `video_index_spatial`,
`video_index_reasoning`. Videos are downloaded one by one from `GMLRVigil/Video-Index`; `VIDEO_INDEX_DIR` points
the task to a local copy.

## Differences from `vi-eval`

| Topic | `vi-eval` | lmms-eval |
|---|---|---|
| `video_index` task | not applicable | the wrapper samples the frames; the first sentence of the prompt is `You are given a video. Answer the question based on this video.` |
| Request rejected for its size | frame count halved, request repeated | handled by the model wrapper |
| Frame encoding | JPEG (quality 85) | set by the wrapper; OpenAI-style wrappers send PNG unless `LMMS_IMAGE_ENCODE_FORMAT=JPEG` |
| Output token limit | `--max-tokens` | `generation_kwargs.max_new_tokens: 16` |
