# Video-Index for Harbor

Converts the 840 Video-Index questions into [Harbor](https://github.com/laude-institute/harbor) tasks, so that agents
(Claude Code, Codex, OpenHands, Terminus, ...) can be evaluated on them. Ready-made task sets are also in the dataset
repository under `harbor/` ([Video-Index/Video-Index](https://huggingface.co/datasets/Video-Index/Video-Index)).

## Task sets

| Protocol | Tasks | Agent receives | Item score |
|---|---|---|---|
| `video` | 840, one per question | the video at `/app/video.mp4` with ffmpeg, OpenCV, Pillow and NumPy installed; the agent chooses its own frames | reward of the task |
| `blind` | 3,360, four per question | question and options only, under four option permutations fixed by `random.Random("42|<item_id>")` | mean reward of the four tasks |

Every task follows the Harbor layout:

```
<task>/
├── task.toml              name video-index/<item>, capability group, source benchmark, duration, chance
├── instruction.md         question, lettered options, where to write the answer
├── environment/Dockerfile python:3.12-slim + ffmpeg; the video is added to /app/video.mp4 at build time
├── solution/solve.sh      oracle: writes the marked letter
└── tests/test.sh          reward 1 when /app/answer.txt starts with the marked letter ("B", "(B)", "b.", "B. text")
```

The agent and the verifier run without network (`network_mode = "no-network"`). The video is downloaded from the dataset
repository at a pinned revision when the image is built; `--embed-videos` copies the mp4 into each task instead, for
offline registries.

## Generate

```bash
cd integrations/harbor
uv run python -m video_index_harbor.main --output-dir datasets/video-index            # 840 video tasks
uv run python -m video_index_harbor.main --output-dir datasets/video-index-blind --protocol blind
# options: --source <hub id | local dir>  --revision <sha>  --embed-videos  --task-ids ID ...  --limit N  --overwrite
```

## Run

```bash
harbor run -c run_video-index.yaml                                        # oracle check, every reward should be 1
harbor run -p datasets/video-index -a claude-code -m anthropic/claude-opus-4-5 -n 4
```

Accuracy is the mean reward. Per capability group, group the rewards by `metadata.capability_group` of `task.toml`.
For the blind set, first average the four tasks of each question (`metadata.item_id`), then average over questions.

## Relation to the paper

The paper's agentic runs used the same setting: one session per question with the stored video, shell tools and image
viewing, the model choosing its own frames, the answer as one option letter, failures counted as wrong. The paper
fixed a 1,800-second limit per question, which is `agent.timeout_sec` here.
