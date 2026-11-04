# Mage-VL on Video-Index

[Mage-VL](https://huggingface.co/microsoft/Mage-VL) (4B, Mage-ViT + Qwen3-4B), evaluated with `vi-eval`.

| Setting | Mean visual tokens | Video | Blind | Gain | Perception | Temporal | Spatial | Reasoning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Official protocol (1 FPS frames, ≤512, short side 224) | 14.9k | 14.3 | 2.9 | 11.4 | 12.9 | 21.9 | 8.6 | 13.8 |
| Codec-native video (`target_canvas=128`) | **7.5k** | **14.4** | 2.9 | **11.5** | 17.1 | 18.6 | 11.9 | 10.0 |

With half the visual tokens, the codec input matches the official protocol.

### Ranking

Mage-VL rows added to Table 1 of the [Video-Index blog](https://www.enxinsong.com/blog/video-index/), ranked by
Video (human volunteers unranked). Mage-VL is 9th overall, 3rd among open models and 1st at its size.

| # | Model | Input | Video | Blind | Gain |
| --- | --- | --- | --- | --- | --- |
| 1 | GPT-6-Astra | agent | 79.3 | 18.7 | 60.6 |
| 2 | Claude Fable 5.1 | agent | 71.5 | 20.7 | 50.8 |
| 3 | Claude Opus 5 | agent | 70.6 | 24.9 | 45.7 |
| 4 | Claude Opus 5 | fixed | 56.8 | 21.3 | 35.5 |
| – | Human volunteers | | 55.0 | 17.5 | 37.5 |
| 5 | Gemini 3.1 Pro | fixed | 54.7 | 23.9 | 30.8 |
| 6 | Gemini 3.5 Flash-Lite | fixed | 22.5 | 17.5 | 5.0 |
| 7 | Molmo2-8B | fixed | 19.0 | 4.8 | 14.2 |
| 8 | InternVL3.5-8B | fixed | 15.2 | 4.4 | 10.8 |
| **9** | **Mage-VL-4B, codec (7.5k tokens)** | native video | **14.4** | 2.9 | **11.5** |
| **9** | **Mage-VL-4B, official (14.9k tokens)** | fixed | **14.3** | 2.9 | **11.4** |
| 10 | LLaVA-OneVision-2-8B | fixed | 14.2 | 1.8 | 12.4 |
| 11 | InternVideo2.5-8B | fixed | 12.4 | 7.5 | 4.9 |
| 11 | VideoLLaMA3-7B | fixed | 12.4 | 5.2 | 7.1 |
| 11 | Video-XL-2 | fixed | 12.4 | 7.3 | 5.1 |
| 14 | Cambrian-S-7B | fixed | 11.3 | 5.3 | 6.0 |
| 15 | Qwen3-VL-8B | fixed | 9.5 | 1.4 | 8.1 |
| 16 | VideoChat3-4B | fixed | 9.4 | 3.8 | 5.7 |

* **Official protocol.** `vi-eval run` unchanged. `mage_vl.py` passes the frames before the text and labels
  them with the real times `vi-eval` provides (Mage-VL's processor otherwise labels pre-decoded frames 0, 1,
  2, … s, which is wrong for the 201 long videos thinned to 512 frames).
* **Codec-native video.** The benchmark's native-video setting (as in lmms-eval's `video_index` task): the
  model receives the video file, with the prompt `You are given a video. Answer the question based on this
  video.` `codec_run.py` feeds it to Mage-VL's HEVC codec processor, which packs up to 1,024 frames (all
  stored frames) into at most 128 canvases. 12 videos the codec cannot read fall back to the official
  frames.
* **Blind** is the same `vi-eval` blind run for both rows. Mean visual tokens count the `<|image_pad|>`
  tokens per item (14.9k = 186.8 frames on average).

## Reproduce

Requirements: `transformers>=5.7` for Mage-VL; for the codec input, `ffmpeg` and `cv-preinfer` from
[`codec-video-prep`](https://pypi.org/project/codec-video-prep/) (set `CV_PREINFER_BIN` if it is not on
`PATH`). The model revision is pinned to `d88b153285f1633a61b2f693c59c8576693af185`.

```bash
pip install -e . "transformers==5.7.0" torch accelerate decord
export PYTHONPATH=integrations/mage_vl

# official protocol (video + blind)
vi-eval run --local mage_vl:MageVL --workers 1 --max-tokens 16 --protocol both --out runs/mage-vl
vi-eval score --run runs/mage-vl

# codec-native video, scored against the same blind rows
python integrations/mage_vl/codec_run.py --target-canvas 128 --out runs/mage-vl
python integrations/mage_vl/codec_run.py --target-canvas 128 --out runs/mage-vl --score
```

Both runners take `--shard i/n` for one process per GPU; give each shard its own `--out` and concatenate the
`.jsonl` files before scoring. Our runs used 8×H800.
