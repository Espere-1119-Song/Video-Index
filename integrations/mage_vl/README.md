# Mage-VL on Video-Index

[Mage-VL](https://huggingface.co/microsoft/Mage-VL) (4B, Mage-ViT + Qwen3-4B), evaluated with `vi-eval`.

| Setting | Mean visual tokens | Video | Blind | Gain | Perception | Temporal | Spatial | Reasoning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Official protocol (1 FPS frames, ≤512, short side 224) | 14.9k | 14.3 | 2.9 | 11.4 | 12.9 | 21.9 | 8.6 | 13.8 |
| Codec-native video (`target_canvas=128`) | **7.5k** | **14.4** | 2.9 | **11.5** | 17.1 | 18.6 | 11.9 | 10.0 |

With half the visual tokens, the codec input matches the official protocol. Both rows place Mage-VL third
among the open models of Table 1, after Molmo2-8B (19.0) and InternVL3.5-8B (15.2), and first among
models of its size (VideoChat3-4B: 9.4).

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
