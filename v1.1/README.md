# Video-Index v1.1

Video-Index v1.1 is a harder subset for agents: **144 questions** from the 840 of Video-Index v1.0. Each comes with its
original source video, including audio. The questions and videos are on Hugging Face at
[Video-Index/Video-Index-v1-1](https://huggingface.co/datasets/Video-Index/Video-Index-v1-1), together with the
text-only traces of all 21 agents. This folder holds the scores.

## Construction

| Step | Questions removed | Remaining |
|---|---:|---:|
| Video-Index v1.0 | | 840 |
| All four agents correct on the v1.0 videos (GPT-6-Astra, Claude Fable 5.1, Claude Opus 5.5, Gemini 4 Argon). For questions without Fable and Astra answers: Opus 5.5 and Gemini 4 both correct | 454 + 84 | 302 |
| All four agents wrong and all chose the same option | 37 | 265 |
| All four agents wrong, then judged by a person as ambiguous or not decidable from the video | 22 | 243 |
| Original source video not retrievable with both its picture and its sound: 9 no longer available, and 7 YouTube videos for which only the audio or only the picture could be downloaded | 16 | 227 |
| Claude Opus 5.5, Gemini 4 Argon and GPT-6-Astra all correct with the original video | 83 | **144** |

Of the 40 questions that every agent got wrong and a person reviewed, 18 were kept: 14 with a correct key and 4 with
a corrected key. Two of the corrected keys remain in v1.1.

The 144 questions span 33 source benchmarks. By capability group: 33 perception, 35 temporal, 39 spatial and
physical, and 37 reasoning and knowledge. 66 videos have an audio track.

## Agent protocol

- **Input.** The original video as `video.mp4` in the working directory. Nothing is pre-extracted, and there is no
  limit on resolution or frame rate.
- **Tools.** A shell with ffmpeg/ffprobe and Python (OpenCV, Pillow, NumPy, and faster-whisper with local Whisper
  large-v3 and small models). Each CLI also has its own image viewer. In Gemini CLI, `read_file` also opens audio
  files.
- **Rules.** No web search, web fetch or downloads. One attempt per question, 3600 seconds per question. A timeout or
  an error counts as wrong.
- **Harnesses.**
  - Claude models: Claude Code 2.1.300, reasoning effort max.
  - Gemini models: Gemini CLI.
  - OpenAI models: Codex CLI 0.154.0, reasoning effort max.

All calls went through the Kaggle model proxy.

## Results

Accuracy on the 144 questions of v1.1. "No answer" counts timeouts and errors; they are scored as wrong. The 227
column is the accuracy before the last step, which removed the questions the top three agents solved; those three
agents are therefore lower on v1.1 by construction. Cost per question in USD:

- **Claude.** The cost reported by Claude Code.
- **Gemini and OpenAI.** Tokens times the proxy list price. The lower bound assumes cached input tokens cost 10% of
  the list price; the upper bound assumes they cost the full price.

Time per question is the mean wall-clock time of one call in minutes, with timeouts counted as 60 minutes. It
depends on the proxy's latency and on how many calls ran at once (12 per Claude or Gemini model, 8 per OpenAI model),
so it compares agents only roughly.

| Model | Harness | Accuracy (144) | No answer | Accuracy (227) | Time per question (min) | Cost per question |
|---|---|---:|---:|---:|---:|---:|
| claude-opus-5-5 | Claude Code 2.1.300 | 47.9 | 12 | 67.0 | 18.9 | 1.72 |
| claude-opus-5 | Claude Code 2.1.300 | 46.5 | 1 | 56.4 | 14.1 | 2.57 |
| gemini-4-argon | Gemini CLI 0.57.0 | 41.0 | 2 | 62.6 | 4.1 | 0.17–0.58 |
| gemini-3.8-flash | Gemini CLI 0.64.0 | 37.5 | 4 | 51.1 | 10.3 | 0.32–1.43 |
| claude-sonnet-5-5 | Claude Code 2.1.300 | 36.8 | 18 | 51.1 | 22.2 | 1.51 |
| gpt-6-astra | Codex CLI 0.154.0 | 35.4 | 12 | 59.0 | 1.8 | 0.41–1.63 |
| gpt-5.6-sol | Codex CLI 0.154.0 | 30.6 | 14 | 41.9 | 4.8 | 0.53–2.87 |
| gemini-3.6-flash | Gemini CLI 0.64.0 | 29.2 | 3 | 36.6 | 10.6 | 0.15–0.62 |
| claude-haiku-5-5 | Claude Code 2.1.300 | 27.8 | 1 | 39.6 | 12.5 | 0.27 |
| claude-opus-4-7 | Claude Code 2.1.300 | 27.1 | 0 | 35.7 | 8.1 | 2.73 |
| claude-opus-4-8 | Claude Code 2.1.300 | 27.1 | 1 | 37.0 | 9.2 | 1.40 |
| claude-sonnet-5 | Claude Code 2.1.300 | 27.1 | 2 | 30.8 | 13.9 | 1.51 |
| claude-opus-4-6 | Claude Code 2.1.300 | 22.2 | 2 | 29.1 | 10.4 | 1.25 |
| gemini-3-flash-preview | Gemini CLI 0.57.0 | 22.2 | 8 | 26.9 | 8.0 | 0.20–0.73 |
| gpt-5.6-terra | Codex CLI 0.154.0 | 21.5 | 15 | 35.2 | 7.8 | 0.44–2.54 |
| gemini-3.1-pro-preview | Gemini CLI 0.57.0 | 20.8 | 53 | 25.6 | 7.8 | 0.42–1.49 |
| gpt-5.6-luna | Codex CLI 0.154.0 | 20.8 | 15 | 29.1 | 7.2 | 0.05–0.26 |
| claude-opus-4-5 | Claude Code 2.1.300 | 18.1 | 1 | 24.7 | 7.0 | 0.78 |
| claude-sonnet-4-6 | Claude Code 2.1.300 | 18.1 | 4 | 25.6 | 11.9 | 1.25 |
| gemini-3.5-flash-lite | Gemini CLI 0.57.0 | 15.3 | 6 | 17.6 | 10.0 | 0.05–0.20 |
| claude-haiku-4-5 | Claude Code 2.1.300 | 14.6 | 8 | 17.6 | 6.8 | 0.22 |

The same numbers are in [scores.csv](scores.csv).

Notes:

- **OpenAI models.** The proxy passes only plain function tools, so Codex ran with its standard shell and image
  tools instead of code mode.
- **Excluded OpenAI models.** GPT-5.5, GPT-5.4, GPT-5.4-mini, GPT-5.4-nano and GPT-5.2 failed most tool calls
  through the proxy and are not reported.
- **Gemini 3.1 Pro.** It often ended its turn without an answer (53 of 144 questions).

## Changes

- **2026-11-03.** The first release had 150 questions. Six of their videos were incomplete YouTube downloads: four
  K9-Bench videos had only the audio track, and one HLV-1K and one ReXTime video had only the picture although the
  YouTube source has sound. We removed these six questions, and one more incomplete download from the 234-question
  evaluation set, and recomputed all scores. The agents' answers on the remaining questions are unchanged.
