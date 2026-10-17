# Video-Index v1.1

Video-Index v1.1 is a harder subset for agents: **150 questions** from the 840 of Video-Index v1.0. Each comes with its
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
| Original source video no longer retrievable | 9 | 234 |
| Claude Opus 5.5, Gemini 4 Argon and GPT-6-Astra all correct with the original video | 84 | **150** |

Of the 40 questions that every agent got wrong and a person reviewed, 18 were kept: 14 with a correct key and 4 with
a corrected key. Two of the corrected keys remain in v1.1.

The 150 questions span 36 source benchmarks. By capability group: 35 perception, 37 temporal, 39 spatial and
physical, and 39 reasoning and knowledge. 70 videos have an audio track.

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

Accuracy on the 150 questions of v1.1. "No answer" counts timeouts and errors; they are scored as wrong. The 234
column is the accuracy before the last step, which removed the questions the top three agents solved; those three
agents are therefore lower on v1.1 by construction. Cost per question in USD:

- **Claude.** The cost reported by Claude Code.
- **Gemini and OpenAI.** Tokens times the proxy list price. The lower bound assumes cached input tokens cost 10% of
  the list price; the upper bound assumes they cost the full price.

| Model | Harness | Accuracy (150) | No answer | Accuracy (234) | Cost per question |
|---|---|---:|---:|---:|---:|
| claude-opus-5-5 | Claude Code 2.1.300 | 46.7 | 14 | 65.8 | 1.69 |
| claude-opus-5 | Claude Code 2.1.300 | 44.7 | 1 | 54.7 | 2.52 |
| gemini-4-argon | Gemini CLI 0.57.0 | 40.0 | 2 | 61.5 | 0.17–0.57 |
| claude-sonnet-5-5 | Claude Code 2.1.300 | 36.0 | 18 | 50.0 | 1.52 |
| gemini-3.8-flash | Gemini CLI 0.64.0 | 36.0 | 4 | 50.0 | 0.31–1.41 |
| gpt-6-astra | Codex CLI 0.154.0 | 35.3 | 13 | 58.5 | 0.41–1.61 |
| gpt-5.6-sol | Codex CLI 0.154.0 | 29.3 | 15 | 40.6 | 0.55–3.01 |
| gemini-3.6-flash | Gemini CLI 0.64.0 | 28.7 | 3 | 35.9 | 0.15–0.61 |
| claude-haiku-5-5 | Claude Code 2.1.300 | 27.3 | 1 | 38.9 | 0.26 |
| claude-opus-4-7 | Claude Code 2.1.300 | 26.0 | 0 | 34.6 | 2.67 |
| claude-opus-4-8 | Claude Code 2.1.300 | 26.0 | 1 | 35.9 | 1.38 |
| claude-sonnet-5 | Claude Code 2.1.300 | 26.0 | 2 | 29.9 | 1.48 |
| claude-opus-4-6 | Claude Code 2.1.300 | 22.0 | 2 | 29.1 | 1.23 |
| gemini-3-flash-preview | Gemini CLI 0.57.0 | 21.3 | 8 | 26.1 | 0.20–0.73 |
| gpt-5.6-terra | Codex CLI 0.154.0 | 20.7 | 15 | 34.2 | 0.45–2.54 |
| gemini-3.1-pro-preview | Gemini CLI 0.57.0 | 20.0 | 56 | 24.8 | 0.42–1.48 |
| gpt-5.6-luna | Codex CLI 0.154.0 | 20.0 | 15 | 28.2 | 0.04–0.26 |
| claude-opus-4-5 | Claude Code 2.1.300 | 18.0 | 1 | 24.8 | 0.77 |
| claude-sonnet-4-6 | Claude Code 2.1.300 | 18.0 | 4 | 25.2 | 1.23 |
| gemini-3.5-flash-lite | Gemini CLI 0.57.0 | 14.7 | 6 | 17.1 | 0.05–0.20 |
| claude-haiku-4-5 | Claude Code 2.1.300 | 14.0 | 8 | 17.1 | 0.22 |

The same numbers are in [scores.csv](scores.csv).

Notes:

- **OpenAI models.** The proxy passes only plain function tools, so Codex ran with its standard shell and image
  tools instead of code mode.
- **Excluded OpenAI models.** GPT-5.5, GPT-5.4, GPT-5.4-mini, GPT-5.4-nano and GPT-5.2 failed most tool calls
  through the proxy and are not reported.
- **Gemini 3.1 Pro.** It often ended its turn without an answer (56 of 150 questions).
