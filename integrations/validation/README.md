# Agreement of the three harnesses

Check of 2026-09-26: `gemini-2.5-flash-lite`, items 624 to 635 of the item file (12 items, 12 source benchmarks,
2 to 5 options), frame cap 8, output limit 16 tokens; blind protocol on items 624 and 625 (8 requests).

| Harness | Backend | Video accuracy | Blind accuracy |
|---|---|---|---|
| `vi-eval` | Gemini API (`provider=gemini`) | 8.33 (1 of 12) | 0.00 |
| VLMEvalKit | `Gemini` wrapper (google-genai) | 8.33 (1 of 12) | 0.00 |
| lmms-eval | `openai` wrapper, OpenAI-compatible endpoint of the Gemini API, `LMMS_IMAGE_ENCODE_FORMAT=JPEG` | 8.33 (1 of 12) | 0.00 |

| Comparison | Video (12 rows) | Blind (8 rows) |
|---|---|---|
| Same correctness in the three harnesses | 12 | 8 |
| Same reply text in the three harnesses | 11 | 8 |
| lmms-eval with the default PNG encoding: same correctness as `vi-eval` | 11 | not run |
| lmms-eval with the default PNG encoding: same reply text as `vi-eval` | 9 | not run |

The reply that differs between the harnesses (K9-Bench_120: `D`, `A`, `D`) is wrong in all three. The slice comes
from the hardest part of the set; the accuracy of the model on it carries no information about the model.

Files:

| File | Content |
|---|---|
| `parity_report.md` | reply and correctness per row |
| `frames_report.md` | frame indices, sizes and pixel differences per video |
| `compare_harnesses.py`, `compare_frames.py` | the scripts that write the two reports |

Frames (`frames_report.md`): the frame indices and sizes agree on the 12 videos. Decoded in stream order, the
frames of decord equal the frames of OpenCV on 12 of 12 videos (pixel difference 0). With random access
(`reader[i]`), decord returned other frames on 7 of 12 videos (mean absolute pixel difference 1.0 to 17.8 of 255);
the integrations therefore decode in stream order.
