# Per-item comparison of the three harnesses

- video: 12 rows, 12 present in all three harnesses, 12 with the same correctness, 11 with the same reply text
- blind: 8 rows, 8 present in all three harnesses, 8 with the same correctness, 8 with the same reply text
- vi-eval, video: accuracy 8.33 over 12 items
- vi-eval, blind: accuracy 0.00 over 2 items
- VLMEvalKit, video: accuracy 8.33 over 12 items
- VLMEvalKit, blind: accuracy 0.00 over 2 items
- lmms-eval, video: accuracy 8.33 over 12 items
- lmms-eval, blind: accuracy 0.00 over 2 items
- lmms-eval with the default PNG encoding of the frames, video: 11 of 12 rows with the correctness of vi-eval, 9 with the same reply text

| Protocol | Item | Option order | vi-eval reply | VLMEvalKit reply | lmms-eval reply | vi-eval correct | VLMEvalKit correct | lmms-eval correct | Same correctness | lmms-eval, PNG frames: reply | correct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| video | AoTBench_5625 | 01 | A | A | A | 0 | 0 | 0 | yes | A | 0 |
| video | K9-Bench_120 | 01234 | D | A | D | 0 | 0 | 0 | yes | B | 0 |
| video | PerceptionTest_1028 | 012 | B | B | B | 0 | 0 | 0 | yes | B | 0 |
| video | SLVMBench_1500 | 01234 | C | C | C | 0 | 0 | 0 | yes | C | 0 |
| video | SiteBench-Video_3048 | 012 | C | C | C | 0 | 0 | 0 | yes | C | 0 |
| video | TempCompass_4224 | 0123 | A | A | A | 0 | 0 | 0 | yes | A | 0 |
| video | UrbanVideo-Bench_3940 | 01234 | E | E | E | 0 | 0 | 0 | yes | E | 0 |
| video | VF-Eval_1249 | 0123 | D | D | D | 0 | 0 | 0 | yes | D | 0 |
| video | VITATECS_11125 | 01 | B | B | B | 1 | 1 | 1 | yes | A | 0 |
| video | Video-MME_2177 | 0123 | D | D | D | 0 | 0 | 0 | yes | D | 0 |
| video | Video-TT_1339 | 0123 | D | D | D | 0 | 0 | 0 | yes | A | 0 |
| video | VideoAds_899 | 0123 | A | A | A | 0 | 0 | 0 | yes | A | 0 |
| blind | TempCompass_4224 | 0312 | A | A | A | 0 | 0 | 0 | yes |  |  |
| blind | TempCompass_4224 | 2013 | The provided text does not contain any i | The provided text does not contain any i | The provided text does not contain any i | 0 | 0 | 0 | yes |  |  |
| blind | TempCompass_4224 | 2310 | D | D | D | 0 | 0 | 0 | yes |  |  |
| blind | TempCompass_4224 | 3201 | The provided text does not contain any i | The provided text does not contain any i | The provided text does not contain any i | 0 | 0 | 0 | yes |  |  |
| blind | UrbanVideo-Bench_3940 | 02134 | A | A | A | 0 | 0 | 0 | yes |  |  |
| blind | UrbanVideo-Bench_3940 | 14302 | B | B | B | 0 | 0 | 0 | yes |  |  |
| blind | UrbanVideo-Bench_3940 | 23041 | C | C | C | 0 | 0 | 0 | yes |  |  |
| blind | UrbanVideo-Bench_3940 | 41302 | A | A | A | 0 | 0 | 0 | yes |  |  |
