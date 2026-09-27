"""Prompts of the attack pyramid, verbatim from the study."""

ANSWER_INSTR = "Reply with ONLY the option letter (or the exact short answer if no options)."

# text attacker: question and options, no video
BLIND = ("You will answer a question about a video WITHOUT seeing the video. "
         "Guess the most likely answer from the question and options alone.\n\n"
         "Question: {q}\n{opts}\n"
         "Reply with ONLY the option letter (or the exact short answer if no options).")

# text attacker: options only
OPTIONS_ONLY = ("You will see ONLY the answer options of a multiple-choice question about a video. "
                "The question and the video are not shown. Choose the option that is most likely "
                "to be the correct answer.\n\nOptions:\n{opts}\n\n"
                "Reply with ONLY the option letter.")

# frames (reference and attacker roles; the shuffle and window probes use the same wording on purpose)
VISUAL = ("You are given {n} frame(s) sampled from a video. "
          "Answer the question based on these frames.\n\n"
          "Question: {q}\n{opts}\n" + ANSWER_INSTR)

# attacker without frames (options-only under permutations)
ATTACKER_OPTIONS_ONLY = ("You are given NO frames from the video and NO question text. "
                         "Using only the answer options below, choose the option most likely to be correct."
                         "\n\n{opts}\n" + ANSWER_INSTR)

# video caption written by the captioner from 32 frames
CAPTION_VIDEO = ("Describe this video in detail based on the frames provided, in temporal order. "
                 "Cover: the scene and setting, every visible object and person, all actions and "
                 "events as they unfold, any on-screen text, and camera movement. Be factual and "
                 "specific; do not speculate beyond what is visible.")

# caption reader, one caption of the whole video
READ_VIDEO_CAPTION = ("You are given a text description of a video INSTEAD of the video "
                      "itself. Answer the question based on the description.\n\n"
                      "Video description: {cap}\n\n"
                      "Question: {q}\n{opts}\n" + ANSWER_INSTR)

# per-frame captions: one frame per captioner call, never the other frames
CAPTION_FRAME = ("Describe this image in one short sentence: the visible objects, people, actions and setting. "
                 "Do not guess what happened before or after.")

READ_FRAME_CAPTIONS = ("You are given short captions of {n} frames sampled from a video, one caption per frame, in frame order. "
                       "You cannot see the video. Answer the question from these captions."
                       "\n\nFrame captions:\n{caps}\n\nQuestion: {q}\n{opts}\n" + ANSWER_INSTR)
