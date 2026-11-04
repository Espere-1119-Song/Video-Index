"""Mage-VL as an in-process model for vi-eval.

    PYTHONPATH=integrations/mage_vl vi-eval run --local mage_vl:MageVL --workers 1 --max-tokens 16 --out runs/mage-vl
"""
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MAGE_VL_REVISION = "d88b153285f1633a61b2f693c59c8576693af185"


class _StampedVideoProcessor:
    """Gives pre-decoded frames their real times.

    Mage-VL's processor labels pre-decoded frames <0.0 seconds>, <1.0 seconds>, ..., which is
    wrong once a long video is thinned to the frame cap.
    """

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "timestamps", None)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        if name == "timestamps":
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)

    def __call__(self, *args, **kwargs):
        out = self._inner(*args, **kwargs)
        if self.timestamps is not None:
            out["frame_timestamps"] = [[float(t) for t in self.timestamps]]
        return out


class MageVL:
    name = "mage-vl-4b"
    threaded = False

    def __init__(self, checkpoint: str = "microsoft/Mage-VL", revision: str = MAGE_VL_REVISION,
                 max_new_tokens: int = 16):
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, revision=revision, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa", trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)
        self.stamped = self.processor.video_processor = _StampedVideoProcessor(self.processor.video_processor)
        self.max_new_tokens = max_new_tokens

    def generate(self, inputs) -> str:
        tok = self.processor.tokenizer
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
        return tok.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def chat_text(self, prompt: str, video: bool) -> str:
        content = ([{"type": "video"}] if video else []) + [{"type": "text", "text": prompt}]
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )

    def answer(self, frames, timestamps, prompt) -> str:
        # Frames before the text, as the protocol prescribes, stamped with their real times.
        self.stamped.timestamps = list(timestamps) if frames else None
        videos = {"videos": [list(frames)]} if frames else {}
        inputs = self.processor(text=[self.chat_text(prompt, bool(frames))], return_tensors="pt",
                                padding=True, **videos)
        return self.generate(inputs)
