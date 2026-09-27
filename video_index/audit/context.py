"""Run context shared by the stages of the audit and onboarding pipelines."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..config import Models, load_yaml

KINDS = {"annotations": "annotations/{bench}", "items": "items/{bench}.jsonl", "samples": "samples/{bench}.jsonl",
         "video_dir": "videos/{video_id}", "video": "videos/{video_id}/video.mp4", "video_meta": "videos/{video_id}/meta.json",
         "captions": "captions/{bench}__{model}.jsonl", "result": "results/{condition}/{bench}__{model}.jsonl",
         "attribution": "attribution/{stage}/{bench}.jsonl", "table": "tables/{name}", "pool": "pool/{name}", "log": "logs/{name}"}

DEFAULTS = dict(seed=42, sample_n=300, delta=0.05, reference_frames=32, reference_long_frames=128, long_video_s=300,
                frame_long_side=768, attacker_short_side=224, workers=8, permutations=8,
                normalize=dict(max_fps=2.0, max_frames=1024, short_side=720, short_side_long=480, long_s=1800))


@dataclass
class Context:
    work_dir: str
    models: Models
    cfg: dict = field(default_factory=dict)

    @classmethod
    def from_files(cls, run_cfg: str, models_cfg: str | None = None, work_dir: str | None = None) -> "Context":
        cfg = {**DEFAULTS, **load_yaml(run_cfg)} if run_cfg else dict(DEFAULTS)
        cfg["normalize"] = {**DEFAULTS["normalize"], **(cfg.get("normalize") or {})}
        models = Models.from_file(models_cfg or cfg.get("models_file") or "configs/models.yaml")
        return cls(os.path.abspath(work_dir or cfg.get("work_dir") or "work"), models, cfg)

    def path(self, kind: str, **kw) -> str:
        if "model" in kw:
            kw["model"] = str(kw["model"]).replace("/", "_")
        p = os.path.join(self.work_dir, KINDS[kind].format(**kw))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def benchmarks(self) -> list[str]:
        d = os.path.join(self.work_dir, "samples")
        return sorted(f[:-6] for f in os.listdir(d) if f.endswith(".jsonl")) if os.path.isdir(d) else []

    def get(self, key, default=None):
        return self.cfg.get(key, default)
