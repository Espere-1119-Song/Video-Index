"""Configuration: YAML files with ``${ENV_VAR}`` / ``${ENV_VAR:-default}`` interpolation, and the model roles.

Roles used by the pipelines (each is one entry under ``models:`` in ``configs/models.yaml``):

============== ==========================================================================================
role           what the model does
============== ==========================================================================================
reference      answers every item from 32 frames; its accuracy is the reference score s* of a benchmark
reference_long 128-frame reference for long-video benchmarks whose 32-frame score is near chance (optional;
               defaults to ``reference``)
attacker       the diagnostic open-weight model of the attack levels (options-only permutations, single
               frame, shuffled frames, first tenth, resolution and frame-rate ladders)
attacker_small the small open-weight model of the pool screen (32 frames)
text_attacker  answers from the question and options alone (blind) and from the options alone
captioner      writes per-video and per-frame captions for the caption attack
caption_reader answers from captions alone
judge_a        first judge of the error attribution
judge_b        second judge of the error attribution
labeler        assigns a capability category to every item
verifier       checks the marked answer of a Video-Index candidate against the frames
============== ==========================================================================================

A role that is not configured falls back along ``FALLBACK`` (for example ``judge_b`` -> ``judge_a`` -> ``reference``).
"""
from __future__ import annotations

import os
import re
from typing import Any

import yaml

from .models import ChatModel, ModelError, build_model

ROLES = ("reference", "reference_long", "attacker", "attacker_small", "text_attacker", "captioner", "caption_reader",
         "judge_a", "judge_b", "labeler", "verifier")
FALLBACK = {"reference_long": "reference", "text_attacker": "reference", "caption_reader": "text_attacker",
            "attacker_small": "attacker", "captioner": "attacker", "judge_a": "reference", "judge_b": "judge_a",
            "labeler": "judge_a", "verifier": "reference"}
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interp(x: Any) -> Any:
    if isinstance(x, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else ""), x)
    if isinstance(x, dict):
        return {k: _interp(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_interp(v) for v in x]
    return x


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return _interp(yaml.safe_load(f) or {})


class Models:
    """Lazy role -> model map. ``models["judge_a"]`` builds the client on first use."""

    def __init__(self, specs: dict[str, dict]):
        self.specs = specs
        self._built: dict[str, ChatModel] = {}

    @classmethod
    def from_file(cls, path: str) -> "Models":
        cfg = load_yaml(path)
        return cls(cfg.get("models", cfg))

    def resolve(self, role: str) -> str:
        seen = set()
        while role not in self.specs:
            if role in seen or role not in FALLBACK:
                raise ModelError(f"no model configured for role '{role}' (configured: {sorted(self.specs)})")
            seen.add(role)
            role = FALLBACK[role]
        return role

    def __contains__(self, role: str) -> bool:
        try:
            self.resolve(role)
            return True
        except ModelError:
            return False

    def __getitem__(self, role: str) -> ChatModel:
        key = self.resolve(role)
        if key not in self._built:
            spec = dict(self.specs[key])
            spec.setdefault("name", spec.get("model", key))
            self._built[key] = build_model(spec)
        return self._built[key]

    def label(self, role: str) -> str:
        key = self.resolve(role)
        return self.specs[key].get("name") or self.specs[key].get("model") or key
