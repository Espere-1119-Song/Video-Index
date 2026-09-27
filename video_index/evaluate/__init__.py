"""Pipeline 3: evaluation on Video-Index (``vi-eval``)."""
from .dataset import HUB_ID, VideoIndex
from .protocol import blind_prompt, permutations_for, score_reply, video_prompt

__all__ = ["HUB_ID", "VideoIndex", "blind_prompt", "permutations_for", "score_reply", "video_prompt"]
