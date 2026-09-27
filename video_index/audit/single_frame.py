"""Frame level: one frame of the video (roles reference and attacker)."""
from .conditions import run_condition, visual_builder

ROLES = ["reference", "attacker"]


def run(ctx, benchmarks=None, limit=None, roles=None):
    return [s for role in (roles or ctx.get("single_frame_roles", ROLES))
            for s in run_condition(ctx, "single_frame", role, visual_builder(ctx, role, "single", 1), benchmarks, limit)]
