"""Order level, probe 2: 32 frames inside one contiguous tenth of the video."""
from .conditions import run_condition, visual_builder

ROLES = ["reference", "attacker"]


def run(ctx, benchmarks=None, limit=None, roles=None):
    k = int(ctx.get("reference_frames", 32))
    return [s for role in (roles or ctx.get("order_roles", ROLES))
            for s in run_condition(ctx, "window", role, visual_builder(ctx, role, "window", k), benchmarks, limit)]
