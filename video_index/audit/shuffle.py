"""Order level, probe 1: the 32 frames of the reference run in a random order; the prompt does not mention the order."""
from .conditions import run_condition, visual_builder

ROLES = ["reference", "attacker"]


def run(ctx, benchmarks=None, limit=None, roles=None):
    k = int(ctx.get("reference_frames", 32))
    return [s for role in (roles or ctx.get("order_roles", ROLES))
            for s in run_condition(ctx, "shuffle", role, visual_builder(ctx, role, "shuffle", k), benchmarks, limit)]
