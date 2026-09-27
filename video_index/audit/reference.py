"""Reference run: 32 uniform frames, long side 768. The reference accuracy s* of the breaking levels.
The attacker role at 32 frames is the second reference and permutation 0 of the option-permutation audit."""
from .conditions import run_condition, visual_builder

ROLES = ["reference", "attacker"]


def run(ctx, benchmarks=None, limit=None, roles=None):
    k = int(ctx.get("reference_frames", 32))
    return [s for role in (roles or ctx.get("reference_roles", ROLES))
            for s in run_condition(ctx, "reference", role, visual_builder(ctx, role, "uniform", k), benchmarks, limit)]
