"""Text level: the text attacker answers from the question and the options, without the video."""
from .conditions import blind_builder, run_condition

ROLES = ["text_attacker"]


def run(ctx, benchmarks=None, limit=None):
    return [s for role in ctx.get("blind_roles", ROLES)
            for s in run_condition(ctx, "blind", role, blind_builder(ctx), benchmarks, limit)]
