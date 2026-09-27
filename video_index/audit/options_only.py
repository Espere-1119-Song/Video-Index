"""Option level: the text attacker sees the options only (no question, no video). Items without options are skipped."""
from .conditions import options_only_builder, run_condition

ROLES = ["text_attacker"]


def run(ctx, benchmarks=None, limit=None):
    return [s for role in ctx.get("options_only_roles", ROLES)
            for s in run_condition(ctx, "options_only", role, options_only_builder(ctx), benchmarks, limit)]
