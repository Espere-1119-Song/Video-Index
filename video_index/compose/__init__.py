"""Composition of Video-Index from the screened pool.

Stages (each a module with ``run(ctx, benchmarks=None, limit=None)``), in order:

    compose_candidates   worst-case attacker percentile per item, the 3,000 hardest items (one per video),
                         capability labels through the ``labeler`` role
    compose_verify       the marked answer of every candidate is checked against the frames by the ``verifier``
                         role (32 frames, then 512 for the unconfirmed items)
    compose_select       per capability group the 210 hardest verified items, one item per video
    compose_export       release layout: items/meta_benchmark.jsonl, videos/, README.md
"""
STAGES = {
    "compose_candidates": "video_index.compose.candidates",
    "compose_verify": "video_index.compose.verify",
    "compose_select": "video_index.compose.select",
    "compose_export": "video_index.compose.export",
}
ORDER = ["compose_candidates", "compose_verify", "compose_select", "compose_export"]
CAPS = ["perception", "temporal", "spatial_physical", "reasoning_knowledge"]
DEFAULTS = dict(candidates=3000, budget=840, verify_frames=32, recheck_frames=512, short_video_s=60.0,
                reference_cap=None, fill_short_groups=False)


def cfg(ctx, key):
    return (ctx.get("compose") or {}).get(key, DEFAULTS.get(key))


def path(ctx, name):
    return ctx.path("pool", name=f"compose/{name}")
