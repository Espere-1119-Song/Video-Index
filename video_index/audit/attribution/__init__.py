"""Error attribution: why the reference model fails on the items it gets wrong at 32 frames.

Stages (each a module with ``run(ctx, benchmarks=None, limit=None)``), in order:

    attribution_traces     rerun every reference error on the same frames with written reasoning
    attribution_preclass   rule-based pre-classification from the text-only run
    attribution_judge      two independent judges, agreement levels, dense-frame recheck
    attribution_checks     unstable-failure check (shifted frame grid), judge self-attribution check
    attribution_tables     share tables, fine-category tables, error matrices
    attribution_discover   optional: clusters the failures the taxonomy does not capture
"""
STAGES = {
    "attribution_traces": "video_index.audit.attribution.traces",
    "attribution_preclass": "video_index.audit.attribution.preclass",
    "attribution_judge": "video_index.audit.attribution.judge",
    "attribution_checks": "video_index.audit.attribution.checks",
    "attribution_tables": "video_index.audit.attribution.tables",
    "attribution_discover": "video_index.audit.attribution.discover",
}
ORDER = ["attribution_traces", "attribution_preclass", "attribution_judge", "attribution_checks", "attribution_tables"]
