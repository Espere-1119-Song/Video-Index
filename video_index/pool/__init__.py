"""Item pool: every multiple-choice item of the audited benchmarks, screened by the attack pyramid, labelled, and
reduced to the coverage-first selection of 10,000 items.

Stages (each a module with ``run(ctx, benchmarks=None, limit=None)``), in order:

    pool_items    item table of all benchmarks (pool/pool_items_base)
    pool_screen   sequential funnel in pyramid order (pool/pool_items, pool/funnel.csv)
    pool_labels   scene type per video and capability per item through the ``labeler`` role
    pool_select   visual scene clusters, coverage-first selection, export of the selected items
"""
STAGES = {
    "pool_items": "video_index.pool.items",
    "pool_screen": "video_index.pool.screen",
    "pool_labels": "video_index.pool.labels",
    "pool_select": "video_index.pool.select",
}
ORDER = ["pool_items", "pool_screen", "pool_labels", "pool_select"]
