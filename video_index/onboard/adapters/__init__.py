"""Adapters: annotation files of a benchmark -> items of the shared schema."""
from __future__ import annotations

from .base import (is_yes_no, normalize_options, options_from_dict, resolve_answer, split_inline_options,
                   strip_sequential_prefix, to_item, yes_no_options)
from .generic import (AdapterError, AdapterSpec, GenericAdapter, detect, load_annotations, load_hook,
                      register_adapter, render)


def build_items(name: str, annotation_dir: str, spec: AdapterSpec | None = None, resolver=None):
    """Run the adapter and resolve every record into an item.

    ``resolver(v) -> video reference or None`` maps the adapter's video reference to a file of the video source
    (:class:`video_index.onboard.sources.VideoIndex`); None keeps the reference as given. Items whose video does not
    resolve are left out and counted. Returns (items, info)."""
    adapter = GenericAdapter(spec, name=name)
    adapter.spec.name = name
    raws, info = adapter.records(annotation_dir)
    items, misses, filtered = [], 0, 0
    for i, raw in enumerate(raws):
        if not adapter.keep(raw):
            filtered += 1
            continue
        ref = raw["v"]
        if resolver is not None:
            ref = resolver(raw["v"])
            if ref is None:
                misses += 1
                continue
        elif isinstance(ref, (tuple, list)):
            ref = dict(archive=ref[-2] if len(ref) >= 2 else "", member=ref[-1])
        it = to_item(name, i, raw)
        it["video_ref"] = ref if isinstance(ref, dict) else str(ref)
        items.append(it)
    total = len(raws) - filtered
    info.update(n_records=len(raws), n_filtered=filtered, n_unresolved=misses, n_items=len(items),
                resolution_rate=(len(items) / total if total else 0.0))
    return items, info


__all__ = ["AdapterError", "AdapterSpec", "GenericAdapter", "build_items", "detect", "is_yes_no", "load_annotations",
           "load_hook", "normalize_options", "options_from_dict", "register_adapter", "render", "resolve_answer",
           "split_inline_options", "strip_sequential_prefix", "to_item", "yes_no_options"]
