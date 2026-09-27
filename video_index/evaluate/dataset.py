"""Loading Video-Index: from the Hugging Face hub (default ``GMLRVigil/Video-Index``) or from a local directory with
the same layout (``items/meta_benchmark.jsonl`` and ``videos/<video_id>.mp4``)."""
from __future__ import annotations

import json
import os

HUB_ID = "GMLRVigil/Video-Index"
ITEMS_FILE = "items/meta_benchmark.jsonl"


class VideoIndex:
    def __init__(self, data: str = HUB_ID, revision: str | None = None, cache_dir: str | None = None):
        self.data, self.revision, self.cache_dir = data, revision, cache_dir
        self.local = os.path.isdir(data)
        self.items = self._load_items()

    def _hub(self, filename: str) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.data, filename, repo_type="dataset", revision=self.revision, cache_dir=self.cache_dir)

    def _load_items(self) -> list[dict]:
        path = os.path.join(self.data, ITEMS_FILE) if self.local else self._hub(ITEMS_FILE)
        items = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["options"] = [str(o) for o in r["options"]]
                    r["answer_idx"] = int(r["answer_idx"])
                    items.append(r)
        return items  # the order of the item file: the toolkit integrations count --offset / --limit in this order

    def video_path(self, item: dict) -> str:
        """Local path of the item's video (downloaded on first use when the data come from the hub)."""
        rel = item.get("video") or f"videos/{item['video_id']}.mp4"
        if self.local:
            p = os.path.join(self.data, rel)
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            return os.path.realpath(p)
        return self._hub(rel)

    def fetch_all_videos(self) -> str:
        """Download every video in one call (hub only); returns the snapshot directory."""
        if self.local:
            return self.data
        from huggingface_hub import snapshot_download
        return snapshot_download(self.data, repo_type="dataset", revision=self.revision, cache_dir=self.cache_dir,
                                 allow_patterns=["items/*", "videos/*"])

    def select(self, limit: int | None = None, shard: str | None = None, seed: int = 0, ids: str | None = None,
               window: str | None = None) -> list[dict]:
        """``ids``: item ids, comma separated or a file with one id per line; ``window`` = "start:stop" takes
        consecutive items in the order of the item file; ``shard`` = "i/n" takes every n-th item; ``limit`` draws
        a seeded random subset (the same for every model)."""
        import random
        items = list(self.items)
        if ids:
            if os.path.isfile(ids):
                with open(ids) as f:
                    wanted = [line.strip() for line in f if line.strip()]
            else:
                wanted = [x.strip() for x in ids.split(",") if x.strip()]
            missing = sorted(set(wanted) - {r["item_id"] for r in items})
            if missing:
                raise KeyError(f"{len(missing)} item id(s) not in Video-Index, first: {missing[0]}")
            wanted = set(wanted)
            items = [r for r in items if r["item_id"] in wanted]
        if window:
            start, _, stop = window.partition(":")
            items = items[int(start or 0):int(stop) if stop else None]
        if shard:
            i, n = (int(x) for x in shard.split("/"))
            items = [r for k, r in enumerate(items) if k % n == i]
        if limit:
            items = random.Random(seed).sample(items, min(limit, len(items)))
        return items
