"""Convert Video-Index questions into Harbor task directories.

One task per question (protocol ``video``), or one task per question and option permutation (protocol ``blind``:
question and options only, four permutations fixed by ``random.Random(f"42|{item_id}")`` as in the paper; the item
score is the mean reward over its four tasks).

Video delivery: by default the task's Dockerfile downloads the video from the dataset repository at a pinned
revision when the image is built (the agent itself runs without network). ``embed_videos=True`` copies the mp4
into the task's ``environment/`` directory instead, for offline use.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

TEMPLATE = Path(__file__).parent / "task-template"
LETTERS = "ABCDEFGHIJKLMNOP"
REPO = "Video-Index/Video-Index"
N_PERM, SEED = 4, 42


def permutations_for(item_id: str, k: int, n_perm: int = N_PERM, seed: int = SEED) -> list[list[int]]:
    rng = random.Random(f"{seed}|{item_id}")
    return [rng.sample(range(k), k) for _ in range(n_perm)]


def task_name(item_id: str, suffix: str = "") -> str:
    s = re.sub(r"[^a-z0-9]+", "-", item_id.lower()).strip("-")
    return s + suffix


@dataclass
class Item:
    item_id: str
    benchmark: str
    capability_group: str
    fine_category: str
    question: str
    options: list[str]
    answer: str
    answer_idx: int
    chance: float
    video: str
    video_id: str
    duration_s: float

    @classmethod
    def from_row(cls, r: dict) -> "Item":
        opts = r["options"]
        if isinstance(opts, str):
            import ast
            opts = ast.literal_eval(opts)
        return cls(r["item_id"], r["benchmark"], r["capability_group"], r.get("fine_category", ""), r["question"].strip(),
                   [str(o).strip() for o in opts], r["answer"], int(r["answer_idx"]), float(r["chance"]), r["video"],
                   r["video_id"], float(r.get("duration_s") or 0))


def load_items(source: str, revision: str | None = None) -> tuple[list[Item], str, str]:
    """(items, video root or None, resolved revision). ``source`` is a local directory with ``items/`` and ``videos/``,
    or a dataset repository id on the Hugging Face hub."""
    if os.path.isdir(source):
        p = Path(source) / "items" / "test.jsonl"
        if not p.exists():
            p = Path(source) / "items" / "meta_benchmark.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        return [Item.from_row(r) for r in rows], str(Path(source)), revision or "main"
    from huggingface_hub import HfApi, hf_hub_download
    rev = revision or HfApi().repo_info(source, repo_type="dataset").sha
    path = hf_hub_download(source, "items/test.jsonl", repo_type="dataset", revision=rev)
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [Item.from_row(r) for r in rows], None, rev


def _fill(text: str, values: dict) -> str:
    for k, v in values.items():
        text = text.replace("{" + k + "}", str(v))
    return text


class VideoIndexAdapter:
    def __init__(self, output_dir: Path, source: str = REPO, revision: str | None = None, protocol: str = "video",
                 embed_videos: bool = False):
        if protocol not in ("video", "blind"):
            raise ValueError("protocol must be 'video' or 'blind'")
        self.output_dir = Path(output_dir)
        self.source, self.protocol, self.embed_videos = source, protocol, embed_videos
        self.items, self.video_root, self.revision = load_items(source, revision)
        self.repo = source if not os.path.isdir(source) else REPO

    def units(self, task_ids: list[str] | None = None, limit: int | None = None):
        items = [it for it in self.items if not task_ids or it.item_id in task_ids or task_name(it.item_id) in task_ids]
        if limit:
            items = items[:limit]
        for it in items:
            if self.protocol == "video":
                yield it, list(range(len(it.options))), task_name(it.item_id)
            else:
                for k, perm in enumerate(permutations_for(it.item_id, len(it.options))):
                    yield it, perm, task_name(it.item_id, f"-p{k}")

    def generate(self, task_ids=None, limit=None, overwrite=False) -> int:
        n = 0
        for it, perm, name in self.units(task_ids, limit):
            out = self.output_dir / name
            if out.exists():
                if not overwrite:
                    continue
                shutil.rmtree(out)
            self._write(it, perm, name, out)
            n += 1
        return n

    def _write(self, it: Item, perm: list[int], name: str, out: Path) -> None:
        (out / "environment").mkdir(parents=True)
        (out / "solution").mkdir()
        (out / "tests").mkdir()
        options = [it.options[j] for j in perm]
        letters = LETTERS[:len(options)]
        answer = letters[perm.index(it.answer_idx)]
        values = dict(task_name=name, item_id=it.item_id, benchmark=it.benchmark.replace('"', "'"),
                      capability_group=it.capability_group, fine_category=it.fine_category, duration_s=it.duration_s,
                      chance=it.chance, protocol=self.protocol, answer=answer, letters=", ".join(letters),
                      letters_compact=letters, question=it.question,
                      options="\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options)),
                      tags=json.dumps(["video-index", "multiple-choice", "video", it.capability_group]))
        (out / "task.toml").write_text(_fill((TEMPLATE / "task.toml").read_text(), values))
        (out / "instruction.md").write_text(_fill((TEMPLATE / f"instruction_{self.protocol}.md").read_text(), values))
        for f in ("solution/solve.sh", "tests/test.sh"):
            (out / f).write_text(_fill((TEMPLATE / f).read_text(), values))
            os.chmod(out / f, 0o755)
        docker = (TEMPLATE / "environment" / f"Dockerfile.{self.protocol}").read_text()
        if self.protocol == "video":
            if self.embed_videos:
                if not self.video_root:
                    from huggingface_hub import hf_hub_download
                    src = hf_hub_download(self.repo, it.video, repo_type="dataset", revision=self.revision)
                else:
                    src = os.path.join(self.video_root, it.video)
                shutil.copy2(os.path.realpath(src), out / "environment" / "video.mp4")
                step = "COPY video.mp4 /app/video.mp4"
            else:
                url = f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{it.video}"
                step = f"ADD {url} /app/video.mp4"
            docker = docker.replace("{video_step}", step)
        (out / "environment" / "Dockerfile").write_text(docker)
