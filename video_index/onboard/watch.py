"""Candidate benchmarks: a list to read, not a trigger.

``vi-onboard watch`` searches the Hugging Face hub (datasets) and arXiv for the configured keywords and merges what it
finds into ``tables/candidates.csv``. Every candidate has a ``status``:

    new         found by the watcher, not looked at
    approved    a person decided that it should be onboarded (edit the file)
    rejected    a person decided against it; the ``note`` column says why
    onboarded   set after ``vi-onboard add`` finished for it

The watcher never onboards anything. :func:`approved` lists the candidates whose status a person set to ``approved``.
With ``--label`` the ``labeler`` role reads title and summary and writes a suggestion into ``suggestion``; the status
stays ``new``.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import requests

from ..runner import log as _log
from .registry import read_csv, upsert

COLS = ["key", "name", "origin", "url", "date", "summary", "status", "suggestion", "note", "discovered"]
KEYWORDS = ["video benchmark", "video question answering benchmark", "video understanding benchmark"]
HF_API = "https://huggingface.co/api/datasets"
ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM = {"a": "http://www.w3.org/2005/Atom"}
LABEL_PROMPT = ("A new dataset or paper is described below. Is it a benchmark that evaluates video understanding with "
                "multiple-choice questions in English? Answer on one line as: yes | no | unclear, then a reason of at "
                "most 15 words.\n\nTitle: {name}\nSummary: {summary}")


def log(m: str) -> None:
    _log(m, "watch")


def search_hub(keyword: str, limit: int = 50) -> list[dict]:
    r = requests.get(HF_API, params=dict(search=keyword.split()[0], filter="video", sort="lastModified", direction=-1,
                                         limit=limit, full="true"), timeout=60)
    r.raise_for_status()
    terms = [t for t in re.split(r"\W+", keyword.lower()) if t]
    out = []
    for d in r.json():
        card = d.get("cardData") or {}
        text = " ".join([d.get("id", ""), str(card.get("pretty_name", "")), " ".join(d.get("tags") or [])]).lower()
        if not all(t in text for t in terms if t not in ("benchmark",)) and "bench" not in text:
            continue
        out.append(dict(key=f"hf:{d['id']}", name=card.get("pretty_name") or d["id"].split("/")[-1], origin="huggingface",
                        url=f"https://huggingface.co/datasets/{d['id']}", date=str(d.get("lastModified", ""))[:10],
                        summary=", ".join(t for t in (d.get("tags") or []) if ":" in t)[:300]))
    return out


def search_arxiv(keyword: str, limit: int = 50) -> list[dict]:
    q = " AND ".join(f'all:"{t}"' for t in [keyword])
    r = requests.get(ARXIV_API, params=dict(search_query=q, sortBy="submittedDate", sortOrder="descending",
                                            max_results=limit), timeout=60)
    r.raise_for_status()
    out = []
    for e in ET.fromstring(r.text).findall("a:entry", _ATOM):
        url = (e.findtext("a:id", "", _ATOM) or "").strip()
        aid = re.sub(r"v\d+$", "", url.rsplit("/", 1)[-1])
        title = " ".join((e.findtext("a:title", "", _ATOM) or "").split())
        out.append(dict(key=f"arxiv:{aid}", name=title.split(":")[0][:80], origin="arxiv", url=url,
                        date=(e.findtext("a:published", "", _ATOM) or "")[:10],
                        summary=" ".join((e.findtext("a:summary", "", _ATOM) or "").split())[:300]))
    return out


def approved(ctx) -> list[dict]:
    return [r for r in read_csv(ctx.path("table", name="candidates.csv")) if r.get("status", "").strip().lower() == "approved"]


def mark(ctx, key: str, status: str, note: str | None = None) -> None:
    upsert(ctx.path("table", name="candidates.csv"), dict(key=key, status=status, note=note), key="key", cols=COLS)


def run(ctx, keywords: list[str] | None = None, days: int = 30, use_labeler: bool = False, limit: int = 100) -> int:
    cfg = ctx.get("watch") or {}
    keywords = keywords or cfg.get("keywords") or KEYWORDS
    since = (datetime.now(timezone.utc) - timedelta(days=days or cfg.get("days", 30))).strftime("%Y-%m-%d")
    path = ctx.path("table", name="candidates.csv")
    known = {r["key"]: r for r in read_csv(path)}
    onboarded = {r["name"].lower() for r in read_csv(ctx.path("table", name="registry.csv"))}
    found: dict[str, dict] = {}
    for kw in keywords:
        for fn in (search_hub, search_arxiv):
            try:
                for c in fn(kw, limit):
                    if c["date"] and c["date"] < since:
                        continue
                    found.setdefault(c["key"], c)
            except Exception as e:  # noqa: BLE001
                log(f"{fn.__name__}('{kw}'): {type(e).__name__}: {str(e)[:120]}")
            time.sleep(3)                                   # arXiv asks for three seconds between requests
    new = [c for k, c in found.items() if k not in known]
    labeler = ctx.models["labeler"] if use_labeler and "labeler" in ctx.models else None
    for c in sorted(new, key=lambda c: c["date"], reverse=True):
        c.update(status="onboarded" if c["name"].lower() in onboarded else "new", suggestion="", note="",
                 discovered=time.strftime("%Y-%m-%d"))
        if labeler is not None and c["status"] == "new":
            try:
                c["suggestion"] = labeler.generate(LABEL_PROMPT.format(**c), max_tokens=60).text.strip().splitlines()[0][:160]
            except Exception as e:  # noqa: BLE001
                c["suggestion"] = f"labeler error: {type(e).__name__}"
        upsert(path, c, key="key", cols=COLS)
    rows = read_csv(path)
    n = lambda s: sum(1 for r in rows if r.get("status") == s)
    log(f"{len(found)} candidates since {since}, {len(new)} new -> {path}")
    print(f"candidates: {len(rows)} (new {n('new')}, approved {n('approved')}, rejected {n('rejected')}, onboarded {n('onboarded')})")
    for c in sorted(new, key=lambda c: c["date"], reverse=True)[:20]:
        print(f"  {c['date']}  {c['origin']:<11} {c['name'][:50]:<50} {c['url']}")
    if new:
        print("Set status to approved in the file for the ones to onboard; then run vi-onboard add for each.")
    return 0
