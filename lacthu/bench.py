"""Measure a configuration against questions with known answers.

A bench file is JSON Lines, one question per line:
  {"question": "...", "source": "docs", "path": "maps/x/map.md", "start_line": 12, "end_line": 30}
A returned chunk counts as the answer when it comes from the same file and overlaps those lines,
so the gold survives changes to the chunker.
"""

import json
import time
from pathlib import Path

from .search import search


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def gold_rank(hits, g: dict) -> int | None:
    for i, h in enumerate(hits, 1):
        if h.source == g["source"] and h.path == g["path"] and h.start_line <= g["end_line"] and h.end_line >= g["start_line"]:
            return i
    return None


def run(con, rows: list[dict], ranker, *, pool: int = 30, expand_links: bool = False) -> dict:
    ranks, t0 = [], time.time()
    for g in rows:
        hits = search(con, g["question"], k=10_000, pool=pool, ranker=ranker, expand_links=expand_links)
        ranks.append(gold_rank(hits, g))
    n = len(rows)
    at = lambda k: sum(1 for r in ranks if r is not None and r <= k)
    return {
        "n": n,
        "in_pool": at(10_000),
        "top1": at(1),
        "top5": at(5),
        "top10": at(10),
        "sec_per_query": round((time.time() - t0) / max(n, 1), 3),
        "ranks": ranks,
    }
