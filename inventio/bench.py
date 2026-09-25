"""Measure a configuration against questions with known answers.

A bench file is JSON Lines, one question per line:
  {"question": "...", "source": "docs", "path": "maps/x/map.md", "start_line": 12, "end_line": 30}
A returned chunk counts as the answer when it comes from the same file and overlaps those lines,
so the gold survives changes to the chunker.
"""

import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

from .search import bm25, rank_key, search, widen_by_facts


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def gold_rank(hits, g: dict) -> int | None:
    for i, h in enumerate(hits, 1):
        if h.source == g["source"] and h.path == g["path"] and h.start_line <= g["end_line"] and h.end_line >= g["start_line"]:
            return i
    return None


def run(con, rows: list[dict], ranker, *, pool: int = 30, expand_links: bool = False,
        by_type: bool = False, facts=None, symbols: bool = True, neighbours: bool = False, scope=None) -> dict:
    ranks, t0 = [], time.time()
    for g in rows:
        hits = search(con, g["question"], k=10_000, pool=pool, ranker=ranker, expand_links=expand_links,
                      by_type=by_type, facts=facts, symbols=symbols, neighbours=neighbours, scope=scope)
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


# ------------------------------------------------------------------ three arms on the same queries
# base     BM25's pool
# facts    the pool plus what the query's predicted categories and judged `about` links add
# control  BM25's pool grown to the facts pool's size, so a gain from facts is not a gain from
#          handing the ranker more candidates
ARMS = ("base", "facts", "control")


def arm_pools(con, q: str, judge, pool: int = 30, scope=None, facts_limit: int = 10) -> dict:
    base = bm25(con, q, pool, scope)
    facts = base + widen_by_facts(con, q, base, judge, limit=facts_limit, scope=scope)
    return {"base": base, "facts": facts, "control": bm25(con, q, len(facts), scope)}


def rank_arms(pools: dict, scores: dict | None) -> dict:
    """Each arm's pool in ranked order; `scores` maps chunk id to the ranker's p (None: BM25 order,
    where added chunks sit after the BM25 ones, so only recall can move)."""
    out = {}
    for arm, hits in pools.items():
        hs = [replace(h) for h in hits]
        if scores is not None:
            for h in hs:
                h.score = scores.get(h.id)
            hs.sort(key=rank_key)
        out[arm] = hs
    return out


def paired(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 7) -> dict:
    """b against a on the same queries: wins, losses, ties, mean difference, bootstrap 95% CI."""
    d = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return {
        "wins": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d), "ties": sum(x == 0 for x in d),
        "mean_diff": round(sum(d) / len(d), 4),
        "ci95": [round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot) - 1], 4)],
    }


def ndcg_one(rank: int | None) -> float:
    return 1 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.0


def run_arms(con, rows: list[dict], ranker, judge, *, pool: int = 30, facts_limit: int = 10, scope=None) -> dict:
    """The three arms per question; each distinct chunk is scored once per question."""
    ranks = {a: [] for a in ARMS}
    sizes = {a: [] for a in ARMS}
    for g in rows:
        pools = arm_pools(con, g["question"], judge, pool, scope, facts_limit=facts_limit)
        scores = None
        if ranker is not None:
            uniq = list({h.id: h for hs in pools.values() for h in hs}.values())
            scores = dict(zip((h.id for h in uniq), ranker.score(g["question"], uniq)))
        for arm, hs in rank_arms(pools, scores).items():
            ranks[arm].append(gold_rank(hs, g))
            sizes[arm].append(len(hs))
    n = len(rows)
    out = {}
    for arm in ARMS:
        at = lambda k, r=ranks[arm]: sum(1 for x in r if x is not None and x <= k)
        out[arm] = {"n": n, "in_pool": at(10_000), "top1": at(1), "top5": at(5), "top10": at(10),
                    "ndcg@10": round(sum(map(ndcg_one, ranks[arm])) / n, 4),
                    "mean_pool": round(sum(sizes[arm]) / n, 1), "ranks": ranks[arm]}
    nd = {a: [ndcg_one(r) for r in ranks[a]] for a in ARMS}
    out["facts_vs_control"] = paired(nd["control"], nd["facts"])
    out["facts_vs_base"] = paired(nd["base"], nd["facts"])
    return out
