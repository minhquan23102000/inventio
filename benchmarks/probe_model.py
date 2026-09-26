"""Does a ranker read the question and the passage, or only match words? Pass/fail probes.

    python benchmarks/probe_model.py <checkpoint dir> [<checkpoint dir> ...]

On pairs no model trained on (the checkpoint's own held-out, and the synthetic items finetune_laya.
synth_held keeps back):

- question: the same 200 (query, passage) pairs, 100 answers and 100 not. A "does it answer?" is
  relevance (AUC). The others are asked in wordings training never saw (finetune_laya.ask_held),
  a different one per pair: B "does it fail to answer?" must move against A (correlation below 0);
  C "is it in Vietnamese?" and E "is it in English?" must follow the language (AUC near 1), not A.
- cut: an answer with the units stating the answer deleted must fall below 0.5 while the whole
  answer stays above it.
- mask: an answer rewritten to share few words with the query must keep its score (mean change),
  and rewritten answers must still rank above rewritten non-answers (AUC).
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import data_dir  # noqa: E402
from finetune_laya import CRITERIA, INSTRUCTIONS, asks, is_vi, synth_held  # noqa: E402
from beir_bench import safe_name  # noqa: E402

A = {"type": "noul", "instructions": INSTRUCTIONS, "criteria": CRITERIA}
UNSEEN = {"B": ("rel", "neg"), "C": ("lang", "pos"), "E": ("lang", "neg")}
PASS = {"A_auc": ">= 0.88", "corr_AB": "< 0", "C_lang_auc": ">= 0.9", "E_english_auc": ">= 0.9", "cut_flipped": ">= 0.7",
        "mask_mean_change": "<= 0.15", "mask_auc": ">= 0.85"}


def auc(s, t) -> float:
    s, t = np.asarray(s, float), np.asarray(t, bool)
    p, n = s[t], s[~t]
    return float((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean())


def probe(agent, holdout: Path) -> dict:
    rows = [json.loads(l) for l in holdout.open(encoding="utf-8") if "relevance" in l]
    rng = random.Random(1)
    pos = [r for r in rows if float(r["p"]) >= 0.5]
    neg = [r for r in rows if float(r["p"]) < 0.5]
    sample = rng.sample(pos, 100) + rng.sample(neg, 100)
    y = np.array([1] * 100 + [0] * 100)
    held = asks(held_out=True)
    P = {k: [] for k in ("A", *UNSEEN)}
    for i, r in enumerate(sample):
        qs = {"A": A} | {k: {"type": "noul", **held[kp][i % len(held[kp])]} for k, kp in UNSEEN.items()}
        ans = agent.predict({"query": r["query"], "passage": r["passage"]}, qs)["answers"]
        for k in qs:
            P[k].append(float(ans[k]["noul"]))
    a, b, c, e = (np.array(P[k]) for k in "ABCE")
    vi = np.array([is_vi(r["passage"]) for r in sample])
    out = {"A_auc": round(auc(a, y), 3), "B_auc": round(auc(b, y), 3), "corr_AB": round(float(np.corrcoef(a, b)[0, 1]), 3),
           "C_lang_auc": round(auc(c, vi), 3), "E_english_auc": round(auc(e, ~vi), 3),
           "corr_AC": round(float(np.corrcoef(a, c)[0, 1]), 3)}

    rel = lambda q, p: float(agent.predict({"query": q, "passage": p}, {"r": A})["answers"]["r"]["noul"])  # noqa: E731
    d = data_dir(None) / "synth"
    flipped, drops = [], []
    for r in map(json.loads, (d / "answer_units.jsonl").open(encoding="utf-8")):
        o = r["out"]
        if not synth_held(r["id"]) or not (o["answerable"] and o["answer_units"]) or o["rest_still_answers"]:
            continue
        head = f"[docs/{safe_name(r['doc'])}.md > {r['title']}]\n" if r.get("title") else f"[docs/{safe_name(r['doc'])}.md]\n"
        rest = "\n".join(u for i, u in enumerate(r["units"]) if i not in set(o["answer_units"]))
        whole, cut = rel(r["query"], head + r["text"]), rel(r["query"], head + rest)
        flipped.append(whole >= 0.5 and cut < 0.5)
        drops.append(whole - cut)
    out |= {"cut_n": len(flipped), "cut_flipped": round(float(np.mean(flipped)), 3), "cut_mean_drop": round(float(np.mean(drops)), 3)}

    change, lab, rew = [], [], []
    for r in map(json.loads, (d / "lexical_mask.jsonl").open(encoding="utf-8")):
        if not synth_held(r["id"]) or not r["out"]["facts_unchanged"]:
            continue
        a, b = rel(r["query"], r["text"]), rel(r["query"], r["out"]["rewrite"])
        change.append(abs(a - b))
        lab.append(r["label"] == "pos")
        rew.append(b)
    out |= {"mask_n": len(change), "mask_mean_change": round(float(np.mean(change)), 3),
            "mask_auc": round(auc(rew, lab), 3) if any(lab) and not all(lab) else None}
    return out


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from inventio.rankers import load_laya

    holdout = Path(sys.argv[1]) / "holdout.jsonl"  # one pair set for every model compared
    for d in sys.argv[1:]:
        os.environ["INVENTIO_DISPOSITIO_MODEL"] = d
        agent, tag = load_laya("dispositio")
        res = probe(agent, holdout)
        ok = {k: bool(eval(f"{res[k]} {rule}")) for k, rule in PASS.items() if res.get(k) is not None}
        print(tag, json.dumps(res))
        print("  passes:", ok, "ALL" if all(ok.values()) else "NOT ALL", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
