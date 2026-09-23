"""Fine-tune Laya on Jev's judgments, on this machine's GPU.

    python benchmarks/finetune_laya.py --time-steps 40      # measure, print the projected run time, exit
    python benchmarks/finetune_laya.py                      # the overnight run
    set INVENTIO_LAYA_MODEL=%LOCALAPPDATA%\\inventio\\laya-tuned   # then Inventio uses the tuned Laya

Teacher data: every judgment Jev left in the maps (the default map and the BEIR benchmark maps):
content categories of chunks and of queries, same-thing judgments between chunks (facts.py) and
relevance of a passage to a query (rankers.py). Each becomes one training item in the form Laya
reads at inference, with Jev's p as a soft target.

Kept out of training, so a later benchmark of the tuned Laya is not measured on what it learned:
- every test query of the three benchmarks (.omp bench-md.jsonl, SciFact test, StackOverflow QA
  test), as a relevance query and as a query-category passage;
- every chunk that is a gold answer of one of those queries, on either side of a pair;
- a fixed 10% of all other chunks (sha1 of the passage), written to <out>/holdout.jsonl so the
  tuned Laya can later be scored against Jev on judgments it never saw.

The loop is the author's (notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb in
NandhaKishorM/laya): policy gradient on a proper scoring rule plus soft cross-entropy, then one
temperature per question type fitted on 400 held-out items. Ported to one GPU with bf16; the
token-embedding matrix (197M of the 322M parameters) is frozen so the optimiser fits in 8 GB.
"""

import argparse
import hashlib
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import data_dir  # noqa: E402
from inventio.facts import category_question, passage as passage_of, query_category_question, same_question  # noqa: E402
from inventio.rankers import CRITERIA, INSTRUCTIONS  # noqa: E402
from inventio.store import default_db  # noqa: E402

HERE = Path(__file__).resolve().parent
BEIR = ("scifact", "coir-stackoverflow-qa")
HOLDOUT = 0.10


def sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------ what is excluded

def omp_tests(con) -> tuple[set[str], set[str]]:
    """The .omp bench questions and the passages of the chunks that answer them."""
    queries, gold = set(), set()
    for line in (HERE.parent / "docs" / "design" / "evidence" / "bench-md.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        queries.add(g["question"])
        for r in con.execute(
            "SELECT f.path, c.heading_path, c.text FROM chunks c JOIN files f ON f.id = c.file_id "
            "JOIN sources s ON s.id = f.source_id WHERE s.name = ? AND f.path = ? AND c.start_line <= ? AND c.end_line >= ?",
            (g["source"], g["path"], g["end_line"], g["start_line"]),
        ):
            gold.add(passage_of(r["path"], r["heading_path"], r["text"]))
    return queries, gold


def beir_tests(ds: Path, con) -> tuple[set[str], set[str]]:
    qtext = {}
    for line in (ds / "queries.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        qtext[r["_id"]] = r["text"]
    queries, docs = set(), set()
    for i, line in enumerate((ds / "qrels" / "test.tsv").open(encoding="utf-8")):
        if i == 0:
            continue
        q, d, s = line.rstrip("\n").split("\t")
        queries.add(qtext[q])
        docs.add(f"docs/{d.replace('/', '_')}.md")
    gold = set()
    for r in con.execute("SELECT f.path, c.heading_path, c.text FROM chunks c JOIN files f ON f.id = c.file_id"):
        if r["path"] in docs:
            gold.add(passage_of(r["path"], r["heading_path"], r["text"]))
    return queries, gold


# ------------------------------------------------------------------------------ teacher data

def collect(maps: list[tuple[str, Path]]) -> tuple[list[dict], dict]:
    rows, test_q, test_p, report = [], set(), set(), {}
    for name, path in maps:
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        q, p = omp_tests(con) if name == "omp" else beir_tests(path.parent, con)
        test_q |= q
        test_p |= p
        n0 = len(rows)
        for r in con.execute("SELECT kind, question, passage, other, p, model, source FROM judgments"):
            rows.append(dict(r))
        for r in con.execute("SELECT query, passage, noul, model, source FROM labels"):
            rows.append({"kind": "relevance", "question": "rel", "passage": r["passage"], "other": r["query"],
                         "p": r["noul"], "model": r["model"], "source": r["source"]})
        report[name] = {"judgments": len(rows) - n0, "test_queries": len(q), "gold_passages": len(p)}
        con.close()
    return rows, {"test_queries": test_q, "test_passages": test_p, "per_map": report}


def split(rows: list[dict], tests: dict) -> tuple[list[dict], list[dict], dict]:
    tq, tp = tests["test_queries"], tests["test_passages"]
    held = lambda s: int(sha(s)[:8], 16) / 0xFFFFFFFF < HOLDOUT
    train, holdout, dropped = [], [], {"test_query": 0, "gold_chunk": 0}
    for r in rows:
        if r["kind"] == "relevance" and r["other"] in tq or r["kind"] == "query_category" and r["passage"] in tq:
            dropped["test_query"] += 1
            continue
        chunks = [r["passage"]] + ([r["other"]] if r["kind"] == "same_thing" else [])
        if r["kind"] != "query_category" and any(c in tp for c in chunks):
            dropped["gold_chunk"] += 1
            continue
        (holdout if r["kind"] != "query_category" and any(held(c) for c in chunks) else train).append(r)
    return train, holdout, dropped


def question(r: dict) -> tuple[dict, dict]:
    """The (state, question) Laya reads for a judgment, identical to inference (facts.py, rankers.py)."""
    k = r["kind"]
    if k == "category":
        return {"passage": r["passage"]}, category_question(r["question"])
    if k == "query_category":
        return {"query": r["passage"]}, query_category_question(r["question"])
    if k == "same_thing":
        return {"passage": r["passage"], "n0": r["other"]}, same_question("n0")
    return {"query": r["other"], "passage": r["passage"]}, {"instructions": INSTRUCTIONS, "criteria": CRITERIA}


def sample(rows: list[dict], n: int, seed: int = 20260923) -> list[dict]:
    """At most n items, spread evenly over (kind, Jev said true / false) so the rare kinds and the
    rare positives are not drowned by eight negative category questions per chunk."""
    rng = random.Random(seed)
    buckets: dict[tuple, list] = {}
    for r in rows:
        buckets.setdefault((r["kind"], r["p"] >= 0.5), []).append(r)
    out, left = [], n
    for i, (b, items) in enumerate(sorted(buckets.items(), key=lambda kv: len(kv[1]))):
        take = min(len(items), left // (len(buckets) - i))
        out += rng.sample(items, take)
        left -= take
    rng.shuffle(out)
    return out


# ------------------------------------------------------------------------------ training

def load_base():
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config

    root = snapshot_download("convaiinnovations/laya")
    d = os.path.join(root, "multilingual")
    _fix_tokenizer_config(d)
    return d


def tokenize(rows, tok, cfg):
    from laya.common import QTYPES, build_sequence, render_options

    items = []
    for r in rows:
        state, q = question(r)
        crit = q["criteria"]
        k = len(render_options({"t": "noul", "crit": crit}))
        seq, markers = build_sequence(tok, state, {"t": "noul", "ins": q["instructions"], "crit": crit},
                                      cfg["max_len"], cfg["head_max_len"])
        if len(markers) != k:
            continue
        p = min(max(float(r["p"]), 0.0), 1.0)
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES["noul"], "target": [1 - p, p],
                      "label": int(p >= 0.5)})
    return items


def collate(items, pad_id):
    import torch

    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"])
    return ids, att, mpos, mmask, target, torch.tensor([it["qtype"] for it in items])


def fit_temp(sel):
    import torch

    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def train(items, base_dir, out_dir: Path, *, epochs, micro, accum, time_steps=0):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from laya.common import build_model, proper_reward

    cfg = json.load(open(os.path.join(base_dir, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(base_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(base_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(base_dir, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    for p in model.encoder.embeddings.tok_embeddings.parameters():
        p.requires_grad = False
    dev = torch.device("cuda")
    model.to(dev).train()

    rng = random.Random(20260922)
    order = list(range(len(items)))
    rng.shuffle(order)
    n_cal = min(400, len(items) // 10)
    calib = [items[i] for i in sorted(order[:n_cal])]
    train_items = [items[i] for i in sorted(order[n_cal:])]
    # batches of similar length waste less padding; shuffled again per epoch below
    train_items.sort(key=lambda it: len(it["ids"]))

    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc, "lr": 2.5e-5}, {"params": head, "lr": 1.0e-4}], weight_decay=0.01)
    batches = [train_items[i:i + micro] for i in range(0, len(train_items), micro)]
    total = max(1, len(batches) // accum * epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-6)
    G, S0, S1 = 4, 0.4, 0.1
    t0, step = time.time(), 0
    print(f"{len(train_items)} train items, {len(calib)} calibration, {len(batches)} batches x {epochs} epochs", flush=True)
    for epoch in range(epochs):
        random.Random(42 + epoch).shuffle(batches)
        sigma = S0 + (S1 - S0) * epoch / max(1, epochs - 1)
        opt.zero_grad(set_to_none=True)
        run_loss = 0.0
        for b, chunk in enumerate(batches, 1):
            ids, att, mpos, mmask, target, qtype = (x.to(dev) for x in collate(chunk, tok.pad_token_id))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, act = model(ids, att, mpos, mmask, qtype)
            logits = logits.float()
            k = mmask.sum(-1, keepdim=True).float()
            eps = torch.randn((G,) + logits.shape, device=dev) * sigma * mmask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mmask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mmask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qtype, mmask, w_sph=0.75, w_rps=1.0)
                adv = (r - r.mean(0, keepdim=True)) / ((r - r.mean(0, keepdim=True)).std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mmask).sum(-1) / (2 * sigma ** 2)
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mmask, -1e4), -1)).sum(-1).mean()
            loss = (-(adv * logp).mean() + loss_ce) / accum + 0.0 * act.sum()
            loss.backward()
            run_loss += loss.item() * accum
            if b % accum == 0 or b == len(batches):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if time_steps and b == time_steps:
                torch.cuda.synchronize()
                per = (time.time() - t0) / b
                print(json.dumps({"sec_per_batch": round(per, 3), "micro_batch": micro, "batches_per_epoch": len(batches),
                                  "epochs": epochs, "projected_hours": round(per * len(batches) * epochs / 3600, 2),
                                  "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}), flush=True)
                return None
            if b % 200 == 0:
                el = time.time() - t0
                done = epoch * len(batches) + b
                print(f"epoch {epoch + 1}/{epochs} batch {b}/{len(batches)} loss {run_loss / 200:.4f} "
                      f"{el / 60:.0f} min, ~{el / done * (len(batches) * epochs - done) / 60:.0f} min left", flush=True)
                run_loss = 0.0
        save(model, tok, cfg, out_dir / "checkpoint_latest", None)
        print(f"epoch {epoch + 1} done in {(time.time() - t0) / 60:.0f} min; checkpoint saved", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(calib), 16):
            ch = calib[i:i + 16]
            ids, att, mpos, mmask, target, qtype = (x.to(dev) for x in collate(ch, tok.pad_token_id))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _ = model(ids, att, mpos, mmask, qtype)
            lg = lg.float().cpu().numpy()
            for j, it in enumerate(ch):
                preds.append((lg[j, : len(it["markers"])], it["target"]))
    temps = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
    temps[2] = fit_temp(preds)  # every item here is a noul question
    save(model, tok, cfg, out_dir, temps)
    return {"minutes": round((time.time() - t0) / 60, 1), "noul_temperature": temps[2], "updates": step}


def save(model, tok, cfg, d: Path, temps):
    from safetensors.torch import save_file

    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}, str(d / "model.safetensors"))
    model.encoder.config.save_pretrained(str(d / "encoder"))
    tok.save_pretrained(str(d / "tokenizer"))
    c = dict(cfg, fine_tuned=True, model_name="laya-inventio-jev")
    if temps is not None:
        c["temperature"] = temps
        c.pop("temperature_by_options", None)
    (d / "rl_agent_config.json").write_text(json.dumps(c, indent=2))


def main() -> int:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "inventio" / "laya-tuned"))
    ap.add_argument("--max-items", type=int, default=60_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8, help="micro-batches per optimiser step (effective batch 64)")
    ap.add_argument("--time-steps", type=int, default=0, help="time this many micro-batches, print the projection, exit")
    args = ap.parse_args()

    maps = [("omp", default_db())] + [(n, data_dir(None) / "beir" / n / "inventio.db") for n in BEIR]
    maps = [(n, p) for n, p in maps if p.exists()]
    rows, tests = collect(maps)
    train_rows, holdout, dropped = split(rows, tests)
    picked = sample(train_rows, args.max_items)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "holdout.jsonl").open("w", encoding="utf-8") as f:
        for r in holdout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = {}
    for r in picked:
        kinds[f"{r['kind']}:{'true' if r['p'] >= 0.5 else 'false'}"] = kinds.get(f"{r['kind']}:{'true' if r['p'] >= 0.5 else 'false'}", 0) + 1
    meta = {"maps": tests["per_map"], "judgments": len(rows), "dropped": dropped, "holdout": len(holdout),
            "trainable": len(train_rows), "picked": len(picked), "picked_by_kind": kinds,
            "epochs": args.epochs, "micro_batch": args.micro_batch, "accum": args.accum}
    print(json.dumps(meta, indent=1), flush=True)

    base = load_base()
    cfg = json.load(open(os.path.join(base, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(base, "tokenizer"))
    items = tokenize(picked, tok, cfg)
    res = train(items, base, out, epochs=args.epochs, micro=args.micro_batch, accum=args.accum,
                time_steps=args.time_steps)
    if res is not None:
        meta["result"] = res
        (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(res), flush=True)
        print(f"tuned Laya in {out}; set INVENTIO_LAYA_MODEL={out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
