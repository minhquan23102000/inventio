"""Fine-tune Laya as Inventio's local ranker, on this machine's GPU, from human labels only.

    python benchmarks/finetune_laya.py --time-steps 40      # measure, print the projected run time, exit
    python benchmarks/finetune_laya.py                      # the full run
    set INVENTIO_LAYA_MODEL=%LOCALAPPDATA%\\inventio\\laya-tuned   # then Inventio uses the tuned Laya

Laya learns the one question the ranker asks (rankers.INSTRUCTIONS / CRITERIA: does the passage
answer the query?), in the exact form it reads at query time: the query and a passage rendered as
`[path > heading]` + text. No label comes from a model. Two kinds of item:

- relevance: the train split of three public benchmarks with human-annotated answers, SciFact
  (English science), StackOverflow QA (English, code) and Zalo legal (Vietnamese law). Each train
  query gives its answer chunks as positives and four of BM25's 30 candidates that are not
  answers as hard negatives (two from ranks 1-10, two from 11-30), the candidates the ranker
  actually has to separate.
- title: a document's own title as the query and its first chunk, heading removed, as the
  positive; BM25's candidates for that title from other documents as negatives, with their
  headings removed too, so a missing heading is not a cue. Titles are kept only when they have at
  least four words and occur once in the corpus. SciFact paper titles and Zalo article titles.

Targets are 0.95 / 0.05, not 1 / 0: qrels are incomplete and a BM25 negative can be an unmarked
answer. On half of the groups the path is dropped from the passage (`[heading]` + text), so the
model cannot lean on file names or document ids.

Kept out of training: every test query of the three benchmarks, and every document that answers
one (no title item is built from it). 10% of the train queries and of the titles (sha1) are held
out; for them all 30 BM25 candidates are scored, in their natural proportion of answers, and split
in two: one half fits the temperature, the other is the report (AUC, Brier, and nDCG@10 of the 30
candidates reordered, against BM25's own order), for the published Laya and the tuned one.

The loop is the one in the Laya author's fine-tuning notebook
(notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb in NandhaKishorM/laya): policy
gradient on a proper scoring rule plus soft cross-entropy. Ported to one GPU with bf16, with a 5%
linear warm-up before the cosine decay; the token-embedding matrix (197M of the 322M parameters)
is frozen so the optimiser fits in 8 GB, which also keeps every language's token vectors as the
base model learned them.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from beir_bench import load_beir, safe_name  # noqa: E402
from data import data_dir  # noqa: E402
from inventio.rankers import CRITERIA, INSTRUCTIONS  # noqa: E402
from inventio.search import bm25  # noqa: E402
from inventio.store import connect  # noqa: E402

DATASETS = ("scifact", "coir-stackoverflow-qa", "zalo-legal")
TITLES = ("scifact", "zalo-legal")  # corpora whose documents carry a real title
HOLDOUT = 0.10
POOL = 30
POS, NEG = 0.95, 0.05
SEED = 20260923


def sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def held(key: str) -> bool:
    return int(sha(key)[:8], 16) / 0xFFFFFFFF < HOLDOUT


def render(h, *, heading: bool, path: bool) -> str:
    """Hit.passage(), optionally without its heading or its path."""
    head = " > ".join(x for x in (h.path if path else "", h.heading_path if heading else "") if x)
    return f"[{head}]\n{h.text}" if head else h.text


def doc_of(h) -> str:
    return Path(h.path).stem


# ------------------------------------------------------------------------------ groups

def relevance_groups(name: str, ds: Path, con, rng):
    """One group per train query: (held out?, a function building the group). BM25 runs only
    when the group is built, so a source whose share is full skips its remaining queries cheaply."""
    _, train_q, train_rel = load_beir(ds, "train")
    _, test_q, _ = load_beir(ds, "test")
    qids = [q for q in train_rel if q in train_q and q not in test_q]
    rng.shuffle(qids)
    for qid in qids:
        q, gold = train_q[qid], {safe_name(d) for d, s in train_rel[qid].items() if s > 0}

        def make(q=q, gold=gold):
            hits = bm25(con, q, POOL, [name])
            got = {doc_of(h) for h in hits}
            extra = [c for c in (gold_chunk(con, name, d) for d in sorted(gold - got)[:1]) if c]
            return {"src": f"{name}:relevance", "query": q, "heading": True, "hits": hits,
                    "gold": [i for i, h in enumerate(hits) if doc_of(h) in gold], "extra": extra}

        yield held(f"{name}\t{q}"), make


def title_groups(name: str, ds: Path, con, rng):
    """One group per unique title: the title as query, its document's first chunk as the answer."""
    _, _, test_rel = load_beir(ds, "test")
    banned = {safe_name(d) for rel in test_rel.values() for d in rel}
    rows = con.execute(
        "SELECT f.path, c.heading_path, c.text, min(c.start_line) FROM chunks c JOIN files f ON f.id = c.file_id "
        "JOIN sources s ON s.id = f.source_id WHERE s.name = ? GROUP BY c.file_id", (name,)).fetchall()
    count: dict[str, int] = {}
    for r in rows:
        count[r["heading_path"]] = count.get(r["heading_path"], 0) + 1
    rows = [r for r in rows if count[r["heading_path"]] == 1 and Path(r["path"]).stem not in banned]
    rng.shuffle(rows)
    for r in rows:
        title = r["heading_path"].split(" > ")[-1]
        q = title.split(". ", 1)[1] if title.startswith("Điều ") and ". " in title else title  # "Điều 5. ..."
        if len(q.split()) < 4:
            continue
        own = SimpleNamespace(path=r["path"], heading_path=r["heading_path"], text=r["text"])

        def make(q=q, own=own):
            hits = [h for h in bm25(con, q, POOL + 5, [name]) if h.path != own.path][:POOL - 1]
            return {"src": f"{name}:title", "query": q, "heading": False, "hits": [own] + hits, "gold": [0], "extra": []}

        yield held(f"{name}\t{r['path']}"), make


def gold_chunk(con, name: str, doc: str):
    """The first chunk of an answer document BM25 did not put among its 30."""
    r = con.execute(
        "SELECT f.path, c.heading_path, c.text FROM chunks c JOIN files f ON f.id = c.file_id "
        "JOIN sources s ON s.id = f.source_id WHERE s.name = ? AND f.path = ? ORDER BY c.start_line LIMIT 1",
        (name, f"docs/{doc}.md")).fetchone()
    return SimpleNamespace(path=r["path"], heading_path=r["heading_path"], text=r["text"]) if r else None


def train_items(g, rng) -> list[dict]:
    """Answers and four hard negatives of a group, in one rendering (path dropped half the time)."""
    path = rng.random() < 0.5
    rend = lambda h: render(h, heading=g["heading"], path=path)  # noqa: E731
    pos = [g["hits"][i] for i in g["gold"][:2]] or g["extra"]
    neg_idx = [i for i in range(len(g["hits"])) if i not in g["gold"]]
    top, rest = [i for i in neg_idx if i < 10], [i for i in neg_idx if i >= 10]
    negs = rng.sample(top, min(2, len(top))) + rng.sample(rest, min(2, len(rest)))
    return ([{"src": g["src"], "query": g["query"], "passage": rend(h), "p": POS} for h in pos]
            + [{"src": g["src"], "query": g["query"], "passage": rend(g["hits"][i]), "p": NEG} for i in negs])


def eval_items(g) -> list[dict]:
    """All candidates of a held-out group, as the ranker sees them at query time (path kept)."""
    gold = set(g["gold"])
    return [{"src": g["src"], "query": g["query"], "passage": render(h, heading=g["heading"], path=True),
             "p": 1.0 if i in gold else 0.0, "rank": i} for i, h in enumerate(g["hits"])]


def build(max_items: int, eval_groups: int, rng) -> tuple[list[dict], list[dict], dict]:
    """Equal shares of the item budget per source; what a source cannot fill passes to the ones
    after it. Held-out groups are scored whole and never trained on."""
    sources = []
    for name in DATASETS:
        ds = data_dir(None) / "beir" / name
        db = ds / "inventio.db"
        if not db.exists():
            print(f"skip {name}: no map at {db} (run benchmarks/beir_bench.py {name} first)", flush=True)
            continue
        con = connect(db)
        sources.append((f"{name}:relevance", relevance_groups(name, ds, con, rng)))
        if name in TITLES:
            sources.append((f"{name}:title", title_groups(name, ds, con, rng)))
    train, held_out, report = [], [], {}
    left = max_items
    for i, (src, groups) in enumerate(sources):
        share, n_train, n_eval, n_groups = left // (len(sources) - i), 0, 0, 0
        for is_held, make in groups:
            if n_train >= share and n_eval >= eval_groups:
                break
            if is_held:
                if n_eval < eval_groups:
                    g = make()
                    if g["gold"]:  # an answer among the 30, or nothing to reorder
                        held_out += eval_items(g)
                        n_eval += 1
            elif n_train < share:
                items = train_items(make(), rng)
                if any(it["p"] == POS for it in items):
                    train += items
                    n_train += len(items)
                    n_groups += 1
        left -= n_train
        report[src] = {"train_items": n_train, "train_groups": n_groups, "eval_groups": n_eval}
        print(src, report[src], flush=True)
    rng.shuffle(train)
    return train, held_out, report


# ------------------------------------------------------------------------------ training

def load_base():
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config

    root = snapshot_download("convaiinnovations/laya")
    d = os.path.join(root, "multilingual")
    _fix_tokenizer_config(d)
    return d


def tokenize(rows, tok, cfg):
    """Items whose state fits: a truncated passage would carry a label about text Laya never reads."""
    from laya.common import QTYPES, build_sequence, render_options

    crit = CRITERIA
    k = len(render_options({"t": "noul", "crit": crit}))
    items, cut = [], 0
    for r in rows:
        seq, markers = build_sequence(tok, {"query": r["query"], "passage": r["passage"]},
                                      {"t": "noul", "ins": INSTRUCTIONS, "crit": crit}, cfg["max_len"], cfg["head_max_len"])
        if len(markers) != k or len(seq) >= cfg["max_len"]:
            cut += 1
            continue
        p = float(r["p"])
        items.append({**r, "ids": seq, "markers": markers, "qtype": QTYPES["noul"], "target": [1 - p, p]})
    return items, cut


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


def logits_of(model, items, tok, dev) -> list:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(items), 16):
            ch = items[i:i + 16]
            ids, att, mpos, mmask, _, qtype = (x.to(dev) for x in collate(ch, tok.pad_token_id))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _ = model(ids, att, mpos, mmask, qtype)
            lg = lg.float().cpu()
            out += [lg[j, :2].tolist() for j in range(len(ch))]
    return out


def fit_temp(logits, items) -> float:
    import torch

    if len(items) < 10:
        return 1.0
    Z = torch.tensor(logits)
    T = torch.tensor([it["target"] for it in items], dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def report(logits, items, temp: float) -> dict:
    """Per source: AUC and Brier on the natural mix, nDCG@10 of each group's candidates reordered."""
    out = {}
    for src in sorted({it["src"] for it in items}):
        sel = [(lg, it) for lg, it in zip(logits, items) if it["src"] == src]
        p = [1 / (1 + math.exp(-(lg[1] - lg[0]) / temp)) for lg, _ in sel]
        y = [it["p"] for _, it in sel]
        pos = [a for a, b in zip(p, y) if b]
        neg = [a for a, b in zip(p, y) if not b]
        auc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / max(1, len(pos) * len(neg))
        groups: dict[str, list] = {}
        for pi, (_, it) in zip(p, sel):
            groups.setdefault(it["query"], []).append((pi, it["rank"], it["p"]))

        def ndcg(order):
            dcg = sum(g / math.log2(i + 2) for i, (_, _, g) in enumerate(order[:10]))
            ideal = sum(1 / math.log2(i + 2) for i in range(min(10, int(sum(g for *_, g in order)))))
            return dcg / ideal if ideal else 0.0

        out[src] = {"items": len(sel), "positives": len(pos), "auc": round(auc, 4),
                    "brier": round(sum((a - b) ** 2 for a, b in zip(p, y)) / len(y), 4),
                    "ndcg10_ranked": round(sum(ndcg(sorted(g, key=lambda x: -x[0])) for g in groups.values()) / len(groups), 4),
                    "ndcg10_bm25": round(sum(ndcg(sorted(g, key=lambda x: x[1])) for g in groups.values()) / len(groups), 4)}
    return out


def train(items, calib, test, base_dir, out_dir: Path, *, epochs, micro, accum, time_steps=0):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from laya.common import build_model, proper_reward

    cfg = json.load(open(os.path.join(base_dir, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(base_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(base_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(base_dir, "model.safetensors")), strict=True)
    dev = torch.device("cuda")
    model.to(dev)
    base_temp = float(cfg.get("temperature", [1.0, 1.0, 1.0])[2])
    before = None
    if not time_steps:
        before = report(logits_of(model, test, tok, dev), test, base_temp)
        print("published Laya on held-out:", json.dumps(before), flush=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    for p in model.encoder.embeddings.tok_embeddings.parameters():
        p.requires_grad = False
    model.train()

    # batches of similar length waste less padding; shuffled again per epoch below
    items = sorted(items, key=lambda it: len(it["ids"]))
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc, "lr": 2.5e-5}, {"params": head, "lr": 1.0e-4}], weight_decay=0.01)
    batches = [items[i:i + micro] for i in range(0, len(items), micro)]
    total = max(1, math.ceil(len(batches) / accum) * epochs)
    warm = max(1, total // 20)
    floor = 1e-6 / 2.5e-5
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    G, S0, S1 = 4, 0.4, 0.1
    t0, step = time.time(), 0
    print(f"{len(items)} train items, {len(calib)} calibration, {len(test)} report, "
          f"{len(batches)} batches x {epochs} epochs, {total} updates ({warm} warm-up)", flush=True)
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

    temp = fit_temp(logits_of(model, calib, tok, dev), calib)
    temps = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
    temps[2] = temp
    save(model, tok, cfg, out_dir, temps)
    after = report(logits_of(model, test, tok, dev), test, temp)
    print("tuned Laya on held-out:", json.dumps(after), flush=True)
    return {"minutes": round((time.time() - t0) / 60, 1), "noul_temperature": temp, "updates": step,
            "published": before, "tuned": after}


def save(model, tok, cfg, d: Path, temps):
    from safetensors.torch import save_file

    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}, str(d / "model.safetensors"))
    model.encoder.config.save_pretrained(str(d / "encoder"))
    tok.save_pretrained(str(d / "tokenizer"))
    c = dict(cfg, fine_tuned=True, model_name="laya-inventio-relevance")
    if temps is not None:
        c["temperature"] = temps
        c.pop("temperature_by_options", None)
    (d / "rl_agent_config.json").write_text(json.dumps(c, indent=2))


def main() -> int:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "inventio" / "laya-tuned"))
    ap.add_argument("--max-items", type=int, default=80_000)
    ap.add_argument("--eval-groups", type=int, default=60, help="held-out groups scored per source (30 candidates each)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8, help="micro-batches per optimiser step (effective batch 64)")
    ap.add_argument("--time-steps", type=int, default=0, help="time this many micro-batches, print the projection, exit")
    args = ap.parse_args()

    rng = random.Random(SEED)
    rows, held_out, per_source = build(args.max_items, args.eval_groups, rng)
    base = load_base()
    cfg = json.load(open(os.path.join(base, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(base, "tokenizer"))
    items, cut = tokenize(rows, tok, cfg)
    evals, cut_eval = tokenize(held_out, tok, cfg)
    by_query = sorted({it["query"] for it in evals})
    cal_q = set(by_query[::2])  # half the held-out queries fit the temperature, the other half is the report
    calib = [it for it in evals if it["query"] in cal_q]
    test = [it for it in evals if it["query"] not in cal_q]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "holdout.jsonl").open("w", encoding="utf-8") as f:
        for it in evals:
            f.write(json.dumps({k: it[k] for k in ("src", "query", "passage", "p", "rank")}, ensure_ascii=False) + "\n")
    meta = {"per_source": per_source, "train_items": len(items), "cut_truncated": cut, "held_out_items": len(evals),
            "cut_truncated_held_out": cut_eval, "epochs": args.epochs, "micro_batch": args.micro_batch, "accum": args.accum}
    print(json.dumps(meta, indent=1), flush=True)
    res = train(items, calib, test, base, out, epochs=args.epochs, micro=args.micro_batch, accum=args.accum,
                time_steps=args.time_steps)
    if res is not None:
        meta["result"] = res
        (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
        print(f"tuned Laya in {out}; set INVENTIO_LAYA_MODEL={out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
