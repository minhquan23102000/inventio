"""The measurement tail of a System One run on Modal: the category judge, the SciFact facts arms and the
SWE-bench Lite type arm, the same scripts on the same public data, spread over cloud machines.

    uvx modal volume put inventio-bench <run dir> /runs/<name>     # once per run (43 MB adapter + head)
    uvx modal run benchmarks/modal_bench.py --run <name> [--only judge,facts,types] [--shard 15]

Data on the volume `inventio-bench` under /bench: categories/, beir/scifact/ (corpus, queries, qrels and the
shared inventio.db the local run copies), swe-lite/lite.jsonl. The SWE repositories are cloned into the image
from the same mirrors the local ones came from. Nothing private is read: these are the public benchmarks.

What changes against the local run, and only this: where it runs. SWE-bench is split into shards of
consecutive issues of one repository (each shard pays one full index, then moves commit to commit as the
local run does), and the type head that shapes the `types` pool is one GPU container every shard asks.
Rows and summaries come back into benchmarks/results/ where the local scripts write them. Times read on a
cloud card are not the laptop's: each merged entry carries `machine`, and latency is quoted from the laptop.
"""

import json
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
BASE, BASE_REV = "Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SWE_REPOS = ("astropy__astropy", "django__django", "matplotlib__matplotlib", "mwaskom__seaborn", "pallets__flask",
             "psf__requests", "pydata__xarray", "pylint-dev__pylint", "pytest-dev__pytest",
             "scikit-learn__scikit-learn", "sphinx-doc__sphinx", "sympy__sympy")
GPU = "H100"


def _fetch_base():
    from huggingface_hub import snapshot_download

    snapshot_download(BASE, revision=BASE_REV)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("torch==2.8.0", "transformers==5.17.0", "peft==0.21.0", "pydantic>=2.9",
                    "flash-linear-attention==0.5.2", "tree-sitter-language-pack==1.20.0", "keyring>=24",
                    "huggingface_hub")
    .run_commands(*[f"git clone -q https://github.com/swe-bench/{r}.git /repos/{r}" for r in SWE_REPOS])
    .env({"HF_HOME": "/hf"})
    .run_function(_fetch_base)
    .env({"PYTHONPATH": "/root/inventio:/root/inventio/benchmarks", "INVENTIO_SERVE": "0",
          "INVENTIO_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(REPO / "inventio", "/root/inventio/inventio", ignore=["**/__pycache__/**"])
    .add_local_dir(REPO / "benchmarks", "/root/inventio/benchmarks", ignore=["results/**", "**/__pycache__/**"])
)
vol = modal.Volume.from_name("inventio-bench")
app = modal.App("inventio-bench", image=image, volumes={"/vol": vol})
WORK = "/root/inventio"


def _env(run: str, data: str) -> dict:
    os.environ.update({"INVENTIO_SYSTEMONE_RUN": f"/vol/runs/{run}", "INVENTIO_SYSTEMONE_DEVICE": "cuda",
                       "INVENTIO_BENCH_DATA": data})
    return dict(os.environ)


def _sh(cmd: list[str], env: dict) -> None:
    import subprocess

    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=WORK, env=env, check=True)


def _read(rel: str) -> str:
    p = Path(WORK) / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


@app.function(gpu=GPU, timeout=3600)
def judge(run: str, tag: str) -> str:
    """systemone.py judge on the category test split; returns results/s1/summary.json as written here."""
    env = _env(run, "/vol/bench")
    _sh(["python", "benchmarks/systemone.py", "judge", tag, "--run", f"/vol/runs/{run}"], env)
    return _read("benchmarks/results/s1/summary.json")


@app.function(gpu=GPU, timeout=4 * 3600, ephemeral_disk=524_288)
def facts(run: str) -> dict:
    """beir_bench.py scifact --arms --judge systemone on a local copy of the SciFact data (the run writes its own
    map copy and a materialized tree beside the corpus, so it must not write into the shared volume)."""
    import shutil

    shutil.copytree("/vol/bench/beir/scifact", "/tmp/bench/beir/scifact")
    env = _env(run, "/tmp/bench")
    _sh(["python", "benchmarks/beir_bench.py", "scifact", "--arms", "--judge", "systemone", "--rankers", "none"], env)
    return {"summary": _read("benchmarks/results/beir-scifact/summary.json"),
            "rows": _read("benchmarks/results/beir-scifact/arms@systemone.jsonl")}


@app.cls(gpu="L4", timeout=3 * 3600, max_containers=2, scaledown_window=300)
class TypeHead:
    """The run's type head, asked by every SWE shard: the choice question over the issue text alone."""
    run: str = modal.parameter()

    @modal.enter()
    def load(self):
        _env(self.run, "/vol/bench")
        from inventio.rankers import SystemOneRanker

        self.ranker = SystemOneRanker()
        self.ranker.types("warm", {"Article": "prose", "SoftwareSourceCode": "code"})

    @modal.method()
    def types(self, query: str, offered: dict) -> dict:
        return self.ranker.types(query, offered)


@app.function(cpu=4, memory=16384, timeout=3 * 3600, ephemeral_disk=524_288)
def swe_shard(run: str, rows: list[dict]) -> str:
    """swe_bench.py --types over these issues (one repository, consecutive), the type head asked remotely."""
    import sys

    data = Path("/tmp/bench/swe-lite")
    data.mkdir(parents=True)
    (data / "lite.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for r in SWE_REPOS:
        (data / r).symlink_to(f"/repos/{r}")
    os.environ["INVENTIO_BENCH_DATA"] = "/tmp/bench"
    os.chdir(WORK)
    import swe_bench

    head = TypeHead(run=run)

    class Remote:
        cloud = False

        def types(self, query, offered):
            return head.types.remote(query, offered)

    local = swe_bench.make_ranker
    swe_bench.make_ranker = lambda name: Remote() if name == "systemone" else local(name)
    sys.argv = ["swe_bench.py", "--types", "--variants", "mixed", "--rankers", "none", "--type-predictor", "systemone"]
    swe_bench.main()
    return _read("benchmarks/results/swe-lite-types/results.jsonl")


def _merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        dst[k] = _merge(dst.get(k, {}), v) if isinstance(v, dict) and isinstance(dst.get(k), dict) else v
    return dst


def _merge_file(path: Path, new: dict, indent=1, sort_keys=True) -> None:
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    path.write_text(json.dumps(_merge(old, new), indent=indent, sort_keys=sort_keys) + "\n", encoding="utf-8")


def _mark(entries: dict, machine: str) -> dict:
    """Every entry this run produced says where it ran (its times are that machine's)."""
    return {k: {**v, "machine": machine} for k, v in entries.items()}


@app.local_entrypoint()
def main(run: str, tag: str = "", only: str = "judge,facts,types", shard: int = 15, merge: bool = True):
    """`--no-merge` (a rehearsal) keeps what came back under results/modal/<tag>/ and touches nothing else."""
    tag, parts, res = tag or run, set(only.split(",")), REPO / "benchmarks" / "results"
    raw = res / "modal" / tag
    raw.mkdir(parents=True, exist_ok=True)
    calls = {}
    if "judge" in parts:
        calls["judge"] = judge.spawn(run, tag)
    if "facts" in parts:
        calls["facts"] = facts.spawn(run)
    shards = []
    if "types" in parts:
        rows = [json.loads(l) for l in (Path(os.environ.get("INVENTIO_BENCH_DATA", Path.home() / "AppData/Local/inventio/bench"))
                                         / "swe-lite" / "lite.jsonl").open(encoding="utf-8")]
        by_repo: dict[str, list] = {}
        for r in rows:
            by_repo.setdefault(r["repo"], []).append(r)
        shards = [rs[i:i + shard] for rs in by_repo.values() for i in range(0, len(rs), shard)]
        print(f"types: {len(rows)} issues in {len(shards)} shards", flush=True)
        calls["types"] = [swe_shard.spawn(run, s) for s in shards]

    if "judge" in calls:
        got = json.loads(calls["judge"].get())
        (raw / "judge.json").write_text(json.dumps(got, indent=1), encoding="utf-8")
        if merge:
            _merge_file(res / "s1" / "summary.json", {"judge": _mark(got["judge"], f"modal {GPU}")})
        print("judge:", json.dumps(got.get("judge", {})), flush=True)
    if "facts" in calls:
        got = calls["facts"].get()
        new = json.loads(got["summary"])
        (raw / "facts.json").write_text(json.dumps(new, indent=1), encoding="utf-8")
        (raw / "arms@systemone.jsonl").write_text(got["rows"], encoding="utf-8")
        if merge:
            (res / "beir-scifact" / "arms@systemone.jsonl").write_text(got["rows"], encoding="utf-8")
            _merge_file(res / "beir-scifact" / "summary.json", {"arms": _mark(new.get("arms", {}), f"modal {GPU}")},
                        indent=2, sort_keys=False)
        print("facts:", json.dumps(new.get("arms", {})), flush=True)
    if "types" in calls:
        out = res / "swe-lite-types" / "results.jsonl"
        keep = [l for l in out.read_text(encoding="utf-8").splitlines()
                if json.loads(l).get("predictor") != "systemone"] if out.exists() else []
        new = [l for h in calls["types"] for l in h.get().splitlines() if l.strip()]
        new = [json.dumps({**json.loads(l), "machine": "modal cpu4 + L4 type head"}) for l in new]
        (raw / "types.jsonl").write_text("".join(l + "\n" for l in new), encoding="utf-8")
        if not merge:
            print(f"types: {len(new)} rows kept in {raw}", flush=True)
            return
        out.write_text("".join(l + "\n" for l in keep + new), encoding="utf-8")
        print(f"types: {len(new)} rows merged into {out}; summary: python benchmarks/swe_bench.py --types "
              f"--variants mixed --rankers none --type-predictor systemone (every row is done, it only sums)", flush=True)
