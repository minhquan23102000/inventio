"""Train a System One run on Modal from a record file built on this machine, with the recipe
`systemone.py train` pins (recipe.trainer_args), so a cloud run is the laptop's run on a bigger card.

    uvx modal run benchmarks/modal_train.py --records <train.jsonl> --name s1-v1.4 --max-steps 50   # smoke
    uvx modal run benchmarks/modal_train.py --records <train.jsonl> --name s1-v1.4 --epochs 2
    uvx modal run benchmarks/modal_train.py --records <delta.jsonl> --name s1-v1.4 --init-from s1-v1.3 --epochs 1

What is the same as the laptop: Kev's checkout at KEV_DIR (its commit, its locked dependencies from
uv.lock), the trainer flags, and the entry `kev_win.py train` (on Linux it only lifts the choice cap and
repeats the kv heads for SDPA). What changes: the card, and the DeltaNet kernels Kev's own Modal image
installs (flash-linear-attention, causal-conv1d), which the Windows run lacks and which only change speed.

The records go to the volume `inventio-bench` at /train/<name>/train.jsonl, the run is written to
/runs/<name> there (where modal_bench.py reads runs), and it comes back to <KEV_DIR>/runs/<name> with a
recipe.json that names the machine, the wall time and the data's hash. A name already on the volume is
refused: a run is never overwritten. `--init-from <run>` trains a delta from an earlier run: its directory
under <KEV_DIR>/runs goes to /runs/<run> on the volume unless it is there already.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recipe import BASE, BASE_REVISION, INIT_FROM, trainer_args  # noqa: E402

HERE = Path(__file__).resolve().parent
KEV = Path(os.environ.get("KEV_DIR", "C:/Users/LEGION/kev"))
GPU_HOURLY = {"H100": 3.95, "A100-80GB": 2.50, "L40S": 1.95, "A10G": 1.10, "L4": 0.80}   # Modal's list price, USD/h
CAUSAL_CONV1D = ("https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/"
                 "causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_sync(uv_project_dir=str(KEV), groups=[])       # Kev's locked dependencies (Linux torch wheels are CUDA builds)
    # as Kev's own modal_app.py: fla refuses its gated chunk backward on Hopper with Triton < 3.7.1, and
    # causal-conv1d without deps so its torch requirement does not put torch's triton 3.4 back
    .uv_pip_install("flash-linear-attention==0.5.2", "triton>=3.7.1")
    .uv_pip_install(CAUSAL_CONV1D, extra_options="--no-deps")
    .env({"HF_HOME": "/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONUNBUFFERED": "1", "TRITON_CACHE_DIR": "/hf/triton-cache", "PYTHONPATH": "/root/kev_src"})
    .add_local_dir(KEV / "kev", "/root/kev_src/kev", ignore=["**/__pycache__/**"])
    .add_local_file(HERE / "kev_win.py", "/root/kev_win.py")
    .add_local_file(HERE / "recipe.py", "/root/recipe.py")
)
vol = modal.Volume.from_name("inventio-bench")
hf = modal.Volume.from_name("inventio-hf", create_if_missing=True)
app = modal.App("inventio-train", image=image, volumes={"/vol": vol, "/hf": hf})


@app.function(gpu="H100", cpu=4, memory=32768, timeout=3 * 3600)
def train_remote(name: str, args: list[str], recipe: dict) -> dict:
    """Kev's trainer on the uploaded records into /vol/runs/<name>; the log is committed every few minutes
    so a run that dies still leaves what it printed."""
    import torch

    out = Path("/vol/runs") / name
    if out.exists():
        raise FileExistsError(f"/runs/{name} already holds a run; pick another --name")
    log = Path("/vol/train") / name / "train.log"
    t0 = time.time()
    cmd = [sys.executable, "/root/kev_win.py", "train", *args]
    print("$ " + " ".join(cmd), flush=True)
    last = time.time()
    with log.open("w", encoding="utf-8") as f, subprocess.Popen(cmd, cwd="/root", stdout=subprocess.PIPE,
                                                                stderr=subprocess.STDOUT, text=True) as p:
        for line in p.stdout:
            f.write(line)
            if line.startswith(("ep", "saved", "dropped", "device")) or "Error" in line or "Traceback" in line:
                print(line.rstrip(), flush=True)
            if time.time() - last > 300:
                f.flush(); vol.commit(); last = time.time()
    rc = p.returncode
    recipe = {**recipe, "machine": f"modal {torch.cuda.get_device_name(0)}", "wall_seconds": round(time.time() - t0),
              "returncode": rc}
    if out.exists():
        (out / "recipe.json").write_text(json.dumps(recipe, indent=1) + "\n", encoding="utf-8")
    vol.commit(); hf.commit()
    if rc:
        raise RuntimeError(f"kev.train exited {rc}; see /train/{name}/train.log on the volume")
    return recipe


@app.local_entrypoint()
def main(records: str, name: str, epochs: int = 2, lr: float = 2e-5, seed: int = 0, max_state: int = 6656,
         max_steps: int = 0, gpu: str = "H100", hours: float = 3.0, batch: int = 8, init_from: str = ""):
    src = Path(records)
    try:
        taken = {Path(e.path).name for e in vol.listdir("/runs")}
    except modal.exception.NotFoundError:
        taken = set()
    if name in taken:
        raise SystemExit(f"/runs/{name} is already on the volume; runs are never overwritten")
    commit = subprocess.run(["git", "-C", str(KEV), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(KEV), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{KEV} has uncommitted changes; the run would not be the commit it records")
    remote_data = f"/vol/train/{name}/train.jsonl"
    init = f"/vol/runs/{init_from}" if init_from else INIT_FROM
    if init_from and init_from not in taken:
        local = KEV / "runs" / init_from
        if not (local / "head.pt").exists():
            raise SystemExit(f"{local} holds no checkpoint (head.pt) to start from")
        with vol.batch_upload() as b:
            b.put_directory(str(local), f"/runs/{init_from}")
        print(f"uploaded {local} -> /runs/{init_from}", flush=True)
    args = trainer_args(remote_data, f"/vol/runs/{name}", max_state=max_state, lr=lr, epochs=epochs, seed=seed,
                        max_steps=max_steps, batch=batch, init_from=init)
    recipe = {"kev_commit": commit, "init_from": init_from or INIT_FROM, "base": BASE, "base_revision": BASE_REVISION,
              "records": str(src), "records_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
              "max_state": max_state, "epochs": epochs, "lr": lr, "seed": seed, "max_steps": max_steps, "batch": batch,
              "trainer_args": args}
    bound = GPU_HOURLY.get(gpu, 4.0) * hours
    print(f"{name}: {sum(1 for _ in src.open(encoding='utf-8'))} records on {gpu}, "
          f"at most {hours:g} h = ${bound:.2f} at list price", flush=True)
    with vol.batch_upload(force=True) as b:
        b.put_file(str(src), f"/train/{name}/train.jsonl")
    res = train_remote.with_options(gpu=gpu, timeout=int(hours * 3600)).remote(name, args, recipe)
    print(json.dumps({k: res[k] for k in ("machine", "wall_seconds")}), flush=True)
    dest = KEV / "runs" / name
    for e in vol.listdir(f"/runs/{name}", recursive=True):
        if e.type != modal.volume.FileEntryType.FILE:
            continue
        rel = Path(e.path.lstrip("/")).relative_to(f"runs/{name}")
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        with (dest / rel).open("wb") as f:
            for chunk in vol.read_file(e.path):
                f.write(chunk)
    print(f"run -> {dest}  (${GPU_HOURLY.get(gpu, 4.0) * res['wall_seconds'] / 3600:.2f} of GPU time at list price)",
          flush=True)
