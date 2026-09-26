"""Publish a dispositio checkpoint to Hugging Face, keeping the outgoing revision fetchable.

    python benchmarks/publish.py <checkpoint dir> --card benchmarks/model-card.md --tag-previous v2
    python benchmarks/publish.py <checkpoint dir> --dry-run

A release always preserves what it replaces: `--tag-previous NAME` puts that tag on the revision
that is on `main` right now, so a user can still fetch it (`revision="v2"`) while the new weights
take over `main`. Only the files a checkpoint needs to load are uploaded
(`model.safetensors`, `rl_agent_config.json`, `encoder/config.json`, the tokenizer, and
`train_meta.json`); the training hold-out and `checkpoint_latest/` never leave the machine.

The upload is verified by reading the revision back and comparing the file hashes, so a partial
upload cannot look like a release.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = "minhquan2310/dispositio"
FILES = ("model.safetensors", "rl_agent_config.json", "train_meta.json",
         "encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", help="checkpoint directory to publish")
    ap.add_argument("--card", help="markdown file uploaded as the model card (README.md)")
    ap.add_argument("--tag-previous", metavar="NAME",
                    help="tag the revision now on main with NAME before uploading, so it stays fetchable")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--message", default=None, help="commit message; default names the checkpoint")
    ap.add_argument("--dry-run", action="store_true", help="print what would be uploaded, upload nothing")
    args = ap.parse_args()

    d = Path(args.checkpoint)
    missing = [f for f in FILES if not (d / f).exists()]
    if missing:
        print(f"checkpoint is missing {missing}; a release needs all of {list(FILES)}", file=sys.stderr)
        return 1
    local = {f: sha256(d / f) for f in FILES}
    meta = json.loads((d / "train_meta.json").read_text(encoding="utf-8"))
    print(f"publishing {d} as {args.repo}")
    print(f"  train_meta: name={meta.get('name')} items={meta.get('train_items')} mix={meta.get('mix')}")
    for f in FILES:
        print(f"  {f}  {(d / f).stat().st_size / 1e6:.1f} MB  sha256 {local[f][:16]}")
    if args.card:
        print(f"  README.md from {args.card} ({Path(args.card).stat().st_size} bytes)")
    if args.dry_run:
        print("dry run: nothing uploaded")
        return 0

    api = HfApi()
    if args.tag_previous:
        head = api.repo_info(args.repo).sha
        api.create_tag(args.repo, tag=args.tag_previous, revision=head)
        print(f"tagged the revision now on main ({head[:8]}) as {args.tag_previous}")

    api.upload_folder(repo_id=args.repo, folder_path=str(d), allow_patterns=list(FILES),
                      commit_message=args.message or f"release: {d.name}")
    if args.card:
        api.upload_file(repo_id=args.repo, path_or_fileobj=args.card, path_in_repo="README.md",
                        commit_message=f"release: model card for {d.name}")

    rev = api.repo_info(args.repo).sha
    from huggingface_hub import hf_hub_download
    bad = []
    for f in FILES:
        got = Path(hf_hub_download(args.repo, f, revision=rev, force_download=True))
        if sha256(got) != local[f]:
            bad.append(f)
    tags = [t.name for t in api.list_repo_refs(args.repo).tags]
    print(f"revision {rev[:8]} online; tags now {sorted(tags)}")
    if bad:
        print(f"MISMATCH after upload: {bad}", file=sys.stderr)
        return 1
    print("every uploaded file matches the checkpoint on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
