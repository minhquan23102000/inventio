"""Show that ingest and query with categories and fact links run on this machine alone.

    python benchmarks/local_proof.py <directory> "<question>"

dispositio judges the categories and links and reranks; the map is a fresh file in a temp
directory. Before anything runs, the TypeSafe key is removed, Hugging Face is put offline
(dispositio must already be in the local cache) and every socket connection to a non-loopback
address raises. At the end the script asserts that no such connection was attempted and that
the TypeSafe SDK was never imported. The transcript is what docs/design/evidence/local-proof.txt
records.
"""

import ipaddress
import os
import socket
import sys
import tempfile
from pathlib import Path

os.environ.pop("TYPESAFE_API_KEY", None)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
attempts = []
_connect = socket.socket.connect


def guarded(self, address):
    host = address[0] if isinstance(address, tuple) else address
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = host == "localhost"
    if not local:
        attempts.append(address)
        raise ConnectionRefusedError(f"local proof: blocked connection to {address}")
    return _connect(self, address)


socket.socket.connect = guarded
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inventio import cli  # noqa: E402


def main() -> int:
    root, question = sys.argv[1], sys.argv[2]
    db = str(Path(tempfile.mkdtemp()) / "map.db")
    print(f"$ inventio --db {db} init {root} --name local --facts --judge dispositio", flush=True)
    assert cli.main(["--db", db, "init", root, "--name", "local", "--facts", "--judge", "dispositio"]) == 0
    print(f'\n$ inventio --db {db} query "{question}" --facts --judge dispositio --ranker dispositio -k 5', flush=True)
    assert cli.main(["--db", db, "query", question, "--facts", "--judge", "dispositio", "--ranker", "dispositio",
                     "-k", "5"]) == 0
    print(f"\nnon-loopback connections attempted: {len(attempts)}")
    print(f"typesafe_sdk imported: {'typesafe_sdk' in sys.modules}")
    assert not attempts and "typesafe_sdk" not in sys.modules
    return 0


if __name__ == "__main__":
    sys.exit(main())
