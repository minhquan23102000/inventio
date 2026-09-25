# Status

Where the work stands, for picking it up on another machine. Last updated 2026-09-25, at
`b75f2b5`.

## Recent changes on `main`

| Commit | Change |
|---|---|
| `b55f28c` | Confluence connector: `_Page.list` shadowed the builtin `list` and crashed on Python below 3.14; renamed `list_items` |
| `06f00fc` | `rankers.py`: load the Hugging Face snapshot with `local_files_only` (skips about 0.44 s of network per query); `MPS_BATCH = 4` on Apple GPUs (3.5 s against 4.9 s for 16 on an M3); results unchanged |
| `cc7662d` | `serve.py`: background server that keeps dispositio loaded, started by the first query, 127.0.0.1 only, token in `serve.json`, exits after `INVENTIO_SERVE_IDLE` (900 s); `INVENTIO_SERVE=0` turns it off; `inventio serve --stop` |
| `727cda6` | Skill: when question and documents are in different languages, translate the whole question and ask again |
| `b75f2b5` | `query --pool` defaults to 15 instead of 30 |

## Measured on a private map (Mac M3, 16 GB)

A wiki space (1,180 pages), a ticket project (1,396 tickets) and two code repositories (1,232
files). 40 questions: 12 written by hand, 28 taken from real links (a ticket's title as the
question, the page or file it links to as the answer). Question set and scripts stay off this
repository because they name internal documents.

| Configuration | Top 1 | Top 10 | MRR@10 | Median per query |
|---|---|---|---|---|
| BM25 | 16/40 | 21/40 | 0.435 | 0.08 s |
| dispositio, pool 30 | 18/40 | 23/40 | 0.483 | 2.70 s |
| dispositio, pool 15 (default now) | 19/40 | 24/40 | 0.508 | 1.47 s |
| dispositio, pool 10 | 16/40 | 20/40 | 0.429 | 0.88 s |

- Served query: first one about 12 s (starts the server), then 1-4 s. Without the server, 9-10 s.
- Shorter reading windows (512 tokens instead of 1024) saved almost nothing: most chunks are
  shorter than 512 tokens. Time follows the number of candidates.
- Tried and slower on MPS: autocast fp16, bf16 (also changes the top 5), CPU. `.half()` crashes.
- Of 322M parameters, 197M are the embedding table; the rest runs at about 1.5 TFLOPS on the M3
  GPU. Core ML on the Neural Engine is the remaining hardware path; not tried.

## Open

- **Public benchmarks not rerun with pool 15.** The README numbers are at `--pool 30`. Risk is
  highest where BM25 often ranks the answer 16-30 (StackOverflow QA, SWE-bench `mixed`).
- **Cross-language questions** are handled only in the skill. A person typing a question in
  one language about documents in another still gets nothing from BM25.
- **Right file, wrong section**: on some questions dispositio picks the section next to the
  answer.
- **No link from a page to code when a table name appears only inside SQL strings**: `mentions`
  links need a definition in the map. Schema cards might fill this; untested.
- **`mirror_dir()` ignores `--db`** and writes into the real data directory, so
  `tests/test_connectors.py::test_where_scopes_every_term_before_bm25` fails from the second run
  on. Undecided: mirrors follow `--db`, or isolate them in tests only. Until then, delete
  `<data dir>/mirrors/notes` after a test run.
- Smaller: printed ranks can appear out of order (results are grouped by document type); the
  first query is slow without warning; `--db` after the subcommand gives a generic argparse error.
- Untested: the server on Windows, and two first queries starting it at once.

## Next

- Try other checkpoints: `INVENTIO_DISPOSITIO_MODEL=<dir or hf id> inventio query ...`; compare on
  the same question set.
- Decide the mirror isolation fix.

## Commands

```sh
uv tool install -q --reinstall-package inventio "$PWD[laya]"     # install from this checkout
uv run -q --python 3.12 --with pytest --with-editable ".[laya]" pytest -q tests
inventio serve --stop                                          # after changing the model
```
