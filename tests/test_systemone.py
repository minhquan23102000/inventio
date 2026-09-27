"""Which checkpoint answers `--ranker dispositio`, and that a checkpoint can be read at all.

The checkpoint in the last test is built here — a two-layer Qwen2 and a randomly initialised pointer
head — so nothing about the *numbers* is under test, only that this repository's code turns a
checkpoint and a state into the answers the served model returned. The probabilities a threshold rests
on come from a real checkpoint; this test is the shape, the schema and the load path.
"""

import json
import shutil

import pytest

from inventio import systemone
from inventio.search import Hit


def tokenizer_dir() -> str | None:
    """A tokenizer the vendored loader can read with no network: the Qwen3.5-0.8B-Base snapshot already
    in the Hugging Face cache, else None (the load test then skips)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    try:
        return snapshot_download("Qwen/Qwen3.5-0.8B-Base", local_files_only=True,
                                 allow_patterns=["*.json", "*.txt", "tokenizer*", "vocab*", "merges*"])
    except Exception:
        return None


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """The registry is a file in the data home; keep it in the test's own directory."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.delenv("INVENTIO_DISPOSITIO_MODEL", raising=False)
    monkeypatch.delenv("INVENTIO_DEVICE", raising=False)


def hits():
    text = "# Rate limits\n\nThe API allows 60 requests a minute per key.\n# Retries\n\nA failed call is retried.\n"
    return [Hit(id=1, source="docs", public=True, root="/docs", path="api.md", start_line=1, end_line=7,
                heading_path="Limits", text=text)]


def tiny_run(tmp_path, max_positions=1024) -> str:
    """A full-weight checkpoint this machine can run without a GPU: two tiny attention layers and a randomly
    initialised pointer head (a hybrid Qwen3.5 base would need a compiled kernel, which is the accelerator's
    business, not these tests'). Skips when the extra or a cached tokenizer is missing."""
    from inventio._systemone import missing
    if missing():
        pytest.skip(missing())
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from inventio._systemone.checkpoint import Meta, write_meta
    from inventio._systemone.model import DecisionModel, load_tokenizer

    base = tokenizer_dir()
    if base is None:
        pytest.skip("no tokenizer in the Hugging Face cache and no network: the load path needs one")
    tok = load_tokenizer(base)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
                                   max_position_embeddings=max_positions)
    backbone = tmp_path / "backbone"
    transformers.AutoModelForCausalLM.from_config(cfg).save_pretrained(backbone)
    run = tmp_path / "run"
    run.mkdir()
    for name in ("config.json", "model.safetensors"):
        shutil.copy(backbone / name, run / name)
    write_meta(run, Meta(base=base, head=DecisionModel(str(backbone), tok, "cpu", head_dim=32).head.state_dict(),
                         head_dim=32, weights="full", weights_dtype="fp32", temperature=1.0))
    return str(run)


def test_run_id_is_the_env_then_what_use_recorded_then_the_published_model(tmp_path, monkeypatch):
    assert systemone.run_id() == systemone.DEFAULT_MODEL
    monkeypatch.setenv("INVENTIO_DISPOSITIO_MODEL", "some/run")
    assert systemone.run_id() == "some/run"
    monkeypatch.delenv("INVENTIO_DISPOSITIO_MODEL")

    run = tmp_path / "run"
    run.mkdir()
    (run / "head.pt").write_bytes(b"x")
    assert systemone.use(str(run)) == str(run)
    assert systemone.run_id() == str(run)
    assert json.loads(systemone.registry_file().read_text(encoding="utf-8"))["run"] == str(run)


def test_use_refuses_a_directory_without_a_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="not a checkpoint"):
        systemone.use(str(tmp_path / "nowhere"))
    # a Hub id needs no directory: it is downloaded at load time
    assert systemone.use("minhquan2310/dispositio@v4") == "minhquan2310/dispositio@v4"
    assert systemone.run_id() == "minhquan2310/dispositio@v4"


def test_a_state_the_server_answered_is_not_refused_here(tmp_path):
    """The reader must encode with the *serving* limits, not the training ones. A line question offers one
    option per line, so a state of a few hundred lines makes a branch far past the training row budget
    (1,024 tokens) — which the server answered, so this must too (measured: it raised ContextOverflow)."""
    run = tiny_run(tmp_path, max_positions=8192)
    body = "\n".join(f"line {i}: the API allows 60 requests a minute per key" for i in range(300))
    hit = Hit(id=1, source="docs", public=True, root="/docs", path="api.md", start_line=1, end_line=300,
              heading_path="Limits", text=body)
    state, pids, lids, owner = systemone.render([hit])
    assert len(lids) > 255, len(lids)   # the branch is past the training row budget
    res = systemone.Model(str(run), "cpu").ask(state, systemone.questions("how many requests a minute?", pids, lids))
    assert res["answers"]["where_line"]["choice"] in lids


def test_the_vendored_reader_loads_a_checkpoint_and_answers_one_state(tmp_path):
    """The in-process path end to end: a run directory -> tokenizer and model -> one state -> the answers
    `kev.serve` would have returned (same request schema, same response keys)."""
    run = tiny_run(tmp_path)
    model = systemone.Model(str(run), "cpu")
    assert model.info()["run"] == str(run) and model.info()["temperature"] == 1.0
    state, pids, lids, owner = systemone.render(hits())
    assert pids == ["P01"] and len(lids) == 4, state

    res = model.ask(state, systemone.questions("how many requests a minute?", pids, lids))
    a = res["answers"]
    assert set(a) == {"where_passage", "where_line", "exists"}
    assert set(a["where_passage"]["probabilities"]) == set(pids)
    assert abs(sum(a["where_passage"]["probabilities"].values()) - 1) < 0.02
    assert 0.0 <= a["exists"]["noul"] <= 1.0
    assert a["where_line"]["choice"] in lids
    assert res["usage"]["input_tokens"] > 0 and res["latency_ms"] >= 0


def test_private_text_is_read_in_process_but_never_sent_to_a_remote_model(tmp_path, monkeypatch):
    """The default reads the map in this process, so a private source is ranked like any other; the same
    ranker pointed at a remote URL refuses it before anything is sent."""
    from dataclasses import replace
    from inventio.rankers import CloudRefused, SystemOneRanker

    monkeypatch.delenv("INVENTIO_DISPOSITIO_URL", raising=False)
    monkeypatch.setenv("INVENTIO_DEVICE", "cpu")
    private = [replace(h, public=False) for h in hits()]
    ps = SystemOneRanker(run=tiny_run(tmp_path)).score("how many requests a minute?", private)
    assert len(ps) == 1 and 0.0 <= ps[0] <= 1.0
    with pytest.raises(CloudRefused):
        SystemOneRanker(url="https://models.example.com").score("how many requests a minute?", private)
def test_the_map_is_judged_and_types_are_predicted_by_the_same_checkpoint(tmp_path, monkeypatch):
    """`inventio facts` and `--types` with the System One model: every prose chunk gets one probability per
    category, a chunk's neighbours are asked in one packed state and every pair gets its answer, the judgments
    are cached under the run that gave them, and the type head answers over exactly the types offered."""
    run = tiny_run(tmp_path)
    monkeypatch.setenv("INVENTIO_DEVICE", "cpu")
    monkeypatch.setenv("INVENTIO_DISPOSITIO_MODEL", run)
    from inventio import facts
    from inventio.ingest import ingest_source
    from inventio.rankers import SystemOneRanker
    from inventio.store import connect

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "limits.md").write_text("# Rate limits\n\nThe API allows 60 requests a minute per key.\n\n"
                                    "# Raising the limit\n\nRun `quota raise --key K` and wait a day.\n", encoding="utf-8")
    (docs / "incident.md").write_text("# 2026-09-01 outage\n\nThe rate limit of 60 requests a minute per key "
                                      "was hit by the nightly job and every call failed.\n", encoding="utf-8")
    con = connect(tmp_path / "map.db")
    with con:
        ingest_source(con, "docs", docs, True, [])
    judge = facts.make_judge("dispositio")
    assert judge.name == f"dispositio:{run}"
    res = facts.build(con, judge, ["docs"], relink=True)

    chunks = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
    assert res["categories"]["asked"] == chunks > 0
    for (cid,) in con.execute("SELECT id FROM chunks"):
        ps = dict(con.execute("SELECT category, p FROM chunk_categories WHERE chunk_id = ?", (cid,)).fetchall())
        assert set(ps) == set(facts.CATEGORIES) and abs(sum(ps.values()) - 1) < 0.02
    assert res["links"]["asked"] == res["links"]["pairs"] - res["links"]["cached"]
    assert {r[0] for r in con.execute("SELECT DISTINCT model FROM judgments")} == {judge.name}

    offered = {"Article": "prose", "SoftwareSourceCode": "code"}
    probs = SystemOneRanker().types("how many requests a minute?", offered)
    assert set(probs) == set(offered) and abs(sum(probs.values()) - 1) < 0.02
