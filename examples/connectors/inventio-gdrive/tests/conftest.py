import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def private_data_home(monkeypatch, tmp_path):
    """Every test gets its own data directory (map, mirrors, model.json, serve.json), on every platform:
    macOS keeps it under ~/Library/Application Support, so HOME is moved too, or a test writes into the
    real one (a stray model.json once pointed the installed CLI at an old checkpoint) and two tests that
    mirror the same source name read each other's files. The Hugging Face cache stays where it is."""
    monkeypatch.setenv("HF_HOME", os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface"))
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
