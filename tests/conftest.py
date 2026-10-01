"""Shared fixtures. Nothing in tests/ requires a GPU."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parent.parent / "eval_mc3" / "kit"


@pytest.fixture
def tmp_store_dir(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    return d


@pytest.fixture
def kit_corpus(tmp_path) -> Path:
    """A private copy of the MC3 sample corpus, with the empty archive/ directory a zip cannot carry."""
    dst = tmp_path / "corpus"
    shutil.copytree(KIT / "mc3-corpus", dst)
    (dst / "archive").mkdir(exist_ok=True)
    return dst


@pytest.fixture
def kit_questions() -> list[dict]:
    return json.loads((KIT / "sample-questions.json").read_text(encoding="utf-8"))["queries"]
