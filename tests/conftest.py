"""Shared fixtures. Nothing in tests/ requires a GPU."""
from __future__ import annotations

import pytest


@pytest.fixture
def tmp_store_dir(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    return d
