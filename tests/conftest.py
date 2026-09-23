"""Shared fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"

requires_hyperfine = pytest.mark.skipif(
    shutil.which("hyperfine") is None, reason="hyperfine is not installed"
)
