"""Headless matplotlib backend for the evaluation plot tests."""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib", exc_type=ImportError)

matplotlib.use("Agg")  # no display in CI / background runs
