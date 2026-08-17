"""Headless matplotlib backend for the evaluation plot tests."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display in CI / background runs
