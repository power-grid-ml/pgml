"""Shared helpers for the pgml runnable examples/benchmarks.

Not part of the pgml public API; imported only by scripts under
``run/examples/pgml/``.
"""

from __future__ import annotations

import time
from typing import Callable

import torch

from pgml.schemas.grid_schema import Line


def _time(fn: Callable[[], object], *, repeat: int, sync: bool = False) -> float:
    """Best-of-``repeat`` wall time of ``fn()`` in seconds (1 warmup call)."""
    fn()
    best = float("inf")
    for _ in range(repeat):
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def _strip_geometry(grid):
    """Clear ``conductor_geometry`` off every Line so the R/X->geometry path is unused."""
    for b in grid.branches:
        if isinstance(b, Line):
            b.conductor_geometry = None
    return grid
