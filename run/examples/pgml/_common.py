"""Shared helpers for the pgml runnable examples/benchmarks.

Not part of the pgml public API; imported only by scripts under
``run/examples/pgml/``.
"""

from __future__ import annotations

import time
from typing import Callable

import torch

from pgml.geometry.synthesis import strip_grid_geometry


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
    """Clear every Line's geometry and harmonic model so the R/X path is used.

    Delegates to :func:`pgml.geometry.strip_grid_geometry`: dropping
    ``conductor_geometry`` alone would leave a line on the ``geometry`` harmonic model
    with nothing to evaluate it from.
    """
    return strip_grid_geometry(grid)
