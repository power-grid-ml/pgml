"""Small framework-bridging helpers for the evaluation package.

The evaluation package consumes solver outputs (torch tensors, possibly on GPU and
possibly carrying autograd history) and turns them into plain numpy for plotting.
Plotting is OUTSIDE the differentiable path, so detaching here is correct and
intended (it never runs on the tape that feeds a loss).
"""

from __future__ import annotations

import numpy as np


def to_float(x) -> float:
    """Coerce a python number OR a 0-d/array-like tensor to a python float.

    Honors the schema's float/tensor duality (a field may be a ``torch.Tensor``).
    """
    if hasattr(x, "detach"):
        return float(x.detach().cpu().reshape(()).item())
    return float(x)


def to_numpy(x) -> np.ndarray:
    """Coerce a torch tensor / array-like to a detached numpy array (plotting only)."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


__all__ = ["to_float", "to_numpy"]
