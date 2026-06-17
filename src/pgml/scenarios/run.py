"""Run a sampled scenario batch through the (already batched) solver.

`run_scenarios` ties the reproducible sampler to the batched power-flow / harmonic
solve: one config -> one batched solve -> results aligned to the sampled inputs.
The whole batch solves in a single call (the solver broadcasts the leading scenario
dim), so generating large ML datasets is one vectorized solve, not a python loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

from pgml.assembly import NodePhaseIndex
from pgml.schemas.grid_schema import Grid
from pgml.solver import solve_harmonic_flow, solve_power_flow

from .config import CartesianConfig, ScenarioConfig
from .sampler import SampledScenarios, cartesian_sample, sample


@dataclass(frozen=True)
class ScenarioResult:
    """Batched results for a scenario config.

    Attributes
    ----------
    v:
        Node voltages — ``[B, N]`` for ``calculation="power_flow"`` or
        ``[B, H, N]`` for ``"harmonic"`` (B = n_samples).
    index:
        The compact :class:`NodePhaseIndex` (row layout of ``v``).
    sampled:
        The :class:`SampledScenarios` (operating points + raw sampled inputs + config)
        — the reproducible input record paired with ``v``.
    frequencies_hz:
        ``[H]`` orders×f0 for harmonic runs, else ``None``.
    """

    v: Tensor
    index: NodePhaseIndex
    sampled: SampledScenarios
    frequencies_hz: Optional[Tensor] = None


def run_scenarios(
    grid: Grid,
    spec: "ScenarioConfig | CartesianConfig | SampledScenarios",
    *,
    calculation: str = "power_flow",
    harmonic_orders: Optional[Sequence[int]] = None,
    slack: str = "ideal",
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
) -> ScenarioResult:
    """Build (if needed) and solve a scenario batch in one batched solve.

    Parameters
    ----------
    grid, slack, dtype, device:
        Passed to the solver (``grid`` is the canonical single grid; scenarios vary
        its operating point via the sampled batched ``operating_point`` override).
    spec:
        A :class:`ScenarioConfig` (random/QMC), a :class:`CartesianConfig` (grid
        sweep), or a pre-built :class:`SampledScenarios`.
    calculation:
        ``"power_flow"`` (fundamental) or ``"harmonic"`` (requires ``harmonic_orders``).
    harmonic_orders:
        Orders for the harmonic calculation (e.g. ``[1, 5, 7]``).
    """
    if isinstance(spec, SampledScenarios):
        sampled = spec
    elif isinstance(spec, CartesianConfig):
        sampled = cartesian_sample(grid, spec)
    else:
        sampled = sample(grid, spec)
    if calculation == "power_flow":
        res = solve_power_flow(
            grid, slack=slack, operating_point=sampled.operating_point,
            dtype=dtype, device=device,
        )
        return ScenarioResult(v=res.v, index=res.index, sampled=sampled)
    if calculation == "harmonic":
        if not harmonic_orders:
            raise ValueError("calculation='harmonic' requires harmonic_orders.")
        res = solve_harmonic_flow(
            grid, harmonic_orders, slack=slack,
            operating_point=sampled.operating_point, dtype=dtype, device=device,
        )
        return ScenarioResult(
            v=res.v, index=res.index, sampled=sampled, frequencies_hz=res.frequencies_hz
        )
    raise ValueError(f"Unknown calculation {calculation!r} (use 'power_flow'/'harmonic').")


__all__ = ["ScenarioResult", "run_scenarios"]
