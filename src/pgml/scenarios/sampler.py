"""Reproducible scenario sampler: ScenarioConfig -> batched operating points.

Turns a `ScenarioConfig` (+ its seed) into a batch of ``B = n_samples`` realized
operating points, as the batched ``operating_point`` override the solver consumes
(``{appliance_id: {"p_w": Tensor[B], "q_var": Tensor[B]}}``), plus the raw sampled
values per parameter (useful as ML inputs/labels). Deterministic: same config+seed
-> identical output.

All three methods produce unit-cube samples ``U in [0,1]^(B, D)`` then map each
column through that parameter's ``distribution.icdf`` — so QMC (Sobol/LHS) and plain
RNG share one transform path. ``D`` = total scalar random variables (one per matched
component for ``per="each"``, one for ``per="shared"``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pgml.schemas.grid_schema import Generator, Grid, Load

from .config import CartesianConfig, ParameterSpec, ScenarioConfig

# A built batch may originate from a random/QMC config or a cartesian config.
_AnyConfig = "ScenarioConfig | CartesianConfig"


@dataclass(frozen=True)
class SampledScenarios:
    """Realized batch of operating points.

    Attributes
    ----------
    operating_point:
        ``{appliance_id: {"p_w": Tensor[B], "q_var": Tensor[B]}}`` — pass straight to
        ``solve_power_flow``/``solve_harmonic_flow`` as ``operating_point=...``.
    samples:
        ``{parameter_name: Tensor[B, d]}`` — the raw sampled values (d = #matched
        components for ``per="each"``, else 1). The reproducible ML input record.
    n_samples:
        Batch size ``B``.
    config:
        The originating :class:`ScenarioConfig` (provenance / reproducibility).
    """

    operating_point: dict
    samples: dict
    n_samples: int
    config: "ScenarioConfig | CartesianConfig"


def _unit_samples(b: int, d: int, method: str, seed: int) -> Tensor:
    """``[b, d]`` quasi-/pseudo-random samples in ``[0, 1)`` (float64, deterministic)."""
    if method == "sobol":
        eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
        return eng.draw(b, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    if method == "independent":
        return torch.rand((b, d), generator=gen, dtype=torch.float64)
    if method == "lhs":
        # Latin hypercube: per dim a random permutation of the b strata + jitter.
        u = torch.empty((b, d), dtype=torch.float64)
        strata = torch.arange(b, dtype=torch.float64)
        for j in range(d):
            perm = torch.randperm(b, generator=gen)
            jitter = torch.rand(b, generator=gen, dtype=torch.float64)
            u[:, j] = (strata[perm] + jitter) / b
        return u
    raise ValueError(f"Unknown sampling method {method!r}.")


def _resolve(grid: Grid, config: ScenarioConfig):
    """List of ``(spec, ids, offset, dim)`` and the total sampling dimension D."""
    resolved = []
    dim = 0
    for spec in config.parameters:
        ids = spec.selector.resolve(grid)
        if not ids:
            raise ValueError(
                f"Parameter {spec.name!r} selector matched no in-service components."
            )
        d = len(ids) if spec.per == "each" else 1
        resolved.append((spec, ids, dim, d))
        dim += d
    if dim == 0:
        raise ValueError("ScenarioConfig has no parameters / sampling dimensions.")
    return resolved, dim


def _apply(entry: dict, spec: ParameterSpec, col: Tensor, p_nom: float, q_nom: float) -> None:
    """Write a sampled per-scenario column ``[B]`` into an operating_point entry."""
    if spec.field in ("p", "pq"):
        entry["p_w"] = col * p_nom if spec.mode == "scale" else col
    if spec.field in ("q", "pq"):
        entry["q_var"] = col * q_nom if spec.mode == "scale" else col


def _nominal(grid: Grid) -> dict:
    return {
        a.id: (float(a.p_nom_w), float(a.q_nom_var))
        for a in grid.appliances
        if isinstance(a, (Load, Generator))
    }


def sample(grid: Grid, config: ScenarioConfig) -> SampledScenarios:
    """Sample ``config.n_samples`` realized operating points from ``grid`` (reproducible)."""
    resolved, dim = _resolve(grid, config)
    b = config.n_samples
    u = _unit_samples(b, dim, config.method, config.seed)  # [B, D] in [0,1)

    nominal = _nominal(grid)
    operating_point: dict = {}
    samples: dict = {}
    for spec, ids, off, d in resolved:
        vals = spec.distribution.icdf(u[:, off : off + d])  # [B, d]
        samples[spec.name] = vals
        for j, cid in enumerate(ids):
            col = vals[:, j] if spec.per == "each" else vals[:, 0]  # [B]
            p_nom, q_nom = nominal[cid]
            _apply(operating_point.setdefault(cid, {}), spec, col, p_nom, q_nom)

    return SampledScenarios(
        operating_point=operating_point, samples=samples, n_samples=b, config=config
    )


def cartesian_sample(grid: Grid, config: CartesianConfig) -> SampledScenarios:
    """Cartesian product of axis levels (pgm-style grid sweep; deterministic).

    ``B = prod(len(axis.values))``. Each axis applies its level to ALL components its
    selector matches (list multiple axes with id selectors for per-component sweeps).
    """
    resolved = [(ax, ax.selector.resolve(grid)) for ax in config.axes]
    for ax, ids in resolved:
        if not ids:
            raise ValueError(f"Cartesian axis {ax.name!r} matched no in-service components.")
    levels = [torch.tensor(ax.values, dtype=torch.float64) for ax, _ in resolved]
    combos = torch.cartesian_prod(*levels)  # [B, n_axes] (or [B] for a single axis)
    if combos.ndim == 1:
        combos = combos.unsqueeze(-1)

    nominal = _nominal(grid)
    operating_point: dict = {}
    samples: dict = {}
    for i, (ax, ids) in enumerate(resolved):
        col = combos[:, i]  # [B]
        samples[ax.name] = col
        for cid in ids:
            p_nom, q_nom = nominal[cid]
            _apply(operating_point.setdefault(cid, {}), ax, col, p_nom, q_nom)

    return SampledScenarios(
        operating_point=operating_point, samples=samples, n_samples=combos.shape[0],
        config=config,
    )


__all__ = ["SampledScenarios", "sample", "cartesian_sample"]
