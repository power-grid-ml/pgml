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
from pgml.errors import InputError
from pgml.schemas.grid_schema import Grid
from pgml.solver import solve_harmonic_flow, solve_power_flow

from .config import (
    CartesianConfig,
    CoherentSpectrumConfig,
    ScenarioConfig,
    SpectrumSweepConfig,
)
from .harmonics import sample_coherent_spectra, spectrum_sweep
from .sampler import SampledScenarios, cartesian_sample, sample


def _slice_range(x, start: int, end: int):
    """Slice the leading scenario dim ``[start:end]`` of a batched tensor (or list of
    them); pass scalars / 0-d / unbatched values through (they broadcast over the chunk)."""
    if isinstance(x, (list, tuple)):
        return type(x)(_slice_range(e, start, end) for e in x)
    if isinstance(x, Tensor) and x.ndim >= 1 and x.shape[0] > 1:
        return x[start:end]
    return x


def _slice_op_range(operating_point: dict, start: int, end: int) -> dict:
    """The ``operating_point`` restricted to scenarios ``[start:end]``."""
    return {
        cid: {k: _slice_range(v, start, end) for k, v in entry.items()}
        for cid, entry in operating_point.items()
    }


def _slice_inj_range(injection: dict, start: int, end: int) -> dict:
    """The batched ``harmonic_injection`` ``{id: {order: (mag, phase)}}`` over ``[start:end]``."""
    return {
        cid: {
            order: tuple(_slice_range(c, start, end) for c in pair)
            for order, pair in od.items()
        }
        for cid, od in injection.items()
    }


@dataclass(frozen=True)
class ScenarioResult:
    """Batched results for a scenario config.

    Attributes
    ----------
    v:
        Node voltages — ``[B, N]`` for ``calculation="power_flow"``, ``[B, H, N]`` for
        ``"harmonic"``, or ``[B, T, H, N]`` for a node-coherent
        :class:`CoherentSpectrumConfig` (T = steps; timestamps in ``sampled.samples``).
    index:
        The compact :class:`NodePhaseIndex` (row layout of ``v``).
    sampled:
        The :class:`SampledScenarios` (operating points + raw sampled inputs + config)
        — the reproducible input record paired with ``v``.
    frequencies_hz:
        ``[H]`` orders×f0 for harmonic runs, else ``None``.
    converged:
        ``True`` iff EVERY scenario converged. A batched run never raises on a failed
        scenario — its best-effort voltages are still returned in ``v``.
    failed_states:
        Flat indices of the scenarios that did not converge (empty when all did). The
        solver also logs an error naming them with the residual + likely cause.
    """

    v: Tensor
    index: NodePhaseIndex
    sampled: SampledScenarios
    frequencies_hz: Optional[Tensor] = None
    converged: bool = True
    failed_states: tuple[int, ...] = ()


def run_scenarios(
    grid: Grid,
    spec: "ScenarioConfig | CartesianConfig | SampledScenarios",
    *,
    calculation: str = "power_flow",
    harmonic_orders: Optional[Sequence[int]] = None,
    slack: str = "ideal",
    symmetry: Optional[str] = None,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    chunk_size: Optional[int] = None,
    output_device: Optional[torch.device] = None,
) -> ScenarioResult:
    """Build (if needed) and solve a scenario batch in one batched solve.

    Parameters
    ----------
    grid, slack, dtype, device:
        Passed to the solver (``grid`` is the canonical single grid; scenarios vary
        its operating point via the sampled batched ``operating_point`` override).
    spec:
        A :class:`ScenarioConfig` (random/QMC), a :class:`CartesianConfig` (grid
        sweep), a :class:`CoherentSpectrumConfig` (node-coherent harmonic sequences —
        forces ``calculation="harmonic"`` and defaults ``harmonic_orders`` to
        ``[1, *config.orders]``), or a pre-built :class:`SampledScenarios`.
    calculation:
        ``"power_flow"`` (fundamental) or ``"harmonic"`` (requires ``harmonic_orders``).
    harmonic_orders:
        Orders for the harmonic calculation (e.g. ``[1, 5, 7]``).
    symmetry:
        Calculation symmetry forwarded to the solver: ``None`` / ``"auto"`` (default;
        per-phase sampled operating points auto-promote to asymmetric), ``"symmetric"``
        (force equal split, ignore per-phase samples), or ``"asymmetric"``.
    chunk_size:
        When set, solve the scenario batch in slices of at most ``chunk_size`` scenarios
        and concatenate, so a batch whose dense ``[B, H, N, N]`` system exceeds memory
        still fits (stream the batch instead of materialising it whole). The result is
        identical to the un-chunked solve (each scenario is independent) and stays
        differentiable (the concatenation preserves the graph). Applies to the node-coherent
        ``[B, T, H, N]`` path too — the slice is along the SCENARIO axis ``B`` (each
        scenario's full ``T``-step sequence solves together). ``None`` (default) solves the
        whole batch.
    output_device:
        Where the RESULT voltages ``v`` are collected. ``None`` (default) keeps them on the
        solve ``device``. When generating a large dataset on the GPU, the full ``[B, ...]``
        result tensor would otherwise accumulate in VRAM and OOM even with a small
        ``chunk_size`` (``chunk_size`` bounds the per-solve WORKSPACE, not the collected
        OUTPUT). Set ``output_device="cpu"`` to move each chunk's result off the GPU as it is
        produced, so VRAM stays bounded to one chunk; the dataset is written from host memory.
        Intended for (non-differentiable) data generation — leave ``None`` to keep ``v`` on the
        solve device for a differentiable GPU pipeline.
    """
    is_coherent = isinstance(spec, CoherentSpectrumConfig)
    if is_coherent:
        sampled = sample_coherent_spectra(grid, spec)
        calculation = "harmonic"
        if harmonic_orders is None:
            harmonic_orders = [1, *spec.orders]
    elif isinstance(spec, SpectrumSweepConfig):
        sampled = spectrum_sweep(grid, spec)
        calculation = "harmonic"
        if harmonic_orders is None:
            harmonic_orders = [1, *spec.orders]
    elif isinstance(spec, SampledScenarios):
        sampled = spec
    elif isinstance(spec, CartesianConfig):
        sampled = cartesian_sample(grid, spec)
    else:
        sampled = sample(grid, spec)

    if calculation == "harmonic" and not harmonic_orders:
        raise InputError("calculation='harmonic' requires harmonic_orders.")

    def _solve(op, inj):
        """Solve one (sub)batch -> (v, index, converged, failed_states, frequencies)."""
        if calculation == "power_flow":
            r = solve_power_flow(
                grid,
                slack=slack,
                operating_point=op,
                symmetry=symmetry,
                dtype=dtype,
                device=device,
            )
            return r.v, r.index, r.converged, r.failed_states, None
        if calculation == "harmonic":
            r = solve_harmonic_flow(
                grid,
                harmonic_orders,
                slack=slack,
                operating_point=op,
                harmonic_injection=inj,
                symmetry=symmetry,
                dtype=dtype,
                device=device,
            )
            return r.v, r.index, r.pf.converged, r.pf.failed_states, r.frequencies_hz
        raise InputError(
            f"Unknown calculation {calculation!r} (use 'power_flow'/'harmonic')."
        )

    op_full = sampled.operating_point
    inj_full = sampled.harmonic_injection or None
    b = int(sampled.n_samples)

    if chunk_size is None or chunk_size >= b or b <= 1:
        v, index, converged, failed, freqs = _solve(op_full, inj_full)
        if output_device is not None:
            v = v.to(output_device)
        return ScenarioResult(
            v=v,
            index=index,
            sampled=sampled,
            frequencies_hz=freqs,
            converged=converged,
            failed_states=failed,
        )

    # Stream the batch in chunks along the SCENARIO axis (a size-1 chunk loses its leading
    # scenario axis in the solver -> add it back so the per-chunk results concatenate into the
    # full [B, ...] tensor). The node-coherent path carries an extra step axis -> [B, T, H, N].
    if calculation == "power_flow":
        batched_ndim = 2
    elif is_coherent:
        batched_ndim = 4
    else:
        batched_ndim = 3
    v_parts: list[Tensor] = []
    failed: list[int] = []
    converged = True
    index = None
    freqs = None
    for start in range(0, b, chunk_size):
        end = min(start + chunk_size, b)
        op_c = _slice_op_range(op_full, start, end)
        inj_c = _slice_inj_range(inj_full, start, end) if inj_full else None
        v_c, index, conv_c, failed_c, freqs = _solve(op_c, inj_c)
        if v_c.ndim == batched_ndim - 1:
            v_c = v_c.unsqueeze(0)
        # move each chunk OFF the solve device as it is produced (when requested) so the
        # collected result does not accumulate in VRAM and OOM regardless of chunk_size.
        if output_device is not None:
            v_c = v_c.to(output_device)
        v_parts.append(v_c)
        converged = converged and conv_c
        failed.extend(start + i for i in failed_c)
    return ScenarioResult(
        v=torch.cat(v_parts, dim=0),
        index=index,
        sampled=sampled,
        frequencies_hz=freqs,
        converged=converged,
        failed_states=tuple(failed),
    )


__all__ = ["ScenarioResult", "run_scenarios"]
