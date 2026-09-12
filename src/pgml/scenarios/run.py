"""Run a sampled scenario batch through the (already batched) solver.

`run_scenarios` ties the reproducible sampler to the batched power-flow / harmonic
solve: one config -> one batched solve -> results aligned to the sampled inputs.
The whole batch solves in a single call (the solver broadcasts the leading scenario
dim), so generating large ML datasets is one vectorized solve, not a python loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor

from pgml.assembly import NodePhaseIndex
from pgml.errors import InputError
from pgml.schemas.grid_schema import Grid
from pgml.solver import prepare_power_flow, solve_harmonic_flow, solve_power_flow

from .sampler import SampledScenarios


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


def _slice_sources(sources, start: int, end: int) -> list:
    """The batched ``node_sources`` over ``[start:end]`` of the SCENARIO axis.

    ``source_power_va`` is a batch-independent constant, so only the spectrum is sliced
    and the network stamp each source contributes is identical across chunks.
    """
    from dataclasses import replace

    return [
        replace(
            src,
            spectrum={
                order: tuple(_slice_range(c, start, end) for c in pair)
                for order, pair in src.spectrum.items()
            },
        )
        for src in sources
    ]


@dataclass(frozen=True)
class ScenarioResult:
    """Batched results for a scenario config.

    Attributes
    ----------
    v:
        Node voltages — ``[B, N]`` for ``calculation="power_flow"``, ``[B, H, N]`` for
        ``"harmonic"``, or ``[B, T, H, N]`` for a sequence batch
        (``sampled.n_steps > 1``; timestamps in ``sampled.samples["time_s"]``).
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
        SCENARIO indices (along the ``B`` axis of ``v``) that did not converge (empty
        when all did). A sequence scenario counts as failed when ANY of its ``T``
        steps failed. The solver also logs an error with the residual + likely cause.
    """

    v: Tensor
    index: NodePhaseIndex
    sampled: SampledScenarios
    frequencies_hz: Optional[Tensor] = None
    converged: bool = True
    failed_states: tuple[int, ...] = ()


@runtime_checkable
class ScenarioSpec(Protocol):
    """What :func:`run_scenarios` needs from a batch specification.

    Any object that can turn a grid into a :class:`SampledScenarios` is a spec, which is
    how a downstream package plugs its own generator into this run path without pgml
    knowing the recipe. The serializable configs of :mod:`pgml.scenarios.config` satisfy
    it themselves.

    Members
    -------
    sample(grid):
        Draw / assemble the batch for ``grid``.
    harmonic_orders:
        Optional hint. A non-empty sequence declares that this spec only makes sense as a
        harmonic calculation: :func:`run_scenarios` then switches ``calculation`` to
        ``"harmonic"`` and uses the sequence as the solved order set unless the caller
        named one. ``None`` leaves both to the caller.
    """

    harmonic_orders: Optional[Sequence[int]]

    def sample(self, grid: Grid) -> SampledScenarios: ...


def _scenario_failures(failed_flat, v_chunk: Tensor, n_steps: int) -> tuple[int, ...]:
    """Solver-flat failed indices -> unique scenario indices along ``B``.

    A sequence batch (``n_steps > 1``) solves a ``[B, T]`` leading batch, so the solver's
    convergence mask flattens over ``B*T`` — a step index maps to its scenario via
    ``// T``. A snapshot batch has one solve per scenario.
    """
    if n_steps > 1:
        if v_chunk.ndim >= 4:
            t = v_chunk.shape[-3]
            return tuple(sorted({i // t for i in failed_flat}))
        # Single-scenario sequence [T, H, N]: any failed step fails scenario 0.
        return (0,) if failed_flat else ()
    return tuple(failed_flat)


def run_scenarios(
    grid: Grid,
    spec: "ScenarioSpec | SampledScenarios",
    *,
    calculation: str = "power_flow",
    harmonic_orders: Optional[Sequence[int]] = None,
    slack: str = "ideal",
    symmetry: Optional[str] = None,
    load_shunt: Optional[str] = None,
    load_shunt_basis: Optional[str] = None,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    chunk_size: Optional[int] = None,
    output_device: Optional[torch.device] = None,
) -> ScenarioResult:
    """Build (if needed) and solve a scenario batch in one batched solve.

    For ``calculation="power_flow"`` the operating-point-independent solve state
    (assembly, slack rows, factorization) is prepared once via
    :func:`~pgml.solver.prepare_power_flow` and reused across the whole batch —
    and across every ``chunk_size`` slice, when chunking — since only the
    operating point differs between scenarios; see the ``system`` parameter of
    :func:`~pgml.solver.solve_power_flow`. The harmonic path assembles per
    order inside :func:`~pgml.solver.solve_harmonic_flow` and does not use this
    reuse.

    Parameters
    ----------
    grid, slack, dtype, device:
        Passed to the solver (``grid`` is the canonical single grid; scenarios vary
        its operating point via the sampled batched ``operating_point`` override).
    spec:
        A :class:`ScenarioSpec` — anything with a ``sample(grid)`` method, which the
        serializable configs (:class:`ScenarioConfig`, :class:`CartesianConfig`,
        :class:`SpectrumSweepConfig`) satisfy — or a pre-built
        :class:`SampledScenarios` (for example from
        :func:`~pgml.scenarios.batch_from_values`). A spec whose
        ``harmonic_orders`` hint is non-empty forces ``calculation="harmonic"`` and
        supplies the default order set.
    calculation:
        ``"power_flow"`` (fundamental) or ``"harmonic"`` (requires ``harmonic_orders``).
    harmonic_orders:
        Orders for the harmonic calculation (e.g. ``[1, 5, 7]``).
    load_shunt:
        Harmonic device Norton shunt forwarded to
        :func:`~pgml.solver.solve_harmonic_flow` (``"none"`` / ``"opendss"`` /
        ``"motor"``; ``None`` = the documented modeling default). Harmonic calculation
        only. A shunt derived from a PER-SCENARIO operating point makes ``Y(h)``
        scenario-dependent, so each scenario is factored on its own — ``"none"`` keeps
        the single shared factorization.
    load_shunt_basis:
        Which power and terminal voltage that shunt is built from
        (:func:`~pgml.solver.solve_harmonic_flow`): ``"operating_point"`` follows each
        scenario, ``"nameplate"`` uses the device's stored P, Q at its rated voltage and
        so keeps ONE factorization per order for the whole batch. ``None`` = the
        documented modeling default ``appliance.harmonic_shunt.basis``.
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
    if isinstance(spec, SampledScenarios):
        sampled = spec
    elif callable(getattr(spec, "sample", None)):
        sampled = spec.sample(grid)
        hint = getattr(spec, "harmonic_orders", None)
        if hint:
            calculation = "harmonic"
            if harmonic_orders is None:
                harmonic_orders = list(hint)
    else:
        raise InputError(
            f"run_scenarios spec {type(spec).__name__!r} is neither a SampledScenarios "
            "nor a scenario spec (an object with a sample(grid) method)."
        )
    sampled.validate(grid)
    n_steps = int(sampled.n_steps)

    if calculation == "harmonic" and not harmonic_orders:
        raise InputError("calculation='harmonic' requires harmonic_orders.")

    # The network side (assembly + slack + factorization) is operating-point
    # independent: prepare it ONCE and reuse it across every chunk. The harmonic
    # path assembles per order inside solve_harmonic_flow and keeps its own flow.
    system = (
        prepare_power_flow(grid, slack=slack, dtype=dtype, device=device)
        if calculation == "power_flow"
        else None
    )

    def _solve(op, inj, sources=()):
        """Solve one (sub)batch -> (v, index, converged, failed_states, frequencies)."""
        if calculation == "power_flow":
            r = solve_power_flow(
                grid,
                slack=slack,
                operating_point=op,
                symmetry=symmetry,
                dtype=dtype,
                device=device,
                system=system,
            )
            return r.v, r.index, r.converged, r.failed_states, None
        if calculation == "harmonic":
            r = solve_harmonic_flow(
                grid,
                harmonic_orders,
                slack=slack,
                operating_point=op,
                harmonic_injection=inj,
                node_sources=list(sources) or None,
                load_shunt=load_shunt,
                load_shunt_basis=load_shunt_basis,
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
    src_full = getattr(sampled, "node_sources", []) or []
    b = int(sampled.n_samples)

    if chunk_size is None or chunk_size >= b or b <= 1:
        v, index, converged, failed, freqs = _solve(op_full, inj_full, src_full)
        if output_device is not None:
            v = v.to(output_device)
        return ScenarioResult(
            v=v,
            index=index,
            sampled=sampled,
            frequencies_hz=freqs,
            converged=converged,
            failed_states=_scenario_failures(failed, v, n_steps),
        )

    # Stream the batch in chunks along the SCENARIO axis (a size-1 chunk loses its leading
    # scenario axis in the solver -> add it back so the per-chunk results concatenate into the
    # full [B, ...] tensor). The node-coherent path carries an extra step axis -> [B, T, H, N].
    if calculation == "power_flow":
        batched_ndim = 2
    else:
        batched_ndim = 4 if n_steps > 1 else 3
    v_parts: list[Tensor] = []
    failed: list[int] = []
    converged = True
    index = None
    freqs = None
    for start in range(0, b, chunk_size):
        end = min(start + chunk_size, b)
        op_c = _slice_op_range(op_full, start, end)
        inj_c = _slice_inj_range(inj_full, start, end) if inj_full else None
        src_c = _slice_sources(src_full, start, end) if src_full else []
        v_c, index, conv_c, failed_c, freqs = _solve(op_c, inj_c, src_c)
        if v_c.ndim == batched_ndim - 1:
            v_c = v_c.unsqueeze(0)
        # move each chunk OFF the solve device as it is produced (when requested) so the
        # collected result does not accumulate in VRAM and OOM regardless of chunk_size.
        if output_device is not None:
            v_c = v_c.to(output_device)
        v_parts.append(v_c)
        converged = converged and conv_c
        failed.extend(start + i for i in _scenario_failures(failed_c, v_c, n_steps))
    return ScenarioResult(
        v=torch.cat(v_parts, dim=0),
        index=index,
        sampled=sampled,
        frequencies_hz=freqs,
        converged=converged,
        failed_states=tuple(failed),
    )


__all__ = ["ScenarioResult", "ScenarioSpec", "run_scenarios"]
