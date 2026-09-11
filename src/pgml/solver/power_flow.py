"""Nonlinear fundamental power flow (const-P / ZIP) via current injection.

Public API
----------
- ``solve_power_flow(grid, *, slack, method, tol, max_iter, dtype, device,
  operating_point, param_overrides) -> PowerFlowResult``

Forward
-------
A current-injection FIXED POINT at the fundamental frequency ``f0 =
grid.base_frequency_hz``. With the passive-network admittance ``Y_net =
assemble_network_ybus`` (plus the source Norton shunt when ``slack="norton"``),
iterate

    V_{k+1} = solve_harmonic(Y_eff, I_slack - I_device(V_k), slack...)

until ``||V_{k+1} - V_k|| < tol`` (or ``max_iter``), where ``I_device`` is the
ZIP voltage-dependent device current from
:func:`pgml.assembly.device_current_injections`. The iteration runs under
``torch.no_grad()`` (CORRECT for implicit diff: the gradient is supplied
analytically, NOT by unrolling).

Backward (IMPLICIT FUNCTION THEOREM, REAL coordinates)
------------------------------------------------------
The power flow is non-holomorphic in ``V`` (``I_device`` uses ``|V|`` and
``conj(V)``), so the residual / Jacobian / adjoint are formed in REAL (re/im
split) coordinates. The converged ``V*`` satisfies the full-row real residual

    R(x, theta) = 0,   x = [Re(V); Im(V)]  (size 2N per system),

where free rows carry the complex power-balance residual ``Re/Im(Y_eff V +
I_device(V) - I_slack)`` and ideal-slack rows carry ``Re/Im(V - V_fixed)``
(pinned). For an output cotangent ``grad_V`` on ``V*`` the IFT gives:

    solve  J^T lambda = grad_x          (ONE real linear solve per system),
    grad_theta = -(dR/dtheta)^T lambda  (a vjp of a single residual eval at V*).

Implemented as a :class:`torch.autograd.Function`: ``forward`` returns the
no_grad ``V*``; ``backward`` builds the real ``[2N, 2N]`` Jacobian at ``V*``,
solves the adjoint, and runs ``torch.autograd.grad`` on a single residual
evaluation to produce gradients at every parameter LEAF (network params, device
P/Q, slack voltage), reached through any derived-parameter expression a Grid
field holds (float/tensor duality: one leaf may feed several fields).

Differentiability + GPU (CLAUDE.md): the differentiable path is the IFT backward
(no unrolling). ``no_grad`` in the forward iteration is expected. No
``.item()/.detach()/.numpy()`` on the autograd tape, no in-place on tracked
tensors, no python control flow on tensor VALUES (the convergence test is a scalar
norm under ``no_grad``). Honors input device/dtype; runs unchanged on CPU/CUDA.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from pgml import defaults
from pgml.assembly import (
    NodePhaseIndex,
    assemble_network_ybus,
    assemble_ybus,
    build_injection_plan,
    build_injections,
    device_current_injections,
    injections_from_plan,
    node_phase_index,
)
from pgml.assembly._stamps import _cdtype, _rdtype
from pgml.assembly._symmetry import log_modeling_summary, resolve_asymmetric
from pgml.assembly.ybus import _stamp_sources, flatten_plan_batch
from pgml.errors import ConnectivityError, InputError, ModelingError
from pgml.schemas.grid_schema import Grid, InjectionAppliance, Load, Source
from pgml.topology import connectivity_report, energized_subgrid, network_fingerprint

from ._pv_bus import PVTerminals, active_power_mismatch, collect_pv_terminals
from .harmonic import (
    estimate_condition,
    lu_factor_system,
    resolve_precision,
    solve_factored,
    solve_harmonic,
)
from .lowrank import (
    LowRankOperator,
    LowRankUpdate,
    branch_state_terms,
    low_rank_update,
    solve_factored_updated,
)

_log = logging.getLogger("pgml")


def _rel_convergence_floor(rdt: torch.dtype, backend: str = "dense") -> float:
    """Smallest per-unit VOLTAGE UPDATE the working precision and backend can resolve.

    The per-row update ``|ΔV| / V_LN`` stops shrinking once it reaches the rounding noise
    of the working precision, so a tolerance below that is unreachable: the floor caps
    the achievable tolerance and is reported by a warning.

    Calibrated on a 5000-scenario batched IEEE-33 run (CPU) by iterating to the plateau:
    ``float64`` dense reaches 1.0e-15 per-row relative (the floor is 16 eps ~ 3.6e-15,
    ~3x headroom and still ~1000x below any useful tolerance); ``float32`` dense was
    still contracting at 8.8e-7, so the 1e-6 floor stands. SuperLU's single-precision
    back-substitution (different pivoting / ordering than the dense torch LU) is
    noisier: the same batch plateaus FLAT at 5.7e-6 per row, which is why the sparse
    float32 floor is 1.2e-5 (~2x headroom) — without it marginal scenarios oscillate
    above a dense-calibrated floor and run to ``max_iter`` at floor-accurate voltages.
    The block backend back-substitutes with the same torch LU as the dense one and
    shares its floor. A mixed-precision factorization is refined at the working
    precision, so it resolves the working precision's floor, not the factors'.
    """
    if rdt == torch.float64:
        return 16.0 * float(torch.finfo(torch.float64).eps)
    return 1.2e-5 if backend == "sparse" else 1.0e-6


def _mismatch_floor_rel(rdt: torch.dtype, backend: str = "dense") -> float:
    """Relative floor of the POWER MISMATCH, as a fraction of a row's ``Y V`` scale.

    The nodal residual is a difference of terms of size ``Σ_j |Y_ij||V_j|``
    (:func:`_abs_row_scale`), so floating point resolves it only down to this fraction
    of that scale — a 20 kV node behind a 1e-4 Ohm source impedance keeps terms of
    ~1e9 A and its mismatch therefore bottoms out near 1e-9 pu at complex128, which no
    iteration count can improve.

    ``float64`` uses 4 eps (measured: the CIGRE LV benchmark bottoms out at 1.05e-9 pu
    where this estimate gives 7.1e-9 pu, so the floor stays below the documented default
    tolerance while covering the observed noise). ``float32`` uses the back-substitution
    precision of the backend, which dominates the cancellation term there.
    """
    if rdt == torch.float64:
        return 4.0 * float(torch.finfo(torch.float64).eps)
    return 4.0e-6 if backend == "sparse" else 1.0e-6


def _resolve_tolerances(
    tol: Optional[float], tol_update_pu: Optional[float], s_base_va: Optional[float]
) -> tuple[float, float, float]:
    """Fill the per-unit convergence settings from the documented defaults.

    ``None`` means "the shipped default" (``solver.convergence.*`` in
    ``pgml/data/defaults.yaml``), so a caller never has to restate the tolerances and
    the defaults stay in one documented place.
    """
    if tol is None:
        tol = float(defaults.get("solver.convergence.mismatch_pu"))
    if tol_update_pu is None:
        tol_update_pu = float(defaults.get("solver.convergence.update_pu"))
    if s_base_va is None:
        s_base_va = float(defaults.get("solver.convergence.s_base_va"))
    if not (tol > 0.0 and tol_update_pu > 0.0 and s_base_va > 0.0):
        raise InputError(
            "tol (per-unit power mismatch), tol_update_pu (per-unit voltage update) "
            f"and s_base_va must be positive; got {tol!r}, {tol_update_pu!r}, "
            f"{s_base_va!r}."
        )
    return float(tol), float(tol_update_pu), float(s_base_va)


def _abs_row_scale(y_eff, v_abs: Tensor) -> Tensor:
    """``Σ_j |Y_ij| |V_j|`` per row — the cancellation scale of the residual ``Y V``.

    The nodal residual is a difference of terms of this size, so floating point
    resolves it only down to ``eps`` times this scale: it is what turns the working
    precision into a per-row floor on the power-mismatch criterion. A switch-state
    sweep's matrix-free operator (``A + U C Vᴴ``) adds the low-rank term's row sums,
    bounded factor by factor — a closed near-ideal switch raises the scale of its
    terminal rows by orders of magnitude, and the floor must follow it.
    """
    if isinstance(y_eff, LowRankOperator):
        base = _apply_y(y_eff.base.abs(), v_abs)  # [*b, N]
        vt = torch.matmul(y_eff.v.abs().mT, v_abs.unsqueeze(-1))  # [k, 1]
        cvt = torch.matmul(y_eff.c.abs(), vt)  # [*states, k, 1]
        return base + torch.matmul(y_eff.u.abs(), cvt).squeeze(-1)
    return _apply_y(y_eff.abs(), v_abs)


#: Once-per-process guard for the plain-complex64 conditioning check (the estimate
#: costs a few back-substitutions, so a scenario sweep must not pay it per solve).
_COMPLEX64_COND_CHECKED = False


def _warn_complex64_conditioning(fac, rdt: torch.dtype) -> None:
    """Warn ONCE when a plain complex64 solve runs on an ill-conditioned system.

    A single-precision solve loses about ``cond(Y) * 1.2e-7`` of relative accuracy, and
    an SI-unit power system is ill-conditioned because the engine carries no per-unit
    normalisation (measured on the factored fundamental system: IEEE-33 ~2.8e3, CIGRE LV
    ~1.7e4 to 5.5e4, decades higher with a stiff source or a near-ideal switch). The
    estimate runs against the factorization the solve already built
    (:func:`~pgml.solver.harmonic.estimate_condition`), ONCE per process, and the
    threshold is the documented ``solver.precision.complex64_cond_warn``.
    """
    global _COMPLEX64_COND_CHECKED
    if _COMPLEX64_COND_CHECKED or rdt != torch.float32 or fac.precision != "full":
        return
    _COMPLEX64_COND_CHECKED = True
    limit = float(defaults.get("solver.precision.complex64_cond_warn"))
    cond = estimate_condition(fac)
    if not math.isfinite(cond) or cond <= limit:
        return
    _log.warning(
        "solve_power_flow: complex64 on a system with an estimated condition number of "
        "%.1e (above the %.0e threshold) keeps only about %.1f significant digits — an "
        "SI-unit feeder's admittance is ill-conditioned because the engine carries no "
        "per-unit normalisation. Solve at dtype=torch.complex128, or at "
        "dtype=torch.complex128 with precision='mixed' (single-precision factorization "
        "refined against double-precision residuals) for complex128 accuracy at "
        "single-precision solve cost, and store complex64.",
        cond,
        limit,
        max(0.0, 7.0 - math.log10(max(cond, 1.0))),
    )


class _PuConvergence:
    """The per-unit convergence test of the nonlinear power flow.

    The engine solves in SI units, so every convergence measure is normalised before
    it meets a tolerance. Per scenario, with free (non-slack) rows ``f``:

    - PRIMARY, the apparent-power mismatch ``max_f |V_f conj(F_f)| / S_base`` with
      ``F = Y_eff V + I_device(V) - I_slack``. This is the quantity pandapower
      (``tolerance_mva`` on a 1 MVA base) and power-grid-model converge on, so an
      iteration count is comparable across the three tools.
    - SECONDARY, the voltage update ``max_rows |ΔV| / V_LN(node)``, the per-row form
      of power-grid-model's voltage criterion.

    Both must hold. Per-row normalisation makes each measure independent of the
    voltage level and of the number of rows, so a multi-voltage grid, and an ensemble
    of grids solved as one block-diagonal system, are judged exactly like a single
    feeder (an absolute norm over the concatenated state is not).

    The mismatch threshold is per row, ``max(tol_mismatch_pu, floor · s_scale_row)``
    with ``s_scale_row`` the row's cancellation scale (:func:`_abs_row_scale`, built
    from the rated voltages, so it is a property of the network and not of the
    iterate) and ``floor`` the working precision's relative resolution
    (:func:`_rel_convergence_floor`). Without it a tolerance tighter than the
    cancellation noise of a stiff node — a 20 kV source behind a 1e-4 Ohm impedance
    reaches ~1e-9 pu at complex128 — would be unreachable and the solve would run to
    ``max_iter`` at a converged voltage.
    """

    def __init__(
        self,
        *,
        v_base: Tensor,
        s_base: float,
        tol_mismatch_pu: float,
        tol_update_pu: float,
        floor_update: float,
        floor_mismatch: float,
        fixed_rows: Optional[Tensor],
        y_eff,
        n: int,
        device,
        rdt: torch.dtype,
        warn: bool = True,
    ) -> None:
        self.v_base = v_base.clamp_min(1e-12)  # [N] line-to-neutral base per row
        self.s_base = float(s_base)
        self.tol_mismatch_pu = float(tol_mismatch_pu)
        self.tol_update_pu = float(tol_update_pu)
        self.floor_update = float(floor_update)
        self.floor_mismatch = float(floor_mismatch)
        free = torch.ones(n, dtype=rdt, device=device)
        if fixed_rows is not None and fixed_rows.numel() > 0:
            # Slack rows absorb mismatch by construction; they carry no equation.
            free = free.index_fill(0, fixed_rows.to(device), 0.0)
        self.free = free
        with torch.no_grad():
            s_scale_pu = (
                self.v_base * _abs_row_scale(y_eff, self.v_base)
            ) / self.s_base  # [*b, N]
            self.thr_mismatch = torch.clamp(
                self.floor_mismatch * s_scale_pu, min=self.tol_mismatch_pu
            )
            self.thr_update = max(self.tol_update_pu, self.floor_update)
            if warn:
                self._warn_unreachable()

    def _warn_unreachable(self) -> None:
        """Report a tolerance the working precision cannot resolve (the floor governs)."""
        floor_m = float((self.thr_mismatch * self.free).max())
        if floor_m > self.tol_mismatch_pu:
            _log.warning(
                "solve_power_flow: the power-mismatch tolerance %.1e pu is below the "
                "precision floor of this system (~%.1e pu, set by the cancellation "
                "scale of Y·V at the working precision); the floor governs "
                "convergence. Use complex128 (or precision='mixed') for a tighter "
                "tolerance.",
                self.tol_mismatch_pu,
                floor_m,
            )
        if self.tol_update_pu < self.floor_update:
            _log.warning(
                "solve_power_flow: the voltage-update tolerance %.1e pu is below the "
                "%.1e relative precision floor of the working dtype / linear-solver "
                "backend; the floor governs convergence. Use complex128 (or "
                "precision='mixed') for a tighter tolerance.",
                self.tol_update_pu,
                self.floor_update,
            )

    def mismatch_rows_pu(self, v: Tensor, fc: Tensor) -> Tensor:
        """Per-row apparent-power mismatch in per unit ``[*b, N]`` (slack rows 0)."""
        return (v.abs() * fc.abs()) * self.free / self.s_base

    def update_rows_pu(self, dv: Tensor) -> Tensor:
        """Per-row voltage update in per unit ``[*b, N]``."""
        return dv.abs() / self.v_base

    def check(
        self, mism_rows: Tensor, upd_rows: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """``(converged_mask, mismatch_max_pu, update_max_pu)``, all per scenario ``[*b]``.

        Each row meets its OWN mismatch threshold (the per-row precision floor), while
        the reported maxima are plain maxima — so a floor-limited row can leave
        ``mismatch_max_pu`` above the requested tolerance on a converged solve, which
        is the honest reading of what the precision allows.
        """
        ok = (mism_rows <= self.thr_mismatch).all(dim=-1) & (
            upd_rows <= self.thr_update
        ).all(dim=-1)
        return ok, mism_rows.amax(dim=-1), upd_rows.amax(dim=-1)


def _validate_block_solver(
    linear_solver: str,
    block_rows: Optional[Sequence[Tensor]],
    *,
    have_system: bool = False,
) -> None:
    """Guard the explicit block-diagonal opt-in (``"auto"`` never selects it).

    The block backend factors each independent sub-grid of a block-diagonal system
    on its own, which is only correct for the caller-supplied row partition — so the
    two arguments must be given together.
    """
    if block_rows is not None and linear_solver != "block":
        raise InputError(
            "block_rows is used only by linear_solver='block' (the block-diagonal "
            f"factorization of an independent-grid ensemble); got {linear_solver!r}."
        )
    if linear_solver == "block" and block_rows is None and not have_system:
        raise InputError(
            "linear_solver='block' needs block_rows=...: one row-index tensor per "
            "independent sub-grid, together partitioning the node-phase rows "
            "(pgml.multigrid.MergedGrid.block_rows() for a merged ensemble)."
        )


def _operating_point_batch_size(operating_point: Optional[dict]) -> int:
    """Leading scenario-batch length carried by a batched ``operating_point`` (else 1).

    Scans every override value (totals and per-phase lists); a value with a leading
    dim > 1 marks a batched scenario sweep.
    """
    if not operating_point:
        return 1
    b = 1
    for entry in operating_point.values():
        for val in entry.values():
            items = val if isinstance(val, (list, tuple)) else (val,)
            for x in items:
                if isinstance(x, Tensor) and x.ndim >= 1 and x.shape[0] > 1:
                    b = max(b, int(x.shape[0]))
    return b


def _slice_operating_point(operating_point: dict, i: int) -> dict:
    """The single-scenario ``operating_point`` at batch index ``i`` (grad-preserving)."""

    def slc(x):
        if isinstance(x, (list, tuple)):
            return type(x)(slc(e) for e in x)
        if isinstance(x, Tensor) and x.ndim >= 1 and x.shape[0] > 1:
            return x[i]
        return x

    return {
        cid: {k: slc(v) for k, v in entry.items()}
        for cid, entry in operating_point.items()
    }


def check_connectivity(grid: Grid) -> None:
    """Raise :class:`~pgml.errors.ConnectivityError` if any row cannot reach a source.

    The pre-solve structural gate: a (node, phase) row with no galvanic path to an
    in-service :class:`~pgml.schemas.grid_schema.Source` (an open switch or an
    out-of-service line / transformer on the only path, or no source at all) makes
    the nodal system singular there, so the solve is refused up front with the
    disconnected nodes and the concrete fixes named
    (:func:`pgml.topology.connectivity_report`).
    """
    report = connectivity_report(grid)
    if not report.connected:
        raise ConnectivityError(
            report.describe(),
            unenergized_nodes=tuple(n for n, _ in report.unenergized),
            islands=report.islands,
            reconnectable=report.reconnectable,
        )


def _zero_series_impedance_branches(grid: Grid) -> list[tuple[int, str, str]]:
    """Branches whose series impedance is exactly zero: ``(id, component, detail)``.

    Such a branch has no primitive admittance — the per-branch impedance matrix is
    singular, so the stamp cannot be formed. It is a common idiom in published network
    data (a bus coupler, a jumper, a zero-length line, an ideal closed switch), which is
    why it is reported by branch id rather than surfacing as a linear-algebra failure
    deep in the assembly. Values are read under ``no_grad`` (a structural check, never
    on the autograd tape) and a branch whose impedance comes from its conductor
    geometry is skipped — the geometry path always yields a finite impedance.
    """
    from pgml.schemas.grid_schema import GenericBranch, Line, Switch, Transformer

    def zero(*vals) -> bool:
        with torch.no_grad():
            for v in vals:
                if v is None:
                    continue
                t = torch.as_tensor(v, dtype=torch.float64)
                if t.numel() and float(t.abs().max()) != 0.0:
                    return False
        return True

    out: list[tuple[int, str, str]] = []
    for b in grid.branches:
        if not getattr(b, "in_service", True):
            continue
        if isinstance(b, Line):
            if b.conductor_geometry is not None:
                continue
            if zero(b.series_resistance_ohm_per_m, b.series_inductance_h_per_m):
                out.append((int(b.id), "line", "R and L per metre are both zero"))
            elif zero(b.length_m):
                out.append((int(b.id), "line", "length_m is zero"))
        elif isinstance(b, Switch):
            if b.closed and zero(b.resistance_ohm, b.inductance_h):
                out.append(
                    (int(b.id), "switch", "closed with zero resistance and inductance")
                )
        elif isinstance(b, GenericBranch):
            if zero(b.series_resistance_ohm, b.series_inductance_h):
                out.append(
                    (int(b.id), "generic_branch", "series R and L are both zero")
                )
        elif isinstance(b, Transformer):
            if zero(b.series_resistance_ohm, b.series_inductance_h):
                out.append(
                    (int(b.id), "transformer", "series (leakage) R and L are both zero")
                )
    return out


def check_branch_impedances(grid: Grid) -> None:
    """Raise :class:`~pgml.errors.ModelingError` for a branch with zero series impedance.

    The pre-solve modeling gate that keeps a singular primitive stamp from surfacing as
    a linear-algebra failure naming an internal batch index: a branch with no series
    impedance (a bus coupler or jumper modelled as a zero-impedance line, a zero-length
    line, an ideal closed switch) cannot be inverted into a primitive admittance. The
    error names every offending branch and the two ways out — give it the documented
    near-ideal resistance (``branch.near_ideal_series_resistance_ohm``), which is what
    the pandapower converter substitutes for a bus-bus switch, or merge its two nodes.
    """
    bad = _zero_series_impedance_branches(grid)
    if not bad:
        return
    r_ideal = float(defaults.get("branch.near_ideal_series_resistance_ohm"))
    shown = "; ".join(f"{kind} {bid} ({why})" for bid, kind, why in bad[:8])
    more = "" if len(bad) <= 8 else f" (+{len(bad) - 8} more)"
    raise ModelingError(
        f"{len(bad)} branch(es) have zero series impedance and therefore no primitive "
        f"admittance: {shown}{more}. A zero-impedance branch is a bus coupler or "
        "jumper, which the nodal formulation cannot stamp as a pi-branch. Give it a "
        f"small finite series resistance (the documented near-ideal value is "
        f"{r_ideal:g} Ohm, what the pandapower converter substitutes for a bus-bus "
        "switch), or merge the two nodes it joins into one."
    )


def _same_device(a: torch.device, b: torch.device) -> bool:
    """Device equality with an unindexed spec matching any index of its type.

    A tensor's device always carries an index (``cuda:0``) while a caller-supplied
    ``torch.device("cuda")`` does not; strict ``==`` would reject that pair even
    though they resolve to the same accelerator.
    """
    if a.type != b.type:
        return False
    return a.index is None or b.index is None or a.index == b.index


def _branch_states_batched(branch_states: Optional[dict]) -> bool:
    """``True`` iff any state carries a leading scenario dim."""
    return bool(branch_states) and any(
        isinstance(v, Tensor) and v.ndim >= 1 and v.numel() > 1
        for v in branch_states.values()
    )


def _check_connectivity_with_states(grid: Grid, branch_states: dict) -> None:
    """Per-scenario connectivity for masked branch states (raises on any dead row).

    The static grid is condensed once: rows are merged over every conducting
    branch NOT listed in ``branch_states`` (union-find, as in
    :func:`pgml.topology.connectivity_report`), giving ``C`` static components of
    which some hold a source. Each masked branch is then a component-level edge
    that conducts where its state is non-zero, so per-scenario energization is a
    boolean propagation over the tiny condensed graph — vectorized over the whole
    scenario batch ``[B, C]`` instead of a per-scenario python search.
    """
    from pgml.topology import _UnionFind, _branch_rows, _conducting

    uf = _UnionFind()
    for node in grid.nodes:
        for ph in node.phases:
            uf.find((int(node.id), ph))
    masked_ids = set(branch_states.keys())
    for b in grid.branches:
        if int(b.id) in masked_ids or not _conducting(b):
            continue
        rows = _branch_rows(b)
        for r in rows[1:]:
            uf.union(rows[0], r)

    roots: dict = {}
    comp_nodes: dict[int, set] = {}
    for node in grid.nodes:
        for ph in node.phases:
            root = uf.find((int(node.id), ph))
            cid = roots.setdefault(root, len(roots))
            comp_nodes.setdefault(cid, set()).add(int(node.id))
    n_comp = len(roots)

    energized0 = torch.zeros(n_comp, dtype=torch.bool)
    for a in grid.appliances:
        if isinstance(a, Source) and getattr(a, "in_service", True):
            for ph in a.phases:
                energized0[roots[uf.find((int(a.node), ph))]] = True

    # Component-level edges of the masked branches, each carrying its state.
    edges_u, edges_v, states = [], [], []
    branch_by_id = {int(b.id): b for b in grid.branches}
    for bid, sval in branch_states.items():
        b = branch_by_id.get(int(bid))
        if b is None:
            raise InputError(f"branch_states references unknown branch id {bid}.")
        comps = {roots[uf.find(r)] for r in _branch_rows(b)}
        comps = sorted(comps)
        s = (
            sval.detach().reshape(-1)
            if isinstance(sval, Tensor)
            else torch.tensor([float(sval)])
        )
        for other in comps[1:]:
            edges_u.append(comps[0])
            edges_v.append(other)
            states.append(s)

    b_size = max((int(s.numel()) for s in states), default=1)
    energized = energized0.expand(b_size, n_comp).clone()  # [B, C]
    if edges_u:
        # A masked branch whose terminals share one static component adds no
        # edge — the component graph already contains it.
        active = torch.stack(
            [(s != 0).expand(b_size).clone() for s in states], dim=-1
        )  # [B, E]
        u_idx = torch.tensor(edges_u, dtype=torch.int64)
        v_idx = torch.tensor(edges_v, dtype=torch.int64)
        for _ in range(len(edges_u) + 1):
            e_u = energized.index_select(1, u_idx)  # [B, E]
            e_v = energized.index_select(1, v_idx)
            # Boolean OR-accumulate along the component axis (int add + clamp:
            # duplicate edge targets accumulate, then saturate to a bool).
            new = energized.to(torch.int32)
            new.index_add_(1, u_idx, (e_v & active).to(torch.int32))
            new.index_add_(1, v_idx, (e_u & active).to(torch.int32))
            new = new.clamp_(max=1).bool()
            if bool((new == energized).all()):
                break
            energized = new

    dead = ~energized  # [B, C]
    if not bool(dead.any()):
        return
    dead_scen = dead.any(dim=1)  # [B]
    scen_idx = torch.nonzero(dead_scen).reshape(-1).tolist()
    worst = int(torch.nonzero(dead_scen).reshape(-1)[0])
    dead_nodes = sorted(
        {
            n
            for c in torch.nonzero(dead[worst]).reshape(-1).tolist()
            for n in comp_nodes[c]
        }
    )
    shown = ", ".join(str(i) for i in scen_idx[:10])
    more = "" if len(scen_idx) <= 10 else f", … (+{len(scen_idx) - 10} more)"
    node_str = ", ".join(str(n) for n in dead_nodes[:12]) + (
        "" if len(dead_nodes) <= 12 else ", …"
    )
    raise ConnectivityError(
        f"branch_states disconnect part of the grid in {len(scen_idx)} of {b_size} "
        f"scenario(s) (indices [{shown}{more}]); e.g. scenario {worst} leaves "
        f"node(s) {node_str} with no path to a source. Keep every scenario's "
        "non-zero states spanning the grid (a state of exactly 0 opens the "
        'branch), drop the offending scenarios, or pass on_disconnected="ignore" '
        "and filter via converged_mask.",
        unenergized_nodes=tuple(dead_nodes),
    )


def _apply_scalar_states(grid: Grid, branch_states: dict) -> Grid:
    """A grid copy whose static flags realize UNBATCHED ``branch_states`` (0 = open).

    Used only for the pre-solve connectivity REPORT of a single masked
    configuration, so the rich :func:`pgml.topology.connectivity_report`
    diagnostics (islands, reconnect hints) apply unchanged.
    """
    conducting = {}
    for bid, sval in branch_states.items():
        s = (
            float(sval.detach().reshape(()).item())
            if isinstance(sval, Tensor)
            else float(sval)
        )
        conducting[int(bid)] = s != 0.0
    branches = []
    for b in grid.branches:
        if int(b.id) in conducting:
            on = conducting[int(b.id)]
            update = {"in_service": on}
            if hasattr(b, "closed"):  # a Switch: the state replaces closed too
                update = {"in_service": True, "closed": on} if on else update
            b = b.model_copy(update=update)
        branches.append(b)
    return grid.model_copy(update={"branches": branches})


def _expand_zeroed_result(grid: Grid, res: PowerFlowResult) -> PowerFlowResult:
    """Scatter a sub-grid solution back to the full grid with 0 V on dropped rows.

    The ``on_disconnected="zero"`` reassembly: ``res`` was solved on
    :func:`pgml.topology.energized_subgrid`; every full-grid row absent from the
    sub-grid is a de-energized conductor and reports 0 V. Out-of-place
    ``index_copy`` so gradients keep flowing into the solved rows.
    """
    full_index = node_phase_index(grid)
    sub_index = res.index
    rows = torch.as_tensor(
        [
            full_index.row(sub_index.node_id_of(r), sub_index.phase_of(r))
            for r in range(sub_index.size)
        ],
        dtype=torch.int64,
        device=res.v.device,
    )
    v_full = torch.zeros(
        (*res.v.shape[:-1], full_index.size), dtype=res.v.dtype, device=res.v.device
    ).index_copy(-1, rows, res.v)
    return PowerFlowResult(
        v=v_full,
        index=full_index,
        iterations=res.iterations,
        residual=res.residual,
        converged=res.converged,
        diagnostics=res.diagnostics,
        converged_mask=res.converged_mask,
        failed_states=res.failed_states,
    )


def _only_appliances(grid: Grid, kind) -> Grid:
    """A grid copy in which only appliances of ``kind`` (plus Sources) are in service.

    The injection-plan split behind a load-only λ ramp: taking the other injecting
    devices out of service drops them from :func:`pgml.assembly.build_injection_plan`
    while every resolution rule (per-phase split, ZIP law, inverter control) applies
    unchanged to the ones that remain. Sources stay in service — they are the boundary,
    not an injection.
    """
    kept = [
        a
        if (isinstance(a, (kind, Source)) or not isinstance(a, InjectionAppliance))
        else a.model_copy(update={"in_service": False})
        for a in grid.appliances
    ]
    return grid.model_copy(update={"appliances": kept})


def _without_appliances(grid: Grid, kind) -> Grid:
    """A grid copy in which appliances of ``kind`` are taken out of service.

    The complement of :func:`_only_appliances` (the un-ramped half of a load-only λ
    ramp: generation and storage at their nameplate values).
    """
    kept = [
        a.model_copy(update={"in_service": False}) if isinstance(a, kind) else a
        for a in grid.appliances
    ]
    return grid.model_copy(update={"appliances": kept})


def _resolve_failed_states(mask: Optional[Tensor]) -> tuple[Optional[Tensor], tuple]:
    """``(converged_mask, failed_indices)`` for a result; unbatched -> ``(None, ())``."""
    if mask is None or mask.ndim == 0:
        return None, ()
    flat = mask.reshape(-1)
    failed = tuple(i for i, ok in enumerate(flat.tolist()) if not ok)
    return mask, failed


@dataclass(frozen=True)
class PowerFlowResult:
    """Converged fundamental power-flow solution.

    Attributes
    ----------
    v:
        Complex node voltages ``[*batch, N]`` (DIFFERENTIABLE through the IFT path).
    index:
        The compact :class:`NodePhaseIndex` describing ``v``'s row layout.
    iterations:
        Number of fixed-point / Newton iterations performed (python int).
    residual:
        Real scalar tensor: the achieved value of the PRIMARY convergence criterion,
        the largest nodal apparent-power mismatch in per unit of ``s_base_va``
        (max over batch). A floor-limited row can leave it above the requested ``tol``
        on a converged solve — see :class:`ConvergenceDiagnostics`, which also reports
        the per-unit voltage update and both SI counterparts.
    converged:
        ``True`` if EVERY batch element met BOTH per-unit criteria (power mismatch and
        voltage update, each capped by the working precision's floor) within
        ``max_iter``.
    diagnostics:
        Non-fatal :class:`ConvergenceDiagnostics` (per-node physical mismatch,
        voltage-band offenders, residual history, worst offenders, likely cause, and
        — when the solve did not converge — an IFT-Jacobian criticality analysis).
    converged_mask:
        Per-scenario convergence flags ``[*batch]`` (bool), or ``None`` for an
        unbatched solve. A batched solve does NOT raise on a failed element — every
        element's best-effort ``V`` is returned and the failures are listed here and
        in :attr:`failed_states`.
    failed_states:
        Flat indices of the batch elements that did NOT converge (empty when all
        converged or unbatched). The companion log record names them with details.
    regulation:
        :class:`VoltageRegulationResult` when the grid carries voltage-regulating
        generators (PV terminals), else ``None``: the solved reactive injection of
        each regulating generator and which of them ended up pinned at a reactive
        limit.
    """

    v: Tensor
    index: NodePhaseIndex
    iterations: int
    residual: Tensor
    converged: bool
    diagnostics: Optional[ConvergenceDiagnostics] = None
    converged_mask: Optional[Tensor] = None
    failed_states: tuple[int, ...] = ()
    regulation: Optional[VoltageRegulationResult] = None


@dataclass(frozen=True)
class VoltageRegulationResult:
    """Solved state of the voltage-regulating generators (the PV terminals).

    Attributes
    ----------
    q_var:
        Generator id -> solved TOTAL reactive injection ``[*batch]`` [var], in the
        generator convention (positive = injected). Recovered from the converged
        residual (``Q = Q_pinned - Im(conj(V) F_c)`` summed over the unit's phases);
        autograd-free, like :class:`ConvergenceDiagnostics`.
    regulating:
        Generator id -> bool ``[*batch]``: ``True`` where the terminal holds its
        voltage setpoint, ``False`` where a reactive limit binds and the unit was
        solved as a PQ injection pinned at that limit.
    switch_rounds:
        Number of PV-to-PQ switching rounds performed beyond the first solve (0 when
        no limit bound or enforcement is off).
    enforce_q_limits:
        Whether reactive limits were enforced in this solve.
    """

    q_var: dict[int, Tensor]
    regulating: dict[int, Tensor]
    switch_rounds: int
    enforce_q_limits: bool


@dataclass
class ConvergenceDiagnostics:
    """Structured power-flow convergence telemetry (autograd-free, computed at ``V*``).

    Cheap state diagnostics are always populated; ``criticality`` is filled only when
    the solve did not converge (it costs a dense Jacobian + SVD).

    Both convergence criteria are reported in PER UNIT — the values the solve is judged
    on — and in SI units under explicitly named fields:

    - ``mismatch_max_pu`` / ``mismatch_max_va``: the PRIMARY criterion, the largest nodal
      apparent-power mismatch over the free (non-slack) rows, per unit of ``s_base_va``
      and in volt-amperes.
    - ``mismatch_max_a``: the same residual as a CURRENT, ``max |F_c|`` in amperes
      (``F = Y_eff V + I_device(V) - I_slack``) — what the nodal equations balance.
    - ``update_max_pu`` / ``update_norm_v``: the SECONDARY criterion, the largest per-row
      voltage update per unit of the node's line-to-neutral rated voltage, and the
      two-norm of the same update in volts.

    Voltages (``v_pu``, ``voltage_band_pu``) are per unit on each node's line-to-neutral
    base.
    """

    converged: bool
    iterations: int
    mismatch_max_pu: float  # PRIMARY: max |V·conj(F)| / s_base_va over free rows [pu]
    update_max_pu: float  # SECONDARY: max |ΔV| / V_LN(node) [pu]
    mismatch_max_va: float  # max |V·conj(F)| over free rows [VA]
    mismatch_max_a: float  # max |F_c| over free rows [A]
    update_norm_v: float  # ||ΔV||_2 of the final update [V]
    s_base_va: float  # power base of the per-unit mismatch [VA]
    voltage_band_pu: tuple[float, float]
    residual_history: list[float] = field(
        default_factory=list
    )  # update_max_pu per iter
    worst_nodes: list[dict] = field(default_factory=list)  # top-k by power mismatch
    out_of_band_nodes: list[dict] = field(default_factory=list)  # |V| outside the band
    likely_cause: str = ""
    criticality: Optional[dict] = None  # IFT-Jacobian analysis (non-convergence only)

    def as_dict(self) -> dict:
        """Plain-dict view (e.g. for :attr:`ConvergenceError.diagnostics`)."""
        return asdict(self)


@dataclass
class LoadabilityResult:
    """λ-ramp loadability analysis — how far the loading scales, and what limits it.

    Scales the injections by ``λ`` (``λ=1`` = the grid's nameplate loading) from a
    feasible base, Newton-corrects at each step, and bisects onto the first ``λ`` the
    corrector can no longer solve.

    ``breaking_lambda`` is therefore the largest ``λ`` at which the Newton corrector
    still CONVERGES, which is a LOWER BOUND on the true P-V nose: a plain corrector
    fails slightly before the singularity, and the gap depends on the corrector's
    tolerance and iteration budget (~4 % on a two-bus feeder whose nose is known in
    closed form). The Jacobian figures (``min_singular_value``, ``condition_number``,
    ``critical_nodes``, ``limiting_loads``, ``nose_voltage_min_pu``) describe that last
    converged point — close to the nose, not the singular point itself. An arc-length
    predictor-corrector continuation, which can turn the nose, is open work.

    ``ramp`` records WHAT ``λ`` multiplied: ``"all"`` (loads and generators / storage
    together, the default) or ``"load"`` (loads only, generation at nameplate — the
    textbook continuation-power-flow ramp).

    When every ramp step up to ``lambda_max`` converges, no limit exists inside the
    ramp: ``capped=True`` and ``breaking_lambda`` (= ``lambda_max``) only says the
    loading scales at least that far — raise ``lambda_max`` to find the limit.
    """

    breaking_lambda: float  # largest λ whose Newton corrector converged (lower bound)
    feasible: bool  # λ* >= 1 -> the nameplate loading solves
    margin: (
        float  # λ* − 1 (headroom above nameplate; negative = infeasible at nameplate)
    )
    nose_voltage_min_pu: float  # lowest |V|/V_LN at the last converged λ
    capped: bool = False  # ramp reached lambda_max without a limit; λ* is a LOWER BOUND
    critical_nodes: list[dict] = field(default_factory=list)  # voltage-collapse mode
    limiting_loads: list[dict] = field(
        default_factory=list
    )  # margin-limiting injections
    min_singular_value: float = 0.0
    condition_number: float = 0.0
    converged_lambdas: list[float] = field(
        default_factory=list
    )  # the λ trace (plotting)
    corrector_iterations: int = 0  # total Newton iterations across the ramp
    ramp: str = "all"  # what λ multiplied: "all" devices or "load" only

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# leaf discovery (tensor-duality + overrides + slack voltage)
# ---------------------------------------------------------------------------
def _tensor_leaves(t: Tensor, out: list, seen: set) -> None:
    """Append the distinct autograd LEAVES reachable from ``t`` (dedup by id).

    A physical Grid field may hold a plain leaf OR a derived expression (float/tensor
    duality: one leaf ``p`` can drive both ``p_nom_w=p`` and ``q_nom_var=p*k``). The IFT
    backward must attach parameter gradients at the true LEAVES, not the derived
    intermediates: if two fields share a leaf, differentiating the residual w.r.t. the
    intermediates would count that leaf once through the intermediate AND again when the
    outer autograd engine walks the intermediate's history — a silent double count. Leaves
    have no history, so differentiating w.r.t. them is unambiguous and the outer engine
    connects to them directly. A leaf ``t`` is appended as itself; a non-leaf is resolved
    by walking its ``grad_fn`` graph to the ``AccumulateGrad`` nodes (whose ``.variable`` is
    the leaf tensor). Detached / no-grad branches are naturally excluded.
    """
    if t.is_leaf:
        if t.requires_grad and id(t) not in seen:
            seen.add(id(t))
            out.append(t)
        return
    grad_fn = t.grad_fn
    if grad_fn is None:
        return
    stack = [grad_fn]
    # Dedup by id while holding a STRONG reference to every visited node: the
    # Python wrappers yielded by ``next_functions`` are transient, so a freed
    # wrapper's address can be reused by a not-yet-visited node -- an id-only
    # set would then skip it and silently drop the leaves behind it (deep
    # graphs, e.g. a neural network driving an operating point).
    fn_seen: dict = {}
    while stack:
        fn = stack.pop()
        if id(fn) in fn_seen:
            continue
        fn_seen[id(fn)] = fn
        var = getattr(fn, "variable", None)  # AccumulateGrad -> the leaf tensor
        if var is not None:
            if var.requires_grad and id(var) not in seen:
                seen.add(id(var))
                out.append(var)
            continue
        for nxt, _ in fn.next_functions:
            if nxt is not None:
                stack.append(nxt)


def _collect_leaves(obj, out: list, seen: set) -> None:
    """Recursively collect the distinct autograd leaves reachable from ``obj``.

    Tensor fields (leaf or derived) are resolved to their true leaves via
    :func:`_tensor_leaves`; containers and pydantic models are walked structurally.
    """
    if isinstance(obj, Tensor):
        if obj.requires_grad:
            _tensor_leaves(obj, out, seen)
        return
    if isinstance(obj, (list, tuple)):
        for el in obj:
            _collect_leaves(el, out, seen)
        return
    if isinstance(obj, dict):
        for el in obj.values():
            _collect_leaves(el, out, seen)
        return
    fields = getattr(type(obj), "model_fields", None)
    if fields is not None:
        for name in fields:
            _collect_leaves(getattr(obj, name), out, seen)


def _grid_param_leaves(
    grid: Grid,
    param_overrides: Optional[dict],
    v_fixed: Optional[Tensor],
    operating_point: Optional[dict] = None,
    branch_states: Optional[dict] = None,
) -> list[Tensor]:
    """All distinct autograd leaves the residual depends on (deterministic order).

    Physical fields may be plain leaves or derived expressions (float/tensor duality);
    both are resolved to their true leaves (see :func:`_tensor_leaves`), so a single leaf
    feeding several fields is captured exactly once and its gradient is not double counted.
    ``operating_point`` entries (per-appliance ``p_w``/``q_var``, possibly batched)
    enter the residual exactly like grid fields, so their leaves are collected too: a
    differentiable operating point — e.g. the output of a neural network — receives
    gradients through the IFT backward. The same holds for ``branch_states``
    (continuous switch / topology states).
    """
    out: list[Tensor] = []
    seen: set = set()
    for node in grid.nodes:
        _collect_leaves(node, out, seen)
    for b in grid.branches:
        _collect_leaves(b, out, seen)
    for a in grid.appliances:
        _collect_leaves(a, out, seen)
    if param_overrides is not None:
        for val in param_overrides.values():
            _collect_leaves(val, out, seen)
    if operating_point is not None:
        _collect_leaves(operating_point, out, seen)
    if branch_states is not None:
        # Continuous switch/topology states are parameters like any other: their
        # leaves receive gradients through the same IFT backward.
        _collect_leaves(branch_states, out, seen)
    if v_fixed is not None and isinstance(v_fixed, Tensor) and v_fixed.requires_grad:
        _tensor_leaves(v_fixed, out, seen)
    return out


# ---------------------------------------------------------------------------
# slack handling
# ---------------------------------------------------------------------------
def _has_uref_scale(operating_point: Optional[dict]) -> bool:
    """True if any operating-point entry carries a per-source ``u_ref_scale``.

    A source-voltage scenario spec (:class:`pgml.scenarios.ParameterSpec` with
    ``field="u_ref"``) writes ``{source_id: {"u_ref_scale": Tensor[*b]}}`` — a
    per-scenario multiplier on the ideal-slack reference. When present, the slack
    reference must be recomputed per scenario rather than read from the cached
    (operating-point-independent) prepared system.
    """
    if not operating_point:
        return False
    return any(
        isinstance(entry, dict) and "u_ref_scale" in entry
        for entry in operating_point.values()
    )


def _slack_rows_and_vref(
    grid: Grid,
    index: NodePhaseIndex,
    rdt: torch.dtype,
    cdt: torch.dtype,
    device,
    operating_point: Optional[dict] = None,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Ideal-slack fixed rows + reference voltages ``u_ref∠u_angle`` at them.

    Returns ``(fixed_rows[int64, S], v_fixed[complex, ...S])`` from in-service
    sources, or ``(None, None)`` if there is no source. ``v_fixed`` stays
    differentiable when ``u_ref``/``u_angle`` are tensors (tensor duality).

    A per-source ``operating_point[source_id]["u_ref_scale"]`` (a per-scenario
    multiplier on ``u_ref_v``) scales that source's reference magnitude while keeping
    its angle — the batched fundamental boundary of a source-voltage scenario sweep.
    A batched scale promotes ``v_fixed`` to ``[*batch, S]`` (the leading scenario dims
    the Schur solve pins per row); gradients flow to the scale leaf via the residual.
    """
    sources = [a for a in grid.appliances if isinstance(a, Source) and a.in_service]
    if not sources:
        return None, None
    rows: list[int] = []
    vref: list[Tensor] = []
    for s in sources:
        u_ref = torch.as_tensor(s.u_ref_v, dtype=rdt, device=device)
        ang = torch.as_tensor(s.u_angle_deg, dtype=rdt, device=device) * (
            math.pi / 180.0
        )
        vth = torch.polar(u_ref, ang).to(cdt)  # [P]
        scale = None
        if operating_point is not None:
            entry = operating_point.get(s.id)
            if isinstance(entry, dict) and "u_ref_scale" in entry:
                # as_tensor keeps the autograd history (tensor duality), so gradients
                # flow to a differentiable u_ref_scale; complex cast scales magnitude only.
                scale = torch.as_tensor(
                    entry["u_ref_scale"], dtype=rdt, device=device
                ).to(cdt)
        for j, ph in enumerate(s.phases):
            rows.append(index.row(s.node, ph))
            vref.append(vth[j] if scale is None else vth[j] * scale)
    fixed_rows = torch.as_tensor(rows, dtype=torch.int64, device=device)
    # A scaled source contributes a ``[*batch]`` entry; broadcast every entry to the
    # common leading shape and stack on a new LAST axis -> ``[*batch, S]`` (``[S]`` when
    # no scale is batched, matching the historical shape).
    vref = list(torch.broadcast_tensors(*vref)) if len(vref) > 1 else vref
    v_fixed = torch.stack(vref, dim=-1)  # [*batch, S]
    return fixed_rows, v_fixed


# ---------------------------------------------------------------------------
# system builders (used by both the forward fixed point and the IFT backward)
# ---------------------------------------------------------------------------
def _apply_y(y_eff, v: Tensor) -> Tensor:
    """``Y @ V`` over the scenario batch, reading a SHARED ``Y`` exactly once.

    When every leading dim of ``y_eff`` is singleton (one network shared by the
    whole batch — the usual case), a broadcast ``matmul`` against ``[..., N, 1]``
    columns degenerates into ``B`` separate matrix-vector products that re-read the
    ``N×N`` matrix per scenario (memory-bandwidth-bound: dominant at large ``N``).
    Folding the batch into the rows of ONE ``[B, N] @ [N, N]`` GEMM reads the
    matrix once. A genuinely batched ``y_eff`` (per-scenario topology) keeps the
    batched matmul — each scenario owns its matrix there. Differentiable in both.

    ``y_eff`` may also be a :class:`~pgml.solver.lowrank.LowRankOperator` — the
    per-state admittance of a Woodbury switch-state sweep, applied as the shared
    base plus each state's rank-``k`` correction instead of a materialised
    ``[B, N, N]`` tensor.
    """
    if isinstance(y_eff, LowRankOperator):
        return _apply_y(y_eff.base, v) + y_eff.correction(v)
    n = y_eff.shape[-1]
    if y_eff.reshape(-1, n, n).shape[0] == 1:
        lead = torch.broadcast_shapes(v.shape[:-1], y_eff.shape[:-2])
        v_b = v.broadcast_to(*lead, n)
        yv = torch.matmul(v_b.reshape(-1, n), y_eff.reshape(n, n).mT)
        return yv.reshape(*lead, n)
    return torch.matmul(y_eff, v.unsqueeze(-1)).squeeze(-1)


def _y_eff_and_islack(
    grid, f0, index, dtype, device, slack, param_overrides, branch_states=None
):
    """Effective admittance ``Y_eff`` ``[N,N]`` / ``[*batch,N,N]`` and slack current ``[N]``.

    ``slack="norton"``: ``Y_eff = Y_net + Y_srcNorton``, ``I_slack`` = source
    Norton current. ``slack="ideal"``: ``Y_eff = Y_net``, ``I_slack`` = 0 (slack
    rows pinned by the Schur solve in :func:`solve_harmonic`).

    The assembly's singleton frequency axis is folded away HERE — it is only known
    to be the frequency axis at this producer — so every leading dim downstream is
    a SCENARIO dim: the fixed point, Newton, and the IFT backward never have to
    guess whether a size-1 dim is the frequency axis or a genuine batch dim of one
    (an operating point batched ``[B, 1]`` carries exactly such a dim, and a
    value-based squeeze would silently mix its scenarios). Batched
    ``branch_states`` promote ``Y_eff`` to per-scenario matrices ``[*batch, N, N]``.
    """
    yb = assemble_network_ybus(
        grid,
        [f0],
        dtype=dtype,
        device=device,
        param_overrides=param_overrides,
        branch_states=branch_states,
    )
    y = yb.Y  # [1, N, N] or [*batch, 1, N, N] (batched branch states)
    has_freq_axis = y.ndim == 3  # ndim > 3: batched states, frequency axis folded next
    if y.ndim > 3:
        y = y.squeeze(-3)  # [*batch, N, N]
    if slack == "norton":
        cdt = _cdtype(dtype)
        rdt = _rdtype(dtype)
        f = torch.as_tensor([f0], dtype=rdt, device=y.device)
        y = _stamp_sources(grid, f, y, index, cdt, rdt, y.device, param_overrides)
        i_slack = build_injections(
            grid,
            [f0],
            index,
            dtype=dtype,
            device=device,
            param_overrides=param_overrides,
        ).squeeze(-2)  # [*b, 1, N] -> [*b, N] (the single-frequency axis, positionally)
    else:
        i_slack = torch.zeros(y.shape[-1], dtype=y.dtype, device=y.device)
    if has_freq_axis:
        y = y.squeeze(-3)  # [*b, 1, N, N] -> [*b, N, N]: the single-frequency axis
    return y, i_slack


def _woodbury_base_states(grid: Grid, branch_states: dict) -> dict:
    """The base configuration a switch-state sweep factors: OMIT what it can.

    A Woodbury update that ADDS admittance is numerically benign, while one that
    REMOVES a near-ideal switch multiplies that switch's voltage drop — two nearly
    equal node voltages whose difference floating point barely resolves — by its
    huge admittance, and spends the precision the fixed point needs. So the base
    leaves every switched branch OUT (state 0) wherever the base network still
    energizes every row: each branch is opened in turn and kept open while
    :func:`check_connectivity` passes. A branch that is a bridge of the base
    network stays IN it (state 1) — the sweep cannot open it either (a state of 0
    there fails the per-scenario connectivity check), and a partial downdate of a
    finite-impedance branch is well conditioned.
    """

    def connected(states: dict) -> bool:
        try:
            check_connectivity(_apply_scalar_states(grid, states))
        except ConnectivityError:
            return False
        return True

    ids = [int(bid) for bid in branch_states]
    all_open = {bid: 0.0 for bid in ids}
    if connected(all_open):
        return all_open  # the usual sweep (normally-open ties): one check
    base = {bid: 1.0 for bid in ids}
    for bid in ids:
        trial = dict(base)
        trial[bid] = 0.0
        if connected(trial):
            base = trial
    return base


def _woodbury_pieces(
    grid,
    f0,
    index,
    dtype,
    device,
    slack,
    param_overrides,
    branch_states,
    fixed_rows,
    factor_backend,
    block_rows,
    precision="full",
):
    """Base admittance, slack current and the per-state low-rank update of a sweep.

    The base network (:func:`_woodbury_base_states`) is assembled and factored
    ONCE; each state is that base plus the rank-``k`` deviation of its switched
    stamps (:func:`~pgml.solver.lowrank.branch_state_terms`). Returns the base as a
    matrix-free :class:`~pgml.solver.lowrank.LowRankOperator` (the residual /
    diagnostics never materialise the ``[B, N, N]`` per-state admittance) together
    with the state-independent slack current and the prepared
    :class:`~pgml.solver.lowrank.LowRankUpdate`.
    """
    base_states = _woodbury_base_states(grid, branch_states)
    y_base, i_slack = _y_eff_and_islack(
        grid, f0, index, dtype, device, slack, param_overrides, base_states
    )
    u, c = branch_state_terms(
        grid,
        index,
        branch_states,
        f0,
        dtype=dtype,
        device=y_base.device,
        param_overrides=param_overrides,
        base_states=base_states,
    )
    fac = lu_factor_system(
        y_base,
        fixed_rows=fixed_rows,
        backend=factor_backend,
        block_rows=block_rows,
        precision=precision,
        refine_steps=0,  # the nonlinear outer iteration IS the refinement loop
    )
    return (
        LowRankOperator(y_base, u, c, u),
        i_slack,
        low_rank_update(fac, u, c),
    )


def _validate_branch_states_method(
    branch_states_method: str, branch_states, method: str = "current_injection"
) -> bool:
    """Resolve the switch-state solve strategy; ``True`` selects the Woodbury path."""
    if branch_states_method not in ("assemble", "woodbury"):
        raise InputError(
            f"Unsupported branch_states_method {branch_states_method!r} "
            "(use 'assemble' or 'woodbury')."
        )
    if branch_states_method == "assemble":
        return False
    if not branch_states:
        raise InputError(
            'branch_states_method="woodbury" needs branch_states: it solves every '
            "state as a low-rank update of one base factorization."
        )
    if method != "current_injection":
        raise InputError(
            'branch_states_method="woodbury" applies to the current-injection '
            f"fixed point (its inner solve is the factored one); method={method!r} "
            "builds its own dense Jacobian per scenario."
        )
    return True


@dataclass(frozen=True)
class PowerFlowSystem:
    """Precomputed solve state for REPEATED solves of one grid (assembly + LU).

    Everything about the network side of a nonlinear power flow is operating-point
    INDEPENDENT: the node-phase index, the effective admittance, the slack rows and
    reference, the factorization, and the grid-side parameter leaves. A chunked
    scenario run (:func:`pgml.scenarios.run_scenarios`) or any solve-in-a-loop
    caller therefore pays assembly + factorization once via
    :func:`prepare_power_flow` and passes the system to every
    :func:`solve_power_flow` call — only the injections change between calls.

    The cached tensors are DETACHED and drive the (already detached) forward
    iteration and diagnostics; when parameter gradients are requested, the IFT
    backward rebuilds its differentiable system from the leaves as always, so
    differentiability is unchanged. The system must come from the SAME grid,
    slack, dtype, device, ``param_overrides`` and ``branch_states`` as the solve
    that consumes it. Validated per solve: slack / dtype / device / size and the
    grid's :func:`~pgml.topology.network_fingerprint` (nodes, branches, sources,
    shunts and their parameter values) — a same-size grid with changed topology or
    impedances is rejected instead of silently reusing the stale factorization.
    ``param_overrides`` / ``branch_states`` equality remains the caller's contract.

    With ``branch_states_method="woodbury"`` the cached system describes the sweep's
    BASE network instead: ``y_eff`` is a matrix-free
    :class:`~pgml.solver.lowrank.LowRankOperator` and ``factorization`` a
    :class:`~pgml.solver.lowrank.LowRankUpdate`, so a consuming solve must request
    the same method.
    """

    index: NodePhaseIndex
    f0: float
    slack: str
    y_eff: Any  # detached [*, N, N] (a LowRankOperator on the woodbury path)
    i_slack: Tensor  # detached [1, N] (norton) or [N] (ideal)
    fixed_rows: Optional[Tensor]
    v_fixed: Optional[Tensor]  # detached slack reference
    factorization: object  # FactoredSystem of y_eff (or its LowRankUpdate)
    static_leaves: tuple[Tensor, ...]  # grid + overrides + states leaves
    network_fp: str = ""  # network_fingerprint(grid) at prepare time
    precision: str = "full"  # working precision of the cached factorization


def prepare_power_flow(
    grid: Grid,
    *,
    slack: str = "ideal",
    dtype: torch.dtype = torch.complex128,
    precision: str = "full",
    device: Optional[torch.device] = None,
    param_overrides: Optional[dict] = None,
    branch_states: Optional[dict] = None,
    branch_states_method: str = "assemble",
    linear_solver: str = "auto",
    block_rows: Optional[Sequence[Tensor]] = None,
) -> PowerFlowSystem:
    """Assemble + factor the operating-point-independent power-flow system once.

    Runs the connectivity check (raising
    :class:`~pgml.errors.ConnectivityError` like :func:`solve_power_flow` with
    ``on_disconnected="raise"``), assembles ``Y_eff`` and the slack quantities,
    and factors ``Y_eff`` with the selected backend
    (:func:`pgml.solver.harmonic.lu_factor_system`; ``linear_solver``,
    ``block_rows`` and ``branch_states_method`` as in :func:`solve_power_flow`).
    Pass the result as ``solve_power_flow(..., system=...)`` to skip that work on
    every subsequent call — with the SAME ``branch_states_method`` and ``precision``
    (``precision="mixed"`` caches single-precision factors, so the consuming solve must
    run its residual-correction iteration).
    """
    if slack not in ("ideal", "norton"):
        raise InputError(f"Unsupported slack {slack!r} (use 'ideal' or 'norton').")
    resolve_precision(precision, _cdtype(dtype))
    _validate_block_solver(linear_solver, block_rows)
    use_woodbury = _validate_branch_states_method(branch_states_method, branch_states)
    check_branch_impedances(grid)
    if branch_states is not None:
        if _branch_states_batched(branch_states):
            _check_connectivity_with_states(grid, branch_states)
        else:
            check_connectivity(_apply_scalar_states(grid, branch_states))
    else:
        check_connectivity(grid)

    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)
    index = node_phase_index(grid)
    f0 = float(grid.base_frequency_hz)
    leaves = _grid_param_leaves(grid, param_overrides, None, None, branch_states)
    if device is None:
        device = leaves[0].device if leaves else torch.device("cpu")

    fixed_rows, v_fixed = (
        _slack_rows_and_vref(grid, index, rdt, cdt, device)
        if slack == "ideal"
        else (None, None)
    )
    with torch.no_grad():
        factor_backend = (
            linear_solver if linear_solver in ("dense", "sparse", "block") else "auto"
        )
        if use_woodbury:
            y_eff, i_slack, fac = _woodbury_pieces(
                grid,
                f0,
                index,
                dtype,
                device,
                slack,
                param_overrides,
                branch_states,
                fixed_rows,
                factor_backend,
                block_rows,
                precision,
            )
        else:
            y_eff, i_slack = _y_eff_and_islack(
                grid, f0, index, dtype, device, slack, param_overrides, branch_states
            )
            fac = lu_factor_system(
                y_eff,
                fixed_rows=fixed_rows,
                backend=factor_backend,
                block_rows=block_rows,
                precision=precision,
                # The nonlinear outer iteration IS the refinement loop.
                refine_steps=0,
            )
    return PowerFlowSystem(
        index=index,
        f0=f0,
        slack=slack,
        y_eff=y_eff,
        i_slack=i_slack,
        fixed_rows=fixed_rows,
        v_fixed=v_fixed.detach() if v_fixed is not None else None,
        factorization=fac,
        static_leaves=tuple(leaves),
        network_fp=network_fingerprint(grid),
        precision=precision,
    )


def solve_power_flow(
    grid: Grid,
    *,
    slack: str = "ideal",
    method: str = "current_injection",
    tol: Optional[float] = None,
    tol_update_pu: Optional[float] = None,
    s_base_va: Optional[float] = None,
    max_iter: int = 100,
    dtype: torch.dtype = torch.complex128,
    precision: str = "full",
    device: Optional[torch.device] = None,
    operating_point: Optional[dict] = None,
    param_overrides: Optional[dict] = None,
    symmetry: Optional[str] = None,
    criticality: str = "auto",
    linear_solver: str = "auto",
    block_rows: Optional[Sequence[Tensor]] = None,
    on_disconnected: str = "raise",
    branch_states: Optional[dict] = None,
    branch_states_method: str = "assemble",
    system: Optional[PowerFlowSystem] = None,
    enforce_q_limits: Optional[bool] = None,
) -> PowerFlowResult:
    """Solve the const-P / ZIP fundamental power flow (differentiable, batched).

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.schemas.grid_schema.Grid` (type_ref expanded).
    slack:
        ``"ideal"`` (default) fixes source-node V = ``u_ref∠u_angle`` exactly via
        the Schur path in :func:`solve_harmonic` (matches pandapower / pgm).
        ``"norton"`` folds the source as a Norton shunt (matches OpenDSS Vsource).
    method:
        ``"current_injection"`` (default) — the fixed-point iteration (fast, batches in
        one solve, but its convergence region is smaller than the feasible region). Or
        ``"newton"`` — Newton on the real residual from a LINEAR const-Z warm start
        (OpenDSS-style); quadratic, and converges near the loadability nose where the
        fixed point oscillates. Both share the IFT gradient path.
    tol:
        PRIMARY convergence tolerance, the largest nodal apparent-power mismatch in PER
        UNIT: ``max_f |V_f conj(F_f)| / s_base_va`` over the free (non-slack) rows, with
        ``F = Y_eff V + I_device(V) - I_slack``. ``None`` (default) resolves the
        documented default ``solver.convergence.mismatch_pu`` (1e-8 pu — pandapower's
        ``tolerance_mva`` default on a 1 MVA base, and the same order as
        power-grid-model's ``error_tolerance``), so an iteration count is comparable
        across the three tools. Per unit, so one value means the same thing on a 400 V
        node and on a 20 kV node.
    tol_update_pu:
        SECONDARY convergence tolerance, the largest per-row voltage update
        ``max_rows |ΔV| / V_LN(node)`` in per unit; both criteria must hold. ``None``
        resolves ``solver.convergence.update_pu`` (1e-8 pu). Per-row normalisation makes
        it independent of the voltage level and of the number of rows, so a multi-voltage
        grid, and an ensemble of grids solved as one block-diagonal system, are judged
        exactly like a single feeder.
    s_base_va:
        Apparent-power base of the per-unit mismatch; ``None`` resolves
        ``solver.convergence.s_base_va`` (1e6 VA, pandapower's default ``sn_mva``).
    max_iter:
        Maximum fixed-point / Newton iterations.
    dtype:
        Complex dtype (``complex128`` for gradcheck; ``complex64`` ok — but see
        ``precision``: an SI-unit feeder's ``Y`` is ill-conditioned, so a plain
        complex64 solve loses about ``cond(Y)·1.2e-7`` of relative accuracy and a
        one-time warning names the estimated condition number when it exceeds the
        documented ``solver.precision.complex64_cond_warn`` threshold).
    precision:
        Working precision of the LINEAR ALGEBRA inside the iteration, independent of
        ``dtype``:

        - ``"full"`` (default) factors and back-substitutes at ``dtype``.
        - ``"mixed"`` factors a complex64 copy of ``Y_eff`` and keeps the iteration,
          the residual and the convergence test at complex128 (required): the fixed
          point runs in its residual-correction form
          ``V_{k+1} = V_k - A_s^{-1} F(V_k)``, so the single-precision factorization
          only preconditions the iteration and the converged voltage carries
          complex128 accuracy. Newton solves its direction in single precision
          (inexact Newton) with the residual and step at complex128. This is the
          recommended recipe for throughput on an ill-conditioned SI-unit feeder,
          above all on a GPU where double precision runs at a fraction of the
          single-precision rate; the plain complex64 path (``dtype=torch.complex64``,
          ``precision="full"``) stays available for comparison.
    device:
        Target device; defaults to the device of the first parameter leaf, else CPU.
    operating_point:
        Optional override of nameplate P/Q (see ``resolve_operating_power``).
    param_overrides:
        Optional differentiability hook (network R/L/C/Z, device P/Q) for gradcheck.
    symmetry:
        Calculation-symmetry mode ``None`` / ``"auto"`` / ``"symmetric"`` /
        ``"asymmetric"`` (``None`` -> config ``calculation.symmetry``). Resolved ONCE
        here (logged once); the resolved decision is threaded into every
        :func:`device_current_injections` call of the iteration (which resolves
        silently — no per-iteration logging).

    criticality:
        When to run the IFT-Jacobian criticality analysis (a dense ``[2N, 2N]``
        Jacobian + SVD): ``"auto"`` (default) only on non-convergence; ``"always"``
        also on a converged solve (a voltage-collapse MARGIN naming the weakest bus);
        ``"never"`` to skip it.
    linear_solver:
        The inner linear-solve backend.

        For ``method="current_injection"`` this selects the factorization of the
        constant ``Y_eff`` (:func:`pgml.solver.harmonic.lu_factor_system`):
        ``"auto"`` (default) uses the scipy SuperLU SPARSE factorization on CPU
        systems of ≥ ~500 rows — a power-grid ``Y`` has O(N) nonzeros, so sparse is
        ~O(N) where dense LU is O(N³) — and the batched dense torch LU everywhere
        else (CUDA is always dense); ``"dense"`` / ``"sparse"`` force the choice
        (``"matrix_free"`` is treated as ``"auto"`` here). ``"block"`` factors a
        BLOCK-DIAGONAL system (an ensemble of independent grids) one sub-grid at a
        time and requires ``block_rows``.

        For ``method="newton"``: ``"dense"`` (the explicit ``[2N, 2N]`` Jacobian +
        direct solve; ``"auto"`` resolves to this) or ``"matrix_free"``
        (Jacobian-free Newton-Krylov — GMRES on finite-difference Jacobian-vector
        products, ``O(N)`` memory for large grids). ``"sparse"`` and ``"block"``
        raise — Newton's Jacobian is built dense.
    block_rows:
        Row partition for ``linear_solver="block"``: one int64 tensor of node-phase
        row indices per independent sub-grid, together covering every row exactly
        once (``pgml.multigrid.MergedGrid.block_rows()`` for a merged ensemble).
        Each sub-grid's diagonal block is factored on its own — ``O(Σ n_k³)``
        instead of the union's ``O((Σ n_k)³)``, with sub-grids of equal size sharing
        one batched LU — which is what makes a many-grid ensemble tractable on CUDA
        (where the union's only alternative is a dense LU of the whole thing). On
        CPU the sparse union backend exploits the same structure and stays the
        better choice. Unsupported with ``on_disconnected="zero"`` (dropping dead
        rows re-indexes the system).
    on_disconnected:
        What to do when the pre-solve connectivity check finds (node, phase) rows
        with no galvanic path to an in-service source (an open switch or
        out-of-service branch on the only path, or no source at all):

        - ``"raise"`` (default) — raise :class:`~pgml.errors.ConnectivityError`
          naming the disconnected nodes, the separating open / out-of-service
          branches, and the concrete fixes.
        - ``"zero"`` — solve the energized sub-grid
          (:func:`pgml.topology.energized_subgrid`) and report 0 V on the
          disconnected rows (a de-energized conductor carries no voltage); the
          result keeps the FULL grid's row layout. Diagnostics describe the
          energized sub-system.
        - ``"ignore"`` — skip the check (the historical behavior: a disconnected
          area surfaces as a singular factorization or non-convergence).

        With ``branch_states``, ``"raise"`` checks every scenario's effective
        topology (vectorized over the batch); ``"zero"`` is unsupported there (a
        per-scenario topology has no single energized sub-grid).
    branch_states:
        Optional topology / switch-state batching ``{branch_id: state}``. A listed
        branch is always stamped and its admittance scaled by the state — a float,
        0-d tensor, or ``[*batch]`` scenario tensor: ``0`` = open, ``1`` = in
        service, intermediate = continuous (differentiable — gradients flow to
        state leaves through the IFT, enabling gradient-based topology search).
        The state OVERRIDES the branch's static ``in_service`` / ``closed`` flags.
        A batched state solves every switch configuration in ONE batched call
        (one assembly, per-scenario ``Y``); it broadcasts against a batched
        ``operating_point`` by the usual rules (align, or use extra leading dims
        for a cartesian sweep). ``method="newton"`` supports batched states OR a
        batched operating point, not both at once.
    branch_states_method:
        How a switch-state sweep reaches each state's linear system.

        - ``"assemble"`` (default) — assemble and factor the admittance of EVERY
          state (``O(S·N³)``, one ``[S, N, N]`` matrix).
        - ``"woodbury"`` — assemble and factor the BASE network ONCE (every
          switched branch closed) and reach each state through a
          Sherman-Morrison-Woodbury low-rank update of that factorization
          (:mod:`pgml.solver.lowrank`): a switched ``P``-phase branch's stamp is a
          rank-``≤ 2P`` term, so a state costs ``O(N²k + k³)`` with
          ``k = Σ 2P`` over the switched branches. Wins whenever ``k ≪ N``, which
          is the switch-sweep regime; it is an explicit opt-in (never chosen by a
          default or an ``"auto"``) because the win depends on that ratio.

        The Woodbury path changes only the FORWARD iteration: a backward pass
        still rebuilds the per-state admittance differentiably through the IFT, so
        gradients (including gradients w.r.t. the state values) are unchanged. It
        requires ``branch_states`` and ``method="current_injection"``.
    enforce_q_limits:
        Whether a voltage-regulating generator's ``q_min_var`` / ``q_max_var`` bound
        its reactive output. ``None`` (default) reads
        ``appliance.generator.enforce_q_limits`` from :mod:`pgml.defaults` (``True``).
        Enforcement is the standard PV-to-PQ switching: a unit whose required
        reactive power leaves its band is re-solved as a PQ injection pinned at the
        violated limit and released when its terminal voltage crosses the setpoint
        from the other side. ``False`` solves every regulating terminal unbounded,
        which is what pandapower's ``runpp(enforce_q_lims=False)`` default does. Read
        only when the grid carries a regulating generator.
    system:
        Optional :class:`PowerFlowSystem` from :func:`prepare_power_flow` — the
        operating-point-independent solve state (index, ``Y_eff``, slack rows,
        factorization, grid leaves) computed ONCE and reused across repeated
        solves of the SAME grid / slack / dtype / device / overrides / states
        (e.g. the chunk loop of :func:`pgml.scenarios.run_scenarios`). Skips
        assembly, factorization, the connectivity check (prepare ran it), and
        the grid leaf walk; the IFT backward still rebuilds differentiably, so
        gradients are unchanged. The ``current_injection`` forward benefits;
        Newton reuses the cached leaves only.

    Returns
    -------
    PowerFlowResult
        ``v`` complex ``[*batch, N]`` (DIFFERENTIABLE via the IFT), the index, the
        iteration count, the achieved per-unit power mismatch (``residual``), the
        convergence flag, a :class:`ConvergenceDiagnostics` reporting both criteria in
        per unit and in SI units, and — for a grid with voltage-regulating generators —
        a :class:`VoltageRegulationResult` with each unit's solved reactive power and
        bus type.

    Notes
    -----
    A grid with a voltage-regulating generator (a
    :class:`~pgml.schemas.grid_schema.VoltageRegulation` block, i.e. a PV terminal) is
    always solved by Newton: the regulated row pair replaces a current-balance row,
    which the current-injection fixed point has no setpoint to iterate on. Such a
    solve logs the method switch.
    """
    if method not in ("current_injection", "newton"):
        raise ModelingError(
            f"Unsupported method {method!r} (use 'current_injection' or 'newton')."
        )
    if slack not in ("ideal", "norton"):
        raise InputError(f"Unsupported slack {slack!r} (use 'ideal' or 'norton').")
    tol, tol_update_pu, s_base_va = _resolve_tolerances(tol, tol_update_pu, s_base_va)
    resolve_precision(precision, _cdtype(dtype))
    if criticality not in ("auto", "always", "never"):
        raise InputError(
            f"Unsupported criticality {criticality!r} (use 'auto'/'always'/'never')."
        )
    if linear_solver not in ("auto", "dense", "sparse", "block", "matrix_free"):
        raise InputError(
            f"Unsupported linear_solver {linear_solver!r} "
            "(use 'auto'/'dense'/'sparse'/'block'/'matrix_free')."
        )
    if method == "newton" and linear_solver in ("sparse", "block"):
        raise InputError(
            "method='newton' supports linear_solver 'auto'/'dense'/'matrix_free' "
            f"(its Jacobian is built dense); {linear_solver!r} selects the "
            "fixed-point factorization backend of method='current_injection'."
        )
    _validate_block_solver(linear_solver, block_rows, have_system=system is not None)
    use_woodbury = _validate_branch_states_method(
        branch_states_method, branch_states, method
    )
    if system is not None and system.precision != precision:
        raise InputError(
            f"The provided PowerFlowSystem was prepared with precision="
            f"{system.precision!r} but the solve requests {precision!r}: the cached "
            "factorization is single precision only in the mixed mode, and only the "
            "mixed mode runs the residual-correction iteration that needs it. Prepare "
            "and solve with the same precision."
        )
    if system is not None and use_woodbury != isinstance(
        system.factorization, LowRankUpdate
    ):
        raise InputError(
            "The provided PowerFlowSystem was prepared with a different "
            "branch_states_method: a woodbury system caches the sweep's BASE "
            "factorization plus its low-rank update, an assemble system the "
            "per-state factorization. Prepare and solve with the same method."
        )
    # Newton's inner solve: 'auto' resolves to the proven dense Jacobian path.
    newton_solver = "dense" if linear_solver == "auto" else linear_solver
    # Fixed-point factorization backend: 'matrix_free' has no meaning there.
    factor_backend = (
        linear_solver if linear_solver in ("dense", "sparse", "block") else "auto"
    )
    if on_disconnected not in ("raise", "zero", "ignore"):
        raise InputError(
            f"Unsupported on_disconnected {on_disconnected!r} "
            "(use 'raise'/'zero'/'ignore')."
        )

    if branch_states is not None and on_disconnected == "zero":
        raise InputError(
            'on_disconnected="zero" is unsupported with branch_states: a '
            "per-scenario topology has no single energized sub-grid. Use "
            '"raise" (per-scenario check) or "ignore".'
        )
    if block_rows is not None and on_disconnected == "zero":
        raise InputError(
            'on_disconnected="zero" is unsupported with block_rows: solving the '
            "energized sub-grid re-indexes the node-phase rows, so the given row "
            'partition no longer describes the system. Use "raise" or "ignore".'
        )
    if system is None:
        # A zero-impedance branch has no primitive stamp at all, so it is refused by
        # name whatever the connectivity policy is (a prepared system already ran it).
        check_branch_impedances(grid)
    if system is None and on_disconnected != "ignore":
        if branch_states is not None:
            if _branch_states_batched(branch_states):
                _check_connectivity_with_states(grid, branch_states)
            else:
                check_connectivity(_apply_scalar_states(grid, branch_states))
        elif on_disconnected == "raise":
            check_connectivity(grid)
        else:
            sub, dropped = energized_subgrid(grid)
            if dropped:
                _log.warning(
                    "solve_power_flow: %d disconnected node(s) %s solved as 0 V "
                    '(on_disconnected="zero"); the energized sub-grid carries the '
                    "solution.",
                    len(dropped),
                    list(dropped[:10]),
                )
                sub_res = solve_power_flow(
                    sub,
                    slack=slack,
                    method=method,
                    tol=tol,
                    tol_update_pu=tol_update_pu,
                    s_base_va=s_base_va,
                    max_iter=max_iter,
                    dtype=dtype,
                    precision=precision,
                    device=device,
                    operating_point=operating_point,
                    param_overrides=param_overrides,
                    symmetry=symmetry,
                    criticality=criticality,
                    linear_solver=linear_solver,
                    on_disconnected="ignore",
                    enforce_q_limits=enforce_q_limits,
                )
                return _expand_zeroed_result(grid, sub_res)

    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)

    index = node_phase_index(grid)
    n = index.size
    f0 = float(grid.base_frequency_hz)

    # Resolve calculation symmetry ONCE (and log once); thread the canonical string
    # into every device_current_injections call so the iteration stays consistent.
    asymmetric = resolve_asymmetric(grid, operating_point, mode=symmetry)
    # Single INFO modeling summary for this entry point (logs once per
    # solve_power_flow call). solve_harmonic_flow does NOT log separately — it
    # delegates its modeling summary to this call, so there is no double logging.
    log_modeling_summary(grid, asymmetric=asymmetric)
    sym_resolved = "asymmetric" if asymmetric else "symmetric"

    if (
        method == "newton"
        and _branch_states_batched(branch_states)
        and _operating_point_batch_size(operating_point) > 1
    ):
        raise InputError(
            "method='newton' does not combine a batched operating_point with "
            "batched branch_states (its per-scenario slicing covers the operating "
            "point only); batch one of the two, or use method='current_injection'."
        )

    if system is not None:
        # The grid-side leaves were walked once in prepare_power_flow; only the
        # per-call operating point can add new ones.
        leaves = list(system.static_leaves)
        seen = {id(t) for t in leaves}
        if operating_point is not None:
            _collect_leaves(operating_point, leaves, seen)
    else:
        leaves = _grid_param_leaves(
            grid, param_overrides, None, operating_point, branch_states
        )
    if device is None:
        device = (
            system.y_eff.device
            if system is not None
            else (leaves[0].device if leaves else torch.device("cpu"))
        )

    # Slack rows are constant indices; the reference VOLTAGE is recomputed fresh
    # from the (possibly tensor) u_ref/u_angle on every residual eval so the
    # graph is not reused across gradcheck's multiple backward passes.
    if system is not None and (
        system.slack != slack
        or system.index.size != n
        or system.y_eff.dtype != cdt
        or not _same_device(system.y_eff.device, torch.device(device))
    ):
        raise InputError(
            "The provided PowerFlowSystem does not match this solve "
            f"(system: slack={system.slack!r}, N={system.index.size}, "
            f"dtype={system.y_eff.dtype}, device={system.y_eff.device}; solve: "
            f"slack={slack!r}, N={n}, dtype={cdt}, device={device}). Prepare it "
            "with the same grid and arguments."
        )
    if (
        system is not None
        and system.network_fp
        and system.network_fp != network_fingerprint(grid)
    ):
        raise InputError(
            "The provided PowerFlowSystem was prepared from a different network: the "
            "grid's nodes/branches/sources/shunts (topology or parameter values) have "
            "changed since prepare_power_flow, so the cached admittance and "
            "factorization are stale. Re-prepare the system for this grid."
        )

    def v_fixed_fn():
        if slack != "ideal":
            return None
        # Recomputed fresh each residual eval (so the graph is not reused across
        # gradcheck's backward passes) and incorporates any per-scenario source
        # u_ref scale from the operating point (the batched slack boundary).
        _, vf = _slack_rows_and_vref(grid, index, rdt, cdt, device, operating_point)
        return vf

    if system is not None:
        # The prepared system caches the operating-point-INDEPENDENT slack reference.
        # A per-scenario u_ref scale is operating-point data, so recompute the (batched)
        # reference here when present; otherwise reuse the cached detached value.
        fixed_rows = system.fixed_rows
        v_fixed = (
            v_fixed_fn()
            if (slack == "ideal" and _has_uref_scale(operating_point))
            else system.v_fixed
        )
    else:
        fixed_rows = (
            _slack_rows_and_vref(grid, index, rdt, cdt, device)[0]
            if slack == "ideal"
            else None
        )
        v_fixed = v_fixed_fn()
    # NOTE: v_fixed is derived from the grid's source fields (u_ref/u_angle) and the
    # operating point's u_ref_scale, whose leaves the grid + operating-point walk above
    # already collected — no second leaf walk is needed.

    # ----- closures over the CURRENT leaf values ----------------------------
    def build_system():
        return _y_eff_and_islack(
            grid, f0, index, dtype, device, slack, param_overrides, branch_states
        )

    def make_residual_complex(op):
        """Build ``F_c(V) = Y_eff @ V + I_device(V) - I_slack`` for an operating point.

        A factory (not a single closure) so the batched-Newton path can build a
        per-scenario residual from a sliced ``op`` while the full-batch ``op`` residual
        drives the IFT backward. Fully DIFFERENTIABLE w.r.t. the parameter leaves
        (rebuilds the injection resolution on every call) — the ``dR/dθ`` half of the
        IFT. The iteration-facing counterpart is :func:`make_fast_residual_complex`.
        """

        def residual_complex(v_cmplx: Tensor, y_eff: Tensor, i_slack: Tensor) -> Tensor:
            i_dev = device_current_injections(
                grid,
                v_cmplx,
                index,
                [f0],
                dtype=dtype,
                device=device,
                operating_point=op,
                param_overrides=param_overrides,
                symmetry=sym_resolved,
            ).squeeze(-2)  # [*b, N]
            return _apply_y(y_eff, v_cmplx) + i_dev - i_slack

        return residual_complex

    def make_fast_residual_complex(op):
        """Plan-based residual for the ITERATION paths (``dR/dV`` only).

        Resolves the operating point ONCE into an :class:`InjectionPlan` (detached)
        and evaluates the residual with pure tensor ops. Correct wherever only the
        dependence on ``V`` matters — the no-grad forward iterations, Newton's line
        search, and the state Jacobian ``J = dR/dx`` (the plan's power tensors are
        constants of that differentiation). The parameter gradients ``dR/dθ`` use
        :func:`make_residual_complex` instead.
        """
        with torch.no_grad():
            plan = build_injection_plan(
                grid,
                index,
                [f0],
                dtype=dtype,
                device=device,
                operating_point=op,
                param_overrides=param_overrides,
                symmetry=sym_resolved,
            )

        def residual_complex(v_cmplx: Tensor, y_eff: Tensor, i_slack: Tensor) -> Tensor:
            i_dev = injections_from_plan(plan, v_cmplx).squeeze(-2)  # [*b, N]
            return _apply_y(y_eff, v_cmplx) + i_dev - i_slack

        residual_complex.plan = plan
        return residual_complex

    # ----- voltage-regulating terminals (PV buses) --------------------------
    # Their residual row pair (active balance + |V|² − V_set²) replaces the terminal's
    # current-balance rows, so the fixed-point iteration — which has no voltage
    # setpoint to iterate on — cannot solve them: route such a grid to Newton.
    pv = collect_pv_terminals(
        grid,
        index,
        rdt,
        cdt,
        device,
        operating_point,
        enforce_q_limits=enforce_q_limits,
    )
    if pv is not None and method != "newton":
        _log.warning(
            "solve_power_flow: the grid has %d voltage-regulating generator(s) (PV "
            "terminal(s)); solving with method='newton' instead of %r. The regulated "
            "row pair replaces a current-balance row, which the current-injection "
            "fixed point cannot iterate on.",
            pv.n_terminals,
            method,
        )
        method = "newton"

    def build_residuals(op, pv_state):
        """The three residual forms for one operating point + PV active set."""
        rc = make_residual_complex(op)
        frc = make_fast_residual_complex(op)
        rr = _make_real_residual(
            build_system,
            rc,
            fixed_rows,
            v_fixed_fn,
            n,
            cdt,
            state_residual_complex=frc,
            pv=pv_state,
        )
        return rc, frc, rr

    op_eff = (
        operating_point if pv is None else pv.pinned_operating_point(operating_point)
    )
    residual_complex, fast_residual_complex, real_res = build_residuals(op_eff, pv)

    # Per-row per-unit bases of both convergence criteria (the voltage base is each
    # node's line-to-neutral rated voltage, as in the flat start and the diagnostics).
    v_base = _node_voltage_bases(grid, index, rdt, device)

    # ----- forward: solve for the detached V* (gradients attached by the IFT) -----
    def _newton_warm_starts(op, pv_state, vf=None):
        """Newton warm starts to try, in order.

        Without voltage-regulating terminals there is exactly one (the const-Z
        solution, built lazily) — the historical path. With them, the balanced
        nominal start that sits ON the setpoints is added, and the order is decided
        by the const-Z seed itself: a COLLAPSED const-Z profile (a node below half
        its nominal) is a poor Newton start on a regulated grid, so the nominal start
        goes first and the const-Z seed stays as the fallback.
        """

        def const_z():
            return _linear_const_z_init(
                grid,
                f0,
                index,
                dtype,
                device,
                slack,
                op,
                param_overrides,
                fixed_rows,
                v_fixed if vf is None else vf,
                branch_states,
            )

        if pv_state is None:
            return [const_z]

        def nominal():
            return _pv_nominal_init(
                grid,
                index,
                pv_state,
                rdt,
                cdt,
                device,
                fixed_rows,
                v_fixed if vf is None else vf,
            )

        seed = const_z()
        if _seed_is_collapsed(grid, index, seed, rdt, device):
            _log.info(
                "solve_power_flow: the const-impedance warm start collapses below "
                "%.2f pu somewhere; starting Newton from the balanced nominal profile "
                "at the voltage setpoints instead (the const-Z seed stays as a "
                "fallback).",
                _SEED_COLLAPSE_PU,
            )
            return [nominal, lambda: seed]
        return [lambda: seed, nominal]

    def run_forward(op, rr, frc, pv_state):
        """One forward solve at a fixed operating point and PV active set."""
        if method == "newton":
            # OpenDSS-style warm start: the LINEAR const-Z solution, then Newton on the
            # full const-P / ZIP residual. Newton's quadratic convergence and far larger
            # convergence region reach solutions the current-injection fixed point cannot
            # (e.g. near the loadability nose — see
            # ``run/examples/current_injection_convergence.py``).
            bsize = _operating_point_batch_size(op)
            if bsize > 1:
                # A batched operating point solves SEQUENTIALLY per scenario: the
                # per-scenario ``jacobian(vectorize=True)`` (one vectorized call per
                # scenario) is measurably faster than a batch-native block-diagonal
                # build — the O(B²·(2N)²) full-map Jacobian does not fit, and the O(B)
                # column-by-column alternative costs 2N JVP evaluations per Newton
                # step (4x slower than this loop at B=64/N=180). Newton is the
                # hard-grid / near-nose solver — for bulk batches prefer the
                # vectorized current-injection method.
                return _newton_forward_sequential(
                    grid,
                    index,
                    device,
                    slack,
                    op,
                    fixed_rows,
                    build_system,
                    make_fast_residual_complex,
                    _newton_warm_starts,
                    bsize,
                    pv_state,
                    n,
                    rdt,
                    cdt,
                    v_base,
                    tol,
                    tol_update_pu,
                    s_base_va,
                    max_iter,
                    newton_solver,
                    precision,
                )
            return _newton_from_starts(
                rr,
                _newton_warm_starts(op, pv_state),
                n,
                rdt,
                cdt,
                device,
                v_base,
                tol,
                tol_update_pu,
                s_base_va,
                max_iter,
                newton_solver,
                precision,
                fixed_rows,
            )
        solve_system = system
        if use_woodbury and solve_system is None:
            # One base assembly + factorization for the whole sweep; each state is a
            # low-rank update of it. Detached like every forward quantity — the IFT
            # backward rebuilds the per-state admittance differentiably.
            with torch.no_grad():
                y_base, i_slack_w, upd = _woodbury_pieces(
                    grid,
                    f0,
                    index,
                    dtype,
                    device,
                    slack,
                    param_overrides,
                    branch_states,
                    fixed_rows,
                    factor_backend,
                    block_rows,
                    precision,
                )
            solve_system = PowerFlowSystem(
                index=index,
                f0=f0,
                slack=slack,
                y_eff=y_base,
                i_slack=i_slack_w,
                fixed_rows=fixed_rows,
                v_fixed=v_fixed.detach() if v_fixed is not None else None,
                factorization=upd,
                static_leaves=tuple(leaves),
                precision=precision,
            )
        return _current_injection_forward(
            grid,
            index,
            build_system,
            fixed_rows,
            v_fixed,
            frc.plan,
            n,
            rdt,
            cdt,
            device,
            v_base,
            tol,
            tol_update_pu,
            s_base_va,
            max_iter,
            factor_backend,
            solve_system,
            block_rows,
            precision,
        )

    # ----- the solve, plus PV-to-PQ switching rounds where a reactive limit binds ---
    # Each round is one complete solve at a FIXED active set (which terminals regulate,
    # which are pinned at a limit), so the residual the IFT differentiates is exactly
    # the converged configuration's. The decision between rounds reads converged values
    # and is off-tape by construction.
    switch_rounds = 0
    while True:
        (
            v_star,
            iterations,
            residual_norm,
            converged,
            update_history,
            y_eff0,
            i_slack0,
            converged_mask,
            residual_vec,
            update_max_pu,
            update_norm_v,
            fc_star,
        ) = run_forward(op_eff, real_res, fast_residual_complex, pv)
        if pv is None or not pv.enforce_q_limits:
            break
        if not converged:
            # A reactive-limit decision reads the converged reactive power; taken at a
            # non-converged iterate it would switch on noise (and pay another full
            # solve per round). Stop with what the solve reached and report it.
            _log.warning(
                "solve_power_flow: the solve did not converge, so the reactive-limit "
                "(PV-to-PQ) switching stopped after %d round(s) with %s.",
                switch_rounds,
                pv.describe_state(),
            )
            break
        pv_next, changed = pv.switch(
            _nodal_residual(fast_residual_complex, v_star, y_eff0, i_slack0), v_star
        )
        if not changed:
            break
        if switch_rounds + 1 >= pv.max_rounds:
            _log.warning(
                "solve_power_flow: the reactive-limit (PV-to-PQ) switching did not "
                "settle in %d rounds; keeping the last consistent solve (%s). Widen "
                "the hysteresis (pgml.defaults appliance.generator."
                "q_limit_hysteresis_pu) or check for a generator whose limit and "
                "setpoint are incompatible.",
                pv.max_rounds,
                pv.describe_state(),
            )
            break
        switch_rounds += 1
        pv = pv_next
        op_eff = pv.pinned_operating_point(operating_point)
        residual_complex, fast_residual_complex, real_res = build_residuals(op_eff, pv)

    # At a regulating row the forward's residual carries the SUBSTITUTED row pair
    # (active balance + setpoint), not the nodal current mismatch, so the reported
    # mismatch and the reactive-power readout use the nodal residual instead.
    fc_nodal = (
        fc_star
        if pv is None
        else _nodal_residual(fast_residual_complex, v_star, y_eff0, i_slack0)
    )

    # Convergence diagnostics at V* (autograd-free; the criticality analysis builds the
    # IFT real Jacobian only when the solve did not converge).
    diagnostics = _build_diagnostics(
        grid,
        index,
        v_star,
        fc_nodal,
        v_base,
        s_base_va,
        real_res,
        fixed_rows,
        update_history,
        bool(converged),
        iterations,
        float(residual_norm),
        float(update_max_pu),
        float(update_norm_v),
        rdt,
        device,
        criticality,
        pv,
    )

    if leaves:
        v_out = _IFTPowerFlow.apply(v_star, real_res, n, rdt, cdt, *leaves)
    else:
        v_out = v_star

    # Per-scenario reporting: a batched solve NEVER raises on a failed element — every
    # element's best-effort V is returned, the failures are listed, and an error record
    # names them with detail (so a large sweep yields data + diagnosable failures).
    cmask_out, failed_states = _resolve_failed_states(converged_mask)
    if failed_states:
        total = int(cmask_out.numel())
        shown = ", ".join(str(i) for i in failed_states[:20])
        more = "" if len(failed_states) <= 20 else f", … (+{len(failed_states) - 20})"
        cause = diagnostics.likely_cause if diagnostics is not None else ""
        _log.error(
            "solve_power_flow: %d/%d scenarios did not converge in %d iterations "
            "(worst power mismatch %.3e pu); returning best-effort voltages. Failed "
            "indices: [%s%s]%s",
            len(failed_states),
            total,
            iterations,
            float(residual_norm),
            shown,
            more,
            f" — {cause}" if cause else "",
        )

    regulation = None
    if pv is not None:
        with torch.no_grad():
            regulation = VoltageRegulationResult(
                q_var=pv.required_q(fc_nodal, v_star),
                regulating=pv.regulating_mask(),
                switch_rounds=switch_rounds,
                enforce_q_limits=pv.enforce_q_limits,
            )
        _log.info(
            "solve_power_flow: %d voltage-regulating terminal(s) solved (%s) in %d "
            "switching round(s); reactive limits %s.",
            pv.n_terminals,
            pv.describe_state(),
            switch_rounds,
            "enforced" if pv.enforce_q_limits else "NOT enforced",
        )

    return PowerFlowResult(
        v=v_out,
        index=index,
        iterations=iterations,
        residual=residual_norm.reshape(()),
        converged=bool(converged),
        diagnostics=diagnostics,
        converged_mask=cmask_out,
        failed_states=failed_states,
        regulation=regulation,
    )


# ---------------------------------------------------------------------------
# forward solvers (detached V*; gradients are attached by the IFT below)
# ---------------------------------------------------------------------------
def _factored_solve(fac, rhs: Tensor, v_fixed: Optional[Tensor]) -> Tensor:
    """Back-substitute against a plain factorization or a low-rank-updated one.

    The switch-state sweep's per-state system is the base factorization plus a
    rank-``k`` update (:class:`~pgml.solver.lowrank.LowRankUpdate`); every other
    path holds a plain :class:`~pgml.solver.harmonic.FactoredSystem`. Both answer
    the same ``[*batch, N]`` contract, so the iteration is identical.
    """
    if isinstance(fac, LowRankUpdate):
        return solve_factored_updated(fac, rhs, v_fixed=v_fixed)
    return solve_factored(fac, rhs, v_fixed=v_fixed)


def _current_injection_forward(
    grid,
    index,
    build_system,
    fixed_rows,
    v_fixed,
    plan,
    n,
    rdt,
    cdt,
    device,
    v_base,
    tol_mismatch_pu,
    tol_update_pu,
    s_base_va,
    max_iter,
    factor_backend="auto",
    system=None,
    block_rows=None,
    precision="full",
):
    """Current-injection fixed point ``V_{k+1} = Y_eff^{-1}(I_slack − I_device(V_k))``.

    ``plan`` is the precomputed :class:`~pgml.assembly.InjectionPlan`: the
    operating point is resolved once and every iteration evaluates
    :func:`injections_from_plan` (pure tensor ops).

    A ``system`` whose factorization is a
    :class:`~pgml.solver.lowrank.LowRankUpdate` (the Woodbury switch-state sweep)
    runs the SAME iteration: ``y_eff0`` is then the matrix-free per-state operator
    and each back-substitution carries the low-rank correction.

    Convergence is per scenario on the two PER-UNIT criteria of
    :class:`_PuConvergence` (apparent-power mismatch and voltage update), which is why
    the iteration also forms the nodal residual ``F(V) = Y_eff V + I_device(V) -
    I_slack`` — one matrix-vector product per iteration on top of the injection
    evaluation the fixed point needs anyway.

    ``precision="mixed"`` turns the iteration into its RESIDUAL-CORRECTION form
    ``V_{k+1} = V_k - A_s^{-1} F(V_k)`` (algebraically the same fixed point) with the
    single-precision factorization ``A_s`` as the correction operator and ``F`` formed
    at the working precision. The outer iteration is then itself the refinement loop:
    an inexact ``A_s^{-1}`` changes only the contraction rate, never the fixed point, so
    the converged voltage carries complex128 accuracy at the cost of single-precision
    back-substitutions.

    Returns ``(v_star, iterations, mismatch_max_pu, converged, update_history, y_eff0,
    i_slack0, converged_mask, mismatch_vec, update_max_pu, update_norm_v, fc)``; the
    maxima are over the batch, while ``converged_mask`` / ``mismatch_vec`` are PER
    scenario and ``fc`` is the final nodal residual (reused by the diagnostics).
    """
    with torch.no_grad():
        if system is not None:
            y_eff0, i_slack0 = system.y_eff, system.i_slack
        else:
            y_eff0, i_slack0 = build_system()
        # Every leading dim of y_eff0 / i_slack0 is a SCENARIO dim (the frequency
        # axis is folded away in _y_eff_and_islack), so this broadcast is pure batch.
        lead = torch.broadcast_shapes(i_slack0.shape[:-1], y_eff0.shape[:-2])
        # Phase-aware balanced warm start: the source reference magnitude rotated by
        # the standard positive-sequence angle of each row's phase (a=0, b=-120,
        # c=+120, n=0). A FLAT start would make DELTA element voltages identically
        # zero (V_a - V_b = 0), giving 0/0 in the const-P device current — so the
        # start must carry the phase rotation. Whole block is under no_grad: it only
        # seeds the fixed point and never enters the converged value / its gradient.
        #
        # Magnitude is derived PER ROW (not by scaling everything by one fixed row):
        # each fixed (source, phase) row is seeded with its OWN |v_fixed|, and every
        # other (non-source) row with a sensible balanced default = the source's
        # Phase-A reference magnitude (not row 0, which may be a non-A phase or a
        # differently-rated source). This is correct when a source's first listed
        # phase is not Phase.A or when sources carry differing per-phase magnitudes.
        phase_codes = index.phase_codes.to(device)
        phase_angle = torch.tensor(
            [0.0, -2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0, 0.0],
            dtype=rdt,
            device=device,
        )
        is_neutral = (phase_codes == 3).to(rdt)  # [N]
        row_ang = phase_angle[phase_codes]  # [N]

        # Resolve the (rows, |v_fixed|) used for both the per-row seed and the
        # balanced default. The UN-SCALED source reference (``sl_vref``, always ``[S]``)
        # is used here: the warm start only needs a ballpark magnitude per row, and a
        # per-scenario ``u_ref`` scale would give a batched ``v_fixed`` that the ``[N]``
        # seed scatter cannot consume (the actual per-scenario slack is pinned by
        # ``solve_factored(v_fixed=...)`` below). For norton (no fixed rows) this is the
        # source reference magnitude directly.
        sl_rows, sl_vref = _slack_rows_and_vref(grid, index, rdt, cdt, device)
        ref_rows = fixed_rows if (fixed_rows is not None) else sl_rows
        ref_v = sl_vref

        # Balanced default magnitude for non-source rows: the source Phase-A
        # reference magnitude where available, else the first reference magnitude,
        # else 1.0 (purely passive grid).
        if ref_v is not None and ref_rows is not None:
            ref_mag = ref_v.abs().to(rdt)  # [S]
            ref_phase = phase_codes.index_select(0, ref_rows)  # [S]
            a_mask = ref_phase == 0
            if bool(a_mask.any()):
                default_mag = ref_mag[a_mask][0]
            else:
                default_mag = ref_mag.reshape(-1)[0]
        else:
            default_mag = torch.ones((), dtype=rdt, device=device)

        # Start from the balanced default on every row, then overwrite each fixed
        # (source, phase) row with its own reference magnitude (out-of-place scatter).
        row_mag = default_mag.to(rdt).reshape(()).expand(n).clone()  # [N]
        if ref_v is not None and ref_rows is not None:
            row_mag = row_mag.scatter(0, ref_rows, ref_v.abs().to(rdt))
        # Neutral rows start at ~0 V (a grounded/return conductor): otherwise V_N
        # would collide with phase A and a WYE-with-neutral element voltage
        # V_A - V_N would be 0 -> 0/0 in the const-P device current.
        row_mag = row_mag * (1.0 - is_neutral)
        v_row = torch.polar(row_mag, row_ang).to(cdt)
        v = v_row.expand(*lead, n).clone()

        mismatch_vec = torch.zeros((), dtype=rdt, device=device)
        mismatch_max = torch.zeros((), dtype=rdt, device=device)
        update_max = torch.zeros((), dtype=rdt, device=device)
        update_norm_v = torch.zeros((), dtype=rdt, device=device)
        converged_mask = torch.zeros((), dtype=torch.bool, device=device)
        iterations = 0
        converged = False
        update_history: list[float] = []
        # Y_eff is the network admittance — constant across iterations (the const-P/ZIP
        # loads enter the RHS as I_device(V), never Y). Factor it ONCE and back-substitute
        # each iteration (the whole fixed point runs under no_grad; the IFT supplies grads).
        fac = (
            system.factorization
            if system is not None
            else lu_factor_system(
                y_eff0,
                fixed_rows=fixed_rows,
                backend=factor_backend,
                block_rows=block_rows,
                precision=precision,
                # The outer iteration below IS the refinement loop, so the inner solve
                # needs no refinement of its own.
                refine_steps=0,
            )
        )
        mixed = precision == "mixed"
        _warn_complex64_conditioning(fac, rdt)
        # The achievable floor of both criteria depends on the RESOLVED backend (SuperLU's
        # single-precision back-substitution is noisier than the dense torch LU).
        # A low-rank (switch-state) update amplifies the base solve's rounding by a
        # measured factor, so the precision this solve path resolves is that much
        # coarser — both floors carry it.
        amp = float(getattr(fac, "amplification", 1.0))
        ctest = _PuConvergence(
            v_base=v_base,
            s_base=s_base_va,
            tol_mismatch_pu=tol_mismatch_pu,
            tol_update_pu=tol_update_pu,
            floor_update=_rel_convergence_floor(rdt, fac.backend) * amp,
            floor_mismatch=_mismatch_floor_rel(rdt, fac.backend) * amp,
            fixed_rows=fixed_rows,
            y_eff=y_eff0,
            n=n,
            device=device,
            rdt=rdt,
        )
        i_dev = injections_from_plan(plan, v).squeeze(-2)  # [*b, N]
        fc = _apply_y(y_eff0, v) + i_dev - i_slack0 if mixed else None
        zero_slack = None if v_fixed is None else torch.zeros_like(v_fixed)
        for k in range(max_iter):
            if mixed and k > 0:
                # Residual-correction form: the slack rows of the correction are held
                # at 0, so the pinned voltages stay exactly the reference.
                v_new = v - _factored_solve(fac, fc, zero_slack)
            else:
                # The factorization carries no frequency axis, so the solution keeps
                # exactly the right-hand side's batch shape — a trailing singleton batch
                # dim (an operating point batched [B, 1]) passes through untouched.
                v_new = _factored_solve(fac, i_slack0 - i_dev, v_fixed)
            # One injection evaluation per iteration serves both the next right-hand
            # side and the residual the per-unit criteria are measured on.
            i_dev = injections_from_plan(plan, v_new).squeeze(-2)  # [*b, N]
            fc = _apply_y(y_eff0, v_new) + i_dev - i_slack0  # [*b, N]
            dv = v_new - v
            mism_rows = ctest.mismatch_rows_pu(v_new, fc)
            upd_rows = ctest.update_rows_pu(dv)
            converged_mask, mismatch_vec, update_vec = ctest.check(mism_rows, upd_rows)
            mismatch_max = mismatch_vec.max()
            update_max = update_vec.max()
            update_norm_v = torch.linalg.vector_norm(dv, dim=-1).max()
            update_history.append(float(update_max))
            v = v_new
            iterations += 1
            if bool(converged_mask.all()):
                converged = True
                break

    return (
        v,
        iterations,
        mismatch_max,
        converged,
        update_history,
        y_eff0,
        i_slack0,
        converged_mask,
        mismatch_vec,
        update_max,
        update_norm_v,
        fc,
    )


def _nodal_residual(fast_residual_complex, v_star, y_eff0, i_slack0) -> Tensor:
    """The complex nodal mismatch ``F_c = Y_eff V* + I_device(V*) - I_slack`` at ``V*``.

    The forward's own residual output carries the SUBSTITUTED row pair at a
    voltage-regulating terminal (active-power balance and the voltage setpoint), so the
    reactive-power readout and the reported mismatch need the plain nodal residual.
    Autograd-free: one injection-plan evaluation plus one matrix-vector product.
    """
    with torch.no_grad():
        return fast_residual_complex(v_star, y_eff0, i_slack0)


def _linear_const_z_init(
    grid,
    f0,
    index,
    dtype,
    device,
    slack,
    operating_point,
    param_overrides,
    fixed_rows,
    v_fixed,
    branch_states=None,
):
    """OpenDSS-style warm start: the LINEAR const-Z solution (one linear solve).

    Folds every load / generator as a constant-impedance shunt at nominal voltage
    (:func:`pgml.assembly.assemble_ybus`, the linear-model assembler) and solves once.
    A far better Newton seed than a flat / nominal start, especially near the
    loadability limit. Detached (a warm start never enters the gradient).
    """
    import logging

    pgml_log = logging.getLogger("pgml")
    prev = pgml_log.level
    # Suppress the duplicate modeling-summary INFO from assemble_ybus (solve_power_flow
    # already logged it for this call).
    pgml_log.setLevel(max(prev, logging.WARNING))
    try:
        yb = assemble_ybus(
            grid,
            [f0],
            dtype=dtype,
            device=device,
            operating_point=operating_point,
            param_overrides=param_overrides,
            branch_states=branch_states,
        )
    finally:
        pgml_log.setLevel(prev)
    if yb.Y.ndim > 3:
        # Batch dims (a batched operating point folded into the const-Z shunts,
        # and/or batched branch states) precede the assembly's singleton frequency
        # axis: fold the frequency axis, KEEP every batch dim — a trailing batch
        # dim of one (an operating point batched [B, 1]) is a scenario dim, not a
        # frequency axis, and must survive into the seed's shape.
        y_lin = yb.Y.squeeze(-3)  # [*batch, N, N]
        drop_freq_axis = False
    else:
        y_lin = yb.Y if yb.Y.ndim == 3 else yb.Y.unsqueeze(0)  # [1, N, N]
        drop_freq_axis = True
    if slack == "norton":
        i_init = build_injections(
            grid,
            [f0],
            index,
            dtype=dtype,
            device=device,
            param_overrides=param_overrides,
        ).squeeze(-2)  # [*b, N] source Norton current (single-frequency axis folded)
    else:
        i_init = torch.zeros(y_lin.shape[-1], dtype=y_lin.dtype, device=device)  # [N]
    with torch.no_grad():
        v0 = solve_harmonic(y_lin, i_init, fixed_rows=fixed_rows, v_fixed=v_fixed)
    if drop_freq_axis:
        v0 = v0.squeeze(-2)  # [1, N] -> [N]: the frequency axis, known positionally
    return v0


def _pv_nominal_init(
    grid, index, pv: PVTerminals, rdt, cdt, device, fixed_rows, v_fixed
) -> Tensor:
    """Balanced nominal warm start with every regulated row AT its setpoint.

    Each row starts at its own node's line-to-neutral nominal magnitude (so a grid
    spanning several voltage levels starts at ~1 pu everywhere, unlike a start built
    from the source's magnitude), with the standard positive-sequence phase rotation,
    neutral rows at 0 V, ideal-slack rows at their reference magnitude, and every
    voltage-regulating row at ``v_set``. This is the classical flat start of a
    transmission solve; it is the SECOND Newton start tried for a grid with PV
    terminals, because the const-Z seed (:func:`_linear_const_z_init`) wins on
    load-dominated networks while this one wins where the const-Z fold depresses the
    profile far from the regulated operating point. Detached — a warm start never
    enters the gradient.
    """
    with torch.no_grad():
        phase_codes = index.phase_codes.to(device)
        ang = torch.tensor(
            [0.0, -2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0, 0.0],
            dtype=rdt,
            device=device,
        )[phase_codes]
        n = index.size
        mag = _node_voltage_bases(grid, index, rdt, device) * (
            1.0 - (phase_codes == 3).to(rdt)
        )  # [N]
        rows, vset = pv.setpoint_volts()  # [R], [*b, R]
        pieces = [(rows, vset.to(rdt))]
        if fixed_rows is not None and v_fixed is not None:
            pieces.append((fixed_rows, v_fixed.abs().to(rdt)))
        lead = torch.broadcast_shapes(*[tuple(t.shape[:-1]) for _r, t in pieces])
        mag = mag.broadcast_to(*lead, n).clone()
        for r, val in pieces:
            mag = mag.scatter(
                -1,
                r.expand(*lead, r.shape[-1]),
                val.broadcast_to(*lead, r.shape[-1]),
            )
        return torch.polar(mag, ang).to(cdt)


#: A const-Z warm start below this fraction of a node's nominal magnitude is a
#: COLLAPSED seed (the const-impedance fold of a load that is large against its local
#: impedance), far outside any steady-state operating point and a poor Newton start.
_SEED_COLLAPSE_PU = 0.5


def _seed_is_collapsed(grid, index, v_seed: Tensor, rdt, device) -> bool:
    """True if a warm start leaves any live row below :data:`_SEED_COLLAPSE_PU`."""
    with torch.no_grad():
        bases = _node_voltage_bases(grid, index, rdt, device).clamp_min(1e-12)
        pu = v_seed.reshape(-1, index.size).abs() / bases[None, :]
        live = (index.phase_codes.to(device) != 3)[None, :]
        return bool((pu[live] < _SEED_COLLAPSE_PU).any())


def _newton_from_starts(
    real_res,
    v_inits,
    n,
    rdt,
    cdt,
    device,
    v_base,
    tol_mismatch_pu,
    tol_update_pu,
    s_base_va,
    max_iter,
    linear_solver,
    precision="full",
    fixed_rows=None,
):
    """Newton from each warm start in turn until one converges; best result wins.

    A single start is the historical path (identical result, one call). With more
    than one, a start that fails to converge is followed by the next one and the
    restart is logged; if none converges the attempt with the smallest final per-unit
    power mismatch is returned, so the diagnostics describe the best iterate reached.
    Returns the 12-tuple of :func:`_newton_forward`.
    """
    best = None
    for attempt, make_init in enumerate(v_inits):
        out = _newton_forward(
            real_res,
            make_init(),
            n,
            rdt,
            cdt,
            device,
            v_base,
            tol_mismatch_pu,
            tol_update_pu,
            s_base_va,
            max_iter,
            linear_solver,
            precision,
            fixed_rows,
        )
        if out[3]:
            if attempt:
                _log.info(
                    "solve_power_flow: Newton converged on warm start %d of %d "
                    "(earlier start(s) did not converge).",
                    attempt + 1,
                    len(v_inits),
                )
            return out
        if best is None or float(out[2]) < float(best[2]):
            best = out
    return best


_NEWTON_MAX_BACKTRACK = 20  # line-search step halvings before accepting the Newton step
_GMRES_RESTART = 100  # Krylov subspace dimension before a restart
_GMRES_MAX_RESTARTS = 20  # restart cycles before giving up the inner solve
_GMRES_RTOL = 1.0e-8  # inner-solve relative tolerance (matrix-free path)


def _gmres(
    matvec, b: Tensor, *, rtol: float, restart: int, max_restarts: int
) -> Tensor:
    """Restarted GMRES(``restart``) for ``A x = b``, matrix-free (real 1-D tensors).

    ``matvec(v)`` returns ``A·v``. Solves to ``‖b − A x‖ ≤ rtol·‖b‖`` or after
    ``max_restarts`` cycles. Classic Arnoldi + Givens rotations; autograd-free (the
    Newton forward runs under ``no_grad``).
    """
    nrows = b.shape[0]
    x = torch.zeros_like(b)
    b_norm = torch.linalg.vector_norm(b)
    if float(b_norm) == 0.0:
        return x
    m = min(restart, nrows)
    for _ in range(max_restarts):
        r = b - matvec(x)
        beta = torch.linalg.vector_norm(r)
        if float(beta) <= rtol * float(b_norm):
            return x
        q = torch.zeros((nrows, m + 1), dtype=b.dtype, device=b.device)
        h = torch.zeros((m + 1, m), dtype=b.dtype, device=b.device)
        cs = torch.zeros(m, dtype=b.dtype, device=b.device)
        sn = torch.zeros(m, dtype=b.dtype, device=b.device)
        g = torch.zeros(m + 1, dtype=b.dtype, device=b.device)
        q[:, 0] = r / beta
        g[0] = beta
        k_used = 0
        for k in range(m):
            w = matvec(q[:, k])
            for j in range(k + 1):  # modified Gram-Schmidt
                h[j, k] = torch.dot(q[:, j], w)
                w = w - h[j, k] * q[:, j]
            h[k + 1, k] = torch.linalg.vector_norm(w)
            if float(h[k + 1, k]) > 1e-300:
                q[:, k + 1] = w / h[k + 1, k]
            for j in range(k):  # apply prior Givens rotations to column k
                t = cs[j] * h[j, k] + sn[j] * h[j + 1, k]
                h[j + 1, k] = -sn[j] * h[j, k] + cs[j] * h[j + 1, k]
                h[j, k] = t
            denom = torch.sqrt(h[k, k] ** 2 + h[k + 1, k] ** 2)
            cs[k] = h[k, k] / denom
            sn[k] = h[k + 1, k] / denom
            h[k, k] = cs[k] * h[k, k] + sn[k] * h[k + 1, k]
            h[k + 1, k] = 0.0
            g[k + 1] = -sn[k] * g[k]
            g[k] = cs[k] * g[k]
            k_used = k + 1
            if float(torch.abs(g[k + 1])) <= rtol * float(b_norm):
                break
        y = torch.linalg.solve_triangular(
            h[:k_used, :k_used], g[:k_used].unsqueeze(-1), upper=True
        ).squeeze(-1)
        x = x + q[:, :k_used] @ y
    return x


def _newton_forward(
    real_res,
    v_init,
    n,
    rdt,
    cdt,
    device,
    v_base,
    tol_mismatch_pu,
    tol_update_pu,
    s_base_va,
    max_iter,
    linear_solver="dense",
    precision="full",
    fixed_rows=None,
):
    """Newton on the real residual ``R(x) = 0`` from the warm start ``v_init``.

    Each step solves ``J·Δx = −R`` with ``J = dR/dx`` and a backtracking line search on
    ``‖R‖∞`` for global robustness, converging on the same two PER-UNIT criteria as the
    fixed point (:class:`_PuConvergence`): the apparent-power mismatch carried by the
    free rows of ``R`` and the per-row voltage step. ``linear_solver``:

    - ``"dense"`` (default): the explicit real ``[2N, 2N]`` Jacobian (autograd) + a direct
      solve. Per-element loop avoids the ``[B, 2N, B, 2N]`` memory of a batched Jacobian.
    - ``"matrix_free"``: Jacobian-free Newton-Krylov — never forms ``J``; solves with
      GMRES using a finite-difference Jacobian-vector product
      ``J·v ≈ (R(x+εv) − R(x))/ε``. ``O(N)`` memory, for large grids where the dense
      Jacobian is prohibitive (accuracy is the ``√eps`` FD floor, ample for a PF solve).

    ``precision="mixed"`` solves for the Newton DIRECTION in single precision while the
    residual, the line search and the step stay at the working precision — an inexact
    Newton method: a direction with a relative error of ``cond(J)·eps_single`` changes
    the contraction rate, not the solution the iteration converges to.

    Returns the same 12-tuple as :func:`_current_injection_forward` (its last entry, the
    nodal residual, is the complex residual rebuilt from the free rows of ``R``).
    """
    state_residual = real_res.state_residual
    build_system = real_res.build_system
    twon = 2 * n
    fd_eps = math.sqrt(torch.finfo(rdt).eps)
    with torch.no_grad():
        y_eff0, i_slack0 = build_system()
        y_eff0 = y_eff0.detach()
        i_slack0 = i_slack0.detach()
        lead = v_init.shape[:-1]
        b = int(torch.tensor(lead).prod().item()) if lead else 1
        x = torch.cat([v_init.real, v_init.imag], dim=-1).reshape(b, twon).to(rdt)
        y_flat = y_eff0.reshape(-1, n, n)
        y_flat = y_flat.expand(b, n, n) if y_flat.shape[0] == 1 else y_flat
        is_flat = i_slack0.reshape(-1, n)
        is_flat = is_flat.expand(b, n) if is_flat.shape[0] == 1 else is_flat
        ctest = _PuConvergence(
            v_base=v_base,
            s_base=s_base_va,
            tol_mismatch_pu=tol_mismatch_pu,
            tol_update_pu=tol_update_pu,
            floor_update=_rel_convergence_floor(rdt),
            floor_mismatch=_mismatch_floor_rel(rdt),
            fixed_rows=fixed_rows,
            y_eff=y_eff0,
            n=n,
            device=device,
            rdt=rdt,
        )

        def res_all(xx: Tensor) -> Tensor:
            return state_residual(
                xx, y_flat.real, y_flat.imag, is_flat.real, is_flat.imag
            )

        def res_one(xb_1d: Tensor, bi: int) -> Tensor:
            return state_residual(
                xb_1d,
                y_flat[bi].real,
                y_flat[bi].imag,
                is_flat[bi].real,
                is_flat[bi].imag,
            )

        def pu_measures(xx: Tensor, rr: Tensor, dxx: Tensor):
            """Per-unit mismatch / update rows of the real state and residual."""
            v_c = torch.complex(xx[..., :n], xx[..., n:])
            fc_c = torch.complex(rr[..., :n], rr[..., n:])
            dv_c = torch.complex(dxx[..., :n], dxx[..., n:])
            return (
                ctest.mismatch_rows_pu(v_c, fc_c),
                ctest.update_rows_pu(dv_c),
                fc_c,
            )

        update_history: list[float] = []
        converged = False
        iterations = 0
        mismatch_max = torch.zeros((), dtype=rdt, device=device)
        update_max = torch.zeros((), dtype=rdt, device=device)
        update_norm_v = torch.zeros((), dtype=rdt, device=device)
        converged_mask = torch.zeros(b, dtype=torch.bool, device=device)
        mismatch_vec = torch.zeros(b, dtype=rdt, device=device)
        r = res_all(x)  # [b, 2N]
        fc = torch.complex(r[..., :n], r[..., n:])
        for _ in range(max_iter):
            if linear_solver == "matrix_free":
                dx = _newton_dir_matrix_free(res_one, x, r, b, fd_eps)
            else:
                dx = _newton_dir_dense(res_all, x, r, precision=precision)
            # PER-ELEMENT backtracking on each scenario's residual infinity-norm
            # (global robustness): a hard scenario halves only its own step.
            r0 = r.abs().amax(dim=-1)  # [b]
            step = torch.ones(b, 1, dtype=rdt, device=x.device)
            for _bt in range(_NEWTON_MAX_BACKTRACK):
                ok = res_all(x + step * dx).abs().amax(dim=-1) <= r0  # [b]
                if bool(ok.all()):
                    break
                step = torch.where(ok.unsqueeze(-1), step, 0.5 * step)
            dx_taken = step * dx
            x = x + dx_taken
            # The residual at the NEW state feeds both the per-unit mismatch criterion
            # and the next step's Jacobian / line search (evaluated once).
            r = res_all(x)
            mism_rows, upd_rows, fc = pu_measures(x, r, dx_taken)
            converged_mask, mismatch_vec, update_vec = ctest.check(mism_rows, upd_rows)
            mismatch_max = mismatch_vec.max()
            update_max = update_vec.max()
            update_norm_v = dx_taken.norm(dim=-1).max()
            update_history.append(float(update_max))
            iterations += 1
            if bool(converged_mask.all()):
                converged = True
                break
        v_star = torch.complex(x[..., :n], x[..., n:]).reshape(*lead, n)
        cmask = converged_mask.reshape(lead) if lead else converged_mask.reshape(())
        mvec = mismatch_vec.reshape(lead) if lead else mismatch_vec.reshape(())
        fc = fc.reshape(*lead, n) if lead else fc.reshape(n)
    return (
        v_star,
        iterations,
        mismatch_max,
        converged,
        update_history,
        y_eff0,
        i_slack0,
        cmask,
        mvec,
        update_max,
        update_norm_v,
        fc,
    )


def _newton_forward_sequential(
    grid,
    index,
    device,
    slack,
    operating_point,
    fixed_rows,
    build_system,
    make_fast_residual_complex,
    warm_starts,
    bsize,
    pv,
    n,
    rdt,
    cdt,
    v_base,
    tol_mismatch_pu,
    tol_update_pu,
    s_base_va,
    max_iter,
    linear_solver,
    precision="full",
):
    """Batched Newton by solving each scenario with the single-grid Newton forward.

    Newton's warm starts and per-element Jacobian are single-grid (the residual
    closes over the operating point), so a batched operating point is handled by slicing
    it per scenario, running the proven single-grid forward, and stacking the detached
    ``V*`` ``[B, N]``. The IFT backward (full op, batch-aligned, block-diagonal) attaches
    batched gradients to the stacked result, so this is forward-only sequencing — the
    differentiability is unchanged. The per-scenario residual comes from
    ``make_fast_residual_complex`` (one detached injection plan per slice — the
    detached forward needs only ``dR/dx``), and the voltage-regulating terminals are
    sliced the same way (per-scenario setpoints and active set). Returns the same
    12-tuple as :func:`_newton_forward`.
    """
    v_list, conv_list, mis_list, upd_list, updv_list, fc_list = [], [], [], [], [], []
    iterations = 0
    y_eff0 = i_slack0 = None
    for i in range(bsize):
        op_i = _slice_operating_point(operating_point, i)
        rc_i = make_fast_residual_complex(op_i)

        # Per-scenario slack reference: a source ``u_ref`` scale makes ``v_fixed``
        # per-scenario, so the residual + warm start use THIS scenario's slice (not the
        # full-batch ``v_fixed`` / ``v_fixed_fn``). Reduces to the shared reference when
        # no scale is present.
        def _vfixed_i(op=op_i):
            if slack != "ideal":
                return None
            return _slack_rows_and_vref(grid, index, rdt, cdt, device, op)[1]

        pv_i = None if pv is None else pv.slice(i)
        rr_i = _make_real_residual(
            build_system, rc_i, fixed_rows, _vfixed_i, n, cdt, pv=pv_i
        )
        (
            v_i,
            it_i,
            mis_i,
            cv_i,
            _,
            y_eff0,
            i_slack0,
            _,
            _,
            upd_i,
            updv_i,
            fc_i,
        ) = _newton_from_starts(
            rr_i,
            warm_starts(op_i, pv_i, _vfixed_i()),
            n,
            rdt,
            cdt,
            device,
            v_base,
            tol_mismatch_pu,
            tol_update_pu,
            s_base_va,
            max_iter,
            linear_solver,
            precision,
            fixed_rows,
        )
        v_list.append(v_i)  # [N]
        conv_list.append(bool(cv_i))
        mis_list.append(mis_i.reshape(()))
        upd_list.append(upd_i.reshape(()))
        updv_list.append(updv_i.reshape(()))
        fc_list.append(fc_i.reshape(-1)[:n])
        iterations = max(iterations, it_i)
    v_star = torch.stack(v_list, 0)  # [B, N]
    converged_mask = torch.tensor(conv_list, dtype=torch.bool, device=device)  # [B]
    mismatch_vec = torch.stack(mis_list, 0)  # [B]
    converged = bool(converged_mask.all())
    # Per-scenario update histories are not aggregated (their lengths differ); the cheap
    # state diagnostics + the per-scenario mismatch carry the per-batch detail.
    return (
        v_star,
        iterations,
        mismatch_vec.max(),
        converged,
        [],
        y_eff0,
        i_slack0,
        converged_mask,
        mismatch_vec,
        torch.stack(upd_list, 0).max(),
        torch.stack(updv_list, 0).max(),
        torch.stack(fc_list, 0),
    )


def _batched_state_jacobian(batched_state_res, x_flat: Tensor) -> Tensor:
    """Block-diagonal real state Jacobian ``J = dR/dx`` ``[B, 2N, 2N]``.

    ``batched_state_res`` maps ``x [B, 2N] -> R [B, 2N]`` where ``R[k]`` depends
    only on ``x[k]``, so the full Jacobian is block diagonal (the off-diagonal
    cross terms are zero). Two ways to get the blocks:

    - small ``B``: differentiate the batched map once (vectorized) and slice the
      diagonal — fast, but the intermediate is ``[B, 2N, B, 2N]`` (O(B²) memory);
    - large ``B``: build the diagonal column-by-column with ``2N`` batched JVPs —
      O(B) memory, using the SAME batch-aligned residual (correct for every batch
      source, incl. batched device params / operating points).

    Both avoid vmap, which does not compose with the assembly's ``index_add_``
    scatter. Shared by the Newton forward and the IFT backward.
    """
    b, twon = x_flat.shape
    if b * b * twon * twon <= _IFT_DENSE_JAC_MAX_ELEMS:
        jac_full = torch.autograd.functional.jacobian(
            batched_state_res, x_flat, create_graph=False, vectorize=True
        )  # [B, 2N, B, 2N]
        idx_b = torch.arange(b, device=x_flat.device)
        return jac_full[idx_b, :, idx_b, :]  # [B, 2N, 2N]
    cols = []
    for j in range(twon):
        tangent = torch.zeros_like(x_flat)
        tangent[:, j] = 1.0
        _, col = torch.autograd.functional.jvp(
            batched_state_res, x_flat, v=tangent
        )  # [B, 2N] = J[..., j]
        cols.append(col)
    return torch.stack(cols, dim=-1)  # [B, 2N, 2N]


def _newton_dir_dense(batched_state_res, x, r, *, precision: str = "full") -> Tensor:
    """Dense Newton direction ``Δx`` solving ``J Δx = −R`` per batch element.

    ``precision="mixed"`` solves the Jacobian system in single precision (an inexact
    Newton direction: the residual and the step stay at the working precision, so the
    iteration converges to the same solution with a slightly degraded rate).
    """
    j = _batched_state_jacobian(batched_state_res, x)  # [b, 2N, 2N]
    if precision == "mixed":
        sdt = torch.float32
        dx = torch.linalg.solve(j.to(sdt), -r.unsqueeze(-1).to(sdt)).squeeze(-1)
        return dx.to(r.dtype)
    return torch.linalg.solve(j, -r.unsqueeze(-1)).squeeze(-1)  # [b, 2N]


def _newton_dir_matrix_free(res_one, x, r, b, fd_eps) -> Tensor:
    """Jacobian-free Newton direction: GMRES with a finite-difference ``J·v``."""
    dx_rows = []
    for bi in range(b):
        xb = x[bi]
        r0 = r[bi]
        x_norm = torch.linalg.vector_norm(xb)

        def matvec(v, xb=xb, r0=r0, x_norm=x_norm, bi=bi):
            nv = torch.linalg.vector_norm(v)
            if float(nv) == 0.0:
                return torch.zeros_like(v)
            eps = fd_eps * (1.0 + float(x_norm)) / float(nv)
            return (res_one(xb + eps * v, bi) - r0) / eps

        dx_rows.append(
            _gmres(
                matvec,
                -r0,
                rtol=_GMRES_RTOL,
                restart=_GMRES_RESTART,
                max_restarts=_GMRES_MAX_RESTARTS,
            )
        )
    return torch.stack(dx_rows, 0)  # [b, 2N]


# ---------------------------------------------------------------------------
# real-coordinate residual + IFT autograd.Function
# ---------------------------------------------------------------------------
def _make_real_residual(
    build_system,
    residual_complex,
    fixed_rows: Optional[Tensor],
    v_fixed_fn,
    n: int,
    cdt: torch.dtype,
    state_residual_complex=None,
    pv: Optional[PVTerminals] = None,
):
    """Return a closure ``R(x) -> [*b, 2N]`` real residual with slack pinning.

    ``x = [Re(V); Im(V)]``. Free rows hold ``Re/Im(F_c)``; ideal-slack rows hold
    ``Re/Im(V - V_fixed)``. The complex system tensors come from ``build_system``
    and the slack reference from ``v_fixed_fn`` (recomputed each call) so they
    stay differentiable w.r.t. the parameter leaves and do not reuse a freed graph.

    ``state_residual_complex`` (optional) is a faster complex residual used ONLY by
    the attached ``state_residual`` — the fixed-system form the Newton iterations,
    the state Jacobian ``J = dR/dx``, and the criticality analysis evaluate. Those
    differentiate w.r.t. ``x`` alone, so a plan-based residual with detached
    parameter tensors is exact there; the full ``real_residual`` keeps the
    differentiable ``residual_complex`` for the ``dR/dθ`` vjp.

    ``pv`` (optional) are the voltage-regulating terminals: their row pair is
    substituted (active balance + voltage setpoint) before the slack pinning, in both
    the full and the state residual, so the Newton step, the IFT Jacobian and the
    adjoint all see the PV equations (:mod:`pgml.solver._pv_bus`).
    """
    state_rc = (
        state_residual_complex
        if state_residual_complex is not None
        else residual_complex
    )

    def _pin_and_split(
        fc: Tensor, v: Tensor, x: Tensor, pv_state: Optional[PVTerminals]
    ) -> Tensor:
        if pv_state is not None:
            fc = pv_state.transform(fc, v)
        v_fixed = v_fixed_fn() if v_fixed_fn is not None else None
        if fixed_rows is not None and v_fixed is not None:
            vf = v_fixed.to(dtype=cdt, device=x.device)  # [S]
            v_at_fixed = v.index_select(-1, fixed_rows)  # [*b, S]
            pin = v_at_fixed - vf  # [*b, S]
            lead = torch.broadcast_shapes(fc.shape[:-1], pin.shape[:-1])
            s = fixed_rows.shape[0]
            fc_b = fc.broadcast_to(*lead, n)
            idx = fixed_rows.expand(*lead, s)
            fc = fc_b.scatter(-1, idx, pin.broadcast_to(*lead, s))
        return torch.cat([fc.real, fc.imag], dim=-1)  # [*b, 2N]

    def real_residual(x: Tensor) -> Tensor:
        """Full real residual; rebuilds the differentiable system from leaves."""
        v_re = x[..., :n]
        v_im = x[..., n:]
        v = torch.complex(v_re, v_im).to(cdt)  # [*b, N]
        y_eff, i_slack = build_system()
        fc = residual_complex(v, y_eff, i_slack)  # [*b, N] complex (all rows)
        return _pin_and_split(fc, v, x, pv)

    def state_residual(
        x: Tensor,
        y_re: Tensor,
        y_im: Tensor,
        islack_re: Tensor,
        islack_im: Tensor,
        *,
        rc=None,
        pv_state: Optional[PVTerminals] = None,
    ) -> Tensor:
        """Real residual at FIXED (real-split) system tensors — for the state Jacobian.

        ``jacrev`` rejects complex inputs, so the (constant) system tensors are
        passed as real/imag pairs and recombined here. Only ``x`` is differentiated.
        ``rc`` overrides the complex state residual (the IFT backward passes a
        flattened-plan variant when the state batch is collapsed to one dim), and
        ``pv_state`` the regulating terminals (flattened the same way).
        """
        v_re = x[..., :n]
        v_im = x[..., n:]
        v = torch.complex(v_re, v_im).to(cdt)
        y_eff = torch.complex(y_re, y_im).to(cdt)
        i_slack = torch.complex(islack_re, islack_im).to(cdt)
        fc = (rc if rc is not None else state_rc)(v, y_eff, i_slack)
        return _pin_and_split(fc, v, x, pv if pv_state is None else pv_state)

    real_residual.state_residual = state_residual
    real_residual.state_rc = state_rc
    real_residual.build_system = build_system
    real_residual.pv = pv
    return real_residual


class _IFTPowerFlow(torch.autograd.Function):
    """Attach the IFT gradient to a detached converged ``V*``.

    ``forward`` returns ``V*`` unchanged. ``backward`` builds the real
    ``[2N, 2N]`` Jacobian of the real residual at ``V*``, solves the adjoint
    ``J^T λ = grad_x`` (one solve per batch system), then forms the parameter
    gradients ``-(dR/dθ)^T λ`` via a single vjp of the residual at ``V*``.

    The captured ``*leaves`` are the true autograd leaves. A Grid field may be a
    derived expression (``q_nom_var = p * k``) sharing history with the outer
    autograd tape, so the residual reaches the leaves THROUGH that shared history;
    differentiating at the leaves (not the intermediates) keeps a leaf feeding
    several fields from being double counted, and the vjp keeps the shared graph
    alive (``retain_graph=True``) for the outer engine.
    """

    @staticmethod
    def forward(ctx, v_star, real_res, n, rdt, cdt, *leaves):
        ctx.real_res = real_res
        ctx.n = n
        ctx.rdt = rdt
        ctx.cdt = cdt
        ctx.num_leaves = len(leaves)
        ctx.save_for_backward(v_star, *leaves)
        return v_star

    @staticmethod
    def backward(ctx, grad_v):
        saved = ctx.saved_tensors
        v_star = saved[0]
        leaves = saved[1 : 1 + ctx.num_leaves]
        n = ctx.n
        rdt = ctx.rdt
        real_res = ctx.real_res
        state_residual = real_res.state_residual
        build_system = real_res.build_system
        twon = 2 * n

        # x* = [Re(V*); Im(V*)] -> [*b, 2N]
        x_star = torch.cat([v_star.real, v_star.imag], dim=-1).to(rdt)
        grad_x = torch.cat([grad_v.real, grad_v.imag], dim=-1).to(rdt)  # [*b, 2N]
        lead = x_star.shape[:-1]
        b = int(torch.tensor(lead).prod().item()) if lead else 1

        # Detached, per-batch system tensors for the STATE Jacobian (J = dR/dx).
        with torch.no_grad():
            y_eff, i_slack = build_system()  # [1,N,N] (or [*b,N,N]); [N] or [*b,N]
        y_eff = y_eff.detach()
        i_slack = i_slack.detach()
        x_flat = x_star.reshape(b, twon)
        gx_flat = grad_x.reshape(b, twon)
        y_flat = y_eff.reshape(-1, n, n)
        y_flat = y_flat.expand(b, n, n) if y_flat.shape[0] == 1 else y_flat
        islack_flat = i_slack.reshape(-1, n)
        islack_flat = (
            islack_flat.expand(b, n) if islack_flat.shape[0] == 1 else islack_flat
        )

        # Real state Jacobian J = dR/dx at x*, per batch system. The batched residual
        # R[k] depends only on x[k], so the Jacobian is block diagonal. Two ways to get
        # the [B, 2N, 2N] blocks (the off-diagonal cross terms are zero):
        #   - small B: differentiate the batched map (vectorized) and slice the diagonal
        #     — fast, but the intermediate is [B, 2N, B, 2N] (O(B²) memory);
        #   - large B: build the diagonal column-by-column with 2N batched JVPs — O(B)
        #     memory, and uses the SAME batch-aligned residual (so it stays correct for
        #     EVERY batch source, incl. batched device params / operating points).
        # Both avoid vmap, which does not compose with the assembly's index_add_ scatter.

        # The state Jacobian runs over a SINGLE flattened [B*T] scenario axis, but a
        # per-step (profiled) operating point gives the plan a [B, T] power batch. Flatten
        # that plan's power to match so each flattened row keeps its own scenario power
        # (a no-op for a scalar / already-1-D operating-point batch — the plan broadcasts).
        rc_flat = None
        state_rc = getattr(real_res, "state_rc", None)
        plan = getattr(state_rc, "plan", None)
        if plan is not None and len(lead) > 1:
            flat_plan = flatten_plan_batch(plan, lead)

            def rc_flat(v, y_eff, i_slack, _p=flat_plan):
                return (
                    _apply_y(y_eff, v)
                    + injections_from_plan(_p, v).squeeze(-2)
                    - i_slack
                )

        # The voltage-regulating terminals' setpoints / limits carry the same scenario
        # batch, so they are collapsed onto the flattened axis alongside the plan.
        pv_flat = getattr(real_res, "pv", None)
        if pv_flat is not None and len(lead) > 1:
            pv_flat = pv_flat.flatten_batch(lead)

        def batched_state_res(xb):
            return state_residual(
                xb,
                y_flat.real,
                y_flat.imag,
                islack_flat.real,
                islack_flat.imag,
                rc=rc_flat,
                pv_state=pv_flat,
            )  # [B, 2N]

        j_batched = _batched_state_jacobian(batched_state_res, x_flat)  # [B, 2N, 2N]
        # Adjoint: J^T λ = grad_x  ->  λ = J^{-T} grad_x  (batched solve).
        lam_flat = torch.linalg.solve(
            j_batched.transpose(-1, -2), gx_flat.unsqueeze(-1)
        ).squeeze(-1)  # [B, 2N]
        lam = lam_flat.reshape(*lead, twon) if lead else lam_flat.reshape(twon)

        # grad_theta = -(dR/dθ)^T λ via a single residual vjp at x* (θ tracking).
        # ``leaves`` are the true autograd leaves (see ``_grid_param_leaves``); the residual
        # reaches them THROUGH any derived-parameter intermediates a Grid field holds
        # (float/tensor duality: ``q_nom_var = p * k``). Because the leaves have no history,
        # this single grad gives each an unambiguous total — no double count from a leaf that
        # feeds several fields — and the outer engine attaches directly to the leaves.
        x_const = x_star.detach()
        with torch.enable_grad():
            r_theta = real_res(x_const)  # [*b', 2N]; depends on the leaves
            # r_theta may carry a broadcast singleton batch dim; align λ to it.
            grad_out = (-lam).reshape(r_theta.shape).to(r_theta.dtype)
            # retain_graph=True: those derived-parameter intermediates were built in the
            # caller's forward pass, so their history is shared with the outer autograd tape.
            # Freeing it here (retain_graph=False) would break the FIRST outer .backward()
            # ("backward through the graph a second time") whenever such an intermediate is
            # also used elsewhere on the outer tape. The freshly built residual sub-graph is
            # dropped normally on scope exit; the outer engine owns the shared history.
            if r_theta.requires_grad:
                grads = torch.autograd.grad(
                    r_theta,
                    leaves,
                    grad_outputs=grad_out,
                    retain_graph=True,
                    allow_unused=True,
                )
            else:
                # No collected leaf reaches the fundamental residual, so every
                # parameter gradient is structurally zero. This happens for a
                # parameter the fundamental system does not contain — e.g. a source
                # series impedance under IDEAL slack, which the pinned slack row makes
                # irrelevant at f0 while the harmonic orders (Norton-stamped) do depend
                # on it. Asking autograd for a gradient of a constant would raise.
                grads = (None,) * len(leaves)

        grad_leaves = tuple(
            g if g is not None else torch.zeros_like(leaf)
            for g, leaf in zip(grads, leaves)
        )
        # forward inputs: (v_star, real_res, n, rdt, cdt, *leaves)
        return (None, None, None, None, None, *grad_leaves)


# ---------------------------------------------------------------------------
# convergence diagnostics (autograd-free) + IFT-Jacobian criticality
# ---------------------------------------------------------------------------
# Diagnostic thresholds — flagging only, NOT modelling decisions.
_DIAG_VBAND_PU = (0.8, 1.2)  # |V|/V_LN outside this band is flagged
_DIAG_VBLOWUP = 5.0  # |V|/V_LN above this (or non-finite) = diverged iterate
# Above this many elements, the IFT backward's dense [B,2N,B,2N] state Jacobian is
# replaced by an O(B) column-by-column JVP build (avoids the B² memory blow-up at the
# cost of 2N batched JVPs). ~2e8 real elems = ~1.6 GB at float64.
_IFT_DENSE_JAC_MAX_ELEMS = 2 * 10**8

_DIAG_TOP_K = 5  # worst offenders / critical nodes reported
_DIAG_COND_SINGULAR = 1.0e8  # Jacobian condition number above this ~ near-singular
_DIAG_MAX_2N = 4000  # skip the dense criticality SVD above this real-state size
_PHASE_NAMES = ("A", "B", "C", "N")


def _phase_name(code: int) -> str:
    return _PHASE_NAMES[code] if 0 <= code < len(_PHASE_NAMES) else "?"


def _node_voltage_bases(grid: Grid, index, rdt, device) -> Tensor:
    """Per-row line-to-neutral voltage base ``[N]`` (the per-unit denominator)."""
    from pgml.assembly._params import phase_voltage_magnitude

    node_by_id = {int(nd.id): nd for nd in grid.nodes}
    bases = [
        phase_voltage_magnitude(
            float(node_by_id[int(nid)].u_rated_v), len(node_by_id[int(nid)].phases)
        )
        for nid in index.node_ids.tolist()
    ]
    return torch.tensor(bases, dtype=rdt, device=device)


def _build_diagnostics(
    grid,
    index,
    v_star,
    fc_star,
    v_base,
    s_base_va,
    real_res,
    fixed_rows,
    residual_history,
    converged,
    iterations,
    mismatch_max_pu,
    update_max_pu,
    update_norm_v,
    rdt,
    device,
    criticality: str = "auto",
    pv: Optional[PVTerminals] = None,
) -> "ConvergenceDiagnostics":
    """Cheap state diagnostics at ``V*`` (+ Jacobian criticality on non-convergence).

    ``fc_star`` is the nodal residual ``F = Y_eff V* + I_device(V*) - I_slack`` at
    ``V*``, so the diagnostics add no further solve or assembly; the per-unit criterion
    VALUES come from the forward and are reported next to their SI counterparts.

    ``pv`` (the voltage-regulating terminals) makes the reported nodal mismatch
    meaningful at a regulated row: the RAW current mismatch there is the reactive
    current the generator supplies, so a REGULATING row reports the ACTIVE component
    ``|Re(conj(V) F_c)| / |V|`` — the part its row pair enforces — while a row pinned
    at a reactive limit keeps its full mismatch.
    """
    n = index.size
    vmin, vmax = _DIAG_VBAND_PU
    node_ids = index.node_ids.tolist()
    phase_codes = index.phase_codes.tolist()
    with torch.no_grad():
        lead = v_star.shape[:-1]
        b = int(torch.tensor(lead).prod().item()) if lead else 1
        vflat = v_star.reshape(b, n)
        # A batched solve can carry a size-one scenario axis that the residual's own
        # reshape collapsed (``[B, 1, N]`` voltages against a ``[B, N]`` residual), so
        # the residual is realigned to the voltage's batch shape before it is read
        # per row — `active_power_mismatch` broadcasts against ``v_star``.
        fc_rows = fc_star.reshape(v_star.shape)
        mismatch = (
            fc_rows.abs() if pv is None else active_power_mismatch(pv, fc_rows, v_star)
        )
        fc_abs = mismatch.reshape(b, n).clone()
        if fixed_rows is not None and fixed_rows.numel() > 0:
            fc_abs[:, fixed_rows] = 0.0  # slack rows absorb mismatch by construction
        bases = v_base.clamp_min(1e-12)
        vpu = vflat.abs() / bases[None, :]  # [B, N]
        mismatch_va = vflat.abs() * fc_abs  # [B, N] apparent-power mismatch
        mismatch_max_a = float(fc_abs.max()) if fc_abs.numel() else 0.0
        mismatch_max_va = float(mismatch_va.max()) if mismatch_va.numel() else 0.0

        worst_nodes: list[dict] = []
        k = min(_DIAG_TOP_K, b * n)
        if k > 0 and mismatch_max_va > 0.0:
            vals, idxs = torch.topk(mismatch_va.reshape(-1), k)
            for val, fi in zip(vals.tolist(), idxs.tolist()):
                bi, r = divmod(int(fi), n)
                worst_nodes.append(
                    {
                        "node_id": int(node_ids[r]),
                        "phase": _phase_name(int(phase_codes[r])),
                        "mismatch_pu": float(val) / s_base_va,
                        "mismatch_va": float(val),
                        "mismatch_a": float(fc_abs[bi, r]),
                        "v_pu": float(vpu[bi, r]),
                        **({"batch": bi} if b > 1 else {}),
                    }
                )

        pc = torch.tensor(phase_codes, device=device)
        nonneutral = (pc != 3)[None, :]
        below, above = (vpu < vmin) & nonneutral, (vpu > vmax) & nonneutral
        dev = torch.where(below, vmin - vpu, torch.zeros_like(vpu))
        dev = torch.where(above, vpu - vmax, dev)  # >0 where violated
        n_viol = int((dev > 0).sum())
        out_of_band: list[dict] = []
        if n_viol > 0:
            vals, idxs = torch.topk(dev.reshape(-1), min(_DIAG_TOP_K, n_viol))
            for val, fi in zip(vals.tolist(), idxs.tolist()):
                if val <= 0.0:
                    continue
                bi, r = divmod(int(fi), n)
                out_of_band.append(
                    {
                        "node_id": int(node_ids[r]),
                        "phase": _phase_name(int(phase_codes[r])),
                        "v_pu": float(vpu[bi, r]),
                        **({"batch": bi} if b > 1 else {}),
                    }
                )

        # Divergence guard: the current-injection fixed point does not stop at the
        # loadability nose — past it (or for a non-contractive map) the iterate blows
        # up. At such an unphysical / non-finite V the Jacobian is not a meaningful
        # loadability test, so we detect it and skip / caveat the criticality analysis.
        finite = bool(torch.isfinite(vflat).all())
        max_vpu = (
            float((vpu * nonneutral.to(vpu.dtype)).max()) if finite else float("inf")
        )
        diverged = (not finite) or (max_vpu > _DIAG_VBLOWUP)
        # The per-unit update IS the relative measure: a large one means the iterate
        # never settled (oscillating), so any band violation on it is an artifact.
        rel_update = update_max_pu if finite else float("inf")

    diag = ConvergenceDiagnostics(
        converged=converged,
        iterations=iterations,
        mismatch_max_pu=mismatch_max_pu,
        update_max_pu=update_max_pu,
        mismatch_max_va=mismatch_max_va,
        mismatch_max_a=mismatch_max_a,
        update_norm_v=update_norm_v,
        s_base_va=s_base_va,
        voltage_band_pu=(vmin, vmax),
        residual_history=residual_history,
        worst_nodes=worst_nodes,
        out_of_band_nodes=out_of_band,
    )
    do_crit = criticality == "always" or (criticality == "auto" and not converged)
    if do_crit and finite and b == 1:
        diag.criticality = _jacobian_criticality(
            real_res, v_star, n, rdt, device, index, fc_abs, diverged
        )
    elif do_crit and b > 1:
        # The IFT-Jacobian criticality is a single-grid loadability diagnostic; on a
        # scenario BATCH the operating point reduces the residual per element, so it is
        # skipped (run a single grid, or use ``loadability_limit``, for the analysis).
        _log.info(
            "criticality analysis skipped for a batched solve (b=%d); it is a "
            "single-grid diagnostic. Re-run one scenario for the Jacobian/SVD.",
            b,
        )
    diag.likely_cause = _likely_cause(diag, n_viol, diverged, max_vpu, rel_update)
    return diag


def _likely_cause(
    diag: "ConvergenceDiagnostics",
    n_viol: int,
    diverged: bool,
    max_vpu: float,
    rel_update: float,
) -> str:
    """One-line heuristic explanation of the convergence outcome."""
    if diag.converged:
        return "converged"
    if diverged:
        reached = (
            f"voltage reached {max_vpu:.1f} pu"
            if math.isfinite(max_vpu)
            else "voltages became non-finite"
        )
        return (
            f"fixed-point iteration diverged ({reached} — unphysical); the operating "
            "point is likely past the loadability limit, or the current-injection map "
            "is non-contractive here — locate the limit with continuation from a "
            "feasible base"
        )
    crit = diag.criticality or {}
    if crit.get("near_singular"):
        names = ", ".join(
            f"{c['node_id']}.{c['phase']}" for c in crit.get("critical_nodes", [])[:3]
        )
        return (
            f"voltage collapse / loadability limit — Jacobian near-singular "
            f"(cond={crit.get('condition_number', float('nan')):.1e}); "
            f"critical node(s): {names}"
        )
    if rel_update > 0.01:  # iterate never settled (oscillating update vs rated voltage)
        return (
            f"fixed-point iteration did not settle (oscillating; final per-row update "
            f"{diag.update_max_pu:.2e} pu, {diag.update_norm_v:.2e} V) — the "
            "current-injection map is not contracting here; try Newton or a "
            "homotopy continuation from a feasible base"
        )
    if n_viol > 0 and diag.out_of_band_nodes:
        worst = min(diag.out_of_band_nodes, key=lambda d: d["v_pu"])
        lo, hi = diag.voltage_band_pu
        return (
            f"{n_viol} node(s) outside [{lo}, {hi}] pu (likely overload / weak source); "
            f"worst {worst['node_id']}.{worst['phase']} at {worst['v_pu']:.3f} pu"
        )
    hist = diag.residual_history
    if len(hist) >= 3 and hist[-1] >= hist[-3]:  # not contracting
        return (
            "fixed-point iteration not contracting (per-row voltage update plateaued at "
            f"{hist[-1]:.2e} pu); a solution may exist — try Newton / a better start / "
            "more iterations"
        )
    wn = diag.worst_nodes[0] if diag.worst_nodes else None
    tail = (
        f"; max mismatch {wn['mismatch_pu']:.2e} pu ({wn['mismatch_a']:.2e} A) at node "
        f"{wn['node_id']}.{wn['phase']}"
        if wn
        else ""
    )
    return (
        f"did not reach tol in {diag.iterations} iterations (power mismatch "
        f"{diag.mismatch_max_pu:.2e} pu, per-row update {diag.update_max_pu:.2e} pu)"
        f"{tail}"
    )


def _jacobian_criticality(
    real_res, v_star, n, rdt, device, index, fc_abs, diverged: bool = False
) -> dict:
    """IFT real Jacobian ``J = dR/dx`` at ``V*`` -> proximity to voltage collapse.

    Builds the same ``[2N, 2N]`` real residual Jacobian the IFT backward uses (for the
    worst batch element), takes its singular values, and reads the critical-bus
    participation from the right singular vector of the SMALLEST singular value (the
    collapse mode). A near-singular ``J`` means a genuine loadability limit and names
    the weakest bus; a well-conditioned ``J`` means the fixed-point map merely failed
    to contract though a solution likely exists.

    The verdict is rigorous AT (or near) a solution. When ``diverged`` the iterate is
    unphysical, so ``J`` there is only a local linearization — the result is annotated
    and the loadability verdict must come from a continuation from a feasible base.
    """
    twon = 2 * n
    if twon > _DIAG_MAX_2N:
        return {
            "skipped": f"state size 2N={twon} exceeds {_DIAG_MAX_2N}; "
            "use a sparse / matrix-free criticality method"
        }
    state_residual = real_res.state_residual
    build_system = real_res.build_system
    lead = v_star.shape[:-1]
    b = int(torch.tensor(lead).prod().item()) if lead else 1
    vflat = v_star.reshape(b, n)
    b_star = int(fc_abs.max(dim=1).values.argmax()) if b > 1 else 0
    v_b = vflat[b_star]
    x = torch.cat([v_b.real, v_b.imag]).to(rdt)  # [2N]
    with torch.no_grad():
        y_eff, i_slack = build_system()
    y_eff = y_eff.detach().reshape(-1, n, n)
    i_slack = i_slack.detach().reshape(-1, n)
    y_b = y_eff[b_star] if y_eff.shape[0] > b_star else y_eff[0]
    is_b = i_slack[b_star] if i_slack.shape[0] > b_star else i_slack[0]

    def f(xb: Tensor) -> Tensor:
        return state_residual(xb, y_b.real, y_b.imag, is_b.real, is_b.imag)

    j = torch.autograd.functional.jacobian(f, x, vectorize=True)  # [2N, 2N]
    if j.numel() != twon * twon:
        # A residual closure built over a scenario batch returns [B, 2N], so its
        # Jacobian carries that leading axis. A size-one batch is still ONE grid, so
        # fold the singleton away and analyse it; anything else is not a single-grid
        # state and the diagnostic reports that instead of raising.
        return {
            "skipped": (
                f"the residual Jacobian came out with shape {tuple(j.shape)} instead of "
                f"({twon}, {twon}); the criticality analysis is a SINGLE-grid diagnostic. "
                "Re-run one scenario for the Jacobian/SVD."
            )
        }
    j = j.reshape(twon, twon)
    # The decomposition is run on the Jacobian of a state that is, by construction, the
    # hard case — a non-converged or barely-converged solve, where J is ill-conditioned or
    # carries repeated singular values. That is exactly where LAPACK's divide-and-conquer
    # driver can fail to converge, and whether it does depends on the BLAS threading of the
    # host. A DIAGNOSTIC must never take down the run it is explaining: report what could
    # not be computed instead.
    try:
        with torch.no_grad():
            svals = torch.linalg.svdvals(j)
            sigma_min, sigma_max = float(svals.min()), float(svals.max())
            cond = sigma_max / max(sigma_min, 1e-300)
            _, _, vh = torch.linalg.svd(j)
            mode = vh[-1]  # right singular vector of the smallest singular value
            part = torch.sqrt(
                mode[:n] ** 2 + mode[n:] ** 2
            )  # [N] per-node participation
            part = part / part.max().clamp_min(1e-30)
            vals, idxs = torch.topk(part, min(_DIAG_TOP_K, n))
    except torch.linalg.LinAlgError as exc:
        return {
            "skipped": (
                "the singular-value decomposition of the Jacobian did not converge "
                f"({exc.__class__.__name__}); the state is too ill-conditioned for this "
                "diagnostic. The convergence verdict and the residual figures above are "
                "unaffected."
            )
        }
    node_ids, phase_codes = index.node_ids.tolist(), index.phase_codes.tolist()
    critical = [
        {
            "node_id": int(node_ids[int(r)]),
            "phase": _phase_name(int(phase_codes[int(r)])),
            "participation": float(v),
        }
        for v, r in zip(vals.tolist(), idxs.tolist())
    ]
    near_singular = (cond > _DIAG_COND_SINGULAR) and not diverged
    if diverged:
        interp = (
            "evaluated at a DIVERGED (unphysical) iterate — J here is only a local "
            "linearization, not a loadability test; use continuation from a feasible "
            "base to locate the limit. Critical nodes are indicative only."
        )
    elif near_singular:
        interp = (
            "Jacobian near-singular — voltage collapse / loadability limit; a solution "
            "at this loading likely does not exist."
        )
    else:
        interp = (
            "Jacobian well-conditioned — the fixed-point iteration failed to contract "
            "though a solution likely exists (try Newton, a better start, or more "
            "iterations)."
        )
    return {
        "min_singular_value": sigma_min,
        "max_singular_value": sigma_max,
        "condition_number": cond,
        "near_singular": bool(near_singular),
        "evaluated_at": "diverged_iterate" if diverged else "final_iterate",
        "critical_nodes": critical,
        "interpretation": interp,
        **({"batch": b_star} if b > 1 else {}),
    }


# ---------------------------------------------------------------------------
# continuation (λ-ramp) loadability analysis
# ---------------------------------------------------------------------------
def loadability_limit(
    grid: Grid,
    *,
    slack: str = "ideal",
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    operating_point: Optional[dict] = None,
    param_overrides: Optional[dict] = None,
    symmetry: Optional[str] = None,
    lambda_max: float = 2.0,
    lambda_step: float = 0.1,
    bisect_tol: float = 1.0e-3,
    tol: Optional[float] = None,
    tol_update_pu: Optional[float] = None,
    s_base_va: Optional[float] = None,
    max_iter: int = 50,
    top_k: int = 5,
    ramp: str = "all",
) -> LoadabilityResult:
    """Step-and-bisect continuation: the largest ``λ`` whose power flow still solves.

    Scales the injections by ``λ`` (the residual is
    ``R(V,λ) = Y_eff·V + λ·I_device(V) − I_slack``; ``λ=1`` is the nameplate loading),
    steps ``λ`` up from a feasible base (``λ=0``, the trivial no-load solve),
    Newton-corrects at each step, and bisects onto the first ``λ`` where the corrector
    no longer converges.

    What ``breaking_lambda`` is — and is not. This is a step-and-bisect on Newton
    FEASIBILITY, not an arc-length predictor-corrector continuation in the augmented
    ``(V, λ)`` space: there is no tangent predictor and no corrector that can turn the
    nose. A plain Newton corrector stops converging slightly BEFORE the singularity
    (the Jacobian becomes ill-conditioned first), so ``breaking_lambda`` is the largest
    ``λ`` at which the corrector still converges and therefore a LOWER BOUND on the true
    P-V nose, with a gap that depends on ``tol``, ``max_iter`` and ``bisect_tol``
    (measured at ~4 % on a two-bus feeder whose nose is known in closed form). The
    Jacobian figures reported at that ``λ`` describe the last converged point, which is
    near the nose, not the singular point itself.

    ``ramp`` chooses WHAT ``λ`` multiplies:

    - ``"all"`` (default) — every injecting device: loads AND generators / storage scale
      together (a joint ramp of the whole operating point, the quantity a scenario sweep
      of a distribution feeder usually wants).
    - ``"load"`` — loads only, with generation held at its nameplate value: the textbook
      continuation-power-flow load ramp. On a feeder with substantial generation the two
      give different limits, because ramping generation with the load offsets the drop
      the ramp is meant to create.

    At the breaking ``λ`` the SVD of the power-flow Jacobian localizes the approaching
    collapse:

    - ``critical_nodes`` (RIGHT singular vector of the smallest σ): the voltage-collapse
      mode — the buses whose voltage gives way (where it breaks).
    - ``limiting_loads`` (LEFT singular vector · each device's current): the injections
      whose apparent power most reduces the margin (which input, at which node, drives
      the non-convergence). ``responsibility`` is normalized to ``[0, 1]``.

    ``breaking_lambda < 1`` means the nameplate loading itself does not solve;
    ``margin = λ* − 1`` is the headroom above nameplate. ``tol`` / ``tol_update_pu`` /
    ``s_base_va`` are the per-unit convergence settings of the Newton corrector (see
    :func:`solve_power_flow`).

    Single grid only (no scenario batch). Detached (a diagnostic, not on the autograd tape).
    """
    if slack not in ("ideal", "norton"):
        raise InputError(f"Unsupported slack {slack!r} (use 'ideal' or 'norton').")
    if ramp not in ("all", "load"):
        raise InputError(
            f"Unsupported ramp {ramp!r}: 'all' scales every injecting device (loads and "
            "generators together), 'load' scales loads only and holds generation at its "
            "nameplate value (the textbook continuation-power-flow ramp)."
        )
    tol, tol_update_pu, s_base_va = _resolve_tolerances(tol, tol_update_pu, s_base_va)
    check_branch_impedances(grid)
    check_connectivity(grid)
    cdt, rdt = _cdtype(dtype), _rdtype(dtype)
    index = node_phase_index(grid)
    n = index.size
    f0 = float(grid.base_frequency_hz)
    asymmetric = resolve_asymmetric(grid, operating_point, mode=symmetry)
    sym_resolved = "asymmetric" if asymmetric else "symmetric"
    leaves = _grid_param_leaves(grid, param_overrides, None, operating_point)
    if device is None:
        device = leaves[0].device if leaves else torch.device("cpu")
    v_base = _node_voltage_bases(grid, index, rdt, device)

    def v_fixed_fn():
        if slack != "ideal":
            return None
        return _slack_rows_and_vref(grid, index, rdt, cdt, device)[1]

    fixed_rows = (
        _slack_rows_and_vref(grid, index, rdt, cdt, device)[0]
        if slack == "ideal"
        else None
    )
    v_fixed = v_fixed_fn()

    def build_system():
        return _y_eff_and_islack(grid, f0, index, dtype, device, slack, param_overrides)

    # Detached injection plans serve every λ step (loadability is a detached
    # diagnostic; λ scales the plan's currents in the residual, not the plan).
    # ``ramp="load"`` needs the two device classes separately, so their currents can be
    # scaled independently; the split is made by taking the other class out of service
    # in a grid copy, which leaves every other resolution rule (per-phase split, ZIP
    # law, inverter control) untouched.
    def plan_for(g):
        with torch.no_grad():
            return build_injection_plan(
                g,
                index,
                [f0],
                dtype=dtype,
                device=device,
                operating_point=operating_point,
                param_overrides=param_overrides,
                symmetry=sym_resolved,
            )

    if ramp == "load":
        plan = plan_for(_only_appliances(grid, Load))
        plan_fixed = plan_for(_without_appliances(grid, Load))
    else:
        plan = plan_for(grid)
        plan_fixed = None

    def make_real_res(lam: float):
        def rc(v: Tensor, y: Tensor, islack: Tensor) -> Tensor:
            i_dev = lam * injections_from_plan(plan, v).squeeze(-2)
            if plan_fixed is not None:
                i_dev = i_dev + injections_from_plan(plan_fixed, v).squeeze(-2)
            return _apply_y(y, v) + i_dev - islack

        return _make_real_residual(build_system, rc, fixed_rows, v_fixed_fn, n, cdt)

    import logging

    pgml_log = logging.getLogger("pgml")
    prev = pgml_log.level
    pgml_log.setLevel(max(prev, logging.WARNING))  # quiet the per-step modeling logs
    try:
        with torch.no_grad():
            y0, islack0 = build_system()
            # y0 carries no frequency axis, so the const-Z start keeps islack0's shape.
            v_good = solve_harmonic(y0, islack0, fixed_rows=fixed_rows, v_fixed=v_fixed)
        lam_good, trace, total_iters = 0.0, [0.0], 0
        nose_found = False
        lam = lambda_step
        newton_args = (
            n,
            rdt,
            cdt,
            device,
            v_base,
            tol,
            tol_update_pu,
            s_base_va,
            max_iter,
            "dense",
            "full",
            fixed_rows,
        )
        while lam <= lambda_max + 1e-12:
            vk, it, _, conv = _newton_forward(make_real_res(lam), v_good, *newton_args)[
                :4
            ]
            total_iters += it
            if conv:
                lam_good, v_good = lam, vk
                trace.append(round(lam, 6))
                lam += lambda_step
                continue
            nose_found = True
            lo, hi = lam_good, lam  # bisect the feasibility boundary
            while hi - lo > bisect_tol:
                mid = 0.5 * (lo + hi)
                vm, itm, _, cm = _newton_forward(
                    make_real_res(mid), v_good, *newton_args
                )[:4]
                total_iters += itm
                if cm:
                    lo, lam_good, v_good = mid, mid, vm
                else:
                    hi = mid
            break
        crit = _nose_criticality(
            make_real_res(lam_good),
            build_system,
            v_good,
            grid,
            index,
            n,
            rdt,
            device,
            operating_point,
            param_overrides,
            sym_resolved,
            f0,
            dtype,
            top_k,
        )
    finally:
        pgml_log.setLevel(prev)

    return LoadabilityResult(
        breaking_lambda=lam_good,
        feasible=lam_good >= 1.0,
        margin=lam_good - 1.0,
        nose_voltage_min_pu=crit["nose_vmin_pu"],
        capped=not nose_found,
        critical_nodes=crit["critical_nodes"],
        limiting_loads=crit["limiting_loads"],
        min_singular_value=crit["sigma_min"],
        condition_number=crit["cond"],
        converged_lambdas=trace,
        corrector_iterations=total_iters,
        ramp=ramp,
    )


def _nose_criticality(
    real_res,
    build_system,
    v_good,
    grid,
    index,
    n,
    rdt,
    device,
    operating_point,
    param_overrides,
    sym_resolved,
    f0,
    dtype,
    top_k,
) -> dict:
    """At the last converged λ: SVD of ``J = dR/dV`` -> collapse mode + limiting loads.

    Evaluated at the largest λ whose corrector converged, which is just short of the
    true nose, so the singular values describe an ill-conditioned — not a singular —
    Jacobian.
    """
    from pgml.schemas.grid_schema import Generator as _Gen, Load as _Load

    state_residual = real_res.state_residual
    vg = v_good.reshape(-1)[:n]
    x = torch.cat([vg.real, vg.imag]).to(rdt)
    with torch.no_grad():
        y0, islack0 = build_system()
        yb = y0.detach().reshape(-1, n, n)[0]
        isb = islack0.detach().reshape(-1, n)[0]
        j = torch.autograd.functional.jacobian(
            lambda xx: state_residual(xx, yb.real, yb.imag, isb.real, isb.imag),
            x,
            vectorize=True,
        )  # [2N, 2N]
        u, s, vh = torch.linalg.svd(j)
        sigma_min, sigma_max = float(s[-1]), float(s[0])
        cond = sigma_max / max(sigma_min, 1e-300)
        right = vh[-1]  # collapse mode in V-space
        left = u[:, -1]  # left null vector
        part = torch.sqrt(right[:n] ** 2 + right[n:] ** 2)
        part = part / part.max().clamp_min(1e-30)
        bases = _node_voltage_bases(grid, index, rdt, device)
        vpu = vg.abs() / bases.clamp_min(1e-12)
        nonneutral = torch.as_tensor(index.phase_codes, device=device) != 3
        nose_vmin = (
            float(vpu[nonneutral].min()) if bool(nonneutral.any()) else float(vpu.min())
        )
        idev = device_current_injections(
            grid,
            vg,
            index,
            [f0],
            dtype=dtype,
            device=device,
            operating_point=operating_point,
            param_overrides=param_overrides,
            symmetry=sym_resolved,
        ).squeeze(-2)
        idev_real = torch.cat([idev.real, idev.imag])

    node_ids, phase_codes = index.node_ids.tolist(), index.phase_codes.tolist()
    vals, idxs = torch.topk(part, min(top_k, n))
    critical_nodes = [
        {
            "node_id": int(node_ids[int(r)]),
            "phase": _phase_name(int(phase_codes[int(r)])),
            "participation": float(v),
        }
        for v, r in zip(vals.tolist(), idxs.tolist())
    ]
    loads = []
    for a in grid.appliances:
        if not (isinstance(a, (_Load, _Gen)) and getattr(a, "in_service", True)):
            continue
        resp = 0.0
        for ph in a.phases:
            try:
                r = index.row(int(a.node), ph)
            except (KeyError, ValueError):
                continue
            resp += float(left[r]) * float(idev_real[r])
            resp += float(left[r + n]) * float(idev_real[r + n])
        p = float(getattr(a, "p_nom_w", 0.0) or 0.0)
        q = float(getattr(a, "q_nom_var", 0.0) or 0.0)
        loads.append(
            {
                "appliance_id": int(a.id),
                "node_id": int(a.node),
                "s_nominal_va": math.hypot(p, q),
                "responsibility": abs(resp),
            }
        )
    max_resp = max((d["responsibility"] for d in loads), default=0.0)
    if max_resp > 0.0:
        for d in loads:
            d["responsibility"] /= max_resp  # normalize to [0, 1]
    loads.sort(key=lambda d: d["responsibility"], reverse=True)
    return {
        "sigma_min": sigma_min,
        "cond": cond,
        "critical_nodes": critical_nodes,
        "limiting_loads": loads[:top_k],
        "nose_vmin_pu": nose_vmin,
    }


__all__ = [
    "check_branch_impedances",
    "check_connectivity",
    "prepare_power_flow",
    "PowerFlowSystem",
    "solve_power_flow",
    "PowerFlowResult",
    "ConvergenceDiagnostics",
    "loadability_limit",
    "LoadabilityResult",
]
