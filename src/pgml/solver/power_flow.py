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
from typing import Optional

import torch
from torch import Tensor

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
from pgml.assembly.ybus import _stamp_sources
from pgml.errors import ConnectivityError, InputError, ModelingError
from pgml.schemas.grid_schema import Grid, Source
from pgml.topology import connectivity_report, energized_subgrid

from .harmonic import lu_factor_system, solve_factored, solve_harmonic

_log = logging.getLogger("pgml")


def _rel_convergence_floor(rdt: torch.dtype, backend: str = "dense") -> float:
    """Smallest relative update ``||ΔV|| / ||V||`` the dtype (and backend) can resolve.

    The fixed-point / Newton update stops shrinking once it reaches the rounding
    noise of the working precision. ``float64`` has ample headroom (eps ~2e-16), so
    its floor is ``0.0`` and the absolute ``tol`` governs unchanged. ``float32``
    (eps ~1.2e-7) cannot resolve an update below ~1e-6 of the voltage scale, so a
    tighter absolute ``tol`` is physically unreachable; the floor caps the
    achievable tolerance and is reported via a one-time warning.

    The floor is also a property of the linear-algebra ``backend``: SuperLU's
    single-precision back-substitution (different pivoting/ordering than the dense
    torch LU) leaves per-iterate rounding noise measured at ~2e-6 relative, so
    marginal scenarios oscillate just above the dense-calibrated floor without ever
    crossing it (measured: batched IEEE-33 at complex64 plateaus flat at 2.0e-6 for
    ~0.2 % of scenarios, floor-accurate but running to ``max_iter``). The sparse
    float32 floor is therefore 4e-6 (2x headroom over the measured plateau).
    """
    if rdt == torch.float64:
        return 0.0
    return 4.0e-6 if backend == "sparse" else 1.0e-6


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
        Number of fixed-point iterations performed (python int).
    residual:
        Real scalar tensor: the final ``||V_{k+1} - V_k||`` (max over batch).
    converged:
        ``True`` if EVERY batch element's residual fell below ``tol`` (or the
        dtype floor) within ``max_iter``.
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
    """

    v: Tensor
    index: NodePhaseIndex
    iterations: int
    residual: Tensor
    converged: bool
    diagnostics: Optional[ConvergenceDiagnostics] = None
    converged_mask: Optional[Tensor] = None
    failed_states: tuple[int, ...] = ()


@dataclass
class ConvergenceDiagnostics:
    """Structured power-flow convergence telemetry (autograd-free, computed at ``V*``).

    Cheap state diagnostics are always populated; ``criticality`` is filled only when
    the solve did not converge (it costs a dense Jacobian + SVD). All voltages are
    per-unit on each node's line-to-neutral base; ``mismatch_a`` is the nodal current
    mismatch ``|F_c|`` [A] of the power-balance residual at the (free) row.
    """

    converged: bool
    iterations: int
    update_norm: float  # final ||ΔV|| (the fixed-point convergence measure)
    power_mismatch_max: float  # max |F_c| over free (non-slack) rows [A]
    voltage_band_pu: tuple[float, float]
    residual_history: list[float] = field(default_factory=list)  # ||ΔV|| per iteration
    worst_nodes: list[dict] = field(default_factory=list)  # top-k by current mismatch
    out_of_band_nodes: list[dict] = field(default_factory=list)  # |V| outside the band
    likely_cause: str = ""
    criticality: Optional[dict] = None  # IFT-Jacobian analysis (non-convergence only)

    def as_dict(self) -> dict:
        """Plain-dict view (e.g. for :attr:`ConvergenceError.diagnostics`)."""
        return asdict(self)


@dataclass
class LoadabilityResult:
    """Continuation (λ-ramp) loadability analysis — where/what limits solvability.

    Ramps the load by ``λ`` (``λ=1`` = the grid's nameplate load) from a feasible base,
    Newton-correcting at each step, until the power-flow Jacobian goes singular (the P-V
    nose). At the nose the singular Jacobian's vectors localize the collapse: the RIGHT
    singular vector is the voltage-collapse mode (the weakest buses), and the LEFT
    singular vector gives the margin's sensitivity to each load (which apparent-power
    injection most reduces the margin).
    """

    breaking_lambda: float  # λ* at the nose (load multiplier of the nameplate load)
    feasible: bool  # λ* >= 1 -> the nameplate load is solvable
    margin: (
        float  # λ* − 1 (headroom above nameplate; negative = infeasible at nameplate)
    )
    nose_voltage_min_pu: float  # lowest |V|/V_LN at the nose
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
def _apply_y(y_eff: Tensor, v: Tensor) -> Tensor:
    """``Y @ V`` over the scenario batch, reading a SHARED ``Y`` exactly once.

    When every leading dim of ``y_eff`` is singleton (one network shared by the
    whole batch — the usual case), a broadcast ``matmul`` against ``[..., N, 1]``
    columns degenerates into ``B`` separate matrix-vector products that re-read the
    ``N×N`` matrix per scenario (memory-bandwidth-bound: dominant at large ``N``).
    Folding the batch into the rows of ONE ``[B, N] @ [N, N]`` GEMM reads the
    matrix once. A genuinely batched ``y_eff`` (per-scenario topology) keeps the
    batched matmul — each scenario owns its matrix there. Differentiable in both.
    """
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
    """Effective admittance ``Y_eff`` ``[1,N,N]`` and slack current ``[1,N]`` or ``[N]``.

    ``slack="norton"``: ``Y_eff = Y_net + Y_srcNorton``, ``I_slack`` = source
    Norton current. ``slack="ideal"``: ``Y_eff = Y_net``, ``I_slack`` = 0 (slack
    rows pinned by the Schur solve in :func:`solve_harmonic`).

    Batched ``branch_states`` promote ``Y_eff`` to per-scenario matrices; the
    singleton frequency axis is folded away then (``[*batch, N, N]``) so every
    leading dim is a scenario dim — the shape the fixed point, Newton, and the
    IFT backward treat uniformly.
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
        )  # [1, N]
    else:
        i_slack = torch.zeros(y.shape[-1], dtype=y.dtype, device=y.device)
    return y, i_slack


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
    that consumes it (validated where cheap: slack / dtype / device / size).
    """

    index: NodePhaseIndex
    f0: float
    slack: str
    y_eff: Tensor  # detached [*, N, N]
    i_slack: Tensor  # detached [1, N] (norton) or [N] (ideal)
    fixed_rows: Optional[Tensor]
    v_fixed: Optional[Tensor]  # detached slack reference
    factorization: object  # FactoredSystem of y_eff
    static_leaves: tuple[Tensor, ...]  # grid + overrides + states leaves


def prepare_power_flow(
    grid: Grid,
    *,
    slack: str = "ideal",
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    param_overrides: Optional[dict] = None,
    branch_states: Optional[dict] = None,
    linear_solver: str = "auto",
) -> PowerFlowSystem:
    """Assemble + factor the operating-point-independent power-flow system once.

    Runs the connectivity check (raising
    :class:`~pgml.errors.ConnectivityError` like :func:`solve_power_flow` with
    ``on_disconnected="raise"``), assembles ``Y_eff`` and the slack quantities,
    and factors ``Y_eff`` with the selected backend
    (:func:`pgml.solver.harmonic.lu_factor_system`; ``linear_solver`` as in
    :func:`solve_power_flow`). Pass the result as ``solve_power_flow(...,
    system=...)`` to skip that work on every subsequent call.
    """
    if slack not in ("ideal", "norton"):
        raise InputError(f"Unsupported slack {slack!r} (use 'ideal' or 'norton').")
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
        y_eff, i_slack = _y_eff_and_islack(
            grid, f0, index, dtype, device, slack, param_overrides, branch_states
        )
        factor_backend = (
            linear_solver if linear_solver in ("dense", "sparse") else "auto"
        )
        fac = lu_factor_system(y_eff, fixed_rows=fixed_rows, backend=factor_backend)
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
    )


def solve_power_flow(
    grid: Grid,
    *,
    slack: str = "ideal",
    method: str = "current_injection",
    tol: float = 1e-8,
    max_iter: int = 100,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    operating_point: Optional[dict] = None,
    param_overrides: Optional[dict] = None,
    symmetry: Optional[str] = None,
    criticality: str = "auto",
    linear_solver: str = "auto",
    on_disconnected: str = "raise",
    branch_states: Optional[dict] = None,
    system: Optional[PowerFlowSystem] = None,
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
        Fixed-point convergence tolerance on ``||ΔV||`` (max over batch).
    max_iter:
        Maximum fixed-point iterations.
    dtype:
        Complex dtype (``complex128`` for gradcheck; ``complex64`` ok).
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
        (``"matrix_free"`` is treated as ``"auto"`` here).

        For ``method="newton"``: ``"dense"`` (the explicit ``[2N, 2N]`` Jacobian +
        direct solve; ``"auto"`` resolves to this) or ``"matrix_free"``
        (Jacobian-free Newton-Krylov — GMRES on finite-difference Jacobian-vector
        products, ``O(N)`` memory for large grids). ``"sparse"`` raises — Newton's
        Jacobian is built dense.
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
        iteration count, the final update-norm residual, the convergence flag, and a
        :class:`ConvergenceDiagnostics`.
    """
    if method not in ("current_injection", "newton"):
        raise ModelingError(
            f"Unsupported method {method!r} (use 'current_injection' or 'newton')."
        )
    if slack not in ("ideal", "norton"):
        raise InputError(f"Unsupported slack {slack!r} (use 'ideal' or 'norton').")
    if criticality not in ("auto", "always", "never"):
        raise InputError(
            f"Unsupported criticality {criticality!r} (use 'auto'/'always'/'never')."
        )
    if linear_solver not in ("auto", "dense", "sparse", "matrix_free"):
        raise InputError(
            f"Unsupported linear_solver {linear_solver!r} "
            "(use 'auto'/'dense'/'sparse'/'matrix_free')."
        )
    if method == "newton" and linear_solver == "sparse":
        raise InputError(
            "method='newton' supports linear_solver 'auto'/'dense'/'matrix_free' "
            "(its Jacobian is built dense); 'sparse' selects the fixed-point "
            "factorization backend of method='current_injection'."
        )
    # Newton's inner solve: 'auto' resolves to the proven dense Jacobian path.
    newton_solver = "dense" if linear_solver == "auto" else linear_solver
    # Fixed-point factorization backend: 'matrix_free' has no meaning there.
    factor_backend = linear_solver if linear_solver in ("dense", "sparse") else "auto"
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
                    max_iter=max_iter,
                    dtype=dtype,
                    device=device,
                    operating_point=operating_point,
                    param_overrides=param_overrides,
                    symmetry=symmetry,
                    criticality=criticality,
                    linear_solver=linear_solver,
                    on_disconnected="ignore",
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

    residual_complex = make_residual_complex(operating_point)
    fast_residual_complex = make_fast_residual_complex(operating_point)
    real_res = _make_real_residual(
        build_system,
        residual_complex,
        fixed_rows,
        v_fixed_fn,
        n,
        cdt,
        state_residual_complex=fast_residual_complex,
    )

    # One-time warning when the absolute `tol` is below what the working precision can
    # resolve at this voltage scale (complex64); the dtype floor governs convergence.
    floor = _rel_convergence_floor(rdt)
    if floor > 0.0:
        _, _sl_vref = _slack_rows_and_vref(grid, index, rdt, cdt, device)
        v_ref = float(_sl_vref.detach().abs().max()) if _sl_vref is not None else 1.0
        if v_ref > 0.0 and tol < floor * v_ref:
            _log.warning(
                "solve_power_flow: tol=%.1e is below the %s precision floor (~%.1e V, "
                "%.0e relative at the ~%.0f V scale); the dtype floor governs "
                "convergence. Use complex128 for a tighter tolerance.",
                tol,
                dtype,
                floor * v_ref,
                floor,
                v_ref,
            )

    # ----- forward: solve for the detached V* (gradients attached by the IFT) -----
    if method == "newton":
        # OpenDSS-style warm start: the LINEAR const-Z solution, then Newton on the
        # full const-P / ZIP residual. Newton's quadratic convergence and far larger
        # convergence region reach solutions the current-injection fixed point cannot
        # (e.g. near the loadability nose — see
        # ``examples/current_injection_convergence.py``).
        bsize = _operating_point_batch_size(operating_point)
        if bsize > 1:
            # A batched operating point solves SEQUENTIALLY per scenario: the
            # per-scenario ``jacobian(vectorize=True)`` (one vectorized call per
            # scenario) is measurably faster than a batch-native block-diagonal
            # build — the O(B²·(2N)²) full-map Jacobian does not fit, and the O(B)
            # column-by-column alternative costs 2N JVP evaluations per Newton
            # step (4x slower than this loop at B=64/N=180). Newton is the
            # hard-grid / near-nose solver — for bulk batches prefer the
            # vectorized current-injection method.
            (
                v_star,
                iterations,
                residual_norm,
                converged,
                residual_history,
                y_eff0,
                i_slack0,
                converged_mask,
                residual_vec,
            ) = _newton_forward_sequential(
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
                v_fixed_fn,
                build_system,
                make_fast_residual_complex,
                bsize,
                n,
                rdt,
                cdt,
                tol,
                max_iter,
                newton_solver,
                branch_states,
            )
        else:
            v_init = _linear_const_z_init(
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
                branch_states,
            )
            (
                v_star,
                iterations,
                residual_norm,
                converged,
                residual_history,
                y_eff0,
                i_slack0,
                converged_mask,
                residual_vec,
            ) = _newton_forward(
                real_res, v_init, n, rdt, cdt, device, tol, max_iter, newton_solver
            )
    else:
        (
            v_star,
            iterations,
            residual_norm,
            converged,
            residual_history,
            y_eff0,
            i_slack0,
            converged_mask,
            residual_vec,
        ) = _current_injection_forward(
            grid,
            index,
            build_system,
            fixed_rows,
            v_fixed,
            fast_residual_complex.plan,
            n,
            rdt,
            cdt,
            device,
            tol,
            max_iter,
            factor_backend,
            system,
        )

    # Convergence diagnostics at V* (autograd-free; the criticality analysis builds the
    # IFT real Jacobian only when the solve did not converge).
    diagnostics = _build_diagnostics(
        grid,
        index,
        v_star,
        y_eff0,
        i_slack0,
        fast_residual_complex,
        real_res,
        fixed_rows,
        residual_history,
        bool(converged),
        iterations,
        float(residual_norm),
        rdt,
        device,
        criticality,
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
            "(worst residual %.3e); returning best-effort voltages. Failed indices: "
            "[%s%s]%s",
            len(failed_states),
            total,
            iterations,
            float(residual_norm),
            shown,
            more,
            f" — {cause}" if cause else "",
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
    )


# ---------------------------------------------------------------------------
# forward solvers (detached V*; gradients are attached by the IFT below)
# ---------------------------------------------------------------------------
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
    tol,
    max_iter,
    factor_backend="auto",
    system=None,
):
    """Current-injection fixed point ``V_{k+1} = Y_eff^{-1}(I_slack − I_device(V_k))``.

    ``plan`` is the precomputed :class:`~pgml.assembly.InjectionPlan`: the
    operating point is resolved once and every iteration evaluates
    :func:`injections_from_plan` (pure tensor ops).

    Returns ``(v_star, iterations, residual_norm, converged, residual_history, y_eff0,
    i_slack0, converged_mask, residual_vec)``; ``residual_norm`` is the final ``||ΔV||``
    (max over batch), while ``converged_mask`` / ``residual_vec`` are PER scenario.
    Convergence is per element on ``||ΔV|| < max(tol, floor·||V||)`` where ``floor`` is
    the resolvable relative precision of the dtype AND linear-algebra backend
    (0 for float64; 1e-6 for float32 dense, 4e-6 for float32 sparse SuperLU —
    see :func:`_rel_convergence_floor`).
    """
    with torch.no_grad():
        if system is not None:
            y_eff0, i_slack0 = system.y_eff, system.i_slack
        else:
            y_eff0, i_slack0 = build_system()
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

        residual_norm = torch.zeros((), dtype=rdt, device=device)
        residual_vec = torch.zeros((), dtype=rdt, device=device)
        converged_mask = torch.zeros((), dtype=torch.bool, device=device)
        iterations = 0
        converged = False
        residual_history: list[float] = []
        # Y_eff is the network admittance — constant across iterations (the const-P/ZIP
        # loads enter the RHS as I_device(V), never Y). Factor it ONCE and back-substitute
        # each iteration (the whole fixed point runs under no_grad; the IFT supplies grads).
        fac = (
            system.factorization
            if system is not None
            else lu_factor_system(y_eff0, fixed_rows=fixed_rows, backend=factor_backend)
        )
        # The achievable update floor depends on the RESOLVED backend (SuperLU's
        # single-precision back-substitution is noisier than the dense torch LU).
        floor = _rel_convergence_floor(rdt, fac.backend)
        for _ in range(max_iter):
            i_dev = injections_from_plan(plan, v).squeeze(-2)  # [*b, N]
            rhs = i_slack0 - i_dev
            v_new = solve_factored(fac, rhs, v_fixed=v_fixed)
            # solve_factored carries Y's leading H=1; drop the singleton axis.
            if v_new.ndim >= 2 and v_new.shape[-2] == 1 and v_new.shape[-1] == n:
                v_new = v_new.squeeze(-2)
            # Per-element update + dtype-aware threshold max(tol, floor*||V||). For
            # float64 floor=0 so this is exactly ||ΔV|| < tol (unchanged); for float32
            # the floor caps tol at the achievable relative precision.
            delta = torch.linalg.vector_norm(v_new - v, dim=-1)  # [*b]
            v_scale = torch.linalg.vector_norm(v_new, dim=-1)  # [*b]
            thresh = torch.clamp(floor * v_scale, min=tol)  # [*b]
            converged_mask = delta < thresh
            residual_vec = delta
            residual_norm = delta.max()
            residual_history.append(float(residual_norm))
            v = v_new
            iterations += 1
            if bool(converged_mask.all()):
                converged = True
                break

    return (
        v,
        iterations,
        residual_norm,
        converged,
        residual_history,
        y_eff0,
        i_slack0,
        converged_mask,
        residual_vec,
    )


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
        y_lin = yb.Y.squeeze(-3)  # [*batch, N, N] (batched branch states)
    elif yb.Y.ndim == 3:
        y_lin = yb.Y  # [1, N, N]
    else:
        y_lin = yb.Y.unsqueeze(0)
    if slack == "norton":
        i_init = build_injections(
            grid,
            [f0],
            index,
            dtype=dtype,
            device=device,
            param_overrides=param_overrides,
        )  # [1, N] source Norton current
    else:
        i_init = torch.zeros(y_lin.shape[-1], dtype=y_lin.dtype, device=device)  # [N]
    with torch.no_grad():
        v0 = solve_harmonic(y_lin, i_init, fixed_rows=fixed_rows, v_fixed=v_fixed)
    if v0.ndim >= 2 and v0.shape[-2] == 1:
        v0 = v0.squeeze(-2)  # drop the singleton H axis -> [*b, N]
    return v0


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
    real_res, v_init, n, rdt, cdt, device, tol, max_iter, linear_solver="dense"
):
    """Newton on the real residual ``R(x) = 0`` from the warm start ``v_init``.

    Each step solves ``J·Δx = −R`` with ``J = dR/dx`` and a backtracking line search on
    ``‖R‖∞`` for global robustness, converging on ``‖Δx‖ < tol`` (the same measure the
    fixed point uses). ``linear_solver``:

    - ``"dense"`` (default): the explicit real ``[2N, 2N]`` Jacobian (autograd) + a direct
      solve. Per-element loop avoids the ``[B, 2N, B, 2N]`` memory of a batched Jacobian.
    - ``"matrix_free"``: Jacobian-free Newton-Krylov — never forms ``J``; solves with
      GMRES using a finite-difference Jacobian-vector product
      ``J·v ≈ (R(x+εv) − R(x))/ε``. ``O(N)`` memory, for large grids where the dense
      Jacobian is prohibitive (accuracy is the ``√eps`` FD floor, ample for a PF solve).

    Returns the same 9-tuple as :func:`_current_injection_forward`.
    """
    state_residual = real_res.state_residual
    build_system = real_res.build_system
    twon = 2 * n
    floor = _rel_convergence_floor(rdt)
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

        residual_history: list[float] = []
        converged = False
        iterations = 0
        dx_norm = torch.zeros((), dtype=rdt, device=device)
        converged_mask = torch.zeros(b, dtype=torch.bool, device=device)
        residual_vec = torch.zeros(b, dtype=rdt, device=device)
        for _ in range(max_iter):
            r = res_all(x)  # [b, 2N]
            if linear_solver == "matrix_free":
                dx = _newton_dir_matrix_free(res_one, x, r, b, fd_eps)
            else:
                dx = _newton_dir_dense(res_all, x, r)
            # PER-ELEMENT backtracking on each scenario's residual infinity-norm
            # (global robustness): a hard scenario halves only its own step.
            r0 = r.abs().amax(dim=-1)  # [b]
            step = torch.ones(b, 1, dtype=rdt, device=x.device)
            for _bt in range(_NEWTON_MAX_BACKTRACK):
                ok = res_all(x + step * dx).abs().amax(dim=-1) <= r0  # [b]
                if bool(ok.all()):
                    break
                step = torch.where(ok.unsqueeze(-1), step, 0.5 * step)
            x = x + step * dx
            # Per-element step norm + dtype-aware threshold max(tol, floor*||x||).
            dxn = (step * dx).norm(dim=-1)  # [b]
            x_scale = x.norm(dim=-1)  # [b]
            converged_mask = dxn < torch.clamp(floor * x_scale, min=tol)
            residual_vec = dxn
            dx_norm = dxn.max()
            residual_history.append(float(dx_norm))
            iterations += 1
            if bool(converged_mask.all()):
                converged = True
                break
        v_star = torch.complex(x[..., :n], x[..., n:]).reshape(*lead, n)
        cmask = converged_mask.reshape(lead) if lead else converged_mask.reshape(())
        rvec = residual_vec.reshape(lead) if lead else residual_vec.reshape(())
    return (
        v_star,
        iterations,
        dx_norm,
        converged,
        residual_history,
        y_eff0,
        i_slack0,
        cmask,
        rvec,
    )


def _newton_forward_sequential(
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
    v_fixed_fn,
    build_system,
    make_fast_residual_complex,
    bsize,
    n,
    rdt,
    cdt,
    tol,
    max_iter,
    linear_solver,
    branch_states=None,
):
    """Batched Newton by solving each scenario with the single-grid Newton forward.

    Newton's const-Z warm start and per-element Jacobian are single-grid (the residual
    closes over the operating point), so a batched operating point is handled by slicing
    it per scenario, running the proven single-grid forward, and stacking the detached
    ``V*`` ``[B, N]``. The IFT backward (full op, batch-aligned, block-diagonal) attaches
    batched gradients to the stacked result, so this is forward-only sequencing — the
    differentiability is unchanged. The per-scenario residual comes from
    ``make_fast_residual_complex`` (one detached injection plan per slice — the
    detached forward needs only ``dR/dx``). Returns the same 9-tuple as
    :func:`_newton_forward`.
    """
    v_list, conv_list, res_list = [], [], []
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

        rr_i = _make_real_residual(build_system, rc_i, fixed_rows, _vfixed_i, n, cdt)
        v_init_i = _linear_const_z_init(
            grid,
            f0,
            index,
            dtype,
            device,
            slack,
            op_i,
            param_overrides,
            fixed_rows,
            _vfixed_i(),
            branch_states,
        )
        v_i, it_i, rn_i, cv_i, _, y_eff0, i_slack0, _, _ = _newton_forward(
            rr_i, v_init_i, n, rdt, cdt, device, tol, max_iter, linear_solver
        )
        v_list.append(v_i)  # [N]
        conv_list.append(bool(cv_i))
        res_list.append(rn_i.reshape(()))
        iterations = max(iterations, it_i)
    v_star = torch.stack(v_list, 0)  # [B, N]
    converged_mask = torch.tensor(conv_list, dtype=torch.bool, device=device)  # [B]
    residual_vec = torch.stack(res_list, 0)  # [B]
    residual_norm = residual_vec.max()
    converged = bool(converged_mask.all())
    # Per-scenario residual histories are not aggregated (their lengths differ); the
    # cheap state diagnostics + the per-element residual_vec carry the per-batch detail.
    return (
        v_star,
        iterations,
        residual_norm,
        converged,
        [],
        y_eff0,
        i_slack0,
        converged_mask,
        residual_vec,
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


def _newton_dir_dense(batched_state_res, x, r) -> Tensor:
    """Dense Newton direction ``Δx`` solving ``J Δx = −R`` per batch element."""
    j = _batched_state_jacobian(batched_state_res, x)  # [b, 2N, 2N]
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
    """
    state_rc = (
        state_residual_complex
        if state_residual_complex is not None
        else residual_complex
    )

    def _pin_and_split(fc: Tensor, v: Tensor, x: Tensor) -> Tensor:
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
        return _pin_and_split(fc, v, x)

    def state_residual(
        x: Tensor, y_re: Tensor, y_im: Tensor, islack_re: Tensor, islack_im: Tensor
    ) -> Tensor:
        """Real residual at FIXED (real-split) system tensors — for the state Jacobian.

        ``jacrev`` rejects complex inputs, so the (constant) system tensors are
        passed as real/imag pairs and recombined here. Only ``x`` is differentiated.
        """
        v_re = x[..., :n]
        v_im = x[..., n:]
        v = torch.complex(v_re, v_im).to(cdt)
        y_eff = torch.complex(y_re, y_im).to(cdt)
        i_slack = torch.complex(islack_re, islack_im).to(cdt)
        fc = state_rc(v, y_eff, i_slack)
        return _pin_and_split(fc, v, x)

    real_residual.state_residual = state_residual
    real_residual.build_system = build_system
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

        def batched_state_res(xb):
            return state_residual(
                xb, y_flat.real, y_flat.imag, islack_flat.real, islack_flat.imag
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
            grads = torch.autograd.grad(
                r_theta,
                leaves,
                grad_outputs=grad_out,
                retain_graph=True,
                allow_unused=True,
            )

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
    y_eff0,
    i_slack0,
    residual_complex,
    real_res,
    fixed_rows,
    residual_history,
    converged,
    iterations,
    update_norm,
    rdt,
    device,
    criticality: str = "auto",
) -> "ConvergenceDiagnostics":
    """Cheap state diagnostics at ``V*`` (+ Jacobian criticality on non-convergence)."""
    n = index.size
    vmin, vmax = _DIAG_VBAND_PU
    node_ids = index.node_ids.tolist()
    phase_codes = index.phase_codes.tolist()
    with torch.no_grad():
        fc = residual_complex(v_star, y_eff0, i_slack0)  # [*B, N] complex (all rows)
        lead = v_star.shape[:-1]
        b = int(torch.tensor(lead).prod().item()) if lead else 1
        vflat = v_star.reshape(b, n)
        fc_abs = fc.reshape(b, n).abs().clone()
        if fixed_rows is not None and fixed_rows.numel() > 0:
            fc_abs[:, fixed_rows] = 0.0  # slack rows absorb mismatch by construction
        bases = _node_voltage_bases(grid, index, rdt, device)
        vpu = vflat.abs() / bases.clamp_min(1e-12)[None, :]  # [B, N]
        power_mismatch_max = float(fc_abs.max()) if fc_abs.numel() else 0.0

        worst_nodes: list[dict] = []
        k = min(_DIAG_TOP_K, b * n)
        if k > 0 and power_mismatch_max > 0.0:
            vals, idxs = torch.topk(fc_abs.reshape(-1), k)
            for val, fi in zip(vals.tolist(), idxs.tolist()):
                bi, r = divmod(int(fi), n)
                worst_nodes.append(
                    {
                        "node_id": int(node_ids[r]),
                        "phase": _phase_name(int(phase_codes[r])),
                        "mismatch_a": float(val),
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
        # Relative final update: a large ||ΔV|| vs the voltage scale means the iterate
        # never settled (oscillating), so any band violation on it is an artifact.
        v_scale = float(vflat.abs().max()) if finite else float("inf")
        rel_update = (
            update_norm / v_scale if (finite and v_scale > 0.0) else float("inf")
        )

    diag = ConvergenceDiagnostics(
        converged=converged,
        iterations=iterations,
        update_norm=update_norm,
        power_mismatch_max=power_mismatch_max,
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
    if rel_update > 0.01:  # iterate never settled (oscillating ||ΔV|| vs voltage scale)
        return (
            f"fixed-point iteration did not settle (oscillating; final ||ΔV|| = "
            f"{diag.update_norm:.2e} V, ~{rel_update * 100:.0f}% of the voltage scale) "
            "— the current-injection map is not contracting here; try Newton or a "
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
            f"fixed-point iteration not contracting (||ΔV|| plateaued at {hist[-1]:.2e}); "
            "a solution may exist — try Newton / a better start / more iterations"
        )
    wn = diag.worst_nodes[0] if diag.worst_nodes else None
    tail = (
        f"; max mismatch {wn['mismatch_a']:.2e} A at node {wn['node_id']}.{wn['phase']}"
        if wn
        else ""
    )
    return (
        f"did not reach tol in {diag.iterations} iterations "
        f"(||ΔV||={diag.update_norm:.2e}){tail}"
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
    with torch.no_grad():
        svals = torch.linalg.svdvals(j)
        sigma_min, sigma_max = float(svals.min()), float(svals.max())
        cond = sigma_max / max(sigma_min, 1e-300)
        _, _, vh = torch.linalg.svd(j)
        mode = vh[-1]  # right singular vector of the smallest singular value
        part = torch.sqrt(mode[:n] ** 2 + mode[n:] ** 2)  # [N] per-node participation
        part = part / part.max().clamp_min(1e-30)
        vals, idxs = torch.topk(part, min(_DIAG_TOP_K, n))
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
    tol: float = 1.0e-8,
    max_iter: int = 50,
    top_k: int = 5,
) -> LoadabilityResult:
    """Continuation power flow: find the loadability nose and WHAT/WHERE limits it.

    Ramps every load/generator by a scalar ``λ`` (the residual is
    ``R(V,λ) = Y_eff·V + λ·I_device(V) − I_slack``; ``λ=1`` is the nameplate load) from a
    feasible base (``λ=0``, the trivial no-load solve), Newton-correcting at each step and
    bisecting onto the breaking ``λ*`` where the corrector fails — the P-V nose. At ``λ*``
    the power-flow Jacobian is (near) singular; its SVD localizes the collapse:

    - ``critical_nodes`` (RIGHT singular vector of the smallest σ): the voltage-collapse
      mode — the buses whose voltage gives way (where it breaks).
    - ``limiting_loads`` (LEFT singular vector · each load's current): the loads whose
      apparent power most reduces the margin (which input, at which node, causes the
      non-convergence). ``responsibility`` is normalized to ``[0, 1]``.

    ``breaking_lambda < 1`` means the nameplate load itself is infeasible (the fixed point
    / Newton cannot converge); ``margin = λ* − 1`` is the headroom above nameplate.

    Single grid only (no scenario batch). Detached (a diagnostic, not on the autograd tape).
    """
    if slack not in ("ideal", "norton"):
        raise InputError(f"Unsupported slack {slack!r} (use 'ideal' or 'norton').")
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

    # One detached injection plan serves every λ step (loadability is a detached
    # diagnostic; λ scales the plan's currents in the residual, not the plan).
    with torch.no_grad():
        plan = build_injection_plan(
            grid,
            index,
            [f0],
            dtype=dtype,
            device=device,
            operating_point=operating_point,
            param_overrides=param_overrides,
            symmetry=sym_resolved,
        )

    def make_real_res(lam: float):
        def rc(v: Tensor, y: Tensor, islack: Tensor) -> Tensor:
            i_dev = injections_from_plan(plan, v).squeeze(-2)
            return _apply_y(y, v) + lam * i_dev - islack

        return _make_real_residual(build_system, rc, fixed_rows, v_fixed_fn, n, cdt)

    import logging

    pgml_log = logging.getLogger("pgml")
    prev = pgml_log.level
    pgml_log.setLevel(max(prev, logging.WARNING))  # quiet the per-step modeling logs
    try:
        with torch.no_grad():
            y0, islack0 = build_system()
            v_good = solve_harmonic(y0, islack0, fixed_rows=fixed_rows, v_fixed=v_fixed)
            if v_good.ndim >= 2 and v_good.shape[-2] == 1:
                v_good = v_good.squeeze(-2)
        lam_good, trace, total_iters = 0.0, [0.0], 0
        lam = lambda_step
        while lam <= lambda_max + 1e-12:
            vk, it, _, conv, _, _, _, _, _ = _newton_forward(
                make_real_res(lam), v_good, n, rdt, cdt, device, tol, max_iter
            )
            total_iters += it
            if conv:
                lam_good, v_good = lam, vk
                trace.append(round(lam, 6))
                lam += lambda_step
                continue
            lo, hi = lam_good, lam  # bisect the feasibility boundary
            while hi - lo > bisect_tol:
                mid = 0.5 * (lo + hi)
                vm, itm, _, cm, _, _, _, _, _ = _newton_forward(
                    make_real_res(mid), v_good, n, rdt, cdt, device, tol, max_iter
                )
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
        critical_nodes=crit["critical_nodes"],
        limiting_loads=crit["limiting_loads"],
        min_singular_value=crit["sigma_min"],
        condition_number=crit["cond"],
        converged_lambdas=trace,
        corrector_iterations=total_iters,
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
    """At the nose: SVD of ``J = dR/dV`` -> collapse mode + margin-limiting loads."""
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
    "check_connectivity",
    "prepare_power_flow",
    "PowerFlowSystem",
    "solve_power_flow",
    "PowerFlowResult",
    "ConvergenceDiagnostics",
    "loadability_limit",
    "LoadabilityResult",
]
