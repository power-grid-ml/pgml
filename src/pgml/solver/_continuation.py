"""The fundamental power flow as a one-parameter family, for continuation studies.

:class:`ContinuationProblem` holds the fundamental power-flow residual of one grid with a
scalar loading parameter ``λ`` that multiplies a chosen set of appliances:

    F_c(V, λ) = Y_eff V + λ · I_scaled(V) + I_fixed(V) − I_slack

``I_scaled`` / ``I_fixed`` are the device currents of the scaled and the unscaled
appliances, each from a detached :class:`~pgml.assembly.InjectionPlan`, so ``λ`` scales
every component of an appliance's power at the solved voltage (the constant-power,
constant-current and constant-impedance parts of a ZIP law alike). The real residual
``R(x, λ)``, ``x = [Re V; Im V]``, is the one :func:`~pgml.solver.solve_power_flow`
iterates on: ideal-slack rows pinned, voltage-regulating generator rows substituted by
the active balance and the setpoint (:mod:`pgml.solver._pv_bus`). Plain solves at a fixed
``λ`` run through the solver's own Newton loop (line search and per-unit convergence
test included); a continuation corrector evaluates ``R`` and its dense Jacobian with
respect to ``(x, λ)`` directly.

Reactive limits of regulating generators are honoured by the same off-tape PV-to-PQ
switching as the solver. A scaled generator that is pinned at a reactive limit keeps its
reactive power at the limit while ``λ`` scales its active power.

A fictitious voltage-controlled reactive source (the QV-curve method) is a per-phase
regulating terminal with zero active power and unbounded reactive power, added to the
same terminal set; its reactive injection is read off the converged residual.

Everything here is detached (an analysis, not on the autograd tape), single grid, single
operating point, on the device and complex dtype the problem was built with.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Optional, Sequence

import torch
from torch import Tensor

from pgml import defaults
from pgml.assembly import (
    InjectionPlan,
    build_injection_plan,
    injections_from_plan,
    node_phase_index,
)
from pgml.assembly._fusion import resolve_fusion, zero_impedance_branches
from pgml.assembly._stamps import _cdtype, _rdtype
from pgml.assembly._symmetry import resolve_asymmetric
from pgml.errors import InputError
from pgml.schemas.grid_schema import Grid, InjectionAppliance, Phase

from ._pv_bus import PVGroup, PVTerminals, collect_pv_terminals
from .equilibration import equilibrated_lu_factor, resolve_equilibration
from .harmonic import solve_harmonic
from .power_flow import (
    _FloorRowScale,
    _PuConvergence,
    _apply_y,
    _check_fused_row_conflicts,
    _linear_const_z_init,
    _make_real_residual,
    _mismatch_floor_rel,
    _newton_forward,
    _node_voltage_bases,
    _operating_point_batch_size,
    _pv_nominal_init,
    _rel_convergence_floor,
    _resolve_tolerances,
    _slack_rows_and_vref,
    _y_eff_and_islack,
    check_branch_impedances,
    check_connectivity,
)

_log = logging.getLogger("pgml")

#: Terminal id of the fictitious voltage-controlled source of a QV study (no grid
#: appliance carries a negative id).
FICTITIOUS_SOURCE_ID = -1


class ContinuationProblem:
    """The power-flow residual of one grid as a function of the state and ``λ``.

    Built by :func:`build_continuation_problem`. ``index`` is the row layout the solve
    runs on (reduced when zero-impedance branches are fused), ``full_index`` the grid's
    own layout that results are reported on (:meth:`prolong`). The terminal set ``pv``
    (regulating generators plus an optional fictitious source) is mutable state:
    :meth:`switch` updates its active set, :meth:`set_source_setpoint` the fictitious
    source's setpoint.
    """

    def __init__(
        self,
        *,
        grid: Grid,
        index,
        full_index,
        fusion,
        scaled_ids: tuple[int, ...],
        operating_point: Optional[dict],
        param_overrides: Optional[dict],
        symmetry: str,
        slack: str,
        dtype: torch.dtype,
        device: torch.device,
        grid_pv: Optional[PVTerminals],
        equilibration: str,
        tol: float,
        tol_update_pu: float,
        s_base_va: float,
        max_iter: int,
    ) -> None:
        self.grid = grid
        self.index = index
        self.full_index = full_index
        self.fusion = fusion
        self.n = index.size
        self.scaled_ids = scaled_ids
        self.operating_point = operating_point
        self.param_overrides = param_overrides
        self.symmetry = symmetry
        self.slack = slack
        self.dtype = dtype
        self.cdt = _cdtype(dtype)
        self.rdt = _rdtype(dtype)
        self.device = device
        self.f0 = float(grid.base_frequency_hz)
        self.equilibration = equilibration
        self.tol = tol
        self.tol_update_pu = tol_update_pu
        self.s_base_va = s_base_va
        self.max_iter = max_iter
        with torch.no_grad():
            if slack == "ideal":
                rows, vf = _slack_rows_and_vref(
                    grid, index, self.rdt, self.cdt, device, operating_point
                )
                self.fixed_rows = rows
                self.v_fixed = None if vf is None else vf.detach()
            else:
                self.fixed_rows, self.v_fixed = None, None
            y, i_slack = _y_eff_and_islack(
                grid,
                self.f0,
                index,
                dtype,
                device,
                slack,
                param_overrides,
                None,
                fusion,
            )
            self.y_eff = y.detach()
            self.i_slack = i_slack.detach()
            self.v_base = _node_voltage_bases(grid, index, self.rdt, device)
            self.full_v_base = _node_voltage_bases(grid, full_index, self.rdt, device)
        self._grid_pv = grid_pv
        self._source: Optional[PVGroup] = None
        self._pv_policy = grid_pv if grid_pv is not None else _default_policy()
        self.ctest = _PuConvergence(
            v_base=self.v_base,
            s_base=s_base_va,
            tol_mismatch_pu=tol,
            tol_update_pu=tol_update_pu,
            floor_update=_rel_convergence_floor(self.rdt),
            floor_mismatch=_mismatch_floor_rel(self.rdt),
            fixed_rows=self.fixed_rows,
            scale=_FloorRowScale(),
            y_eff=self.y_eff,
            n=self.n,
            device=device,
            rdt=self.rdt,
        )
        self._rebuild_plans()

    # -- terminal set --------------------------------------------------------
    @property
    def pv(self) -> Optional[PVTerminals]:
        """Every regulating terminal of the residual (grid generators + source)."""
        groups = () if self._grid_pv is None else self._grid_pv.groups
        if self._source is not None:
            groups = (*groups, self._source)
        if not groups:
            return None
        return replace(self._pv_policy, groups=groups)

    def _rebuild_plans(self) -> None:
        """Injection plans of the scaled and the unscaled appliances at the active set.

        A regulating generator's reactive power is pinned into the operating point
        (zero while it regulates, the binding limit once pinned), exactly as the solver
        does. A SCALED generator pinned at a limit is split: its active power stays in
        the scaled plan, its pinned reactive power moves to the unscaled one, so ``λ``
        never multiplies a reactive limit.
        """
        op = dict(self.operating_point) if self.operating_point else {}
        if self._grid_pv is not None:
            op = self._grid_pv.pinned_operating_point(op)
        pinned_scaled: dict[int, Tensor] = {}
        if self._grid_pv is not None:
            for g in self._grid_pv.groups:
                for ki, gid in enumerate(g.gen_ids):
                    if gid in self.scaled_ids and int(g.state[..., ki]) != 0:
                        pinned_scaled[gid] = op[gid]["q_var"]
        op_scaled = dict(op)
        op_fixed = dict(op)
        for gid, q in pinned_scaled.items():
            op_scaled[gid] = {**op[gid], "q_var": 0.0}
            fixed_entry = {
                k: v for k, v in op[gid].items() if k not in ("p_per_phase_w",)
            }
            op_fixed[gid] = {**fixed_entry, "p_w": 0.0, "q_var": q}
        scaled = set(self.scaled_ids)
        fixed_ids = {
            int(a.id)
            for a in self.grid.appliances
            if isinstance(a, InjectionAppliance) and int(a.id) not in scaled
        } | set(pinned_scaled)
        self._plan_scaled = (
            self._plan(_keep_injections(self.grid, scaled), op_scaled)
            if scaled
            else None
        )
        self._plan_fixed = self._plan(_keep_injections(self.grid, fixed_ids), op_fixed)

    def _plan(self, grid: Grid, op: dict) -> InjectionPlan:
        with torch.no_grad():
            return build_injection_plan(
                grid,
                self.index,
                [self.f0],
                dtype=self.dtype,
                device=self.device,
                operating_point=op or None,
                param_overrides=self.param_overrides,
                symmetry=self.symmetry,
            )

    def add_voltage_source(self, node_id: int, v_set_pu: float) -> tuple[Phase, ...]:
        """Hold every phase of ``node_id`` at ``v_set_pu`` with a free reactive source.

        The fictitious source is a per-phase regulating terminal (each phase-to-ground
        magnitude at the setpoint, each with its own reactive power) of zero active
        power and unbounded reactive power. Returns the phases it holds (the node's
        non-neutral phases, in layout order).
        """
        node = next((nd for nd in self.grid.nodes if int(nd.id) == int(node_id)), None)
        if node is None:
            raise InputError(f"node {node_id} is not in the grid.")
        phases = tuple(ph for ph in node.phases if ph != Phase.N)
        if not phases:
            raise InputError(f"node {node_id} has no phase conductor to hold.")
        rows = [self.index.row(int(node_id), ph) for ph in phases]
        rows_t = torch.as_tensor([rows], dtype=torch.int64, device=self.device)
        held = set(rows)
        if self.fixed_rows is not None and held & set(self.fixed_rows.tolist()):
            raise InputError(
                f"node {node_id} is (or is fused to) an ideal-slack source node; its "
                "voltage is already fixed, so it has no QV curve."
            )
        if self._grid_pv is not None and held & set(self._grid_pv.rows.tolist()):
            raise InputError(
                f"node {node_id} is (or is fused to) the terminal of a voltage-regulating "
                "generator; its voltage is already held, so it has no QV curve."
            )
        v0 = self.v_base[rows[0]].reshape(1)
        inf = torch.full((1,), math.inf, dtype=self.rdt, device=self.device)
        self._source = PVGroup(
            gen_ids=(FICTITIOUS_SOURCE_ID,),
            rows=rows_t,
            flat_rows=rows_t.reshape(-1),
            v0=v0,
            v_set_pu=torch.full(
                (1,), float(v_set_pu), dtype=self.rdt, device=self.device
            ),
            q_min=-inf,
            q_max=inf,
            per_phase=True,
            seq=None,
            state=torch.zeros(1, dtype=torch.int8, device=self.device),
        )
        return phases

    @property
    def source_rows(self) -> Tensor:
        """Solve-layout rows the fictitious source holds (int64 ``[P]``)."""
        if self._source is None:
            raise InputError("no fictitious voltage source has been added.")
        return self._source.flat_rows

    def set_source_setpoint(self, v_set_pu: float) -> None:
        """Move the fictitious source's voltage setpoint [pu of the node's L-N base]."""
        if self._source is None:
            raise InputError("no fictitious voltage source has been added.")
        self._source = replace(
            self._source,
            v_set_pu=torch.full(
                (1,), float(v_set_pu), dtype=self.rdt, device=self.device
            ),
        )

    def source_q_per_phase(self, v: Tensor, lam: float = 0.0) -> Tensor:
        """Reactive power the fictitious source injects per held phase ``[P]`` [var]."""
        pv = self.pv
        if self._source is None or pv is None:
            raise InputError("no fictitious voltage source has been added.")
        with torch.no_grad():
            return pv.required_q_per_phase(self.nodal_residual(v, lam), v)[
                FICTITIOUS_SOURCE_ID
            ]

    # -- residuals -----------------------------------------------------------
    def nodal_residual(self, v: Tensor, lam) -> Tensor:
        """Complex nodal current mismatch ``F_c(V, λ)`` ``[*b, N]`` [A] (no row edits)."""
        out = _apply_y(self.y_eff, v) - self.i_slack
        out = out + injections_from_plan(self._plan_fixed, v).squeeze(-2)
        if self._plan_scaled is not None:
            out = out + lam * injections_from_plan(self._plan_scaled, v).squeeze(-2)
        return out

    def _real_residual(self, lam):
        """The solver's real-residual closure at a fixed ``λ`` (pinning + PV rows)."""
        y, i_s = self.y_eff, self.i_slack

        def rc(v: Tensor, _y: Tensor, _i: Tensor) -> Tensor:
            return self.nodal_residual(v, lam)

        return _make_real_residual(
            lambda: (y, i_s),
            rc,
            self.fixed_rows,
            (lambda: self.v_fixed) if self.fixed_rows is not None else None,
            self.n,
            self.cdt,
            pv=self.pv,
            equilibrate=self.equilibration,
        )

    def residual(self, x: Tensor, lam) -> Tensor:
        """Real residual ``R(x, λ)`` ``[*b, 2N]`` exactly as the solver forms it.

        ``x = [Re V; Im V]`` [V]; ``lam`` a float or a real 0-d tensor (differentiable,
        for ``∂R/∂λ``). Free rows carry the current mismatch [A], ideal-slack rows
        ``V − V_ref`` [V], regulating rows the active balance and the setpoint.
        """
        rr = self._real_residual(lam)
        y = self.y_eff
        return rr.state_residual(
            x, y.real, y.imag, self.i_slack.real, self.i_slack.imag
        )

    def scaled_power(self, v: Tensor, lam: float) -> Tensor:
        """Complex power drawn by the scaled appliances ``λ Σ V conj(I_scaled)`` [VA].

        Load convention (positive = consumed); a generating appliance contributes a
        negative active power.
        """
        if self._plan_scaled is None:
            return torch.zeros((), dtype=self.cdt, device=self.device)
        with torch.no_grad():
            i = injections_from_plan(self._plan_scaled, v).squeeze(-2)
            return lam * (v * torch.conj(i)).sum(-1)

    def factor(self, a: Tensor):
        """Equilibrated LU of a square real matrix ``[m, m]`` (the solver's scaling)."""
        return equilibrated_lu_factor(a.unsqueeze(0), mode=self.equilibration)

    def converged(self, x: Tensor, r: Tensor, dx: Tensor) -> tuple[bool, bool]:
        """The solver's per-unit test on a state, its residual and the last update.

        Returns ``(ok, in_floor_band)``: both criteria met, and both met within the
        band above the precision floor inside which the solver accepts a stalled
        iterate as converged.
        """
        n = self.n
        v = torch.complex(x[..., :n], x[..., n:])
        fc = torch.complex(r[..., :n], r[..., n:])
        dv = torch.complex(dx[..., :n], dx[..., n:])
        ok, in_band, _, _ = self.ctest.check(
            self.ctest.mismatch_rows_pu(v, fc), self.ctest.update_rows_pu(dv)
        )
        return bool(ok), bool(in_band)

    # -- fixed-λ solves ------------------------------------------------------
    def _seeds(self, lam: float) -> list:
        """Newton starts at a fixed ``λ``: no-load linear, const-Z, nominal profile."""

        def no_load():
            with torch.no_grad():
                return solve_harmonic(
                    self.y_eff,
                    self.i_slack,
                    fixed_rows=self.fixed_rows,
                    v_fixed=self.v_fixed,
                    equilibrate=self.equilibration,
                ).reshape(self.n)

        def const_z():
            return _linear_const_z_init(
                self.grid,
                self.f0,
                self.index,
                self.dtype,
                self.device,
                self.slack,
                self.operating_point,
                self.param_overrides,
                self.fixed_rows,
                self.v_fixed,
                None,
                self.fusion,
                self.equilibration,
            ).reshape(self.n)

        seeds = [const_z, no_load] if lam >= 0.5 else [no_load, const_z]
        pv = self.pv
        if pv is not None:
            seeds.append(
                lambda: _pv_nominal_init(
                    self.grid,
                    self.index,
                    pv,
                    self.rdt,
                    self.cdt,
                    self.device,
                    self.fixed_rows,
                    self.v_fixed,
                ).reshape(self.n)
            )
        return seeds

    def solve(self, lam: float, v_init: Optional[Tensor] = None) -> tuple[Tensor, bool]:
        """Newton solve at fixed ``λ`` and the current active set.

        Uses the solver's Newton loop from ``v_init`` (complex ``[N]``), or from the
        standard warm starts when it is ``None``. Returns ``(V [N], converged)``.
        """
        rr = self._real_residual(float(lam))
        starts = [lambda: v_init] if v_init is not None else self._seeds(float(lam))
        v = None
        for start in starts:
            out = _newton_forward(
                rr,
                start(),
                self.n,
                self.rdt,
                self.cdt,
                self.device,
                self.v_base,
                self.tol,
                self.tol_update_pu,
                self.s_base_va,
                self.max_iter,
                "dense",
                "full",
                self.fixed_rows,
            )
            v, conv = out[0].reshape(self.n), bool(out[3])
            if conv:
                return v, True
        return v, False

    def solve_with_switching(
        self, lam: float, v_init: Optional[Tensor] = None
    ) -> tuple[Tensor, bool]:
        """:meth:`solve` plus PV-to-PQ switching rounds until the active set settles."""
        v, conv = self.solve(lam, v_init)
        rounds = 0
        while conv:
            changed, _ = self.switch(v, lam)
            if not changed:
                return v, True
            rounds += 1
            if rounds >= self._pv_policy.max_rounds:
                _log.warning(
                    "continuation: the reactive-limit switching did not settle in %d "
                    "rounds at lambda=%.6g.",
                    rounds,
                    lam,
                )
                return v, False
            v, conv = self.solve(lam, v)
        return v, conv

    # -- reactive limits -----------------------------------------------------
    def would_switch(self, v: Tensor, lam: float) -> bool:
        """Would a reactive-limit check at ``(V, λ)`` change any terminal's bus type?"""
        pv = self._grid_pv
        if pv is None or not pv.enforce_q_limits:
            return False
        with torch.no_grad():
            _, changed = pv.switch(self.nodal_residual(v, lam), v)
        return bool(changed)

    def switch(self, v: Tensor, lam: float) -> tuple[bool, tuple[int, ...]]:
        """Apply one switching round at ``(V, λ)``; returns ``(changed, generator ids)``."""
        pv = self._grid_pv
        if pv is None or not pv.enforce_q_limits:
            return False, ()
        with torch.no_grad():
            nxt, changed = pv.switch(self.nodal_residual(v, lam), v)
            if not changed:
                return False, ()
            _, ids = pv.unsettled(nxt)
        self._grid_pv = nxt
        self._rebuild_plans()
        return True, ids

    # -- layout --------------------------------------------------------------
    def prolong(self, v: Tensor) -> Tensor:
        """Solve-layout voltages ``[..., N_red]`` -> the grid's full layout ``[..., N]``."""
        return v if self.fusion is None else self.fusion.prolong(v)


def _default_policy() -> PVTerminals:
    """Terminal-set policy when the grid has no regulating generator of its own."""
    return PVTerminals(
        groups=(),
        enforce_q_limits=False,
        hysteresis_pu=float(defaults.get("appliance.generator.q_limit_hysteresis_pu")),
        hysteresis_rel=float(
            defaults.get("appliance.generator.q_limit_hysteresis_rel")
        ),
        max_rounds=int(defaults.get("appliance.generator.q_limit_switch_rounds_max")),
    )


def _keep_injections(grid: Grid, ids: set[int]) -> Grid:
    """A grid copy in which only the injecting appliances ``ids`` stay in service.

    Taking the others out of service drops them from the injection plan while every
    resolution rule (per-phase split, ZIP law, inverter control) applies unchanged to
    the ones that remain. Sources and shunts are untouched.
    """
    kept = [
        a.model_copy(update={"in_service": False})
        if isinstance(a, InjectionAppliance) and int(a.id) not in ids
        else a
        for a in grid.appliances
    ]
    return grid.model_copy(update={"appliances": kept})


def build_continuation_problem(
    grid: Grid,
    *,
    scaled_ids: Sequence[int] = (),
    slack: str = "ideal",
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    operating_point: Optional[dict] = None,
    param_overrides: Optional[dict] = None,
    symmetry: Optional[str] = None,
    enforce_q_limits: Optional[bool] = None,
    equilibrate: Optional[str] = None,
    tol: Optional[float] = None,
    tol_update_pu: Optional[float] = None,
    s_base_va: Optional[float] = None,
    max_iter: int = 50,
) -> ContinuationProblem:
    """Validate ``grid`` like :func:`~pgml.solver.solve_power_flow` and build the problem.

    ``scaled_ids`` are the in-service injecting appliances ``λ`` multiplies. The
    operating point must describe a single scenario. Tolerances and ``enforce_q_limits``
    resolve their documented defaults when ``None``.
    """
    if slack not in ("ideal", "norton"):
        raise InputError(f"Unsupported slack {slack!r} (use 'ideal' or 'norton').")
    if _operating_point_batch_size(operating_point) > 1:
        raise InputError(
            "a continuation study takes a single operating point, not a scenario batch."
        )
    by_id = {int(a.id): a for a in grid.appliances}
    for aid in scaled_ids:
        a = by_id.get(int(aid))
        if a is None or not isinstance(a, InjectionAppliance):
            raise InputError(
                f"appliance {aid} is not a Load, Generator or Storage of the grid."
            )
        if not a.in_service:
            raise InputError(f"appliance {aid} is out of service; it cannot be scaled.")
    eq_mode = resolve_equilibration(equilibrate)
    tol, tol_update_pu, s_base_va = _resolve_tolerances(tol, tol_update_pu, s_base_va)
    zero = zero_impedance_branches(grid, param_overrides=param_overrides)
    fusion = resolve_fusion(grid, None, param_overrides=param_overrides, zero=zero)
    check_branch_impedances(
        grid, fusion=fusion, param_overrides=param_overrides, zero_branches=zero
    )
    check_connectivity(grid)
    full_index = node_phase_index(grid)
    index = fusion.index if fusion is not None else full_index
    if device is None:
        device = torch.device("cpu")
    device = torch.device(device)
    rdt, cdt = _rdtype(dtype), _cdtype(dtype)
    asymmetric = resolve_asymmetric(grid, operating_point, mode=symmetry)
    if fusion is not None:
        fixed = (
            _slack_rows_and_vref(grid, full_index, rdt, cdt, device)[0]
            if slack == "ideal"
            else None
        )
        _check_fused_row_conflicts(grid, fusion, full_index, fixed)
    grid_pv = collect_pv_terminals(
        grid,
        index,
        rdt,
        cdt,
        device,
        operating_point,
        enforce_q_limits=enforce_q_limits,
    )
    return ContinuationProblem(
        grid=grid,
        index=index,
        full_index=full_index,
        fusion=fusion,
        scaled_ids=tuple(int(a) for a in scaled_ids),
        operating_point=operating_point,
        param_overrides=param_overrides,
        symmetry="asymmetric" if asymmetric else "symmetric",
        slack=slack,
        dtype=dtype,
        device=device,
        grid_pv=grid_pv,
        equilibration=eq_mode,
        tol=tol,
        tol_update_pu=tol_update_pu,
        s_base_va=s_base_va,
        max_iter=max_iter,
    )
