"""Voltage-stability curves: continuation power flow (P-V) and the Q-V curve.

:func:`pv_curve` follows the fundamental power-flow solution while a loading parameter
``λ`` scales the power of a set of appliances, THROUGH the nose of the P-V curve, and
reports the maximum loadability ``λ*``. :func:`qv_curve` holds one bus at a sweep of
voltage setpoints with a fictitious reactive source and records the reactive power that
source must inject; the minimum of that curve is the bus's reactive power margin.

Both studies solve the phase-domain fundamental power flow of pgml (unbalanced grids,
ZIP loads, inverter control laws, voltage-regulating generators with reactive limits,
fused zero-impedance branches), with the same residual, per-unit convergence test and
equilibration as :func:`pgml.solver.solve_power_flow`.

Continuation method (:func:`pv_curve`)
--------------------------------------
- Parameter: ``λ`` multiplies the complex power of the selected appliances at the solved
  voltage (every component of a ZIP law); ``λ = 1`` is the base operating point (the
  grid's nameplate values, or ``config.operating_point``). The other appliances stay at
  their base values.
- Coordinates: ``ẑ = (x / V_base, λ / λ_ref)`` with ``x = [Re V; Im V]`` and
  ``λ_ref = max(1, |λ|)`` re-based along the trace, so the arc length mixes the relative
  change of ``λ`` with the per-unit voltage change (Euclidean over the node-phase rows).
  ``step`` is measured in that arc length; the fold looks the same whether the nose lies
  at ``λ* = 2`` or at ``λ* = 700``.
- Predictor: the unit tangent ``t`` of the solution curve, from
  ``[∂R/∂ẑ; t_prevᵀ] t = e`` (oriented along the previous tangent, initially
  increasing ``λ``).
- Corrector: Newton on the pseudo-arc-length system
  ``R(x, λ) = 0, tᵀ(ẑ − ẑ_pred) = 0``, whose Jacobian stays regular at the nose; dense
  real Jacobian by automatic differentiation, equilibrated LU.
- Step control: the step grows or shrinks with the corrector iteration count (target 4,
  factor clipped to ``[0.5, 2]``); a failed or over-long correction, or a tangent that
  turns by more than 0.25 rad within the step, halves it.
- Nose: the sign change of the tangent's ``λ`` component; the fold is then located by
  bisection on the arc length to ``1e-6`` and inserted as a curve point.
- Reactive limits: a PV-to-PQ (or release) switch of a regulating generator is located by
  bisection on the arc length, the active set switched there, and the point re-solved. A
  switch after which ``λ`` can only decrease is a limit-induced nose.

Outputs are host numpy arrays; neither study is differentiable (both run detached), and
neither mutates its inputs. Every tensor operation honours ``device`` and ``dtype``
(complex64 or complex128); a single grid and a single operating point per call.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Literal, Optional, Union

import numpy as np
import torch
from torch import Tensor

from pgml.errors import ConvergenceError, InputError
from pgml.schemas.grid_schema import (
    ApplianceBase,
    Generator,
    Grid,
    InjectionAppliance,
    Load,
    Phase,
)
from pgml.simulation import SimulationConfig
from pgml.solver._continuation import ContinuationProblem, build_continuation_problem

from ._common import check_study, resolve_complex_dtype

_PHASE_VALUES = tuple(p.value for p in (Phase.A, Phase.B, Phase.C, Phase.N))
_CORRECTOR_MAX_ITER = 10
_CORRECTOR_TARGET_ITER = 4
_LOCATE_TOL = 1.0e-6
_LOCATE_MAX_ITER = 60
_SWITCH_TRIAL_STEP = 1.0e-3
_QV_SUBSTEP_DEPTH = 4
_MAX_TURN_COS = math.cos(0.25)  # largest tangent rotation accepted in one step
_REGULAR_STOPS = ("nose", "lambda_max", "v_stop", "lambda_start")


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class PVCurve:
    """A traced P-V curve (:func:`pv_curve`).

    Attributes
    ----------
    lambdas:
        Loading parameter of every curve point ``[K]``; ``λ = 1`` is the base operating
        point. Increases up to the nose and decreases after it.
    p_total_w:
        Total active power of the scaled appliances at each point ``[K]`` [W], at the
        solved voltage (``λ · P_base`` for constant-power appliances). Consumption is
        positive for ``direction="load"``, generation is positive for
        ``direction="generation"``.
    v_pu:
        Voltage magnitude of every node-phase row ``[K, R]`` [pu of the node's
        line-to-neutral rated voltage], in the grid's row layout.
    node_ids, phases:
        Row index of ``v_pu``: node id ``[R]`` and phase of each row, as the schema
        value of :class:`~pgml.schemas.Phase` (``"a"``, ``"b"``, ``"c"``, ``"n"``).
    appliance_ids:
        Ids of the appliances ``λ`` scaled.
    direction:
        ``"load"`` or ``"generation"``.
    nose_index:
        Index of the nose point in the curve, ``None`` when the trace stopped before it.
    lambda_max:
        Maximum loadability ``λ*`` (``lambdas[nose_index]``), ``None`` without a nose.
    nose_kind:
        ``"saddle_node"`` (the Jacobian fold) or ``"limit_induced"`` (a reactive limit
        reached at which ``λ`` can only decrease); ``None`` without a nose.
    critical_node, critical_phase:
        The row with the lowest voltage at the nose (neutral rows excluded); the phase
        as its schema value (``"a"``, ``"b"``, ``"c"``).
    converged:
        ``True`` when the trace ended on a regular stop (``stop_reason`` one of
        ``"nose"``, ``"lambda_max"``, ``"v_stop"``, ``"lambda_start"``). Every recorded
        point is a converged power flow either way.
    stop_reason:
        Why the trace ended: ``"nose"`` (``stop_at_nose``), ``"lambda_max"`` (the cap
        was reached before a nose), ``"v_stop"`` (a voltage fell below ``v_stop_pu``),
        ``"lambda_start"`` (the lower branch returned below the starting ``λ``),
        ``"max_steps"``, ``"step_min"`` (the corrector failed at the smallest step), or
        ``"switch_failed"`` (no solution after a reactive-limit switch).
    events:
        Reactive-limit switches of regulating generators:
        ``{"index", "lambda", "generator_ids"}`` per event, ``index`` the curve point
        at which the new active set applies.
    """

    lambdas: np.ndarray
    p_total_w: np.ndarray
    v_pu: np.ndarray
    node_ids: np.ndarray
    phases: tuple[str, ...]
    appliance_ids: tuple[int, ...]
    direction: str
    nose_index: Optional[int]
    lambda_max: Optional[float]
    nose_kind: Optional[str]
    critical_node: Optional[int]
    critical_phase: Optional[str]
    converged: bool
    stop_reason: str
    events: tuple[dict, ...] = field(default_factory=tuple)

    def voltage_pu(
        self, node_id: int, phase: Optional[Union[str, Phase]] = None
    ) -> np.ndarray:
        """Voltage of one node along the curve: ``[K]`` for a phase, else ``[K, P]``.

        ``phase`` is a :class:`~pgml.schemas.Phase` or its letter (``"a"``; case is
        ignored); without it the node's rows are returned in layout order.
        """
        mask = self.node_ids == int(node_id)
        if phase is not None:
            mask &= np.asarray(self.phases) == _phase_value(phase)
        if not mask.any():
            raise InputError(
                f"node {node_id} phase {phase} is not in the curve's rows."
            )
        out = self.v_pu[:, mask]
        return out[:, 0] if phase is not None else out

    def to_dict(self) -> dict:
        """JSON-ready dict (plain lists, units in the key names)."""
        return {
            "study": "pv_curve",
            "direction": self.direction,
            "appliance_ids": [int(a) for a in self.appliance_ids],
            "lambdas": self.lambdas.tolist(),
            "p_total_w": self.p_total_w.tolist(),
            "node_ids": self.node_ids.tolist(),
            "phases": list(self.phases),
            "v_pu": self.v_pu.tolist(),
            "nose_index": self.nose_index,
            "lambda_max": self.lambda_max,
            "nose_kind": self.nose_kind,
            "critical_node": self.critical_node,
            "critical_phase": self.critical_phase,
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "events": [dict(e) for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PVCurve":
        """Inverse of :meth:`to_dict`.

        Raises :class:`~pgml.errors.InputError` unless ``data["study"]`` is
        ``"pv_curve"``.
        """
        check_study(data, "pv_curve")
        n_rows = len(data["node_ids"])
        return cls(
            lambdas=np.asarray(data["lambdas"], dtype=np.float64),
            p_total_w=np.asarray(data["p_total_w"], dtype=np.float64),
            v_pu=np.asarray(data["v_pu"], dtype=np.float64).reshape(-1, n_rows),
            node_ids=np.asarray(data["node_ids"], dtype=np.int64),
            phases=tuple(data["phases"]),
            appliance_ids=tuple(int(a) for a in data["appliance_ids"]),
            direction=data["direction"],
            nose_index=data["nose_index"],
            lambda_max=data["lambda_max"],
            nose_kind=data["nose_kind"],
            critical_node=data["critical_node"],
            critical_phase=data["critical_phase"],
            converged=bool(data["converged"]),
            stop_reason=data["stop_reason"],
            events=tuple(
                {
                    "index": int(e["index"]),
                    "lambda": float(e["lambda"]),
                    "generator_ids": [int(g) for g in e["generator_ids"]],
                }
                for e in data.get("events", ())
            ),
        )


@dataclass(frozen=True, eq=False)
class QVCurve:
    """A Q-V curve at one bus (:func:`qv_curve`).

    Attributes
    ----------
    node_id, phases:
        The bus and the phases the fictitious source holds (its non-neutral phases),
        each as its schema value (``"a"``, ``"b"``, ``"c"``).
    v_pu:
        Voltage setpoints ``[K]`` [pu of the node's line-to-neutral rated voltage],
        ascending; every held phase is at this magnitude.
    q_var:
        Total reactive power the fictitious source injects ``[K]`` [var] (positive =
        injected into the grid, i.e. capacitive support); ``NaN`` where the power flow
        did not converge.
    q_phase_var:
        The same per held phase ``[K, P]`` [var].
    converged:
        Per setpoint ``[K]`` (bool).
    q_margin_var:
        Reactive power margin ``−min(q_var)`` [var]: how much reactive LOAD the bus can
        take before the voltage collapses (negative: the bus needs that much reactive
        support to operate at all). ``None`` when no point converged.
    v_at_min_pu:
        Voltage at the minimum of the curve [pu].
    minimum_in_range:
        ``False`` when the minimum lies at the end of the converged range, so the
        true minimum may lie outside ``v_range_pu`` (``q_margin_var`` is then an
        upper bound on the margin).
    """

    node_id: int
    phases: tuple[str, ...]
    v_pu: np.ndarray
    q_var: np.ndarray
    q_phase_var: np.ndarray
    converged: np.ndarray
    q_margin_var: Optional[float]
    v_at_min_pu: Optional[float]
    minimum_in_range: bool

    def to_dict(self) -> dict:
        """JSON-ready dict (plain lists, ``None`` for a non-converged value)."""
        return {
            "study": "qv_curve",
            "node_id": int(self.node_id),
            "phases": list(self.phases),
            "v_pu": self.v_pu.tolist(),
            "q_var": _nan_to_none(self.q_var),
            "q_phase_var": _nan_to_none(self.q_phase_var),
            "converged": [bool(c) for c in self.converged],
            "q_margin_var": self.q_margin_var,
            "v_at_min_pu": self.v_at_min_pu,
            "minimum_in_range": self.minimum_in_range,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QVCurve":
        """Inverse of :meth:`to_dict`.

        Raises :class:`~pgml.errors.InputError` unless ``data["study"]`` is
        ``"qv_curve"``.
        """
        check_study(data, "qv_curve")
        n_ph = len(data["phases"])
        return cls(
            node_id=int(data["node_id"]),
            phases=tuple(data["phases"]),
            v_pu=np.asarray(data["v_pu"], dtype=np.float64),
            q_var=_none_to_nan(data["q_var"]),
            q_phase_var=_none_to_nan(data["q_phase_var"]).reshape(-1, n_ph),
            converged=np.asarray(data["converged"], dtype=bool),
            q_margin_var=data["q_margin_var"],
            v_at_min_pu=data["v_at_min_pu"],
            minimum_in_range=bool(data["minimum_in_range"]),
        )


def _phase_value(phase: Union[str, Phase]) -> str:
    """Schema value (``"a"`` ...) of a :class:`Phase` or a phase letter (any case)."""
    if isinstance(phase, Phase):
        return phase.value
    value = str(phase).lower()
    if value not in _PHASE_VALUES:
        raise InputError(
            f"phase must be a Phase or one of {list(_PHASE_VALUES)}, got {phase!r}."
        )
    return value


def _nan_to_none(a: np.ndarray) -> list:
    if a.ndim == 1:
        return [None if math.isnan(x) else float(x) for x in a]
    return [_nan_to_none(row) for row in a]


def _none_to_nan(values) -> np.ndarray:
    def conv(v):
        if isinstance(v, list):
            return [conv(e) for e in v]
        return math.nan if v is None else float(v)

    return np.asarray(conv(values), dtype=np.float64)


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------
def pv_curve(
    grid: Grid,
    *,
    selector: Optional[Union[Iterable[int], Callable[[ApplianceBase], bool]]] = None,
    direction: Literal["load", "generation"] = "load",
    config: Optional[SimulationConfig] = None,
    lambda_start: float = 1.0,
    lambda_max: Optional[float] = None,
    step: float = 0.05,
    step_min: float = 1.0e-4,
    step_max: Optional[float] = None,
    max_steps: int = 500,
    v_stop_pu: float = 0.3,
    stop_at_nose: bool = False,
    dtype: Union[str, torch.dtype] = torch.complex128,
    device: Optional[Union[str, torch.device]] = None,
) -> PVCurve:
    """Trace the P-V curve through its nose by predictor-corrector continuation.

    ``λ`` multiplies the complex power of the selected appliances (all components of a
    ZIP law, at the solved voltage); ``λ = 1`` is the base operating point. The trace
    starts with a plain power-flow solve at ``lambda_start`` and follows the solution
    curve with a tangent predictor and a pseudo-arc-length Newton corrector, which stays
    regular where a plain ``λ`` sweep diverges, so it passes the nose (the maximum
    loadability ``λ*``) and continues on the lower branch. The method is described in the
    module documentation.

    Parameters
    ----------
    grid:
        The grid; not modified.
    selector:
        Appliances to scale: an iterable of appliance ids or a predicate on an
        appliance. ``None`` selects every in-service :class:`~pgml.schemas.Load`
        (``direction="load"``) or :class:`~pgml.schemas.Generator`
        (``direction="generation"``). Selected appliances must be in-service loads,
        generators or storage units.
    direction:
        ``"load"`` or ``"generation"``: the default selection, and the sign of
        :attr:`PVCurve.p_total_w`.
    config:
        :class:`~pgml.simulation.SimulationConfig` for ``slack``, ``symmetry``,
        ``operating_point`` (the base, ``λ = 1``), the per-unit tolerances, ``max_iter``
        of the starting solve and ``enforce_q_limits``; ``calculation`` and the harmonic
        fields are ignored. ``None`` uses the defaults.
    lambda_start:
        ``λ`` of the first curve point (default ``1.0``, the base; ``0.0`` starts with
        the selected appliances switched off, which always solves when the rest of the
        grid does).
    lambda_max:
        Stop when ``λ`` reaches this value before a nose (the last point lies exactly
        on it). ``None``: no cap.
    step:
        Initial arc-length step (relative change of ``λ`` mixed with the per-unit
        voltage change; ``0.05`` is about ``Δλ = 0.05`` at ``λ = 1`` on a stiff grid).
    step_min, step_max:
        Bounds of the adaptive step; ``step_max=None`` leaves it unbounded (the
        curvature guard still limits it). Set ``step_max`` for a denser curve.
    max_steps:
        Largest number of curve points.
    v_stop_pu:
        Stop once the lowest phase voltage falls below this value [pu], on either
        branch (a curve without a nose, e.g. of constant-impedance loads, ends here).
    stop_at_nose:
        Stop right after the nose is located instead of tracing the lower branch.
    dtype:
        ``torch.complex128`` (default) or ``torch.complex64``, or their names
        (``"complex128"``, ``"complex64"``).
    device:
        Device of the computation (default CPU).

    Returns
    -------
    PVCurve
        Curve points, the nose and the stop reason.

    Raises
    ------
    InputError
        Bad options (including an unknown ``direction`` or ``dtype``), an empty
        selection, or an appliance that cannot be scaled.
    ConvergenceError
        The power flow at ``lambda_start`` does not solve.

    Notes
    -----
    Every tensor operation runs on ``device`` at ``dtype``; at complex64 the per-unit
    tolerances are capped by the precision floor of the solver. The dense real Jacobian
    is ``[2N, 2N + 1]`` for ``N`` node-phase rows, so memory grows with ``N²``.
    Detached: not differentiable.
    """
    if direction not in ("load", "generation"):
        raise InputError(
            f"direction must be 'load' or 'generation', got {direction!r}."
        )
    if not (step > 0 and step_min > 0 and step_min <= step):
        raise InputError("need 0 < step_min <= step.")
    step_max = math.inf if step_max is None else float(step_max)
    if step_max < step:
        raise InputError("step_max must be at least step.")
    if lambda_max is not None and lambda_max <= lambda_start:
        raise InputError("lambda_max must exceed lambda_start.")
    if max_steps < 2:
        raise InputError("max_steps must be at least 2.")
    cdt = resolve_complex_dtype(dtype)
    cfg = config or SimulationConfig(calculation="power_flow")
    ids = _select(grid, selector, direction)
    with _quiet_pgml():
        problem = _problem(grid, cfg, cdt, device, scaled_ids=ids)
        return _Tracer(problem).trace(
            ids=ids,
            direction=direction,
            lambda_start=float(lambda_start),
            lambda_cap=None if lambda_max is None else float(lambda_max),
            step=float(step),
            step_min=float(step_min),
            step_max=step_max,
            max_steps=int(max_steps),
            v_stop_pu=float(v_stop_pu),
            stop_at_nose=bool(stop_at_nose),
            start_max_iter=int(cfg.max_iter),
        )


def qv_curve(
    grid: Grid,
    node_id: int,
    *,
    v_range_pu: tuple[float, float] = (0.8, 1.1),
    steps: int = 31,
    config: Optional[SimulationConfig] = None,
    dtype: Union[str, torch.dtype] = torch.complex128,
    device: Optional[Union[str, torch.device]] = None,
) -> QVCurve:
    """Q-V curve of one bus: the reactive power that holds it at each voltage.

    A fictitious voltage-controlled reactive source at ``node_id`` holds every
    non-neutral phase at the setpoint (phase-to-ground magnitude, each phase with its own
    reactive power, zero active power, no reactive limit); the setpoint sweeps
    ``steps`` equally spaced values over ``v_range_pu`` and the source's reactive
    injection is recorded. Fixing the voltage removes the fold, so each point is a
    regular power flow even on the unstable side of the curve. The sweep starts at the
    setpoint nearest the bus's base voltage and walks outward in both directions, each
    solve warm-started from its neighbour.

    The minimum of the curve is refined by a parabola through the lowest sampled point
    and its two neighbours; ``q_margin_var = −q_min``.

    Parameters
    ----------
    grid:
        The grid; not modified.
    node_id:
        The bus; not a slack bus and not the terminal of a regulating generator.
    v_range_pu:
        ``(low, high)`` setpoint range [pu of the node's line-to-neutral rated voltage].
    steps:
        Number of setpoints (at least 3).
    config:
        As for :func:`pv_curve` (slack, symmetry, operating point, tolerances,
        ``max_iter``, ``enforce_q_limits`` of the grid's own regulating generators).
    dtype, device:
        As for :func:`pv_curve`.

    Returns
    -------
    QVCurve

    Raises
    ------
    InputError
        Bad options (including an unknown ``dtype``), an unknown node, or a node
        whose voltage is already held.
    ConvergenceError
        The base power flow (without the fictitious source) does not solve.

    Notes
    -----
    Every tensor operation runs on ``device`` at ``dtype``. Detached: not
    differentiable.
    """
    lo, hi = (float(v) for v in v_range_pu)
    if not (0.0 < lo < hi):
        raise InputError("v_range_pu must be (low, high) with 0 < low < high.")
    if steps < 3:
        raise InputError("steps must be at least 3.")
    cdt = resolve_complex_dtype(dtype)
    cfg = config or SimulationConfig(calculation="power_flow")
    with _quiet_pgml():
        problem = _problem(grid, cfg, cdt, device, scaled_ids=())
        problem.max_iter = int(cfg.max_iter)
        v_base, ok = problem.solve_with_switching(0.0)
        if not ok:
            raise ConvergenceError(
                "qv_curve: the base power flow does not converge, so there is no "
                "operating point to start the sweep from."
            )
        phases = problem.add_voltage_source(int(node_id), 1.0)
        rows = problem.source_rows
        v0_pu = float((v_base[rows].abs() / problem.v_base[rows]).mean().cpu())
        setpoints = np.linspace(lo, hi, int(steps))
        k0 = int(np.argmin(np.abs(setpoints - v0_pu)))
        q_phase = np.full((len(setpoints), len(phases)), np.nan)
        conv = np.zeros(len(setpoints), dtype=bool)
        for order in (range(k0, len(setpoints)), range(k0 - 1, -1, -1)):
            v_prev, sp_prev = v_base, v0_pu
            for k in order:
                v, ok = _held_solve(problem, v_prev, sp_prev, float(setpoints[k]))
                if ok:
                    q_phase[k] = problem.source_q_per_phase(v).cpu().numpy()
                    conv[k] = True
                    v_prev, sp_prev = v, float(setpoints[k])
    q = q_phase.sum(axis=1)
    q[~conv] = np.nan
    q_min, v_min, inside = _curve_minimum(setpoints, q, conv)
    return QVCurve(
        node_id=int(node_id),
        phases=tuple(ph.value for ph in phases),
        v_pu=setpoints,
        q_var=q,
        q_phase_var=q_phase,
        converged=conv,
        q_margin_var=None if q_min is None else -q_min,
        v_at_min_pu=v_min,
        minimum_in_range=inside,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@contextmanager
def _quiet_pgml() -> Iterator[None]:
    """Keep the per-solve modelling INFO logs of the many inner solves out of the log."""
    log = logging.getLogger("pgml")
    prev = log.level
    log.setLevel(max(prev, logging.WARNING))
    try:
        yield
    finally:
        log.setLevel(prev)


def _problem(grid, cfg: SimulationConfig, dtype: torch.dtype, device, *, scaled_ids):
    return build_continuation_problem(
        grid,
        scaled_ids=scaled_ids,
        slack=cfg.slack,
        dtype=dtype,
        device=torch.device(device) if device is not None else None,
        operating_point=cfg.operating_point,
        symmetry=cfg.symmetry,
        enforce_q_limits=cfg.enforce_q_limits,
        tol=cfg.tol,
        tol_update_pu=cfg.tol_update_pu,
        s_base_va=cfg.s_base_va,
        max_iter=_CORRECTOR_MAX_ITER,
    )


def _select(grid: Grid, selector, direction: str) -> tuple[int, ...]:
    """Resolve ``selector`` to the ids of in-service injecting appliances."""
    if selector is None:
        kind = Load if direction == "load" else Generator
        ids = tuple(
            int(a.id) for a in grid.appliances if isinstance(a, kind) and a.in_service
        )
        if not ids:
            raise InputError(
                f"the grid has no in-service {kind.__name__}; nothing to scale for "
                f"direction={direction!r}."
            )
        return ids
    if callable(selector):
        ids = tuple(
            int(a.id)
            for a in grid.appliances
            if isinstance(a, InjectionAppliance) and a.in_service and selector(a)
        )
    else:
        ids = tuple(dict.fromkeys(int(i) for i in selector))
    if not ids:
        raise InputError("the selector selects no appliance.")
    return ids


def _held_solve(
    problem: ContinuationProblem, v_prev: Tensor, sp_prev: float, sp: float
) -> tuple[Tensor, bool]:
    """Solve with the held bus at ``sp``, warm-started from the solution at ``sp_prev``.

    The held rows of the warm start are moved onto the new setpoint first. A setpoint
    the solve cannot reach in one move is approached in halved sub-steps from the last
    converged one (up to :data:`_QV_SUBSTEP_DEPTH` halvings).
    """
    rows = problem.source_rows
    for depth in range(_QV_SUBSTEP_DEPTH + 1):
        n_sub = 2**depth
        v, start, ok = v_prev, sp_prev, True
        for j in range(1, n_sub + 1):
            target = sp_prev + (sp - sp_prev) * j / n_sub
            seed = v.clone()
            seed[rows] = v[rows] * (target / start)
            problem.set_source_setpoint(target)
            v, ok = problem.solve_with_switching(0.0, seed)
            if not ok:
                break
            start = target
        if ok:
            return v, True
    return v, False


def _curve_minimum(
    v: np.ndarray, q: np.ndarray, conv: np.ndarray
) -> tuple[Optional[float], Optional[float], bool]:
    """Minimum of a sampled curve, refined by a parabola through its neighbours."""
    if not conv.any():
        return None, None, False
    idx = np.flatnonzero(conv)
    k = int(idx[np.argmin(q[idx])])
    inside = k - 1 >= 0 and k + 1 < len(v) and conv[k - 1] and conv[k + 1]
    if not inside:
        return float(q[k]), float(v[k]), False
    x0, x1, x2 = v[k - 1], v[k], v[k + 1]
    y0, y1, y2 = q[k - 1], q[k], q[k + 1]
    den = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
    if a <= 0.0:
        return float(y1), float(x1), True
    xv = -b / (2.0 * a)
    yv = y1 + a * (xv - x1) ** 2 + (2.0 * a * x1 + b) * (xv - x1)
    return float(yv), float(xv), True


# ---------------------------------------------------------------------------
# the continuation
# ---------------------------------------------------------------------------
@dataclass
class _Point:
    lam: float
    v: Tensor  # solve-layout voltages [N] (complex)
    p: float  # active power of the scaled appliances, load convention [W]


class _Tracer:
    """Pseudo-arc-length continuation of :class:`ContinuationProblem` in ``λ``."""

    def __init__(self, problem: ContinuationProblem) -> None:
        self.p = problem
        n = problem.n
        one = torch.ones(1, dtype=problem.rdt, device=problem.device)
        self.d = torch.cat([problem.v_base.repeat(2), one])
        self.e_lam = torch.zeros(2 * n + 1, dtype=problem.rdt, device=problem.device)
        self.e_lam[-1] = 1.0

    # -- coordinates ---------------------------------------------------------
    def to_z(self, v: Tensor, lam: float) -> Tensor:
        lam_t = torch.full((1,), lam, dtype=self.p.rdt, device=self.p.device)
        return torch.cat([v.real, v.imag, lam_t]) / self.d

    def rescale(self, zh: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Re-base the ``λ`` coordinate on ``max(1, |λ|)`` once it is off by 2x.

        Keeps arc length in RELATIVE changes of ``λ``, so the fold has the same
        geometry whether the nose lies at ``λ* = 2`` or ``λ* = 700``. Returns the
        point and the unit tangent in the new coordinates.
        """
        lam = float((zh[-1] * self.d[-1]).cpu())
        ref = max(1.0, abs(lam))
        if 0.5 <= ref / float(self.d[-1].cpu()) <= 2.0:
            return zh, t
        z, dz = zh * self.d, t * self.d
        self.d = self.d.clone()
        self.d[-1] = ref
        t_new = dz / self.d
        return z / self.d, t_new / torch.linalg.vector_norm(t_new)

    def split(self, zh: Tensor) -> tuple[Tensor, Tensor]:
        z = zh * self.d
        return z[:-1], z[-1]

    def voltage(self, zh: Tensor) -> tuple[Tensor, float]:
        x, lam = self.split(zh)
        n = self.p.n
        return torch.complex(x[:n], x[n:]), float(lam)

    # -- linear algebra ------------------------------------------------------
    def residual(self, zh: Tensor) -> Tensor:
        x, lam = self.split(zh)
        return self.p.residual(x, lam)

    def jacobian(self, zh: Tensor) -> Tensor:
        return torch.autograd.functional.jacobian(self.residual, zh, vectorize=True)

    def tangent(self, zh: Tensor, t_prev: Tensor) -> Tensor:
        with torch.no_grad():
            a = torch.cat([self.jacobian(zh), t_prev.unsqueeze(0)], dim=0)
            t = self.p.factor(a).solve(self.e_lam.unsqueeze(0))[0]
            return t / torch.linalg.vector_norm(t)

    def correct(self, z_pred: Tensor, t: Tensor) -> tuple[Tensor, int, bool]:
        """Newton on ``R = 0, tᵀ(ẑ − ẑ_pred) = 0`` from ``ẑ_pred``.

        Converged when the solver's per-unit criteria hold, or when the iteration ends
        inside the solver's band above the precision floor (the complex64 case near the
        nose, where the floor, not the tolerance, limits the attainable mismatch).
        """
        zh = z_pred
        in_band = False
        with torch.no_grad():
            for it in range(1, _CORRECTOR_MAX_ITER + 1):
                r = self.residual(zh)
                a = torch.cat([self.jacobian(zh), t.unsqueeze(0)], dim=0)
                g = (t * (zh - z_pred)).sum().reshape(1)
                dz = self.p.factor(a).solve(-torch.cat([r, g]).unsqueeze(0))[0]
                if not bool(torch.isfinite(dz).all()):
                    return zh, it, False
                zh = zh + dz
                x, _ = self.split(zh)
                ok, in_band = self.p.converged(x, self.residual(zh), (dz * self.d)[:-1])
                if ok:
                    return zh, it, True
        return zh, _CORRECTOR_MAX_ITER, in_band

    def locate(
        self, z0: Tensor, t0: Tensor, h: float, flag
    ) -> tuple[Optional[tuple[Tensor, Tensor]], list[tuple[Tensor, Tensor]]]:
        """Bisect ``s ∈ (0, h]`` for the first arc point where ``flag(ẑ, t)`` holds.

        ``flag`` is false at ``s = 0`` and true at ``s = h``. Returns the closest
        point found on the true side (``None`` if no bisection point landed there)
        and every corrected point evaluated.
        """
        lo, hi = 0.0, h
        hit = None
        seen: list[tuple[Tensor, Tensor]] = []
        for _ in range(_LOCATE_MAX_ITER):
            if hi - lo <= _LOCATE_TOL:
                break
            s = 0.5 * (lo + hi)
            zs, _, ok = self.correct(z0 + s * t0, t0)
            if not ok:
                break
            ts = self.tangent(zs, t0)
            seen.append((zs, ts))
            if flag(zs, ts):
                hi, hit = s, (zs, ts)
            else:
                lo = s
        return hit, seen

    def branch_direction(self, zh: Tensor, t_prev: Tensor, h: float) -> Tensor:
        """Continuation direction after a reactive-limit switch at ``ẑ``.

        The tangent of the new active set, oriented along ``t_prev``, unless a short
        trial step along it would switch the active set straight back (a unit pinned at
        its upper limit whose voltage would rise above the setpoint, or the converse):
        that half of the new branch is not a solution of the limited model, so the
        trace continues along the other half. A resulting decrease of ``λ`` is the
        limit-induced nose.
        """
        t_new = self.tangent(zh, t_prev)
        eps = min(h, _SWITCH_TRIAL_STEP)
        z_try, _, ok = self.correct(zh + eps * t_new, t_new)
        if ok and self.p.would_switch(*self.voltage(z_try)):
            return -t_new
        return t_new

    # -- bookkeeping ---------------------------------------------------------
    def point(self, zh: Tensor) -> _Point:
        v, lam = self.voltage(zh)
        return _Point(lam=lam, v=v, p=float(self.p.scaled_power(v, lam).real.cpu()))

    def min_v_pu(self, v: Tensor) -> float:
        live = self.p.index.phase_codes.to(v.device) != 3
        return float((v.abs() / self.p.v_base)[live].min().cpu())

    # -- the trace -----------------------------------------------------------
    def trace(
        self,
        *,
        ids: tuple[int, ...],
        direction: str,
        lambda_start: float,
        lambda_cap: Optional[float],
        step: float,
        step_min: float,
        step_max: float,
        max_steps: int,
        v_stop_pu: float,
        stop_at_nose: bool,
        start_max_iter: int,
    ) -> PVCurve:
        prob = self.p
        corrector_iter = prob.max_iter
        prob.max_iter = start_max_iter
        v0, ok = prob.solve_with_switching(lambda_start)
        prob.max_iter = corrector_iter
        if not ok:
            raise ConvergenceError(
                f"pv_curve: the power flow at lambda_start={lambda_start:g} does not "
                "converge; start lower (lambda_start=0 switches the scaled appliances "
                "off)."
            )
        zh = self.to_z(v0, lambda_start)
        t = self.tangent(zh, self.e_lam)
        points = [self.point(zh)]
        events: list[dict] = []
        nose_index: Optional[int] = None
        nose_kind: Optional[str] = None
        h = step
        stop = "max_steps"
        while len(points) < max_steps:
            z_new, iters, ok = self.correct(zh + h * t, t)
            if ok and float(torch.linalg.vector_norm(z_new - (zh + h * t))) > h:
                ok = False  # the corrector jumped: the predictor overshot
            if not ok:
                h *= 0.5
                if h < step_min:
                    stop = "step_min"
                    break
                continue
            t_new = self.tangent(z_new, t)
            if float((t * t_new).sum()) < _MAX_TURN_COS and h > step_min:
                h = max(step_min, 0.5 * h)  # the curve bends inside the step
                continue
            v_new, lam_new = self.voltage(z_new)

            if prob.would_switch(v_new, lam_new):
                hit, _ = self.locate(
                    zh, t, h, lambda zs, ts: prob.would_switch(*self.voltage(zs))
                )
                z_sw = hit[0] if hit is not None else z_new
                v_sw, lam_sw = self.voltage(z_sw)
                _, gen_ids = prob.switch(v_sw, lam_sw)
                v_sw, ok = prob.solve_with_switching(lam_sw, v_sw)
                if not ok:
                    stop = "switch_failed"
                    break
                z_new = self.to_z(v_sw, lam_sw)
                t_new = self.branch_direction(z_new, t, h)
                points.append(self.point(z_new))
                events.append(
                    {
                        "index": len(points) - 1,
                        "lambda": lam_sw,
                        "generator_ids": [int(g) for g in gen_ids],
                    }
                )
                if nose_index is None and float(t[-1]) > 0 and float(t_new[-1]) <= 0:
                    nose_index, nose_kind = len(points) - 1, "limit_induced"
                    if stop_at_nose:
                        stop = "nose"
                        break
                zh, t = self.rescale(z_new, t_new)
                continue

            if nose_index is None and float(t[-1]) > 0 and float(t_new[-1]) <= 0:
                # The fold lies between the last point and this one: the corrected
                # point of largest λ among the bisection points is the nose.
                _, seen = self.locate(zh, t, h, lambda zs, ts: float(ts[-1]) <= 0)
                best = max(
                    (zs for zs, _ in seen),
                    key=lambda zs: self.voltage(zs)[1],
                    default=None,
                )
                nose_is_new = False
                if best is not None and self.voltage(best)[1] > max(
                    lam_new, points[-1].lam
                ):
                    points.append(self.point(best))
                elif lam_new >= points[-1].lam:
                    points.append(self.point(z_new))
                    nose_is_new = True
                nose_index, nose_kind = len(points) - 1, "saddle_node"
                if stop_at_nose:
                    stop = "nose"
                    break
                if not nose_is_new:
                    points.append(self.point(z_new))
                zh, t = self.rescale(z_new, t_new)
                h = _adapt(h, iters, step_min, step_max)
                continue

            if nose_index is None and lambda_cap is not None and lam_new >= lambda_cap:
                hit, _ = self.locate(
                    zh, t, h, lambda zs, ts: float(self.split(zs)[1]) >= lambda_cap
                )
                z_cap = hit[0] if hit is not None else z_new
                v_cap, ok = prob.solve(lambda_cap, self.voltage(z_cap)[0])
                if ok:
                    points.append(self.point(self.to_z(v_cap, lambda_cap)))
                else:
                    points.append(self.point(z_cap))
                stop = "lambda_max"
                break

            points.append(self.point(z_new))
            zh, t = self.rescale(z_new, t_new)
            if self.min_v_pu(v_new) < v_stop_pu:
                stop = "v_stop"
                break
            if nose_index is not None and lam_new < lambda_start:
                stop = "lambda_start"
                break
            h = _adapt(h, iters, step_min, step_max)
        return self.result(points, ids, direction, nose_index, nose_kind, stop, events)

    def result(
        self,
        points: list[_Point],
        ids: tuple[int, ...],
        direction: str,
        nose_index: Optional[int],
        nose_kind: Optional[str],
        stop: str,
        events: list[dict],
    ) -> PVCurve:
        prob = self.p
        full = prob.full_index
        with torch.no_grad():
            v_red = torch.stack([pt.v for pt in points])  # [K, N_red]
            v_full = prob.prolong(v_red)  # [K, N]
            v_pu = (v_full.abs() / prob.full_v_base).cpu().numpy().astype(np.float64)
        phase_codes = full.phase_codes.tolist()
        phases = tuple(_PHASE_VALUES[c] for c in phase_codes)
        node_ids = np.asarray(full.node_ids.tolist(), dtype=np.int64)
        sign = 1.0 if direction == "load" else -1.0
        crit_node = crit_phase = None
        lam_max = None
        if nose_index is not None:
            lam_max = points[nose_index].lam
            live = np.asarray(phase_codes) != 3
            row = int(np.flatnonzero(live)[np.argmin(v_pu[nose_index, live])])
            crit_node, crit_phase = int(node_ids[row]), phases[row]
        return PVCurve(
            lambdas=np.asarray([pt.lam for pt in points], dtype=np.float64),
            p_total_w=sign * np.asarray([pt.p for pt in points], dtype=np.float64),
            v_pu=v_pu,
            node_ids=node_ids,
            phases=phases,
            appliance_ids=ids,
            direction=direction,
            nose_index=nose_index,
            lambda_max=lam_max,
            nose_kind=nose_kind,
            critical_node=crit_node,
            critical_phase=crit_phase,
            converged=stop in _REGULAR_STOPS,
            stop_reason=stop,
            events=tuple(events),
        )


def _adapt(h: float, iters: int, step_min: float, step_max: float) -> float:
    """Next arc-length step from the corrector iteration count of the last one."""
    factor = min(2.0, max(0.5, _CORRECTOR_TARGET_ITER / max(iters, 1)))
    return min(step_max, max(step_min, h * factor))


__all__ = ["PVCurve", "QVCurve", "pv_curve", "qv_curve"]
