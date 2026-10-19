"""Hosting capacity: how much new generation or demand each bus can take.

For every candidate bus, :func:`hosting_capacity` finds the largest additional active power
a new device at that bus can inject (``direction="generation"``) or draw
(``direction="load"``)
while every limit of :class:`~pgml.studies.HostingLimits` holds, with every other bus
left at the base operating point. The search is a bisection on the device's power, run
for all candidates at once: each step solves ONE scenario batch in which scenario ``k``
carries candidate ``k``'s trial power at its own bus and nothing anywhere else.

Method
------
1. The grid is extended (on a copy) by one inert device per candidate bus, a
   :class:`~pgml.schemas.grid_schema.Generator` or :class:`~pgml.schemas.grid_schema.Load`
   with zero nameplate, WYE-connected on the requested phases. The base operating point
   (``base_operating_point``) and the trial powers reach the solver as one batched
   operating point through :func:`~pgml.scenarios.batch_from_values`; a scenario sets
   the power of its own candidate device only.
2. The base case (every candidate at zero) is solved first. A limit element (a node
   phase, a line, a transformer) that the base case already violates is reported in
   ``base_violations`` and, from then on, is binding only where a candidate makes its
   violation WORSE: its relative excess over the limit may not grow by more than the
   working-precision tolerance (``1e-7`` of the limit in complex128, ``100`` float32
   epsilons, about ``1.2e-5``, in complex64). A limit the base case satisfies stays a
   hard limit. So one overloaded line elsewhere in the grid does not block the study,
   while new generation at a bus that is already above the voltage band reports
   ``p_max_w = 0`` with that voltage as binding. ``base_case_feasible`` is ``False``
   whenever ``base_violations`` is not empty. A base case that does not converge has no
   operating point to compare against: every candidate then reports ``p_max_w = 0``
   with binding ``{"kind": "convergence"}``.
3. Bracketing. With ``p_max_w`` given, every candidate is evaluated at that cap; a
   candidate that keeps every limit there reports the cap with binding ``{"kind":
   "none"}``. Without a cap, the trial power starts at 1 kW and doubles, per candidate,
   until a limit is violated; the last trial is clipped to the 1 TW ceiling, so a
   candidate that keeps every limit up to it reports exactly ``p_cap_w`` with binding
   ``"none"``.
4. Bisection between the largest feasible and the smallest violating power, all open
   candidates in lockstep, until the bracket is narrower than the tolerance (by default
   0.1 % of the candidate's upper bracket). The reported ``p_max_w`` is the lower end, a
   power at which every limit was verified; the binding limit is the one exceeded most
   (relative to its limit, or for a limit violated in the base case relative to its base
   excess) at the upper end.

Each batched solve carries one scenario per open candidate. The open candidates are
solved in slices that fit the memory budget ``studies.hosting.batch_budget_mb`` (at least
one scenario per slice); a scenario depends only on its own candidate, so the result does
not depend on the slicing.

A scenario whose power flow does not converge counts as a violation (binding
``{"kind": "convergence"}``): beyond the loadability nose there is no operating point.
The bisection assumes the feasible powers form one interval starting at zero, which holds
when every limit tightens monotonically with the added power. With the default
``"worst_case"`` base that is the usual situation: no load offsets new generation (and no
generation offsets new demand), so voltage rise or drop and the line and transformer
currents grow with the added power from the start. With a ``"nameplate"`` or custom base
it need not hold: new generation first REDUCES the current of a line that feeds local
load (reverse flow overtakes it only beyond that load), and a single-phase device can
first reduce an existing unbalance (VUF) or harmonic distortion (THD). Such a limit can
be violated at a small power, satisfied over a middle range and violated again; the
doubling bracket can step over that structure, and the search reports the end of the
first feasible interval it brackets, which need not be the first one.

The fundamental power flow is the default calculation. A harmonic
:class:`~pgml.SimulationConfig` solves every scenario at its harmonic orders (order 1
must be among them), which is what the ``thd_max`` limit needs; the line and
transformer currents are then RMS values over the solved orders, the voltage band and
unbalance stay fundamental quantities.

Differentiability
-----------------
None of the outputs is differentiable: a bisection decides on feasibility, a boolean of
the solved state. The study runs without building an autograd graph and returns host
numpy arrays. It does not modify the grid passed in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from pgml import defaults
from pgml.assembly import base_voltage_per_row, node_phase_index
from pgml.errors import InputError
from pgml.geometry.sequence import fortescue_matrix
from pgml.scenarios import batch_from_values
from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    Line,
    Load,
    Phase,
    Source,
    Transformer,
    WindingConnection,
)
from pgml.simulation import SimulationConfig, SolvedState
from pgml.solver import prepare_power_flow, solve_harmonic_flow, solve_power_flow

from ._common import check_study, resolve_complex_dtype
from .limits import HostingLimits, branch_ratings

#: Sides a hosting-capacity study can add: new generation or new demand.
_DIRECTIONS = ("generation", "load")
#: First trial power of the doubling bracket when no search cap is given (W).
_P_START_W = 1.0e3
#: Largest power the doubling bracket tries; a candidate feasible there reports it (W).
_P_CEILING_W = 1.0e12
#: Default bisection tolerance, relative to the candidate's upper bracket.
_REL_TOL = 1.0e-3
# Keys of HostingCapacity.margins, in report order.
_MARGIN_KEYS = (
    "v_min_pu",
    "v_max_pu",
    "line_loading",
    "transformer_loading",
    "vuf",
    "thd",
    "neutral_loading",
)

BaseOperatingPoint = Union[None, str, Mapping[int, Mapping[str, Any]]]


@dataclass(frozen=True, eq=False)
class HostingCapacity:
    """Result of :func:`hosting_capacity`: one entry per candidate bus.

    Attributes
    ----------
    node_ids:
        Candidate bus ids, in the order of every per-candidate array ``[n]``.
    direction:
        ``"generation"`` or ``"load"``.
    p_max_w:
        ``[n]`` float64, W: the largest added active power (total over the device's
        phases) at which every limit was verified (a limit already violated in the base
        case: not made worse, see :mod:`pgml.studies.hosting`). ``0`` when even a small
        addition violates a limit, and when the base case does not converge. Exactly
        :attr:`p_cap_w` for a candidate that reached no limit.
    binding:
        ``n`` dicts naming what stops the candidate at ``p_max_w``:
        ``{"kind": "voltage", "limit": "v_max_pu" | "v_min_pu", "node_id", "phase"}``,
        ``{"kind": "line", "branch_id"}``, ``{"kind": "line_neutral", "branch_id"}``,
        ``{"kind": "transformer", "branch_id"}``, ``{"kind": "vuf", "node_id"}``,
        ``{"kind": "thd", "node_id", "phase"}``, ``{"kind": "convergence"}`` (no
        operating point beyond), or ``{"kind": "none"}`` when no limit was reached up to
        :attr:`p_cap_w`.
    margins:
        ``{key: [n] float64}`` for the keys ``v_min_pu``, ``v_max_pu``,
        ``line_loading``, ``transformer_loading``, ``vuf``, ``thd`` and
        ``neutral_loading``, evaluated at ``p_max_w``: how far the tightest element of
        each limit is from it, in that limit's unit (pu for the voltage band, a fraction
        for loading, unbalance and THD); positive is inside the limit. An element
        violated in the base case counts too, so the margin of its limit is negative.
        ``NaN`` where the limit is disabled or has no element to check (no rated line,
        no line with a neutral conductor, no three-phase node).
    iterations:
        ``[n]`` int64: power flows solved for that candidate (base case excluded); one
        batched solve serves every open candidate.
    p_cap_w:
        Search cap in W: ``p_max_w`` as given, else the doubling ceiling (1 TW).
    tol_w:
        ``[n]`` float64, W: bisection tolerance per candidate; ``p_max_w`` is within it
        of the limit. ``NaN`` where no bisection ran (base case not converged).
    base_case_feasible:
        Whether the base operating point (no new device) converges and keeps every
        limit, i.e. ``base_violations`` is empty.
    base_operating_point:
        ``"worst_case"``, ``"nameplate"`` or ``"custom"``: which base the search used.
    phases:
        ``"balanced"`` or a phase letter: how the new device was connected.
    power_factor:
        Power factor of the new device (sign convention of :func:`hosting_capacity`).
    limits:
        The :class:`~pgml.studies.HostingLimits` that were checked.
    base_violations:
        The limit elements the base case violates, in row / branch order: the element
        dict of ``binding`` plus ``"excess"``, its relative excess over the limit
        (``0.05`` = 5 % beyond it). ``({"kind": "convergence"},)`` when the base case
        does not converge.
    """

    node_ids: tuple[int, ...]
    direction: str
    p_max_w: np.ndarray
    binding: tuple[dict, ...]
    margins: dict[str, np.ndarray]
    iterations: np.ndarray
    p_cap_w: float
    tol_w: np.ndarray
    base_case_feasible: bool
    base_operating_point: str
    phases: str
    power_factor: float
    limits: HostingLimits = field(default_factory=HostingLimits)
    base_violations: tuple[dict, ...] = ()

    def result_for(self, node_id: int) -> dict:
        """The JSON-ready entry of one candidate (``p_max_w``, ``binding``, margins)."""
        try:
            k = self.node_ids.index(int(node_id))
        except ValueError:
            raise InputError(
                f"node {node_id} is not a candidate of this study"
            ) from None
        return {
            "node_id": int(node_id),
            "p_max_w": float(self.p_max_w[k]),
            "binding": dict(self.binding[k]),
            "margins": {key: _json_float(self.margins[key][k]) for key in _MARGIN_KEYS},
        }

    def to_dict(self) -> dict:
        """JSON-ready dict: plain lists and ids, units in the key names, ``NaN`` as ``None``.

        Keys: ``study`` (``"hosting_capacity"``), ``direction``, ``node_ids``,
        ``p_max_w``, ``binding``, ``margins`` (``{key: list}``), ``iterations``,
        ``p_cap_w``, ``tol_w``, ``base_case_feasible``, ``base_violations``,
        ``base_operating_point``, ``phases``, ``power_factor``, ``limits``
        (:meth:`HostingLimits.to_dict`).
        """
        return {
            "study": "hosting_capacity",
            "direction": self.direction,
            "node_ids": [int(i) for i in self.node_ids],
            "p_max_w": [float(x) for x in self.p_max_w],
            "binding": [dict(b) for b in self.binding],
            "margins": {
                key: [_json_float(x) for x in self.margins[key]] for key in _MARGIN_KEYS
            },
            "iterations": [int(x) for x in self.iterations],
            "p_cap_w": float(self.p_cap_w),
            "tol_w": [_json_float(x) for x in self.tol_w],
            "base_case_feasible": bool(self.base_case_feasible),
            "base_violations": [dict(v) for v in self.base_violations],
            "base_operating_point": self.base_operating_point,
            "phases": self.phases,
            "power_factor": float(self.power_factor),
            "limits": self.limits.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HostingCapacity":
        """Rebuild a result from :meth:`to_dict` output.

        Raises :class:`~pgml.errors.InputError` unless ``data["study"]`` is
        ``"hosting_capacity"``.
        """
        check_study(data, "hosting_capacity")

        def _arr(values) -> np.ndarray:
            return np.array(
                [np.nan if x is None else float(x) for x in values], dtype=np.float64
            )

        return cls(
            node_ids=tuple(int(i) for i in data["node_ids"]),
            direction=str(data["direction"]),
            p_max_w=_arr(data["p_max_w"]),
            binding=tuple(dict(b) for b in data["binding"]),
            margins={key: _arr(data["margins"][key]) for key in _MARGIN_KEYS},
            iterations=np.array(data["iterations"], dtype=np.int64),
            p_cap_w=float(data["p_cap_w"]),
            tol_w=_arr(data["tol_w"]),
            base_case_feasible=bool(data["base_case_feasible"]),
            base_operating_point=str(data["base_operating_point"]),
            phases=str(data["phases"]),
            power_factor=float(data["power_factor"]),
            limits=HostingLimits.from_dict(data["limits"]),
            base_violations=tuple(dict(v) for v in data["base_violations"]),
        )


def _json_float(x) -> Optional[float]:
    """A float, or ``None`` for NaN (JSON has no NaN)."""
    x = float(x)
    return None if math.isnan(x) else x


@dataclass
class _Evaluation:
    """Verdict of one batched solve: per scenario feasibility, binding limit, margins.

    ``converged`` is the solve's own verdict (finite limit quantities included);
    ``excess`` the raw relative excess of every limit element, ``(value - limit) /
    limit`` (``> 0`` violates), before the base-case comparison.
    """

    feasible: np.ndarray  # [B] bool
    binding: list  # [B] dict | None (None where feasible)
    margins: dict  # key -> [B] float64
    converged: np.ndarray  # [B] bool
    excess: np.ndarray  # [B, E] float64

    @staticmethod
    def concat(parts: list["_Evaluation"]) -> "_Evaluation":
        """The evaluations of consecutive scenario slices as one."""
        if len(parts) == 1:
            return parts[0]
        return _Evaluation(
            feasible=np.concatenate([e.feasible for e in parts]),
            binding=[b for e in parts for b in e.binding],
            margins={
                key: np.concatenate([e.margins[key] for e in parts])
                for key in _MARGIN_KEYS
            },
            converged=np.concatenate([e.converged for e in parts]),
            excess=np.concatenate([e.excess for e in parts], axis=0),
        )


def _live_phases(phases: Sequence[Phase]) -> tuple[Phase, ...]:
    """The phase conductors of a terminal (its neutral excluded)."""
    return tuple(p for p in phases if p != Phase.N)


def _candidate_phases(grid: Grid, node_ids: Sequence[int], phases) -> list:
    """Phases each candidate device connects to, validated against its node."""
    nodes = {n.id: n for n in grid.nodes}
    if phases == "balanced":
        chosen = []
        for nid in node_ids:
            live = _live_phases(nodes[nid].phases)
            if not live:
                raise InputError(f"candidate node {nid} has no phase conductor")
            chosen.append(live)
        return chosen
    try:
        phase = Phase(str(getattr(phases, "value", phases)).lower())
    except ValueError:
        raise InputError(
            f"phases must be 'balanced' or one of 'a', 'b', 'c', got {phases!r}"
        ) from None
    if phase == Phase.N:
        raise InputError("a candidate device cannot connect to the neutral alone")
    missing = [nid for nid in node_ids if phase not in nodes[nid].phases]
    if missing:
        raise InputError(f"candidate nodes {missing} carry no phase {phase.value!r}")
    return [(phase,) for _ in node_ids]


def _resolve_candidates(grid: Grid, candidates) -> list[int]:
    """Candidate bus ids: as given (validated), else every bus without a source."""
    node_ids = [n.id for n in grid.nodes]
    if candidates is None:
        source_nodes = {
            a.node for a in grid.appliances if isinstance(a, Source) and a.in_service
        }
        return [
            n.id
            for n in grid.nodes
            if n.id not in source_nodes and _live_phases(n.phases)
        ]
    chosen = [int(c) for c in candidates]
    unknown = sorted(set(chosen) - set(node_ids))
    if unknown:
        raise InputError(f"candidates name unknown nodes {unknown}")
    if len(set(chosen)) != len(chosen):
        raise InputError("candidates must not repeat a node")
    if not chosen:
        raise InputError("candidates must name at least one node")
    return chosen


def _base_values(
    grid: Grid, direction: str, base_operating_point: BaseOperatingPoint, real, device
) -> tuple[str, dict]:
    """The base operating point as ``batch_from_values`` mappings, plus its label.

    ``None`` / ``"worst_case"``: for generation every in-service load draws nothing and
    the existing generation runs at nameplate; for load every in-service generator
    injects nothing and the loads draw nameplate. Storage keeps its nameplate setpoint in
    both. ``"nameplate"``: every device at nameplate. A mapping: ``{appliance_id:
    {"p_w", "q_var", "p_per_phase_w", "q_per_phase_var"}}`` overrides on nameplate.
    """
    values: dict = {"p_w": {}, "q_var": {}, "p_per_phase_w": {}, "q_per_phase_var": {}}

    def _t(x) -> Tensor:
        return torch.as_tensor(x, dtype=real, device=device)

    if base_operating_point is None or base_operating_point == "worst_case":
        zeroed = Load if direction == "generation" else Generator
        for a in grid.appliances:
            if a.in_service and isinstance(a, zeroed):
                values["p_w"][a.id] = _t(0.0)
                values["q_var"][a.id] = _t(0.0)
        return "worst_case", values
    if base_operating_point == "nameplate":
        return "nameplate", values
    if isinstance(base_operating_point, str):
        raise InputError(
            "base_operating_point must be None, 'worst_case', 'nameplate' or a mapping "
            f"of appliance ids, got {base_operating_point!r}"
        )
    for cid, entry in base_operating_point.items():
        unknown = set(entry) - set(values)
        if unknown:
            raise InputError(
                f"base_operating_point[{cid}] has unknown keys {sorted(unknown)}; use "
                f"{sorted(values)}"
            )
        for key, value in entry.items():
            if key in ("p_per_phase_w", "q_per_phase_var"):
                values[key][int(cid)] = [_t(x) for x in value]
            else:
                values[key][int(cid)] = _t(value)
    return "custom", values


class _Study:
    """The extended grid, the solve settings and the static limit tables of one study."""

    def __init__(
        self,
        grid: Grid,
        node_ids: list[int],
        cand_phases: list,
        *,
        direction: str,
        limits: HostingLimits,
        base_values: dict,
        power_factor: float,
        harmonic_emission: Optional[Mapping[int, tuple]],
        config: SimulationConfig,
        dtype: torch.dtype,
        device: Optional[torch.device],
    ) -> None:
        self.limits = limits
        self.config = config
        self.dtype = dtype
        self.device = device
        self.real = torch.empty((), dtype=dtype).real.dtype
        # Relative excesses closer than this to the largest one are ties: the solve's
        # 1e-8 pu tolerance, or the resolution of single precision.
        self.tie_tol = max(1.0e-7, 100.0 * torch.finfo(self.real).eps)
        self.base_values = base_values
        self.harmonic = config.calculation == "harmonic"
        self.orders = [int(o) for o in config.harmonic_orders] if self.harmonic else [1]
        self.h1 = self.orders.index(1)
        self.emission = (
            {int(o): tuple(pair) for o, pair in harmonic_emission.items()}
            if harmonic_emission
            else None
        )

        # Reactive power per watt of the candidate device (power-factor convention of
        # hosting_capacity: positive = same direction as the active power).
        pf = float(power_factor)
        self.q_per_w = math.copysign(math.tan(math.acos(abs(pf))), pf)

        # The extended grid: one inert device per candidate, fresh ids above the grid's.
        first_id = max((int(a.id) for a in grid.appliances), default=0) + 1
        self.device_ids = [first_id + k for k in range(len(node_ids))]
        cls = Generator if direction == "generation" else Load
        added = [
            cls(
                id=did,
                node=nid,
                phases=ph,
                connection=WindingConnection.WYE,
                p_nom_w=0.0,
                q_nom_var=0.0,
            )
            for did, nid, ph in zip(self.device_ids, node_ids, cand_phases)
        ]
        self.grid = grid.model_copy(
            update={"appliances": list(grid.appliances) + added}
        )
        self.system = (
            None
            if self.harmonic
            else prepare_power_flow(
                self.grid, slack=config.slack, dtype=dtype, device=device
            )
        )
        self._build_tables()
        # Base-case comparison (set by set_base): elements the base case violates are
        # binding only where their relative excess grows beyond the tie tolerance.
        self.base_excess: Optional[np.ndarray] = None
        self.base_violated: Optional[np.ndarray] = None
        # Scenarios per batched solve: the dense Newton step holds a real [2N, 2N]
        # Jacobian, its factorization and the directional derivatives it is built from.
        real_size = torch.empty((), dtype=self.real).element_size()
        per_scenario = 3 * (2 * self.n_rows) ** 2 * real_size
        budget = float(defaults.get("studies.hosting.batch_budget_mb")) * 1024.0**2
        self.batch_size = max(1, int(budget // per_scenario))

    # -- static tables ------------------------------------------------------ #
    def _build_tables(self) -> None:
        """Row and branch index tables of every element a limit is checked on."""
        grid = self.grid
        index = node_phase_index(grid)
        self.n_rows = index.size
        base = base_voltage_per_row(grid, dtype=torch.float64)

        live, neutral = [], []
        for node in grid.nodes:
            n_row = index.row(node.id, Phase.N) if Phase.N in node.phases else -1
            for ph in _live_phases(node.phases):
                live.append(index.row(node.id, ph))
                neutral.append(n_row)
        self.live_rows = torch.tensor(live, dtype=torch.long)
        self.neutral_rows = torch.tensor(neutral, dtype=torch.long)
        self.live_base = base[self.live_rows].numpy()
        self.live_desc = [(index.node_id_of(r), index.phase_of(r).value) for r in live]

        abc = [
            [index.row(n.id, p) for p in (Phase.A, Phase.B, Phase.C)]
            for n in grid.nodes
            if all(p in n.phases for p in (Phase.A, Phase.B, Phase.C))
        ]
        self.abc_rows = torch.tensor(abc, dtype=torch.long).reshape(-1, 3)
        self.abc_nodes = [
            n.id
            for n in grid.nodes
            if all(p in n.phases for p in (Phase.A, Phase.B, Phase.C))
        ]

        ratings = branch_ratings(grid)
        # Per rated line: (id, ampacity, live-conductor masks at the from / to terminal).
        self.lines = []
        for b in grid.branches:
            if not (isinstance(b, Line) and b.in_service and ratings[b.id].i_max_a):
                continue
            live_from = torch.tensor([p != Phase.N for p in b.from_phases])
            live_to = torch.tensor([p != Phase.N for p in b.to_phases])
            self.lines.append((b.id, ratings[b.id].i_max_a, live_from, live_to))
        self.neutral_lines = [
            entry for entry in self.lines if not (entry[2].all() and entry[3].all())
        ]
        self.transformers = []
        for b in grid.branches:
            if not (isinstance(b, Transformer) and b.in_service):
                continue
            s_rated = ratings[b.id].s_rated_va
            if not s_rated:
                continue
            terminals = []
            for node_id, phases in (
                (b.from_node, b.from_phases),
                (b.to_node, b.to_phases),
            ):
                mask = torch.tensor([p != Phase.N for p in phases])
                n_live = int(mask.sum())
                u_ln = float(base[index.row(node_id, _live_phases(phases)[0])])
                # loading = |I| * U_ln * n_live / S_rated per phase conductor
                terminals.append((mask, u_ln * n_live / s_rated))
            self.transformers.append((b.id, terminals))

    # -- one batched evaluation -------------------------------------------- #
    def set_base(self, base: _Evaluation) -> None:
        """Compare later evaluations against this converged base case (scenario 0)."""
        self.base_excess = base.excess[0]
        self.base_violated = base.excess[0] > 0.0

    def evaluate(self, trial_w: np.ndarray, cand: np.ndarray) -> _Evaluation:
        """Scenario ``k`` = candidate ``cand[k]`` at ``trial_w[k]`` W, in budget slices."""
        n = int(trial_w.shape[0])
        step = self.batch_size
        return _Evaluation.concat(
            [
                self._evaluate_slice(trial_w[i : i + step], cand[i : i + step])
                for i in range(0, n, step)
            ]
        )

    def _evaluate_slice(self, trial_w: np.ndarray, cand: np.ndarray) -> _Evaluation:
        """Solve one batch, scenario ``k`` = candidate ``cand[k]`` at ``trial_w[k]`` W."""
        b = int(trial_w.shape[0])
        p = torch.as_tensor(trial_w, dtype=self.real, device=self.device)
        # [B, B]: column k is candidate cand[k]'s power in every scenario of the batch
        p_diag = torch.diag(p)
        q_diag = p_diag * self.q_per_w
        ids = [self.device_ids[int(c)] for c in cand]
        p_w = {**self.base_values["p_w"]}
        q_var = {**self.base_values["q_var"]}
        p_w.update({did: p_diag[:, k] for k, did in enumerate(ids)})
        q_var.update({did: q_diag[:, k] for k, did in enumerate(ids)})
        injection = (
            {did: self.emission for did in ids}
            if (self.emission and self.harmonic)
            else None
        )
        batch = batch_from_values(
            self.grid,
            n_samples=b,
            p_w=p_w,
            q_var=q_var,
            p_per_phase_w=self.base_values["p_per_phase_w"] or None,
            q_per_phase_var=self.base_values["q_per_phase_var"] or None,
            harmonic_injection=injection,
        )
        op = batch.operating_point
        cfg = self.config
        solve_kwargs = dict(
            slack=cfg.slack,
            operating_point=op,
            tol=cfg.tol,
            tol_update_pu=cfg.tol_update_pu,
            s_base_va=cfg.s_base_va,
            enforce_q_limits=cfg.enforce_q_limits,
            max_iter=cfg.max_iter,
            symmetry=cfg.symmetry,
            criticality="never",
            dtype=self.dtype,
            device=self.device,
        )
        if self.harmonic:
            res = solve_harmonic_flow(
                self.grid,
                self.orders,
                harmonic_injection=batch.harmonic_injection or None,
                load_shunt=cfg.load_shunt,
                load_shunt_basis=cfg.load_shunt_basis,
                **solve_kwargs,
            )
            pf, v, freqs = res.pf, res.v, res.frequencies_hz
        else:
            res = solve_power_flow(self.grid, system=self.system, **solve_kwargs)
            pf, v = res, res.v.unsqueeze(-2)
            freqs = torch.tensor(
                [float(self.grid.base_frequency_hz)], dtype=self.real, device=v.device
            )
        h = len(self.orders)
        v = v.reshape(b, h, self.n_rows)
        mask = res.converged_mask
        ok = (
            mask.reshape(-1).cpu().numpy().astype(bool)
            if mask is not None
            else np.full(b, bool(res.converged))
        )

        state = SolvedState(
            grid=self.grid,
            config=cfg,
            v=v,
            frequencies_hz=freqs,
            index=res.index,
            converged=bool(res.converged),
            iterations=int(pf.iterations),
            residual=float(pf.residual),
            dtype=self.dtype,
            device=self.device,
            fusion=res.fusion,
            harmonic_injection=batch.harmonic_injection or None,
            resolved_operating_point=pf.resolved_operating_point(op),
        )
        return self._judge(v, state, ok)

    def _judge(self, v: Tensor, state: SolvedState, ok: np.ndarray) -> _Evaluation:
        """Relative excess of every limit element, the verdict and the margins."""
        lim = self.limits
        b = v.shape[0]
        # Each part is [B, E_k]: (value - limit) / limit per element, > 0 violates.
        parts: list[np.ndarray] = []
        descs: list[dict] = []
        margins = {key: np.full(b, np.nan) for key in _MARGIN_KEYS}

        # Phase-to-neutral voltage of every phase conductor (phase-to-ground without N).
        v_live = v[..., self.live_rows.to(v.device)]
        has_n = (self.neutral_rows >= 0).to(v.device)
        v_n = v[..., self.neutral_rows.clamp_min(0).to(v.device)]
        v_pn = v_live - torch.where(has_n, v_n, torch.zeros_like(v_n))  # [B, H, R]
        mag = v_pn.abs().to(torch.float64).cpu().numpy()
        v_pu = mag[:, self.h1, :] / self.live_base  # [B, R]
        if lim.v_max_pu is not None:
            parts.append((v_pu - lim.v_max_pu) / lim.v_max_pu)
            descs += [
                {"kind": "voltage", "limit": "v_max_pu", "node_id": n, "phase": ph}
                for n, ph in self.live_desc
            ]
            margins["v_max_pu"] = lim.v_max_pu - v_pu.max(axis=1)
        if lim.v_min_pu is not None:
            parts.append((lim.v_min_pu - v_pu) / lim.v_min_pu)
            descs += [
                {"kind": "voltage", "limit": "v_min_pu", "node_id": n, "phase": ph}
                for n, ph in self.live_desc
            ]
            margins["v_min_pu"] = v_pu.min(axis=1) - lim.v_min_pu

        if lim.vuf_max is not None and self.abc_nodes:
            v1 = v[:, self.h1, :][:, self.abc_rows.to(v.device)]  # [B, K, 3]
            a_inv = torch.linalg.inv(fortescue_matrix(v1.dtype, v1.device))
            seq = torch.einsum("ij,bkj->bki", a_inv, v1)  # [B, K, (0, +, -)]
            vuf = (
                (seq[..., 2].abs() / seq[..., 1].abs()).to(torch.float64).cpu().numpy()
            )
            parts.append((vuf - lim.vuf_max) / lim.vuf_max)
            descs += [{"kind": "vuf", "node_id": n} for n in self.abc_nodes]
            margins["vuf"] = lim.vuf_max - vuf.max(axis=1)

        if lim.thd_max is not None:
            sq = mag**2
            thd = (
                np.sqrt(np.clip(sq.sum(axis=1) - sq[:, self.h1, :], 0.0, None))
                / mag[:, self.h1, :]
            )
            parts.append((thd - lim.thd_max) / lim.thd_max)
            descs += [
                {"kind": "thd", "node_id": n, "phase": ph} for n, ph in self.live_desc
            ]
            margins["thd"] = lim.thd_max - thd.max(axis=1)

        need_lines = lim.line_loading is not None and self.lines
        need_neutral = lim.neutral_loading is not None and self.neutral_lines
        need_trafos = lim.transformer_loading is not None and self.transformers
        if need_lines or need_neutral or need_trafos:
            currents = {bc.branch_id: bc for bc in state.branch_currents()}

            def _rms(i: Tensor) -> Tensor:  # [B, H, P] -> [B, P]
                return i.abs().pow(2).sum(dim=-2).sqrt()

            def _line_loading(entries, live: bool) -> np.ndarray:
                """Largest conductor loading [B, L]: phase (``live``) or neutral ones."""
                cols = []
                for bid, i_max, m_from, m_to in entries:
                    bc = currents[bid]
                    sel_from = (m_from if live else ~m_from).to(v.device)
                    sel_to = (m_to if live else ~m_to).to(v.device)
                    i_cond = torch.cat(
                        [_rms(bc.i_from)[..., sel_from], _rms(bc.i_to)[..., sel_to]],
                        dim=-1,
                    )
                    cols.append(i_cond.amax(dim=-1) / i_max)
                out = torch.stack(cols, dim=-1).reshape(b, -1)
                return out.to(torch.float64).cpu().numpy()

            if need_lines:
                loading = _line_loading(self.lines, live=True)
                parts.append((loading - lim.line_loading) / lim.line_loading)
                descs += [{"kind": "line", "branch_id": e[0]} for e in self.lines]
                margins["line_loading"] = lim.line_loading - loading.max(axis=1)
            if need_neutral:
                loading = _line_loading(self.neutral_lines, live=False)
                lim_n = lim.neutral_loading
                parts.append((loading - lim_n) / lim_n)
                descs += [
                    {"kind": "line_neutral", "branch_id": e[0]}
                    for e in self.neutral_lines
                ]
                margins["neutral_loading"] = lim_n - loading.max(axis=1)
            if need_trafos:
                cols = []
                for bid, ((m_from, k_from), (m_to, k_to)) in self.transformers:
                    bc = currents[bid]
                    i_from = _rms(bc.i_from)[..., m_from.to(v.device)] * k_from
                    i_to = _rms(bc.i_to)[..., m_to.to(v.device)] * k_to
                    cols.append(torch.cat([i_from, i_to], dim=-1).amax(dim=-1))
                loading = torch.stack(cols, dim=-1).reshape(b, -1)
                loading = loading.to(torch.float64).cpu().numpy()
                lim_t = lim.transformer_loading
                parts.append((loading - lim_t) / lim_t)
                descs += [
                    {"kind": "transformer", "branch_id": bid}
                    for bid, _ in self.transformers
                ]
                margins["transformer_loading"] = lim_t - loading.max(axis=1)

        raw = np.concatenate(parts, axis=1) if parts else np.zeros((b, 0), np.float64)
        self.descs = descs
        finite = np.isfinite(raw).all(axis=1)
        ok = ok & finite
        excess = raw
        if self.base_violated is not None:
            # An element the base case violates binds only where the candidate makes its
            # violation worse than the base by more than the working precision.
            grown = raw - self.base_excess - self.tie_tol
            excess = np.where(self.base_violated, grown, raw)
        violated = (excess > 0.0).any(axis=1)
        feasible = ok & ~violated
        # Binding element: the largest relative excess; elements within the working
        # precision of it count as tied and the first in row / branch order wins, so a
        # tie (e.g. the equal phases of a balanced bus) resolves the same on every device.
        if excess.shape[1]:
            safe = np.where(np.isfinite(excess), excess, -np.inf)
            top = safe.max(axis=1, keepdims=True)
            pick = np.argmax(safe >= top - self.tie_tol, axis=1)
        else:
            pick = np.zeros(b, dtype=np.int64)
        binding = [
            None
            if feasible[k]
            else {"kind": "convergence"}
            if not ok[k]
            else dict(descs[int(pick[k])])
            for k in range(b)
        ]
        for key in _MARGIN_KEYS:
            margins[key] = np.where(ok, margins[key], np.nan)
        return _Evaluation(
            feasible=feasible,
            binding=binding,
            margins=margins,
            converged=ok,
            excess=raw,
        )


def hosting_capacity(
    grid: Grid,
    candidates: Optional[Sequence[int]] = None,
    *,
    direction: Literal["load", "generation"] = "generation",
    limits: Optional[HostingLimits] = None,
    base_operating_point: BaseOperatingPoint = None,
    phases: Union[str, Phase] = "balanced",
    power_factor: float = 1.0,
    p_max_w: Optional[float] = None,
    tol_w: Optional[float] = None,
    harmonic_emission: Optional[Mapping[int, tuple[float, float]]] = None,
    config: Optional[SimulationConfig] = None,
    dtype: Union[str, torch.dtype] = torch.complex128,
    device: Optional[Union[str, torch.device]] = None,
) -> HostingCapacity:
    """Largest new generation or demand each candidate bus can take within the limits.

    See the module documentation of :mod:`pgml.studies.hosting` for the method (batched
    lockstep bisection, one scenario per candidate).

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.Grid` (``type_ref`` of every line and transformer
        expanded; the line catalog is still read for ampacities). Not modified.
    candidates:
        Bus ids to place the new device at. ``None``: every bus with a phase conductor
        that carries no in-service :class:`~pgml.schemas.grid_schema.Source`.
    direction:
        ``"generation"`` (default; a new injecting device, e.g. PV) or ``"load"`` (new
        demand, e.g. an EV charger or a heat pump).
    limits:
        :class:`~pgml.studies.HostingLimits`; ``None`` = its defaults (0.90-1.10 pu,
        100 % line and transformer loading).
    base_operating_point:
        What every other device does. ``None`` or ``"worst_case"`` (default): the worst
        case for ``direction`` -- for generation every in-service load draws nothing and
        existing generators run at nameplate, for load every in-service generator injects
        nothing and the loads draw nameplate; storage keeps its nameplate setpoint.
        ``"nameplate"``: every device at nameplate. A mapping ``{appliance_id: {"p_w",
        "q_var", "p_per_phase_w", "q_per_phase_var"}}`` (W, var; any subset) overrides
        the named devices on top of nameplate.
    phases:
        ``"balanced"`` (default): the device connects WYE to every phase conductor of
        its bus and the power splits equally. ``"a"``, ``"b"`` or ``"c"``: a
        single-phase device on that phase (to neutral where the bus has one, else to
        ground); every candidate must carry the phase.
    power_factor:
        Power factor of the new device, ``0 < |power_factor| <= 1``. Positive: the
        reactive power flows with the active power (a generator injects ``Q``,
        over-excited; a load draws ``Q``, inductive). Negative: against it (a generator
        absorbs ``Q``, under-excited, the usual voltage-rise countermeasure; a load is
        capacitive). ``Q = sign(pf) * P * tan(arccos|pf|)``.
    p_max_w:
        Search cap in W (total over the device's phases). ``None``: no cap, the bracket
        doubles from 1 kW until a limit is violated, up to 1 TW (``p_cap_w``).
    tol_w:
        Bisection tolerance in W. ``None``: 0.1 % of each candidate's upper bracket.
    harmonic_emission:
        Harmonic current spectrum of the new device, ``{order: (magnitude_pu,
        phase_deg)}`` in per unit of its own fundamental current (the format of
        :func:`~pgml.scenarios.batch_from_values`). Harmonic calculation only; ``None``:
        a linear device.
    config:
        :class:`~pgml.SimulationConfig` for slack model, symmetry, convergence control
        and, with ``calculation="harmonic"``, the solved orders (order 1 required) and
        the harmonic device shunt. ``None``: the fundamental power flow with the config
        defaults. Its ``operating_point`` must be unset: the base case is
        ``base_operating_point``.
    dtype, device:
        Complex working dtype (``torch.complex64`` / ``torch.complex128`` or their
        names) and device of every solve.

    Returns
    -------
    HostingCapacity
        Host numpy arrays, one entry per candidate in ``candidates`` order.

    Notes
    -----
    Device and dtype: every solve runs on ``device`` at ``dtype`` (CPU or CUDA,
    complex64 or complex128); the trial powers are real tensors of the matching
    precision on that device. Only the feasibility verdicts and margins move to the
    host, once per batched solve. At complex64 the voltage resolves to about 1e-6 pu,
    which moves ``p_max_w`` by less than the default tolerance on the tested grids.

    Memory: a batched solve carries one scenario per open candidate; the candidates are
    solved in slices that fit ``studies.hosting.batch_budget_mb`` (dense Newton step,
    ``3 * (2N)^2`` real numbers per scenario for ``N`` node-phase rows).

    Raises
    ------
    InputError
        Unknown or repeated candidate, an unknown ``direction`` / ``phases`` /
        ``dtype`` value, an invalid power factor, cap or tolerance, ``thd_max`` or
        ``harmonic_emission`` without a harmonic calculation, or a set
        ``config.operating_point``.

    Examples
    --------
    PV hosting capacity under a 3 % voltage-rise limit, at unity power factor::

        result = hosting_capacity(grid, limits=HostingLimits(v_max_pu=1.03))
        for node_id, p, why in zip(result.node_ids, result.p_max_w, result.binding):
            print(node_id, p / 1e3, "kW", why["kind"])
    """
    if direction not in _DIRECTIONS:
        raise InputError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
    dtype = resolve_complex_dtype(dtype)
    limits = HostingLimits() if limits is None else limits
    config = SimulationConfig(calculation="power_flow") if config is None else config
    if config.operating_point is not None:
        raise InputError(
            "hosting_capacity: config.operating_point must be unset; pass the base case "
            "as base_operating_point."
        )
    if config.calculation == "harmonic" and 1 not in [
        int(o) for o in config.harmonic_orders
    ]:
        raise InputError("a harmonic hosting-capacity study must solve order 1")
    if limits.thd_max is not None and config.calculation != "harmonic":
        raise InputError(
            "HostingLimits.thd_max needs a harmonic calculation: pass a "
            "SimulationConfig(calculation='harmonic', harmonic_orders=[1, ...])."
        )
    if harmonic_emission and config.calculation != "harmonic":
        raise InputError("harmonic_emission needs a harmonic calculation")
    pf = float(power_factor)
    if not (0.0 < abs(pf) <= 1.0):
        raise InputError(f"power_factor must satisfy 0 < |pf| <= 1, got {pf}")
    if p_max_w is not None and not (math.isfinite(p_max_w) and p_max_w > 0.0):
        raise InputError(f"p_max_w must be a positive finite power, got {p_max_w}")
    if tol_w is not None and not (math.isfinite(tol_w) and tol_w > 0.0):
        raise InputError(f"tol_w must be a positive finite power, got {tol_w}")
    if isinstance(device, str):
        device = torch.device(device)

    node_ids = _resolve_candidates(grid, candidates)
    cand_phases = _candidate_phases(grid, node_ids, phases)
    phase_label = "balanced" if phases == "balanced" else cand_phases[0][0].value
    real = torch.empty((), dtype=dtype).real.dtype
    base_label, base_values = _base_values(
        grid, direction, base_operating_point, real, device
    )

    with torch.no_grad():
        study = _Study(
            grid,
            node_ids,
            cand_phases,
            direction=direction,
            limits=limits,
            base_values=base_values,
            power_factor=pf,
            harmonic_emission=harmonic_emission,
            config=config,
            dtype=dtype,
            device=device,
        )
        n = len(node_ids)
        base = study.evaluate(np.zeros(1), np.zeros(1, dtype=np.int64))
        cap = float(p_max_w) if p_max_w is not None else _P_CEILING_W
        iterations = np.zeros(n, dtype=np.int64)
        if not base.converged[0]:
            no_solution = {"kind": "convergence"}
            return HostingCapacity(
                node_ids=tuple(node_ids),
                direction=direction,
                p_max_w=np.zeros(n),
                binding=tuple(dict(no_solution) for _ in range(n)),
                margins={k: np.full(n, np.nan) for k in _MARGIN_KEYS},
                iterations=iterations,
                p_cap_w=cap,
                tol_w=np.full(n, np.nan),
                base_case_feasible=False,
                base_operating_point=base_label,
                phases=phase_label,
                power_factor=pf,
                limits=limits,
                base_violations=(dict(no_solution),),
            )
        base_violations = tuple(
            {**study.descs[j], "excess": float(base.excess[0, j])}
            for j in np.flatnonzero(base.excess[0] > 0.0)
        )
        study.set_base(base)

        lo = np.zeros(n)
        hi = np.full(n, np.nan)
        lo_margins = {k: np.full(n, base.margins[k][0]) for k in _MARGIN_KEYS}
        hi_binding: list = [None] * n

        def _record(ev: _Evaluation, idx: np.ndarray, trial: np.ndarray) -> None:
            iterations[idx] += 1
            ok, bad = idx[ev.feasible], idx[~ev.feasible]
            lo[ok] = trial[ev.feasible]
            for key in _MARGIN_KEYS:
                lo_margins[key][ok] = ev.margins[key][ev.feasible]
            hi[bad] = trial[~ev.feasible]
            for j, k in enumerate(np.flatnonzero(~ev.feasible)):
                hi_binding[int(bad[j])] = ev.binding[int(k)]

        everyone = np.arange(n)
        if p_max_w is not None:
            trial = np.full(n, cap)
            _record(study.evaluate(trial, everyone), everyone, trial)
        else:
            # Double per candidate; the last trial is the cap itself, so a candidate that
            # never violates reports exactly the cap.
            trial_all = np.full(n, _P_START_W)
            active = everyone
            while active.size:
                trial = trial_all[active]
                ev = study.evaluate(trial, active)
                _record(ev, active, trial)
                grown = active[ev.feasible & (trial < cap)]
                trial_all[grown] = np.minimum(2.0 * trial_all[grown], cap)
                active = grown

        bracketed = ~np.isnan(hi)
        if tol_w is not None:
            tol = np.full(n, float(tol_w))
        else:
            tol = _REL_TOL * np.where(bracketed, hi, cap)
        while True:
            active = np.flatnonzero(bracketed & (hi - lo > tol))
            if not active.size:
                break
            mid = 0.5 * (lo[active] + hi[active])
            _record(study.evaluate(mid, active), active, mid)

    binding = tuple(
        dict(hi_binding[k]) if bracketed[k] else {"kind": "none"} for k in range(n)
    )
    return HostingCapacity(
        node_ids=tuple(node_ids),
        direction=direction,
        p_max_w=lo,
        binding=binding,
        margins=lo_margins,
        iterations=iterations,
        p_cap_w=cap,
        tol_w=tol,
        base_case_feasible=not base_violations,
        base_operating_point=base_label,
        phases=phase_label,
        power_factor=pf,
        limits=limits,
        base_violations=base_violations,
    )


__all__ = ["HostingCapacity", "hosting_capacity"]
