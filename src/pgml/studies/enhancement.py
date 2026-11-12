"""Enhancement sensitivities: which grid upgrade relieves the most stress.

:func:`enhancement_sensitivities` solves ONE power flow at a base operating point,
measures how close every element is to its limit with a smooth stress metric, and
differentiates that metric with ONE backward pass with respect to every candidate
enhancement at once: the series impedance and the thermal rating of every line and
transformer, every transformer's off-nominal tap, and new active and reactive power at
every bus. A finite-difference ranking costs one power flow per candidate; the gradient
costs about one more linear solve in total, whatever the number of candidates.

Stress metric
-------------
Each limit of :class:`~pgml.studies.HostingLimits` is turned into a utilisation ``u``
per element, ``0`` at the reference and ``1`` exactly at the limit:

- voltage band, per phase conductor (phase-to-neutral; phase-to-ground on a node
  without a neutral): ``u = (v - 1) / (v_max_pu - 1)`` above and
  ``u = (1 - v) / (1 - v_min_pu)`` below the nominal ``1`` pu;
- line loading, per rated line: the largest phase-conductor current over both
  terminals per ``line_loading * max_i_a``; neutral loading the same for the neutral
  conductor against ``neutral_loading * max_i_a``;
- transformer loading: the per-phase current times the terminal's nominal
  line-to-neutral voltage, against ``transformer_loading * s_rated_va`` per phase
  conductor, largest over both terminals. Two terminals that agree to the tie
  tolerance of the hosting search (``1e-7`` relative in complex128, ``100`` float32
  epsilons in complex64) count with their mean: at a nominal tap both terminals of a
  transformer without a core branch carry the same loading while their tap slopes
  differ, and the mean makes the tap derivative the central one instead of the slope
  of whichever terminal rounding ranks larger;
- voltage unbalance factor ``|V2| / |V1|`` per ``vuf_max`` and voltage THD per
  ``thd_max``, where those limits are set.

The quantities are the ones :func:`~pgml.studies.hosting_capacity` judges, computed by
the same code, so the two studies agree on what is near a limit. The stress of one
element is the squared excess of its utilisation over the onset ``c``
(``stress_onset``), rescaled so that it reaches ``1`` at the limit::

    s = max(0, (u - c) / (1 - c)) ** 2

and the total stress ``S`` is the sum over every element. A row of a per-phase
equivalent (``metadata.tags["phase_mode"] == "single_phase_equiv"``) stands for the
three phase conductors of the system, so its voltage and THD terms count three times;
``S`` is then the same number for a network in every representation that describes
its conductors (per-phase equivalent, three-phase, as-built). ``S`` is continuously
differentiable; it is zero (and so is every gradient) while every element stays below
``c`` of its limit, which is why the onset lies below ``1``: an element approaching its
limit already contributes a gradient. ``S = 1`` means, for instance, one element exactly
at its limit and every other one below the onset.

Sensitivities
-------------
Every enhancement is an elasticity ``dS / d ln k`` of an upgrade factor ``k`` at
``k = 1``; NEGATIVE means the upgrade relieves stress, and its magnitude is the stress
removed per unit of ``ln k`` to first order (``k = 1.1``, a 10 % upgrade, removes about
``0.095 * |elasticity|``):

- impedance: the series impedance becomes ``Z / k`` (resistance and inductance
  together, the effect of a larger cross-section or a lower-impedance transformer);
- rating: the ampacity or the rated power becomes ``k * rating``; it acts through the
  loading terms only and is always ``<= 0``.

The transformer tap is a plain derivative ``dS / dn`` of the off-nominal ratio
magnitude ``n`` (``tap.ratio_magnitude``). Bus injections are ``dS / dP`` per kW and
``dS / dQ`` per kvar of a new constant-power device connected wye over every phase
conductor of the bus with the power split equally, in LOAD convention: positive ``P``
is new demand, so new generation changes the stress by the negative of the reported
value. ``P`` and ``Q`` are three-phase SYSTEM powers in every representation of a
network: on a per-phase equivalent (``metadata.tags["phase_mode"] ==
"single_phase_equiv"``, which stores a third of every power) one kW of new power is a
third of a kW on the stored phase (:func:`pgml.representation.system_totals`), so the
same network reports the same sensitivity per kW whatever its representation.

The derivatives are local: they rank candidates at the base operating point rather
than size them. Re-solve a shortlist with the actual upgrade before deciding.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Union

import numpy as np
import torch
from torch import Tensor

from pgml.errors import ConvergenceError, InputError
from pgml.representation import system_totals
from pgml.scenarios import batch_from_values
from pgml.schemas.grid_schema import (
    Grid,
    Line,
    Load,
    LoadModel,
    Transformer,
    WindingConnection,
)
from pgml.simulation import SimulationConfig, SolvedState, simulate

from ._common import (
    BaseOperatingPoint,
    base_operating_values,
    check_study,
    live_phases,
    resolve_candidates,
    resolve_complex_dtype,
)
from .limits import HostingLimits, _LimitElements

#: Sides a base operating point can be the worst case for.
_DIRECTIONS = ("generation", "load")
#: Stress kinds, in report order (keys of ``stress_by_kind``).
_STRESS_KINDS = ("voltage", "line", "line_neutral", "transformer", "vuf", "thd")
# Per-element arrays of the result: (attribute, id attribute), in to_dict order.
_ARRAYS = (
    ("line_impedance_elasticity", "line_ids"),
    ("line_impedance_elasticity_per_km", "line_ids"),
    ("line_rating_elasticity", "line_ids"),
    ("transformer_impedance_elasticity", "transformer_ids"),
    ("transformer_rating_elasticity", "transformer_ids"),
    ("transformer_tap_gradient", "transformer_ids"),
    ("d_stress_d_p_per_kw", "node_ids"),
    ("d_stress_d_q_per_kvar", "node_ids"),
)


@dataclass(frozen=True, eq=False)
class EnhancementSensitivities:
    """Result of :func:`enhancement_sensitivities`: the stress and its gradient.

    Elasticities are ``dS / d ln k`` of an upgrade factor ``k`` (dimensionless, stress
    per unit log change); negative = the upgrade relieves stress. ``NaN`` marks a
    candidate the sensitivity does not apply to.

    Attributes
    ----------
    stress:
        Total stress ``S`` (dimensionless, ``>= 0``) at the base operating point.
    stress_by_kind:
        ``S`` split by limit kind: ``{"voltage", "line", "line_neutral",
        "transformer", "vuf", "thd"}`` -> float; ``0`` for a kind that is disabled or
        has no stressed element.
    stressed:
        Every element with a positive stress term, largest first: the element dict of
        :attr:`pgml.studies.HostingCapacity.binding` (``{"kind": "voltage", "limit",
        "node_id", "phase"}``, ``{"kind": "line" | "line_neutral" | "transformer",
        "branch_id"}``, ``{"kind": "vuf", "node_id"}``, ``{"kind": "thd", "node_id",
        "phase"}``) plus ``"value"`` (the checked quantity: pu, loading fraction, VUF
        or THD fraction), ``"utilisation"`` (``1`` = at the limit) and ``"stress"``
        (of one conductor: a voltage or THD entry of a per-phase equivalent counts
        three times in :attr:`stress`).
    line_ids:
        Every in-service :class:`~pgml.schemas.grid_schema.Line`, in grid order
        (the order of the ``line_*`` arrays ``[L]``).
    line_impedance_elasticity:
        ``[L]`` float64: series impedance ``Z / k``. ``NaN`` for a line whose impedance
        comes from a ``conductor_geometry`` (not scaled).
    line_impedance_elasticity_per_km:
        ``[L]`` float64, per km: the impedance elasticity divided by the line length,
        the benefit per unit of cable to replace.
    line_rating_elasticity:
        ``[L]`` float64: ampacity ``k * max_i_a`` (phase and neutral loading terms).
        ``NaN`` for a line without an ampacity or when both line loading limits are
        disabled.
    transformer_ids:
        Every in-service :class:`~pgml.schemas.grid_schema.Transformer`, in grid order
        (the order of the ``transformer_*`` arrays ``[T]``).
    transformer_impedance_elasticity:
        ``[T]`` float64: leakage impedance ``Z / k`` (positive sequence, and an
        explicit zero-sequence leakage with it).
    transformer_rating_elasticity:
        ``[T]`` float64: rated power ``k * s_rated_va``. ``NaN`` without a rating or
        with ``transformer_loading`` disabled.
    transformer_tap_gradient:
        ``[T]`` float64: ``dS / dn`` per unit of the off-nominal tap ratio magnitude.
    node_ids:
        Every bus with a phase conductor that carries no in-service source (the order
        of the injection arrays ``[n]``).
    d_stress_d_p_per_kw:
        ``[n]`` float64, 1/kW: ``dS / dP`` of new constant-power demand at the bus
        (load convention; new generation: the negative), per kW of three-phase system
        power in every grid representation.
    d_stress_d_q_per_kvar:
        ``[n]`` float64, 1/kvar: ``dS / dQ`` of new reactive demand (inductive
        positive) at the bus, per kvar of three-phase system power.
    direction:
        ``"generation"`` or ``"load"``: the side a ``"worst_case"`` base was built for.
    base_operating_point:
        ``"worst_case"``, ``"nameplate"`` or ``"custom"``.
    calculation:
        ``"power_flow"`` or ``"harmonic"``.
    stress_onset:
        The onset ``c`` of every stress term (fraction of the limit).
    limits:
        The :class:`~pgml.studies.HostingLimits` the stress is built from.
    iterations:
        Iterations of the fundamental power flow.
    solve_time_s:
        Wall time of the forward pass in s: assembly, power flow, limit quantities and
        stress.
    backward_time_s:
        Wall time of the single backward pass in s.
    """

    stress: float
    stress_by_kind: dict[str, float]
    stressed: tuple[dict, ...]
    line_ids: tuple[int, ...]
    line_impedance_elasticity: np.ndarray
    line_impedance_elasticity_per_km: np.ndarray
    line_rating_elasticity: np.ndarray
    transformer_ids: tuple[int, ...]
    transformer_impedance_elasticity: np.ndarray
    transformer_rating_elasticity: np.ndarray
    transformer_tap_gradient: np.ndarray
    node_ids: tuple[int, ...]
    d_stress_d_p_per_kw: np.ndarray
    d_stress_d_q_per_kvar: np.ndarray
    direction: str
    base_operating_point: str
    calculation: str
    stress_onset: float
    limits: HostingLimits = field(default_factory=HostingLimits)
    iterations: int = 0
    solve_time_s: float = math.nan
    backward_time_s: float = math.nan

    def ranking(
        self, sensitivity: str, k: Optional[int] = None, *, ascending: bool = True
    ) -> list[tuple[int, float]]:
        """``(id, value)`` pairs of one sensitivity array, sorted; ``NaN`` left out.

        ``ascending=True`` (default) puts the most negative value first: for an
        elasticity the upgrade that relieves the most stress, for
        ``d_stress_d_p_per_kw`` the bus where new demand relieves the most (where new
        generation hurts the most). ``ascending=False`` reverses it, e.g. the bus
        where new demand hurts the most. ``k`` keeps the first ``k`` entries.
        """
        ids_attr = dict(_ARRAYS).get(sensitivity)
        if ids_attr is None:
            raise InputError(
                f"sensitivity must be one of {[a for a, _ in _ARRAYS]}, "
                f"got {sensitivity!r}"
            )
        values = getattr(self, sensitivity)
        ids = getattr(self, ids_attr)
        keep = np.flatnonzero(~np.isnan(values))
        order = keep[np.argsort(values[keep], kind="stable")]
        if not ascending:
            order = order[::-1]
        pairs = [(int(ids[j]), float(values[j])) for j in order]
        return pairs if k is None else pairs[:k]

    def to_dict(self) -> dict:
        """JSON-ready dict: plain lists and ids, ``NaN`` as ``None``.

        Keys: ``study`` (``"enhancement_sensitivities"``), ``stress``,
        ``stress_by_kind``, ``stressed``, ``line_ids``, ``transformer_ids``,
        ``node_ids``, one list per sensitivity array (the attribute names),
        ``direction``, ``base_operating_point``, ``calculation``, ``stress_onset``,
        ``limits`` (:meth:`HostingLimits.to_dict`), ``iterations``, ``solve_time_s``,
        ``backward_time_s``.
        """
        out: dict[str, Any] = {
            "study": "enhancement_sensitivities",
            "stress": float(self.stress),
            "stress_by_kind": {k: float(v) for k, v in self.stress_by_kind.items()},
            "stressed": [dict(e) for e in self.stressed],
            "line_ids": [int(i) for i in self.line_ids],
            "transformer_ids": [int(i) for i in self.transformer_ids],
            "node_ids": [int(i) for i in self.node_ids],
        }
        for name, _ in _ARRAYS:
            out[name] = [_json_float(x) for x in getattr(self, name)]
        out.update(
            direction=self.direction,
            base_operating_point=self.base_operating_point,
            calculation=self.calculation,
            stress_onset=float(self.stress_onset),
            limits=self.limits.to_dict(),
            iterations=int(self.iterations),
            solve_time_s=_json_float(self.solve_time_s),
            backward_time_s=_json_float(self.backward_time_s),
        )
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnhancementSensitivities":
        """Rebuild a result from :meth:`to_dict` output.

        Raises :class:`~pgml.errors.InputError` unless ``data["study"]`` is
        ``"enhancement_sensitivities"``.
        """
        check_study(data, "enhancement_sensitivities")

        def _arr(values) -> np.ndarray:
            return np.array(
                [np.nan if x is None else float(x) for x in values], dtype=np.float64
            )

        def _time(x) -> float:
            return math.nan if x is None else float(x)

        return cls(
            stress=float(data["stress"]),
            stress_by_kind={k: float(v) for k, v in data["stress_by_kind"].items()},
            stressed=tuple(dict(e) for e in data["stressed"]),
            line_ids=tuple(int(i) for i in data["line_ids"]),
            transformer_ids=tuple(int(i) for i in data["transformer_ids"]),
            node_ids=tuple(int(i) for i in data["node_ids"]),
            **{name: _arr(data[name]) for name, _ in _ARRAYS},
            direction=str(data["direction"]),
            base_operating_point=str(data["base_operating_point"]),
            calculation=str(data["calculation"]),
            stress_onset=float(data["stress_onset"]),
            limits=HostingLimits.from_dict(data["limits"]),
            iterations=int(data["iterations"]),
            solve_time_s=_time(data["solve_time_s"]),
            backward_time_s=_time(data["backward_time_s"]),
        )


def _json_float(x) -> Optional[float]:
    """A float, or ``None`` for NaN (JSON has no NaN)."""
    x = float(x)
    return None if math.isnan(x) else x


def _stress(u: Tensor, onset: float) -> Tensor:
    """Squared excess of the utilisation ``u`` over the onset, ``1`` at ``u = 1``."""
    return torch.relu((u - onset) / (1.0 - onset)).pow(2)


def _check_inputs(
    limits: HostingLimits, config: SimulationConfig, direction: str, onset: float
) -> None:
    """Refuse options the stress metric or the solve cannot honour."""
    if direction not in _DIRECTIONS:
        raise InputError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
    if not (math.isfinite(onset) and 0.0 <= onset < 1.0):
        raise InputError(
            f"stress_onset must satisfy 0 <= stress_onset < 1, got {onset}"
        )
    if all(v is None for v in limits.to_dict().values()):
        raise InputError("limits disable every limit; the stress would be empty")
    if limits.v_max_pu is not None and limits.v_max_pu <= 1.0:
        raise InputError(
            f"the stress is measured from 1 pu: v_max_pu must exceed 1, got "
            f"{limits.v_max_pu}"
        )
    if limits.v_min_pu is not None and limits.v_min_pu >= 1.0:
        raise InputError(
            f"the stress is measured from 1 pu: v_min_pu must be below 1, got "
            f"{limits.v_min_pu}"
        )
    if config.operating_point is not None:
        raise InputError(
            "enhancement_sensitivities: config.operating_point must be unset; pass the "
            "base case as base_operating_point."
        )
    harmonic = config.calculation == "harmonic"
    if harmonic and 1 not in [int(o) for o in config.harmonic_orders]:
        raise InputError("a harmonic enhancement study must solve order 1")
    if limits.thd_max is not None and not harmonic:
        raise InputError(
            "HostingLimits.thd_max needs a harmonic calculation: pass a "
            "SimulationConfig(calculation='harmonic', harmonic_orders=[1, ...])."
        )


def _as_real(value, real: torch.dtype, device) -> Tensor:
    """A schema value (float, nested list or tensor) as a real tensor."""
    return torch.as_tensor(value, dtype=real, device=device)


@dataclass
class _Forward:
    """One forward pass: total stress, its split by kind, the terms and the state.

    ``terms[kind]`` lists ``(descriptors, value, utilisation, stress, weight)`` per
    limit of that kind: ``value``, ``utilisation`` and ``stress`` ``[E]`` float64
    tensors on the tape, ``weight`` the conductors each element stands for (``S`` sums
    ``weight * stress``).
    """

    total: Tensor
    by_kind: dict[str, Tensor]
    terms: dict[str, list]
    state: SolvedState

    def stressed(self) -> tuple[dict, ...]:
        """Every element with a positive stress term, largest first (host values)."""
        out = []
        for parts in self.terms.values():
            for descs, value, u, s, _ in parts:
                s_host = s.detach().cpu().numpy()
                value_host = value.detach().cpu().numpy()
                u_host = u.detach().cpu().numpy()
                for j in np.flatnonzero(s_host > 0.0):
                    out.append(
                        {
                            **descs[j],
                            "value": float(value_host[j]),
                            "utilisation": float(u_host[j]),
                            "stress": float(s_host[j]),
                        }
                    )
        out.sort(key=lambda e: -e["stress"])
        return tuple(out)


class _Problem:
    """The extended grid, the limit tables and the stress as a function of the leaves.

    The leaves, in :meth:`forward` order: ``ln_k_line`` ``[L]`` (log impedance
    upgrade of every in-service line), ``ln_k_trafo`` ``[T]`` (the same for every
    in-service transformer), ``tap`` ``[T]`` (off-nominal ratio magnitudes),
    ``ln_k_ampacity`` ``[L_rated]`` (log ampacity upgrade of every rated line, the
    order of ``elements.lines``), ``ln_k_s_rated`` ``[T_rated]`` (``elements.
    transformers``), ``p_kw`` and ``q_kvar`` ``[n]`` (new system power at every bus of
    ``node_ids``, three-phase kW / kvar; the probe setpoint is that over the
    representation's power factor). :meth:`initial_leaves` is the as-built grid at the base operating
    point.
    """

    def __init__(
        self,
        grid: Grid,
        *,
        limits: HostingLimits,
        base_label: str,
        base_values: dict,
        config: SimulationConfig,
        onset: float,
        dtype: torch.dtype,
        device: Optional[torch.device],
    ) -> None:
        self.limits = limits
        self.base_label = base_label
        self.base_values = base_values
        self.config = config
        self.onset = onset
        self.dtype = dtype
        self.device = device
        self.real = torch.empty((), dtype=dtype).real.dtype
        # Relative tie tolerance of hosting_capacity: the solve's 1e-8 pu tolerance,
        # or the resolution of single precision.
        self.tie_tol = max(1.0e-7, 100.0 * torch.finfo(self.real).eps)
        harmonic = config.calculation == "harmonic"
        self.orders = [int(o) for o in config.harmonic_orders] if harmonic else [1]
        self.two_pi_f0 = 2.0 * math.pi * float(grid.base_frequency_hz)
        # Stored watts per kW of system power: 1e3 / 3 on a per-phase equivalent.
        self.stored_w_per_kw = 1.0e3 / system_totals(grid).power_factor

        # One zero-power constant-power probe load per bus carries the new-power leaves.
        self.node_ids = resolve_candidates(grid, None)
        nodes = {n.id: n for n in grid.nodes}
        first_id = max((int(a.id) for a in grid.appliances), default=0) + 1
        self.probe_ids = [first_id + k for k in range(len(self.node_ids))]
        probes = [
            Load(
                id=pid,
                node=nid,
                phases=live_phases(nodes[nid].phases),
                connection=WindingConnection.WYE,
                load_model=LoadModel.CONST_POWER,
                p_nom_w=0.0,
                q_nom_var=0.0,
            )
            for pid, nid in zip(self.probe_ids, self.node_ids)
        ]
        self.grid = grid.model_copy(
            update={"appliances": list(grid.appliances) + probes}
        )
        self.elements = _LimitElements(self.grid)
        self.lines = [b for b in grid.branches if isinstance(b, Line) and b.in_service]
        self.trafos = [
            b for b in grid.branches if isinstance(b, Transformer) and b.in_service
        ]
        # A line whose impedance comes from its conductor geometry is not scaled.
        self.scaled_lines = np.array(
            [ln.conductor_geometry is None for ln in self.lines], dtype=bool
        )

    def initial_leaves(self) -> tuple[Tensor, ...]:
        """The leaves of the as-built grid at the base point, ``requires_grad`` set."""

        def _leaf(values) -> Tensor:
            t = torch.as_tensor(values, dtype=self.real, device=self.device)
            return t.requires_grad_(True)

        tap = [
            float(_as_real(t.tap.ratio_magnitude, self.real, None)) for t in self.trafos
        ]
        return (
            _leaf([0.0] * len(self.lines)),
            _leaf([0.0] * len(self.trafos)),
            _leaf(tap),
            _leaf([0.0] * len(self.elements.lines)),
            _leaf([0.0] * len(self.elements.transformers)),
            _leaf([0.0] * len(self.node_ids)),
            _leaf([0.0] * len(self.node_ids)),
        )

    def _overrides(self, ln_k_line: Tensor, ln_k_trafo: Tensor, tap: Tensor) -> dict:
        """``param_overrides`` scaling every series impedance and setting every tap."""
        real, device = self.real, self.device
        overrides: dict = {}
        for k, ln in enumerate(self.lines):
            if not self.scaled_lines[k]:
                continue
            down = torch.exp(-ln_k_line[k])
            for name in ("series_resistance_ohm_per_m", "series_inductance_h_per_m"):
                overrides[("line", ln.id, name)] = (
                    _as_real(getattr(ln, name), real, device) * down
                )
        for k, t in enumerate(self.trafos):
            down = torch.exp(-ln_k_trafo[k])
            overrides[("transformer", t.id, "series_resistance_ohm")] = (
                _as_real(t.series_resistance_ohm, real, device) * down
            )
            overrides[("transformer", t.id, "series_inductance_h")] = (
                _as_real(t.series_inductance_h, real, device) * down
            )
            if t.zero_sequence is not None:
                zs = t.zero_sequence
                overrides[("transformer", t.id, "zero_sequence_resistance_ohm")] = (
                    _as_real(zs.r0_ohm, real, device) * down
                )
                overrides[("transformer", t.id, "zero_sequence_inductance_h")] = (
                    _as_real(zs.x0_ohm, real, device) / self.two_pi_f0 * down
                )
            overrides[("transformer", t.id, "tap_magnitude")] = tap[k]
        return overrides

    def forward(
        self,
        ln_k_line: Tensor,
        ln_k_trafo: Tensor,
        tap: Tensor,
        ln_k_ampacity: Tensor,
        ln_k_s_rated: Tensor,
        p_kw: Tensor,
        q_kvar: Tensor,
    ) -> _Forward:
        """Solve at the leaves' values and build the stress on the autograd tape."""
        p_w = {**self.base_values["p_w"]}
        q_var = {**self.base_values["q_var"]}
        scale = self.stored_w_per_kw
        p_w.update({pid: scale * p_kw[k] for k, pid in enumerate(self.probe_ids)})
        q_var.update({pid: scale * q_kvar[k] for k, pid in enumerate(self.probe_ids)})
        batch = batch_from_values(
            self.grid,
            n_samples=1,
            p_w=p_w,
            q_var=q_var,
            p_per_phase_w=self.base_values["p_per_phase_w"] or None,
            q_per_phase_var=self.base_values["q_per_phase_var"] or None,
        )
        state = simulate(
            self.grid,
            self.config.model_copy(update={"operating_point": batch.operating_point}),
            dtype=self.dtype,
            device=self.device,
            param_overrides=self._overrides(ln_k_line, ln_k_trafo, tap),
            strict=False,
        )
        if not state.converged:
            raise ConvergenceError(
                f"enhancement_sensitivities: the {self.base_label!r} base operating "
                f"point does not converge ({state.iterations} iterations, residual "
                f"{state.residual:.3e}); there is no operating point to differentiate "
                "at. Use a lighter base_operating_point.",
                iterations=state.iterations,
                residual=state.residual,
            )
        v = state.v.reshape(1, len(self.orders), -1)
        lim, el, onset = self.limits, self.elements, self.onset
        quantities = el.evaluate(v, state, lim, h1=self.orders.index(1))
        dev = v.device

        terms: dict[str, list] = {kind: [] for kind in _STRESS_KINDS}

        def _add(
            kind: str, descs: list, value: Tensor, u: Tensor, weight: float = 1.0
        ) -> None:
            terms[kind].append((descs, value, u, _stress(u, onset), weight))

        # A per-phase-equivalent row stands for this many phase conductors.
        per_row = el.conductors_per_row

        if "v_pu" in quantities:
            v_pu = quantities["v_pu"][0]
            for key, bound, sign in (
                ("v_max_pu", lim.v_max_pu, 1.0),
                ("v_min_pu", lim.v_min_pu, -1.0),
            ):
                if bound is not None:
                    descs = [
                        {"kind": "voltage", "limit": key, "node_id": n, "phase": ph}
                        for n, ph in el.live_desc
                    ]
                    u = sign * (v_pu - 1.0) / abs(bound - 1.0)
                    _add("voltage", descs, v_pu, u, per_row)
        ampacity = torch.exp(ln_k_ampacity.to(device=dev, dtype=torch.float64))
        if "line_loading" in quantities:
            value = quantities["line_loading"][0]
            descs = [{"kind": "line", "branch_id": e[0]} for e in el.lines]
            _add("line", descs, value, value / lim.line_loading / ampacity)
        if "neutral_loading" in quantities:
            value = quantities["neutral_loading"][0]
            position = {e[0]: j for j, e in enumerate(el.lines)}
            pick = torch.tensor([position[e[0]] for e in el.neutral_lines], device=dev)
            descs = [
                {"kind": "line_neutral", "branch_id": e[0]} for e in el.neutral_lines
            ]
            _add(
                "line_neutral",
                descs,
                value,
                value / lim.neutral_loading / ampacity[pick],
            )
        if "transformer_loading" in quantities:
            # The larger terminal, as hosting judges it. At a nominal tap both terminals
            # of a transformer without magnetizing branch carry the same loading, but
            # their tap slopes differ (the from-side current is referred through the
            # ratio): terminals within the tie tolerance count as tied and the term
            # takes their mean, whose derivative is the central one, instead of the
            # side that rounding happens to rank larger.
            terminal = quantities["transformer_terminal_loading"][0]  # [T, 2]
            hard = terminal.amax(dim=-1)
            tied = (terminal[:, 0] - terminal[:, 1]).abs() <= self.tie_tol * hard
            value = torch.where(tied, terminal.mean(dim=-1), hard)
            s_rated = torch.exp(ln_k_s_rated.to(device=dev, dtype=torch.float64))
            descs = [
                {"kind": "transformer", "branch_id": e[0]} for e in el.transformers
            ]
            _add("transformer", descs, value, value / lim.transformer_loading / s_rated)
        if "vuf" in quantities:
            value = quantities["vuf"][0]
            descs = [{"kind": "vuf", "node_id": n} for n in el.abc_nodes]
            _add("vuf", descs, value, value / lim.vuf_max)
        if "thd" in quantities:
            value = quantities["thd"][0]
            descs = [
                {"kind": "thd", "node_id": n, "phase": ph} for n, ph in el.live_desc
            ]
            _add("thd", descs, value, value / lim.thd_max, per_row)

        zero = torch.zeros((), dtype=torch.float64, device=dev)
        by_kind = {
            kind: sum((part[4] * part[3].sum() for part in parts), zero)
            for kind, parts in terms.items()
        }
        total = sum(by_kind.values(), zero)
        return _Forward(total=total, by_kind=by_kind, terms=terms, state=state)


def enhancement_sensitivities(
    grid: Grid,
    *,
    limits: Optional[HostingLimits] = None,
    base_operating_point: BaseOperatingPoint = "nameplate",
    direction: Literal["load", "generation"] = "generation",
    stress_onset: float = 0.8,
    config: Optional[SimulationConfig] = None,
    dtype: Union[str, torch.dtype] = torch.complex128,
    device: Optional[Union[str, torch.device]] = None,
) -> EnhancementSensitivities:
    """Rank every candidate grid enhancement by its first-order relief of limit stress.

    One power flow at ``base_operating_point`` and one backward pass give the
    elasticity of a smooth stress metric with respect to the impedance and rating of
    every line and transformer, the tap of every transformer, and new power at every
    bus. See :mod:`pgml.studies.enhancement` for the metric and the sign conventions.

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.Grid` (``type_ref`` of every line and transformer
        expanded; the line catalog is still read for ampacities). Not modified.
    limits:
        :class:`~pgml.studies.HostingLimits` the stress is built from; ``None`` = its
        defaults (0.90-1.10 pu, 100 % line and transformer loading). A set voltage
        bound must lie on its side of 1 pu, the reference of the voltage utilisation.
    base_operating_point:
        The operating point to differentiate at, with the semantics of
        :func:`~pgml.studies.hosting_capacity`: ``"nameplate"`` (default) every device
        at nameplate; ``None`` or ``"worst_case"`` the worst case for ``direction``
        (for generation every load at zero, for load every generator at zero); a
        mapping ``{appliance_id: {"p_w", "q_var", "p_per_phase_w",
        "q_per_phase_var"}}`` (W, var) overrides the named devices on top of
        nameplate; its values are in the grid's stored units, like the appliance
        fields they replace (on a per-phase equivalent a third of the device's
        three-phase power).
    direction:
        ``"generation"`` (default) or ``"load"``: which worst case ``"worst_case"``
        builds. Not used by any other base.
    stress_onset:
        Fraction ``c`` of each limit at which its stress term starts,
        ``0 <= c < 1`` (default 0.8).
    config:
        :class:`~pgml.SimulationConfig` for slack model, symmetry, convergence control
        and, with ``calculation="harmonic"``, the solved orders (order 1 required;
        line and transformer currents are then RMS over the orders and ``thd_max`` can
        be used). ``None``: the fundamental power flow with the config defaults. Its
        ``operating_point`` must be unset.
    dtype, device:
        Complex working dtype (``torch.complex64`` / ``torch.complex128`` or their
        names) and device of the solve and the backward pass.

    Returns
    -------
    EnhancementSensitivities
        Host numpy arrays and floats.

    Raises
    ------
    ConvergenceError
        The base operating point does not converge: there is no operating point to
        differentiate at. Choose a lighter base (``"worst_case"``, or a mapping with
        realistic, e.g. diversified, powers).
    InputError
        An unknown ``direction`` / ``dtype`` / base, ``stress_onset`` outside
        ``[0, 1)``, every limit disabled, a voltage bound on the wrong side of 1 pu,
        ``thd_max`` without a harmonic calculation, or a set
        ``config.operating_point``.

    Notes
    -----
    Voltages are the phase-to-neutral voltages of the phase conductors and line
    loading compares each phase conductor's current with the line type's ``max_i_a``,
    the conventions of :func:`~pgml.studies.hosting_capacity`.

    Every sensitivity comes from one call of :func:`torch.autograd.grad` on all
    parameter tensors: the power flow is differentiated through the implicit function
    theorem at the converged state, so the backward pass costs one adjoint solve,
    independent of the number of candidates and of the iteration count. The new-power
    sensitivities are taken on zero-power constant-power loads added to a copy of the
    grid, one per bus, which leave the base operating point unchanged.

    Device and dtype: the solve, the stress and the backward pass run on ``device`` at
    ``dtype``; the limit quantities are formed in float64 from the solved voltages and
    currents. Only the results move to the host. complex128 (default) is the precision
    the gradient checks run in.

    Examples
    --------
    The five lines whose reinforcement relieves the most stress::

        result = enhancement_sensitivities(grid)
        for line_id, value in result.ranking("line_impedance_elasticity", 5):
            print(line_id, value)
    """
    dtype = resolve_complex_dtype(dtype)
    limits = HostingLimits() if limits is None else limits
    config = SimulationConfig(calculation="power_flow") if config is None else config
    onset = float(stress_onset)
    _check_inputs(limits, config, direction, onset)
    if isinstance(device, str):
        device = torch.device(device)
    real = torch.empty((), dtype=dtype).real.dtype
    base_label, base_values = base_operating_values(
        grid, direction, base_operating_point, real, device
    )
    problem = _Problem(
        grid,
        limits=limits,
        base_label=base_label,
        base_values=base_values,
        config=config,
        onset=onset,
        dtype=dtype,
        device=device,
    )
    leaves = problem.initial_leaves()
    sync = (
        torch.cuda.synchronize
        if (device is not None and device.type == "cuda")
        else None
    )
    t_start = time.perf_counter()
    fwd = problem.forward(*leaves)
    if sync is not None:
        sync()
    t_forward = time.perf_counter()
    if fwd.total.requires_grad:
        grads = torch.autograd.grad(fwd.total, leaves, allow_unused=True)
    else:
        grads = (None,) * len(leaves)
    if sync is not None:
        sync()
    t_backward = time.perf_counter()

    def _host(g: Optional[Tensor], leaf: Tensor) -> np.ndarray:
        if g is None:
            return np.zeros(leaf.shape[0], dtype=np.float64)
        return g.detach().to(torch.float64).cpu().numpy()

    g_line, g_trafo, g_tap, g_amp, g_rated, g_p, g_q = (
        _host(g, leaf) for g, leaf in zip(grads, leaves)
    )
    lines, trafos, elements = problem.lines, problem.trafos, problem.elements
    line_imp = np.where(problem.scaled_lines, g_line, np.nan)
    length_km = np.array(
        [float(_as_real(ln.length_m, torch.float64, None)) / 1.0e3 for ln in lines]
    )
    line_rating = np.full(len(lines), np.nan)
    if limits.line_loading is not None or limits.neutral_loading is not None:
        position = {ln.id: j for j, ln in enumerate(lines)}
        for j, entry in enumerate(elements.lines):
            line_rating[position[entry[0]]] = g_amp[j]
    trafo_rating = np.full(len(trafos), np.nan)
    if limits.transformer_loading is not None:
        position = {t.id: j for j, t in enumerate(trafos)}
        for j, (bid, _) in enumerate(elements.transformers):
            trafo_rating[position[bid]] = g_rated[j]

    return EnhancementSensitivities(
        stress=float(fwd.total.detach()),
        stress_by_kind={k: float(v.detach()) for k, v in fwd.by_kind.items()},
        stressed=fwd.stressed(),
        line_ids=tuple(int(ln.id) for ln in lines),
        line_impedance_elasticity=line_imp,
        line_impedance_elasticity_per_km=line_imp / length_km,
        line_rating_elasticity=line_rating,
        transformer_ids=tuple(int(t.id) for t in trafos),
        transformer_impedance_elasticity=g_trafo,
        transformer_rating_elasticity=trafo_rating,
        transformer_tap_gradient=g_tap,
        node_ids=tuple(problem.node_ids),
        d_stress_d_p_per_kw=g_p,
        d_stress_d_q_per_kvar=g_q,
        direction=direction,
        base_operating_point=base_label,
        calculation=config.calculation,
        stress_onset=onset,
        limits=limits,
        iterations=int(fwd.state.iterations),
        solve_time_s=t_forward - t_start,
        backward_time_s=t_backward - t_forward,
    )


__all__ = ["EnhancementSensitivities", "enhancement_sensitivities"]
