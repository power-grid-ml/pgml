"""Read and migrate a grid's representation: per-phase equivalent vs three-phase totals.

A :class:`~pgml.schemas.grid_schema.Grid` stores powers, ratings and voltages in the
representation it was built in, named by ``GridMetadata.tags["phase_mode"]``
(:data:`PHASE_MODE_TAG`):

- no tag -- the grid is read exactly as the schema defines it: values as stored,
  ``Node.u_rated_v`` line-to-line on a node with three or more phase rows (a neutral row
  counts) and line-to-neutral otherwise, appliance powers and ratings totals over their
  connected phases. A one-phase node is a real single-phase circuit.
- ``"single_phase_equiv"`` -- one phase of a balanced three-phase grid. Every node is
  one-phase and stores the line-to-neutral nominal ``U_LL/sqrt(3)``; loads, generators,
  storage, inverter ratings and transformer ``s_rated_va`` carry one third of the
  three-phase totals; transformer rated voltages are line-to-neutral too (their ratio is
  unchanged); per-phase impedances, admittances and ampacities are unchanged. The default
  output of the pandapower, power-grid-model and OpenDSS converters.
- ``"three_phase"`` -- an explicit phase-domain build (``(A, B, C)`` or source-native
  phases). Values as for an untagged grid.
- ``"as_built"`` -- the unbalanced phase-domain network as a data source records it
  (source-native phases, single-phase and two-phase branches kept), written by a loader
  that does not reduce the network to an equivalent. Values as for an untagged grid.
- ``"legacy_ll"`` -- a single-phase equivalent written before the per-phase equivalent
  existed: its one-phase nodes carry the line-to-line voltage and its appliances the
  three-phase totals, so a branch current is sqrt(3) too large. Such files are tagged
  with :func:`mark_legacy` (or migrated with :func:`to_per_phase_equivalent`); untagged,
  they would be read as single-phase circuits.

Every consumer that compares grids across representations goes through
:func:`system_totals` (system voltages and totals from any representation), and
:func:`to_per_phase_equivalent` migrates a ``"legacy_ll"`` grid. The conversion rule
itself is written once, in :mod:`pgml.convert._common` (``equivalent_voltage``,
``equivalent_power``, ``equivalent_control``, ``equivalent_voltage_regulation``); this
module reuses it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pgml.errors import InputError
from pgml.schemas.grid_schema import Grid, Line, Node, Phase

__all__ = [
    "PHASE_MODES",
    "PHASE_MODE_TAG",
    "SystemTotals",
    "mark_legacy",
    "phase_mode_of",
    "positive_sequence_resistance_ohm_per_m",
    "system_totals",
    "to_per_phase_equivalent",
]

_logger = logging.getLogger(__name__)

#: ``GridMetadata.tags`` key naming a grid's representation.
PHASE_MODE_TAG = "phase_mode"
#: The representations a grid may name under :data:`PHASE_MODE_TAG`: the converters'
#: ``PhaseMode`` values, the phase-domain ``"as_built"`` of dataset loaders and
#: ``"legacy_ll"``, the line-to-line single-phase equivalent of older converters.
PHASE_MODES: tuple[str, ...] = (
    "single_phase_equiv",
    "three_phase",
    "as_built",
    "legacy_ll",
)
# The tag value of a legacy line-to-line single-phase equivalent (``mark_legacy``).
_LEGACY_LL = "legacy_ll"

_SQRT3 = math.sqrt(3.0)
_untagged_logged = False


def _to_float(value: Any) -> float:
    """Plain ``float`` from a number or 0-d array-like (grid float/tensor duality)."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "reshape") and hasattr(value, "item"):
        return float(value.reshape(()).item())
    return float(value)


def phase_mode_of(grid: Grid) -> Optional[str]:
    """The representation ``grid`` names in ``metadata.tags["phase_mode"]``.

    Parameters
    ----------
    grid:
        Any grid.

    Returns
    -------
    str or None
        One of :data:`PHASE_MODES`, or ``None`` for an untagged grid (read by the schema
        rule).

    Raises
    ------
    InputError
        When the tag holds a value outside :data:`PHASE_MODES`. Every error of this
        module is a :class:`~pgml.errors.InputError`, which subclasses
        :class:`ValueError`, so ``except ValueError`` still catches it.
    """
    tags = getattr(getattr(grid, "metadata", None), "tags", None) or {}
    mode = tags.get(PHASE_MODE_TAG)
    if mode is None:
        return None
    if mode not in PHASE_MODES:
        raise InputError(
            f"grid {getattr(grid.metadata, 'name', None)!r} has "
            f"tags[{PHASE_MODE_TAG!r}] = {mode!r}; expected one of {PHASE_MODES}."
        )
    return mode


@dataclass(frozen=True)
class SystemTotals:
    """Converts one grid's stored values to three-phase system quantities.

    Built by :func:`system_totals`; holds no reference to the grid. Results are plain
    ``float`` (tensor-valued fields are read out, off the differentiable path).

    Attributes
    ----------
    phase_mode:
        The grid's representation (:data:`PHASE_MODES`), ``None`` for an untagged grid.
    power_factor:
        Multiplier from a stored appliance power (W, var) or transformer rating (VA) to
        the three-phase total: ``3`` for ``"single_phase_equiv"``, else ``1``.
    """

    phase_mode: Optional[str]
    power_factor: float

    @property
    def legacy(self) -> bool:
        """Whether the grid is a line-to-line single-phase equivalent (``"legacy_ll"``)."""
        return self.phase_mode == _LEGACY_LL

    def node_voltage_ll_v(self, node: Node) -> float:
        """Line-to-line nominal voltage of ``node`` [V].

        ``Node.u_rated_v`` on a node with three or more phase rows; ``sqrt(3)`` times it
        on a node with fewer (stored line-to-neutral, the schema rule), except on a node
        with two phase conductors that sets ``Node.u_rated_ln_v``: there ``u_rated_v`` is
        already the phase-to-phase voltage (two phases of a three-phase system, or the
        X1-X2 voltage of a split phase) and is returned as stored. A ``"legacy_ll"``
        grid's node is read as stored at any phase count.
        """
        u = _to_float(node.u_rated_v)
        phases = tuple(node.phases)
        if self.legacy or len(phases) >= 3:
            return u
        conductors = sum(1 for p in phases if p != Phase.N)
        if conductors == 2 and getattr(node, "u_rated_ln_v", None) is not None:
            return u
        return u * _SQRT3

    def transformer_voltage_ll_v(self, value: Any) -> float:
        """Line-to-line value [V] of a stored transformer rated voltage.

        ``sqrt(3)`` times the stored value for ``"single_phase_equiv"`` (per-phase
        nameplates), the stored value otherwise (line-to-line nameplates).
        """
        u = _to_float(value)
        return u * _SQRT3 if self.phase_mode == "single_phase_equiv" else u

    def total(self, value: Any) -> float:
        """Three-phase total [W, var or VA] of a stored appliance power or rating.

        ``None`` reads as ``0.0``.
        """
        if value is None:
            return 0.0
        return _to_float(value) * self.power_factor


def system_totals(grid: Grid) -> SystemTotals:
    """The :class:`SystemTotals` reader for ``grid``.

    Parameters
    ----------
    grid:
        Any grid; its representation is read from ``metadata.tags["phase_mode"]``.

    Returns
    -------
    SystemTotals
        ``power_factor = 3`` for ``"single_phase_equiv"``, ``1`` for every other
        representation and for untagged grids. The first untagged grid with a node of
        fewer than three phase rows logs one INFO line on logger ``pgml.representation``
        (it is read as a single-phase circuit; a file written by an older converter needs
        :func:`mark_legacy`).

    Raises
    ------
    InputError
        When the tag holds an unknown representation.
    """
    global _untagged_logged
    mode = phase_mode_of(grid)
    if mode is None and not _untagged_logged:
        if any(len(tuple(n.phases)) < 3 for n in grid.nodes):
            _untagged_logged = True
            _logger.info(
                "grid %r carries no %r tag: its one-phase nodes are read as "
                "single-phase circuits with a line-to-neutral rating (schema rule; "
                "a file from an older converter needs mark_legacy; logged once).",
                getattr(grid.metadata, "name", None),
                PHASE_MODE_TAG,
            )
    return SystemTotals(
        phase_mode=mode, power_factor=3.0 if mode == "single_phase_equiv" else 1.0
    )


def _matrix(value: Any) -> list[list[float]]:
    """A per-phase matrix (nested sequence or array-like) as nested plain floats."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [[float(x) for x in row] for row in value]


def positive_sequence_resistance_ohm_per_m(line: Line) -> float:
    """Per-conductor (positive-sequence) series resistance of ``line`` [Ohm/m].

    Reduces the phase matrix ``series_resistance_ohm_per_m`` over the line's live
    (non-neutral) rows ``P``:

    - ``|P| >= 2``: mean self minus mean mutual resistance over ``P``. For a symmetric
      three-phase matrix this is ``Re Z1``; for a matrix built from sequence data it
      returns ``R1`` exactly.
    - ``|P| = 1`` with a neutral row: the phase self term minus the phase-neutral mutual
      term.
    - otherwise: the first diagonal entry.

    Subtracting the mutual term removes an earth-return resistance shared by self and
    mutual terms (Carson matrices), so a single-phase equivalent (``[[R1]]``), an explicit
    three-phase build and a phase-domain line with neutral give the same value for the
    same cable.

    Returns
    -------
    float
        The resistance in Ohm/m; ``0.0`` when the line defers its matrix (``None``) to a
        line type or conductor geometry.
    """
    raw = getattr(line, "series_resistance_ohm_per_m", None)
    if raw is None:
        return 0.0
    r = _matrix(raw)
    if not r:
        return 0.0
    phases = tuple(line.from_phases)
    if len(phases) != len(r):
        return r[0][0]
    live = [i for i, p in enumerate(phases) if p != Phase.N]
    neutral = [i for i, p in enumerate(phases) if p == Phase.N]
    if len(live) >= 2:
        self_mean = sum(r[i][i] for i in live) / len(live)
        mutual = [r[i][j] for i in live for j in live if i != j]
        return self_mean - sum(mutual) / len(mutual)
    if len(live) == 1 and neutral:
        p, n = live[0], neutral[0]
        return r[p][p] - r[p][n]
    return r[0][0]


# =============================================================================
# Scaling between the system and the per-phase representation
# =============================================================================
_POWER_FIELDS = (
    "p_nom_w",
    "q_nom_var",
    "p_nom_per_phase_w",
    "q_nom_per_phase_var",
    "energy_capacity_wh",
    "p_rated_w",
)


def _map(value: Any, fn: Callable[[Any], Any]) -> Any:
    """``fn`` over a float, a tensor or the entries of a tuple/list; ``None`` stays."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return type(value)(_map(v, fn) for v in value)
    return fn(value)


def _update(model: Any, names: tuple[str, ...], fn: Callable[[Any], Any]) -> Any:
    update = {
        n: _map(getattr(model, n), fn)
        for n in names
        if getattr(model, n, None) is not None
    }
    return model.model_copy(update=update) if update else model


def _rescale_grid(
    grid: Grid, volt: Callable[[Any], Any], power: Callable[[Any], Any]
) -> Grid:
    """Copy of ``grid`` with every system voltage / power field mapped by ``volt`` /
    ``power``. Impedances, admittances, ampacities and per-unit fields are untouched."""
    from pgml.convert._common import _CONTROL_POWER_FIELDS

    nodes = [n.model_copy(update={"u_rated_v": volt(n.u_rated_v)}) for n in grid.nodes]
    branches = []
    for br in grid.branches:
        if getattr(br, "component", None) == "transformer":
            br = _update(br, ("u_rated_from_v", "u_rated_to_v"), volt)
            br = _update(br, ("s_rated_va",), power)
        branches.append(br)
    appliances = []
    for ap in grid.appliances:
        if getattr(ap, "u_ref_v", None) is not None:
            ap = _update(ap, ("u_ref_v",), volt)
        ap = _update(ap, _POWER_FIELDS, power)
        control = getattr(ap, "control", None)
        if control is not None:
            ap = ap.model_copy(
                update={"control": _update(control, _CONTROL_POWER_FIELDS, power)}
            )
        regulation = getattr(ap, "voltage_regulation", None)
        if regulation is not None:
            ap = ap.model_copy(
                update={
                    "voltage_regulation": _update(
                        regulation, ("q_min_var", "q_max_var"), power
                    )
                }
            )
        appliances.append(ap)
    transformers = {
        k: _update(
            _update(t, ("u_rated_from_v", "u_rated_to_v"), volt), ("s_rated_va",), power
        )
        for k, t in grid.types.transformers.items()
    }
    types = grid.types.model_copy(update={"transformers": transformers})
    return grid.model_copy(
        update={
            "nodes": nodes,
            "branches": branches,
            "appliances": appliances,
            "types": types,
        }
    )


def _is_one_phase(grid: Grid) -> bool:
    return bool(grid.nodes) and all(len(tuple(n.phases)) == 1 for n in grid.nodes)


def _represented_phases(totals: SystemTotals, node: Node, phases: Any) -> int:
    """Live conductors a terminal with ``phases`` on ``node`` stands for.

    Three on a node with fewer than three phase rows of a ``"single_phase_equiv"`` or
    ``"legacy_ll"`` grid (the one phase represents a balanced three-phase terminal),
    otherwise the non-neutral entries of ``phases`` (at least one).
    """
    if len(tuple(node.phases)) < 3 and totals.phase_mode in (
        "single_phase_equiv",
        _LEGACY_LL,
    ):
        return 3
    return max(1, sum(1 for p in phases if p != Phase.N))


def _with_tag(grid: Grid, mode: str) -> Grid:
    tags = dict(grid.metadata.tags)
    tags[PHASE_MODE_TAG] = mode
    return grid.model_copy(
        update={"metadata": grid.metadata.model_copy(update={"tags": tags})}
    )


def mark_legacy(grid: Grid) -> Grid:
    """Tag a grid written by an older converter as a line-to-line equivalent.

    Converters and loaders that predate the per-phase equivalent wrote single-phase
    equivalents with the line-to-line voltage on their one-phase nodes and three-phase
    totals on their appliances, without a ``phase_mode`` tag. Read untagged, such a grid
    is taken for a network of real single-phase circuits. This returns a copy tagged
    ``"legacy_ll"``, which :func:`system_totals` reads the old way and
    :func:`to_per_phase_equivalent` migrates. No value is changed.

    Only apply it to grids KNOWN to come from such a converter: an untagged grid built by
    hand with line-to-neutral one-phase nodes is already read correctly.

    Parameters
    ----------
    grid:
        An untagged grid whose nodes all have exactly one phase. Not modified.

    Returns
    -------
    Grid
        The tagged copy; ``grid`` itself, unchanged, when it is already ``"legacy_ll"``.

    Raises
    ------
    InputError
        When the grid carries another ``phase_mode`` tag, has a node with more than one
        phase, or has no nodes.
    """
    mode = phase_mode_of(grid)
    if mode == _LEGACY_LL:
        return grid
    if mode is not None:
        raise InputError(
            f"grid is tagged {PHASE_MODE_TAG}={mode!r}; only an untagged grid is "
            f"marked {_LEGACY_LL!r}."
        )
    if not _is_one_phase(grid):
        raise InputError(
            f"only a grid whose nodes are all one-phase can be marked {_LEGACY_LL!r}."
        )
    return _with_tag(grid, _LEGACY_LL)


def to_per_phase_equivalent(grid: Grid, *, assume_legacy: bool = False) -> Grid:
    """Migrate a line-to-line single-phase equivalent to the per-phase equivalent.

    A ``"legacy_ll"`` grid (:func:`mark_legacy`) carries the line-to-line voltage on its
    one-phase nodes and three-phase totals on its appliances. This returns a copy in the
    representation the converters now write (:class:`pgml.convert.PhaseMode`
    ``SINGLE_PHASE_EQUIV``): node, source and transformer rated voltages divided by
    sqrt(3); load, generator, storage, inverter control, PV reactive-limit and
    transformer ``s_rated_va`` powers divided by 3; per-phase impedances, admittances,
    ampacities, per-unit fields and spectra unchanged; tag ``"single_phase_equiv"`` set.
    Branch currents of the result are the conductor currents. The rule is the converters'
    own (``pgml.convert._common.equivalent_*``).

    Parameters
    ----------
    grid:
        A ``"legacy_ll"`` grid, or an untagged grid together with
        ``assume_legacy=True``; every node must have exactly one phase. Not modified.
    assume_legacy:
        Read an UNTAGGED grid as ``"legacy_ll"``. Without it an untagged grid is refused:
        by the schema its one-phase nodes are already line-to-neutral, and migrating a
        grid that really holds line-to-neutral values would scale it twice.

    Returns
    -------
    Grid
        The migrated copy; ``grid`` itself, unchanged, when it is already tagged
        ``"single_phase_equiv"``.

    Raises
    ------
    InputError
        When the grid carries another ``phase_mode`` tag, is untagged without
        ``assume_legacy``, has a node with more than one phase, or has no nodes.
    """
    from pgml.convert._common import PhaseMode, equivalent_power, equivalent_voltage

    mode = phase_mode_of(grid)
    if mode == "single_phase_equiv":
        return grid
    if mode is None and not assume_legacy:
        raise InputError(
            f"grid carries no {PHASE_MODE_TAG!r} tag and is read by the schema rule "
            f"(line-to-neutral one-phase nodes); tag it with mark_legacy, or pass "
            "assume_legacy=True, when it is a line-to-line equivalent of an older "
            "converter."
        )
    if mode not in (None, _LEGACY_LL):
        raise InputError(
            f"grid is tagged {PHASE_MODE_TAG}={mode!r}; only a {_LEGACY_LL!r} grid is "
            "migrated."
        )
    if not _is_one_phase(grid):
        raise InputError(
            "only a grid whose nodes are all one-phase can be migrated to the "
            "per-phase equivalent."
        )
    single = PhaseMode.SINGLE_PHASE_EQUIV
    out = _rescale_grid(
        grid,
        lambda u: equivalent_voltage(single, u),
        lambda p: equivalent_power(single, p),
    )
    return _with_tag(out, single.value)
