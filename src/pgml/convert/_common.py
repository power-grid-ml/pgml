"""Shared converter scaffold (library-agnostic plumbing for every ``to_grid``).

This module drains the duplicated mechanics out of the per-library converters
(``pandapower``, ``pgm``, ``opendss``) into one documented place so each converter
only contains the source-specific field reading. It provides:

- :class:`IdCounter` — the single monotonic id allocator.
- :class:`PhaseMode` — single-phase positive-sequence equivalent vs genuine abc.
- :func:`phases_for` — the one place the node/branch phase tuple is decided.
- :func:`zero_sequence_ratios` — the defaults-driven R0/R1, X0/X1, C0/C1 ratios.
- :func:`sequence_to_phase_matrices` — sequence (1/0) quantities -> 3x3 phase
  matrices via the symmetric-component identity ``self=(Z0+2*Z1)/3``,
  ``mutual=(Z0-Z1)/3`` (and likewise for the shunt C).
- :func:`single_phase_matrix` — the trivial 1x1 wrapper for the positive-sequence
  equivalent path (preserves the exact ``[[value]]`` output of the old converters).
- :func:`thevenin_from_z` / :func:`thevenin_from_sk` — source Thevenin (R, L)
  from an explicit impedance or from short-circuit power + R/X ratio.
- Emit helpers — :func:`build_node`, :func:`build_load`, :func:`build_source`,
  :func:`build_line_from_sequence`, :func:`build_line_from_matrices` — the single
  place the :class:`PhaseMode` decision and the per-phase mapping live, so all three
  converters stamp identical schema objects for the same physical input.

Zero-sequence assumption
------------------------
When a positive-sequence (``r1``/``x1``/``c1``) line is expanded to a genuine
3-phase phase-domain matrix (:data:`PhaseMode.THREE_PHASE`) and the source dataset
carries no native zero-sequence data, the zero-sequence quantities default to
``r1 * (R0/R1)`` etc. using the ratios in ``pgml.defaults`` (``line.zero_sequence.*``).
An explicit per-line ``r0``/``x0``/``c0`` always wins over these defaults.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Optional

from pgml import defaults
from pgml.schemas.grid_schema import (
    ConstantParam,
    Generator,
    GridMetadata,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Provenance,
    ResistanceFrequencyModel,
    Source,
    WindingConnection,
)

# Default abc phase tuple for an expanded three-phase node / branch.
_ABC: tuple[Phase, ...] = (Phase.A, Phase.B, Phase.C)
_PHASE_A: tuple[Phase, ...] = (Phase.A,)


# =============================================================================
# Id allocation
# =============================================================================
class IdCounter:
    """Monotonically increasing integer id generator (shared across converters).

    Allocate node, branch and appliance ids from one instance per conversion so
    the integer ids never collide across element classes.
    """

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


# =============================================================================
# Phase mode
# =============================================================================
class PhaseMode(str, Enum):
    """How a converted grid represents phases.

    - :data:`SINGLE_PHASE_EQUIV` — today's positive-sequence single-phase
      equivalent: every node/branch is ``phases=(Phase.A,)`` and lines carry 1x1
      matrices. Reproduces the historical converter output byte-for-byte.
    - :data:`THREE_PHASE` — genuine abc: nodes/branches become ``(A, B, C)`` (or a
      source-native phase tuple, e.g. OpenDSS including ``Phase.N``); lines carry
      n x n matrices built from sequence data or supplied explicitly.
    """

    SINGLE_PHASE_EQUIV = "single_phase_equiv"
    THREE_PHASE = "three_phase"


def phases_for(
    mode: PhaseMode, *, native: Optional[tuple[Phase, ...]] = None
) -> tuple[Phase, ...]:
    """Return the phase tuple for a node/branch under ``mode``.

    Parameters
    ----------
    mode:
        The :class:`PhaseMode`.
    native:
        For :data:`PhaseMode.THREE_PHASE`, an explicit phase tuple from the source
        (OpenDSS passes the real phases, possibly including ``Phase.N``). Ignored
        for :data:`PhaseMode.SINGLE_PHASE_EQUIV`.

    Returns
    -------
    tuple[Phase, ...]
        ``(Phase.A,)`` for the single-phase equivalent; for three-phase the
        ``native`` tuple when given, else ``(Phase.A, Phase.B, Phase.C)``.
    """
    if mode is PhaseMode.SINGLE_PHASE_EQUIV:
        return _PHASE_A
    return native if native is not None else _ABC


# =============================================================================
# Zero-sequence defaults and the sequence -> phase identity
# =============================================================================
def zero_sequence_ratios() -> tuple[float, float, float]:
    """Return ``(r0_over_r1, x0_over_x1, c0_over_c1)`` from ``pgml.defaults``.

    These are the assumed zero/positive-sequence ratios used to synthesize a
    zero-sequence quantity when a positive-sequence line is expanded to abc and
    the dataset has no native zero-sequence data (``line.zero_sequence.*``).
    """
    return (
        float(defaults.get("line.zero_sequence.r0_over_r1")),
        float(defaults.get("line.zero_sequence.x0_over_x1")),
        float(defaults.get("line.zero_sequence.c0_over_c1")),
    )


def _circulant_3x3(self_val: float, mutual_val: float) -> list[list[float]]:
    """Return a 3x3 symmetric circulant matrix with ``self`` on the diagonal and
    ``mutual`` off-diagonal (the phase matrix of a balanced sequence quantity)."""
    return [[self_val if i == j else mutual_val for j in range(3)] for i in range(3)]


def sequence_to_phase_matrices(
    r1: float,
    x1: float,
    c1: float,
    *,
    r0: Optional[float] = None,
    x0: Optional[float] = None,
    c0: Optional[float] = None,
    two_pi_f0: float,
    g0: float = 0.0,
    g1: float = 0.0,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    """Build per-phase 3x3 matrices ``(R, L, C, G)`` from sequence quantities.

    Uses the symmetric-component identity (see
    :class:`~pgml.schemas.grid_schema.LineSequenceInput`)::

        Q_self   = (Q0 + 2*Q1) / 3
        Q_mutual = (Q0 -   Q1) / 3

    applied independently to R, X, C and G. For a balanced sequence input this
    yields a symmetric circulant matrix (off-diagonal == mutual).

    Parameters
    ----------
    r1, x1, c1:
        Positive-sequence series resistance [Ohm/m], series reactance [Ohm/m] and
        shunt capacitance [F/m] per unit length.
    r0, x0, c0:
        Zero-sequence counterparts. When ``None`` each defaults to the positive
        value times the corresponding ratio from :func:`zero_sequence_ratios`.
    two_pi_f0:
        ``2*pi*f0`` used to convert reactance to inductance ``L = X / (2*pi*f0)``.
    g0, g1:
        Optional zero/positive-sequence shunt conductance [S/m]; default 0.

    Returns
    -------
    (R, L, C, G):
        Four 3x3 nested python lists (SI per-length). ``L`` is derived from the
        per-phase reactance matrix divided by ``two_pi_f0``.
    """
    rr0, xr0, cr0 = zero_sequence_ratios()
    r0v = r1 * rr0 if r0 is None else r0
    x0v = x1 * xr0 if x0 is None else x0
    c0v = c1 * cr0 if c0 is None else c0
    g0v = g0  # no config ratio for conductance; defaults handled by caller

    def split(q0: float, q1: float) -> tuple[float, float]:
        return (q0 + 2.0 * q1) / 3.0, (q0 - q1) / 3.0

    r_self, r_mut = split(r0v, r1)
    x_self, x_mut = split(x0v, x1)
    c_self, c_mut = split(c0v, c1)
    g_self, g_mut = split(g0v, g1)

    r_mat = _circulant_3x3(r_self, r_mut)
    x_mat = _circulant_3x3(x_self, x_mut)
    c_mat = _circulant_3x3(c_self, c_mut)
    g_mat = _circulant_3x3(g_self, g_mut)

    l_mat = [[x_mat[i][j] / two_pi_f0 for j in range(3)] for i in range(3)]
    return r_mat, l_mat, c_mat, g_mat


def single_phase_matrix(value: float) -> list[list[float]]:
    """Wrap a scalar into the trivial 1x1 matrix ``[[value]]``.

    Used by the single-phase positive-sequence equivalent line path; preserves the
    exact ``[[r1]]`` / ``[[l1]]`` / ``[[c1]]`` output of the historical converters.
    """
    return [[value]]


# =============================================================================
# Source Thevenin
# =============================================================================
# Fallback Thevenin values for a near-ideal source (ideal-slack mode ignores Z).
_FALLBACK_R = 1.0e-6  # Ohm
_FALLBACK_L = 1.0e-12  # H


def thevenin_from_z(
    r_ohm: float, x_ohm: float, two_pi_f0: float
) -> tuple[float, float]:
    """Return Thevenin ``(R [Ohm], L [H])`` from an explicit series impedance.

    ``L = X / (2*pi*f0)``. Both quantities are floored at tiny positive values so a
    degenerate (zero) impedance does not produce a singular Norton stamp.
    """
    r_s = max(r_ohm, _FALLBACK_R)
    l_s = x_ohm / two_pi_f0 if two_pi_f0 > 0.0 else _FALLBACK_L
    l_s = max(l_s, _FALLBACK_L)
    return r_s, l_s


def thevenin_from_sk(
    u_rated_v: float,
    sk_va: float,
    rx_ratio: float,
    two_pi_f0: float,
) -> tuple[float, float]:
    """Derive Thevenin ``(R [Ohm], L [H])`` from short-circuit power and R/X ratio.

    ::

        |Z_s| = u_rated_v^2 / sk_va        (positive-sequence, 3-phase base)
        X_s   = |Z_s| / sqrt(1 + rx_ratio^2)
        R_s   = X_s * rx_ratio
        L_s   = X_s / two_pi_f0

    Falls back to tiny values when ``sk_va`` is non-positive or unreasonably large
    (> 1e15 VA) to avoid overflow; near-zero results are floored. (Moved verbatim
    from the pgm converter so its numerics are unchanged.)
    """
    if sk_va <= 0.0 or sk_va > 1.0e15:
        return _FALLBACK_R, _FALLBACK_L

    z_mag = (u_rated_v**2) / sk_va
    denom = math.sqrt(1.0 + rx_ratio**2)
    x_s = z_mag / denom
    r_s = x_s * rx_ratio
    l_s = x_s / two_pi_f0 if two_pi_f0 > 0.0 else _FALLBACK_L

    r_s = max(r_s, _FALLBACK_R)
    l_s = max(l_s, _FALLBACK_L)
    return r_s, l_s


# =============================================================================
# Provenance / metadata helpers
# =============================================================================
def make_metadata(name: str, description: str) -> GridMetadata:
    """Build a :class:`~pgml.schemas.grid_schema.GridMetadata` (thin shared helper)."""
    return GridMetadata(name=name, description=description)


# =============================================================================
# Emit helpers (the single place the phase decision + per-phase mapping live)
# =============================================================================
def build_node(
    *,
    id: int,
    u_rated_v: float,
    mode: PhaseMode,
    name: Optional[str] = None,
    native_phases: Optional[tuple[Phase, ...]] = None,
) -> Node:
    """Build a :class:`~pgml.schemas.grid_schema.Node` with mode-resolved phases.

    Parameters
    ----------
    id, u_rated_v, name:
        Node id, rated voltage [V] and optional name.
    mode:
        :class:`PhaseMode`; selects ``(A,)`` vs ``(A,B,C)`` / ``native_phases``.
    native_phases:
        Source-native phase tuple used under :data:`PhaseMode.THREE_PHASE`.
    """
    return Node(
        id=id,
        name=name,
        u_rated_v=u_rated_v,
        phases=phases_for(mode, native=native_phases),
    )


def build_load(
    *,
    id: int,
    node: int,
    mode: PhaseMode,
    p_total_w: float,
    q_total_var: float,
    name: Optional[str] = None,
    connection: Optional[WindingConnection] = None,
    p_per_phase_w: Optional[tuple[float, ...]] = None,
    q_per_phase_var: Optional[tuple[float, ...]] = None,
    native_phases: Optional[tuple[Phase, ...]] = None,
    load_model: Optional[LoadModel] = None,
) -> Load:
    """Build a :class:`~pgml.schemas.grid_schema.Load`, phase decision centralized.

    Under :data:`PhaseMode.SINGLE_PHASE_EQUIV` the load is emitted with
    ``phases=(Phase.A,)`` and the TOTAL P/Q only (today's exact behaviour); any
    per-phase split is collapsed by the caller before reaching here. Under
    :data:`PhaseMode.THREE_PHASE` the load is emitted on ``(A,B,C)`` (or
    ``native_phases``) and threads ``connection`` plus the per-phase nameplate
    split when given.

    Parameters
    ----------
    id, node, name:
        Load id, host node id and optional name.
    mode:
        :class:`PhaseMode`.
    p_total_w, q_total_var:
        Total active [W] / reactive [VAr] nameplate power.
    connection:
        WYE / DELTA; ``None`` (default) resolves from config at assembly. Threaded
        only under :data:`PhaseMode.THREE_PHASE`.
    p_per_phase_w, q_per_phase_var:
        Optional per-phase nameplate split (must sum to the totals). Honored only
        under :data:`PhaseMode.THREE_PHASE`; ignored for the single-phase
        equivalent (a 1-phase asymmetric load keeps its single phase via
        ``native_phases``).
    native_phases:
        Source-native phase tuple used under :data:`PhaseMode.THREE_PHASE` (e.g. a
        genuine 1-phase load is ``(Phase.A,)``).
    load_model:
        Optional :class:`~pgml.schemas.grid_schema.LoadModel`; ``None`` leaves the
        schema default (``CONST_POWER``) — used to keep the pandapower output
        byte-identical (it never set ``load_model``).
    """
    kwargs: dict[str, Any] = {
        "id": id,
        "name": name,
        "node": node,
        "p_nom_w": p_total_w,
        "q_nom_var": q_total_var,
    }
    if load_model is not None:
        kwargs["load_model"] = load_model

    if mode is PhaseMode.SINGLE_PHASE_EQUIV:
        kwargs["phases"] = _PHASE_A
        return Load(**kwargs)

    kwargs["phases"] = phases_for(mode, native=native_phases)
    if connection is not None:
        kwargs["connection"] = connection
    if p_per_phase_w is not None:
        kwargs["p_nom_per_phase_w"] = p_per_phase_w
    if q_per_phase_var is not None:
        kwargs["q_nom_per_phase_var"] = q_per_phase_var
    return Load(**kwargs)


def build_generator(
    *,
    id: int,
    node: int,
    mode: PhaseMode,
    p_total_w: float,
    q_total_var: float,
    name: Optional[str] = None,
    connection: Optional[WindingConnection] = None,
    native_phases: Optional[tuple[Phase, ...]] = None,
    consumer_type: Optional[str] = None,
) -> Generator:
    """Build a :class:`~pgml.schemas.grid_schema.Generator` (PQ injection).

    Mirrors :func:`build_load` with the GENERATION-POSITIVE nameplate convention:
    ``p_total_w > 0`` injects into the grid (the assembly applies the −1 sign for
    the :class:`Generator` component). Used for source-library static generators
    (pandapower ``sgen``, power-grid-model ``sym_gen``).
    """
    kwargs: dict[str, Any] = {
        "id": id,
        "name": name,
        "node": node,
        "p_nom_w": p_total_w,
        "q_nom_var": q_total_var,
    }
    if consumer_type is not None:
        kwargs["consumer_type"] = consumer_type
    if mode is PhaseMode.SINGLE_PHASE_EQUIV:
        kwargs["phases"] = _PHASE_A
        return Generator(**kwargs)
    kwargs["phases"] = phases_for(mode, native=native_phases)
    if connection is not None:
        kwargs["connection"] = connection
    return Generator(**kwargs)


def warn_dropped_elements(logger, source_name: str, dropped: dict) -> None:
    """Log one WARNING per non-empty element table the converter does not read.

    A silently dropped element (a shunt, a 3-winding transformer, ...) yields a
    grid that solves but is physically WRONG relative to the source network —
    the caller must be told. ``dropped`` maps ``element kind -> count`` (zero /
    falsy counts are skipped).
    """
    for kind, count in dropped.items():
        if count:
            logger.warning(
                "%s -> Grid: %d %r element(s) present in the source network are "
                "NOT converted (unsupported by the converter) — the converted "
                "grid omits their physics.",
                source_name,
                count,
                kind,
            )


def build_source(
    *,
    id: int,
    node: int,
    mode: PhaseMode,
    u_ref_v: float,
    u_angle_deg: float,
    r_ohm: float,
    l_h: float,
    name: Optional[str] = None,
    native_phases: Optional[tuple[Phase, ...]] = None,
) -> Source:
    """Build a :class:`~pgml.schemas.grid_schema.Source` (balanced Thevenin).

    Under :data:`PhaseMode.SINGLE_PHASE_EQUIV` the source is 1-phase
    (``phases=(A,)``, scalar ``u_ref``/angle, 1x1 R/L). Under
    :data:`PhaseMode.THREE_PHASE` it becomes a BALANCED 3-phase Thevenin: equal
    magnitudes, angles ``u_angle_deg`` and ``-120`` / ``+120`` offsets, and a
    diagonal per-phase R/L (positive-sequence value on every phase).

    Parameters
    ----------
    u_ref_v, u_angle_deg:
        Reference voltage magnitude [V] and angle [deg], given as the LINE-TO-LINE
        magnitude (matching ``Node.u_rated_v``). Under :data:`PhaseMode.THREE_PHASE`
        it is converted to the per-phase line-to-neutral phase-to-ground EMF
        (``u_ref_v / sqrt(3)``) for the balanced wye source; under
        :data:`PhaseMode.SINGLE_PHASE_EQUIV` it is used directly (positive-sequence
        equivalent).
    r_ohm, l_h:
        Per-phase Thevenin resistance [Ohm] and inductance [H] (placed on the
        diagonal under three-phase).
    native_phases:
        Source-native phase tuple under :data:`PhaseMode.THREE_PHASE`.
    """
    if mode is PhaseMode.SINGLE_PHASE_EQUIV:
        return Source(
            id=id,
            name=name,
            node=node,
            phases=_PHASE_A,
            u_ref_v=(u_ref_v,),
            u_angle_deg=(u_angle_deg,),
            resistance_ohm=[[r_ohm]],
            inductance_h=[[l_h]],
        )

    phases = phases_for(mode, native=native_phases)
    n = len(phases)
    # The ideal-slack solve pins each phase-to-GROUND row to this phasor, so a balanced
    # wye-grounded 3-phase source must carry the line-to-NEUTRAL magnitude. Converters
    # supply ``u_ref_v`` as the line-to-line magnitude (like ``Node.u_rated_v``), so
    # divide by sqrt(3) for a 3-/4-wire node — the same WYE convention the const-Z/ZIP
    # load model and the per-unit reporting use (see ``phase_voltage_magnitude``).
    u_ln = u_ref_v / math.sqrt(3.0) if n >= 3 else u_ref_v
    angles = tuple(u_angle_deg - 120.0 * i for i in range(n))
    r_mat = [[r_ohm if i == j else 0.0 for j in range(n)] for i in range(n)]
    l_mat = [[l_h if i == j else 0.0 for j in range(n)] for i in range(n)]
    return Source(
        id=id,
        name=name,
        node=node,
        phases=phases,
        u_ref_v=tuple(u_ln for _ in range(n)),
        u_angle_deg=angles,
        resistance_ohm=r_mat,
        inductance_h=l_mat,
    )


def build_line_from_matrices(
    *,
    id: int,
    from_node: int,
    to_node: int,
    phases: tuple[Phase, ...],
    length_m: float,
    r_matrix: list[list[float]],
    l_matrix: list[list[float]],
    c_matrix: list[list[float]],
    g_matrix: Optional[list[list[float]]] = None,
    name: Optional[str] = None,
    provenance: Optional[Provenance] = None,
) -> Line:
    """Build a :class:`~pgml.schemas.grid_schema.Line` from explicit n x n matrices.

    The general per-length form used by every converter; OpenDSS supplies the
    native n x n R/L/C matrices directly, while pandapower/pgm route through
    :func:`build_line_from_sequence`. ``from_phases`` and ``to_phases`` are both
    set to ``phases`` (a series branch shares its phase set on both ends).

    Parameters
    ----------
    phases:
        The branch phase tuple (e.g. ``(Phase.A,)`` or ``(A, B, C)``); also fixes
        the matrix row/column order.
    r_matrix, l_matrix, c_matrix:
        Per-length series resistance [Ohm/m], series inductance [H/m] and shunt
        capacitance [F/m] matrices (each ``len(phases) x len(phases)``).
    g_matrix:
        Optional shunt conductance [S/m] matrix; ``None`` means zeros.
    """
    return Line(
        id=id,
        name=name,
        from_node=from_node,
        to_node=to_node,
        from_phases=phases,
        to_phases=phases,
        length_m=length_m,
        series_resistance_ohm_per_m=r_matrix,
        series_inductance_h_per_m=l_matrix,
        shunt_capacitance_f_per_m=c_matrix,
        shunt_conductance_s_per_m=g_matrix,
        resistance_frequency=ResistanceFrequencyModel(
            multiplier=ConstantParam(value=1.0)
        ),
        provenance=provenance,
    )


def build_line_from_sequence(
    *,
    id: int,
    from_node: int,
    to_node: int,
    mode: PhaseMode,
    length_m: float,
    r1: float,
    x1: float,
    c1: float,
    two_pi_f0: float,
    r0: Optional[float] = None,
    x0: Optional[float] = None,
    c0: Optional[float] = None,
    g1: float = 0.0,
    g0: float = 0.0,
    name: Optional[str] = None,
    provenance: Optional[Provenance] = None,
) -> Line:
    """Build a :class:`~pgml.schemas.grid_schema.Line` from sequence (1/0) quantities.

    Under :data:`PhaseMode.SINGLE_PHASE_EQUIV` the line uses 1x1 positive-sequence
    matrices (``[[r1]]``, ``[[x1/two_pi_f0]]``, ``[[c1]]``; conductance only when
    ``g1`` is non-zero) — byte-identical to the historical pandapower/pgm output.
    Under :data:`PhaseMode.THREE_PHASE` the 3x3 phase matrices come from
    :func:`sequence_to_phase_matrices` (zero-sequence defaulted from config when
    ``r0``/``x0``/``c0`` are ``None``).

    Parameters
    ----------
    r1, x1, c1:
        Positive-sequence series resistance [Ohm/m], series reactance [Ohm/m],
        shunt capacitance [F/m] per unit length.
    two_pi_f0:
        ``2*pi*f0`` (``L = X / two_pi_f0``).
    r0, x0, c0:
        Optional native zero-sequence quantities (else config defaults).
    g1, g0:
        Optional positive/zero-sequence shunt conductance [S/m].
    """
    if mode is PhaseMode.SINGLE_PHASE_EQUIV:
        l1 = x1 / two_pi_f0
        r_mat = single_phase_matrix(r1)
        l_mat = single_phase_matrix(l1)
        c_mat = single_phase_matrix(c1)
        g_mat = single_phase_matrix(g1) if g1 != 0.0 else None
        phases = phases_for(mode)
    else:
        r_mat, l_mat, c_mat, g_full = sequence_to_phase_matrices(
            r1,
            x1,
            c1,
            r0=r0,
            x0=x0,
            c0=c0,
            two_pi_f0=two_pi_f0,
            g0=g0,
            g1=g1,
        )
        g_mat = g_full if (g1 != 0.0 or g0 != 0.0) else None
        phases = phases_for(mode)

    return build_line_from_matrices(
        id=id,
        name=name,
        from_node=from_node,
        to_node=to_node,
        phases=phases,
        length_m=length_m,
        r_matrix=r_mat,
        l_matrix=l_mat,
        c_matrix=c_mat,
        g_matrix=g_mat,
        provenance=provenance,
    )


__all__ = [
    "IdCounter",
    "PhaseMode",
    "phases_for",
    "zero_sequence_ratios",
    "sequence_to_phase_matrices",
    "single_phase_matrix",
    "thevenin_from_z",
    "thevenin_from_sk",
    "make_metadata",
    "build_node",
    "build_load",
    "build_generator",
    "warn_dropped_elements",
    "build_source",
    "build_line_from_matrices",
    "build_line_from_sequence",
]
