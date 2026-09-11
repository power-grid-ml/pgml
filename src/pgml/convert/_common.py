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
- :func:`resolve_converted_line_models` — turn the configured default harmonic line
  model into per-line :attr:`~pgml.schemas.grid_schema.Line.harmonic_line_model`
  values and log the one line naming what was applied.
- :class:`ZeroSequenceDefaults` — tally of the lines whose zero-sequence data was
  invented from the configured ratios, warned once per converted grid.
- :func:`source_zero_sequence_ratios` — the defaults-driven R0/R1, X0/X1 of a
  source Thevenin, used when the dataset carries no zero-sequence source data.
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

The same rule applies to a 3-phase :class:`~pgml.schemas.grid_schema.Source`: its
Thevenin matrix is built from ``(R1, X1)`` and ``(R0, X0)`` through
``Z_self = (Z0 + 2*Z1)/3``, ``Z_mutual = (Z0 - Z1)/3``, with
``source.zero_sequence.{r0_over_r1, x0_over_x1}`` as the documented fallback and a
WARNING naming the source whenever that fallback is used.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
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
    ZipCoefficients,
)

_logger = logging.getLogger(__name__)

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

    - :data:`SINGLE_PHASE_EQUIV` — the default positive-sequence single-phase
      equivalent: every node/branch is ``phases=(Phase.A,)`` and lines carry 1x1
      matrices.
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


@dataclass
class ZeroSequenceDefaults:
    """Tally of lines whose zero-sequence data was invented from configured ratios.

    A positive-sequence dataset (the common case for pandapower / power-grid-model LV
    and MV feeders) carries no ``r0``/``x0``/``c0``, so a three-phase expansion has to
    synthesise them from the ``line.zero_sequence.*`` ratios. Those ratios are overhead
    line rules of thumb; on a cable-dominated LV feeder ``R0/R1`` is typically much
    closer to 1-2, and the assumption moves every unbalanced and triplen result. The
    converter therefore counts the affected lines and warns ONCE per grid, naming the
    ratios it used.
    """

    lines: int = 0
    r0: int = 0
    x0: int = 0
    c0: int = 0

    def note(self, *, r0=None, x0=None, c0=None) -> None:
        """Record one converted line: ``None`` means the value had to be invented."""
        self.lines += 1
        self.r0 += r0 is None
        self.x0 += x0 is None
        self.c0 += c0 is None

    def warn(self, logger: logging.Logger, *, tool: str) -> None:
        """Emit the one-per-grid WARNING (no-op when nothing was invented)."""
        if not (self.r0 or self.x0 or self.c0):
            return
        rr0, xr0, cr0 = zero_sequence_ratios()
        logger.warning(
            "%s conversion: %d of %d three-phase lines carry no zero-sequence data; "
            "R0 invented for %d (R0/R1=%g), X0 for %d (X0/X1=%g), C0 for %d "
            "(C0/C1=%g) from line.zero_sequence.*. These are overhead-line rules of "
            "thumb: every unbalanced or triplen-harmonic result on this grid rests on "
            "them (a cable feeder typically has a much lower R0/R1). Supply native "
            "zero-sequence data, set the ratios in the modeling defaults, or give the "
            "lines a conductor_geometry.",
            tool,
            max(self.r0, self.x0, self.c0),
            self.lines,
            self.r0,
            rr0,
            self.x0,
            xr0,
            self.c0,
            cr0,
        )


def resolve_converted_line_models(
    grid,
    logger: logging.Logger,
    *,
    tool: str,
    requested: Optional[str] = None,
) -> None:
    """Give every converted R/X line its harmonic line model, and log what was applied.

    A source library has no concept of a frequency-dependent line model, so the
    converted grid would otherwise reach a harmonic solve with an unresolved model and
    be assembled with constant ``R`` and ``X ∝ h`` — the naive model the modeling
    defaults deliberately do not choose. The default
    (``line.harmonic_model.three_phase`` = ``sequence_aware``,
    ``.single_phase`` = ``positive_sequence``) is therefore applied here, at conversion
    time, where it can be logged; ``requested`` overrides it for every line
    (``"none"`` leaves the lines unresolved, reproducing the raw stored parameters).
    Lines that carry a ``conductor_geometry`` keep the full Carson model.
    """
    from pgml.geometry.synthesis import resolve_harmonic_line_models

    counts = resolve_harmonic_line_models(grid, model=requested)
    if not counts:
        return
    applied = ", ".join(f"{n} x {name}" for name, n in sorted(counts.items()))
    if requested is None:
        logger.info(
            "%s conversion: harmonic line model applied from the modeling defaults "
            "(line.harmonic_model.*): %s. Pass harmonic_line_model= to to_grid() to "
            "choose another model, or give the lines a conductor_geometry for the full "
            "Carson/Deri model.",
            tool,
            applied,
        )
    else:
        logger.info(
            "%s conversion: harmonic line model %r applied as requested: %s.",
            tool,
            requested,
            applied,
        )


def source_zero_sequence_ratios() -> tuple[float, float]:
    """Return ``(r0_over_r1, x0_over_x1)`` of a source Thevenin from ``pgml.defaults``.

    The assumed zero/positive-sequence ratios used to synthesize a source's
    zero-sequence impedance when it is expanded to a genuine 3-phase Thevenin and
    the dataset carries no native zero-sequence data
    (``source.zero_sequence.*``). Native data always wins.
    """
    return (
        float(defaults.get("source.zero_sequence.r0_over_r1")),
        float(defaults.get("source.zero_sequence.x0_over_x1")),
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

    Used by the single-phase positive-sequence equivalent line path, which
    represents a line by its plain ``[[r1]]`` / ``[[l1]]`` / ``[[c1]]`` entries.
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
    (> 1e15 VA) to avoid overflow; near-zero results are floored.
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
    zip_coefficients: Optional[ZipCoefficients] = None,
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
    zip_coefficients:
        Optional :class:`~pgml.schemas.grid_schema.ZipCoefficients` for a
        voltage-dependent load. Setting it implies ``load_model=ZIP`` (an explicit
        conflicting ``load_model`` raises). The nonlinear solver honors the mix;
        the linear const-Z assembler keeps using the base P/Q.
    """
    if zip_coefficients is not None:
        if load_model is not None and load_model is not LoadModel.ZIP:
            raise ValueError(
                f"zip_coefficients conflicts with load_model={load_model!r} "
                "(coefficients imply LoadModel.ZIP)."
            )
        load_model = LoadModel.ZIP
    kwargs: dict[str, Any] = {
        "id": id,
        "name": name,
        "node": node,
        "p_nom_w": p_total_w,
        "q_nom_var": q_total_var,
    }
    if load_model is not None:
        kwargs["load_model"] = load_model
    if zip_coefficients is not None:
        kwargs["zip_coefficients"] = zip_coefficients

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
    control: Optional[Any] = None,
) -> Generator:
    """Build a :class:`~pgml.schemas.grid_schema.Generator` (PQ injection).

    Mirrors :func:`build_load` with the GENERATION-POSITIVE nameplate convention:
    ``p_total_w > 0`` injects into the grid (the assembly applies the −1 sign for
    the :class:`Generator` component). Used for source-library static generators
    (pandapower ``sgen``, power-grid-model ``sym_gen``).

    ``control`` optionally attaches an
    :data:`~pgml.schemas.grid_schema.InverterControl` block (constant power factor,
    ``cosphi(P)``, Volt-VAr, Volt-Watt), making the injection voltage-dependent. A
    controlled generator's reactive nameplate is never read — the control law
    supplies Q — so pass ``q_total_var=0.0`` with one.
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
    if control is not None:
        kwargs["control"] = control
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
    r0_ohm: Optional[float] = None,
    x0_ohm: Optional[float] = None,
    two_pi_f0: Optional[float] = None,
    element: Optional[str] = None,
) -> Source:
    """Build a :class:`~pgml.schemas.grid_schema.Source` (balanced Thevenin).

    Under :data:`PhaseMode.SINGLE_PHASE_EQUIV` the source is 1-phase
    (``phases=(A,)``, scalar ``u_ref``/angle, 1x1 R/L) — a positive-sequence
    equivalent has no zero sequence, so ``r0_ohm``/``x0_ohm`` are ignored there.
    Under :data:`PhaseMode.THREE_PHASE` it becomes a BALANCED 3-phase Thevenin:
    equal magnitudes, angles ``u_angle_deg`` and ``-120`` / ``+120`` offsets, and a
    SEQUENCE-AWARE per-phase R/L matrix built from the positive- and
    zero-sequence impedances through the symmetric-component identity
    (:func:`sequence_to_phase_matrices`)::

        Z_self   = (Z0 + 2*Z1) / 3
        Z_mutual = (Z0 -   Z1) / 3

    so the off-diagonal terms carry the difference between Z0 and Z1. With
    ``Z0 == Z1`` the mutual term vanishes and the matrix is the plain diagonal
    ``Z1`` stamp. The identity is applied to the A/B/C rows only; any further
    conductor (an explicit ``Phase.N`` on a 4-wire source terminal) keeps the
    positive-sequence value on its diagonal and no mutual coupling.

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
        Positive-sequence Thevenin resistance [Ohm] and inductance [H].
    native_phases:
        Source-native phase tuple under :data:`PhaseMode.THREE_PHASE`.
    r0_ohm, x0_ohm:
        Native zero-sequence resistance [Ohm] and reactance at ``f0`` [Ohm]. When
        either is ``None`` it is synthesized from the positive-sequence value and
        the documented ratio in :func:`source_zero_sequence_ratios`, and a WARNING
        naming ``element`` (or the source id) and the ratio is logged — but only
        under :data:`PhaseMode.THREE_PHASE`, where the zero sequence exists.
    two_pi_f0:
        ``2*pi*f0``, required to convert ``x0_ohm`` into the stored inductance
        (``L0 = X0 / (2*pi*f0)``). Required whenever the zero-sequence path is
        built (i.e. under :data:`PhaseMode.THREE_PHASE` with >= 3 phases).
    element:
        Source-library element name used in the fallback WARNING.
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
    r_mat, l_mat = _source_impedance_matrices(
        n=n,
        phases=phases,
        r_ohm=r_ohm,
        l_h=l_h,
        r0_ohm=r0_ohm,
        x0_ohm=x0_ohm,
        two_pi_f0=two_pi_f0,
        element=element if element is not None else f"source {id}",
    )
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


def _source_impedance_matrices(
    *,
    n: int,
    phases: tuple[Phase, ...],
    r_ohm: float,
    l_h: float,
    r0_ohm: Optional[float],
    x0_ohm: Optional[float],
    two_pi_f0: Optional[float],
    element: str,
) -> tuple[list[list[float]], list[list[float]]]:
    """Per-phase (R, L) matrices of a 3-phase source Thevenin, zero-sequence aware.

    Returns the plain diagonal positive-sequence stamp when fewer than three A/B/C
    conductors are present (no zero sequence is defined for a 1- or 2-conductor
    terminal); otherwise the symmetric-component self/mutual split over the A/B/C
    rows, leaving any additional conductor (``Phase.N``) diagonal.
    """
    abc_idx = [k for k, ph in enumerate(phases) if ph in _ABC]
    r_mat = [[r_ohm if i == j else 0.0 for j in range(n)] for i in range(n)]
    l_mat = [[l_h if i == j else 0.0 for j in range(n)] for i in range(n)]
    if len(abc_idx) != 3:
        if r0_ohm is not None or x0_ohm is not None:
            _logger.warning(
                "%s: zero-sequence Thevenin data (R0/X0) is ignored — the source "
                "terminal carries %d of the A/B/C conductors, and the symmetric-"
                "component self/mutual split needs all three.",
                element,
                len(abc_idx),
            )
        return r_mat, l_mat

    if two_pi_f0 is None or two_pi_f0 <= 0.0:
        raise ValueError(
            f"{element}: two_pi_f0 must be positive to build the zero-sequence "
            "Thevenin matrix of a 3-phase source."
        )
    x_ohm = l_h * two_pi_f0
    if r0_ohm is None or x0_ohm is None:
        rr0, xr0 = source_zero_sequence_ratios()
        r0_ohm = r_ohm * rr0 if r0_ohm is None else r0_ohm
        x0_ohm = x_ohm * xr0 if x0_ohm is None else x0_ohm
        _logger.warning(
            "%s: no zero-sequence Thevenin data; assuming R0/R1 = %g and X0/X1 = %g "
            "(pgml defaults `source.zero_sequence.*`). The zero-sequence source "
            "impedance shapes triplen/residual voltage wherever zero-sequence "
            "current reaches the source.",
            element,
            rr0,
            xr0,
        )

    # Floor the zero-sequence pair exactly like the positive-sequence Thevenin
    # (:func:`thevenin_from_z`): an all-zero Z0 would make the per-phase matrix
    # singular in the zero sequence, and its inverse (the Norton stamp) undefined.
    r0_ohm = max(r0_ohm, _FALLBACK_R)
    x0_ohm = max(x0_ohm, two_pi_f0 * _FALLBACK_L)
    if r0_ohm == r_ohm and x0_ohm == x_ohm:
        # Z0 == Z1: the mutual term is exactly zero and the self term is exactly
        # Z1, so keep the plain diagonal stamp unchanged (no round-trip through
        # (Z0 + 2*Z1)/3, which would move the diagonal by one unit in the last place).
        return r_mat, l_mat

    r_self, r_mut = (r0_ohm + 2.0 * r_ohm) / 3.0, (r0_ohm - r_ohm) / 3.0
    x_self, x_mut = (x0_ohm + 2.0 * x_ohm) / 3.0, (x0_ohm - x_ohm) / 3.0
    for i in abc_idx:
        for j in abc_idx:
            r_mat[i][j] = r_self if i == j else r_mut
            l_mat[i][j] = (x_self if i == j else x_mut) / two_pi_f0
    return r_mat, l_mat


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
    to_phases: Optional[tuple[Phase, ...]] = None,
) -> Line:
    """Build a :class:`~pgml.schemas.grid_schema.Line` from explicit n x n matrices.

    The general per-length form used by every converter; OpenDSS supplies the
    native n x n R/L/C matrices directly, while pandapower/pgm route through
    :func:`build_line_from_sequence`. By default ``from_phases`` and ``to_phases``
    are both set to ``phases`` (a series branch usually shares its phase set on
    both ends); pass ``to_phases`` for a phase-transposing connection (e.g. an
    OpenDSS line wired ``bus1=a.1 bus2=b.2``) — conductor ``k`` then lands on
    ``phases[k]`` at the from terminal and ``to_phases[k]`` at the to terminal.

    Parameters
    ----------
    phases:
        The FROM-terminal phase tuple (e.g. ``(Phase.A,)`` or ``(A, B, C)``); also
        fixes the matrix row/column (conductor) order.
    r_matrix, l_matrix, c_matrix:
        Per-length series resistance [Ohm/m], series inductance [H/m] and shunt
        capacitance [F/m] matrices (each ``len(phases) x len(phases)``).
    g_matrix:
        Optional shunt conductance [S/m] matrix; ``None`` means zeros.
    to_phases:
        Optional TO-terminal phase tuple, aligned conductor-by-conductor with
        ``phases`` (must have the same length); ``None`` reuses ``phases``.
    """
    if to_phases is not None and len(to_phases) != len(phases):
        raise ValueError(
            f"to_phases must match the conductor count of phases "
            f"({len(to_phases)} != {len(phases)})."
        )
    return Line(
        id=id,
        name=name,
        from_node=from_node,
        to_node=to_node,
        from_phases=phases,
        to_phases=phases if to_phases is None else to_phases,
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
    "ZeroSequenceDefaults",
    "resolve_converted_line_models",
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
