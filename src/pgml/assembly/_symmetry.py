"""Calculation-symmetry / load-connection resolution + final-modeling logging.

These pure (torch-free) helpers turn a grid + the simulation config into the two
modeling decisions the asymmetric assembly path needs, and log the FINAL modeling
so a user can see exactly what was built:

1. ``resolve_asymmetric`` — is the calculation ASYMMETRIC (per-phase operating
   points honored) or SYMMETRIC (each appliance's total split equally over its
   phases)?  pgml is always phase-domain, so this is operating-point resolution,
   not a network change — mirroring power-grid-model's ``symmetric=True/False``
   (a symmetric calc averages an asymmetric load).
2. ``resolve_connection`` — is a Load/Generator WYE (phase-to-neutral/ground) or
   DELTA (phase-to-phase)?  Explicit ``connection`` wins; otherwise the config
   default (single-phase vs multi-phase).

``log_modeling_summary`` emits an INFO summary: the resolved calculation symmetry,
whether a NEUTRAL is being modeled (a node carries ``Phase.N``), the WYE/DELTA load mix
and the frequency-dependent line models in use. It WARNS on the two silent modeling
traps: a line with no harmonic model, and a radius-based conductor internal-inductance
model applied to a synthesized geometry whose radius is a placeholder. Citations +
rationale: ``docs/pgml/modeling/asymmetric.md``.

Runs once per assemble/solve call on the python schema objects — no tensors, no
autograd, no per-node loops on the tape.
"""

from __future__ import annotations

import logging
from typing import Optional

from pgml import defaults
from pgml.errors import InputError
from pgml.schemas.grid_schema import (
    Grid,
    InjectionAppliance,
    Line,
    Phase,
    WindingConnection,
    _has_resistance_law,
)

logger = logging.getLogger("pgml")

_VALID_MODES = ("auto", "symmetric", "asymmetric")


def _has_per_phase_appliance(grid: Grid) -> bool:
    """True if any injecting appliance carries an explicit per-phase nameplate split."""
    return any(
        isinstance(a, InjectionAppliance)
        and (a.p_nom_per_phase_w is not None or a.q_nom_per_phase_var is not None)
        for a in grid.appliances
    )


def _has_per_phase_operating_point(operating_point: Optional[dict]) -> bool:
    """True if any operating-point entry carries per-phase P or Q."""
    if not operating_point:
        return False
    return any(
        isinstance(entry, dict)
        and ("p_per_phase_w" in entry or "q_per_phase_var" in entry)
        for entry in operating_point.values()
    )


def resolve_asymmetric(
    grid: Grid,
    operating_point: Optional[dict] = None,
    *,
    mode: Optional[str] = None,
) -> bool:
    """Resolve the calculation symmetry to a bool (``True`` == asymmetric/per-phase).

    Parameters
    ----------
    grid:
        The grid (inspected for per-phase appliance data in ``auto`` mode).
    operating_point:
        Optional operating-point override (inspected for per-phase keys in ``auto``).
    mode:
        ``None`` -> read the config default ``calculation.symmetry``. Otherwise one of
        ``"symmetric"`` (balanced equal split), ``"asymmetric"`` (per-phase honored),
        or ``"auto"`` (asymmetric iff any appliance / operating point carries per-phase
        data, else symmetric — the power-grid-model rule).

    PURE (no logging / no side effects): it is called once per assemble/solve AND on
    every power-flow residual evaluation, so it must not spam logs. The single INFO
    summary is emitted by :func:`log_modeling_summary`.
    """
    if mode is None:
        mode = defaults.get("calculation.symmetry")
    m = str(mode).lower()
    if m not in _VALID_MODES:
        raise InputError(
            f"calculation symmetry must be one of {_VALID_MODES}; got {mode!r}."
        )

    if m == "symmetric":
        return False
    if m == "asymmetric":
        return True
    # auto
    return _has_per_phase_appliance(grid) or _has_per_phase_operating_point(
        operating_point
    )


def resolve_connection(appliance) -> WindingConnection:
    """Effective connection of a Load/Generator: explicit if set, else modeling default.

    A 1-phase appliance defaults to ``appliance.load.single_phase_connection``; a
    multi-phase appliance to ``appliance.load.default_connection`` (both WYE by
    default). WYE_GROUNDED is treated as WYE for an appliance terminal (the return is
    the node's ``Phase.N`` row if present, else ground).
    """
    explicit = getattr(appliance, "connection", None)
    if explicit is not None:
        return explicit
    key = (
        "appliance.load.single_phase_connection"
        if len(appliance.phases) == 1
        else "appliance.load.default_connection"
    )
    return WindingConnection(defaults.get(key))


def log_modeling_summary(grid: Grid, *, asymmetric: bool) -> None:
    """INFO-log the FINAL modeling: neutral handling + load connections + symmetry.

    Makes implicit modeling explicit, e.g. that a neutral is being modeled because a
    node carries ``Phase.N`` (so WYE loads there return into the neutral, not ground).
    Call once per assemble/solve after :func:`resolve_asymmetric`.

    The summary counts nodes and appliances, which on a large grid costs more than the
    log call itself, so it is built only when the INFO level is enabled. The modeling
    WARNINGS of :func:`log_line_models` are emitted either way.
    """
    if logger.isEnabledFor(logging.INFO):
        neutral_nodes = [nd.id for nd in grid.nodes if Phase.N in nd.phases]
        if neutral_nodes:
            logger.info(
                "pgml: NEUTRAL modeled as a solved row at %d node(s): %s "
                "(WYE appliances there return into Phase.N, not ground).",
                len(neutral_nodes),
                neutral_nodes,
            )
        else:
            logger.info(
                "pgml: no Phase.N present — WYE appliances return to ground "
                "(3-wire / solidly-grounded model)."
            )

        appliances = [a for a in grid.appliances if isinstance(a, InjectionAppliance)]
        if appliances:
            counts: dict[str, int] = {}
            for a in appliances:
                c = resolve_connection(a).value
                counts[c] = counts.get(c, 0) + 1
            logger.info(
                "pgml: %d load/gen connection(s): %s; calculation = %s.",
                len(appliances),
                ", ".join(f"{k}x{v}" for k, v in sorted(counts.items())),
                "ASYMMETRIC (per-phase)"
                if asymmetric
                else "SYMMETRIC (balanced split)",
            )

    log_line_models(grid)


def _line_model_name(line: Line) -> str:
    """Which frequency-dependent model one line is assembled with, as a name.

    ``"unresolved"`` when neither a typed ``harmonic_line_model`` nor an explicit
    ``resistance_frequency`` law says, which is what the modeling WARNING reports.
    """
    return line.harmonic_line_model or (
        "explicit resistance_frequency"
        if _has_resistance_law(line.resistance_frequency)
        else "unresolved"
    )


def log_line_models(grid: Grid) -> None:
    """INFO-log which frequency-dependent line model each line uses.

    A line whose ``harmonic_line_model`` is still unresolved is assembled from its
    stored parameters (constant ``R``, ``X`` proportional to ``h``), which is the naive
    model the modeling defaults deliberately do not choose — so an unresolved line is
    logged as a WARNING naming the count, the first few line ids and the entry point that
    resolves it. This matters above the fundamental only, and it is not cosmetic: the
    model a three-phase lumped line resolves to is the sequence-aware one, which carries
    a zero-sequence earth-return term the naive model has no equivalent for. Converted
    grids are resolved at conversion time; a grid built without a converter is resolved by
    ``pgml.geometry.apply_default_harmonic_model``.
    """
    lines = [b for b in grid.branches if isinstance(b, Line) and b.in_service]
    if not lines:
        return
    unresolved_ids = [
        int(ln.id) for ln in lines if _line_model_name(ln) == "unresolved"
    ]
    if logger.isEnabledFor(logging.INFO):
        counts: dict[str, int] = {}
        for ln in lines:
            name = _line_model_name(ln)
            counts[name] = counts.get(name, 0) + 1
        logger.info(
            "pgml: %d line harmonic model(s): %s.",
            len(lines),
            ", ".join(f"{k}x{v}" for k, v in sorted(counts.items())),
        )
    if unresolved_ids:
        logger.warning(
            "pgml: %d of %d lines have no harmonic line model and are assembled from "
            "their stored parameters (R constant, X proportional to h). Above the "
            "fundamental this is the naive model, which differs from the resolved one "
            "(a three-phase lumped line resolves to the sequence-aware model, whose "
            "zero-sequence earth-return term the naive model omits). First id(s): %s. "
            "Apply the documented default with "
            "pgml.geometry.apply_default_harmonic_model(grid) (the converters do it "
            "for you) or set Line.harmonic_line_model.",
            len(unresolved_ids),
            len(lines),
            unresolved_ids[:10],
        )
    log_synthesized_geometry_radius(lines)


def log_synthesized_geometry_radius(lines) -> None:
    """WARN when a radius-based internal-inductance model meets a synthesized geometry.

    ``pgml.geometry.synthesize_line_geometry`` fits the GMR to the line's reactance and
    keeps the modeling-default radius as a placeholder, so a low-reactance line ends up
    with ``GMR >> radius`` (flagged ``synth_unphysical``). Every internal-inductance
    model except ``"gmr"`` reads that placeholder radius, which then dominates the
    self-impedance: measured on the CIGRE LV residential feeder, the series ``Z`` above
    1 kHz moves by a factor of 20. The combination is a modeling error, not a refinement.
    """
    model = defaults.get("line.geometry.internal_inductance")
    affected_models = [
        ln.conductor_geometry.internal_inductance or model
        for ln in lines
        if ln.conductor_geometry is not None
        and (ln.conductor_geometry.internal_inductance or model) != "gmr"
        and ln.conductor_geometry.provenance is not None
        and ln.conductor_geometry.provenance.extra.get("synth_unphysical") == "True"
    ]
    if affected_models:
        logger.warning(
            "pgml: resolved internal_inductance=%s uses the conductor RADIUS, but "
            "%d line(s) carry a synthesized geometry whose radius is a placeholder "
            "(GMR >= radius, tagged synth_unphysical). Their harmonic impedance will be "
            "dominated by that placeholder. Use 'gmr' for synthesized geometries, or "
            "give these lines measured conductor data.",
            ", ".join(sorted(set(affected_models))),
            len(affected_models),
        )


__all__ = [
    "resolve_asymmetric",
    "resolve_connection",
    "log_modeling_summary",
    "log_line_models",
    "log_synthesized_geometry_radius",
]
