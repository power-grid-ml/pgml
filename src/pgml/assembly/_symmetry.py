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
whether a NEUTRAL is being modeled (a node carries ``Phase.N``), and the WYE/DELTA
load mix. Citations + rationale: ``references/asymmetric_modeling.md``.

Runs once per assemble/solve call on the python schema objects — no tensors, no
autograd, no per-node loops on the tape.
"""

from __future__ import annotations

import logging
from typing import Optional

from pgml import config
from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    Load,
    Phase,
    WindingConnection,
)

logger = logging.getLogger("pgml")

_VALID_MODES = ("auto", "symmetric", "asymmetric")


def _has_per_phase_appliance(grid: Grid) -> bool:
    """True if any Load/Generator carries an explicit per-phase nameplate split."""
    return any(
        isinstance(a, (Load, Generator))
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

    Logs the resolved decision and its reason at INFO.
    """
    if mode is None:
        mode = config.get("calculation.symmetry")
    m = str(mode).lower()
    if m not in _VALID_MODES:
        raise ValueError(
            f"calculation symmetry must be one of {_VALID_MODES}; got {mode!r}."
        )

    if m == "symmetric":
        resolved, reason = False, "forced symmetric"
    elif m == "asymmetric":
        resolved, reason = True, "forced asymmetric"
    else:  # auto
        pp_grid = _has_per_phase_appliance(grid)
        pp_op = _has_per_phase_operating_point(operating_point)
        resolved = pp_grid or pp_op
        if not resolved:
            reason = "auto: no per-phase data"
        else:
            src = []
            if pp_grid:
                src.append("appliance *_per_phase_*")
            if pp_op:
                src.append("operating point")
            reason = "auto: per-phase data in " + " + ".join(src)

    logger.info(
        "pgml calculation symmetry: %s (%s).",
        "ASYMMETRIC (per-phase)" if resolved else "SYMMETRIC (balanced split)",
        reason,
    )
    return resolved


def resolve_connection(appliance) -> WindingConnection:
    """Effective connection of a Load/Generator: explicit if set, else config default.

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
    return WindingConnection(config.get(key))


def log_modeling_summary(grid: Grid, *, asymmetric: bool) -> None:
    """INFO-log the FINAL modeling: neutral handling + load connections + symmetry.

    Makes implicit modeling explicit, e.g. that a neutral is being modeled because a
    node carries ``Phase.N`` (so WYE loads there return into the neutral, not ground).
    Call once per assemble/solve after :func:`resolve_asymmetric`.
    """
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

    appliances = [a for a in grid.appliances if isinstance(a, (Load, Generator))]
    if appliances:
        counts: dict[str, int] = {}
        for a in appliances:
            c = resolve_connection(a).value
            counts[c] = counts.get(c, 0) + 1
        logger.info(
            "pgml: %d load/gen connection(s): %s; calculation = %s.",
            len(appliances),
            ", ".join(f"{k}x{v}" for k, v in sorted(counts.items())),
            "ASYMMETRIC (per-phase)" if asymmetric else "SYMMETRIC (balanced split)",
        )


__all__ = [
    "resolve_asymmetric",
    "resolve_connection",
    "log_modeling_summary",
]
