"""Split-phase (centre-tapped) service transformers of North-American OpenDSS feeders.

North-American distribution models write a split-phase service transformer as one
single-phase, three-winding ``Transformer``: a primary coil and two equal half
secondaries in series, ``buses=(hv.3, lv.1.0, lv.0.2)`` (the centre tap is the
ground node ``0`` or the neutral conductor ``4``). :func:`read_centre_tapped_unit`
recognises exactly that arrangement and returns its data on the full-secondary
basis that :class:`~pgml.schemas.grid_schema.Transformer` (with its
:class:`~pgml.schemas.grid_schema.CentreTap` block) stores; anything else is a
:class:`~pgml.errors.ConversionError` that names the first deviation.

Impedance reduction (all in percent on the unit's kVA, ``kV_x`` one half winding,
``Z_base(kV) = kV^2 * 1000 / kVA``):

- The star reactances of the three windings are ``X_H = (XHL + XHT - XLT) / 2`` and
  ``X_L = (XHL + XLT - XHT) / 2``, ``X_T = (XHT + XLT - XHL) / 2`` for the two halves.
  :class:`~pgml.schemas.grid_schema.CentreTap` has two equal halves, so each half
  carries ``X_X = (X_L + X_T) / 2 = XLT / 2``; the sum, and with it the
  primary-to-full-secondary impedance, is exact, but a unit with ``XHL != XHT`` (the
  halves differ) loses that difference (``asymmetry`` below). ``%R_H`` and ``%R_X`` are
  the winding resistances.
- One half winding: ``Z_half = (%R_X + j X_X) / 100 * Z_base(kV_x)``, in its own
  coil ohms.
- Primary to the full secondary (X1-X2 shorted, centre tap open), referred to the
  full secondary coil of rating ``2 kV_x``:
  ``Z_a = (%R_H + j X_H) / 100 * Z_base(2 kV_x) + 2 Z_half``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pgml.errors import ConversionError
from pgml.schemas.grid_schema import Phase

__all__ = ["CentreTappedUnit", "read_centre_tapped_unit"]

#: Bus-node indices that can be the centre tap: the ground reference or the neutral.
_CENTRE_NODES = (0, 4)
_PHASE_NODES = (1, 2, 3)
_REL_TOL = 1e-6


@dataclass(frozen=True)
class CentreTappedUnit:
    """A split-phase unit on the full-secondary basis (SI units).

    ``primary_phases`` are the real conductors of the primary coil (one for a coil
    to earth, two for a phase-to-phase coil); ``x1`` / ``x2`` the secondary outer
    conductors; ``centre`` the centre-tap node (0 = ground reference, 4 = neutral
    conductor).
    """

    primary_bus: str
    secondary_bus: str
    primary_phases: tuple[Phase, ...]
    primary_delta: bool
    x1: Phase
    x2: Phase
    centre: int
    s_rated_va: float
    u_rated_from_v: float
    u_rated_to_v: float
    ratio: float
    series_resistance_ohm: float
    series_reactance_ohm: float
    half_resistance_ohm: float
    half_reactance_ohm: float
    noload_loss_pct: float
    imag_pct: float
    xr_constant: bool
    asymmetry: float


def _nodes(bus: str) -> tuple[str, list[int]]:
    name, *rest = bus.split(".")
    return name, [int(p) for p in rest if p.isdigit()]


def _phase(node: int) -> Phase:
    return {1: Phase.A, 2: Phase.B, 3: Phase.C}[node]


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=1e-12)


def read_centre_tapped_unit(dss: Any, name: str) -> CentreTappedUnit:
    """Read the active three-winding transformer as a split-phase unit.

    Parameters
    ----------
    dss:
        The ``opendssdirect`` module, with the transformer ``name`` the active
        element of the ``Transformers`` iterator.
    name:
        Lowercase transformer name (error messages and the text-interface reads).

    Raises
    ------
    ConversionError
        When the unit is not a single-phase transformer with a primary coil and two
        equal half secondaries ``(x, c)`` and ``(c, y)`` in series on one bus, or
        when its windings differ in rating, voltage, resistance or tap, or carry a
        neutral earthing impedance.
    """

    def refuse(why: str) -> ConversionError:
        return ConversionError(
            f"OpenDSS transformer '{name}' is not split-phase: {why}"
        )

    dss.Circuit.SetActiveElement(f"Transformer.{name}")
    if dss.Transformers.NumWindings() != 3 or dss.CktElement.NumPhases() != 1:
        raise refuse("a centre-tapped unit has one phase and three windings.")
    buses = [b.lower() for b in dss.CktElement.BusNames()]
    if len(buses) != 3:
        raise refuse(f"expected 3 terminals, found {len(buses)}.")

    kv, kva, pct_r, tap, delta = [], [], [], [], []
    for w in (1, 2, 3):
        dss.Transformers.Wdg(w)
        kv.append(float(dss.Transformers.kV()))
        kva.append(float(dss.Transformers.kVA()))
        pct_r.append(float(dss.Transformers.R()))
        tap.append(float(dss.Transformers.Tap()))
        delta.append(bool(dss.Transformers.IsDelta()))
        if (
            float(dss.Transformers.Rneut()) > 0.0
            or float(dss.Transformers.Xneut()) != 0.0
        ):
            raise refuse(
                f"winding {w} has a neutral earthing impedance (Rneut/Xneut); the "
                "centre tap is stamped solidly earthed."
            )
    xhl = float(dss.Transformers.Xhl())
    xht = float(dss.Transformers.Xht())
    xlt = float(dss.Transformers.Xlt())

    if delta[1] or delta[2]:
        raise refuse("the half secondaries must be wye windings.")
    if not (_same(kva[0], kva[1]) and _same(kva[0], kva[2])):
        raise refuse(f"winding kVA ratings differ ({kva}).")
    if not _same(kv[1], kv[2]):
        raise refuse(f"the half secondaries differ in voltage ({kv[1]} vs {kv[2]} kV).")
    if not _same(pct_r[1], pct_r[2]):
        raise refuse("the half secondaries differ in %R.")
    if not _same(tap[1], tap[2]):
        raise refuse("the half secondaries differ in tap.")

    p_bus, p_nodes = _nodes(buses[0])
    s2_bus, n2 = _nodes(buses[1])
    s3_bus, n3 = _nodes(buses[2])
    if s2_bus != s3_bus:
        raise refuse("the half secondaries sit on different buses.")
    if len(n2) != 2 or len(n3) != 2:
        raise refuse(
            "the half secondaries need explicit two-node terminals "
            f"(found {buses[1]!r}, {buses[2]!r})."
        )
    centre = n2[1]
    if centre not in _CENTRE_NODES or n3[0] != centre:
        raise refuse(
            "the half secondaries must meet at one centre-tap node "
            f"(ground 0 or neutral 4), as (x.c, c.y); found {buses[1]!r}, {buses[2]!r}."
        )
    x1, x2 = n2[0], n3[1]
    if x1 not in _PHASE_NODES or x2 not in _PHASE_NODES or x1 == x2:
        raise refuse(f"the outer conductors must be two distinct phases ({x1}, {x2}).")

    live = [n for n in p_nodes if n != 0]
    if delta[0]:
        if len(live) != 2 or len(p_nodes) != 2:
            raise refuse("a delta primary needs two phase conductors.")
    else:
        if (
            len(live) != 1
            or len(p_nodes) > 2
            or (len(p_nodes) == 2 and p_nodes[1] != 0)
        ):
            raise refuse(
                "a wye primary needs one phase conductor (earth return) "
                f"found {buses[0]!r}."
            )
    if any(n not in _PHASE_NODES for n in live):
        raise refuse(f"primary conductors must be phases (found {buses[0]!r}).")

    dss.Text.Command(f"? Transformer.{name}.%noloadloss")
    noload = float(dss.Text.Result().strip() or 0.0)
    dss.Text.Command(f"? Transformer.{name}.%imag")
    imag = float(dss.Text.Result().strip() or 0.0)
    dss.Text.Command(f"? Transformer.{name}.xrconst")
    xr_const = dss.Text.Result().strip().lower() in ("yes", "true", "1")

    kv_x = kv[1]
    z_base_x = kv_x**2 * 1e3 / kva[0]
    z_base_full = (2.0 * kv_x) ** 2 * 1e3 / kva[0]
    x_h = (xhl + xht - xlt) / 2.0
    x_x = xlt / 2.0
    r_half = pct_r[1] / 100.0 * z_base_x
    x_half = x_x / 100.0 * z_base_x
    r_a = pct_r[0] / 100.0 * z_base_full + 2.0 * r_half
    x_a = x_h / 100.0 * z_base_full + 2.0 * x_half

    return CentreTappedUnit(
        primary_bus=p_bus,
        secondary_bus=s2_bus,
        primary_phases=tuple(_phase(n) for n in live),
        primary_delta=delta[0],
        x1=_phase(x1),
        x2=_phase(x2),
        centre=centre,
        s_rated_va=kva[0] * 1e3,
        u_rated_from_v=kv[0] * 1e3,
        u_rated_to_v=2.0 * kv_x * 1e3,
        ratio=tap[0] / tap[1],
        series_resistance_ohm=r_a,
        series_reactance_ohm=x_a,
        half_resistance_ohm=r_half,
        half_reactance_ohm=x_half,
        noload_loss_pct=noload,
        imag_pct=imag,
        xr_constant=xr_const,
        asymmetry=abs(xhl - xht) / max(abs(xlt), 1e-12),
    )
