"""Synthesize conductor geometry that reproduces a line's R/X at the fundamental.

The standard feeders (IEEE-33, CIGRE LV) are defined by R/X per km with NO conductor
geometry (pandapower line types carry only R/X/C/type). To exercise the Carson/Deri
geometry path on them, we synthesize a single-conductor earth-return geometry whose
fundamental self-impedance equals the given ``R1 + jX1`` (closed form: GMR sets the
reactance, Rdc the resistance, with a couple of skin-effect fixed-point steps), and
TRACK it via the geometry's :class:`Provenance`. The same synthesized geometry is fed
to both pgml and OpenDSS, so the harmonic comparison is an apples-to-apples Carson
check on a realistic topology.

Capacitance of the synthesized conductor is tiny (and identical on both sides), so it
does not materially change the c=0 feeders. Heights follow the line ``type`` (overhead
vs cable). This is a deliberate positive-sequence-equivalent earth-return synthesis,
not a unique physical reconstruction — hence the provenance note.
"""

from __future__ import annotations

import cmath
import math
from typing import Optional

from pgml.schemas.grid_schema import (
    ConductorPlacement,
    Grid,
    Line,
    LineGeometry,
    Phase,
    Provenance,
    SourceConvention,
)

from .carson import MU0

_DEFAULT_HEIGHT = {"ol": 10.0, "cs": 1.0}  # m: overhead vs cable (Deri needs y>0)
_DEFAULT_RADIUS = 0.0102  # m, typical ACSR-ish (capacitance only; negligible here)


def _self_ze(h: float, rho: float, f: float) -> complex:
    """Deri earth-return self term Ze(i,i) at one conductor (Ω/m)."""
    fw = 2.0 * math.pi * f
    fme = cmath.sqrt(complex(0.0, fw * MU0 / rho))
    hterm = 2.0 * h + 2.0 / fme
    return (1j * fw * MU0 / (2.0 * math.pi)) * cmath.log(hterm)


def _re_zint(rdc: float, f: float) -> float:
    """Real part of the skin-effect internal impedance at frequency f (Ω/m)."""
    alpha = (1 + 1j) * math.sqrt(f * MU0 / rdc)
    if abs(alpha) > 35.0:
        i0i1 = 1.0 + 0j
    else:
        import scipy.special as sp

        i0i1 = sp.iv(0, alpha) / sp.iv(1, alpha)
    return ((1 + 1j) * i0i1 * cmath.sqrt(complex(rdc * f * MU0)) / 2.0).real


def synthesize_line_geometry(
    r1_ohm_per_m: float,
    x1_ohm_per_m: float,
    *,
    f0: float,
    phase: Phase = Phase.A,
    line_type: str = "ol",
    earth_resistivity_ohm_m: float = 100.0,
    height_m: Optional[float] = None,
    radius_m: float = _DEFAULT_RADIUS,
    notes: str = "",
) -> LineGeometry:
    """Single-conductor :class:`LineGeometry` reproducing ``r1 + j x1`` (Ω/m) at ``f0``.

    GMR is set in closed form from the reactance; Rdc from the resistance with a few
    skin-effect fixed-point refinements (clamped to a small positive value if the
    target resistance is below the earth-return floor). Provenance records the origin.
    """
    h = height_m if height_m is not None else _DEFAULT_HEIGHT.get(line_type, 10.0)
    ze = _self_ze(h, earth_resistivity_ohm_m, f0)
    coef = f0 * MU0  # = Fw*mu0/(2*pi), the spacing-reactance coefficient
    gmr = math.exp(-(x1_ohm_per_m - ze.imag) / coef)

    rdc = max(r1_ohm_per_m - ze.real, 1e-7)
    for _ in range(6):  # skin effect makes Re(Zint) > Rdc; correct toward the target
        residual = r1_ohm_per_m - ze.real - _re_zint(rdc, f0)
        rdc = max(rdc + residual, 1e-7)

    return LineGeometry(
        conductors=[
            ConductorPlacement(
                phase=phase,
                x_m=0.0,
                y_m=h,
                gmr_m=gmr,
                radius_m=radius_m,
                r_dc_ohm_per_m=rdc,
                is_neutral=False,
            )
        ],
        earth_resistivity_ohm_m=earth_resistivity_ohm_m,
        provenance=Provenance(
            source_convention=SourceConvention.GEOMETRY,
            notes=(
                notes
                or "synthesized single-conductor earth-return geometry "
                f"reproducing R1={r1_ohm_per_m:.6g}, X1={x1_ohm_per_m:.6g} Ω/m @ {f0} Hz"
            ),
            extra={
                "synth_height_m": str(h),
                "synth_gmr_m": f"{gmr:.6g}",
                "synth_rdc_ohm_per_m": f"{rdc:.6g}",
                # GMR >= radius is physically impossible for a real conductor: it means
                # the target X1 is below the single-conductor earth-return reactance
                # floor (typical of cables / low-X positive-sequence lines). The result
                # still reproduces R1/X1 at f0 and matches OpenDSS on the SAME geometry,
                # but is not a physical conductor — see geometry/CONTEXT.md.
                "synth_unphysical": str(gmr >= radius_m),
            },
        ),
    )


def synthesize_grid_geometry(grid: Grid, *, f0: Optional[float] = None) -> Grid:
    """In place: give every single-phase R/X line a synthesized ``conductor_geometry``.

    Uses each line's diagonal ``R`` and ``X = 2*pi*f0*L`` (Ω/m). Multi-phase lines are
    skipped (the single-conductor synthesis is positive-sequence). Returns ``grid``.
    """
    f0 = float(f0 if f0 is not None else grid.base_frequency_hz)
    n_unphysical = 0
    n_synth = 0
    for ln in grid.branches:
        if not (isinstance(ln, Line) and ln.conductor_geometry is None):
            continue
        if len(ln.from_phases) != 1:
            continue
        r1 = float(ln.series_resistance_ohm_per_m[0][0])
        x1 = 2.0 * math.pi * f0 * float(ln.series_inductance_h_per_m[0][0])
        ltype = (ln.tags or {}).get("pp_type", "ol")
        ln.conductor_geometry = synthesize_line_geometry(
            r1, x1, f0=f0, phase=ln.from_phases[0], line_type=ltype
        )
        n_synth += 1
        if ln.conductor_geometry.provenance.extra.get("synth_unphysical") == "True":
            n_unphysical += 1
    if n_unphysical:
        import warnings

        warnings.warn(
            f"synthesize_grid_geometry: {n_unphysical}/{n_synth} lines have X1 below the "
            "single-conductor earth-return reactance floor, yielding a non-physical "
            "GMR (>= radius). The geometry still reproduces R1/X1 at f0 and matches "
            "OpenDSS on the same geometry, but is not a physical conductor (typical of "
            "cables / low-X positive-sequence feeders). See geometry/CONTEXT.md.",
            stacklevel=2,
        )
    return grid


__all__ = ["synthesize_line_geometry", "synthesize_grid_geometry"]
