"""Synthesize conductor geometry that reproduces a line's R/X at the fundamental.

The standard feeders (IEEE-33, CIGRE LV) are defined by R/X per km with NO conductor
geometry (pandapower line types carry only R/X/C/type). To exercise the Carson/Deri
geometry path on them, we synthesize a conductor geometry whose Carson impedance equals
the line's sequence data at the fundamental: a SINGLE-conductor earth-return geometry
reproducing ``R1 + jX1`` for 1-phase lines (closed form: GMR sets the reactance, Rdc the
resistance), and an equilateral THREE-conductor geometry reproducing ``Z1`` and the
zero-sequence reactance ``X0`` for 3-phase lines (GMR + spacing fitted by Newton on the
Carson forward; ``R0`` follows from the earth return). Each is TRACKED via the geometry's
:class:`Provenance`. The same synthesized geometry is fed to both pgml and OpenDSS, so the
harmonic comparison is an apples-to-apples Carson check on a realistic topology — at every
order, including the triplen / zero-sequence orders for 3-phase lines.

Capacitance of the synthesized conductor is tiny (and identical on both sides), so it
does not materially change the c=0 feeders. Heights follow the line ``type`` (overhead
vs cable). This is a deliberate positive-sequence-equivalent earth-return synthesis,
not a unique physical reconstruction — hence the provenance note.
"""

from __future__ import annotations

import cmath
import math
from typing import Optional

from pgml.config import get as _cfg
from pgml.errors import InputError, ModelingError
from pgml.schemas.grid_schema import (
    AnalyticParam,
    ConductorPlacement,
    Grid,
    Line,
    LineGeometry,
    Phase,
    Provenance,
    ResistanceFrequencyModel,
    SourceConvention,
)

from .carson import MU0

# Defaults sourced from the top-level config (`pgml.config` / defaults.yaml).
_DEFAULT_HEIGHT = {  # m: overhead vs cable (Deri needs y>0)
    "ol": _cfg("line.conductor.height_overhead_m"),
    "cs": _cfg("line.conductor.height_cable_m"),
}
_DEFAULT_RADIUS = _cfg("line.conductor.radius_m")  # m, ~ACSR (capacitance only)
_DEFAULT_EARTH_RHO = _cfg("line.earth_return.resistivity_ohm_m")  # ohm*m


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
    earth_resistivity_ohm_m: float = _DEFAULT_EARTH_RHO,
    height_m: Optional[float] = None,
    radius_m: float = _DEFAULT_RADIUS,
    notes: str = "",
) -> LineGeometry:
    """Single-conductor :class:`LineGeometry` reproducing ``r1 + j x1`` (Ω/m) at ``f0``.

    GMR is set in closed form from the reactance; Rdc from the resistance with a few
    skin-effect fixed-point refinements (clamped to a small positive value if the
    target resistance is below the earth-return floor). Provenance records the origin.
    Defaults (radius, heights, earth resistivity) come from ``pgml.config``.
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


def _seq_from_phase_matrix(mat, f0: float) -> tuple[float, float]:
    """Symmetric (transposed) sequence values from a 3x3 R or L matrix.

    Returns ``(seq1, seq0)`` where ``seq1 = self - mutual`` and ``seq0 = self + 2*mutual``
    using the mean diagonal (self) and mean off-diagonal (mutual). For an inductance
    matrix the caller multiplies by ``2*pi*f0`` to get a reactance.
    """
    diag = [_float0(mat[i][i]) for i in range(3)]
    off = [_float0(mat[i][j]) for i in range(3) for j in range(3) if i != j]
    self_ = sum(diag) / 3.0
    mutual = sum(off) / 6.0
    return self_ - mutual, self_ + 2.0 * mutual


def _equilateral_xy(spacing: float, height: float):
    """Coordinates of an equilateral triangle of side ``spacing``, base at ``height``."""
    apex = height + spacing * math.sqrt(3.0) / 2.0
    xs = [-spacing / 2.0, spacing / 2.0, 0.0]
    ys = [height, height, apex]
    return xs, ys


def _carson_sequence_z(xs, ys, gmr, rdc, radius, rho, f0):
    """Carson (Z0, Z1) of an equilateral 3-conductor geometry at ``f0`` (complex Ω/m)."""
    import torch

    from .carson import line_constants
    from .sequence import sequence_impedances

    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    g = torch.full((3,), float(gmr), dtype=torch.float64)
    r = torch.full((3,), float(rdc), dtype=torch.float64)
    rad = torch.full((3,), float(radius), dtype=torch.float64)
    z, _c = line_constants(
        x, y, g, r, rad, float(rho), torch.tensor([f0], dtype=torch.float64), 3
    )
    z0, z1, _z2 = sequence_impedances(z[0])  # [3,3] at the single frequency
    return complex(z0.item()), complex(z1.item())


def synthesize_three_phase_geometry(
    r1_ohm_per_m: float,
    x1_ohm_per_m: float,
    x0_ohm_per_m: float,
    *,
    f0: float,
    phases=(Phase.A, Phase.B, Phase.C),
    line_type: str = "ol",
    earth_resistivity_ohm_m: float = _DEFAULT_EARTH_RHO,
    height_m: Optional[float] = None,
    radius_m: float = _DEFAULT_RADIUS,
    notes: str = "",
) -> LineGeometry:
    """Three-conductor :class:`LineGeometry` reproducing ``Z1`` and ``X0`` at ``f0``.

    An equilateral 3-conductor arrangement (identical conductors, pairwise spacing
    ``D``, base height ``height_m``). The geometric mean radius ``GMR`` and spacing
    ``D`` are fitted (Newton on the Carson forward) so the synthesized line's
    positive-sequence reactance ``X1`` and zero-sequence reactance ``X0`` match the
    targets; ``Rdc`` is fitted to ``R1``. The zero-sequence RESISTANCE ``R0`` then
    follows from the Carson earth-return physics (``R0 ≈ R1 + 3·R_earth(f0)``) — it is
    not a free target, so a sequence dataset's assumed ``R0`` is replaced by the
    geometry's physical value, recorded in the provenance.

    The SAME geometry can be fed to both pgml and OpenDSS, so the harmonic comparison
    is an apples-to-apples Carson check (earth-return on both ``Z1`` and ``Z0`` at every
    harmonic, including the triplen / zero-sequence orders). Defaults come from
    ``pgml.config``.
    """
    h = height_m if height_m is not None else _DEFAULT_HEIGHT.get(line_type, 10.0)
    rho = earth_resistivity_ohm_m
    coef = f0 * MU0  # reactance per unit ln (MU0*f0)

    # Rdc seed from the positive-sequence skin fit (earth-free); refined in the loop.
    import torch

    from .sequence import fit_equivalent_rdc

    rdc = float(
        fit_equivalent_rdc(r1_ohm_per_m, f0, torch.tensor([f0], dtype=torch.float64))
    )

    # Newton on (ln D, ln GMR) for (X1, X0); constant analytic Jacobian
    # J = [[dX1/dlnD, dX1/dlnGMR],[dX0/dlnD, dX0/dlnGMR]] = coef*[[1,-1],[-2,-1]].
    ln_gmr = math.log(_cfg("line.conductor.gmr_over_radius") * radius_m)
    ln_d = math.log(_cfg("line.conductor.phase_spacing_m"))
    inv_det = 1.0 / (-3.0 * coef * coef)
    for _ in range(40):
        xs, ys = _equilateral_xy(math.exp(ln_d), h)
        z0, z1 = _carson_sequence_z(xs, ys, math.exp(ln_gmr), rdc, radius_m, rho, f0)
        e1 = z1.imag - x1_ohm_per_m
        e0 = z0.imag - x0_ohm_per_m
        # [dlnD, dlnGMR] = -J^{-1} e ; J^{-1} = inv_det * coef*[[-1,1],[2,1]]
        d_lnd = -inv_det * coef * (-1.0 * e1 + 1.0 * e0)
        d_lngmr = -inv_det * coef * (2.0 * e1 + 1.0 * e0)
        ln_d += d_lnd
        ln_gmr += d_lngmr
        # refine Rdc against the synthesized R1 (earth cancels in Z1.real)
        rdc = max(rdc + (r1_ohm_per_m - z1.real), 1e-9)
        if abs(e1) < 1e-12 and abs(e0) < 1e-12:
            break

    gmr = math.exp(ln_gmr)
    spacing = math.exp(ln_d)
    xs, ys = _equilateral_xy(spacing, h)
    z0_fit, z1_fit = _carson_sequence_z(xs, ys, gmr, rdc, radius_m, rho, f0)

    conductors = [
        ConductorPlacement(
            phase=phases[i],
            x_m=xs[i],
            y_m=ys[i],
            gmr_m=gmr,
            radius_m=radius_m,
            r_dc_ohm_per_m=rdc,
            is_neutral=False,
        )
        for i in range(3)
    ]
    return LineGeometry(
        conductors=conductors,
        earth_resistivity_ohm_m=rho,
        provenance=Provenance(
            source_convention=SourceConvention.GEOMETRY,
            notes=(
                notes
                or "synthesized equilateral 3-conductor Carson geometry reproducing "
                f"R1={r1_ohm_per_m:.6g}, X1={x1_ohm_per_m:.6g}, X0={x0_ohm_per_m:.6g} "
                f"Ω/m @ {f0} Hz (R0 follows from earth return)"
            ),
            extra={
                "synth_kind": "three_phase_equilateral",
                "synth_height_m": f"{h:.6g}",
                "synth_spacing_m": f"{spacing:.6g}",
                "synth_gmr_m": f"{gmr:.6g}",
                "synth_rdc_ohm_per_m": f"{rdc:.6g}",
                "synth_r0_ohm_per_m": f"{z0_fit.real:.6g}",
                # GMR >= radius means the target X1/X0 are below the geometric floor for
                # this spacing (non-physical conductor; still matches OpenDSS on the SAME
                # geometry). See geometry/CONTEXT.md.
                "synth_unphysical": str(gmr >= radius_m),
            },
        ),
    )


def synthesize_grid_geometry(grid: Grid, *, f0: Optional[float] = None) -> Grid:
    """In place: give every R/X line a synthesized ``conductor_geometry``.

    Single-phase lines get a single-conductor earth-return geometry reproducing
    ``R1 + jX1``; 3-phase lines get an equilateral 3-conductor geometry reproducing
    ``Z1`` and ``X0`` (:func:`synthesize_three_phase_geometry`). The SAME geometry is
    fed to both pgml and OpenDSS for an apples-to-apples Carson harmonic comparison.
    Other phase counts (2-phase) are skipped. Returns ``grid``.
    """
    f0 = float(f0 if f0 is not None else grid.base_frequency_hz)
    n_unphysical = 0
    n_synth = 0
    for ln in grid.branches:
        if not (isinstance(ln, Line) and ln.conductor_geometry is None):
            continue
        p = len(ln.from_phases)
        ltype = (ln.tags or {}).get("pp_type", "ol")
        if p == 1:
            r1 = float(ln.series_resistance_ohm_per_m[0][0])
            x1 = 2.0 * math.pi * f0 * float(ln.series_inductance_h_per_m[0][0])
            ln.conductor_geometry = synthesize_line_geometry(
                r1, x1, f0=f0, phase=ln.from_phases[0], line_type=ltype
            )
        elif p == 3:
            r1, _r0 = _seq_from_phase_matrix(ln.series_resistance_ohm_per_m, f0)
            l1, l0 = _seq_from_phase_matrix(ln.series_inductance_h_per_m, f0)
            two_pi_f0 = 2.0 * math.pi * f0
            ln.conductor_geometry = synthesize_three_phase_geometry(
                r1,
                two_pi_f0 * l1,
                two_pi_f0 * l0,
                f0=f0,
                phases=ln.from_phases,
                line_type=ltype,
            )
        else:
            continue
        n_synth += 1
        if ln.conductor_geometry.provenance.extra.get("synth_unphysical") == "True":
            n_unphysical += 1
    if n_unphysical:
        import warnings

        warnings.warn(
            f"synthesize_grid_geometry: {n_unphysical}/{n_synth} lines have a reactance "
            "below the earth-return / spacing floor, yielding a non-physical GMR "
            "(>= radius). The geometry still reproduces the target Z at f0 and matches "
            "OpenDSS on the same geometry, but is not a physical conductor (typical of "
            "cables / low-X feeders). See geometry/CONTEXT.md.",
            stacklevel=2,
        )
    return grid


def _float0(v) -> float:
    """Representative python float from a possibly-tensor scalar (skin-curve shape only)."""
    if hasattr(v, "detach"):
        return float(v.detach().reshape(-1)[0])
    return float(v)


def positive_sequence_resistance_model(
    r1_ohm_per_m: float, *, f0: float
) -> ResistanceFrequencyModel:
    """A :class:`ResistanceFrequencyModel` carrying the positive-sequence skin curve.

    Encodes the ``carson_skin_multiplier`` analytic law (evaluated differentiably in
    assembly via :func:`pgml.geometry.sequence.skin_resistance_multiplier`): the Bessel
    ``I0/I1`` internal-resistance growth WITHOUT the earth-return floor. ``base_value``
    is the f0 multiplier (1.0); ``params`` carry the representative ``R1`` and ``f0``.
    """
    return ResistanceFrequencyModel(
        multiplier=AnalyticParam(
            law="carson_skin_multiplier",
            base_value=1.0,
            params={"r1_ohm_per_m": float(r1_ohm_per_m), "f0_hz": float(f0)},
        )
    )


def _line_representative_r1(ln: Line) -> float:
    """Mean of the diagonal series resistance (Ω/m) — the representative ``R1``."""
    diag = ln.series_resistance_ohm_per_m
    p = len(ln.from_phases)
    return sum(_float0(diag[i][i]) for i in range(p)) / p


def _apply_positive_sequence_line(ln: Line, *, f0: float, skin: bool) -> None:
    """Tag a single R/X line with the positive-sequence (Z1) harmonic model."""
    if skin:
        ln.resistance_frequency = positive_sequence_resistance_model(
            _line_representative_r1(ln), f0=f0
        )
    else:
        ln.resistance_frequency = ResistanceFrequencyModel()  # constant 1.0


def _apply_sequence_aware_line(ln: Line, *, skin: bool, coeff: float) -> None:
    """Tag a single 3-phase R/X line with the sequence-aware (Z1+Z0) harmonic model."""
    ln.tags = dict(ln.tags or {})
    ln.tags["harmonic_line_model"] = "sequence_aware"
    ln.tags["seq_skin"] = "true" if skin else "false"
    ln.tags["seq_earth_coeff"] = repr(coeff)


def apply_positive_sequence_harmonic_model(
    grid: Grid, *, f0: Optional[float] = None, skin: Optional[bool] = None
) -> Grid:
    """In place: give every R/X line the positive-sequence harmonic model (no earth floor).

    The corrected positive-sequence harmonic line model: the EXPLICIT R/L/C path
    already scales the geometric reactance ``X(h)=X1*h`` (constant ``L``) with NO earth
    return; this adds the physically-correct skin-effect growth on ``R`` via the line's
    :class:`ResistanceFrequencyModel` (``carson_skin_multiplier``). Unlike
    :func:`synthesize_grid_geometry`, it does NOT reverse-synthesise a single-conductor
    earth-return geometry, so it never hits the GMR floor and stays well defined for
    cables / low-X feeders. Lines that already carry a ``conductor_geometry`` (genuine
    geometry -> full Carson) are left untouched. ``skin`` defaults to the config value
    ``line.harmonic_model.skin_effect``; ``skin=False`` keeps R constant (naive model).
    Returns ``grid``.
    """
    f0 = float(f0 if f0 is not None else grid.base_frequency_hz)
    skin = _cfg("line.harmonic_model.skin_effect") if skin is None else bool(skin)
    for ln in grid.branches:
        if not (isinstance(ln, Line) and ln.conductor_geometry is None):
            continue
        _apply_positive_sequence_line(ln, f0=f0, skin=skin)
    return grid


def apply_sequence_aware_harmonic_model(
    grid: Grid,
    *,
    skin: Optional[bool] = None,
    earth_resistance_coeff: Optional[float] = None,
) -> Grid:
    """In place: tag every 3-phase R/X line for the sequence-aware harmonic model.

    For ASYMMETRIC (unbalanced) 4-wire studies. Assembly decomposes each tagged line's
    reference-frequency phase matrix ``Z_abc(f0)`` into ``Z1``/``Z0`` and
    frequency-corrects each sequence separately: ``Z1`` stays earth-free (``X∝h`` + skin)
    while ``Z0`` carries the Carson earth-return resistance damping (see
    :func:`pgml.geometry.sequence.sequence_aware_phase_z`). The zero-sequence path that an
    unbalanced/neutral-return current excites is therefore modelled with its earth-return
    damping, which a balanced positive-sequence current never sees.

    Needs a full 3x3 R/L matrix (the off-diagonal mutuals are what carry ``Z0``); a
    diagonal matrix gives ``Z0 = Z1`` plus the universal earth term. Single-/two-phase and
    geometry-defined lines are left untouched. ``skin`` and ``earth_resistance_coeff``
    default to the config (``line.harmonic_model.skin_effect`` /
    ``line.earth_return.resistance_coeff_ohm_per_m_per_hz``); pass them to override.
    Returns ``grid``.
    """
    skin = _cfg("line.harmonic_model.skin_effect") if skin is None else bool(skin)
    coeff = (
        _cfg("line.earth_return.resistance_coeff_ohm_per_m_per_hz")
        if earth_resistance_coeff is None
        else float(earth_resistance_coeff)
    )
    for ln in grid.branches:
        if not (isinstance(ln, Line) and ln.conductor_geometry is None):
            continue
        if len(ln.from_phases) != 3:
            continue
        _apply_sequence_aware_line(ln, skin=skin, coeff=coeff)
    return grid


def _line_has_explicit_harmonic_model(ln: Line) -> bool:
    """Whether a line already carries a user-set harmonic model (precedence 1, explicit)."""
    if (ln.tags or {}).get("harmonic_line_model"):
        return True
    rfm = getattr(ln, "resistance_frequency", None)
    mult = getattr(rfm, "multiplier", None)
    # A non-default (non-constant, or constant != 1.0) resistance-frequency law counts.
    if getattr(mult, "kind", None) == "constant":
        return float(getattr(mult, "value", 1.0)) != 1.0
    return mult is not None


def apply_default_harmonic_model(
    grid: Grid, *, f0: Optional[float] = None, skin: Optional[bool] = None
) -> Grid:
    """In place: apply the CONFIG-DEFAULT harmonic line model to each R/X line.

    The single deliberate entry point that turns the documented defaults in
    ``pgml.config`` (``line.harmonic_model.three_phase`` / ``.single_phase``) into actual
    per-line models — so the choice is explicit and config-sourced, never silently
    implicit at solve time. Per the precedence contract, a line that ALREADY carries an
    explicit harmonic model (a ``harmonic_line_model`` tag or a non-default
    ``resistance_frequency``) or a ``conductor_geometry`` is left untouched; only the
    remaining R/X lines get the config default (``sequence_aware`` for 3-phase,
    ``positive_sequence`` for 1-/2-phase by default). ``skin`` defaults to
    ``line.harmonic_model.skin_effect``. Returns ``grid``.
    """
    f0 = float(f0 if f0 is not None else grid.base_frequency_hz)
    skin = _cfg("line.harmonic_model.skin_effect") if skin is None else bool(skin)
    coeff = _cfg("line.earth_return.resistance_coeff_ohm_per_m_per_hz")
    model_3ph = _cfg("line.harmonic_model.three_phase")
    model_other = _cfg("line.harmonic_model.single_phase")

    for ln in grid.branches:
        if not (isinstance(ln, Line) and ln.conductor_geometry is None):
            continue
        if _line_has_explicit_harmonic_model(ln):  # precedence 1: explicit wins
            continue
        model = model_3ph if len(ln.from_phases) == 3 else model_other
        if model == "sequence_aware":
            if len(ln.from_phases) != 3:
                raise ModelingError(
                    f"Line {ln.id}: sequence_aware needs 3 phases "
                    f"(config single_phase={model_other!r} should not be sequence_aware)."
                )
            _apply_sequence_aware_line(ln, skin=skin, coeff=coeff)
        elif model == "positive_sequence":
            _apply_positive_sequence_line(ln, f0=f0, skin=skin)
        elif model == "naive":
            _apply_positive_sequence_line(ln, f0=f0, skin=False)
        elif model == "none":
            continue
        else:
            raise InputError(
                f"Unknown harmonic line model {model!r} in config "
                "(expected sequence_aware | positive_sequence | naive | none)."
            )
    return grid


__all__ = [
    "synthesize_line_geometry",
    "synthesize_three_phase_geometry",
    "synthesize_grid_geometry",
    "positive_sequence_resistance_model",
    "apply_positive_sequence_harmonic_model",
    "apply_sequence_aware_harmonic_model",
    "apply_default_harmonic_model",
]
