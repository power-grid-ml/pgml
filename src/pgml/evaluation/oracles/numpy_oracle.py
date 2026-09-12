"""Pure-numpy harmonic oracle — a REGRESSION guard on pgml's Y-bus formulas.

NOT independent validation: the formulas are deliberately pgml's own (and the
3-phase transformer stamp imports ``pgml.assembly._transformer``'s incidence
builder directly), so a bug in those formulas passes this parity check by
construction. Independent physics validation lives in the live-OpenDSS and
pandapower / power-grid-model oracle tests.

This module provides two harmonic oracles that operate without a live OpenDSS process:

- :func:`numpy_harmonic_profiles` — single-phase feeder oracle (lines + source Norton,
  R const / X∝h), returns :class:`~pgml.evaluation.data.HarmonicProfile` objects.

- :func:`numpy_harmonic_voltages` — full multi-element oracle (lines, switches,
  transformers, source Norton) for both single-phase and three-phase grids.  Reimplements
  pgml's EXACT Y-bus formulas (R const / X∝h) giving machine-precision parity (~1e-13 V)
  vs :func:`pgml.solver.solve_harmonic_flow`.

The two functions share the private helpers :func:`_build_numpy_ybus` (stamps all passive
network elements) and :func:`_stamp_transformer_numpy` (mirrors ``pgml.assembly._transformer``
exactly).

All heavy imports (pgml modules) are deferred to function scope where practical.
``numpy`` is imported at module level (core dependency).
"""

from __future__ import annotations

import cmath
import math
from typing import Optional, Sequence

import numpy as np

from pgml.schemas.grid_schema import (
    Grid,
    InjectionAppliance,
    Line,
    Load,
    Phase,
    Source,
    StaticSpectrum,
    Switch,
    Transformer,
)

from pgml.evaluation._util import to_float
from pgml.evaluation.data import HarmonicProfile
from pgml.evaluation.topology import distance_from_slack


# ---------------------------------------------------------------------------
# ideal (zero-impedance) branches: the oracle's own dense bus fusion
# ---------------------------------------------------------------------------
def _is_ideal_branch(b) -> bool:
    """A closed branch with exactly zero series impedance (an ideal conductor).

    Such a branch has no primitive admittance, so the oracle does not stamp it: its two
    terminals are the SAME electrical node, which :func:`fusion_prolongation` expresses.
    """
    if not getattr(b, "in_service", True):
        return False
    if isinstance(b, Switch):
        if not b.closed:
            return False
        return to_float(b.resistance_ohm) == 0.0 and to_float(b.inductance_h) == 0.0
    if isinstance(b, Line):
        if b.conductor_geometry is not None:
            return False
        if to_float(b.length_m) == 0.0:
            return True
        r = b.series_resistance_ohm_per_m
        ell = b.series_inductance_h_per_m
        if r is None or ell is None:
            return False
        flat_r = [to_float(x) for row in r for x in row]
        flat_l = [to_float(x) for row in ell for x in row]
        return not any(flat_r) and not any(flat_l)
    return False


def fusion_prolongation(grid: Grid, index) -> Optional[np.ndarray]:
    """Dense 0/1 prolongation ``P`` ``[N, M]`` merging the ideal branches' terminal rows.

    The textbook form of exact bus fusion, written out for the oracle: with ``V = P v``
    the reduced system is ``(P^T Y P) v = P^T I``, which is what
    :func:`solve_with_fusion` solves. ``None`` when the grid has no ideal branch.

    pgml's solver reaches the same system through a many-to-one row index instead of a
    matrix (:func:`pgml.assembly.fusion_map`), so this is an independent implementation
    of the same algebra.
    """
    pairs = []
    for b in grid.branches:
        if not _is_ideal_branch(b):
            continue
        for pf, pt in zip(b.from_phases, b.to_phases):
            pairs.append(
                (
                    index.row(int(b.from_node), pf),
                    index.row(int(b.to_node), pt),
                )
            )
    if not pairs:
        return None
    parent = list(range(index.size))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, c in pairs:
        ra, rc = find(a), find(c)
        if ra != rc:
            parent[rc] = ra
    col_of: dict[int, int] = {}
    cols = []
    for row in range(index.size):
        root = find(row)
        if root not in col_of:
            col_of[root] = len(col_of)
        cols.append(col_of[root])
    p_mat = np.zeros((index.size, len(col_of)), dtype=complex)
    p_mat[np.arange(index.size), cols] = 1.0
    return p_mat


def solve_with_fusion(y: np.ndarray, i: np.ndarray, p_mat) -> np.ndarray:
    """Solve ``Y V = I``, collapsing the fused rows when ``p_mat`` is given."""
    if p_mat is None:
        return np.linalg.solve(y, i)
    return p_mat @ np.linalg.solve(p_mat.conj().T @ y @ p_mat, p_mat.conj().T @ i)


def _resolve_spectrum(app, harmonic_injection):
    if harmonic_injection is not None and app.id in harmonic_injection:
        return dict(harmonic_injection[app.id])
    spec = getattr(app, "spectrum", None)
    if isinstance(spec, StaticSpectrum):
        return {
            c.order: (to_float(c.magnitude_pu), to_float(c.phase_deg))
            for c in spec.spectrum.components
        }
    return None


def _stamp_transformer_numpy(
    y: np.ndarray,
    b: Transformer,
    h: int,
    w0: float,
    fr_rows: list,
    to_rows: list,
    p: int,
) -> None:
    """Stamp a two-winding transformer into the numpy Y-bus (in-place).

    Mirrors :mod:`pgml.assembly._transformer` exactly:

    - **P == 1** (single-phase / positive-sequence equivalent): the vector group is
      folded into a complex line-to-line ratio ``t = (u_from/u_to) · tap_mag ·
      e^{j·shift_deg}`` and the textbook off-nominal-tap pi is applied::

          Y_ff = y_se / |t|² + y_m,   Y_ft = −y_se / conj(t)
          Y_tf = −y_se / t,            Y_tt = y_se

    - **P == 3** (phase-domain): the nodal block is built as ``Nᵀ·Y_winding·N``
      where ``N = blockdiag(N_hv, N_lv)`` (wye-grounded → ``I₃``, delta → ``M`` or
      ``Mᵀ`` per clock), and the coil turns ratio is::

          τ = (coil_from / coil_to) · tap_mag

      with delta coils rated at the line-to-line voltage and wye coils at
      ``u_rated / √3``.  The magnetizing shunt ``y_m = G_m + jB_m`` is added to the
      HV terminal phase diagonal outside the incidence transform — identical to pgml.

    ``y``, ``fr_rows``, ``to_rows``, and ``p`` are passed in to avoid re-computing
    them; ``b`` supplies all other transformer parameters.
    """
    from pgml.assembly._transformer import block_incidence, resolve_vector_group

    r_t = to_float(b.series_resistance_ohm)
    l_t = to_float(b.series_inductance_h)
    z_se = r_t + 1j * h * w0 * l_t
    y_se = 1.0 / z_se
    gm = to_float(b.magnetizing_conductance_s)
    lm = b.magnetizing_inductance_h
    bm = -1.0 / (h * w0 * to_float(lm)) if lm is not None else 0.0
    ym = complex(gm, bm)

    u_from = to_float(b.u_rated_from_v)
    u_to = to_float(b.u_rated_to_v)
    tap_mag = to_float(b.tap.ratio_magnitude)

    if p == 1:
        # Single-phase equivalent: nominal ratio = u_from/u_to (LL/LL), folded
        # together with the off-nominal tap and the clock phase shift. The
        # stored leakage is TO-coil-referred; the scalar pi consumes the
        # line-to-line equivalent (y_LL = 3·y_coil for a delta TO winding).
        vg1 = resolve_vector_group(b, n_phases=1)
        y_se = (3.0 if vg1.to_side.kind == "delta" else 1.0) * y_se
        shift_rad = to_float(b.tap.shift_deg) * math.pi / 180.0
        t = (u_from / u_to) * tap_mag * cmath.exp(1j * shift_rad)
        abs_t2 = abs(t) ** 2
        y_ff = y_se / abs_t2 + ym
        y_ft = -y_se / complex(t).conjugate()
        y_tf = -y_se / t
        y_tt = y_se
        fr = fr_rows[0]
        to = to_rows[0]
        y[fr, fr] += y_ff
        y[fr, to] += y_ft
        y[to, fr] += y_tf
        y[to, to] += y_tt
    else:
        # Phase-domain: winding-incidence primitive Nᵀ Y_winding N.
        import torch as _torch

        vg = resolve_vector_group(b, n_phases=p)

        # Coil turns ratio: delta → LL voltage, wye/zigzag → LN voltage = u/√3
        # (mirrors pgml.assembly._transformer.nominal_turns_ratio).
        sqrt3 = math.sqrt(3.0)
        coil_from = u_from if vg.from_side.kind == "delta" else u_from / sqrt3
        coil_to = u_to if vg.to_side.kind == "delta" else u_to / sqrt3
        tau = (coil_from / coil_to) * tap_mag

        # Block incidence N [2P, 2P] (constant, real).
        rdt = _torch.float64
        n_blk = (
            block_incidence(vg, p, rdt, _torch.device("cpu")).numpy().astype(complex)
        )

        # 6×6 winding primitive Y_winding.
        eye_p = np.eye(p, dtype=complex)
        y_w = np.block(
            [
                [(y_se / tau**2) * eye_p, -(y_se / tau) * eye_p],
                [-(y_se / tau) * eye_p, y_se * eye_p],
            ]
        )
        # Nodal block: Nᵀ Y_winding N  [2P, 2P].
        y_node = n_blk.T @ y_w @ n_blk

        # Magnetizing shunt on the HV terminal diagonal (outside incidence).
        y_node[:p, :p] += ym * eye_p

        # Scatter into global Y.
        all_rows = list(fr_rows) + list(to_rows)
        for i in range(2 * p):
            for j in range(2 * p):
                y[all_rows[i], all_rows[j]] += y_node[i, j]


def _numpy_skin_multiplier(r1: float, f0: float, f: float) -> float:
    """Skin-effect resistance multiplier ``m(f)`` of a conductor whose ``R(f0) = r1``.

    An INDEPENDENT formulation of ``pgml.geometry.sequence.skin_resistance_multiplier``:
    the internal impedance of a round conductor is
    ``Zint = (1+j)*sqrt(Rdc*f*mu0)/2 * I0(alpha)/I1(alpha)`` with
    ``alpha = (1+j)*sqrt(f*mu0/Rdc)``, evaluated here with ``scipy.special.iv`` instead
    of the differentiable continued fraction, and ``Rdc`` found by the same contraction
    (``Re(Zint(Rdc, f0)) = r1``). ``m(f) = Re(Zint(f))/Re(Zint(f0))``, so ``m(f0) = 1``.
    """
    import scipy.special as sp

    mu0 = 4.0e-7 * math.pi

    def re_zint(rdc: float, freq: float) -> float:
        alpha = (1.0 + 1.0j) * math.sqrt(freq * mu0 / rdc)
        i0i1 = 1.0 + 0.0j if abs(alpha) > 35.0 else sp.iv(0, alpha) / sp.iv(1, alpha)
        return ((1.0 + 1.0j) * i0i1 * cmath.sqrt(complex(rdc * freq * mu0)) / 2.0).real

    rdc = r1
    for _ in range(40):
        rdc = max(rdc + (r1 - re_zint(rdc, f0)), 1e-12)
    return re_zint(rdc, f) / re_zint(rdc, f0)


def _numpy_line_series_z(b: Line, h: int, f0: float) -> np.ndarray:
    """Series impedance matrix ``Z(h)`` ``[P, P]`` (Ohm) of one line, per its model.

    An INDEPENDENT transcription of the line models of ``pgml.assembly.ybus``, written
    straight from the equations in ``docs/pgml/modeling/harmonic-line-model.md``:

    - ``None`` / ``naive``: ``Z(h) = R + j*h*2*pi*f0*L``.
    - ``positive_sequence``: the skin multiplier scales the CONDUCTOR part of ``R``
      (diagonal minus mean mutual); the mutual entries, which carry the earth return,
      are left alone.
    - ``sequence_aware``: ``Z_abc(f0)`` is split into ``Z1``/``Z0``, each frequency
      corrected on its own (``Z1`` earth-free; ``Z0`` with the Carson earth-return
      resistance and the configured ``X0`` law) and recombined.
    """
    from pgml import defaults as _d

    w0 = 2.0 * math.pi * f0
    length = to_float(b.length_m)
    p = len(b.from_phases)
    r_mat = (
        np.array(
            [
                [to_float(b.series_resistance_ohm_per_m[i][j]) for j in range(p)]
                for i in range(p)
            ]
        )
        * length
    )
    l_mat = (
        np.array(
            [
                [to_float(b.series_inductance_h_per_m[i][j]) for j in range(p)]
                for i in range(p)
            ]
        )
        * length
    )
    model = b.harmonic_line_model
    skin = b.harmonic_skin_effect
    if skin is None:
        skin = bool(_d.get("line.harmonic_model.skin_effect"))

    if model == "sequence_aware":
        if p != 3:
            raise ValueError(f"Line {b.id}: sequence_aware needs 3 phases.")
        z = r_mat + 1j * w0 * l_mat
        zs = np.mean(np.diag(z))
        zm = (z.sum() - np.trace(z)) / 6.0
        z1, z0 = zs - zm, zs + 2.0 * zm
        er = b.earth_return
        coeff = getattr(er, "resistance_coeff_ohm_per_m_per_hz", None)
        if coeff is None:
            coeff = _d.get("line.earth_return.resistance_coeff_ohm_per_m_per_hz")
        kx = getattr(er, "reactance_coeff_ohm_per_m_per_hz", None)
        if kx is None:
            kx = _d.get("line.earth_return.reactance_coeff_ohm_per_m_per_hz")
        law = getattr(er, "x0_frequency", None) or _d.get(
            "line.earth_return.x0_frequency"
        )
        expo = getattr(er, "x0_exponent", None)
        if expo is None:
            expo = _d.get("line.earth_return.x0_exponent")
        inc = getattr(er, "r0_includes_earth_return", None)
        if inc is None:
            inc = _d.get("line.zero_sequence.r0_includes_earth_return")
        coeff, kx, expo = (
            to_float(coeff) * length,
            to_float(kx) * length,
            to_float(expo),
        )

        m1 = _numpy_skin_multiplier(z1.real, f0, h * f0) if skin else 1.0
        z1_h = complex(z1.real * m1, z1.imag * h)
        r0_cond = z0.real - 3.0 * coeff * f0 if inc else z0.real
        r_earth = 3.0 * coeff * (h * f0) if inc else 3.0 * coeff * (h * f0 - f0)
        m0 = _numpy_skin_multiplier(r0_cond, f0, h * f0) if skin else 1.0
        x0_h = z0.imag * h**expo
        if law == "carson_sublinear":
            x0_h -= 1.5 * kx * f0 * h * math.log(h)
        z0_h = complex(r0_cond * m0 + r_earth, x0_h)
        z_self, z_mut = (z0_h + 2.0 * z1_h) / 3.0, (z0_h - z1_h) / 3.0
        return np.where(np.eye(3, dtype=bool), z_self, z_mut)

    mult = 1.0
    if model == "positive_sequence" and skin:
        self_ = np.mean(np.diag(r_mat))
        mutual = 0.0 if p == 1 else (r_mat.sum() - np.trace(r_mat)) / (p * (p - 1))
        mult = _numpy_skin_multiplier(self_ - mutual, f0, h * f0)
    elif model is None and b.resistance_frequency is not None:
        law = getattr(b.resistance_frequency.multiplier, "law", None)
        if law == "carson_skin_multiplier":
            params = b.resistance_frequency.multiplier.params
            mult = to_float(b.resistance_frequency.multiplier.base_value) * (
                _numpy_skin_multiplier(
                    to_float(params["r1_ohm_per_m"]) * length,
                    to_float(params["f0_hz"]),
                    h * f0,
                )
            )
        elif getattr(b.resistance_frequency.multiplier, "kind", None) == "constant":
            mult = to_float(b.resistance_frequency.multiplier.value)
    r_earth_mat = np.zeros_like(r_mat)
    if p > 1 and mult != 1.0:
        off = r_mat * (1.0 - np.eye(p))
        r_earth_mat = off + np.diag(off.sum(axis=1) / (p - 1))
    return (r_mat - r_earth_mat) * mult + r_earth_mat + 1j * h * w0 * l_mat


def _build_numpy_ybus(grid: Grid, h: int, index) -> np.ndarray:
    """Build the full per-harmonic Y-bus (numpy) mirroring pgml's assembly.

    Stamps Lines (series + shunt, R const / X∝h), Switches (series RL),
    Transformers (off-nominal complex-tap leakage-pi, magnetizing shunt on HV
    diagonal), and Source Norton shunts — exactly the same formulas as
    ``pgml.assembly.ybus._stamp_network`` + ``_stamp_sources``.

    Works for single-phase (P=1) and three-phase (P=3) grids: phase matrices
    are stamped into the compact ``NodePhaseIndex`` rows via ``index.rows()``.

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.schemas.grid_schema.Grid`.
    h:
        Harmonic order (integer >= 1).
    index:
        :class:`~pgml.assembly.NodePhaseIndex` for this grid.

    Returns
    -------
    numpy.ndarray
        Complex ``[N, N]`` admittance matrix.
    """
    n = index.size
    f0 = float(grid.base_frequency_hz)
    w0 = 2.0 * math.pi * f0
    y = np.zeros((n, n), dtype=complex)

    # Lines: series pi + shunt capacitance
    for b in grid.branches:
        if _is_ideal_branch(b):
            continue  # an ideal conductor: no stamp, its rows are fused instead
        if not (isinstance(b, Line) and getattr(b, "in_service", True)):
            continue
        if getattr(b, "conductor_geometry", None) is not None:
            # Carson geometry lines: not supported in this oracle (use
            # opendss_geometry_systemy for Carson validation)
            continue
        length = to_float(b.length_m)
        phases_b = b.from_phases
        p = len(phases_b)
        fr_rows = index.rows(b.from_node)
        to_rows = index.rows(b.to_node)
        r_mat = (
            np.array(
                [
                    [to_float(b.series_resistance_ohm_per_m[i][j]) for j in range(p)]
                    for i in range(p)
                ]
            )
            * length
        )
        l_mat = (
            np.array(
                [
                    [to_float(b.series_inductance_h_per_m[i][j]) for j in range(p)]
                    for i in range(p)
                ]
            )
            * length
        )
        c_mat = (
            np.array(
                [
                    [to_float(b.shunt_capacitance_f_per_m[i][j]) for j in range(p)]
                    for i in range(p)
                ]
            )
            * length
            if b.shunt_capacitance_f_per_m is not None
            else np.zeros((p, p))
        )
        ys = np.linalg.inv(_numpy_line_series_z(b, h, f0))
        ysh = 1j * h * w0 * c_mat
        half_ysh = 0.5 * ysh
        for i_ph, (fr, to) in enumerate(zip(fr_rows, to_rows)):
            for j_ph in range(p):
                fr2 = fr_rows[j_ph]
                to2 = to_rows[j_ph]
                y[fr_rows[i_ph], fr2] += ys[i_ph, j_ph] + (
                    half_ysh[i_ph, j_ph] if i_ph == j_ph else 0.0
                )
                y[to_rows[i_ph], to2] += ys[i_ph, j_ph] + (
                    half_ysh[i_ph, j_ph] if i_ph == j_ph else 0.0
                )
                y[fr_rows[i_ph], to2] -= ys[i_ph, j_ph]
                y[to_rows[i_ph], fr2] -= ys[i_ph, j_ph]

    # Switches: series RL only (no shunt)
    for b in grid.branches:
        if _is_ideal_branch(b):
            continue  # an ideal conductor: no stamp, its rows are fused instead
        if not (isinstance(b, Switch) and getattr(b, "in_service", True) and b.closed):
            continue
        phases_b = b.from_phases
        p = len(phases_b)
        fr_rows = index.rows(b.from_node)
        to_rows = index.rows(b.to_node)
        r_sw = to_float(b.resistance_ohm)
        l_sw = to_float(b.inductance_h)
        z_sw = r_sw + 1j * h * w0 * l_sw
        ys_sw = 1.0 / z_sw
        # Diagonal per phase (pgml stamps each phase independently for switches)
        for i_ph in range(p):
            fr = fr_rows[i_ph]
            to = to_rows[i_ph]
            y[fr, fr] += ys_sw
            y[to, to] += ys_sw
            y[fr, to] -= ys_sw
            y[to, fr] -= ys_sw

    # Transformers: winding-incidence primitive (new vector-group model).
    # See pgml.assembly._transformer for the full derivation.
    for b in grid.branches:
        if not (isinstance(b, Transformer) and getattr(b, "in_service", True)):
            continue
        p = len(b.from_phases)
        fr_rows = index.rows(b.from_node)
        to_rows = index.rows(b.to_node)
        _stamp_transformer_numpy(y, b, h, w0, fr_rows, to_rows, p)

    # Sources: Norton shunt Y_s = Z_s(h)^-1 (held at zero harmonic voltage)
    for a in grid.appliances:
        if not (isinstance(a, Source) and getattr(a, "in_service", True)):
            continue
        phases_a = a.phases
        p = len(phases_a)
        src_rows = index.rows(a.node)
        r_mat = np.array(
            [[to_float(a.resistance_ohm[i][j]) for j in range(p)] for i in range(p)]
        )
        l_mat = np.array(
            [[to_float(a.inductance_h[i][j]) for j in range(p)] for i in range(p)]
        )
        z_mat = r_mat + 1j * h * w0 * l_mat
        ys_mat = np.linalg.inv(z_mat)
        for i_ph in range(p):
            for j_ph in range(p):
                y[src_rows[i_ph], src_rows[j_ph]] += ys_mat[i_ph, j_ph]

    return y


def _apply_node_sources_numpy(
    grid: Grid,
    node_sources: Sequence,
    v1: np.ndarray,
    index,
    h: int,
    y_h: np.ndarray,
    i_h: np.ndarray,
) -> None:
    """Stamp per-node harmonic disturbance sources in-place for order ``h``.

    Implements the physics from ``docs/pgml/modeling/error-injection.md``:

    - ``V_base`` = node ``u_rated_v`` (L-N for ≥3-phase, else L-L).
    - ``Y_s = S_sc / V_base^2`` (real, frequency-flat).
    - ``E_h = (mag_h/mag_1) * |V1_row| * exp(j*(ang_h + h*(arg(V1_row) - ang_1)))``.
    - ``I_N = E_h * Y_s``.
    - ``kind="voltage"``: add ``Y_s`` to ``y_h[row, row]``; add ``I_N`` to ``i_h[row]``.
    - ``kind="current"``: add ``I_N`` to ``i_h[row]`` only.

    Modifies ``y_h`` and ``i_h`` in-place (numpy arrays).
    """
    node_map = {int(nd.id): nd for nd in grid.nodes}
    from pgml.assembly._params import phase_voltage_magnitude

    for ns in node_sources:
        nid = int(ns.node_id)
        node = node_map[nid]
        n_phases_node = len(node.phases)
        v_base = phase_voltage_magnitude(float(node.u_rated_v), n_phases_node)
        y_s = float(ns.source_power_va) / (v_base * v_base)  # real admittance S

        spec = {
            int(o): (float(mag), float(ang)) for o, (mag, ang) in ns.spectrum.items()
        }
        mag1, ang1 = spec.get(1, (1.0, 0.0))
        ang1_rad = math.radians(ang1)
        if mag1 == 0.0:
            continue

        mag_h, ang_h = spec.get(h, (0.0, 0.0))
        if mag_h == 0.0:
            continue
        ang_h_rad = math.radians(ang_h)
        ratio = mag_h / mag1

        phases = ns.phases if ns.phases is not None else list(node.phases)
        for phase in phases:
            row = index.row(nid, phase)
            v1_row = v1[row]
            e_h = (
                ratio
                * abs(v1_row)
                * cmath.exp(1j * (ang_h_rad + h * (cmath.phase(v1_row) - ang1_rad)))
            )
            i_n = e_h * y_s

            i_h[row] += i_n
            if ns.kind == "voltage":
                y_h[row, row] += y_s


def _operating_power_numpy(appliance, operating_point, n_ph: int):
    """Per-phase ``(P, Q)`` lists of an appliance: override, per-phase nameplate, split.

    Mirrors :func:`pgml.assembly._params.resolve_operating_power` for the plain-float
    (oracle) case, so the shunt and the injected current read the same operating point.
    """
    p_total = to_float(appliance.p_nom_w)
    q_total = to_float(appliance.q_nom_var)
    p_pp = (
        [to_float(x) for x in appliance.p_nom_per_phase_w]
        if getattr(appliance, "p_nom_per_phase_w", None) is not None
        else [p_total / n_ph] * n_ph
    )
    q_pp = (
        [to_float(x) for x in appliance.q_nom_per_phase_var]
        if getattr(appliance, "q_nom_per_phase_var", None) is not None
        else [q_total / n_ph] * n_ph
    )
    if operating_point and appliance.id in operating_point:
        op = operating_point[appliance.id]
        if "p_per_phase_w" in op:
            p_pp = [to_float(x) for x in op["p_per_phase_w"]]
        elif "p_w" in op:
            p_pp = [to_float(op["p_w"]) / n_ph] * n_ph
        if "q_per_phase_var" in op:
            q_pp = [to_float(x) for x in op["q_per_phase_var"]]
        elif "q_var" in op:
            q_pp = [to_float(op["q_var"]) / n_ph] * n_ph
    return p_pp, q_pp


def _device_shunt_numpy(
    appliance,
    load_shunt: str,
    p_elem: float,
    q_elem: float,
    v_rated: float,
    h: int,
    *,
    kva_base: Optional[float] = None,
) -> complex:
    """The harmonic device shunt of one element, written out in plain python complex.

    Mirrors OpenDSS ``Load.pas`` ``CalcYPrimMatrix`` from the equations, not from the
    torch implementation::

        Y_eq     = (P - jQ)/V_rated**2
        Y_par(h) = (1-s)Re(Y_eq) + j(1-s)Im(Y_eq)/h
        Z_ser    = 1/(s*Y_eq)  (or X/xr + jX for the motor model), scaled Re + j*h*Im

    ``p_elem``/``q_elem`` carry the device's sign (a generator's are negative).
    ``kva_base`` is the motor reactance's apparent-power base (defaults to the element's
    own). Returns ``0`` for the ``none`` model or a device at zero power.
    """
    from pgml.assembly._load_shunt import resolve_harmonic_shunt

    spec = resolve_harmonic_shunt(appliance, load_shunt)
    if spec.kind == "none":
        return 0.0 + 0.0j
    y_eq = complex(p_elem, -q_elem) / (v_rated * v_rated)
    if y_eq == 0.0:
        return 0.0 + 0.0j
    s = spec.series_rl_fraction
    y = complex((1.0 - s) * y_eq.real, (1.0 - s) * y_eq.imag / h)
    if s > 0.0:
        if spec.kind == "motor":
            kva = abs(complex(p_elem, q_elem)) if kva_base is None else kva_base
            if kva == 0.0:
                return y
            x = v_rated * v_rated / (kva * s) * spec.motor_x_harm_pu
            z_ser = complex(x / spec.motor_xr_harm, x)
        else:
            z_ser = 1.0 / (s * y_eq)
        y += 1.0 / complex(z_ser.real, h * z_ser.imag)
    return y


def _stamp_device_shunts_numpy(
    y: np.ndarray, grid, index, h: int, load_shunt: str, operating_point=None
) -> None:
    """Add every in-service device's harmonic shunt to the diagonal of ``y`` (in place).

    Phase-to-ground (WYE) per device phase, which is the connection these oracles
    support; the admittance is :func:`_device_shunt_numpy`. ``load_shunt == "none"``
    does nothing. Shared by the numpy and the live-OpenDSS harmonic oracles so both
    compare against the SAME device model the engine solves.
    """
    if load_shunt == "none":
        return
    nodes = {int(nd.id): nd for nd in grid.nodes}
    for a in grid.appliances:
        if not (isinstance(a, InjectionAppliance) and getattr(a, "in_service", True)):
            continue
        n_ph = len(a.phases)
        node = nodes[int(a.node)]
        v0 = (
            to_float(node.u_rated_v) / math.sqrt(3.0)
            if len(node.phases) >= 3
            else to_float(node.u_rated_v)
        )
        sign = 1.0 if isinstance(a, Load) else -1.0
        p_pp, q_pp = _operating_power_numpy(a, operating_point, n_ph)
        for k_ph, row in enumerate(index.rows(a.node)[:n_ph]):
            y[row, row] += _device_shunt_numpy(
                a, load_shunt, sign * p_pp[k_ph], sign * q_pp[k_ph], v0, h
            )


def numpy_harmonic_profiles(
    grid: Grid,
    v1,
    index,
    orders: Sequence[int],
    *,
    slack=None,
    label: str = "numpy",
    unit: str = "pu",
    harmonic_injection=None,
    operating_point=None,
    load_shunt: Optional[str] = None,
) -> list[HarmonicProfile]:
    """Independent numpy harmonic solve -> one :class:`HarmonicProfile` per order.

    Single-phase only. ``v1`` is the converged fundamental node-voltage vector (numpy
    or tensor, aligned to ``index`` rows) — the SHARED operating point. For each order
    ``h > 1`` this builds ``Y(h)`` (line series/shunt with R const & X∝h, the source
    Norton shunt and each device's harmonic shunt), injects each device's harmonic
    current per the OpenDSS convention, and solves ``Y(h) V(h) = I(h)``. Order 1 returns
    ``v1``. ``load_shunt`` selects the device shunt model (``None`` = the documented
    modeling default, as in :func:`pgml.solver.solve_harmonic_flow`).
    """
    from pgml.assembly._load_shunt import resolve_shunt_model_name

    shunt = resolve_shunt_model_name(load_shunt)
    for node in grid.nodes:
        if len(node.phases) != 1:
            raise ValueError(
                "numpy_harmonic_profiles supports single-phase grids only."
            )
    v1 = np.asarray(v1.detach().cpu().numpy() if hasattr(v1, "detach") else v1).reshape(
        -1
    )
    f0 = float(grid.base_frequency_hz)
    w0 = 2.0 * math.pi * f0
    n = index.size
    # An ideal (zero-impedance) branch has no stamp; its terminals are one node.
    p_fuse = fusion_prolongation(grid, index)

    def _build_y(h: int) -> np.ndarray:
        y = np.zeros((n, n), dtype=complex)
        for b in grid.branches:
            if _is_ideal_branch(b):
                continue  # an ideal conductor: no stamp, its rows are fused instead
            if not (isinstance(b, Line) and getattr(b, "in_service", True)):
                continue
            length = to_float(b.length_m)
            c = (
                to_float(b.shunt_capacitance_f_per_m[0][0]) * length
                if b.shunt_capacitance_f_per_m
                else 0.0
            )
            ys = 1.0 / complex(_numpy_line_series_z(b, h, f0)[0, 0])
            ysh = 1j * h * w0 * c
            fr = index.row(b.from_node, Phase.A)
            to = index.row(b.to_node, Phase.A)
            y[fr, fr] += ys + 0.5 * ysh
            y[to, to] += ys + 0.5 * ysh
            y[fr, to] -= ys
            y[to, fr] -= ys
        for a in grid.appliances:
            if isinstance(a, Source) and getattr(a, "in_service", True):
                z = to_float(a.resistance_ohm[0][0]) + 1j * h * w0 * to_float(
                    a.inductance_h[0][0]
                )
                r0 = index.row(a.node, Phase.A)
                y[r0, r0] += 1.0 / z
        return y

    # Per-device fundamental current I1 (load convention) for the injection scaling.
    devs = []
    for a in grid.appliances:
        if not (isinstance(a, InjectionAppliance) and getattr(a, "in_service", True)):
            continue
        spec = _resolve_spectrum(a, harmonic_injection)
        if spec is None:
            continue
        row = index.row(a.node, Phase.A)
        sign = 1.0 if isinstance(a, Load) else -1.0
        p = to_float(a.p_nom_w)
        q = to_float(a.q_nom_var)
        if operating_point and a.id in operating_point:
            op = operating_point[a.id]
            if "p_w" in op:
                p = to_float(op["p_w"])
            if "q_var" in op:
                q = to_float(op["q_var"])
        s0 = complex(sign * p, sign * q)
        i1 = np.conj(s0) / np.conj(v1[row])
        devs.append((spec, row, i1))

    out: list[HarmonicProfile] = []
    dist = distance_from_slack(grid, slack)
    for h in orders:
        if h == 1:
            vh = v1
        else:
            y = _build_y(h)
            _stamp_device_shunts_numpy(y, grid, index, h, shunt, operating_point)
            i = np.zeros(n, dtype=complex)
            for spec, row, i1 in devs:
                mag1, ang1 = spec.get(1, (1.0, 0.0))
                mag_h, ang_h = spec.get(h, (0.0, 0.0))
                i_drawn = (
                    (mag_h / mag1)
                    * abs(i1)
                    * cmath.exp(
                        1j
                        * (
                            math.radians(ang_h)
                            + h * (cmath.phase(i1) - math.radians(ang1))
                        )
                    )
                )
                i[row] += -i_drawn  # nodal injection
            vh = solve_with_fusion(y, i, p_fuse)
        ds, mags, angs, nids = [], [], [], []
        for node in grid.nodes:
            row = index.row(int(node.id), Phase.A)
            base = (to_float(node.u_rated_v)) if unit == "pu" else 1.0
            ds.append(dist[int(node.id)])
            mags.append(abs(vh[row]) / base)
            angs.append(math.degrees(cmath.phase(complex(vh[row]))))
            nids.append(int(node.id))
        o = np.argsort(ds)
        out.append(
            HarmonicProfile(
                distances_km=np.asarray(ds)[o],
                magnitude=np.asarray(mags)[o],
                angle_deg=np.asarray(angs)[o],
                order=int(h),
                frequency_hz=float(h * f0),
                label=label,
                node_ids=np.asarray(nids)[o],
                unit=unit,
            )
        )
    return out


def numpy_harmonic_voltages(
    grid: Grid,
    harmonic_injection: Optional[dict],
    orders: Sequence[int],
    *,
    slack: str = "norton",
    v1: Optional[np.ndarray] = None,
    operating_point: Optional[dict] = None,
    node_sources: Optional[Sequence] = None,
    load_shunt: Optional[str] = None,
) -> np.ndarray:
    """Pure-numpy harmonic voltage oracle — exact pgml parity (R const / X∝h).

    Returns complex node voltages ``[len(orders), N]`` aligned to
    :func:`pgml.assembly.node_phase_index` rows, solving the same linear harmonic
    system as :func:`pgml.solver.solve_harmonic_flow`.

    This is the **regression oracle**: it reimplements pgml's EXACT Y-bus formulas
    (R const / X∝h for all elements including transformers and source Norton) in
    pure numpy, giving machine-precision parity (~1e-13 V absolute) vs
    ``solve_harmonic_flow``.  No live OpenDSS circuit is built; the
    ``import opendssdirect`` dependency is not required.

    **Model (exact pgml parity, R const / X∝h)**

    - Lines: series pi (``R`` fixed, ``X(h) = h·X(f0)``), shunt capacitance
      (``B(h) = h·B(f0)``).  Conductor-geometry lines are skipped (the Carson
      path lives in :func:`opendss_harmonic_voltages`).
    - Switches: series RL (identical scaling).
    - Transformers: per-phase diagonal off-nominal-tap leakage-pi stamp —
      ``Y_ff = y_se/|t|² + y_m``, ``Y_ft = −y_se/t*``, ``Y_tf = −y_se/t``,
      ``Y_tt = y_se`` — where ``y_se = (R + j·h·2πf₀·L)⁻¹`` and
      ``t = ratio_magnitude · exp(j·shift_deg)``.
    - Sources (Norton mode): shunt ``Y_s(h) = Z_s(h)⁻¹`` on the source diagonal.
    - Harmonic injection: OpenDSS convention (``docs/pgml/modeling/references/opendss/harmonics.md``).

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.schemas.grid_schema.Grid`.
    harmonic_injection:
        Per-device harmonic-current spec —
        ``{appliance_id: {order: (magnitude_pu, phase_deg)}}``.
    orders:
        Harmonic orders to solve (e.g. ``[1, 5, 11]``).
    slack:
        Only ``"norton"`` is implemented.
    v1:
        Optional pre-computed fundamental voltage vector ``[N]``.  When ``None``,
        a linear const-Z fundamental is solved internally.
    operating_point:
        Optional per-device operating-point override
        ``{appliance_id: {order: value}}`` — passed through to the linear
        fundamental solve when ``v1 is None``.  Currently unused when ``v1`` is
        provided (the passed ``v1`` already encodes the operating point).
    node_sources:
        Optional sequence of :class:`~pgml.solver.NodeHarmonicSource` — per-node
        harmonic disturbance sources applied ONLY at ``h > 1``.  Each source stamps
        its Norton current ``I_N`` (and, for ``kind="voltage"``, the shunt ``Y_s``)
        into the harmonic system using the same physics as
        :func:`pgml.solver.solve_harmonic_flow`.  When ``None`` (default), the oracle
        is byte-identical to the pre-node-source behaviour.

    Returns
    -------
    numpy.ndarray
        Complex ``[H, N]``.

    See Also
    --------
    opendss_harmonic_voltages : Live OpenDSS oracle (Carson lines).
    numpy_harmonic_profiles : Single-phase numpy oracle (lines + source only).
    """
    if slack != "norton":
        raise ValueError(
            f"numpy_harmonic_voltages supports only slack='norton'; got {slack!r}"
        )
    from pgml.assembly import node_phase_index
    from pgml.assembly._load_shunt import resolve_shunt_model_name
    from pgml.schemas.grid_schema import Generator as PgmlGen, Load as PgmlLoad

    shunt = resolve_shunt_model_name(load_shunt)
    orders_list = [int(h) for h in orders]
    index = node_phase_index(grid)
    # An ideal (zero-impedance) branch has no stamp; its terminals are one electrical
    # node, which the dense prolongation expresses (see `fusion_prolongation`).
    p_fuse = fusion_prolongation(grid, index)
    n = index.size
    f0 = float(grid.base_frequency_hz)
    w0 = 2.0 * math.pi * f0

    # --- Fundamental voltage (operating point for injection scaling) ---
    if v1 is None:
        # Linear fundamental: network Y + const-Z load shunts + source Norton current
        y1 = _build_numpy_ybus(grid, 1, index)
        node_map_v = {nd.id: nd for nd in grid.nodes}
        for a in grid.appliances:
            if not (
                isinstance(a, (PgmlLoad, PgmlGen)) and getattr(a, "in_service", True)
            ):
                continue
            node_v = node_map_v[a.node]
            u_rated = to_float(node_v.u_rated_v)
            phases_a = a.phases
            n_ph = len(phases_a)
            src_rows_v = index.rows(a.node)
            sign = 1.0 if isinstance(a, PgmlLoad) else -1.0
            p_total = to_float(a.p_nom_w)
            q_total = to_float(a.q_nom_var)
            p_ph = p_total / n_ph
            q_ph = q_total / n_ph
            v0 = u_rated / math.sqrt(3.0) if n_ph >= 3 else u_rated
            y_elem = (sign * p_ph - 1j * sign * q_ph) / (v0**2)
            for r in src_rows_v:
                y1[r, r] += y_elem
        i1 = np.zeros(n, dtype=complex)
        for a in grid.appliances:
            if not (isinstance(a, Source) and getattr(a, "in_service", True)):
                continue
            phases_a = a.phases
            p = len(phases_a)
            src_rows_v = index.rows(a.node)
            r_mat = np.array(
                [
                    [to_float(a.resistance_ohm[i_][j_]) for j_ in range(p)]
                    for i_ in range(p)
                ]
            )
            l_mat = np.array(
                [
                    [to_float(a.inductance_h[i_][j_]) for j_ in range(p)]
                    for i_ in range(p)
                ]
            )
            z_mat = r_mat + 1j * w0 * l_mat
            ys_mat = np.linalg.inv(z_mat)
            u_ref = np.array([to_float(a.u_ref_v[k]) for k in range(p)])
            u_ang = np.array(
                [to_float(a.u_angle_deg[k]) * math.pi / 180.0 for k in range(p)]
            )
            v_th = u_ref * np.exp(1j * u_ang)
            i_s = ys_mat @ v_th
            for k in range(p):
                i1[src_rows_v[k]] += i_s[k]
        v1_eff = solve_with_fusion(y1, i1, p_fuse)
    else:
        v1_arr = np.asarray(v1).reshape(-1)
        if v1_arr.shape[0] != n:
            raise ValueError(
                f"v1 has {v1_arr.shape[0]} entries but grid has N={n} rows"
            )
        v1_eff = v1_arr.astype(complex)

    # --- Per-device fundamental current + spectrum (for h > 1 injection) ---
    devs = []
    for a in grid.appliances:
        if not (isinstance(a, (PgmlLoad, PgmlGen)) and getattr(a, "in_service", True)):
            continue
        # Resolve spectrum from override or stored spectrum
        if harmonic_injection is not None and a.id in harmonic_injection:
            spec: dict = {
                int(o): (float(mag), float(ang))
                for o, (mag, ang) in harmonic_injection[a.id].items()
            }
        else:
            s = getattr(a, "spectrum", None)
            if not isinstance(s, StaticSpectrum):
                continue
            spec = {
                c.order: (to_float(c.magnitude_pu), to_float(c.phase_deg))
                for c in s.spectrum.components
            }
        if not spec:
            continue
        phases_a = a.phases
        n_ph = len(phases_a)
        src_rows = index.rows(a.node)
        sign = 1.0 if isinstance(a, PgmlLoad) else -1.0
        p_total = to_float(a.p_nom_w)
        q_total = to_float(a.q_nom_var)
        # Per-phase S0: honor an explicit per-phase nameplate (asymmetric loads), else
        # split the total equally — mirroring pgml's resolve_operating_power so the
        # per-phase fundamental current I1 (and thus the harmonic injection) matches.
        p_pp = (
            [to_float(x) for x in a.p_nom_per_phase_w]
            if getattr(a, "p_nom_per_phase_w", None) is not None
            else [p_total / n_ph] * n_ph
        )
        q_pp = (
            [to_float(x) for x in a.q_nom_per_phase_var]
            if getattr(a, "q_nom_per_phase_var", None) is not None
            else [q_total / n_ph] * n_ph
        )
        # Fundamental current per phase: I1 = conj(S0_ph) / conj(V_term)
        i1_list = []
        for k_ph, row in enumerate(src_rows):
            v_term = v1_eff[row]
            s0_ph = complex(sign * p_pp[k_ph], sign * q_pp[k_ph])
            if abs(v_term) < 1e-300:
                i1_list.append(0.0 + 0j)
            else:
                i1_list.append(np.conj(s0_ph) / np.conj(v_term))
        devs.append((spec, src_rows, i1_list))

    # --- Per-order solve ---
    result_slices: list[np.ndarray] = []
    for h in orders_list:
        if h == 1:
            result_slices.append(v1_eff)
            continue
        y_h = _build_numpy_ybus(grid, h, index)
        _stamp_device_shunts_numpy(y_h, grid, index, h, shunt, operating_point)
        i_h = np.zeros(n, dtype=complex)
        for spec, src_rows, i1_list in devs:
            mag1, ang1 = spec.get(1, (1.0, 0.0))
            mag_h, ang_h = spec.get(h, (0.0, 0.0))
            if mag1 == 0.0:
                continue
            ratio = mag_h / mag1
            for i1_val, row in zip(i1_list, src_rows):
                i_drawn = (
                    ratio
                    * abs(i1_val)
                    * cmath.exp(
                        1j
                        * (
                            math.radians(ang_h)
                            + h * (cmath.phase(i1_val) - math.radians(ang1))
                        )
                    )
                )
                i_h[row] += -i_drawn  # nodal injection (drawn = negative source)

        # --- Per-node harmonic disturbance sources ---
        if node_sources:
            _apply_node_sources_numpy(grid, node_sources, v1_eff, index, h, y_h, i_h)

        result_slices.append(solve_with_fusion(y_h, i_h, p_fuse))

    return np.stack(result_slices, axis=0)  # [H, N]


__all__ = [
    "fusion_prolongation",
    "numpy_harmonic_profiles",
    "numpy_harmonic_voltages",
    "solve_with_fusion",
]
