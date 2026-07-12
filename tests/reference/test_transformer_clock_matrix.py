"""Vector-group clock matrix: sequence-domain oracle for the transformer stamp.

Every supported winding pairing (wye / grounded-wye / delta / zigzag, both sides)
is swept over every clock number of the correct parity, and the assembled
phase-domain nodal block ``Nᵀ Y_winding N`` is checked against the textbook
positive-sequence off-nominal-tap pi::

    Y_ff = y/|t|²,  Y_ft = −y/conj(t),  Y_tf = −y/t,  Y_tt = y
    t = n_LL · e^{j·clock·30°}          (positive shift = LV lags HV)

by projecting each 3×3 quadrant onto the positive-sequence phasor set. The
projection is exact (all incidence blocks are circulant), so the comparison is
to machine precision — this is the numerical pin for the clock sign convention,
the delta / zigzag orientation selection and the coil-rating normalisation.

The LV-referred admittance convention this pins (the converter contract): the
schema's ``series_*`` leakage is referred to the TO-side COIL, so the scalar
line-to-line admittance is ``y_LL = 3·y_coil`` for a delta TO winding and
``y_LL = y_coil`` for wye / zigzag TO windings.

Zero-sequence structure is asserted per pairing: delta and zigzag windings block
zero-sequence transfer; a grounded zigzag keeps a low-impedance zero-sequence
self path (grounding-transformer property); grounded wye passes zero sequence.
"""

from __future__ import annotations

import cmath
import math

import pytest
import torch

from pgml.assembly._transformer import (
    SideConn,
    VectorGroup,
    block_incidence,
    nominal_turns_ratio,
    resolve_vector_group,
    winding_leakage_block,
)
from pgml.errors import ModelingError
from pgml.schemas.grid_schema import ComplexTap, Transformer, Phase, WindingConnection

ABC = (Phase.A, Phase.B, Phase.C)

U_FROM = 20_000.0
U_TO = 400.0
Z_COIL = 0.02 + 0.06j  # leakage referred to the TO-side coil

_KINDS = {
    "wye_grounded": SideConn("wye_grounded", grounded=True),
    "wye": SideConn("wye", grounded=False),
    "delta": SideConn("delta", grounded=False),
    "zigzag_grounded": SideConn("zigzag_grounded", grounded=True),
    "zigzag": SideConn("zigzag", grounded=False),
}

_A = cmath.exp(2j * math.pi / 3)
_V_POS = torch.tensor([1.0, _A**2, _A], dtype=torch.complex128) / math.sqrt(3.0)
_V_ZERO = torch.tensor([1.0, 1.0, 1.0], dtype=torch.complex128) / math.sqrt(3.0)


def _pairings():
    for fk in _KINDS:
        for tk in _KINDS:
            if "zigzag" in fk and "zigzag" in tk:
                continue  # rejected pairing
            shifting = ("delta" in fk or "zigzag" in fk) != (
                "delta" in tk or "zigzag" in tk
            )
            clocks = range(1, 12, 2) if shifting else range(0, 12, 2)
            for clock in clocks:
                yield fk, tk, clock


def _assembled_block(fk: str, tk: str, clock: int) -> torch.Tensor:
    """[6, 6] nodal leakage block for one unit at the fundamental."""
    vg = VectorGroup(_KINDS[fk], _KINDS[tk], clock)
    n_blk = block_incidence(vg, 3, torch.float64, torch.device("cpu"))
    y_coil = torch.tensor([[1.0 / Z_COIL]], dtype=torch.complex128)  # [H=1,K=1]
    u_from = torch.tensor(U_FROM, dtype=torch.float64)
    u_to = torch.tensor(U_TO, dtype=torch.float64)
    tau = nominal_turns_ratio(vg, u_from, u_to)[None]  # [K=1]
    return winding_leakage_block(y_coil, tau, n_blk)[0, 0]


def _project(block: torch.Tensor, v: torch.Tensor) -> dict[str, complex]:
    """Project each 3×3 quadrant of the 6×6 block onto the phasor set ``v``."""
    quads = {
        "ff": block[:3, :3],
        "ft": block[:3, 3:],
        "tf": block[3:, :3],
        "tt": block[3:, 3:],
    }
    return {k: complex(v.conj() @ q.to(torch.complex128) @ v) for k, q in quads.items()}


@pytest.mark.parametrize("fk,tk,clock", list(_pairings()))
def test_positive_sequence_equals_scalar_tap_pi(fk, tk, clock):
    block = _assembled_block(fk, tk, clock)
    pos = _project(block, _V_POS)

    y_ll = (3.0 if tk == "delta" else 1.0) / Z_COIL
    n_ll = U_FROM / U_TO
    t = n_ll * cmath.exp(1j * math.radians(clock * 30.0))

    assert pos["tt"] == pytest.approx(y_ll, rel=1e-12), "LV self"
    assert pos["ff"] == pytest.approx(y_ll / abs(t) ** 2, rel=1e-12), "HV self"
    assert pos["ft"] == pytest.approx(-y_ll / t.conjugate(), rel=1e-12), "HV-LV"
    assert pos["tf"] == pytest.approx(-y_ll / t, rel=1e-12), "LV-HV"


@pytest.mark.parametrize("fk,tk,clock", list(_pairings()))
def test_zero_sequence_structure(fk, tk, clock):
    block = _assembled_block(fk, tk, clock)
    zero = _project(block, _V_ZERO)

    def blocks_zero_seq(kind: str) -> bool:
        return kind in ("delta", "wye", "zigzag")

    # Transfer: ANY delta or zigzag winding interrupts the zero-sequence path
    # (zigzag MMF cancels per limb; delta circulates it; a floating star has no
    # return). Both couplings vanish together.
    transfer_blocked = (
        blocks_zero_seq(fk) or blocks_zero_seq(tk) or "zigzag" in fk or "zigzag" in tk
    )
    if transfer_blocked:
        assert abs(zero["ft"]) < 1e-12 and abs(zero["tf"]) < 1e-12
    else:
        assert abs(zero["ft"]) > 1e-6 and abs(zero["tf"]) > 1e-6

    # Self path on each side.
    y_coil = 1.0 / Z_COIL
    if tk == "zigzag_grounded":
        # Grounding-transformer property: full leakage-limited zero-seq self path
        # on the zigzag side even though nothing transfers to the far winding.
        assert zero["tt"] == pytest.approx(y_coil, rel=1e-12)
    if tk in ("wye", "zigzag", "delta"):
        assert abs(zero["tt"]) < 1e-12
    if fk in ("wye", "delta"):
        assert abs(zero["ff"]) < 1e-12
    if fk == "zigzag_grounded":
        assert abs(zero["ff"]) > 1e-6


def _transformer(fc: WindingConnection, tc: WindingConnection, shift: float):
    return Transformer(
        id=1,
        from_node=1,
        to_node=2,
        from_phases=ABC,
        to_phases=ABC,
        s_rated_va=0.4e6,
        u_rated_from_v=U_FROM,
        u_rated_to_v=U_TO,
        from_connection=fc,
        to_connection=tc,
        series_resistance_ohm=Z_COIL.real,
        series_inductance_h=Z_COIL.imag / (2.0 * math.pi * 50.0),
        tap=ComplexTap(ratio_magnitude=1.0, shift_deg=shift),
    )


def test_parity_mismatch_raises():
    t = _transformer(WindingConnection.DELTA, WindingConnection.WYE_GROUNDED, 60.0)
    with pytest.raises(ModelingError, match="clock 2 is inconsistent"):
        resolve_vector_group(t, n_phases=3)
    t = _transformer(WindingConnection.WYE, WindingConnection.WYE_GROUNDED, 150.0)
    with pytest.raises(ModelingError, match="clock 5 is inconsistent"):
        resolve_vector_group(t, n_phases=3)


def test_zigzag_zigzag_rejected():
    t = _transformer(
        WindingConnection.ZIGZAG_GROUNDED, WindingConnection.ZIGZAG_GROUNDED, 0.0
    )
    with pytest.raises(ModelingError, match="zigzag-zigzag"):
        resolve_vector_group(t, n_phases=3)


@pytest.mark.parametrize(
    "fc,tc,clock",
    [
        (WindingConnection.DELTA, WindingConnection.WYE_GROUNDED, 5),  # Dyn5
        (WindingConnection.WYE_GROUNDED, WindingConnection.DELTA, 5),  # YNd5
        (WindingConnection.WYE, WindingConnection.ZIGZAG_GROUNDED, 5),  # Yzn5
        (WindingConnection.DELTA, WindingConnection.WYE_GROUNDED, 11),  # Dyn11
        (WindingConnection.WYE_GROUNDED, WindingConnection.WYE_GROUNDED, 4),  # YNyn4
        (WindingConnection.DELTA, WindingConnection.DELTA, 6),  # Dd6
    ],
)
def test_solved_lv_angle_matches_clock(fc, tc, clock):
    """End-to-end: solved LV phase-A angle lags HV by ≈ clock·30° (light load)."""
    from pgml.assembly import assemble_ybus, build_injections, node_phase_index
    from pgml.schemas.grid_schema import Grid, Load, Node, Source
    from pgml.solver import solve_harmonic

    shift = clock * 30.0
    src = Source(
        id=10,
        node=1,
        phases=ABC,
        u_ref_v=(U_FROM / math.sqrt(3.0),) * 3,
        u_angle_deg=(0.0, -120.0, 120.0),
        resistance_ohm=[[0.05 if i == j else 0.0 for j in range(3)] for i in range(3)],
        inductance_h=[[1e-4 if i == j else 0.0 for j in range(3)] for i in range(3)],
    )
    xfmr = _transformer(fc, tc, shift)
    load = Load(id=30, node=2, phases=ABC, p_nom_w=1.0e3, q_nom_var=100.0)
    grid = Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=U_FROM, phases=ABC),
            Node(id=2, u_rated_v=U_TO, phases=ABC),
        ],
        branches=[xfmr],
        appliances=[src, load],
    )
    idx = node_phase_index(grid)
    yb = assemble_ybus(grid, [50.0], dtype=torch.complex128)
    inj = build_injections(grid, [50.0], idx, dtype=torch.complex128)
    v = solve_harmonic(yb.Y, inj)[0]
    va_hv = math.degrees(cmath.phase(complex(v[idx.row(1, Phase.A)])))
    va_lv = math.degrees(cmath.phase(complex(v[idx.row(2, Phase.A)])))
    lag = (va_hv - va_lv) % 360.0
    assert lag == pytest.approx(shift % 360.0, abs=1.5), (
        f"LV should lag HV by {shift}°, got {lag:.3f}°"
    )
    # Balanced magnitudes near nominal line-to-neutral on the LV side.
    v_lv = [abs(complex(v[idx.row(2, p)])) for p in ABC]
    for m in v_lv:
        assert m == pytest.approx(U_TO / math.sqrt(3.0), rel=0.05)


def test_non_clock_shift_rejected_three_phase_exact_single_phase():
    t = _transformer(
        WindingConnection.WYE_GROUNDED, WindingConnection.WYE_GROUNDED, 7.5
    )
    with pytest.raises(ModelingError, match="not a multiple of 30"):
        resolve_vector_group(t, n_phases=3)
    vg = resolve_vector_group(t, n_phases=1)
    assert vg.shift_exact_deg == pytest.approx(7.5)
    assert vg.clock == 0  # nearest clock, used only for parity/grouping
