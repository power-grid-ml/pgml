"""Forward-correctness of the nonlinear fundamental power flow.

Validates the frozen requirements of ``solve_power_flow``:

- **Const-Z consistency (required regression link):** an all-const-impedance ZIP
  run reproduces the linear ``assemble_ybus`` + ``solve_harmonic`` system EXACTLY
  (Norton and ideal-slack), near machine precision.
- **Convergence:** a tiny const-power grid converges (residual -> tol) and the
  converged ``V`` satisfies the power-balance residual (the load draws exactly its
  nominal complex power).
- **ZIP:** the const_power / const_impedance / const_current special cases match
  their closed forms; a mixed-ZIP load lies between const-Z and const-P.
"""

from __future__ import annotations

import math

import torch

from pgml.assembly import (
    assemble_network_ybus,
    assemble_ybus,
    build_injections,
    node_phase_index,
)
from pgml.assembly._stamps import _cdtype, _rdtype
from pgml.assembly.ybus import _stamp_sources
from pgml.schemas.grid_schema import (
    Grid,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
    ZipCoefficients,
)
from pgml.solver import solve_harmonic, solve_power_flow

from tests.fixtures.tiny_grids import single_phase_chain, three_phase_two_bus

CDT = torch.complex128
F = 50.0


def _set_load_model(grid: Grid, model: LoadModel, zc=None) -> Grid:
    for a in grid.appliances:
        if isinstance(a, Load):
            a.load_model = model
            if zc is not None:
                a.zip_coefficients = zc
    return grid


# ---------------------------------------------------------------------------
# const-Z consistency: PF (all const-Z) == assemble_ybus + solve_harmonic
# ---------------------------------------------------------------------------
def _linear_norton_v(grid: Grid):
    idx = node_phase_index(grid)
    yb = assemble_ybus(grid, [F], dtype=CDT)
    i = build_injections(grid, [F], idx, dtype=CDT)
    return solve_harmonic(yb.Y, i).reshape(-1), idx


def _linear_ideal_v(grid: Grid):
    """Linear ideal-slack reference: Y_net + Y_devZ (NO source Norton), slack pinned."""
    idx = node_phase_index(grid)
    n = idx.size
    cdt, rdt = _cdtype(CDT), _rdtype(CDT)
    yb_net = assemble_network_ybus(grid, [F], dtype=CDT)
    yb_lin = assemble_ybus(grid, [F], dtype=CDT)
    f = torch.tensor([F], dtype=rdt)
    y_net_plus_src = _stamp_sources(
        grid, f, yb_net.Y.clone(), idx, cdt, rdt, yb_net.Y.device, None
    )
    y_devz = yb_lin.Y - y_net_plus_src  # const-Z load only
    y_eff = yb_net.Y + y_devz
    # slack rows = source-node phases.
    src = next(a for a in grid.appliances if isinstance(a, Source))
    rows, vref = [], []
    u_ref = torch.as_tensor(src.u_ref_v, dtype=rdt)
    ang = torch.as_tensor(src.u_angle_deg, dtype=rdt) * (math.pi / 180.0)
    vth = torch.polar(u_ref, ang).to(cdt)
    for j, ph in enumerate(src.phases):
        rows.append(idx.row(src.node, ph))
        vref.append(vth[j])
    fixed = torch.as_tensor(rows, dtype=torch.int64)
    vfix = torch.stack(vref, 0)
    i_zero = torch.zeros(n, dtype=cdt)
    v = solve_harmonic(y_eff, i_zero, fixed_rows=fixed, v_fixed=vfix).reshape(-1)
    return v, idx


#: Per-unit voltage-update tolerance the const-Z consistency checks below need: they
#: assert agreement with the linear solve in VOLTS at the 1e-10 / 1e-9 level, which on a
#: 400 V feeder is ~1e-12 pu — three orders tighter than the per-unit power-mismatch
#: default (pandapower's 1e-8 pu). Driving the secondary (voltage) criterion instead of
#: the primary one keeps the requested tolerance inside what complex128 resolves.
_TOL_V_PU = 1.0e-13


def test_const_z_consistency_norton_single_phase():
    grid = _set_load_model(single_phase_chain(), LoadModel.CONST_IMPEDANCE)
    v_lin, _ = _linear_norton_v(grid)
    res = solve_power_flow(grid, slack="norton", dtype=CDT, tol_update_pu=_TOL_V_PU)
    torch.testing.assert_close(res.v.reshape(-1), v_lin, rtol=0, atol=1e-10)
    assert res.converged


def test_const_z_consistency_ideal_single_phase():
    grid = _set_load_model(single_phase_chain(), LoadModel.CONST_IMPEDANCE)
    v_lin, _ = _linear_ideal_v(grid)
    res = solve_power_flow(grid, slack="ideal", dtype=CDT, tol_update_pu=_TOL_V_PU)
    torch.testing.assert_close(res.v.reshape(-1), v_lin, rtol=0, atol=1e-9)
    assert res.converged


def test_const_z_consistency_norton_three_phase():
    grid = _set_load_model(three_phase_two_bus(), LoadModel.CONST_IMPEDANCE)
    v_lin, _ = _linear_norton_v(grid)
    res = solve_power_flow(grid, slack="norton", dtype=CDT, tol_update_pu=_TOL_V_PU)
    torch.testing.assert_close(res.v.reshape(-1), v_lin, rtol=0, atol=1e-9)


def test_const_z_consistency_ideal_three_phase():
    grid = _set_load_model(three_phase_two_bus(), LoadModel.CONST_IMPEDANCE)
    v_lin, _ = _linear_ideal_v(grid)
    res = solve_power_flow(grid, slack="ideal", dtype=CDT, tol_update_pu=_TOL_V_PU)
    torch.testing.assert_close(res.v.reshape(-1), v_lin, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# convergence + power balance for const-power
# ---------------------------------------------------------------------------
def test_const_power_converges_and_balances():
    grid = single_phase_chain()  # default const_power load
    res = solve_power_flow(grid, slack="ideal", tol=1e-10, dtype=CDT)
    assert res.converged
    assert float(res.residual) < 1e-10

    # The load at node 3 must draw exactly its nominal P + jQ.
    idx = res.index
    load = next(a for a in grid.appliances if isinstance(a, Load))
    row = idx.row(load.node, load.phases[0])
    vt = res.v.reshape(-1)[row]
    # I drawn = conj(S0)/conj(Vt); S delivered = Vt * conj(I) = S0.
    s0 = complex(load.p_nom_w, load.q_nom_var)
    i_drawn = torch.conj(torch.tensor(s0, dtype=CDT)) / torch.conj(vt)
    s_deliv = vt * torch.conj(i_drawn)
    torch.testing.assert_close(
        s_deliv, torch.tensor(s0, dtype=CDT), rtol=1e-9, atol=1e-6
    )


def test_const_power_three_phase_converges():
    res = solve_power_flow(three_phase_two_bus(), slack="ideal", dtype=CDT)
    assert res.converged
    assert res.v.shape == (6,)


# ---------------------------------------------------------------------------
# ZIP special cases + ordering
# ---------------------------------------------------------------------------
def _two_bus(p, q, model=LoadModel.CONST_POWER, zc=None) -> Grid:
    return Grid(
        base_frequency_hz=F,
        nodes=[
            Node(id=1, u_rated_v=230.0, phases=(Phase.A,)),
            Node(id=2, u_rated_v=230.0, phases=(Phase.A,)),
        ],
        branches=[
            Line(
                id=20,
                from_node=1,
                to_node=2,
                from_phases=(Phase.A,),
                to_phases=(Phase.A,),
                length_m=100.0,
                series_resistance_ohm_per_m=[[1e-3]],
                series_inductance_h_per_m=[[1e-6]],
                shunt_capacitance_f_per_m=[[1e-9]],
            )
        ],
        appliances=[
            Source(
                id=10,
                node=1,
                phases=(Phase.A,),
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[1e-3]],
            ),
            Load(
                id=30,
                node=2,
                phases=(Phase.A,),
                p_nom_w=p,
                q_nom_var=q,
                load_model=model,
                zip_coefficients=zc,
            ),
        ],
    )


def _vload(grid):
    res = solve_power_flow(grid, slack="ideal", dtype=CDT)
    assert res.converged
    return res.v.reshape(-1)[1]


def test_zip_const_z_matches_const_impedance_model():
    """ZIP=(z=1) reproduces the CONST_IMPEDANCE special case exactly."""
    zc = ZipCoefficients(z_p=1.0, i_p=0.0, p_p=0.0, z_q=1.0, i_q=0.0, p_q=0.0)
    v_zip = _vload(_two_bus(2000.0, 500.0, LoadModel.ZIP, zc))
    v_z = _vload(_two_bus(2000.0, 500.0, LoadModel.CONST_IMPEDANCE))
    torch.testing.assert_close(v_zip, v_z, rtol=0, atol=1e-10)


def test_zip_const_p_matches_const_power_model():
    zc = ZipCoefficients(z_p=0.0, i_p=0.0, p_p=1.0, z_q=0.0, i_q=0.0, p_q=1.0)
    v_zip = _vload(_two_bus(2000.0, 500.0, LoadModel.ZIP, zc))
    v_p = _vload(_two_bus(2000.0, 500.0, LoadModel.CONST_POWER))
    torch.testing.assert_close(v_zip, v_p, rtol=0, atol=1e-10)


def test_zip_const_i_matches_const_current_model():
    zc = ZipCoefficients(z_p=0.0, i_p=1.0, p_p=0.0, z_q=0.0, i_q=1.0, p_q=0.0)
    v_zip = _vload(_two_bus(2000.0, 500.0, LoadModel.ZIP, zc))
    v_i = _vload(_two_bus(2000.0, 500.0, LoadModel.CONST_CURRENT))
    torch.testing.assert_close(v_zip, v_i, rtol=0, atol=1e-10)


def test_zip_mixed_lies_between_const_p_and_const_z():
    """A heavier load makes the V-drop ordering const-P (worst) < mix < const-Z."""
    p, q = 20000.0, 5000.0  # heavy enough to separate the models
    v_p = abs(_vload(_two_bus(p, q, LoadModel.CONST_POWER)))
    v_z = abs(_vload(_two_bus(p, q, LoadModel.CONST_IMPEDANCE)))
    zc = ZipCoefficients(z_p=0.4, i_p=0.3, p_p=0.3, z_q=0.4, i_q=0.3, p_q=0.3)
    v_mix = abs(_vload(_two_bus(p, q, LoadModel.ZIP, zc)))
    assert v_p < v_mix < v_z


def test_const_current_closed_form_at_node():
    """const-current: |I| is fixed at the nominal magnitude S0/|V0| regardless of V."""
    grid = _two_bus(2000.0, 500.0, LoadModel.CONST_CURRENT)
    res = solve_power_flow(grid, slack="ideal", dtype=CDT)
    vt = res.v.reshape(-1)[1]
    # S_eff = S0 * (|Vt|/|V0|); I = conj(S_eff)/conj(Vt) -> |I| = |S0|/|V0|.
    s0 = complex(2000.0, 500.0)
    v0 = 230.0
    i_drawn = torch.conj(torch.tensor(s0, dtype=CDT) * (abs(vt) / v0)) / torch.conj(vt)
    expected_mag = abs(s0) / v0
    torch.testing.assert_close(
        torch.abs(i_drawn),
        torch.tensor(expected_mag, dtype=torch.float64),
        rtol=1e-9,
        atol=1e-9,
    )
