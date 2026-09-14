"""Differentiability gate for the voltage-regulating generator (PV terminal).

The regulated terminal replaces a reactive power-balance row with
``|V|**2 - V_set**2`` (``pgml.solver._pv_bus``), so the implicit-function-theorem
backward has to carry gradients through a residual row the other appliances never
touch. The quantities checked with float64 ``gradcheck``:

- ``dV*/dv_set`` — the setpoint, from the schema field and from a per-scenario
  operating-point override (the batched form);
- ``dV*/dP_gen`` — the regulated machine's active power, which still enters the
  active-balance row of the same terminal;
- ``dV*/dq_limit`` — the BINDING reactive limit of a terminal that has switched to
  PQ: the limit value then enters the residual as that machine's reactive power, so
  the gradient is the ordinary injection sensitivity at a FIXED active set (the
  switching decision itself is piecewise constant in the parameters and is
  differentiated nowhere, exactly as a complementarity constraint's active set is
  not);
- ``dV*/dR_line`` — an ordinary network parameter, to show the PV row does not break
  the existing path;
- ``dV*/du_rated`` — the node rating, which the setpoint's per-unit base refers to.

The replacement row is quadratic in ``V`` (no ``abs``, no ``sqrt``), so the residual is
smooth and no smoothing parameter is needed. Every check runs at float64 with the
Newton forward; the finite-difference step is chosen against the scale of the
perturbed quantity.
"""

from __future__ import annotations

import math

import torch

from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    Line,
    Load,
    Node,
    Phase,
    RegulatedQuantity,
    Source,
    VoltageRegulation,
)
from pgml.solver import solve_power_flow

CDT = torch.complex128
RDT = torch.float64
ABC = (Phase.A, Phase.B, Phase.C)
U_RATED = 400.0
GEN_ID = 2


def _src():
    return Source(
        id=0,
        node=0,
        phases=ABC,
        u_ref_v=[U_RATED / math.sqrt(3.0)] * 3,
        u_angle_deg=[0.0, -120.0, 120.0],
        resistance_ohm=[[0.05 if i == j else 0.0 for j in range(3)] for i in range(3)],
        inductance_h=[[2e-4 if i == j else 0.0 for j in range(3)] for i in range(3)],
    )


def _line(r=None):
    return Line(
        id=0,
        from_node=0,
        to_node=1,
        from_phases=ABC,
        to_phases=ABC,
        length_m=300.0,
        series_resistance_ohm_per_m=(
            r
            if r is not None
            else [[3e-4 if i == j else 0.0 for j in range(3)] for i in range(3)]
        ),
        series_inductance_h_per_m=[
            [1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)
        ],
        shunt_capacitance_f_per_m=[[0.0] * 3 for _ in range(3)],
    )


def _nodes(u_rated=U_RATED):
    return [
        Node(id=0, u_rated_v=U_RATED, phases=ABC),
        Node(id=1, u_rated_v=u_rated, phases=ABC),
    ]


def _grid(
    *,
    v_set=1.02,
    p_gen=6000.0,
    q_min=None,
    q_max=None,
    u_rated=U_RATED,
    r=None,
    regulated=RegulatedQuantity.POSITIVE_SEQUENCE,
):
    gen = Generator(
        id=GEN_ID,
        node=1,
        phases=ABC,
        p_nom_w=p_gen,
        voltage_regulation=VoltageRegulation(
            v_set_pu=v_set, q_min_var=q_min, q_max_var=q_max, regulated=regulated
        ),
    )
    load = Load(id=3, node=1, phases=ABC, p_nom_w=9000.0, q_nom_var=3000.0)
    return Grid(
        nodes=_nodes(u_rated), branches=[_line(r)], appliances=[_src(), load, gen]
    )


def _solve(grid, **kw):
    return solve_power_flow(
        grid,
        method="newton",
        tol=1e-10,
        max_iter=60,
        dtype=CDT,
        criticality="never",
        **kw,
    ).v


def test_gradcheck_voltage_setpoint():
    """``dV*/dv_set``: the setpoint row's own parameter."""
    v_set = torch.tensor(1.02, dtype=RDT, requires_grad=True)

    def fn(v_set):
        return _solve(_grid(v_set=v_set))

    assert torch.autograd.gradcheck(fn, (v_set,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_voltage_setpoint_per_phase_mode():
    """The per-phase regulated quantity is differentiable in the same way."""
    v_set = torch.tensor(1.02, dtype=RDT, requires_grad=True)

    def fn(v_set):
        return _solve(_grid(v_set=v_set, regulated=RegulatedQuantity.PER_PHASE))

    assert torch.autograd.gradcheck(fn, (v_set,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_batched_setpoint_override():
    """A per-scenario setpoint (the operating-point form) carries per-scenario
    gradients through the batched IFT backward."""
    v_set = torch.tensor([0.99, 1.02, 1.04], dtype=RDT, requires_grad=True)

    def fn(v_set):
        return _solve(_grid(), operating_point={GEN_ID: {"v_set_pu": v_set}})

    assert torch.autograd.gradcheck(fn, (v_set,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_generator_active_power():
    """``dV*/dP_gen``: the regulated terminal's active-balance row."""
    p = torch.tensor(6000.0, dtype=RDT, requires_grad=True)

    def fn(p):
        return _solve(_grid(p_gen=p))

    assert torch.autograd.gradcheck(fn, (p,), eps=1e-2, atol=1e-5, rtol=1e-3)


def test_gradcheck_binding_reactive_limit():
    """``dV*/dq_max`` at a terminal pinned by its upper limit.

    The setpoint needs far more reactive power than 1.5 kvar, so the unit switches to
    PQ at ``q_max`` and the limit becomes its reactive injection. The active set is
    the same for every finite-difference step (the limit stays binding), so the
    gradient is exact there.
    """
    q_max = torch.tensor(1500.0, dtype=RDT, requires_grad=True)

    def fn(q_max):
        grid = _grid(v_set=1.08, q_min=-1500.0, q_max=q_max)
        res = solve_power_flow(
            grid,
            method="newton",
            tol=1e-10,
            max_iter=60,
            dtype=CDT,
            criticality="never",
        )
        assert not bool(res.regulation.regulating[GEN_ID]), "the limit must bind"
        return res.v

    assert torch.autograd.gradcheck(fn, (q_max,), eps=1e-2, atol=1e-5, rtol=1e-3)


def test_gradcheck_line_resistance_with_a_regulated_terminal():
    """An ordinary network parameter still gets its gradient with a PV row present."""
    r = torch.tensor(
        [[3e-4 if i == j else 0.0 for j in range(3)] for i in range(3)],
        dtype=RDT,
        requires_grad=True,
    )

    def fn(r):
        return _solve(_grid(r=r))

    assert torch.autograd.gradcheck(fn, (r,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_node_rating_behind_the_per_unit_setpoint():
    """``dV*/du_rated``: the regulated magnitude is ``v_set`` times the node's
    line-to-neutral base, so the rating is on the differentiable path too."""
    u_rated = torch.tensor(U_RATED, dtype=RDT, requires_grad=True)

    def fn(u_rated):
        return _solve(_grid(u_rated=u_rated))

    assert torch.autograd.gradcheck(fn, (u_rated,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_finite_difference_spot_check_of_dv_dvset():
    """One hand-computed sensitivity, to back the gradcheck assertions.

    ``d|V_1| / dv_set`` must be the node's line-to-neutral base: the regulated
    magnitude IS ``v_set`` times that base, a number known without the solver.
    """
    v_set = torch.tensor(1.02, dtype=RDT, requires_grad=True)
    v = _solve(_grid(v_set=v_set))
    row = 3  # node 1 phase A: rows are (0,a), (0,b), (0,c), (1,a), ...
    (grad,) = torch.autograd.grad(v[row].abs(), v_set)
    assert abs(float(grad) - U_RATED / math.sqrt(3.0)) < 1e-6 * U_RATED
