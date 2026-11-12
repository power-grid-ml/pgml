"""float64 gradcheck through two-phase DELTA appliances and the node L-N rating.

The phase-to-phase element's power reaches the voltages through the nonlinear
fundamental solve (implicit-function adjoint), the harmonic current injection and the
harmonic load shunt; ``Node.u_rated_ln_v`` as a tensor reaches them through the
constant-impedance reference of a wye load on a split-phase node.
"""

from __future__ import annotations

import torch

from pgml.schemas.grid_schema import Load, LoadModel, Phase
from pgml.solver import solve_harmonic_flow, solve_power_flow
from tests.asymmetric.test_two_phase_delta import SPEC, _delta, _grid

CDT = torch.complex128
P = Phase


def _leaf(v: float) -> torch.Tensor:
    return torch.tensor([v], dtype=torch.float64, requires_grad=True)


def _overrides(p, q):
    return {
        ("load", 20, "p_nom_per_phase_w"): p,
        ("load", 20, "q_nom_per_phase_var"): q,
    }


def test_gradcheck_power_flow_wrt_element_power():
    """Constant-power phase-to-phase load: ``dV/dP``, ``dV/dQ`` by the IFT adjoint."""
    grid = _grid([_delta()])

    def fn(p, q):
        res = solve_power_flow(
            grid, dtype=CDT, param_overrides=_overrides(p, q), tol=1e-12
        )
        return torch.view_as_real(res.v).reshape(-1)

    assert torch.autograd.gradcheck(
        fn, (_leaf(9e3), _leaf(3e3)), eps=1e-2, atol=1e-6, rtol=1e-3
    )


def test_gradcheck_harmonic_flow_with_shunt_wrt_element_power():
    """Order-3 voltages: injection and harmonic shunt of a ZIP phase-to-phase load."""
    grid = _grid([_delta(model=LoadModel.CONST_CURRENT, spectrum=SPEC)])

    def fn(p, q):
        res = solve_harmonic_flow(
            grid,
            [1, 3],
            dtype=CDT,
            load_shunt="opendss",
            param_overrides=_overrides(p, q),
            tol=1e-12,
        )
        return torch.view_as_real(res.v).reshape(-1)

    assert torch.autograd.gradcheck(
        fn, (_leaf(9e3), _leaf(3e3)), eps=1e-2, atol=1e-6, rtol=1e-3
    )


def test_gradcheck_split_phase_rating_leaf():
    """A tensor ``u_rated_ln_v`` reaches the voltages through the const-Z reference."""
    abn = (P.A, P.B, P.N)
    half = Load(
        id=21,
        node=2,
        phases=(P.A,),
        p_nom_w=4e3,
        q_nom_var=1e3,
        load_model=LoadModel.CONST_IMPEDANCE,
        spectrum=SPEC,
    )

    def fn(ln):
        base = _grid(
            [half, _delta(model=LoadModel.CONST_IMPEDANCE)],
            phases=abn,
            u=440.0,
            ln=220.0,
        )
        nodes = [
            n.model_copy(update={"u_rated_ln_v": ln[0]}) if n.id == 2 else n
            for n in base.nodes
        ]
        grid = base.model_copy(update={"nodes": nodes})
        res = solve_harmonic_flow(
            grid, [1, 3], dtype=CDT, load_shunt="opendss", tol=1e-12
        )
        return torch.view_as_real(res.v).reshape(-1)

    ln = _leaf(220.0)
    (grad,) = torch.autograd.grad(fn(ln).square().sum(), ln)
    assert grad.abs().item() > 0.0  # the rating is on the tape
    assert torch.autograd.gradcheck(fn, (ln,), eps=1e-4, atol=1e-6, rtol=1e-3)
