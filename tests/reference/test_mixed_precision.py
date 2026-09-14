"""Mixed-precision solving: complex64 factors, complex128 accuracy.

An SI-unit feeder's admittance is ill-conditioned (no per-unit normalisation anywhere in
the engine), so a plain complex64 solve loses about ``cond(Y) * 1.2e-7`` of relative
accuracy — on a physical feeder that is most of the digits. ``precision="mixed"`` keeps
the factorization and the back-substitutions in single precision and recovers
complex128 accuracy from residuals formed at the working precision:

- the LINEAR solve by classic iterative refinement (:func:`pgml.solver.lu_factor_system`),
- the NONLINEAR solve because its fixed point then runs in residual-correction form, so
  the single-precision factorization only preconditions the iteration and never moves the
  solution it converges to.

These tests pin the accuracy against a complex128 reference on IEEE-33 and CIGRE LV
(three-phase), for all three factorization backends, and the one-time warning a plain
complex64 solve emits when the estimated condition number is high.
"""

from __future__ import annotations

import logging

import pytest
import torch

from pgml.assembly import node_phase_index
from pgml.assembly._params import phase_voltage_magnitude
from pgml.errors import InputError
from pgml.grids import cigre_lv_full_grid, ieee33_geometry_grid
from pgml.solver import solve_power_flow
from pgml.solver.harmonic import estimate_condition, lu_factor_system, solve_factored

CDT = torch.complex128
CF = torch.complex64

#: Accuracy a mixed-precision run must reach against the complex128 reference.
MIXED_ATOL_PU = 1.0e-9


def _bases(grid, index) -> torch.Tensor:
    nb = {int(n.id): n for n in grid.nodes}
    return torch.tensor(
        [
            phase_voltage_magnitude(float(nb[int(i)].u_rated_v), len(nb[int(i)].phases))
            for i in index.node_ids.tolist()
        ],
        dtype=torch.float64,
    )


def _err_pu(v, v_ref, bases) -> float:
    return float(((v.to(CDT) - v_ref).abs() / bases).max())


@pytest.fixture(scope="module")
def ieee33():
    grid, _ = ieee33_geometry_grid()
    return grid


@pytest.fixture(scope="module")
def cigre_3ph():
    from pgml.convert.pandapower import PhaseMode

    grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.THREE_PHASE)
    return grid


def _accuracy_table(grid, **kw) -> dict[str, float]:
    """``{mode: max |ΔV| in pu}`` of complex64 and mixed against complex128."""
    index = node_phase_index(grid)
    bases = _bases(grid, index)
    ref = solve_power_flow(grid, dtype=CDT, tol_update_pu=1e-12, **kw)
    out = {}
    for name, dtype, precision in (
        ("complex64", CF, "full"),
        ("mixed", CDT, "mixed"),
    ):
        res = solve_power_flow(
            grid, dtype=dtype, precision=precision, tol_update_pu=1e-12, **kw
        )
        assert res.converged, f"{name} did not converge"
        out[name] = _err_pu(res.v.detach(), ref.v.detach(), bases)
    return out


def test_mixed_precision_accuracy_ieee33(ieee33):
    """The pinned claim: mixed precision stays below 1e-9 pu where complex64 does not."""
    err = _accuracy_table(ieee33)
    assert err["mixed"] < MIXED_ATOL_PU, err
    # And it is a real gain: the plain single-precision solve is orders worse.
    assert err["complex64"] > 100.0 * err["mixed"], err


def test_mixed_precision_accuracy_cigre_lv_three_phase(cigre_3ph):
    err = _accuracy_table(cigre_3ph)
    assert err["mixed"] < MIXED_ATOL_PU, err
    assert err["complex64"] > 100.0 * err["mixed"], err


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_every_factorization_backend_refines(ieee33, backend):
    err = _accuracy_table(ieee33, linear_solver=backend)
    assert err["mixed"] < MIXED_ATOL_PU, (backend, err)


def test_mixed_precision_newton_matches_full_precision(ieee33):
    """Inexact Newton: a single-precision direction, a complex128 solution."""
    index = node_phase_index(ieee33)
    bases = _bases(ieee33, index)
    ref = solve_power_flow(ieee33, dtype=CDT, method="newton", tol_update_pu=1e-12)
    mixed = solve_power_flow(
        ieee33, dtype=CDT, method="newton", precision="mixed", tol_update_pu=1e-12
    )
    assert mixed.converged
    assert _err_pu(mixed.v.detach(), ref.v.detach(), bases) < MIXED_ATOL_PU


def test_mixed_precision_needs_a_complex128_working_dtype(ieee33):
    """complex64 + mixed is refused: there is no wider precision to refine against."""
    with pytest.raises(InputError, match="complex128"):
        solve_power_flow(ieee33, dtype=CF, precision="mixed")


def test_refined_linear_solve_reaches_double_precision():
    """The refinement itself: a deliberately ill-conditioned system, solved twice.

    The ill conditioning here is pure SCALING (one row multiplied by 1e5, the decade
    spread a stiff source row gives an SI feeder), so it also pins what equilibration
    does with it: the condition estimate of the matrix AS ASSEMBLED is above 1e4, and of
    the same matrix as FACTORED (equilibrated, the default) below 1e3.
    """
    n = 40
    torch.manual_seed(0)
    a = torch.randn(n, n, dtype=CDT) + n * torch.eye(n, dtype=CDT)
    a[0] = a[0] * 1.0e5  # cond ~ 1e5, the scale an SI feeder reaches easily
    b = torch.randn(n, dtype=CDT)
    exact = torch.linalg.solve(a, b)
    single = torch.linalg.solve(a.to(CF), b.to(CF)).to(CDT)
    refined = solve_factored(lu_factor_system(a, precision="mixed"), b)
    rel = lambda x: float((x - exact).abs().max() / exact.abs().max())  # noqa: E731
    assert rel(single) > 1.0e-9  # single precision loses most digits here
    assert rel(refined) < 1.0e-13  # refinement recovers them
    assert estimate_condition(lu_factor_system(a, equilibrate="off")) > 1.0e4
    assert estimate_condition(lu_factor_system(a)) < 1.0e3


def test_plain_complex64_warns_once_on_an_ill_conditioned_system(ieee33, caplog):
    """The documented conditioning warning, with the recommended recipe in it."""
    import pgml.solver.power_flow as pf

    pf._COMPLEX64_COND_CHECKED = False
    try:
        with caplog.at_level(logging.WARNING, logger="pgml"):
            solve_power_flow(ieee33, dtype=CF)
        msgs = [r.message for r in caplog.records if "condition number" in r.message]
        assert msgs, "expected a conditioning warning for a complex64 feeder solve"
        assert "mixed" in msgs[0] and "complex128" in msgs[0]
        # ONE warning per process, not one per solve (a scenario sweep must stay quiet).
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="pgml"):
            solve_power_flow(ieee33, dtype=CF)
        assert not [r for r in caplog.records if "condition number" in r.message]
    finally:
        pf._COMPLEX64_COND_CHECKED = False


def test_mixed_precision_does_not_warn_about_conditioning(ieee33, caplog):
    import pgml.solver.power_flow as pf

    pf._COMPLEX64_COND_CHECKED = False
    try:
        with caplog.at_level(logging.WARNING, logger="pgml"):
            solve_power_flow(ieee33, dtype=CDT, precision="mixed")
        assert not [r for r in caplog.records if "condition number" in r.message]
    finally:
        pf._COMPLEX64_COND_CHECKED = False


def test_prepared_system_must_match_the_requested_precision(ieee33):
    from pgml.solver import prepare_power_flow

    system = prepare_power_flow(ieee33, dtype=CDT, precision="mixed")
    assert solve_power_flow(
        ieee33, system=system, precision="mixed", dtype=CDT
    ).converged
    with pytest.raises(InputError, match="precision"):
        solve_power_flow(ieee33, system=system, dtype=CDT)
