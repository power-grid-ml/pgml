"""CPU-vs-CUDA parity for the Woodbury switch-state update-solve (needs CUDA).

A batched switch sweep is the GPU's case: the assemble path stores one ``[N, N]``
admittance per state, while the low-rank path stores one base factorization plus a
``[S, k, k]`` capacitance matrix. These checks assert that the path runs unchanged
on CUDA — every derived tensor lands on the input's device, the requested dtype is
honored, the solved voltages match the CPU run, and gradients still flow through
the implicit-function-theorem backward.
"""

from __future__ import annotations

import pytest
import torch

from pgml.assembly import assemble_ybus, node_phase_index
from pgml.grids import synthetic_feeder
from pgml.solver import prepare_power_flow, solve_power_flow
from pgml.solver.harmonic import lu_factor_system
from pgml.solver.lowrank import (
    LowRankUpdate,
    branch_state_terms,
    low_rank_update,
    solve_factored_updated,
)

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():  # pragma: no cover - CPU-only host
    pytest.skip("CUDA not available", allow_module_level=True)

CUDA = torch.device("cuda")
TIE = 30000


@pytest.fixture(scope="module")
def grid():
    return synthetic_feeder(24, n_feeders=4, tie_switches=3)


def _states(dtype=torch.float64, device=None):
    s = torch.tensor([0.0, 1.0, 0.5, 0.25], dtype=dtype, device=device)
    return {30000 + i: s.roll(i) for i in range(3)}


def test_update_terms_live_on_the_input_device(grid):
    index = node_phase_index(grid)
    states = _states(device=CUDA)
    base = {bid: 0.0 for bid in states}
    y = assemble_ybus(grid, [50.0], branch_states=base, device=CUDA).Y
    u, c = branch_state_terms(grid, index, states, 50.0, device=CUDA, base_states=0.0)
    assert u.device.type == c.device.type == "cuda"
    upd = low_rank_update(lu_factor_system(y), u, c)
    assert upd.w.device.type == upd.k_lu.device.type == "cuda"
    b = torch.randn(y.shape[-1], dtype=y.dtype, device=CUDA)
    v = solve_factored_updated(upd, b)
    assert v.device.type == "cuda" and v.dtype == y.dtype

    y_cpu = assemble_ybus(grid, [50.0], branch_states=base).Y
    u_c, c_c = branch_state_terms(grid, index, _states(), 50.0, base_states=0.0)
    v_cpu = solve_factored_updated(lu_factor_system(y_cpu), b.cpu(), u=u_c, c=c_c)
    assert torch.allclose(v.cpu(), v_cpu, atol=1e-9 * float(v_cpu.abs().max()))


@pytest.mark.parametrize("slack", ["ideal", "norton"])
def test_power_flow_sweep_parity(grid, slack):
    ref = solve_power_flow(grid, branch_states=_states(), slack=slack)
    got = solve_power_flow(
        grid,
        branch_states=_states(device=CUDA),
        slack=slack,
        device=CUDA,
        branch_states_method="woodbury",
    )
    assert got.converged and got.v.device.type == "cuda"
    assert torch.allclose(ref.v, got.v.cpu(), atol=1e-6)


def test_complex64_honors_dtype(grid):
    system = prepare_power_flow(
        grid,
        dtype=torch.complex64,
        device=CUDA,
        branch_states=_states(dtype=torch.float32, device=CUDA),
        branch_states_method="woodbury",
    )
    assert isinstance(system.factorization, LowRankUpdate)
    assert system.factorization.k_lu.dtype == torch.complex64
    res = solve_power_flow(
        grid,
        dtype=torch.complex64,
        device=CUDA,
        branch_states=_states(dtype=torch.float32, device=CUDA),
        branch_states_method="woodbury",
        system=system,
    )
    assert res.converged and res.v.dtype == torch.complex64
    ref = solve_power_flow(grid, branch_states=_states())
    rel = (
        (res.v.abs().cpu().to(torch.float64) - ref.v.abs()).abs() / ref.v.abs()
    ).max()
    assert float(rel) < 1e-4  # single-precision floor, not a parity claim


def test_gradients_flow_on_cuda(grid):
    """The IFT backward reaches the state leaf through the CUDA low-rank forward."""
    # Load ONE feeder heavily so the tie switch actually carries current (its state
    # is otherwise nearly irrelevant between two identically loaded feeder ends).
    op = {a.id: {"p_w": 3.0e6} for a in grid.appliances if (a.id - 20000) % 4 == 1}
    grads = []
    for device in (torch.device("cpu"), CUDA):
        s = torch.tensor(
            [0.3, 0.9], dtype=torch.float64, device=device, requires_grad=True
        )
        res = solve_power_flow(
            grid,
            device=device,
            operating_point=op,
            branch_states={TIE: s},
            branch_states_method="woodbury",
        )
        res.v.abs().sum().backward()
        assert s.grad is not None and torch.isfinite(s.grad).all()
        grads.append(s.grad.cpu())
    assert torch.allclose(grads[0], grads[1], rtol=1e-6, atol=1e-9)
