"""Topology / switch-state batching via branch_states (admittance masking).

One assembly covers a whole batch of switch configurations: each listed branch's
primitive stamp is scaled by its (possibly batched, possibly continuous) state,
overriding the static ``in_service`` / ``closed`` flags. Verified here: masking
semantics in the assembly, batched == per-configuration loop in the solver,
cartesian broadcasting against a batched operating point, per-scenario
connectivity errors, branch currents, harmonics, and Newton parity.
"""

from __future__ import annotations

import pytest
import torch

from pgml.assembly import assemble_network_ybus, branch_currents
from pgml.errors import ConnectivityError, InputError
from pgml.grids import synthetic_feeder
from pgml.solver import solve_harmonic_flow, solve_power_flow

TIE = 30000  # first tie switch id of synthetic_feeder
LINE1 = 10001  # first segment of feeder 1


@pytest.fixture(scope="module")
def grid():
    return synthetic_feeder(20, n_feeders=2, tie_switches=1)


# ---------------------------------------------------------------------------
# assembly masking semantics
# ---------------------------------------------------------------------------
def test_state_zero_equals_unstamped_and_one_equals_closed(grid):
    y_open = assemble_network_ybus(grid, [50.0]).Y  # tie is open by default
    closed = grid.model_copy(
        update={
            "branches": [
                b.model_copy(update={"closed": True}) if b.id == TIE else b
                for b in grid.branches
            ]
        }
    )
    y_closed = assemble_network_ybus(closed, [50.0]).Y
    assert torch.allclose(
        assemble_network_ybus(grid, [50.0], branch_states={TIE: 0.0}).Y, y_open
    )
    assert torch.allclose(
        assemble_network_ybus(grid, [50.0], branch_states={TIE: 1.0}).Y, y_closed
    )


def test_state_overrides_static_flags(grid):
    """State 1 on an out-of-service line stamps it as if in service."""
    off = grid.model_copy(
        update={
            "branches": [
                b.model_copy(update={"in_service": False}) if b.id == LINE1 else b
                for b in grid.branches
            ]
        }
    )
    y_ref = assemble_network_ybus(grid, [50.0]).Y
    y_masked = assemble_network_ybus(off, [50.0], branch_states={LINE1: 1.0}).Y
    assert torch.allclose(y_masked, y_ref)


def test_batched_states_promote_y(grid):
    s = torch.tensor([0.0, 1.0, 0.5])
    y = assemble_network_ybus(grid, [50.0], branch_states={TIE: s}).Y
    n = y.shape[-1]
    assert y.shape == (3, 1, n, n)
    y0 = assemble_network_ybus(grid, [50.0]).Y
    y1 = assemble_network_ybus(grid, [50.0], branch_states={TIE: 1.0}).Y
    assert torch.allclose(y[0], y0)
    assert torch.allclose(y[1], y1)
    # a continuous state scales the tie's stamp linearly between open and closed
    assert torch.allclose(y[2], y0 + 0.5 * (y1 - y0), atol=1e-12)


# ---------------------------------------------------------------------------
# solver: batched == loop, cartesian broadcast
# ---------------------------------------------------------------------------
def test_solve_batched_equals_per_config_loop(grid):
    s = torch.tensor([0.0, 1.0, 0.5], dtype=torch.float64)
    rb = solve_power_flow(grid, branch_states={TIE: s})
    assert rb.converged
    assert rb.v.shape == (3, rb.index.size)
    for i, sv in enumerate(s.tolist()):
        ri = solve_power_flow(grid, branch_states={TIE: sv})
        assert torch.allclose(rb.v[i], ri.v, atol=1e-6)


def test_cartesian_states_times_operating_point(grid):
    s = torch.tensor([0.0, 1.0], dtype=torch.float64)  # [K]
    p = torch.tensor([[1.0e5], [2.0e5]], dtype=torch.float64)  # [B, 1]
    r = solve_power_flow(
        grid, branch_states={TIE: s}, operating_point={20001: {"p_w": p}}
    )
    assert r.converged
    assert r.v.shape == (2, 2, r.index.size)
    r00 = solve_power_flow(
        grid, branch_states={TIE: 0.0}, operating_point={20001: {"p_w": 1.0e5}}
    )
    r11 = solve_power_flow(
        grid, branch_states={TIE: 1.0}, operating_point={20001: {"p_w": 2.0e5}}
    )
    assert torch.allclose(r.v[0, 0], r00.v, atol=1e-6)
    assert torch.allclose(r.v[1, 1], r11.v, atol=1e-6)


def test_newton_matches_current_injection(grid):
    s = torch.tensor([0.0, 1.0], dtype=torch.float64)
    rci = solve_power_flow(grid, branch_states={TIE: s})
    rnw = solve_power_flow(grid, branch_states={TIE: s}, method="newton")
    assert rnw.converged
    assert torch.allclose(rci.v, rnw.v, atol=1e-5)


def test_newton_rejects_double_batching(grid):
    s = torch.tensor([0.0, 1.0], dtype=torch.float64)
    op = {20001: {"p_w": torch.tensor([1.0e5, 2.0e5], dtype=torch.float64)}}
    with pytest.raises(InputError, match="newton"):
        solve_power_flow(
            grid, branch_states={TIE: s}, operating_point=op, method="newton"
        )


# ---------------------------------------------------------------------------
# connectivity
# ---------------------------------------------------------------------------
def test_batched_disconnecting_state_raises_with_scenario_indices(grid):
    s = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    with pytest.raises(ConnectivityError, match=r"scenario") as exc:
        solve_power_flow(grid, branch_states={LINE1: s})
    assert "1" in str(exc.value)  # the failing scenario index


def test_scalar_disconnecting_state_raises_static_report(grid):
    with pytest.raises(ConnectivityError, match="no galvanic path"):
        solve_power_flow(grid, branch_states={LINE1: 0.0})


def test_open_tie_state_keeps_grid_connected(grid):
    # the tie is redundant (feeders are radial from the source): any state is fine
    r = solve_power_flow(grid, branch_states={TIE: torch.tensor([0.0, 1.0])})
    assert r.converged


def test_zero_mode_rejected_with_states(grid):
    with pytest.raises(InputError, match="zero"):
        solve_power_flow(grid, branch_states={TIE: 1.0}, on_disconnected="zero")


def test_unknown_branch_id_raises(grid):
    with pytest.raises(InputError, match="unknown branch"):
        solve_power_flow(grid, branch_states={999999: torch.tensor([1.0, 1.0])})


# ---------------------------------------------------------------------------
# branch currents
# ---------------------------------------------------------------------------
def test_branch_currents_scale_with_state(grid):
    s = torch.tensor([0.0, 1.0], dtype=torch.float64)
    states = {TIE: s}
    r = solve_power_flow(grid, branch_states=states)
    bc = branch_currents(grid, r.v, [50.0], r.index, branch_states=states)
    tie = next(c for c in bc if c.branch_id == TIE)
    # scenario 0: open -> exactly zero current; scenario 1: closed -> flowing
    assert torch.allclose(tie.i_from[0], torch.zeros_like(tie.i_from[0]))
    assert float(tie.i_from[1].abs().max()) > 0.0


def test_branch_currents_include_masked_out_of_service_branch(grid):
    off = grid.model_copy(
        update={
            "branches": [
                b.model_copy(update={"in_service": False}) if b.id == LINE1 else b
                for b in grid.branches
            ]
        }
    )
    r = solve_power_flow(off, branch_states={LINE1: 1.0})
    bc = branch_currents(off, r.v, [50.0], r.index, branch_states={LINE1: 1.0})
    assert any(c.branch_id == LINE1 for c in bc)


# ---------------------------------------------------------------------------
# harmonic flow
# ---------------------------------------------------------------------------
def test_harmonic_flow_batched_states(grid):
    from pgml.schemas.grid_schema import (
        HarmonicComponent,
        SpectrumPoint,
        StaticSpectrum,
    )

    g = grid.model_copy(deep=True)
    load = next(a for a in g.appliances if a.id == 20005)
    load.spectrum = StaticSpectrum(
        spectrum=SpectrumPoint(
            components=[
                HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
                HarmonicComponent(order=5, magnitude_pu=0.2, phase_deg=0.0),
            ]
        )
    )
    s = torch.tensor([0.0, 1.0], dtype=torch.float64)
    rb = solve_harmonic_flow(g, [1, 5], branch_states={TIE: s})
    assert rb.converged
    assert rb.v.shape == (2, 2, rb.index.size)
    for i, sv in enumerate(s.tolist()):
        ri = solve_harmonic_flow(g, [1, 5], branch_states={TIE: sv})
        assert torch.allclose(rb.v[i], ri.v, atol=1e-6)
