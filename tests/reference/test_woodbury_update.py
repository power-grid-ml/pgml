"""Woodbury low-rank update-solve: parity with assembling every state.

A switch-state sweep changes ``Y`` only through the switched branches' primitive
stamps, a rank-``≤ 2P`` term per branch, so one base factorization plus a low-rank
correction must reproduce — to round-off — the per-state assemble-and-factor path
(:func:`pgml.solver.solve_power_flow` with ``branch_states_method="assemble"``).
Checked here at the linear-algebra level (:mod:`pgml.solver.lowrank` against a
fresh dense solve of the modified matrix) and end to end on grids with 1-4
switched branches of every kind the stamp registry supports — lines, switches and
a transformer — in both slack modes, with scalar, batched and cartesian
(states x operating point) batching, and at the exact open (``s = 0``), closed
(``s = 1``) and intermediate scalings.
"""

from __future__ import annotations

import pytest
import torch

from pgml.assembly import assemble_ybus, node_phase_index
from pgml.convert.pandapower import PhaseMode
from pgml.errors import InputError
from pgml.grids import cigre_lv_full_grid, synthetic_feeder
from pgml.schemas.grid_schema import Line, Transformer
from pgml.solver import prepare_power_flow, solve_power_flow
from pgml.solver.harmonic import lu_factor_system, solve_factored
from pgml.solver.lowrank import (
    LowRankUpdate,
    branch_state_terms,
    low_rank_update,
    solve_factored_updated,
)

RTOL = 1.0e-9
TIE = 30000  # first tie switch of synthetic_feeder
LINE1 = 10001  # first line segment of feeder 1


def _close(a, b, rtol=RTOL):
    return torch.allclose(a, b, rtol=rtol, atol=rtol * float(b.abs().max()))


@pytest.fixture(scope="module")
def feeder():
    """Two feeders joined by one normally-open tie switch."""
    return synthetic_feeder(20, n_feeders=2, tie_switches=1)


@pytest.fixture(scope="module")
def meshed():
    """Four feeders joined by three ties (up to k = 3 x 6 = 18 update columns)."""
    return synthetic_feeder(40, n_feeders=4, tie_switches=3)


@pytest.fixture(scope="module")
def cigre():
    """CIGRE LV (three 20/0.4 kV transformers + 37 lines), three-phase."""
    grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.THREE_PHASE)
    return grid


# --- linear-system level ---------------------------------------------------- #
def _terms(grid, states, base_states=0.0):
    index = node_phase_index(grid)
    return branch_state_terms(grid, index, states, 50.0, base_states=base_states)


def _base_y(grid, states, base_states=0.0):
    """The base admittance matching ``_terms``' base (every switched branch at it)."""
    base = {int(bid): float(base_states) for bid in states}
    return assemble_ybus(grid, [50.0], branch_states=base).Y


@pytest.mark.parametrize("state", [0.0, 0.25, 1.0])
def test_updated_solve_matches_a_fresh_factorization(feeder, state):
    """``(A + UCU^H)^-1 b`` equals a dense solve of the assembled state matrix."""
    states = {TIE: state}
    y_base = _base_y(feeder, states)
    y_state = assemble_ybus(feeder, [50.0], branch_states=states).Y
    u, c = _terms(feeder, states)
    b = torch.randn(y_base.shape[-1], dtype=y_base.dtype)
    ref = torch.linalg.solve(y_state.squeeze(0), b)
    got = solve_factored_updated(lu_factor_system(y_base), b, u=u, c=c).squeeze(0)
    assert _close(got, ref)


@pytest.mark.parametrize("state", [0.5, 1.0])
def test_updated_solve_from_a_closed_base(feeder, state):
    """The DOWNDATE direction (base holds the branch) on a finite-impedance branch."""
    states = {LINE1: state}
    y_base = _base_y(feeder, states, base_states=1.0)
    y_state = assemble_ybus(feeder, [50.0], branch_states=states).Y
    u, c = _terms(feeder, states, base_states=1.0)
    b = torch.randn(y_base.shape[-1], dtype=y_base.dtype)
    ref = torch.linalg.solve(y_state.squeeze(0), b)
    got = solve_factored_updated(lu_factor_system(y_base), b, u=u, c=c).squeeze(0)
    assert _close(got, ref)


def test_zero_core_reproduces_the_base_solve_exactly(feeder):
    """``C = 0`` (every branch at the base state) is the identity update."""
    y = assemble_ybus(feeder, [50.0], branch_states={TIE: 1.0}).Y
    u, c = _terms(feeder, {TIE: 1.0}, base_states=1.0)  # C == 0
    assert float(c.abs().max()) == 0.0
    fac = lu_factor_system(y)
    b = torch.randn(4, y.shape[-1], dtype=y.dtype)
    assert torch.equal(solve_factored_updated(fac, b, u=u, c=c), solve_factored(fac, b))


def test_batched_states_share_one_base_factorization(meshed):
    """One update object answers a whole batch of states, matching per-state solves."""
    s = torch.tensor([0.0, 1.0, 0.5, 0.75], dtype=torch.float64)
    states = {30000 + i: s.roll(i) for i in range(3)}
    y_base = _base_y(meshed, states)
    u, c = _terms(meshed, states)
    upd = low_rank_update(lu_factor_system(y_base), u, c)
    n = y_base.shape[-1]
    b = torch.randn(n, dtype=y_base.dtype)
    got = solve_factored_updated(upd, b)
    assert got.shape == (4, n)
    y_states = assemble_ybus(meshed, [50.0], branch_states=states).Y.squeeze(-3)
    ref = torch.linalg.solve(y_states, b.expand(4, n).unsqueeze(-1)).squeeze(-1)
    assert _close(got, ref)
    assert upd.rank == 3 * 6  # three 3-phase switches, 2P rows each


def test_ideal_slack_update_touching_a_slack_row(meshed):
    """A switched branch incident to the slack modifies ``Y_fs`` too."""
    states = {LINE1: torch.tensor([0.6, 1.0], dtype=torch.float64)}  # leaves node 0
    grid = meshed
    index = node_phase_index(grid)
    slack_rows = torch.tensor(
        [index.row(0, ph) for ph in grid.nodes[0].phases], dtype=torch.int64
    )
    y_base = _base_y(grid, states, base_states=1.0)
    y_states = assemble_ybus(grid, [50.0], branch_states=states).Y.squeeze(-3)
    u, c = _terms(grid, states, base_states=1.0)
    n = y_base.shape[-1]
    v_fixed = torch.full((slack_rows.numel(),), 11547.0 + 0j, dtype=y_base.dtype)
    b = torch.randn(n, dtype=y_base.dtype)
    got = solve_factored_updated(
        lu_factor_system(y_base, fixed_rows=slack_rows),
        b,
        u=u,
        c=c,
        v_fixed=v_fixed,
    )
    from pgml.solver import solve_harmonic

    ref = solve_harmonic(y_states, b, fixed_rows=slack_rows, v_fixed=v_fixed)
    assert _close(got, ref)
    assert _close(got.index_select(-1, slack_rows), v_fixed.expand(2, 3))


@pytest.mark.parametrize("backend", ["dense", "sparse", "block"])
@pytest.mark.parametrize("ideal", [False, True])
def test_every_factorization_backend_supports_the_update(feeder, backend, ideal):
    """Dense, sparse (SuperLU) and block-diagonal factors all take the update."""
    states = {TIE: torch.tensor([0.0, 0.5], dtype=torch.float64)}
    y_base = _base_y(feeder, states)
    u, c = _terms(feeder, states)
    n = y_base.shape[-1]
    b = torch.randn(n, dtype=y_base.dtype)
    kw = {}
    if ideal:
        index = node_phase_index(feeder)
        kw["fixed_rows"] = torch.tensor(
            [index.row(0, ph) for ph in feeder.nodes[0].phases], dtype=torch.int64
        )
        kw["v_fixed"] = torch.full((3,), 11547.0 + 0j, dtype=y_base.dtype)
    # a single feeder is one galvanically independent block
    rows = [torch.arange(n)] if backend == "block" else None
    ref = solve_factored_updated(
        lu_factor_system(y_base, backend="dense", fixed_rows=kw.get("fixed_rows")),
        b,
        u=u,
        c=c,
        v_fixed=kw.get("v_fixed"),
    )
    got = solve_factored_updated(
        lu_factor_system(
            y_base,
            backend=backend,
            block_rows=rows,
            fixed_rows=kw.get("fixed_rows"),
        ),
        b,
        u=u,
        c=c,
        v_fixed=kw.get("v_fixed"),
    )
    assert _close(got, ref)


def test_shape_errors_are_explicit(feeder):
    y = assemble_ybus(feeder, [50.0], branch_states={TIE: 1.0}).Y
    fac = lu_factor_system(y)
    u, c = _terms(feeder, {TIE: 0.0}, base_states=1.0)
    b = torch.randn(y.shape[-1], dtype=y.dtype)
    with pytest.raises(InputError, match=r"\[N, k\]"):
        solve_factored_updated(fac, b, u=u[:-1], c=c)
    with pytest.raises(InputError, match=r"k, k"):
        solve_factored_updated(fac, b, u=u, c=c[:-1])
    with pytest.raises(InputError, match="not both"):
        solve_factored_updated(low_rank_update(fac, u, c), b, u=u, c=c)
    with pytest.raises(InputError, match="needs the update terms"):
        solve_factored_updated(fac, b)


# --- end-to-end power flow -------------------------------------------------- #
@pytest.mark.parametrize("slack", ["ideal", "norton"])
def test_power_flow_sweep_matches_assemble_per_state(feeder, slack):
    """s = 0 (open), 1 (closed) and an intermediate scaling, both slack modes."""
    states = {TIE: torch.tensor([0.0, 1.0, 0.5], dtype=torch.float64)}
    ref = solve_power_flow(feeder, branch_states=states, slack=slack)
    got = solve_power_flow(
        feeder, branch_states=states, slack=slack, branch_states_method="woodbury"
    )
    assert got.converged and got.v.shape == ref.v.shape
    assert _close(got.v, ref.v)


def test_power_flow_scalar_state(feeder):
    for s in (0.0, 0.5, 1.0):
        ref = solve_power_flow(feeder, branch_states={TIE: s})
        got = solve_power_flow(
            feeder, branch_states={TIE: s}, branch_states_method="woodbury"
        )
        assert _close(got.v, ref.v)


def test_power_flow_multi_branch_and_cartesian_operating_point(meshed):
    """Three switched branches x a batched operating point (cartesian broadcast)."""
    s = torch.tensor([0.0, 1.0, 0.5, 1.0], dtype=torch.float64)
    states = {30000 + i: s.roll(i) for i in range(3)}
    op = {20001: {"p_w": torch.tensor([[1.0e5], [2.0e5]], dtype=torch.float64)}}
    ref = solve_power_flow(meshed, branch_states=states, operating_point=op)
    got = solve_power_flow(
        meshed,
        branch_states=states,
        operating_point=op,
        branch_states_method="woodbury",
    )
    assert got.v.shape == (2, 4, got.index.size)
    assert _close(got.v, ref.v)


def test_power_flow_transformer_and_line_states(cigre):
    """A transformer's vector-group primitive is as low-rank as any other stamp."""
    xf = next(b.id for b in cigre.branches if isinstance(b, Transformer))
    lines = [b.id for b in cigre.branches if isinstance(b, Line)]
    s = torch.tensor([1.0, 0.5], dtype=torch.float64)
    states = {
        xf: s,
        lines[3]: torch.tensor([1.0, 1.0], dtype=torch.float64),
        lines[7]: torch.tensor([0.9, 1.0], dtype=torch.float64),
    }
    ref = solve_power_flow(cigre, branch_states=states)
    got = solve_power_flow(cigre, branch_states=states, branch_states_method="woodbury")
    assert got.converged
    assert _close(got.v, ref.v)
    # the sweep really moves the network (otherwise the parity is vacuous)
    assert float((got.v[0] - got.v[1]).abs().max()) > 1.0


def test_near_ideal_switch_stays_accurate(feeder):
    """A milli/micro-ohm switch: the base OMITS it, so the update only ADDS."""
    for r_sw in (1.0e-4, 1.0e-6):
        grid = feeder.model_copy(
            update={
                "branches": [
                    b.model_copy(update={"resistance_ohm": r_sw}) if b.id == TIE else b
                    for b in feeder.branches
                ]
            }
        )
        states = {TIE: torch.tensor([0.0, 1.0], dtype=torch.float64)}
        ref = solve_power_flow(grid, branch_states=states)
        got = solve_power_flow(
            grid, branch_states=states, branch_states_method="woodbury"
        )
        assert got.converged, f"r_sw={r_sw}"
        assert _close(got.v, ref.v), f"r_sw={r_sw}"


def test_bridge_branch_keeps_the_base_connected(meshed):
    """A branch the base cannot drop (a bridge) stays in it; parity is unaffected."""
    states = {LINE1: torch.tensor([1.0, 0.7], dtype=torch.float64)}
    ref = solve_power_flow(meshed, branch_states=states)
    got = solve_power_flow(
        meshed, branch_states=states, branch_states_method="woodbury"
    )
    assert _close(got.v, ref.v)


def test_prepared_woodbury_system_is_reusable(feeder):
    states = {TIE: torch.tensor([0.0, 1.0], dtype=torch.float64)}
    system = prepare_power_flow(
        feeder, branch_states=states, branch_states_method="woodbury"
    )
    assert isinstance(system.factorization, LowRankUpdate)
    ref = solve_power_flow(feeder, branch_states=states)
    got = solve_power_flow(
        feeder,
        branch_states=states,
        branch_states_method="woodbury",
        system=system,
    )
    assert _close(got.v, ref.v)


def test_method_mismatch_and_unsupported_combinations(feeder):
    states = {TIE: 0.5}
    with pytest.raises(InputError, match="branch_states_method"):
        solve_power_flow(feeder, branch_states=states, branch_states_method="auto")
    with pytest.raises(InputError, match="needs branch_states"):
        solve_power_flow(feeder, branch_states_method="woodbury")
    with pytest.raises(InputError, match="current-injection"):
        solve_power_flow(
            feeder,
            branch_states=states,
            method="newton",
            branch_states_method="woodbury",
        )
    system = prepare_power_flow(feeder, branch_states=states)
    with pytest.raises(InputError, match="different branch_states_method"):
        solve_power_flow(
            feeder,
            branch_states=states,
            branch_states_method="woodbury",
            system=system,
        )
