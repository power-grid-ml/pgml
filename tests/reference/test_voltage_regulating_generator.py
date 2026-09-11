"""The voltage-regulating generator (PV terminal) against an independent oracle.

A :class:`~pgml.schemas.grid_schema.Generator` with a
:class:`~pgml.schemas.grid_schema.VoltageRegulation` block holds its terminal voltage
magnitude at the setpoint and supplies whatever reactive power that takes, bounded by
its limits (``docs/pgml/modeling/der-pv-storage.md`` section 4.5). The solver reaches
it by replacing the terminal's reactive power-balance row with ``|V|**2 - V_set**2``.

What is checked here, all against quantities computed OUTSIDE the solver:

- the setpoint is held EXACTLY (not approximately, as the Volt-VAr droop of
  ``GenMode.VOLT_VAR_APPROX`` does), on one- and three-phase terminals;
- the reported reactive power is the one that makes the grid balance: a plain PQ
  generator given that same reactive power reproduces the regulated solution, and a
  hand-written numpy nodal balance at the converged voltages closes to rounding;
- the active power is untouched by the regulation;
- reactive limits bind by PV-to-PQ switching: the unit sits exactly at the violated
  limit, the voltage lands off the setpoint on the correct side, and the solution
  equals the PQ solve at that limit;
- limit enforcement can be turned off (pandapower's ``runpp`` default), and then the
  setpoint is held with an unbounded reactive power;
- per-phase regulation holds every phase magnitude on an UNBALANCED three-phase
  terminal, where positive-sequence regulation holds ``|V1|`` instead;
- the configurations the row substitution cannot express raise.

The cross-tool comparisons live in ``tests/reference/test_pandapower_pv_bus.py``
(pandapower ``net.gen`` on the MATPOWER benchmarks) and
``tests/reference/test_opendss_pv_bus.py`` (OpenDSS ``Generator model=3``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pgml.errors import InputError, ModelingError
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
    WindingConnection,
)
from pgml.solver import solve_power_flow

CDT = torch.complex128
PH1 = (Phase.A,)
PH3 = (Phase.A, Phase.B, Phase.C)
U_RATED = 20_000.0
GEN_ID = 22


def _mat(n: int, diag: float) -> list[list[float]]:
    return [[diag if i == j else 0.0 for j in range(n)] for i in range(n)]


def _grid(
    *,
    phases=PH1,
    elem_phases=None,
    regulation: VoltageRegulation | None = None,
    p_gen_w: float = 0.3e6,
    q_gen_var: float = 0.0,
    load_w: float = 1.0e6,
    load_var: float = 0.4e6,
    load_per_phase=None,
    gen_phases=None,
) -> Grid:
    """Two-bus feeder: ideal slack, a 1 km line, a load and one generator at bus 1."""
    elem_phases = elem_phases or phases
    n = len(elem_phases)
    gen_phases = gen_phases or elem_phases
    v_ln = U_RATED / (math.sqrt(3.0) if len(phases) >= 3 else 1.0)
    nodes = [
        Node(id=0, name="slack", u_rated_v=U_RATED, phases=phases),
        Node(id=1, name="pv", u_rated_v=U_RATED, phases=phases),
    ]
    branches = [
        Line(
            id=10,
            name="l",
            from_node=0,
            to_node=1,
            from_phases=phases,
            to_phases=phases,
            length_m=1_000.0,
            series_resistance_ohm_per_m=_mat(len(phases), 4.0e-4),
            series_inductance_h_per_m=_mat(
                len(phases), 3.0e-4 / (2.0 * math.pi * 50.0)
            ),
            shunt_capacitance_f_per_m=_mat(len(phases), 0.0),
        )
    ]
    load_kwargs = dict(p_nom_w=load_w, q_nom_var=load_var)
    if load_per_phase is not None:
        load_kwargs = dict(
            p_nom_w=float(sum(load_per_phase)),
            q_nom_var=load_var,
            p_nom_per_phase_w=tuple(load_per_phase),
        )
    appliances = [
        Source(
            id=20,
            name="src",
            node=0,
            phases=elem_phases,
            u_ref_v=[v_ln] * n,
            u_angle_deg=[0.0, -120.0, 120.0][:n],
            resistance_ohm=_mat(n, 1.0e-6),
            inductance_h=_mat(n, 1.0e-12),
        ),
        Load(id=21, name="load", node=1, phases=elem_phases, **load_kwargs),
        Generator(
            id=GEN_ID,
            name="g",
            node=1,
            phases=gen_phases,
            p_nom_w=p_gen_w,
            q_nom_var=q_gen_var,
            voltage_regulation=regulation,
        ),
    ]
    return Grid(
        base_frequency_hz=50.0, nodes=nodes, branches=branches, appliances=appliances
    )


def _v_ln(grid: Grid) -> float:
    node = grid.nodes[1]
    return float(node.u_rated_v) / (math.sqrt(3.0) if len(node.phases) >= 3 else 1.0)


def _solve(grid: Grid, **kw):
    res = solve_power_flow(
        grid, slack="ideal", method="newton", tol=1e-8, max_iter=60, dtype=CDT, **kw
    )
    assert res.converged, f"did not converge (residual {float(res.residual):.3e})"
    return res


def _bus1_pu(grid: Grid, res) -> np.ndarray:
    rows = [res.index.row(1, ph) for ph in grid.nodes[1].phases]
    return np.array([float(res.v.reshape(-1)[r].abs()) for r in rows]) / _v_ln(grid)


def _nodal_balance_residual(grid: Grid, res, q_gen_total: float) -> float:
    """Largest |sum of currents| [A] at bus 1, recomputed in plain numpy.

    Independent of the solver: the line's series admittance from R and L, the load's
    constant-power current, and the generator's current at the SOLVED reactive power.
    """
    n = len(grid.nodes[1].phases)
    line = grid.branches[0]
    z = (
        np.asarray(line.series_resistance_ohm_per_m, float) * line.length_m
        + 1j
        * (2.0 * np.pi * 50.0)
        * np.asarray(line.series_inductance_h_per_m, float)
        * line.length_m
    )
    y_series = np.linalg.inv(z)
    v0 = np.array(
        [
            complex(res.v.reshape(-1)[res.index.row(0, ph)])
            for ph in grid.nodes[0].phases
        ]
    )
    v1 = np.array(
        [
            complex(res.v.reshape(-1)[res.index.row(1, ph)])
            for ph in grid.nodes[1].phases
        ]
    )
    load = grid.appliances[1]
    gen = grid.appliances[2]
    p_load = np.asarray(
        load.p_nom_per_phase_w
        if load.p_nom_per_phase_w is not None
        else [load.p_nom_w / n] * n,
        float,
    )
    q_load = np.full(n, load.q_nom_var / n)
    s_load = p_load + 1j * q_load
    s_gen = (gen.p_nom_w / n) + 1j * (q_gen_total / n)
    i_line = y_series @ (v1 - v0)  # current leaving bus 1 into the line
    i_load = np.conj(s_load) / np.conj(v1)
    i_gen = np.conj(s_gen) / np.conj(v1)
    return float(np.abs(i_line + i_load - i_gen).max())


# ---------------------------------------------------------------------------
# 1. The setpoint is held exactly
# ---------------------------------------------------------------------------
class TestSetpointHeldExactly:
    @pytest.mark.parametrize("v_set", [0.97, 1.0, 1.04])
    @pytest.mark.parametrize("phases", [PH1, PH3])
    def test_single_and_three_phase(self, v_set, phases):
        grid = _grid(phases=phases, regulation=VoltageRegulation(v_set_pu=v_set))
        res = _solve(grid)
        assert _bus1_pu(grid, res) == pytest.approx(v_set, abs=1e-12)
        assert res.regulation is not None
        assert bool(res.regulation.regulating[GEN_ID])
        assert res.regulation.switch_rounds == 0

    def test_reported_reactive_power_closes_the_numpy_nodal_balance(self):
        grid = _grid(regulation=VoltageRegulation(v_set_pu=1.02))
        res = _solve(grid)
        q = float(res.regulation.q_var[GEN_ID])
        # A hand-written balance at the solved voltages, with that reactive power.
        assert _nodal_balance_residual(grid, res, q) < 1e-6  # A, on a ~50 A feeder

    def test_a_pq_generator_at_the_solved_reactive_power_reproduces_the_solution(self):
        """The regulated solve is the PQ solve at the reactive power it reports."""
        grid = _grid(regulation=VoltageRegulation(v_set_pu=1.02))
        res = _solve(grid)
        q = float(res.regulation.q_var[GEN_ID])
        pq = _solve(_grid(regulation=None, q_gen_var=q))
        assert float((res.v - pq.v).abs().max()) < 1e-6  # V, on a 20 kV feeder

    def test_active_power_is_untouched(self):
        """Regulation frees Q only: the active power balance still carries p_nom."""
        grid = _grid(regulation=VoltageRegulation(v_set_pu=1.02), p_gen_w=0.45e6)
        res = _solve(grid)
        assert (
            _nodal_balance_residual(grid, res, float(res.regulation.q_var[GEN_ID]))
            < 1e-6
        )

    def test_nameplate_reactive_power_is_ignored_when_regulating(self):
        """A regulating unit's ``q_nom_var`` never enters the solution."""
        a = _solve(_grid(regulation=VoltageRegulation(v_set_pu=1.02), q_gen_var=0.0))
        b = _solve(_grid(regulation=VoltageRegulation(v_set_pu=1.02), q_gen_var=9.9e6))
        assert float((a.v - b.v).abs().max()) < 1e-9

    def test_current_injection_is_routed_to_newton_with_a_warning(self, caplog):
        grid = _grid(regulation=VoltageRegulation(v_set_pu=1.02))
        with caplog.at_level("WARNING", logger="pgml"):
            res = solve_power_flow(
                grid, method="current_injection", tol=1e-8, max_iter=60, dtype=CDT
            )
        assert res.converged
        assert _bus1_pu(grid, res) == pytest.approx(1.02, abs=1e-12)
        assert any(
            "voltage-regulating" in r.message and "newton" in r.message
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# 2. Reactive limits (PV -> PQ switching)
# ---------------------------------------------------------------------------
class TestReactiveLimits:
    def test_upper_limit_binds_and_equals_the_pq_solve_at_that_limit(self):
        """The setpoint needs ~29 Mvar; capped at 0.05 Mvar the unit pins there."""
        q_max = 0.05e6
        grid = _grid(
            regulation=VoltageRegulation(v_set_pu=1.02, q_min_var=-1e6, q_max_var=q_max)
        )
        res = _solve(grid)
        assert not bool(res.regulation.regulating[GEN_ID])
        assert float(res.regulation.q_var[GEN_ID]) == pytest.approx(q_max, rel=1e-9)
        assert res.regulation.switch_rounds == 1
        # Pinned at the limit, the bus lands BELOW the setpoint.
        assert _bus1_pu(grid, res).max() < 1.02
        pq = _solve(_grid(regulation=None, q_gen_var=q_max))
        assert float((res.v - pq.v).abs().max()) < 1e-6

    def test_lower_limit_binds_when_the_setpoint_is_below_the_natural_voltage(self):
        q_min = -0.05e6
        grid = _grid(
            regulation=VoltageRegulation(v_set_pu=0.90, q_min_var=q_min, q_max_var=1e6)
        )
        res = _solve(grid)
        assert not bool(res.regulation.regulating[GEN_ID])
        assert float(res.regulation.q_var[GEN_ID]) == pytest.approx(q_min, rel=1e-9)
        assert _bus1_pu(grid, res).min() > 0.90  # above the setpoint, as expected

    def test_a_limit_that_does_not_bind_leaves_the_unit_regulating(self):
        grid = _grid(
            regulation=VoltageRegulation(v_set_pu=1.0, q_min_var=-50e6, q_max_var=50e6)
        )
        res = _solve(grid)
        assert bool(res.regulation.regulating[GEN_ID])
        assert _bus1_pu(grid, res) == pytest.approx(1.0, abs=1e-12)
        assert res.regulation.switch_rounds == 0

    def test_enforcement_off_holds_the_setpoint_past_the_limit(self):
        """``enforce_q_limits=False`` is pandapower ``runpp``'s own default."""
        grid = _grid(
            regulation=VoltageRegulation(
                v_set_pu=1.02, q_min_var=-1e6, q_max_var=0.05e6
            )
        )
        res = _solve(grid, enforce_q_limits=False)
        assert bool(res.regulation.regulating[GEN_ID])
        assert res.regulation.enforce_q_limits is False
        assert _bus1_pu(grid, res) == pytest.approx(1.02, abs=1e-12)
        assert float(res.regulation.q_var[GEN_ID]) > 0.05e6

    def test_unbounded_by_default(self):
        """No limit given = unbounded, so nothing switches however large Q gets."""
        grid = _grid(regulation=VoltageRegulation(v_set_pu=1.04))
        res = _solve(grid)
        assert bool(res.regulation.regulating[GEN_ID])
        assert float(res.regulation.q_var[GEN_ID]) > 1e6


# ---------------------------------------------------------------------------
# 3. Regulated quantity on an unbalanced three-phase terminal
# ---------------------------------------------------------------------------
class TestRegulatedQuantity:
    UNBALANCED = (0.6e6, 1.0e6, 1.4e6)  # per-phase active load [W]

    def test_per_phase_holds_every_phase(self):
        grid = _grid(
            phases=PH3,
            load_per_phase=self.UNBALANCED,
            regulation=VoltageRegulation(
                v_set_pu=1.01, regulated=RegulatedQuantity.PER_PHASE
            ),
        )
        res = _solve(grid, symmetry="asymmetric")
        assert _bus1_pu(grid, res) == pytest.approx([1.01] * 3, abs=1e-12)

    def test_positive_sequence_holds_the_sequence_magnitude_not_each_phase(self):
        grid = _grid(
            phases=PH3,
            load_per_phase=self.UNBALANCED,
            regulation=VoltageRegulation(
                v_set_pu=1.01, regulated=RegulatedQuantity.POSITIVE_SEQUENCE
            ),
        )
        res = _solve(grid, symmetry="asymmetric")
        v = np.array([complex(res.v.reshape(-1)[res.index.row(1, ph)]) for ph in PH3])
        a = np.exp(2j * np.pi / 3.0)
        v1 = (v[0] + a * v[1] + a**2 * v[2]) / 3.0
        assert abs(v1) / _v_ln(grid) == pytest.approx(1.01, abs=1e-12)
        # The individual phases spread around it (the terminal is unbalanced).
        assert float(np.ptp(_bus1_pu(grid, res))) > 1e-4

    def test_both_modes_agree_on_a_balanced_terminal(self):
        kw = dict(phases=PH3)
        per_phase = _solve(
            _grid(
                regulation=VoltageRegulation(
                    v_set_pu=1.02, regulated=RegulatedQuantity.PER_PHASE
                ),
                **kw,
            )
        )
        seq = _solve(
            _grid(
                regulation=VoltageRegulation(
                    v_set_pu=1.02, regulated=RegulatedQuantity.POSITIVE_SEQUENCE
                ),
                **kw,
            )
        )
        assert float((per_phase.v - seq.v).abs().max()) < 1e-7  # V


# ---------------------------------------------------------------------------
# 4. Configurations the row substitution cannot express
# ---------------------------------------------------------------------------
class TestGuards:
    def test_delta_connection_raises(self):
        grid = _grid(phases=PH3, regulation=VoltageRegulation(v_set_pu=1.0))
        grid.appliances[2].connection = WindingConnection.DELTA
        with pytest.raises(ModelingError, match="WYE"):
            _solve(grid)

    def test_neutral_return_raises_and_ground_return_solves(self):
        phases = (Phase.A, Phase.B, Phase.C, Phase.N)
        grid = _grid(
            phases=phases,
            elem_phases=PH3,
            regulation=VoltageRegulation(v_set_pu=1.0),
        )
        with pytest.raises(ModelingError, match="neutral"):
            _solve(grid)
        grid.appliances[2].return_path = "ground"
        res = _solve(grid)
        rows = [res.index.row(1, ph) for ph in PH3]
        v = np.array([float(res.v.reshape(-1)[r].abs()) for r in rows]) / _v_ln(grid)
        assert v == pytest.approx(1.0, abs=1e-10)

    def test_regulating_generator_on_the_slack_node_raises(self):
        grid = _grid(regulation=VoltageRegulation(v_set_pu=1.0))
        grid.appliances[2].node = 0
        with pytest.raises(ModelingError, match="Source"):
            _solve(grid)

    def test_two_phase_positive_sequence_raises(self):
        phases = (Phase.A, Phase.B)
        grid = _grid(phases=phases, regulation=VoltageRegulation(v_set_pu=1.0))
        with pytest.raises(ModelingError, match="positive-sequence"):
            _solve(grid)

    def test_setpoint_override_on_a_plain_appliance_raises(self):
        grid = _grid(regulation=VoltageRegulation(v_set_pu=1.0))
        with pytest.raises(InputError, match="v_set_pu"):
            _solve(grid, operating_point={21: {"v_set_pu": 1.0}})

    def test_reactive_override_on_a_regulating_generator_warns(self, caplog):
        grid = _grid(regulation=VoltageRegulation(v_set_pu=1.02))
        with caplog.at_level("WARNING", logger="pgml"):
            res = _solve(grid, operating_point={GEN_ID: {"q_var": 1.0e6}})
        assert _bus1_pu(grid, res) == pytest.approx(1.02, abs=1e-12)
        assert any("VOLTAGE-REGULATING" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 5. Batched operating points
# ---------------------------------------------------------------------------
class TestBatched:
    def test_batched_setpoint(self):
        v_set = torch.tensor([0.98, 1.0, 1.02, 1.05], dtype=torch.float64)
        grid = _grid(regulation=VoltageRegulation(v_set_pu=1.0))
        res = _solve(grid, operating_point={GEN_ID: {"v_set_pu": v_set}})
        row = res.index.row(1, Phase.A)
        got = res.v[:, row].abs() / _v_ln(grid)
        assert got.detach().numpy() == pytest.approx(v_set.numpy(), abs=1e-12)
        assert res.regulation.q_var[GEN_ID].shape == (4,)
        # A rising setpoint needs monotonically more reactive power.
        q = res.regulation.q_var[GEN_ID]
        assert bool((q[1:] > q[:-1]).all())

    def test_batched_setpoint_matches_the_single_solves(self):
        v_set = torch.tensor([0.99, 1.03], dtype=torch.float64)
        grid = _grid(regulation=VoltageRegulation(v_set_pu=1.0))
        batched = _solve(grid, operating_point={GEN_ID: {"v_set_pu": v_set}})
        for i, v in enumerate(v_set.tolist()):
            one = _solve(_grid(regulation=VoltageRegulation(v_set_pu=v)))
            assert float((batched.v[i] - one.v).abs().max()) < 1e-7

    def test_batched_setpoint_with_a_per_scenario_limit(self):
        """A limit that binds in only some scenarios switches only those.

        The feeder is stiff (0.3 ohm of reactance), so holding 1.04 pu needs tens of
        Mvar while 0.9995 pu needs well under one: the same +/-5 Mvar limit binds in
        the second scenario only.
        """
        v_set = torch.tensor([0.9995, 1.04], dtype=torch.float64)
        grid = _grid(
            regulation=VoltageRegulation(v_set_pu=1.0, q_min_var=-5e6, q_max_var=5e6)
        )
        res = _solve(grid, operating_point={GEN_ID: {"v_set_pu": v_set}})
        regulating = res.regulation.regulating[GEN_ID]
        assert bool(regulating[0]) and not bool(regulating[1])
        assert float(res.regulation.q_var[GEN_ID][1]) == pytest.approx(5e6, rel=1e-9)
