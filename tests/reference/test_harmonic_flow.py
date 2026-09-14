"""Forward-correctness of the harmonic power flow (`solve_harmonic_flow`).

The harmonic INJECTION convention is OpenDSS-exact (verified in
``docs/pgml/modeling/references/opendss/harmonics.md``). The network harmonic IMPEDANCE uses the
standard ``R const, X∝h`` model (OpenDSS adds a Carson earth-return correction we
postpone), so the rigorous correctness check is an INDEPENDENT numpy reimplementation
of that same model; the OpenDSS bus-voltage match is asserted only to ballpark.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
import torch

from pgml.schemas.grid_schema import (
    Grid,
    HarmonicComponent,
    HarmonicShuntModel,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
)
from pgml.solver import solve_harmonic_flow, solve_power_flow

CDT = torch.complex128
F0 = 50.0
W0 = 2.0 * math.pi * F0
# Line / source ohms at fundamental, and the load.
R_LINE, X_LINE = 0.5, 0.5
R_SRC, X_SRC = 0.1, 0.1
P_LOAD, Q_LOAD = 2000.0, 500.0
SPEC = [(1, 1.0, 0.0), (5, 0.2, 0.0), (7, 0.14, 0.0)]  # (order, mag_pu, phase_deg)


def _grid(spec=SPEC) -> Grid:
    comps = [
        HarmonicComponent(order=o, magnitude_pu=m, phase_deg=a) for o, m, a in spec
    ]
    return Grid(
        base_frequency_hz=F0,
        nodes=[
            Node(id=1, u_rated_v=230.0, phases=(Phase.A,)),
            Node(id=2, u_rated_v=230.0, phases=(Phase.A,)),
        ],
        branches=[
            Line(
                id=1,
                from_node=1,
                to_node=2,
                from_phases=(Phase.A,),
                to_phases=(Phase.A,),
                length_m=1.0,
                series_resistance_ohm_per_m=[[R_LINE]],
                series_inductance_h_per_m=[[X_LINE / W0]],
                shunt_capacitance_f_per_m=[[0.0]],
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=(Phase.A,),
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[R_SRC]],
                inductance_h=[[X_SRC / W0]],
            ),
            Load(
                id=2,
                node=2,
                phases=(Phase.A,),
                p_nom_w=P_LOAD,
                q_nom_var=Q_LOAD,
                load_model=LoadModel.CONST_POWER,
                spectrum=StaticSpectrum(spectrum=SpectrumPoint(components=comps)),
            ),
        ],
    )


def _numpy_harmonic_v_ld(
    v_ld_fundamental: complex, orders, *, series_rl: float = 0.5
) -> dict[int, complex]:
    """Independent numpy harmonic solve (simple R-const, X∝h model) at the load bus.

    Uses the fundamental load-bus voltage to form ``I1 = conj(S0)/conj(V1)`` then,
    per harmonic, builds the 2x2 Y (line series + source Norton shunt + the load's own
    harmonic shunt), injects the nodal harmonic current ``-I_h`` at the load bus, and
    solves. ``series_rl`` is the shunt's series fraction ``s``; ``None`` leaves the load
    a pure current source.
    """
    s0 = complex(P_LOAD, Q_LOAD)
    i1 = np.conj(s0) / np.conj(v_ld_fundamental)
    spec = {o: (m, a) for o, m, a in SPEC}
    mag1, ang1 = spec[1]
    out = {}
    for h in orders:
        if h == 1:
            out[1] = v_ld_fundamental
            continue
        z_line = R_LINE + 1j * h * X_LINE
        z_src = R_SRC + 1j * h * X_SRC
        y_line = 1.0 / z_line
        y_src = 1.0 / z_src
        y_load = 0.0 + 0.0j
        if series_rl is not None:
            # The load's harmonic Norton shunt at its RATED voltage: a parallel R-L
            # branch plus a series R-L branch, both derived from conj(S0)/V_rated**2.
            y_eq = np.conj(s0) / 230.0**2
            s = series_rl
            y_load = complex((1.0 - s) * y_eq.real, (1.0 - s) * y_eq.imag / h)
            if s > 0.0:
                z_ser = 1.0 / (s * y_eq)
                y_load += 1.0 / complex(z_ser.real, h * z_ser.imag)
        Y = np.array(
            [[y_src + y_line, -y_line], [-y_line, y_line + y_load]], dtype=complex
        )
        mag_h, ang_h = spec.get(h, (0.0, 0.0))
        i_drawn = (
            (mag_h / mag1)
            * abs(i1)
            * cmath.exp(
                1j * (math.radians(ang_h) + h * (cmath.phase(i1) - math.radians(ang1)))
            )
        )
        rhs = np.array([0.0, -i_drawn], dtype=complex)  # nodal injection = -I_drawn
        v = np.linalg.solve(Y, rhs)
        out[h] = complex(v[1])
    return out


def test_matches_numpy_oracle():
    """The shipped default: the load is a current source in parallel with its shunt."""
    grid = _grid()
    orders = [1, 5, 7]
    res = solve_harmonic_flow(grid, orders, slack="norton", dtype=CDT)
    assert res.pf.converged
    ld = res.index.row(2, Phase.A)
    v_fund = complex(res.v[orders.index(1), ld])
    ref = _numpy_harmonic_v_ld(v_fund, orders)
    for k, h in enumerate(orders):
        got = complex(res.v[k, ld])
        np.testing.assert_allclose(
            [got.real, got.imag],
            [ref[h].real, ref[h].imag],
            rtol=1e-7,
            atol=1e-9,
            err_msg=f"order {h} mismatch vs numpy oracle",
        )


def test_matches_numpy_oracle_without_the_device_shunt():
    """``load_shunt="none"``: the pure current-source model, same hand-written oracle."""
    grid = _grid()
    orders = [1, 5, 7]
    res = solve_harmonic_flow(
        grid, orders, slack="norton", dtype=CDT, load_shunt="none"
    )
    ld = res.index.row(2, Phase.A)
    ref = _numpy_harmonic_v_ld(
        complex(res.v[orders.index(1), ld]), orders, series_rl=None
    )
    for k, h in enumerate(orders):
        got = complex(res.v[k, ld])
        np.testing.assert_allclose(
            [got.real, got.imag],
            [ref[h].real, ref[h].imag],
            rtol=1e-7,
            atol=1e-9,
            err_msg=f"order {h} mismatch vs numpy oracle (no device shunt)",
        )


@pytest.mark.parametrize("series_rl", [0.0, 1.0])
def test_matches_numpy_oracle_for_every_split(series_rl):
    """The two limits of the split, against the same hand-written oracle."""
    grid = _grid()
    for a in grid.appliances:
        if isinstance(a, Load):
            a.harmonic_model = HarmonicShuntModel(series_rl_fraction=series_rl)
    orders = [1, 5, 7]
    res = solve_harmonic_flow(grid, orders, slack="norton", dtype=CDT)
    ld = res.index.row(2, Phase.A)
    ref = _numpy_harmonic_v_ld(
        complex(res.v[orders.index(1), ld]), orders, series_rl=series_rl
    )
    for k, h in enumerate(orders):
        got = complex(res.v[k, ld])
        np.testing.assert_allclose(
            [got.real, got.imag],
            [ref[h].real, ref[h].imag],
            rtol=1e-7,
            atol=1e-9,
            err_msg=f"order {h} mismatch vs numpy oracle (s={series_rl})",
        )


def test_fundamental_matches_power_flow():
    grid = _grid()
    res = solve_harmonic_flow(grid, [1, 5, 7], slack="norton", dtype=CDT)
    # Order 1 is exactly the fundamental power-flow solution carried in res.pf.
    torch.testing.assert_close(res.v[0], res.pf.v, rtol=0, atol=0)
    # And it agrees with an independent solve to fixed-point tolerance.
    pf = solve_power_flow(grid, slack="norton", dtype=CDT)
    torch.testing.assert_close(res.v[0], pf.v, rtol=0, atol=1e-8)


def test_opendss_ballpark():
    """Regression guard against OpenDSS values recorded once (harmonics.md), not a live
    oracle call. Fundamental exact; harmonics within ~4% (the residual is OpenDSS Carson
    earth-return + load-Y, both deferred). The live OpenDSS harmonic comparison is
    `test_cigre_lv_live_opendss.py` / `test_carson_harmonics_feeders.py`."""
    grid = _grid()
    orders = [1, 5, 7]
    res = solve_harmonic_flow(grid, orders, slack="norton", dtype=CDT)
    ld = res.index.row(2, Phase.A)
    # OpenDSS AllBusVolts (NeglectLoadY=yes), |V_ld| per order, recorded once:
    recorded_mag = {1: 223.24690, 5: 5.51315, 7: 5.30916}
    for k, h in enumerate(orders):
        mag = abs(complex(res.v[k, ld]))
        rel = abs(mag - recorded_mag[h]) / recorded_mag[h]
        tol = 1e-4 if h == 1 else 0.04
        assert rel < tol, (
            f"order {h}: |V|={mag:.5f} vs recorded OpenDSS {recorded_mag[h]} (rel {rel:.4f})"
        )


def test_orders_shapes_and_frequencies():
    grid = _grid()
    orders = [1, 5, 7, 11]
    res = solve_harmonic_flow(grid, orders, slack="norton", dtype=CDT)
    assert res.v.shape == (len(orders), res.index.size)
    np.testing.assert_allclose(
        res.frequencies_hz.numpy(), [o * F0 for o in orders], rtol=0, atol=0
    )


def test_harmonic_only_orders_without_fundamental():
    """Requesting only harmonics (no order 1) still works (PF runs internally)."""
    grid = _grid()
    res = solve_harmonic_flow(grid, [5, 7], slack="norton", dtype=CDT)
    assert res.v.shape == (2, res.index.size)
    # Compare to the full-orders run's 5th/7th.
    full = solve_harmonic_flow(grid, [1, 5, 7], slack="norton", dtype=CDT)
    torch.testing.assert_close(res.v[0], full.v[1], rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(res.v[1], full.v[2], rtol=1e-7, atol=1e-9)


class TestSolverOptions:
    """The harmonic entry point exposes the solver options, and runs the safety gates.

    A harmonic study's expensive part is the fundamental solve plus one direct solve per
    order, all of which are factorizations of systems with the same sparsity — so the
    caller has to be able to pick the backend, the equilibration and the criticality
    policy for the whole study, and the pre-solve connectivity check must not be
    silently skipped by any of it.
    """

    @staticmethod
    def _orders():
        return [1, 5, 13]

    def test_every_backend_gives_the_same_voltages(self):
        grid = _grid()
        orders = self._orders()
        ref = solve_harmonic_flow(
            grid, orders, slack="norton", dtype=CDT, linear_solver="dense"
        )
        got = solve_harmonic_flow(
            grid, orders, slack="norton", dtype=CDT, linear_solver="sparse"
        )
        torch.testing.assert_close(got.v, ref.v, rtol=1e-10, atol=1e-12)

    def test_backend_reaches_the_per_order_solve(self):
        """A forced backend is used by the HARMONIC orders, not only the fundamental.

        Without this the option would be inert for the part of the study it is meant to
        control, which is invisible in the result and shows up only as identical timings.
        """
        import pgml.solver.harmonic_flow as hf

        seen = []
        original = hf.lu_factor_system

        def spy(y, **kw):
            seen.append(kw.get("backend"))
            return original(y, **kw)

        hf.lu_factor_system = spy
        try:
            solve_harmonic_flow(
                _grid(),
                self._orders(),
                slack="norton",
                dtype=CDT,
                linear_solver="sparse",
            )
        finally:
            hf.lu_factor_system = original
        assert seen and all(b == "sparse" for b in seen)

    def test_equilibration_option_reaches_the_orders(self):
        grid = _grid()
        orders = self._orders()
        ref = solve_harmonic_flow(
            grid, orders, slack="norton", dtype=CDT, equilibrate="off"
        )
        got = solve_harmonic_flow(
            grid, orders, slack="norton", dtype=CDT, equilibrate="symmetric"
        )
        torch.testing.assert_close(got.v, ref.v, rtol=1e-10, atol=1e-10)

    def test_block_backend_solves_the_ensemble_at_every_order(self):
        """`block_rows` reaches the per-order solves too (the CUDA ensemble path).

        A disjoint union of two feeders has a block-diagonal admittance at EVERY order, so
        the member-by-member factorization must answer the same voltages as the dense union
        at the fundamental and at each harmonic.
        """
        from pgml.grids import synthetic_feeder
        from pgml.multigrid import merge_grids

        merged = merge_grids([synthetic_feeder(6), synthetic_feeder(6)])
        orders = self._orders()
        ref = solve_harmonic_flow(merged.grid, orders, dtype=CDT, linear_solver="dense")
        blk = solve_harmonic_flow(
            merged.grid,
            orders,
            dtype=CDT,
            linear_solver="block",
            block_rows=merged.block_rows(),
        )
        # Volts on a 20 kV feeder; the dense union and the per-member factorization differ
        # only in their rounding.
        assert float((blk.v - ref.v).abs().max()) < 1e-8

    def test_criticality_option_is_accepted(self):
        res = solve_harmonic_flow(
            _grid(), self._orders(), slack="norton", dtype=CDT, criticality="always"
        )
        assert res.pf.diagnostics.criticality is not None

    def test_connectivity_is_checked_by_default(self):
        """The default run raises on a de-energized row; only "ignore" skips the check."""
        import pytest

        from pgml.errors import ConnectivityError

        grid = _grid()
        grid.branches[0].in_service = False  # cuts the load bus off the source
        with pytest.raises(ConnectivityError):
            solve_harmonic_flow(grid, [1, 5], slack="norton", dtype=CDT)
        with pytest.raises(ConnectivityError):
            solve_harmonic_flow(
                grid, [1, 5], slack="norton", dtype=CDT, on_disconnected="raise"
            )
        zeroed = solve_harmonic_flow(
            grid, [1, 5], slack="norton", dtype=CDT, on_disconnected="zero"
        )
        assert zeroed.v.shape[-1] == len(grid.nodes)


class TestScenarioBatchedSystem:
    """A per-scenario ``Y(h)`` is assembled and factored in budgeted chunks."""

    ORDERS = [1, 3, 5]

    @staticmethod
    def _batched_op(scales):
        return {
            2: {
                "p_w": torch.tensor([P_LOAD * s for s in scales], dtype=torch.float64),
                "q_var": torch.tensor(
                    [Q_LOAD * s for s in scales], dtype=torch.float64
                ),
            }
        }

    def _solve(self, op, **kw):
        return solve_harmonic_flow(
            _grid(),
            self.ORDERS,
            slack="norton",
            dtype=CDT,
            operating_point=op,
            load_shunt="opendss",
            **kw,
        )

    def test_chunked_solve_equals_the_whole_batch(self, monkeypatch):
        """Chunking is an implementation detail: the voltages must be identical.

        The device shunt on the default basis makes ``Y(h)`` per scenario, so a batch is
        assembled and factored in chunks that fit ``solver.harmonic.system_budget_mb``.
        """
        import pgml.solver.harmonic_flow as hf

        op = self._batched_op([0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 0.3])
        whole = self._solve(op)
        assert tuple(whole.v.shape) == (7, len(self.ORDERS), 2)  # [B, H, N]
        monkeypatch.setattr(hf, "_harmonic_system_budget_bytes", lambda: 1)
        chunked = self._solve(op)
        assert tuple(chunked.v.shape) == tuple(whole.v.shape)
        torch.testing.assert_close(chunked.v, whole.v, rtol=1e-12, atol=1e-12)

    def test_the_budget_decides_the_chunk_size(self):
        """One scenario is always attempted, and a generous budget keeps one chunk."""
        import pgml.solver.harmonic_flow as hf

        n_orders, n = 12, 294
        per = 2 * hf._harmonic_system_bytes(1, n_orders, n, CDT)
        assert hf._harmonic_chunk(1024, n_orders, n, CDT) >= 1
        assert hf._harmonic_chunk(4, n_orders, n, CDT) == 4  # 1 GiB default fits four
        # 18 GB for 1024 scenarios of a 294-row grid at 13 orders is the case the budget
        # exists for: the default must not select the whole batch.
        assert hf._harmonic_chunk(1024, n_orders, n, CDT) < 1024
        assert hf._harmonic_chunk(1024, n_orders, n, CDT) == (
            hf._harmonic_system_budget_bytes() // per
        )

    def test_the_nameplate_basis_needs_no_chunking(self, monkeypatch):
        """On the nameplate basis ``Y(h)`` is shared, so the budget never binds."""
        import pgml.solver.harmonic_flow as hf

        calls = []
        orig = hf._harmonic_chunk
        monkeypatch.setattr(
            hf,
            "_harmonic_chunk",
            lambda *a, **k: calls.append(a) or orig(*a, **k),
        )
        op = self._batched_op([0.5, 1.0, 1.5])
        res = self._solve(op, load_shunt_basis="nameplate")
        assert tuple(res.v.shape) == (3, len(self.ORDERS), 2)
        assert calls == []

    def test_a_scenario_batch_matches_the_single_scenario_studies(self):
        """Per-scenario systems: each scenario's voltages equal its own study's."""
        scales = [0.5, 1.0, 1.5]
        batched = self._solve(self._batched_op(scales))
        for k, s in enumerate(scales):
            single = solve_harmonic_flow(
                _grid(),
                self.ORDERS,
                slack="norton",
                dtype=CDT,
                operating_point={2: {"p_w": P_LOAD * s, "q_var": Q_LOAD * s}},
                load_shunt="opendss",
            )
            assert float((batched.v[k] - single.v).abs().max()) < 1e-12
