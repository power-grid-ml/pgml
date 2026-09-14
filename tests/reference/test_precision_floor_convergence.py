"""What a solve does when its tolerance is below what the arithmetic resolves.

A single-precision fixed point does not settle on a fixed point of the exact equations:
it reaches a fixed point of the ROUNDED map (voltage update exactly zero) or a small
limit cycle whose amplitude is one back-substitution's rounding. The engine therefore

- judges every scenario of a batch on its OWN criteria and holds it at the iterate that
  met them, so a batched solve returns what the unbatched solve of that scenario returns
  and does not require every scenario to satisfy the criteria in the same iteration;
- detects a scenario that has stopped making progress, and either converges it at its
  precision floor (reported, with a warning naming the level) or reports the stall and
  stops, instead of spending every remaining iteration on an iterate that cannot improve.

The grids are the tiny chain fixtures; the behaviour under test is the solver's, and the
documented knobs (``solver.convergence.stall_*``) are driven through the module helpers
so the tests do not depend on a machine's rounding.
"""

from __future__ import annotations

import logging

import torch

import pgml.solver.power_flow as pf_mod
from pgml.solver.power_flow import solve_power_flow
from tests.fixtures.tiny_grids import single_phase_chain

CDT = torch.complex128
CF = torch.complex64


def _chain_op(p_values, q_value=300.0):
    """The single-phase chain and an operating point batching the node-3 load's P."""
    grid = single_phase_chain()
    p = torch.tensor(p_values, dtype=torch.float64)
    q = torch.full((len(p_values),), float(q_value), dtype=torch.float64)
    return grid, {30: {"p_w": p, "q_var": q}}


class TestPerScenarioExit:
    def test_batched_complex64_equals_the_single_scenario_solves(self) -> None:
        """Each scenario stops on its own criteria, so the batch is the loop of singles.

        At complex64 the scenarios of a batch reach the floor in different iterations,
        and an exit condition that required all of them at once would keep iterating the
        ones that were already done — changing their answer by the rounding of every
        further iteration.
        """
        p_vals = [1500.0, 3000.0, 4500.0, 6000.0]
        grid, op = _chain_op(p_vals)
        batched = solve_power_flow(grid, operating_point=op, dtype=CF)
        assert batched.converged
        for k, p in enumerate(p_vals):
            single = solve_power_flow(
                grid,
                operating_point={30: {"p_w": float(p), "q_var": 300.0}},
                dtype=CF,
            )
            assert single.converged
            assert torch.equal(batched.v[k], single.v)

    def test_large_complex64_batch_does_not_spin_to_max_iter(self) -> None:
        """A batch of many scenarios converges in the same few iterations as a few.

        The per-scenario plateau of a single-precision solve varies from scenario to
        scenario, so the more scenarios a batch carries the less likely it is that all
        of them sit inside the floor in ONE iteration.
        """
        torch.manual_seed(0)
        p = 1000.0 + 5000.0 * torch.rand(4096, dtype=torch.float64)
        grid = single_phase_chain()
        op = {30: {"p_w": p, "q_var": torch.full((4096,), 300.0, dtype=torch.float64)}}
        r = solve_power_flow(grid, operating_point=op, dtype=CF, max_iter=100)
        assert r.converged
        assert r.iterations < 20
        assert bool(r.converged_mask.all())


class TestStallAtTheFloor:
    """A stall inside the floor's band converges and is reported as floor-governed."""

    @staticmethod
    def _tight_floor(monkeypatch, *, floor: float, factor: float) -> None:
        """Make the requested tolerance unreachable with a band of ``factor`` above it.

        The float32 plateau of this grid is a property of the machine's arithmetic, so
        the test moves the FLOOR (and with it the accepted band) around the plateau
        instead of assuming where the plateau is: ``floor`` far below it and a wide band
        means "the tolerance cannot be met, but the stall is inside the band".
        """
        monkeypatch.setattr(
            pf_mod, "_rel_convergence_floor", lambda rdt, backend="dense": floor
        )
        monkeypatch.setattr(
            pf_mod, "_mismatch_floor_rel", lambda rdt, backend="dense": floor
        )
        monkeypatch.setattr(pf_mod, "_stall_tolerance_factor", lambda: factor)

    def test_converges_at_the_floor_and_reports_it(self, monkeypatch, caplog) -> None:
        self._tight_floor(monkeypatch, floor=1e-12, factor=1e9)  # band 1e-12 … 1e-3
        grid, op = _chain_op([1500.0, 3000.0, 4500.0])
        with caplog.at_level(logging.WARNING, logger="pgml"):
            r = solve_power_flow(
                grid,
                operating_point=op,
                dtype=CF,
                tol=1e-12,
                tol_update_pu=1e-12,
                max_iter=100,
            )
        assert r.converged
        assert r.iterations < 100  # it stopped when it stopped improving
        d = r.diagnostics
        assert d.floor_governed and d.n_floor_governed >= 1
        assert d.update_floor_pu == 1e-12
        assert d.n_stalled == 0
        # Neither criterion reached the requested 1e-12 pu; the band did.
        assert d.mismatch_max_pu > 1e-12
        assert "PRECISION FLOOR" in "".join(rec.message for rec in caplog.records)
        assert d.likely_cause.startswith("converged at the precision floor")

    def test_stall_above_the_band_fails_early_and_names_the_plateau(
        self, monkeypatch, caplog
    ) -> None:
        self._tight_floor(monkeypatch, floor=1e-12, factor=1.0)  # no band at all
        grid, op = _chain_op([1500.0, 3000.0])
        with caplog.at_level(logging.ERROR, logger="pgml"):
            r = solve_power_flow(
                grid,
                operating_point=op,
                dtype=CF,
                tol=1e-12,
                tol_update_pu=1e-12,
                max_iter=100,
            )
        assert not r.converged
        assert r.iterations < 30  # loudly and EARLY, not at the cap
        d = r.diagnostics
        assert d.n_stalled == 2 and d.n_floor_governed == 0
        # The plateau of whichever criterion blocked is named (here the mismatch: at
        # float32 this grid reaches an exact fixed point of the rounded map, so the
        # voltage update is 0 while the mismatch sits at its cancellation floor).
        assert d.stall_mismatch_pu > 0.0
        assert "stopped making progress" in d.likely_cause
        assert r.failed_states == (0, 1)
        assert any("did not converge" in rec.message for rec in caplog.records)

    def test_complex128_is_unaffected(self, caplog) -> None:
        """The double-precision path converges on its tolerance, with no floor report."""
        grid, op = _chain_op([1500.0, 3000.0, 4500.0])
        with caplog.at_level(logging.WARNING, logger="pgml"):
            r = solve_power_flow(grid, operating_point=op, dtype=CDT)
        assert r.converged
        d = r.diagnostics
        assert not d.floor_governed and d.n_floor_governed == 0 and d.n_stalled == 0
        assert d.update_max_pu <= 1e-8
        assert not any("PRECISION FLOOR" in rec.message for rec in caplog.records)

    def test_mixed_precision_is_unaffected(self) -> None:
        """A mixed-precision solve reaches the complex128 tolerance, not a floor."""
        grid, op = _chain_op([1500.0, 3000.0, 4500.0])
        full = solve_power_flow(grid, operating_point=op, dtype=CDT)
        mixed = solve_power_flow(grid, operating_point=op, dtype=CDT, precision="mixed")
        assert mixed.converged
        assert not mixed.diagnostics.floor_governed
        assert mixed.diagnostics.n_stalled == 0
        assert torch.max(torch.abs(mixed.v - full.v)) < 1e-9 * torch.max(
            torch.abs(full.v)
        )


class TestProgressMeasure:
    def test_a_rising_update_with_a_falling_mismatch_is_progress(
        self, monkeypatch
    ) -> None:
        """Newton's voltage update is not monotonic; its power mismatch is.

        A line search can raise the per-row update from one iteration to the next while
        the mismatch falls by a decade, so progress is improvement in EITHER criterion —
        otherwise such an iteration would be called stalled and stopped before it
        converges.
        """
        state = pf_mod._BatchIterationState(
            shape=(1,),
            device=torch.device("cpu"),
            rdt=torch.float64,
            ctest=_ctest(),
        )
        no = torch.zeros(1, dtype=torch.bool)
        upd = [8.0e-2, 2.3e-3, 6.0e-2, 1.1e-2, 2.9e-3, 1.1e-2, 4.2e-3, 2.7e-2]
        mism = [1.6e2, 1.05e2, 5.5e1, 4.0e1, 3.7e1, 2.7e1, 2.4e1, 3.0e0]
        for u, m in zip(upd, mism):
            finished, *_ = state.step(
                no,
                no,
                torch.tensor([u], dtype=torch.float64),
                torch.tensor([m], dtype=torch.float64),
            )
            assert not finished
        assert int(state.stalled.sum()) == 0

    def test_a_flat_plateau_in_both_is_a_stall(self, monkeypatch) -> None:
        state = pf_mod._BatchIterationState(
            shape=(1,),
            device=torch.device("cpu"),
            rdt=torch.float64,
            ctest=_ctest(),
        )
        no = torch.zeros(1, dtype=torch.bool)
        finished = False
        for _ in range(8):
            finished, *_ = state.step(
                no,
                no,
                torch.tensor([4.0e-6], dtype=torch.float64),
                torch.tensor([3.2e-4], dtype=torch.float64),
            )
            if finished:
                break
        assert finished and int(state.stalled.sum()) == 1


def _ctest() -> pf_mod._PuConvergence:
    """A one-row convergence test with a 1e-8 pu tolerance on both criteria."""
    return pf_mod._PuConvergence(
        v_base=torch.tensor([230.0], dtype=torch.float64),
        s_base=1.0e6,
        tol_mismatch_pu=1e-8,
        tol_update_pu=1e-8,
        floor_update=1e-15,
        floor_mismatch=1e-15,
        fixed_rows=None,
        y_eff=torch.tensor([[1.0 + 0.0j]], dtype=CDT),
        n=1,
        device=torch.device("cpu"),
        rdt=torch.float64,
        warn=False,
    )
