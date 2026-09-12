"""Power-flow convergence diagnostics + IFT-Jacobian criticality.

Asserts the non-fatal :class:`~pgml.solver.power_flow.ConvergenceDiagnostics` carried on
``PowerFlowResult`` (and surfaced on :class:`~pgml.errors.ConvergenceError`):

- a converged solve fills the cheap state diagnostics (physical mismatch, residual
  history, worst offenders, voltage band) and skips the criticality analysis under the
  ``"auto"`` policy;
- ``criticality="always"`` adds the IFT-Jacobian voltage-collapse MARGIN at a converged
  solution — the smallest singular value shrinks as the load approaches the nose;
- a load far past loadability makes the current-injection fixed point DIVERGE (it does
  not stop at the nose), which is detected and reported honestly (the Jacobian there is
  only a local linearization — a definitive limit needs continuation);
- ``simulate(strict=True)`` raises ``ConvergenceError`` carrying the rich diagnostics.
"""

from __future__ import annotations

import math

import pytest
import torch

import pgml
from pgml.errors import ConvergenceError
from pgml.solver.power_flow import ConvergenceDiagnostics, solve_power_flow
from tests.fixtures.tiny_grids import single_phase_chain

CDT = torch.complex128


def grid_rows(res) -> int:
    """Node-phase row count of a solved result (the state's last dimension)."""
    return res.index.size


def _chain(p_w: float, q_var: float):
    """The radial single-phase chain with the node-3 load set to ``(p_w, q_var)``."""
    g = single_phase_chain()
    for a in g.appliances:
        if getattr(a, "p_nom_w", None) is not None and int(a.id) == 30:
            a.p_nom_w = p_w
            a.q_nom_var = q_var
    return g


class TestConvergenceDiagnostics:
    def test_converged_populates_cheap_diagnostics(self) -> None:
        r = solve_power_flow(_chain(2000.0, 500.0), slack="ideal", dtype=CDT)
        assert r.converged
        d = r.diagnostics
        assert isinstance(d, ConvergenceDiagnostics)
        assert d.likely_cause == "converged"
        assert d.mismatch_max_a < 1e-3  # ~0 nodal current mismatch at the solution
        # Both criteria are reported in per unit, with the SI values alongside.
        assert d.mismatch_max_pu < 1e-8  # the primary criterion (per unit of s_base)
        assert d.mismatch_max_va == pytest.approx(d.mismatch_max_pu * d.s_base_va)
        assert d.update_max_pu < 1e-8 and d.update_norm_v > 0.0
        assert len(d.residual_history) == r.iterations
        assert d.residual_history[-1] < 1e-6  # the per-unit update, below tol
        assert d.worst_nodes and "node_id" in d.worst_nodes[0]
        assert d.out_of_band_nodes == []  # healthy voltages, in band
        assert d.criticality is None  # "auto" + converged -> skipped
        assert d.as_dict()["likely_cause"] == "converged"  # dict round-trip

    def test_criticality_survives_a_failing_decomposition(self, monkeypatch) -> None:
        """A decomposition that does not converge is REPORTED, never raised.

        The criticality analysis runs on the hard cases by construction — a diverged or
        barely-converged state, where LAPACK's driver can fail depending on the host's
        BLAS threading. The diagnostic then has to explain what it could not compute
        instead of taking the solve down with it.
        """
        import pgml.solver.power_flow as pf

        def _boom(*_args, **_kwargs):
            raise torch.linalg.LinAlgError(
                "linalg.svd: The algorithm failed to converge"
            )

        monkeypatch.setattr(pf.torch.linalg, "svd", _boom)
        r = solve_power_flow(
            _chain(2000.0, 500.0), slack="ideal", dtype=CDT, criticality="always"
        )
        assert r.converged  # the solve itself is untouched
        assert "skipped" in r.diagnostics.criticality
        assert "did not converge" in r.diagnostics.criticality["skipped"]

    def test_criticality_always_is_a_collapse_margin(self) -> None:
        light = solve_power_flow(
            _chain(2000.0, 500.0), slack="ideal", dtype=CDT, criticality="always"
        )
        heavy = solve_power_flow(
            _chain(50000.0, 12500.0), slack="ideal", dtype=CDT, criticality="always"
        )
        assert light.converged and heavy.converged
        cl, ch = light.diagnostics.criticality, heavy.diagnostics.criticality
        for c in (cl, ch):
            assert c["min_singular_value"] > 0.0
            assert math.isfinite(c["condition_number"])
            assert c["evaluated_at"] == "final_iterate"
            assert c["near_singular"] is False  # both far from collapse
            assert c["critical_nodes"] and "participation" in c["critical_nodes"][0]
        # The margin shrinks toward the loadability nose: a heavier (still converged)
        # load leaves the residual Jacobian closer to singular.
        assert ch["min_singular_value"] < cl["min_singular_value"]

    def test_overload_does_not_converge_and_is_reported_honestly(self) -> None:
        # Far past loadability the current-injection fixed point does not settle (it
        # oscillates or diverges — it never lands at the nose), and the diagnosis says so
        # and points at Newton / continuation rather than fabricating a collapse verdict.
        r = solve_power_flow(
            _chain(5.0e5, 1.25e5), slack="ideal", dtype=CDT, max_iter=80
        )
        assert not r.converged
        cause = r.diagnostics.likely_cause.lower()
        assert any(w in cause for w in ("diverg", "settle", "contract", "collapse"))
        # criticality (under "auto", since not converged) must NOT fabricate a
        # loadability limit at this non-solution iterate.
        c = r.diagnostics.criticality
        if c is not None:
            assert c["evaluated_at"] in ("diverged_iterate", "final_iterate")
            assert c["near_singular"] is False

    def test_criticality_handles_a_batch_of_one(self) -> None:
        """A one-element scenario batch is still ONE grid: the analysis runs on it.

        A batched operating point keeps a leading axis on the state (``[1, N]`` instead
        of ``[N]``) and therefore on the residual and its Jacobian. The analysis folds
        that singleton away and returns the same numbers as the unbatched solve; a
        diagnostic must never raise on the run it is explaining.
        """
        grid = _chain(2000.0, 500.0)
        p_batch = torch.tensor([2000.0], dtype=torch.float64)
        op = {30: {"p_w": p_batch, "q_var": 0.25 * p_batch}}
        plain = solve_power_flow(grid, slack="ideal", dtype=CDT, criticality="always")
        batched = solve_power_flow(
            grid,
            slack="ideal",
            dtype=CDT,
            operating_point=op,
            criticality="always",
        )
        assert tuple(batched.v.shape) == (1, grid_rows(plain))
        cb, cp = batched.diagnostics.criticality, plain.diagnostics.criticality
        assert "skipped" not in cb
        assert cb["min_singular_value"] == pytest.approx(cp["min_singular_value"])
        assert cb["condition_number"] == pytest.approx(cp["condition_number"])
        assert [n["node_id"] for n in cb["critical_nodes"]] == [
            n["node_id"] for n in cp["critical_nodes"]
        ]

    def test_criticality_never_skips_even_on_failure(self) -> None:
        r = solve_power_flow(
            _chain(5.0e5, 1.25e5),
            slack="ideal",
            dtype=CDT,
            max_iter=80,
            criticality="never",
        )
        assert not r.converged
        assert r.diagnostics.criticality is None

    def test_invalid_criticality_mode_raises(self) -> None:
        with pytest.raises(pgml.errors.InputError):
            solve_power_flow(_chain(2000.0, 500.0), criticality="sometimes")

    def test_simulate_strict_raises_with_rich_diagnostics(self) -> None:
        cfg = pgml.SimulationConfig(calculation="power_flow", max_iter=80)
        with pytest.raises(ConvergenceError) as excinfo:
            pgml.simulate(_chain(5.0e5, 1.25e5), cfg)
        err = excinfo.value
        assert err.diagnostics  # non-empty structured dict
        assert "likely_cause" in err.diagnostics
        assert "worst_nodes" in err.diagnostics
        assert "diverg" in str(err).lower() or "did not converge" in str(err).lower()
