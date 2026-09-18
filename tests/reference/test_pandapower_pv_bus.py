"""Oracle test: pandapower ``net.gen`` as an exact PV bus vs a live ``runpp``.

``GenMode.VOLTAGE_REGULATING`` (the converter default) maps each ``net.gen`` row onto a
:class:`~pgml.schemas.grid_schema.Generator` with a
:class:`~pgml.schemas.grid_schema.VoltageRegulation` block, which the solver holds
exactly. Together with ``net.shunt`` conversion this makes the MATPOWER transmission
benchmarks importable, so the comparison below is against the benchmark AS PUBLISHED —
no element taken out of service, no approximation parameter to tune.

Tolerances and what limits them
-------------------------------
``case9``, ``case39`` and ``case57`` agree to 4.4e-16, 1.8e-15 and 3.1e-15 pu
respectively, so the tolerance is the solver's own convergence, not a modelling gap.
``case14`` and ``case30`` land at 6.2e-12 and 8.9e-12 pu: that is pandapower's own
``tolerance_mva=1e-10`` mismatch translated into voltage, not a pgml deviation.

``case118`` and ``case300`` carry transformers with a NONZERO ``i0_percent`` (4 and 18
of them; the MATPOWER branch charging susceptance of a ratio branch, which
``from_ppc`` stores as a negative magnetizing current). pgml places the magnetizing
branch on the external HV terminal, OUTSIDE the winding-incidence transform, where
pandapower splits it across the pi-model — a documented topological deviation
(``docs/pgml/modeling/transformer.md``). That, and nothing else, sets their residual:
with ``i0_percent`` zeroed in BOTH tools the same two benchmarks agree to 6.7e-16 and
3.0e-14 pu (asserted below), which is what pins the deviation on the magnetizing
branch rather than on the PV-bus row or the shunt conversion.

Reactive limits
---------------
``pp.runpp``'s default is ``enforce_q_lims=False`` (limits ignored); pgml's default is
to enforce them. Both settings are compared, and in the enforcing run the generators
that pandapower switches to PQ must be the same ones pgml pins, at the same reactive
power.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgml.convert.pandapower import to_grid
from pgml.solver import solve_power_flow

pp = pytest.importorskip("pandapower", exc_type=ImportError)
pn = pytest.importorskip("pandapower.networks", exc_type=ImportError)

CDT = torch.complex128

#: Per-bus |V| tolerance [pu] and angle tolerance [deg] for the benchmarks whose
#: transformers carry no magnetizing branch. Both are solver-convergence floors.
TOL_VM_PU = 1.0e-10
TOL_VA_DEG = 1.0e-8
#: Reactive-power tolerance [Mvar] on every generator (same origin).
TOL_Q_MVAR = 1.0e-7


def _compare(
    case,
    *,
    qlim: bool,
    zero_i0: bool = False,
    tol_vm=TOL_VM_PU,
    tol_va=TOL_VA_DEG,
    tol_q=TOL_Q_MVAR,
):
    """Solve one benchmark in both tools; assert and return the deviations."""
    net = case()
    if zero_i0:
        net.trafo["i0_percent"] = 0.0
        net.trafo["pfe_kw"] = 0.0
    pp.runpp(
        net,
        enforce_q_lims=qlim,
        calculate_voltage_angles=True,
        numba=False,
        tolerance_mva=1e-10,
    )
    grid, id_map = to_grid(net)
    res = solve_power_flow(
        grid,
        slack="ideal",
        method="newton",
        tol=1e-6,
        max_iter=60,
        dtype=CDT,
        criticality="never",
        enforce_q_limits=qlim,
    )
    assert res.converged, f"pgml did not converge (residual {float(res.residual):.3e})"
    rows = np.array([res.index.row(id_map["bus"][b], "a") for b in net.bus.index])
    v = res.v.detach().numpy()[rows]
    base = net.bus.vn_kv.values * 1_000.0
    d_vm = np.abs(np.abs(v) / base - net.res_bus.vm_pu.values)
    d_va = np.abs(np.degrees(np.angle(v)) - net.res_bus.va_degree.values)
    d_q, pinned = [], 0
    for pp_idx, gid in id_map["gen"].items():
        q_pgml = float(res.regulation.q_var[gid]) / 1.0e6
        d_q.append(abs(q_pgml - float(net.res_gen.q_mvar.at[pp_idx])))
        pinned += not bool(res.regulation.regulating[gid])
    assert d_vm.max() < tol_vm, f"max |dV| = {d_vm.max():.3e} pu"
    assert d_va.max() < tol_va, f"max |d angle| = {d_va.max():.3e} deg"
    assert max(d_q) < tol_q, f"max |dQ_gen| = {max(d_q):.3e} Mvar"
    return dict(
        d_vm=d_vm.max(),
        d_va=d_va.max(),
        d_q=max(d_q),
        pinned=pinned,
        iterations=res.iterations,
        switch_rounds=res.regulation.switch_rounds,
        n_gen=len(id_map["gen"]),
    )


# ---------------------------------------------------------------------------
# 1. The quick suite: one small case and the case the approximation could not do
# ---------------------------------------------------------------------------
class TestQuick:
    def test_case14(self):
        """14 buses, 4 generators, 1 capacitive shunt, 5 transformers."""
        out = _compare(pn.case14, qlim=False)
        assert out["n_gen"] == 4
        assert out["pinned"] == 0

    def test_case39(self):
        """New England 39-bus: 9 generators, 11 tapped transformers.

        The earlier Volt-VAr approximation settled on the collapsed low-voltage
        branch here (0.49 pu off, every machine pinned at its reactive limit); the
        exact row pair reproduces pandapower to 1.8e-15 pu.
        """
        out = _compare(pn.case39, qlim=False)
        assert out["n_gen"] == 9
        assert out["d_vm"] < 1.0e-12

    def test_case39_with_reactive_limits(self):
        """With limits enforced in both tools, one machine switches to PQ."""
        out = _compare(pn.case39, qlim=True)
        assert out["pinned"] == 1
        assert out["switch_rounds"] == 1


# ---------------------------------------------------------------------------
# 2. The rest of the ladder
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestLadder:
    def test_case9(self):
        _compare(pn.case9, qlim=False)

    def test_case30(self):
        """2 shunts, no transformer."""
        _compare(pn.case30, qlim=False)

    def test_case57(self):
        """17 transformers and 3 shunts, none with a magnetizing branch."""
        out = _compare(pn.case57, qlim=False)
        assert out["d_vm"] < 1.0e-13

    @pytest.mark.parametrize("qlim", [False, True])
    def test_case118_magnetizing_branch_is_the_whole_residual(self, qlim):
        """As published the residual is the magnetizing-branch placement; with
        ``i0_percent`` zeroed in both tools the benchmark agrees exactly."""
        as_published = _compare(
            pn.case118, qlim=qlim, tol_vm=1.0e-2, tol_va=1.0e-1, tol_q=1.0e2
        )
        assert as_published["d_vm"] > 1.0e-4  # the deviation is real, not noise
        exact = _compare(pn.case118, qlim=qlim, zero_i0=True)
        assert exact["d_vm"] < 1.0e-13
        if qlim:
            assert exact["pinned"] == 6

    def test_case300_magnetizing_branch_is_the_whole_residual(self):
        """300 buses, 68 generators, 29 shunts, 128 transformers (18 with a
        magnetizing branch). ``enforce_q_lims=True`` is not compared: pandapower's
        own solve does not converge on this benchmark with limits enforced."""
        as_published = _compare(
            pn.case300, qlim=False, tol_vm=5.0e-2, tol_va=1.0, tol_q=2.0e2
        )
        assert as_published["d_vm"] > 1.0e-4
        exact = _compare(pn.case300, qlim=False, zero_i0=True)
        assert exact["d_vm"] < 1.0e-12
