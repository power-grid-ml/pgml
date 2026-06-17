"""Oracle test: IEEE 33-bus (Baran & Wu) — nonlinear const-power solver vs pandapower.

Test strategy
-------------
1. Load ``case33bw()`` from pandapower.
2. Run ``pp.runpp(net)`` with DEFAULT constant-power loads (no const_z_percent override).
   This is the real AC power flow Newton-Raphson solve.
3. Convert the net to our Grid using ``pgml.convert.pandapower.to_grid``.
   Loads default to ``LoadModel.CONST_POWER`` so no additional setup is needed.
4. Run ``solve_power_flow(grid, slack="ideal")`` (current-injection fixed-point,
   IFT-differentiable backward).
5. Compare node voltages in per-unit (|V|/V_rated and angle) against pandapower's
   ``net.res_bus``.

Both solvers solve the SAME nonlinear constant-power equation system, so agreement
should be extremely tight.  Empirically we achieve ~3e-9 pu on |V| and ~1.3e-7 deg
on angle, well within the 1e-6 pu target.

Comparison notes
----------------
- The slack bus is pinned to ``vm_pu * vn_kv * 1000 V`` (line-to-line, per our
  single-phase positive-sequence convention) via ``slack="ideal"``.
- Load voltages are in pu relative to each bus's rated voltage (``vn_kv * 1000``).
- The linear const-Z oracle (``test_ieee33_pandapower.py``) is kept as a regression
  and is NOT modified by this file.

Tolerance targets
-----------------
- Node voltage magnitude: atol = 1e-6 pu.
- Node voltage angle:      atol = 1e-5 deg.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# --- numpy 2.x compatibility shim for pandapower 2.14 ----------------------
np.Inf = np.inf  # type: ignore[attr-defined]
np.in1d = np.isin  # type: ignore[attr-defined]

import pandapower as pp  # noqa: E402
import pandapower.networks as pn  # noqa: E402

from pgml.convert.pandapower import to_grid  # noqa: E402
from pgml.schemas.grid_schema import Phase  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _build_ref_net() -> pp.pandapowerNet:
    """Return a case33bw net with default (constant-power) loads and converged PF."""
    net = pn.case33bw()
    # Do NOT set const_z_percent — use the default constant-power model.
    pp.runpp(net, numba=False)
    assert net.converged, "pandapower did not converge — check the test setup"
    return net


def _angle_diff_deg(a: float, b: float) -> float:
    """Signed angle difference a - b in degrees, wrapped to (-180, 180]."""
    diff = (a - b) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


def _cmath_angle(c: complex) -> float:
    return math.atan2(c.imag, c.real)


# ---------------------------------------------------------------------------
# main oracle test
# ---------------------------------------------------------------------------


class TestIEEE33ConstPowerVsPandapower:
    """Compare our nonlinear const-power solve vs pandapower on IEEE 33-bus."""

    # Documented achievable tolerances (both sides solve identical equations)
    ATOL_VM_PU: float = 1e-6  # magnitude tolerance in per-unit
    ATOL_VA_DEG: float = 1e-5  # angle tolerance in degrees

    def test_node_voltages_match_pandapower_const_power(self) -> None:
        """End-to-end: convert -> solve_power_flow(slack='ideal') -> compare."""
        net = _build_ref_net()
        grid, id_map = to_grid(net)

        result = solve_power_flow(
            grid,
            slack="ideal",
            tol=1e-10,
            max_iter=100,
            dtype=torch.complex128,
        )

        assert result.converged, (
            f"solve_power_flow did not converge (residual={float(result.residual):.3e})"
        )

        for pp_bus_idx, node_id in id_map["bus"].items():
            row = result.index.row(node_id, Phase.A)
            v_complex = result.v.reshape(-1)[row].item()

            u_rated_v = net.bus.at[pp_bus_idx, "vn_kv"] * 1_000.0
            vm_pu_ours = abs(v_complex) / u_rated_v
            va_deg_ours = math.degrees(_cmath_angle(v_complex))

            vm_pu_ref = float(net.res_bus.at[pp_bus_idx, "vm_pu"])
            va_deg_ref = float(net.res_bus.at[pp_bus_idx, "va_degree"])

            vm_err = abs(vm_pu_ours - vm_pu_ref)
            va_err = abs(_angle_diff_deg(va_deg_ours, va_deg_ref))

            assert vm_err < self.ATOL_VM_PU, (
                f"Bus {pp_bus_idx}: |V| mismatch "
                f"ours={vm_pu_ours:.8f} pp={vm_pu_ref:.8f} err={vm_err:.2e} pu"
            )
            assert va_err < self.ATOL_VA_DEG, (
                f"Bus {pp_bus_idx}: angle mismatch "
                f"ours={va_deg_ours:.6f} pp={va_deg_ref:.6f} err={va_err:.2e} deg"
            )

    def test_voltages_array_allclose(self) -> None:
        """Vectorised allclose check with worst-bus report."""
        net = _build_ref_net()
        grid, id_map = to_grid(net)

        result = solve_power_flow(
            grid,
            slack="ideal",
            tol=1e-10,
            max_iter=100,
            dtype=torch.complex128,
        )
        assert result.converged

        n_buses = len(id_map["bus"])
        vm_pu_ours = np.empty(n_buses)
        va_deg_ours = np.empty(n_buses)
        vm_pu_pp = np.empty(n_buses)
        va_deg_pp = np.empty(n_buses)

        for i, (pp_bus_idx, node_id) in enumerate(sorted(id_map["bus"].items())):
            row = result.index.row(node_id, Phase.A)
            v_c = result.v.reshape(-1)[row].item()
            u_rated = net.bus.at[pp_bus_idx, "vn_kv"] * 1_000.0
            vm_pu_ours[i] = abs(v_c) / u_rated
            va_deg_ours[i] = math.degrees(_cmath_angle(v_c))
            vm_pu_pp[i] = float(net.res_bus.at[pp_bus_idx, "vm_pu"])
            va_deg_pp[i] = float(net.res_bus.at[pp_bus_idx, "va_degree"])

        np.testing.assert_allclose(
            vm_pu_ours,
            vm_pu_pp,
            atol=self.ATOL_VM_PU,
            rtol=0,
            err_msg="Voltage magnitude (pu) mismatch vs pandapower const-power",
        )
        angle_diff = np.array(
            [_angle_diff_deg(a, b) for a, b in zip(va_deg_ours, va_deg_pp)]
        )
        assert np.all(np.abs(angle_diff) < self.ATOL_VA_DEG), (
            f"Voltage angle mismatch > {self.ATOL_VA_DEG} deg: "
            f"max err = {np.max(np.abs(angle_diff)):.4e} deg at bus indices "
            f"{np.where(np.abs(angle_diff) >= self.ATOL_VA_DEG)[0].tolist()}"
        )

    def test_convergence_metadata(self) -> None:
        """Solver must converge within reasonable iteration count."""
        net = _build_ref_net()
        grid, id_map = to_grid(net)
        result = solve_power_flow(
            grid,
            slack="ideal",
            tol=1e-10,
            max_iter=100,
            dtype=torch.complex128,
        )
        assert result.converged
        assert result.iterations <= 100
        assert float(result.residual) < 1e-10

    def test_worst_bus_error_documented(self) -> None:
        """Document the worst-case bus and confirm it stays below tolerance."""
        net = _build_ref_net()
        grid, id_map = to_grid(net)
        result = solve_power_flow(
            grid,
            slack="ideal",
            tol=1e-10,
            max_iter=100,
            dtype=torch.complex128,
        )
        assert result.converged

        errors = {}
        for pp_bus_idx, node_id in id_map["bus"].items():
            row = result.index.row(node_id, Phase.A)
            v_c = result.v.reshape(-1)[row].item()
            u_rated = net.bus.at[pp_bus_idx, "vn_kv"] * 1_000.0
            vm_pu_ours = abs(v_c) / u_rated
            vm_pu_ref = float(net.res_bus.at[pp_bus_idx, "vm_pu"])
            errors[pp_bus_idx] = abs(vm_pu_ours - vm_pu_ref)

        worst_bus = max(errors, key=errors.__getitem__)
        worst_err = errors[worst_bus]
        # Documented: worst bus is bus 32, error ~3e-9 pu
        assert worst_err < self.ATOL_VM_PU, (
            f"Worst bus {worst_bus}: error={worst_err:.3e} pu exceeds {self.ATOL_VM_PU}"
        )
