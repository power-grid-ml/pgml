"""Oracle test: CIGRE LV feeder (3 MV/LV transformer feeders) vs pandapower.

Network description
-------------------
``pandapower.networks.create_cigre_network_lv()`` — 44 buses, 3 MV/LV transformers
(Dyn30, 20/0.4 kV), 37 LV lines, 16 loads. Three parallel feeders (R, I, C) each
fed through a 20 kV / 400 V transformer with shift_degree=30.0 (Dyn connection).
The MV buses (0, 1, 20, 23) are coupled by closed bus-bus switches (CB type),
which we model as near-ideal Switch elements (resistance_ohm=1e-4 Ohm).

This test validates our transformer conversion and the nonlinear power flow
for a MULTI-VOLTAGE-LEVEL network with a significant turns ratio (50:1) and a
30-degree phase shift.

Transformer conversion convention
----------------------------------
The assembly's off-nominal-tap pi-model stamp uses leakage ``y_se`` referred to the
**TO (LV) side** so that the SI admittance matrix entries are correct for both HV and
LV voltage levels.  Concretely, given pandapower's HV-referred leakage:

    Z_sc_LV = Z_sc_HV / n^2     (where n = vn_hv_kv / vn_lv_kv)

The assembly formula ``Y_ff = y_se/|t|^2, Y_tt = y_se`` then gives:

    Y_ff = y_se_LV / n^2 = y_se_HV  (HV SI)   ✓
    Y_tt = y_se_LV = n^2 * y_se_HV  (LV SI)   ✓

This matches pandapower's SI Ybus (verified against ``net._ppc["internal"]["Ybus"]``).

Bus-bus switches
----------------
Pandapower merges buses 0, 1, 20, 23 (all 20 kV, connected by closed CB switches)
into one internal ppc bus.  We instead keep them as separate grid nodes connected
by near-ideal Switch elements (resistance_ohm=1e-4 Ohm).  The voltage drop across
a 1e-4 Ohm switch carrying the CIGRE LV load current is negligible (< 0.1 mV).

Tolerance targets
-----------------
- Voltage magnitude: atol = 1e-5 pu  (empirically ~1e-7 pu achieved)
- Voltage angle:     atol = 1e-4 deg (empirically ~2e-6 deg achieved)
"""

from __future__ import annotations

import math

import pandapower as pp
import pandapower.networks as pn
import pytest
import torch

from pgml.convert.pandapower import to_grid
from pgml.schemas.grid_schema import Phase
from pgml.solver import solve_power_flow


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_cigre_net() -> pp.pandapowerNet:
    """Return a converged CIGRE LV network (default const-power loads)."""
    net = pn.create_cigre_network_lv()
    pp.runpp(net, numba=False)
    assert net.converged, "pandapower did not converge on CIGRE LV"
    return net


def _angle_diff_deg(a: float, b: float) -> float:
    """Signed angle difference a - b in degrees, wrapped to (-180, 180]."""
    diff = (a - b) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


# ---------------------------------------------------------------------------
# main oracle test
# ---------------------------------------------------------------------------


class TestCIGRELVVsPandapower:
    """Compare our nonlinear const-power solve vs pandapower on CIGRE LV."""

    ATOL_VM_PU: float = 1e-5  # achievable: ~1e-7 pu
    ATOL_VA_DEG: float = 1e-4  # achievable: ~2e-6 deg

    def test_node_voltages_match_pandapower(self) -> None:
        """End-to-end: convert CIGRE LV -> solve_power_flow -> compare voltages."""
        net = _build_cigre_net()
        grid, id_map = to_grid(net)

        result = solve_power_flow(
            grid,
            slack="ideal",
            tol_update_pu=1e-12,
            max_iter=100,
            dtype=torch.complex128,
        )

        assert result.converged, (
            f"solve_power_flow did not converge (residual={float(result.residual):.3e}, "
            f"iterations={result.iterations})"
        )

        for pp_bus_idx, node_id in id_map["bus"].items():
            row = result.index.row(node_id, Phase.A)
            v_complex = result.v.reshape(-1)[row].item()

            u_rated_v = net.bus.at[pp_bus_idx, "vn_kv"] * 1_000.0
            vm_pu_ours = abs(v_complex) / u_rated_v
            va_deg_ours = math.degrees(math.atan2(v_complex.imag, v_complex.real))

            vm_pu_ref = float(net.res_bus.at[pp_bus_idx, "vm_pu"])
            va_deg_ref = float(net.res_bus.at[pp_bus_idx, "va_degree"])

            vm_err = abs(vm_pu_ours - vm_pu_ref)
            va_err = abs(_angle_diff_deg(va_deg_ours, va_deg_ref))

            assert vm_err < self.ATOL_VM_PU, (
                f"Bus {pp_bus_idx} ({net.bus.at[pp_bus_idx, 'name']}): "
                f"|V| err={vm_err:.2e} pu "
                f"(ours={vm_pu_ours:.6f}, pp={vm_pu_ref:.6f})"
            )
            assert va_err < self.ATOL_VA_DEG, (
                f"Bus {pp_bus_idx} ({net.bus.at[pp_bus_idx, 'name']}): "
                f"angle err={va_err:.2e} deg "
                f"(ours={va_deg_ours:.4f}, pp={va_deg_ref:.4f})"
            )

    def test_convergence_metadata(self) -> None:
        """Solver must converge within 100 iterations, at the tolerance it reports."""
        net = _build_cigre_net()
        grid, id_map = to_grid(net)
        tol_pu = 1e-8  # per-unit power mismatch (pandapower's own default)
        result = solve_power_flow(
            grid,
            slack="ideal",
            tol=tol_pu,
            max_iter=100,
            dtype=torch.complex128,
        )
        assert result.converged
        assert result.iterations <= 100
        # ``residual`` is the achieved PRIMARY criterion: the largest nodal
        # apparent-power mismatch in per unit of the 1 MVA base.
        assert float(result.residual) < tol_pu
        d = result.diagnostics
        assert d.mismatch_max_pu == pytest.approx(float(result.residual))
        assert d.mismatch_max_va == pytest.approx(d.mismatch_max_pu * d.s_base_va)
        assert d.update_max_pu < 1e-8

    def test_transformer_count_in_grid(self) -> None:
        """Converter must produce exactly 3 Transformer objects (one per CIGRE trafo)."""
        from pgml.schemas.grid_schema import Transformer

        net = pn.create_cigre_network_lv()
        grid, id_map = to_grid(net)
        trafos = [b for b in grid.branches if isinstance(b, Transformer)]
        assert len(trafos) == 3, f"Expected 3 trafos, got {len(trafos)}"
        assert len(id_map["trafo"]) == 3

    def test_switch_count_in_grid(self) -> None:
        """Converter must produce exactly 3 Switch objects (one per bus-bus CB)."""
        from pgml.schemas.grid_schema import Switch

        net = pn.create_cigre_network_lv()
        grid, id_map = to_grid(net)
        switches = [b for b in grid.branches if isinstance(b, Switch)]
        assert len(switches) == 3, f"Expected 3 switches, got {len(switches)}"
        assert len(id_map["switch"]) == 3
