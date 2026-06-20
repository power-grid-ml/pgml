"""Oracle test: IEEE 33-bus (Baran & Wu) — our solver vs pandapower reference.

Test strategy
-------------
1. Load ``case33bw()`` from pandapower.
2. Set ALL loads to ``const_z_percent=100`` so both sides solve the SAME linear
   system (constant-impedance model; no Newton iteration needed).
3. Run ``pp.runpp`` to get the reference voltages.
4. Convert the net to our Grid using ``pgml.convert.pandapower.to_grid``.
5. Assemble Y-bus + current injections, then solve with **ideal-slack** mode
   (fix the slack bus voltage exactly to the ext_grid phasor).
6. Compare node voltages in per-unit (|V|/V_rated and angle).

const-Z reference-voltage convention (verified and documented)
--------------------------------------------------------------
pandapower const-Z: ``y = conj(S_rated) / V_rated^2`` where ``V_rated`` is the
bus rated voltage (``vn_kv * 1000``) which is **line-to-line** in pandapower's
positive-sequence convention.

Our schema stores single-phase nodes with ``u_rated_v = vn_kv * 1000`` (LL).
The assembly uses ``phase_voltage_magnitude(u_rated_v, n_phases=1)`` which returns
``u_rated_v`` unchanged (the sqrt(3) division only applies for n_phases >= 3).
Therefore both sides compute ``y = (P - jQ) / V_LL^2``, which is the 3-phase
total admittance of a Y-connected load expressed as a single scalar in the
positive-sequence equivalent network.  The two formulations are identical.

Slack voltage
-------------
``ext_grid.vm_pu * vn_kv * 1000`` gives the complex phasor magnitude in Volt
(line-to-line, consistent with our 1-phase-node convention).  The angle is
``ext_grid.va_degree`` in degrees. This phasor is fixed directly with
``solve_harmonic(..., fixed_rows=..., v_fixed=...)``.

Ybus comparison
---------------
pandapower's ``net._ppc["internal"]["Ybus"]`` is a scipy sparse matrix in
per-unit on its internal base (S_base = ``net._ppc["baseMVA"]`` MVA,
V_base = bus vn_kv kV). We convert our SI Y to the same per-unit base and
compare the in-service bus submatrix.

Tolerance targets
-----------------
- Node voltage magnitude: atol = 1e-6 pu (measured ~4.7e-9).
- Node voltage angle:      atol = 1e-5 deg (measured ~3.2e-8).
- Y-bus:                   rtol = 1e-4, atol = 1e-4 (looser; different
  formulation of const-Z shunt placement vs MATPOWER sparse structure).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

# --- numpy 2.x compatibility shim for pandapower 2.14 ----------------------
# pandapower 2.14 references removed numpy aliases (Inf, in1d).
# Applying the shim before the first pandapower import fixes the ImportError.
np.Inf = np.inf  # type: ignore[attr-defined]
np.in1d = np.isin  # type: ignore[attr-defined]

import pandapower as pp  # noqa: E402
import pandapower.networks as pn  # noqa: E402

from pgml.assembly import assemble_ybus, build_injections, node_phase_index  # noqa: E402
from pgml.convert.pandapower import to_grid  # noqa: E402
from pgml.solver import solve_harmonic  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _build_ref_net() -> pp.pandapowerNet:
    """Return a case33bw net with const-Z loads and a converged power flow."""
    net = pn.case33bw()
    # Switch ALL loads to 100 % constant-impedance so the system is linear.
    # This makes both pandapower and our linear solver solve the SAME system.
    net.load["const_z_percent"] = 100.0
    net.load["const_i_percent"] = 0.0
    pp.runpp(net, numba=False)
    assert net.converged, "pandapower did not converge — check the test setup"
    return net


# ---------------------------------------------------------------------------
# main oracle test
# ---------------------------------------------------------------------------


class TestIEEE33VsReference:
    """Compare our solved node voltages to pandapower on IEEE 33-bus."""

    # Tolerance targets (documented in module docstring). Measured const-Z error
    # is ~4.7e-9 pu / ~3.2e-8 deg; tolerances set with safe headroom.
    ATOL_VM_PU: float = 1e-6  # magnitude tolerance in per-unit
    ATOL_VA_DEG: float = 1e-5  # angle tolerance in degrees

    def test_node_voltages_match_pandapower(self) -> None:
        """End-to-end: convert -> assemble -> ideal-slack solve -> compare."""
        net = _build_ref_net()
        grid, id_map = to_grid(net)

        f0 = grid.base_frequency_hz  # 60 Hz for case33bw
        index = node_phase_index(grid)

        # ---- assemble Y and I -----------------------------------------------
        ybus = assemble_ybus(grid, [f0], dtype=torch.complex128)
        i_inj = build_injections(grid, [f0], index, dtype=torch.complex128)

        # ---- ideal-slack setup ----------------------------------------------
        # The slack bus is the node converted from ext_grid bus 0.
        slack_pp_bus = int(net.ext_grid.at[0, "bus"])
        slack_node_id = id_map["bus"][slack_pp_bus]
        from pgml.schemas.grid_schema import Phase

        slack_row = index.row(slack_node_id, Phase.A)
        fixed_rows = torch.tensor([slack_row], dtype=torch.int64)

        # Slack voltage: vm_pu * vn_kv * 1e3  (LL phasor, matches our 1-phase convention)
        v_slack_complex = id_map["slack_v_complex"]  # stored by to_grid
        v_fixed = torch.tensor([v_slack_complex], dtype=torch.complex128)

        # ---- solve -----------------------------------------------------------
        v_all = solve_harmonic(
            ybus.Y,  # [1, N, N]
            i_inj,  # [1, N]
            fixed_rows=fixed_rows,
            v_fixed=v_fixed,
        )  # [1, N]

        # ---- compare per bus ------------------------------------------------
        for pp_bus_idx, node_id in id_map["bus"].items():
            row = index.row(node_id, Phase.A)
            v_complex = v_all[0, row].item()  # Python complex from torch

            # Our voltage magnitude in pu
            u_rated_v = net.bus.at[pp_bus_idx, "vn_kv"] * 1_000.0
            vm_pu_ours = abs(v_complex) / u_rated_v
            va_deg_ours = math.degrees(cmath_angle(v_complex))

            # Reference from pandapower
            vm_pu_ref = float(net.res_bus.at[pp_bus_idx, "vm_pu"])
            va_deg_ref = float(net.res_bus.at[pp_bus_idx, "va_degree"])

            vm_err = abs(vm_pu_ours - vm_pu_ref)
            va_err = abs(_angle_diff_deg(va_deg_ours, va_deg_ref))

            assert vm_err < self.ATOL_VM_PU, (
                f"Bus {pp_bus_idx}: |V| mismatch "
                f"ours={vm_pu_ours:.6f} pp={vm_pu_ref:.6f} err={vm_err:.2e} pu"
            )
            assert va_err < self.ATOL_VA_DEG, (
                f"Bus {pp_bus_idx}: angle mismatch "
                f"ours={va_deg_ours:.4f} pp={va_deg_ref:.4f} err={va_err:.2e} deg"
            )


# ---------------------------------------------------------------------------
# Ybus comparison (secondary check)
# ---------------------------------------------------------------------------


class TestIEEE33YbusVsReference:
    """Secondary check: our Y(60 Hz) vs pandapower's internal Ybus (pu -> SI).

    Important caveat: pandapower's internal ``Ybus`` is the PURE NETWORK admittance
    matrix (lines + GS/BS shunts from net.shunt) WITHOUT const-Z load shunts.
    pandapower keeps const-Z loads as PQ injections and converts them to
    equivalent shunts INSIDE its Newton-Raphson loop using the CURRENT iteration
    voltage, not the rated voltage. Therefore:

    - Off-diagonal entries match our Y exactly (network topology only).
    - Diagonal entries in our Y include the const-Z load shunt
      ``y_load = conj(S)/V_rated^2`` that is absent from pandapower's Ybus.
    - The slack bus diagonal also differs because we stamp the tiny Source
      Thevenin shunt (irrelevant in ideal-slack mode) which pandapower does not.

    Tests:
    1. Off-diagonal (non-zero) entries: should agree to rtol=1e-4.
    2. Diagonal (minus our load shunts and Source shunts): should match the
       pure network contribution visible in pandapower's Ybus.
    """

    RTOL_OFF: float = 1e-4  # relative tolerance for off-diagonal entries
    ATOL_OFF: float = 1e-9  # negligible entries treated as zero

    def test_ybus_off_diagonal_matches(self) -> None:
        """Off-diagonal entries of our SI Y-bus match pandapower's (pu->SI)."""
        from pgml.schemas.grid_schema import Phase

        net = _build_ref_net()
        grid, id_map = to_grid(net)

        f0 = grid.base_frequency_hz
        ybus = assemble_ybus(grid, [f0], dtype=torch.complex128)
        Y_ours = ybus.Y[0].numpy()

        Ybus_pp_pu = net._ppc["internal"]["Ybus"].toarray()
        base_mva = float(net._ppc["baseMVA"])
        base_kv = float(net._ppc["bus"][0, 9])
        z_base = (base_kv**2) / base_mva
        y_base = 1.0 / z_base
        Ybus_pp_si = Ybus_pp_pu * y_base

        bus_lookup = net._pd2ppc_lookups["bus"]

        for pp_i, node_i in id_map["bus"].items():
            ppc_i = int(bus_lookup[pp_i])
            row_i = index_for_node(ybus.index, node_i, Phase.A)
            for pp_j, node_j in id_map["bus"].items():
                if pp_j == pp_i:
                    continue
                ppc_j = int(bus_lookup[pp_j])
                row_j = index_for_node(ybus.index, node_j, Phase.A)
                y_pp = Ybus_pp_si[ppc_i, ppc_j]
                if abs(y_pp) < self.ATOL_OFF:
                    continue  # negligible/zero entry — both sides should be ~0
                y_ours = Y_ours[row_i, row_j]
                err = abs(y_ours - y_pp)
                assert err < self.RTOL_OFF * abs(y_pp), (
                    f"Off-diagonal Y mismatch at ({pp_i},{pp_j}): "
                    f"ours={y_ours:.4e} pp={y_pp:.4e} err={err:.3e}"
                )

    def test_ybus_diagonal_network_only_matches(self) -> None:
        """Diagonal entries MINUS our const-Z and Source shunts match pandapower's Ybus.

        pandapower's Ybus diagonal = pure line/branch contributions.
        Our diagonal = same lines + Source Thevenin shunt (slack) + const-Z load shunts.
        After subtracting our load and source shunts, the remainder should match.
        """
        from pgml.schemas.grid_schema import (
            Load as GridLoad,
            Phase,
            Source as GridSource,
        )

        net = _build_ref_net()
        grid, id_map = to_grid(net)

        f0 = grid.base_frequency_hz
        ybus = assemble_ybus(grid, [f0], dtype=torch.complex128)
        Y_ours = ybus.Y[0].numpy()

        Ybus_pp_pu = net._ppc["internal"]["Ybus"].toarray()
        base_mva = float(net._ppc["baseMVA"])
        base_kv = float(net._ppc["bus"][0, 9])
        z_base = (base_kv**2) / base_mva
        y_base = 1.0 / z_base
        Ybus_pp_si = Ybus_pp_pu * y_base

        bus_lookup = net._pd2ppc_lookups["bus"]

        # Build a map: node_id -> sum of load shunt admittances at that node
        # y_load = conj(S) / V_rated^2  (our formula, same as rated-V const-Z)
        node_id_of = {v: k for k, v in id_map["bus"].items()}
        load_shunt: dict[int, complex] = {}
        for app in grid.appliances:
            if isinstance(app, GridLoad) and app.in_service:
                node_id = app.node
                pp_bus_idx = node_id_of[node_id]
                u_rated_v = float(net.bus.at[pp_bus_idx, "vn_kv"]) * 1_000.0
                y = complex(app.p_nom_w, -app.q_nom_var) / (u_rated_v**2)
                load_shunt[node_id] = load_shunt.get(node_id, 0.0) + y

        # Source (Thevenin) Norton shunt: Y_s = Z_s^{-1} at frequency f0
        # Z_s = R_tiny + j*2*pi*f0*L_tiny
        two_pi_f0 = 2.0 * math.pi * f0
        source_shunt: dict[int, complex] = {}
        for app in grid.appliances:
            if isinstance(app, GridSource) and app.in_service:
                r = app.resistance_ohm[0][0]
                lh = app.inductance_h[0][0]
                z = complex(r, two_pi_f0 * lh)
                y_s = 1.0 / z
                source_shunt[app.node] = source_shunt.get(app.node, 0.0) + y_s

        for pp_bus_idx, node_id in id_map["bus"].items():
            ppc_idx = int(bus_lookup[pp_bus_idx])
            row = index_for_node(ybus.index, node_id, Phase.A)

            y_diag_ours = Y_ours[row, row]
            # Subtract our shunts to get pure network diagonal
            y_net_ours = (
                y_diag_ours
                - load_shunt.get(node_id, 0.0)
                - source_shunt.get(node_id, 0.0)
            )
            y_diag_pp = Ybus_pp_si[ppc_idx, ppc_idx]

            err = abs(y_net_ours - y_diag_pp)
            ref_scale = max(abs(y_diag_pp), 1e-12)
            assert err < 1e-4 * ref_scale + 1e-9, (
                f"Network Y diagonal mismatch at bus {pp_bus_idx}: "
                f"ours(net)={y_net_ours:.4f} pp={y_diag_pp:.4f} err={err:.3e}"
            )


# ---------------------------------------------------------------------------
# converter unit tests
# ---------------------------------------------------------------------------


class TestToGridConverter:
    """Unit tests for the to_grid converter (no solve needed)."""

    def test_bus_count(self) -> None:
        net = pn.case33bw()
        grid, id_map = to_grid(net)
        n_buses = net.bus["in_service"].sum()
        assert len(grid.nodes) == n_buses

    def test_line_count_in_service(self) -> None:
        net = pn.case33bw()
        grid, id_map = to_grid(net)
        n_lines_in_service = net.line["in_service"].sum()
        assert len(grid.branches) == n_lines_in_service

    def test_load_count(self) -> None:
        net = pn.case33bw()
        grid, id_map = to_grid(net)
        n_loads = net.load["in_service"].sum()
        # +1 for the ext_grid Source
        n_appliances_expected = n_loads + net.ext_grid["in_service"].sum()
        assert len(grid.appliances) == n_appliances_expected

    def test_base_frequency(self) -> None:
        net = pn.case33bw()
        grid, _ = to_grid(net)
        assert grid.base_frequency_hz == pytest.approx(float(net.f_hz))

    def test_node_rated_voltage(self) -> None:
        net = pn.case33bw()
        grid, id_map = to_grid(net)
        for pp_bus, node_id in id_map["bus"].items():
            node = next(n for n in grid.nodes if n.id == node_id)
            expected_v = float(net.bus.at[pp_bus, "vn_kv"]) * 1_000.0
            assert node.u_rated_v == pytest.approx(expected_v)

    def test_line_unit_conversion(self) -> None:
        """First in-service line: verify SI per-length params."""
        net = pn.case33bw()
        grid, id_map = to_grid(net)
        f0 = float(net.f_hz)
        # First in-service line
        pp_line_idx = net.line[net.line["in_service"]].index[0]
        line_id = id_map["line"][pp_line_idx]
        from pgml.schemas.grid_schema import Line as GridLine

        line = next(
            b for b in grid.branches if isinstance(b, GridLine) and b.id == line_id
        )
        pp_row = net.line.loc[pp_line_idx]
        assert line.length_m == pytest.approx(float(pp_row["length_km"]) * 1_000.0)
        assert line.series_resistance_ohm_per_m[0][0] == pytest.approx(
            float(pp_row["r_ohm_per_km"]) / 1_000.0
        )
        assert line.series_inductance_h_per_m[0][0] == pytest.approx(
            float(pp_row["x_ohm_per_km"]) / 1_000.0 / (2 * math.pi * f0)
        )

    def test_id_map_completeness(self) -> None:
        net = pn.case33bw()
        _, id_map = to_grid(net)
        # All in-service buses should appear
        for pp_bus in net.bus[net.bus["in_service"]].index:
            assert pp_bus in id_map["bus"]
        # All in-service lines whose buses are present
        for pp_line in net.line[net.line["in_service"]].index:
            row = net.line.loc[pp_line]
            fb, tb = int(row["from_bus"]), int(row["to_bus"])
            if fb in id_map["bus"] and tb in id_map["bus"]:
                assert pp_line in id_map["line"]

    def test_slack_v_complex_stored(self) -> None:
        net = pn.case33bw()
        _, id_map = to_grid(net)
        assert "slack_v_complex" in id_map
        v = id_map["slack_v_complex"]
        # Should be close to vm_pu * vn_kv * 1000 at va_degree angle
        vm_pu = float(net.ext_grid.at[0, "vm_pu"])
        vn_v = float(net.bus.at[0, "vn_kv"]) * 1_000.0
        expected_mag = vm_pu * vn_v
        assert abs(v) == pytest.approx(expected_mag, rel=1e-9)


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------


def cmath_angle(c: complex) -> float:
    """Phase angle of complex number c in radians (math.atan2 convention)."""
    return math.atan2(c.imag, c.real)


def _angle_diff_deg(a: float, b: float) -> float:
    """Signed angle difference a - b in degrees, wrapped to (-180, 180]."""
    diff = (a - b) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


def index_for_node(index, node_id: int, phase):
    """Convenience: row of node_id / phase from a NodePhaseIndex."""
    return index.row(node_id, phase)
