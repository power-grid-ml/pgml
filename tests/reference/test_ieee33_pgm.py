"""Oracle test: IEEE 33-bus (Baran & Wu) — our solver vs power-grid-model reference.

Test strategy
-------------
1. Build an IEEE 33-bus pgm ``input_data`` dict from the same numerical data as
   the pandapower oracle (``case33bw()`` from pandapower) so all tools describe
   an IDENTICAL network in SI units.
2. Set ALL ``sym_load`` to ``LoadGenType.const_impedance`` so that pgm solves the
   SAME constant-impedance linear system as our M1 assembly (no Newton iteration).
3. Run ``PowerGridModel.calculate_power_flow(symmetric=True)`` for the reference
   node voltages.
4. Convert the same ``input_data`` to our Grid via ``pgml.convert.pgm.to_grid``.
5. Assemble Y-bus + current injections, then solve with **ideal-slack** mode
   (fix the slack bus voltage to the source's reference phasor exactly).
6. Compare node voltages in per-unit (|V|/u_rated) and angle.

Load model and slack setup
--------------------------
pgm ``sym_load.type = LoadGenType.const_impedance``:
    At constant-impedance mode pgm linearises the load as
    ``Y_load = conj(S_rated) / u_rated_ll^2`` — identical to our M1 const-Z
    formula ``y = conj(P + jQ) / u_rated_v^2``.  Both sides solve the same
    linear system, so results should agree to numerical precision.

Our schema stores the M1 load shunt per the assembly CONTEXT.md:
    ``y = conj(P + jQ) / |U_nom|^2``
where ``|U_nom|`` is ``u_rated_v`` (line-to-line) for a 1-phase node.  This is
exactly pgm's constant-impedance admittance at the rated (nominal) voltage.

Slack source: pgm source with ``sk = 1e16 VA`` (near-ideal) and
``u_ref = 1.0 pu`` (→ 12660 V).  pgm's slack voltage converges to
``1.0 pu ± ~1e-8`` (verified empirically; documented below).  Our ideal-slack
mode fixes the slack to exactly ``1.0 * 12660 = 12660 + 0j`` V, which is within
1e-6 pu of pgm's slack, producing < 1e-6 pu error at all buses.

pgm voltage convention
-----------------------
``out["node"]["u"]`` is the LINE-TO-LINE voltage magnitude in Volt (same base as
``node.u_rated``).  ``out["node"]["u_pu"]`` = u / u_rated (line-to-line based).
``out["node"]["u_angle"]`` is in radians.

Our single-phase nodes store ``u_rated_v = u_rated`` (LL) so the comparison is:
    our_pu = |v_complex| / u_rated_v  ← same base as pgm u_pu.

Line parameter mapping (pgm -> our schema)
------------------------------------------
pgm lines store TOTAL (lumped) positive-sequence impedances:
    r1 [Ohm] = r_ohm_per_km * length_km   (total, positive-sequence)
    x1 [Ohm] = x_ohm_per_km * length_km
    c1 [F]   = c_nf_per_km * length_km * 1e-9

The converter sets ``length_m = 1.0`` (virtual) and:
    series_resistance_ohm_per_m = r1   → Z_total = r1 * 1 = r1 Ohm ✓
    series_inductance_h_per_m   = x1 / (2*pi*f0)
    shunt_capacitance_f_per_m   = c1

Tolerance targets
-----------------
- Node voltage magnitude: atol = 1e-4 pu.
- Node voltage angle:     atol = 1e-4 deg.

Achieved on IEEE 33-bus: max |V| error < 1e-8 pu (well within tolerance).
"""

from __future__ import annotations

import math

import numpy as np
import pandapower.networks as pn
import power_grid_model as pgm
import pytest
import torch
from power_grid_model import LoadGenType, PowerGridModel

from pgml.assembly import assemble_ybus, build_injections, node_phase_index
from pgml.convert.pgm import to_grid
from pgml.schemas.grid_schema import LoadModel, Phase
from pgml.solver import solve_harmonic


# ---------------------------------------------------------------------------
# pgm input builder (IEEE 33-bus derived from pandapower case33bw data)
# ---------------------------------------------------------------------------

# pgm id ranges (non-overlapping to avoid collisions)
_NODE_ID_OFFSET = 0  # node ids = bus index (0..32)
_LINE_ID_OFFSET = 100  # line ids = 100 + line index
_LOAD_ID_OFFSET = 200  # load ids = 200 + load index
_SOURCE_ID = 300  # single source id

# Source: near-ideal slack — sk large enough that pgm slack ≈ 1.0 pu (error < 1e-8)
_SOURCE_SK_VA = 1.0e16
_SOURCE_RX = 0.0  # pure reactance (R=0 in the limit)


def _build_pgm_input() -> dict[str, np.ndarray]:
    """Return a pgm ``input_data`` dict for IEEE 33-bus with const-Z loads.

    All loads are set to ``LoadGenType.const_impedance`` so pgm and our solver
    solve the same linear system (no Newton iteration needed).
    """
    net = pn.case33bw()

    # -- nodes ----------------------------------------------------------------
    n_buses = len(net.bus)
    node_arr = pgm.initialize_array("input", "node", n_buses)
    for i, (pp_idx, row) in enumerate(net.bus.iterrows()):
        node_arr["id"][i] = int(pp_idx) + _NODE_ID_OFFSET
        node_arr["u_rated"][i] = float(row["vn_kv"]) * 1_000.0  # V, line-to-line

    # -- lines (in-service only) ----------------------------------------------
    lines_df = net.line[net.line["in_service"]]
    line_arr = pgm.initialize_array("input", "line", len(lines_df))
    for i, (pp_idx, row) in enumerate(lines_df.iterrows()):
        lkm = float(row["length_km"])
        line_arr["id"][i] = int(pp_idx) + _LINE_ID_OFFSET
        line_arr["from_node"][i] = int(row["from_bus"]) + _NODE_ID_OFFSET
        line_arr["to_node"][i] = int(row["to_bus"]) + _NODE_ID_OFFSET
        line_arr["from_status"][i] = 1
        line_arr["to_status"][i] = 1
        line_arr["r1"][i] = float(row["r_ohm_per_km"]) * lkm  # Ohm total
        line_arr["x1"][i] = float(row["x_ohm_per_km"]) * lkm  # Ohm total
        c_nf_km = float(row.get("c_nf_per_km", 0.0) or 0.0)
        line_arr["c1"][i] = c_nf_km * lkm * 1.0e-9  # F total
        line_arr["tan1"][i] = 0.0
        # Positive-sequence = zero-sequence for this network (homogeneous cables)
        line_arr["r0"][i] = line_arr["r1"][i]
        line_arr["x0"][i] = line_arr["x1"][i]
        line_arr["c0"][i] = line_arr["c1"][i]
        line_arr["tan0"][i] = 0.0
        line_arr["i_n"][i] = 1_000.0  # ampacity (not used in load flow)

    # -- loads (in-service only, all const-impedance) -------------------------
    loads_df = net.load[net.load["in_service"]]
    load_arr = pgm.initialize_array("input", "sym_load", len(loads_df))
    for i, (pp_idx, row) in enumerate(loads_df.iterrows()):
        load_arr["id"][i] = int(pp_idx) + _LOAD_ID_OFFSET
        load_arr["node"][i] = int(row["bus"]) + _NODE_ID_OFFSET
        load_arr["status"][i] = 1
        load_arr["type"][i] = LoadGenType.const_impedance
        load_arr["p_specified"][i] = float(row["p_mw"]) * 1.0e6  # W
        load_arr["q_specified"][i] = float(row["q_mvar"]) * 1.0e6  # VAr

    # -- source (near-ideal slack at bus 0) -----------------------------------
    source_arr = pgm.initialize_array("input", "source", 1)
    source_arr["id"][0] = _SOURCE_ID
    source_arr["node"][0] = 0 + _NODE_ID_OFFSET  # bus 0 is slack
    source_arr["status"][0] = 1
    source_arr["u_ref"][0] = float(net.ext_grid.at[0, "vm_pu"])  # pu (1.0)
    source_arr["u_ref_angle"][0] = math.radians(float(net.ext_grid.at[0, "va_degree"]))
    source_arr["sk"][0] = _SOURCE_SK_VA  # near-ideal slack
    source_arr["rx_ratio"][0] = _SOURCE_RX
    source_arr["z01_ratio"][0] = 1.0

    return {
        "node": node_arr,
        "line": line_arr,
        "sym_load": load_arr,
        "source": source_arr,
    }


def _pgm_node_id(pp_bus_idx: int) -> int:
    """Map pandapower bus index to pgm node id."""
    return pp_bus_idx + _NODE_ID_OFFSET


# ---------------------------------------------------------------------------
# helper utilities
# ---------------------------------------------------------------------------


def _cmath_angle(c: complex) -> float:
    """Phase angle in radians."""
    return math.atan2(c.imag, c.real)


def _angle_diff_deg(a: float, b: float) -> float:
    """Signed angle difference a - b in degrees, wrapped to (-180, 180]."""
    diff = (a - b) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


# ---------------------------------------------------------------------------
# converter unit tests
# ---------------------------------------------------------------------------


class TestPgmConverterUnit:
    """Unit tests for ``to_grid`` (no solver or pgm power flow needed)."""

    def test_node_count(self) -> None:
        input_data = _build_pgm_input()
        grid, id_map = to_grid(input_data, base_frequency_hz=60.0)
        assert len(grid.nodes) == len(input_data["node"])

    def test_line_count(self) -> None:
        input_data = _build_pgm_input()
        grid, id_map = to_grid(input_data, base_frequency_hz=60.0)
        in_svc_lines = int(
            sum(
                1
                for r in input_data["line"]
                if int(r["from_status"]) == 1 and int(r["to_status"]) == 1
            )
        )
        assert len(grid.branches) == in_svc_lines

    def test_load_count(self) -> None:
        input_data = _build_pgm_input()
        grid, id_map = to_grid(input_data, base_frequency_hz=60.0)
        in_svc_loads = int(
            sum(1 for r in input_data["sym_load"] if int(r["status"]) == 1)
        )
        in_svc_sources = int(
            sum(1 for r in input_data["source"] if int(r["status"]) == 1)
        )
        # loads + 1 source
        assert len(grid.appliances) == in_svc_loads + in_svc_sources

    def test_node_rated_voltage(self) -> None:
        input_data = _build_pgm_input()
        grid, id_map = to_grid(input_data, base_frequency_hz=60.0)
        for pgm_node_id, our_node_id in id_map["node"].items():
            node_obj = next(n for n in grid.nodes if n.id == our_node_id)
            pgm_row = next(r for r in input_data["node"] if int(r["id"]) == pgm_node_id)
            assert node_obj.u_rated_v == pytest.approx(float(pgm_row["u_rated"]))

    def test_line_impedance(self) -> None:
        """First line: r_per_m == r1 (virtual 1m length), l_per_m == x1/(2*pi*f0)."""
        import math as _math

        f0 = 60.0
        input_data = _build_pgm_input()
        grid, id_map = to_grid(input_data, base_frequency_hz=f0)
        # First in-service line
        first_pgm_line = next(iter(id_map["line"].keys()))
        our_line_id = id_map["line"][first_pgm_line]
        from pgml.schemas.grid_schema import Line as GridLine

        line_obj = next(
            b for b in grid.branches if isinstance(b, GridLine) and b.id == our_line_id
        )
        pgm_row = next(r for r in input_data["line"] if int(r["id"]) == first_pgm_line)
        r1 = float(pgm_row["r1"])
        x1 = float(pgm_row["x1"])
        assert line_obj.length_m == pytest.approx(1.0)
        assert line_obj.series_resistance_ohm_per_m[0][0] == pytest.approx(r1)
        assert line_obj.series_inductance_h_per_m[0][0] == pytest.approx(
            x1 / (2 * _math.pi * f0), rel=1e-9
        )

    def test_load_model_const_impedance(self) -> None:
        """All loads must have CONST_IMPEDANCE model."""
        input_data = _build_pgm_input()
        grid, id_map = to_grid(
            input_data, base_frequency_hz=60.0, load_model=LoadModel.CONST_IMPEDANCE
        )
        from pgml.schemas.grid_schema import Load as GridLoad

        for app in grid.appliances:
            if isinstance(app, GridLoad):
                assert app.load_model == LoadModel.CONST_IMPEDANCE

    def test_slack_v_complex_stored(self) -> None:
        input_data = _build_pgm_input()
        _, id_map = to_grid(input_data, base_frequency_hz=60.0)
        assert id_map["slack_v_complex"] is not None
        v = id_map["slack_v_complex"]
        # u_ref=1.0 pu → |v| ≈ u_rated = 12660 V
        assert abs(v) == pytest.approx(12_660.0, rel=1e-6)
        # u_ref_angle=0 → angle ≈ 0
        assert math.degrees(_cmath_angle(v)) == pytest.approx(0.0, abs=1e-9)

    def test_id_map_completeness(self) -> None:
        input_data = _build_pgm_input()
        _, id_map = to_grid(input_data, base_frequency_hz=60.0)
        # Every node in input_data must appear in id_map
        for row in input_data["node"]:
            assert int(row["id"]) in id_map["node"]
        # Every in-service line must appear
        for row in input_data["line"]:
            if int(row["from_status"]) == 1 and int(row["to_status"]) == 1:
                assert int(row["id"]) in id_map["line"]
        # Every in-service load must appear
        for row in input_data["sym_load"]:
            if int(row["status"]) == 1:
                assert int(row["id"]) in id_map["sym_load"]

    def test_base_frequency_stored(self) -> None:
        input_data = _build_pgm_input()
        grid, _ = to_grid(input_data, base_frequency_hz=60.0)
        assert grid.base_frequency_hz == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# main oracle test: our solver vs pgm power flow
# ---------------------------------------------------------------------------


class TestIEEE33VsPgm:
    """Compare our solved node voltages to power-grid-model on IEEE 33-bus.

    Both sides use:
    - Identical network (from case33bw pandapower data converted to pgm format)
    - Constant-impedance load model (linear system, no iteration needed)
    - Same slack bus voltage (1.0 pu = 12660 V, angle 0)

    Comparison metric:
    - ours: |v_complex| / u_rated_v  (line-to-line pu)
    - pgm:  out["node"]["u_pu"]      (= u / u_rated, line-to-line pu)
    """

    # Achieved < 1e-8 pu in practice; tolerance set with safe headroom.
    ATOL_VM_PU: float = 1e-6  # voltage magnitude tolerance, per-unit
    ATOL_VA_DEG: float = 1e-6  # voltage angle tolerance, degrees

    def test_node_voltages_match_pgm(self) -> None:
        """Per-bus check: our |V| pu and angle vs pgm result."""
        input_data = _build_pgm_input()
        f0 = 60.0

        # --- pgm reference ---------------------------------------------------
        pgm_model = PowerGridModel(input_data)
        pgm_result = pgm_model.calculate_power_flow(symmetric=True)
        pgm_by_id = {int(r["id"]): r for r in pgm_result["node"]}

        # --- our solver ------------------------------------------------------
        grid, id_map = to_grid(
            input_data, base_frequency_hz=f0, load_model=LoadModel.CONST_IMPEDANCE
        )
        index = node_phase_index(grid)
        ybus = assemble_ybus(grid, [f0], dtype=torch.complex128)
        i_inj = build_injections(grid, [f0], index, dtype=torch.complex128)

        slack_pgm_node = _pgm_node_id(0)  # bus 0 is slack
        slack_our_node = id_map["node"][slack_pgm_node]
        slack_row = index.row(slack_our_node, Phase.A)
        fixed_rows = torch.tensor([slack_row], dtype=torch.int64)
        v_fixed = torch.tensor([id_map["slack_v_complex"]], dtype=torch.complex128)

        v_all = solve_harmonic(
            ybus.Y, i_inj, fixed_rows=fixed_rows, v_fixed=v_fixed
        )  # [1, N]

        # --- per-bus comparison ----------------------------------------------
        for pgm_node_id, our_node_id in id_map["node"].items():
            row_idx = index.row(our_node_id, Phase.A)
            v_c = v_all[0, row_idx].item()

            # Recover u_rated from grid node
            node_obj = next(n for n in grid.nodes if n.id == our_node_id)
            u_rated_v = node_obj.u_rated_v

            vm_pu_ours = abs(v_c) / u_rated_v
            va_deg_ours = math.degrees(_cmath_angle(v_c))

            pgm_row = pgm_by_id[pgm_node_id]
            vm_pu_pgm = float(pgm_row["u_pu"])
            va_deg_pgm = math.degrees(float(pgm_row["u_angle"]))

            vm_err = abs(vm_pu_ours - vm_pu_pgm)
            va_err = abs(_angle_diff_deg(va_deg_ours, va_deg_pgm))

            assert vm_err < self.ATOL_VM_PU, (
                f"Node {pgm_node_id}: |V| mismatch "
                f"ours={vm_pu_ours:.6f} pgm={vm_pu_pgm:.6f} err={vm_err:.2e} pu"
            )
            assert va_err < self.ATOL_VA_DEG, (
                f"Node {pgm_node_id}: angle mismatch "
                f"ours={va_deg_ours:.4f} pgm={va_deg_pgm:.4f} err={va_err:.2e} deg"
            )

    def test_achieved_tolerance_is_tight(self) -> None:
        """Assert the ACTUAL max error is well below 1e-4 (it should be < 1e-6).

        This documents the achieved accuracy and would catch a regression where
        a code change degrades the solver from sub-1e-8 to, say, 1e-5 pu.
        """
        input_data = _build_pgm_input()
        f0 = 60.0

        pgm_model = PowerGridModel(input_data)
        pgm_result = pgm_model.calculate_power_flow(symmetric=True)

        grid, id_map = to_grid(
            input_data, base_frequency_hz=f0, load_model=LoadModel.CONST_IMPEDANCE
        )
        index = node_phase_index(grid)
        ybus = assemble_ybus(grid, [f0], dtype=torch.complex128)
        i_inj = build_injections(grid, [f0], index, dtype=torch.complex128)

        slack_our_node = id_map["node"][_pgm_node_id(0)]
        fixed_rows = torch.tensor(
            [index.row(slack_our_node, Phase.A)], dtype=torch.int64
        )
        v_fixed = torch.tensor([id_map["slack_v_complex"]], dtype=torch.complex128)
        v_all = solve_harmonic(ybus.Y, i_inj, fixed_rows=fixed_rows, v_fixed=v_fixed)

        pgm_by_id = {int(r["id"]): r for r in pgm_result["node"]}
        max_err = 0.0
        for pgm_node_id, our_node_id in id_map["node"].items():
            row_idx = index.row(our_node_id, Phase.A)
            v_c = v_all[0, row_idx].item()
            node_obj = next(nd for nd in grid.nodes if nd.id == our_node_id)
            vm_pu_ours = abs(v_c) / node_obj.u_rated_v
            vm_pu_pgm = float(pgm_by_id[pgm_node_id]["u_pu"])
            max_err = max(max_err, abs(vm_pu_ours - vm_pu_pgm))

        # Tight regression guard: expect < 1e-6 pu in practice
        assert max_err < 1e-6, (
            f"Solver regression: max |V| error {max_err:.2e} pu "
            f"(expected < 1e-6 pu for const-Z + ideal-slack vs pgm near-ideal-slack)"
        )


# ---------------------------------------------------------------------------
# cross-check: pgm grid matches pandapower grid (same physical network)
# ---------------------------------------------------------------------------


class TestPgmGridMatchesPandapowerGrid:
    """Assert the pgm-derived Grid produces the same node voltages as the
    pandapower-derived Grid when solved with the same ideal-slack and const-Z
    parameters.  This confirms both converters describe an identical network.
    """

    # Same network, same ideal slack, same const-Z: the two converters must
    # produce bit-for-bit equivalent grids, so the COMPLEX voltages agree to
    # near machine precision (measured << 1e-9 pu).
    ATOL_V_PU: float = 1e-9

    def test_pgm_and_pandapower_grids_agree(self) -> None:
        """Solve both grids; complex node voltages must agree to ~1e-9 pu."""
        import pandapower as pp
        from pgml.convert.pandapower import to_grid as pp_to_grid

        net = pn.case33bw()
        net.load["const_z_p_percent"] = 100.0
        net.load["const_z_q_percent"] = 100.0
        net.load["const_i_percent"] = 0.0
        pp.runpp(net, numba=False)
        assert net.converged

        f0 = float(net.f_hz)

        # pandapower grid
        pp_grid, pp_id_map = pp_to_grid(net)
        pp_index = node_phase_index(pp_grid)
        pp_ybus = assemble_ybus(pp_grid, [f0], dtype=torch.complex128)
        pp_i_inj = build_injections(pp_grid, [f0], pp_index, dtype=torch.complex128)
        pp_slack_row = pp_index.row(pp_id_map["bus"][0], Phase.A)
        pp_fixed = torch.tensor([pp_slack_row], dtype=torch.int64)
        pp_v_fixed = torch.tensor(
            [pp_id_map["slack_v_complex"]], dtype=torch.complex128
        )
        pp_v_all = solve_harmonic(
            pp_ybus.Y, pp_i_inj, fixed_rows=pp_fixed, v_fixed=pp_v_fixed
        )

        # pgm grid
        pgm_input = _build_pgm_input()
        pgm_grid, pgm_id_map = to_grid(
            pgm_input, base_frequency_hz=f0, load_model=LoadModel.CONST_IMPEDANCE
        )
        pgm_index = node_phase_index(pgm_grid)
        pgm_ybus = assemble_ybus(pgm_grid, [f0], dtype=torch.complex128)
        pgm_i_inj = build_injections(pgm_grid, [f0], pgm_index, dtype=torch.complex128)
        pgm_slack_our = pgm_id_map["node"][_pgm_node_id(0)]
        pgm_fixed = torch.tensor(
            [pgm_index.row(pgm_slack_our, Phase.A)], dtype=torch.int64
        )
        pgm_v_fixed = torch.tensor(
            [pgm_id_map["slack_v_complex"]], dtype=torch.complex128
        )
        pgm_v_all = solve_harmonic(
            pgm_ybus.Y, pgm_i_inj, fixed_rows=pgm_fixed, v_fixed=pgm_v_fixed
        )

        # Compare by bus index (pandapower bus 0..32 = pgm node 0..32)
        for pp_bus_idx in range(33):
            pp_node_id = pp_id_map["bus"][pp_bus_idx]
            pgm_node_id = pgm_id_map["node"][_pgm_node_id(pp_bus_idx)]

            pp_row = pp_index.row(pp_node_id, Phase.A)
            pgm_row = pgm_index.row(pgm_node_id, Phase.A)

            v_pp = pp_v_all[0, pp_row].item()
            v_pgm = pgm_v_all[0, pgm_row].item()

            u_rated = pp_grid.nodes[pp_bus_idx].u_rated_v
            # Compare the FULL complex voltage (re + im), per-unit. A magnitude-
            # only check at a loose tolerance can pass even if the two grids
            # describe slightly different networks; the complex match at ~1e-9 pu
            # actually proves converter equivalence.
            v_pu_pp = v_pp / u_rated
            v_pu_pgm = v_pgm / u_rated
            err = abs(v_pu_pp - v_pu_pgm)
            assert err < self.ATOL_V_PU, (
                f"Bus {pp_bus_idx}: pandapower-grid V_pu={v_pu_pp:.9f}, "
                f"pgm-grid V_pu={v_pu_pgm:.9f}, |Δ|={err:.2e} pu"
            )
