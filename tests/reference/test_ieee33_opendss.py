"""Oracle test: IEEE 33-bus (Baran & Wu) — our Y(60 Hz) vs OpenDSS System Y.

Test strategy
-------------
1. Load ``case33bw()`` from pandapower (the CANONICAL source of truth for
   this test) and convert it to our Grid via
   ``pgml.convert.pandapower.to_grid``.
2. Build an EQUIVALENT OpenDSS circuit from the same numerical data:
   - 1-phase Vsource at bus 0 with the same tiny Thevenin impedance.
   - 32 in-service lines with exactly the same R/X (from pandapower columns).
   - NO loads — this gives a pure passive-network Y matrix.
3. Call ``dss.Solution.Solve()`` and extract ``dss.Circuit.SystemY()``
   (dense complex array, siemens).
4. Align the DSS ``YNodeOrder`` (bus.phase names) to our compact
   ``NodePhaseIndex`` row ordering.
5. Assemble our Y(60 Hz) from the pandapower-derived Grid.
6. Subtract the const-Z load shunts that our assembly includes (because
   ``assemble_ybus`` stamps them by default) from our diagonal — leaving
   only the passive network contributions + the Vsource Norton shunt.
7. Compare entry-by-entry in absolute siemens:
   - Off-diagonal: direct comparison (no load shunts on off-diagonal).
   - Diagonal: compare after subtracting load shunts (source shunts cancel
     because both DSS and our assembly use the same tiny Thevenin Z).

EE convention notes
-------------------
- ``dss.Circuit.SystemY()`` returns the system admittance matrix with ALL
  enabled elements stamped: line pi-model series + shunt branches, PLUS the
  Vsource Norton shunt ``Y_s = Z_s^{-1}`` on the source bus diagonal.
  Disabled elements (out-of-service lines, open switches) are excluded.
- OpenDSS Load stamping: const-P loads (model=1) add a Norton shunt to Y
  at the operating voltage. Const-Z loads (model=2) add ``conj(S)/|kV|^2``
  at rated voltage. Since we add NO loads to the DSS circuit, the DSS Y is
  the passive network (lines + Vsource shunt) only.
- Our ``assemble_ybus`` stamps Load appliances as ``conj(S)/u_rated_v^2``
  (constant-impedance M1 model). By subtracting these from our Y diagonal
  before comparison, both sides represent the same passive network.
- Vsource shunts: both OpenDSS and our assembly use ``Z_s = (1e-6 + j*X)``
  ohm (tiny Thevenin), so ``Y_s = Z_s^{-1} ≈ 1e6`` S on the source bus.
  These contributions are identical and cancel in the comparison.

Tolerance targets
-----------------
- Off-diagonal (network topology): rtol = 1e-10 (floating-point only;
  actual achieved is ~7e-16 on this circuit).
- Diagonal (after load shunt subtraction): rtol = 1e-10.
- Source bus diagonal: abs tolerance 1e-4 S (the large vsource shunt ~1e6 S
  is well within floating point on both sides).

Round-trip converter test
-------------------------
Also verifies that ``convert.opendss.to_grid`` reproduces the same line
parameters (R/X per meter, length) as the canonical pandapower-converted
Grid, to floating-point precision (rtol = 1e-12).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

# numpy 2.x compatibility shim for pandapower 2.14
np.Inf = np.inf  # type: ignore[attr-defined]
np.in1d = np.isin  # type: ignore[attr-defined]

import pandapower.networks as pn  # noqa: E402

import opendssdirect as dss  # noqa: E402

from pgml.assembly import assemble_ybus, node_phase_index  # noqa: E402
from pgml.convert.pandapower import to_grid as pp_to_grid  # noqa: E402
from pgml.convert.opendss import to_grid as dss_to_grid  # noqa: E402
from pgml.schemas.grid_schema import (  # noqa: E402
    Line as GridLine,
    Load as GridLoad,
    Phase,
    Source as GridSource,
)


# ---------------------------------------------------------------------------
# Canonical data / circuit builders
# ---------------------------------------------------------------------------


def _canonical_net():
    """Return the canonical case33bw pandapower network (no modifications)."""
    return pn.case33bw()


def _build_dss_circuit_passive(net) -> None:
    """Load a single-phase positive-sequence IEEE 33-bus circuit into OpenDSS.

    Uses the same numerical line data as the pandapower network but adds
    NO loads (so ``SystemY`` reflects the pure passive network).  The Vsource
    uses the same tiny Thevenin impedance as the pandapower converter
    (``R=1e-6`` Ohm, ``L=1e-12`` H).

    Parameters
    ----------
    net:
        A ``case33bw()`` pandapower network (before ``runpp``).
    """
    f0 = float(net.f_hz)
    vn_kv = float(net.bus.at[0, "vn_kv"])  # 12.66 kV (all buses same)

    # Tiny Thevenin impedance for the Vsource (matches pandapower converter)
    r1_tiny = 1.0e-6  # Ohm
    x1_tiny = 2.0 * math.pi * f0 * 1.0e-12  # Ohm (X = 2*pi*f*L, L=1e-12 H)

    dss.Text.Command("Clear")
    dss.Text.Command(
        f"New Circuit.ieee33_passive basekv={vn_kv} pu=1.0 phases=1 "
        f"bus1=bus0.1 r1={r1_tiny} x1={x1_tiny} frequency={f0}"
    )

    for pp_idx, row in net.line[net.line["in_service"]].iterrows():
        fb = int(row["from_bus"])
        tb = int(row["to_bus"])
        r1 = float(row["r_ohm_per_km"])
        x1 = float(row["x_ohm_per_km"])
        c1 = float(row.get("c_nf_per_km", 0.0) or 0.0)
        length_km = float(row["length_km"])
        dss.Text.Command(
            f"New Line.line{pp_idx} phases=1 "
            f"bus1=bus{fb}.1 bus2=bus{tb}.1 "
            f"r1={r1} x1={x1} c1={c1} length={length_km} units=km"
        )

    dss.Text.Command(f"Set voltagebases=[{vn_kv}]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")
    assert dss.Solution.Converged(), "OpenDSS circuit did not converge"


def _build_dss_circuit_with_loads(net) -> None:
    """Load the full IEEE 33-bus circuit into OpenDSS including loads.

    Used for the ``to_grid`` round-trip converter test.
    """
    f0 = float(net.f_hz)
    vn_kv = float(net.bus.at[0, "vn_kv"])

    r1_tiny = 1.0e-6
    x1_tiny = 2.0 * math.pi * f0 * 1.0e-12

    dss.Text.Command("Clear")
    dss.Text.Command(
        f"New Circuit.ieee33_full basekv={vn_kv} pu=1.0 phases=1 "
        f"bus1=bus0.1 r1={r1_tiny} x1={x1_tiny} frequency={f0}"
    )

    for pp_idx, row in net.line[net.line["in_service"]].iterrows():
        fb = int(row["from_bus"])
        tb = int(row["to_bus"])
        r1 = float(row["r_ohm_per_km"])
        x1 = float(row["x_ohm_per_km"])
        c1 = float(row.get("c_nf_per_km", 0.0) or 0.0)
        length_km = float(row["length_km"])
        dss.Text.Command(
            f"New Line.line{pp_idx} phases=1 "
            f"bus1=bus{fb}.1 bus2=bus{tb}.1 "
            f"r1={r1} x1={x1} c1={c1} length={length_km} units=km"
        )

    for pp_idx, row in net.load[net.load["in_service"]].iterrows():
        bus = int(row["bus"])
        p_kw = float(row["p_mw"]) * 1_000.0
        q_kvar = float(row["q_mvar"]) * 1_000.0
        dss.Text.Command(
            f"New Load.load{pp_idx} phases=1 bus1=bus{bus}.1 "
            f"kv={vn_kv} kw={p_kw} kvar={q_kvar} model=1"
        )

    dss.Text.Command(f"Set voltagebases=[{vn_kv}]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")
    assert dss.Solution.Converged(), "OpenDSS circuit did not converge"


def _extract_dss_y() -> tuple[np.ndarray, list[str]]:
    """Extract system Y matrix and node order from the active DSS circuit.

    Returns
    -------
    (Y_dss, node_order)
        ``Y_dss``: complex ``[N, N]`` numpy array, siemens.
        ``node_order``: list of strings ``"BUSNAME.PHASE"`` (uppercase) of
        length N defining the row/column ordering.
    """
    node_order: list[str] = dss.Circuit.YNodeOrder()
    n = len(node_order)
    y_raw = dss.Circuit.SystemY()  # flat [G00, B00, G01, B01, ...] row-major
    y_arr = np.array(y_raw, dtype=np.float64)
    y_complex = y_arr[0::2] + 1j * y_arr[1::2]
    y_matrix = y_complex.reshape(n, n)
    return y_matrix, node_order


def _build_alignment(
    node_order: list[str], grid, id_map: dict, index
) -> dict[int, int]:
    """Map DSS Y matrix row index -> our compact node-phase row index.

    The circuit is single-phase (all entries end in ``.1``), so every DSS
    row corresponds to exactly one :class:`Phase.A` row in our index.

    Parameters
    ----------
    node_order:
        DSS ``YNodeOrder`` list, e.g. ``["BUS0.1", "BUS1.1", ...]``.
    grid:
        Our ``Grid`` object (from pandapower converter).
    id_map:
        The ``id_map`` from ``pp_to_grid``.
    index:
        Our ``NodePhaseIndex`` for the grid.

    Returns
    -------
    dict mapping DSS row index -> our compact row index.
    """
    alignment: dict[int, int] = {}
    for dss_i, entry in enumerate(node_order):
        # entry: "BUS{n}.1" (uppercase, single-phase)
        bus_name = entry.upper().split(".")[0]  # "BUSn"
        bus_num = int(bus_name[3:])  # strip "BUS" prefix
        node_id = id_map["bus"][bus_num]
        our_row = index.row(node_id, Phase.A)
        alignment[dss_i] = our_row
    return alignment


def _compute_load_shunts(grid, index) -> np.ndarray:
    """Compute the const-Z load shunt admittances that assemble_ybus stamps.

    Returns a complex 1-D array of length ``index.size`` with the aggregate
    load shunt ``y = conj(S) / u_rated_v^2`` at each row (zero for rows with
    no load).
    """
    load_shunts = np.zeros(index.size, dtype=complex)
    node_by_id = {n.id: n for n in grid.nodes}
    for app in grid.appliances:
        if isinstance(app, GridLoad) and app.in_service:
            node = node_by_id[app.node]
            u_v = node.u_rated_v
            y_load = complex(app.p_nom_w, -app.q_nom_var) / (u_v**2)
            our_row = index.rows(app.node)[0]  # Phase.A row
            load_shunts[our_row] += y_load
    return load_shunts


# ---------------------------------------------------------------------------
# Y-bus oracle tests
# ---------------------------------------------------------------------------


class TestIEEE33YBusVsOpenDSS:
    """Validate our assembled Y(60 Hz) against OpenDSS SystemY on IEEE 33-bus.

    The comparison is done on the PASSIVE NETWORK (lines + Vsource Norton shunt):
    - DSS circuit has no loads → DSS Y = passive network only.
    - Our Y is assembled from the pandapower-converted grid (which includes
      load shunts). We subtract the load shunts from our diagonal before
      comparing.

    Tolerance targets (documented at module level):
    - Off-diagonal: rtol = 1e-10
    - Diagonal: rtol = 1e-10 (after load shunt subtraction)
    """

    RTOL_OFF_DIAG: float = 1e-10
    RTOL_DIAG: float = 1e-10

    @pytest.fixture(autouse=True, scope="class")
    def _setup(self, request) -> None:
        """Build canonical grid + DSS circuit once for the whole class."""
        net = _canonical_net()
        _build_dss_circuit_passive(net)
        Y_dss, node_order = _extract_dss_y()

        grid, id_map = pp_to_grid(net)
        f0 = grid.base_frequency_hz
        index = node_phase_index(grid)
        ybus = assemble_ybus(grid, [f0], dtype=torch.complex128)
        Y_ours = ybus.Y[0].numpy()  # [N, N] complex128

        alignment = _build_alignment(node_order, grid, id_map, index)
        load_shunts = _compute_load_shunts(grid, index)

        # Reorder DSS Y to match our compact node-phase layout
        n = index.size
        Y_dss_aligned = np.zeros((n, n), dtype=complex)
        for dss_i in range(len(node_order)):
            our_i = alignment[dss_i]
            for dss_j in range(len(node_order)):
                our_j = alignment[dss_j]
                Y_dss_aligned[our_i, our_j] = Y_dss[dss_i, dss_j]

        # Our Y minus load shunts (diagonal only)
        Y_ours_net = Y_ours.copy()
        for i in range(n):
            Y_ours_net[i, i] -= load_shunts[i]

        # Attach to class for test methods
        request.cls._Y_dss_aligned = Y_dss_aligned
        request.cls._Y_ours_net = Y_ours_net
        request.cls._n = n
        request.cls._node_order = node_order
        request.cls._f0 = f0

    def test_off_diagonal_matches(self) -> None:
        """Off-diagonal entries match DSS to rtol = 1e-10 (floating-point only).

        Off-diagonal entries carry only line series admittances; loads/sources
        do not affect them. The error is expected to be at or below floating-
        point rounding (~7e-16) on this circuit.
        """
        Y_dss = self._Y_dss_aligned
        Y_ours = self._Y_ours_net
        n = self._n

        max_err = 0.0
        max_rtol = 0.0
        worst = (0, 0)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dss_val = Y_dss[i, j]
                our_val = Y_ours[i, j]
                if abs(dss_val) < 1e-10:
                    continue  # structural zero — skip
                err = abs(dss_val - our_val)
                rtol = err / abs(dss_val)
                if rtol > max_rtol:
                    max_rtol = rtol
                    max_err = err
                    worst = (i, j)

        assert max_rtol < self.RTOL_OFF_DIAG, (
            f"Off-diagonal Y mismatch: max rtol={max_rtol:.2e} at row ({worst}): "
            f"dss={Y_dss[worst]:.4e} ours={Y_ours[worst]:.4e} abs_err={max_err:.2e} S"
        )

    def test_diagonal_after_load_subtraction_matches(self) -> None:
        """Diagonal (minus load shunts) matches DSS to rtol = 1e-10.

        After subtracting the const-Z load shunts that our assembly stamps,
        the diagonal should equal the DSS passive-network diagonal (which
        includes the Vsource Norton shunt on bus 0 and only line contributions
        on the remaining buses).
        """
        Y_dss = self._Y_dss_aligned
        Y_ours = self._Y_ours_net
        n = self._n
        node_order = self._node_order

        max_rtol = 0.0
        worst_idx = 0

        for i in range(n):
            dss_val = Y_dss[i, i]
            our_val = Y_ours[i, i]
            err = abs(dss_val - our_val)
            rtol = err / max(abs(dss_val), 1e-6)
            if rtol > max_rtol:
                max_rtol = rtol
                worst_idx = i

        assert max_rtol < self.RTOL_DIAG, (
            f"Diagonal Y mismatch: max rtol={max_rtol:.2e} at row {worst_idx} "
            f"({node_order[worst_idx] if worst_idx < len(node_order) else 'unknown'}): "
            f"dss={Y_dss[worst_idx, worst_idx]:.4e} "
            f"ours(net)={Y_ours[worst_idx, worst_idx]:.4e}"
        )

    def test_source_bus_diagonal_dominated_by_vsource_shunt(self) -> None:
        """Verify bus-0 diagonal is dominated by the Vsource Norton shunt.

        Both DSS and our assembly stamp ``Y_s = 1/(1e-6 + j*2*pi*f*1e-12)``,
        so their bus-0 diagonals should match to floating-point precision
        (the Vsource shunt ~1e6 S is 5 orders of magnitude larger than the
        line admittances ~1-10 S, so rtol=1e-10 on the total is trivially met).
        """
        Y_dss = self._Y_dss_aligned
        Y_ours = self._Y_ours_net
        # Row 0 is bus 0 (the Vsource bus, id_map['bus'][0] is the first node)
        # Both should have the same large diagonal
        dss_diag0 = Y_dss[0, 0]
        our_diag0 = Y_ours[0, 0]
        assert abs(dss_diag0).real > 1e5, (
            f"Expected large Vsource shunt, got {dss_diag0}"
        )
        err = abs(dss_diag0 - our_diag0)
        rtol = err / abs(dss_diag0)
        assert rtol < self.RTOL_DIAG, (
            f"Source bus diagonal mismatch: dss={dss_diag0:.4e} ours={our_diag0:.4e} "
            f"rtol={rtol:.2e}"
        )

    def test_matrix_symmetry(self) -> None:
        """Both Y matrices are symmetric (passive network is reciprocal)."""
        Y_dss = self._Y_dss_aligned
        Y_ours = self._Y_ours_net

        sym_err_dss = np.max(np.abs(Y_dss - Y_dss.T))
        sym_err_ours = np.max(np.abs(Y_ours - Y_ours.T))

        assert sym_err_dss < 1e-10, (
            f"DSS Y is not symmetric: max|Y-Y^T|={sym_err_dss:.2e}"
        )
        assert sym_err_ours < 1e-10, (
            f"Our Y is not symmetric: max|Y-Y^T|={sym_err_ours:.2e}"
        )

    def test_topology_matches_expected_sparsity(self) -> None:
        """Non-zero off-diagonal entries match IEEE 33-bus topology (32 lines)."""
        Y_ours = self._Y_ours_net
        n = self._n
        # Each in-service line contributes two off-diagonal non-zero entries
        # (symmetric), so we expect 2 * 32 = 64 non-zero off-diagonal entries.
        mask = ~np.eye(n, dtype=bool)
        n_nonzero = int(np.sum(np.abs(Y_ours[mask]) > 1e-10))
        assert n_nonzero == 64, (
            f"Expected 64 non-zero off-diagonal entries (32 lines * 2), got {n_nonzero}"
        )


# ---------------------------------------------------------------------------
# Load and Vsource shunt accounting tests
# ---------------------------------------------------------------------------


class TestShuntAccounting:
    """Document and verify the shunt admittances that explain the diagonal delta.

    These tests verify the EE convention noted in the module docstring:
    ``our_Y[i,i] - load_shunt[i] == DSS_Y[i,i]`` for all buses (excluding
    the Vsource bus where both sides include the same large ``Y_s``).
    """

    ATOL_SHUNT: float = 1e-6  # S (absolute tolerance on shunt verification)

    def test_load_shunt_formula(self) -> None:
        """Verify load shunt formula: y = conj(P+jQ) / u_rated_v^2.

        For load on bus 1 (p=0.1 MW, q=0.06 MVAr, vn_kv=12.66 kV):
        y = (0.1e6 - 0.06e6j) / (12660)^2 = (6.239e-4 - 3.744e-4j) S.
        This should equal (our_Y[bus1,bus1] - DSS_Y_passive[bus1,bus1]).
        """
        net = _canonical_net()
        _build_dss_circuit_passive(net)
        Y_dss, node_order = _extract_dss_y()

        grid, id_map = pp_to_grid(net)
        f0 = grid.base_frequency_hz
        index = node_phase_index(grid)
        ybus = assemble_ybus(grid, [f0], dtype=torch.complex128)
        Y_ours = ybus.Y[0].numpy()

        alignment = _build_alignment(node_order, grid, id_map, index)

        # Bus 1 in pandapower maps to node id_map['bus'][1]
        our_row_1 = index.row(id_map["bus"][1], Phase.A)
        dss_row_1 = next(k for k, v in alignment.items() if v == our_row_1)

        delta = Y_ours[our_row_1, our_row_1] - Y_dss[dss_row_1, dss_row_1]

        # Expected: load on bus 1 has p=0.1 MW, q=0.06 MVAr
        p_w = float(net.load[net.load["bus"] == 1]["p_mw"].iloc[0]) * 1e6
        q_var = float(net.load[net.load["bus"] == 1]["q_mvar"].iloc[0]) * 1e6
        v_ll = float(net.bus.at[1, "vn_kv"]) * 1_000.0
        y_expected = complex(p_w, -q_var) / (v_ll**2)

        err = abs(delta - y_expected)
        assert err < self.ATOL_SHUNT, (
            f"Load shunt mismatch at bus 1: "
            f"delta={delta:.4e} expected={y_expected:.4e} err={err:.2e} S"
        )

    def test_all_load_shunts_account_for_diagonal_delta(self) -> None:
        """For every load bus: our_Y[i,i] - DSS_Y[i,i] ≈ sum(load_shunts_at_i).

        The maximum absolute error across all 32 load buses should be < 1e-6 S.
        """
        net = _canonical_net()
        _build_dss_circuit_passive(net)
        Y_dss, node_order = _extract_dss_y()

        grid, id_map = pp_to_grid(net)
        f0 = grid.base_frequency_hz
        index = node_phase_index(grid)
        ybus = assemble_ybus(grid, [f0], dtype=torch.complex128)
        Y_ours = ybus.Y[0].numpy()

        alignment = _build_alignment(node_order, grid, id_map, index)
        load_shunts = _compute_load_shunts(grid, index)

        max_err = 0.0
        for dss_i, our_i in alignment.items():
            dss_val = Y_dss[dss_i, dss_i]
            our_val = Y_ours[our_i, our_i]
            delta = our_val - dss_val
            expected_delta = load_shunts[our_i]
            err = abs(delta - expected_delta)
            if err > max_err:
                max_err = err

        assert max_err < self.ATOL_SHUNT, (
            f"Load shunt accounting mismatch: max abs error = {max_err:.2e} S "
            f"(expected < {self.ATOL_SHUNT:.1e} S)"
        )


# ---------------------------------------------------------------------------
# OpenDSS converter round-trip tests
# ---------------------------------------------------------------------------


class TestOpenDSSConverterRoundTrip:
    """Verify that ``convert.opendss.to_grid`` reproduces the canonical Grid.

    The canonical Grid is obtained from the pandapower converter; the DSS
    converter re-derives it from the OpenDSS engine state.  Line parameters
    should match to floating-point precision.
    """

    RTOL_LINE_PARAMS: float = 1e-12  # relative tolerance for per-meter params

    @pytest.fixture(autouse=True, scope="class")
    def _setup(self, request) -> None:
        net = _canonical_net()
        _build_dss_circuit_with_loads(net)

        grid_dss, id_map_dss = dss_to_grid(dss)
        grid_pp, id_map_pp = pp_to_grid(net)

        request.cls._grid_dss = grid_dss
        request.cls._grid_pp = grid_pp
        request.cls._id_map_dss = id_map_dss
        request.cls._id_map_pp = id_map_pp
        request.cls._net = net

    def test_node_count(self) -> None:
        """DSS-converted grid has 33 nodes."""
        assert len(self._grid_dss.nodes) == 33

    def test_branch_count(self) -> None:
        """DSS-converted grid has 32 branches (in-service lines only)."""
        assert len(self._grid_dss.branches) == 32

    def test_base_frequency(self) -> None:
        """Base frequency matches pandapower network."""
        assert self._grid_dss.base_frequency_hz == pytest.approx(
            self._grid_pp.base_frequency_hz
        )

    def test_line_r_per_m_matches_pandapower(self) -> None:
        """Series resistance per meter matches the pandapower converter to rtol=1e-12."""
        id_map_pp = self._id_map_pp
        pp_lines = {b.id: b for b in self._grid_pp.branches if isinstance(b, GridLine)}
        dss_lines = {
            b.name: b for b in self._grid_dss.branches if isinstance(b, GridLine)
        }

        max_rtol = 0.0
        for pp_idx in id_map_pp["line"]:
            pp_line = pp_lines[id_map_pp["line"][pp_idx]]
            dss_line = dss_lines.get(f"line{pp_idx}")
            if dss_line is None:
                continue

            pp_r = pp_line.series_resistance_ohm_per_m[0][0]
            dss_r = dss_line.series_resistance_ohm_per_m[0][0]
            err = abs(pp_r - dss_r) / max(abs(pp_r), 1e-20)
            if err > max_rtol:
                max_rtol = err

        assert max_rtol < self.RTOL_LINE_PARAMS, (
            f"series_resistance_ohm_per_m round-trip rtol={max_rtol:.2e} "
            f"(expected < {self.RTOL_LINE_PARAMS:.1e})"
        )

    def test_line_l_per_m_matches_pandapower(self) -> None:
        """Series inductance per meter matches the pandapower converter to rtol=1e-12."""
        id_map_pp = self._id_map_pp
        pp_lines = {b.id: b for b in self._grid_pp.branches if isinstance(b, GridLine)}
        dss_lines = {
            b.name: b for b in self._grid_dss.branches if isinstance(b, GridLine)
        }

        max_rtol = 0.0
        for pp_idx in id_map_pp["line"]:
            pp_line = pp_lines[id_map_pp["line"][pp_idx]]
            dss_line = dss_lines.get(f"line{pp_idx}")
            if dss_line is None:
                continue

            pp_l = pp_line.series_inductance_h_per_m[0][0]
            dss_l = dss_line.series_inductance_h_per_m[0][0]
            err = abs(pp_l - dss_l) / max(abs(pp_l), 1e-20)
            if err > max_rtol:
                max_rtol = err

        assert max_rtol < self.RTOL_LINE_PARAMS, (
            f"series_inductance_h_per_m round-trip rtol={max_rtol:.2e} "
            f"(expected < {self.RTOL_LINE_PARAMS:.1e})"
        )

    def test_id_map_buses_present(self) -> None:
        """All 33 buses appear in the DSS id_map."""
        id_map_dss = self._id_map_dss
        # DSS bus names are bus0 .. bus32
        for bus_num in range(33):
            assert f"bus{bus_num}" in id_map_dss["bus"], (
                f"bus{bus_num} missing from DSS id_map"
            )

    def test_vsource_in_id_map(self) -> None:
        """The Vsource (named 'source' by OpenDSS default) appears in id_map."""
        assert "source" in self._id_map_dss["vsource"]

    def test_load_count(self) -> None:
        """DSS-converted grid has 32 loads (one per load bus in case33bw)."""
        loads = [a for a in self._grid_dss.appliances if isinstance(a, GridLoad)]
        assert len(loads) == 32

    def test_vsource_count(self) -> None:
        """DSS-converted grid has exactly 1 Vsource."""
        sources = [a for a in self._grid_dss.appliances if isinstance(a, GridSource)]
        assert len(sources) == 1

    def test_load_p_matches_pandapower(self) -> None:
        """Total active load power matches pandapower to 0.1 W."""
        net = self._net
        pp_total_p = float(net.load[net.load["in_service"]]["p_mw"].sum()) * 1e6
        dss_loads = [a for a in self._grid_dss.appliances if isinstance(a, GridLoad)]
        dss_total_p = sum(a.p_nom_w for a in dss_loads)
        assert abs(pp_total_p - dss_total_p) < 0.1, (
            f"Total P mismatch: pp={pp_total_p:.1f} W, dss={dss_total_p:.1f} W"
        )
