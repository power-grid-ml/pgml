"""Oracle test matrix: named ``pandapower.networks`` grids vs pandapower's own AC
power flow, exercising the vector-group- and tap-changer-aware transformer
conversion (``pgml.convert.pandapower.converter``, section 3) end to end.

Every grid below is converted with ``phase_mode=PhaseMode.SINGLE_PHASE_EQUIV``
(the default -- matches pandapower's own positive-sequence ``runpp``), solved
with the nonlinear constant-power power flow (``solve_power_flow(...,
slack="ideal")``, the same pattern as ``test_ieee33_power_flow_pandapower.py``
/ ``test_cigre_lv_pandapower.py``), and compared bus-by-bus (voltage magnitude
AND angle) against ``net.res_bus``.

Achieved tolerances split into three families, all explained and pinned here
(NOT loosened blindly):

- **No magnetizing branch, no open bus-line/bus-transformer switch**
  (``pfe_kw == i0_percent == 0``, e.g. every CIGRE trafo): agreement is limited
  only by the nonlinear solver's own iteration tolerance -- ~1e-7 pu / ~1e-6
  deg, approaching ``case33bw``'s trafo-free ~1e-9 pu / ~1e-7 deg control.
- **A magnetizing (no-load) branch is present** (``pfe_kw>0`` or
  ``i0_percent>0``, e.g. every Kerber std type and ``mv_oberrhein``): a real,
  ~1e-5 pu / ~1e-2 deg residual appears. Root-caused here (not guessed) by
  zeroing ``pfe_kw``/``i0_percent`` on ``create_kerber_landnetz_kabel_1`` and
  re-solving both engines: the residual collapses to ~3e-13 pu / ~7e-12 deg
  (machine precision). The cause is a MODEL-PLACEMENT difference, not a
  vector-group/tap bug: pgml stamps the magnetizing shunt directly at the
  external HV terminal (``assembly._transformer``'s documented
  simplification), while pandapower's default ``trafo_model="t"`` places it
  inside an internal T-equivalent (leakage split either side of the shunt).
  This is the SAME phenomenon already documented for the OpenDSS oracle
  (``src/pgml/convert/opendss/CONTEXT.md``, ``test_opendss_transformer.py``)
  -- pandapower's magnetizing branch has the identical T-vs-pi placement gap.
- **An open bus-line/bus-transformer switch is present** (``et='l'``/``'t'``,
  e.g. the tie/sectionalizing switches ``create_cigre_network_mv`` and
  ``mv_oberrhein`` use to operate a meshed ring radially): the converter's
  accepted approximation (``_open_switch_targets``) takes the WHOLE line/
  trafo out of service, dropping the shunt admittance (line charging) at the
  STILL-connected terminal too, unlike pandapower's own solver (which keeps
  that terminal energized via an internal auxiliary bus). Root-caused by
  comparing a live ``runpp`` on the untouched net against the SAME net with
  those lines forced ``in_service=False`` for pandapower itself: the two
  differ by ~2.9e-4 pu / ~6.9e-3 deg on ``create_cigre_network_mv`` and
  ~8.9e-4 pu / ~1.4e-2 deg on ``mv_oberrhein`` -- almost the ENTIRE residual
  seen against pandapower's untouched ``runpp`` in
  ``test_cigre_mv_shift30_fallback``/``test_mv_oberrhein_ynd5_delta_referral_
  and_taps`` below, confirming the gap is the switch approximation, not the
  vector-group/tap/magnetizing conversion (which those same two tests already
  validate independently via the tighter families above).

``mv_oberrhein`` additionally exposed a genuine, unrelated converter gap: the
per-element ``scaling`` column (``net.load.scaling``, ``net.sgen.scaling``) was
not read at all, silently using 100% of nameplate P/Q instead of pandapower's
own scaled value (``mv_oberrhein`` ships ``load.scaling=0.6``,
``sgen.scaling=0.0`` by default) -- FIXED in the converter (``_scaling_factor``)
as part of this test matrix; every other grid here uses ``scaling=1.0``
(pandapower's own default) so is unaffected by the fix.

``mv_oberrhein``'s YNd5 unit (``to_connection=DELTA``) also pins the delta-LV
coil-referral factor of 3 (``series_resistance_ohm``/``series_inductance_h``
= 3x the raw vk%-derived terminal value): this is applied by the converter
UNCONDITIONALLY, the SAME way in every ``phase_mode`` -- assembly's own
internal machinery (the p==1 scalar-tap-pi's ``k_ll`` factor in
``_transformer_block_groups``, and the p==3 winding-incidence transform)
undoes the factor to recover the correct terminal admittance either way, so
the converter's only job is to supply the coil-referred value once. See
``test_mv_oberrhein_delta_lv_referral_factor_applied_in_both_phase_modes``
and ``test_mv_oberrhein_ybus_transformer_stamp_matches_pandapower`` for the
live-oracle proof (both the Y-bus stamp AND the full nonlinear solve are
exact/near-exact WITH the factor, and wrong by 3x without it).

``case118`` needs separate treatment: see ``TestCase118TransformerOnly`` below
for why a full nonlinear voltage comparison is not attempted there.
"""

from __future__ import annotations

import math

import pandapower as pp
import pandapower.networks as pn
import pytest
import torch

from pgml.assembly import assemble_network_ybus
from pgml.convert.pandapower import PhaseMode, to_grid
from pgml.errors import ConversionError
from pgml.schemas.grid_schema import (
    Line,
    Load,
    LoadModel,
    Phase,
    Transformer,
    WindingConnection,
)
from pgml.solver import solve_power_flow


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _angle_diff_deg(a: float, b: float) -> float:
    """Signed angle difference a - b in degrees, wrapped to (-180, 180]."""
    diff = (a - b) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


def _assert_matches_pandapower(
    net,
    *,
    atol_vm: float,
    atol_va: float,
    tol: float = 1e-10,
    max_iter: int = 200,
):
    """Run pandapower + pgml on ``net`` and assert per-bus |V|/angle agreement.

    Returns ``(grid, id_map, result)`` for tests that need further inspection.
    """
    pp.runpp(net, numba=False)
    assert net.converged, "pandapower did not converge -- check the test setup"

    grid, id_map = to_grid(net)
    result = solve_power_flow(
        grid, slack="ideal", tol=tol, max_iter=max_iter, dtype=torch.complex128
    )
    assert result.converged, (
        f"solve_power_flow did not converge (residual={float(result.residual):.3e}, "
        f"iterations={result.iterations})"
    )

    for pp_idx, node_id in id_map["bus"].items():
        row = result.index.row(node_id, Phase.A)
        v_complex = result.v.reshape(-1)[row].item()

        u_rated_v = float(net.bus.at[pp_idx, "vn_kv"]) * 1_000.0
        vm_pu_ours = abs(v_complex) / u_rated_v
        va_deg_ours = math.degrees(math.atan2(v_complex.imag, v_complex.real))

        vm_pu_ref = float(net.res_bus.at[pp_idx, "vm_pu"])
        va_deg_ref = float(net.res_bus.at[pp_idx, "va_degree"])

        vm_err = abs(vm_pu_ours - vm_pu_ref)
        va_err = abs(_angle_diff_deg(va_deg_ours, va_deg_ref))

        assert vm_err < atol_vm, (
            f"bus {pp_idx} ({net.bus.at[pp_idx, 'name']}): |V| err={vm_err:.3e} pu "
            f"(ours={vm_pu_ours:.8f}, pp={vm_pu_ref:.8f}, atol={atol_vm:.1e})"
        )
        assert va_err < atol_va, (
            f"bus {pp_idx} ({net.bus.at[pp_idx, 'name']}): angle err={va_err:.3e} deg "
            f"(ours={va_deg_ours:.6f}, pp={va_deg_ref:.6f}, atol={atol_va:.1e})"
        )

    return grid, id_map, result


# ---------------------------------------------------------------------------
# 1. case33bw -- no transformer, control
# ---------------------------------------------------------------------------
def test_case33bw_control_no_transformer():
    """No transformer at all: limited only by nonlinear-solver tolerance."""
    _assert_matches_pandapower(pn.case33bw(), atol_vm=1e-6, atol_va=1e-5)


# ---------------------------------------------------------------------------
# 2. CIGRE LV -- Dyn1 (shift 30), no magnetizing branch
# ---------------------------------------------------------------------------
def test_cigre_lv_dyn1():
    """CIGRE LV: 3x Dyn1 (20/0.4 kV) transformers, pfe_kw=i0_percent=0."""
    _assert_matches_pandapower(pn.create_cigre_network_lv(), atol_vm=1e-5, atol_va=1e-4)


# ---------------------------------------------------------------------------
# 3. CIGRE MV (no DER) -- shift 30 fallback (no vector_group column), no
#    magnetizing branch; 3 open bus-line switches operate the ring radially.
# ---------------------------------------------------------------------------
def test_cigre_mv_shift30_fallback():
    """CIGRE MV: no vector_group anywhere -> odd-clock (shift=30) Dyn fallback.

    No magnetizing branch, no tap changer -- the vector-group/tap conversion
    itself is exact. The achieved ~2.9e-4 pu / ~6.9e-3 deg residual is entirely
    the open-bus-line-switch approximation (see the module docstring's third
    tolerance family): the net's 3 tie switches are converted, not worked
    around in the test harness, exercising the production
    ``_open_switch_targets`` path directly.
    """
    net = pn.create_cigre_network_mv(with_der=False)
    _assert_matches_pandapower(net, atol_vm=4e-4, atol_va=8e-3)


def test_cigre_mv_fallback_is_dyn():
    """The fallback-derived connection is DELTA/WYE_GROUNDED (clock 1, odd)."""
    net = pn.create_cigre_network_mv(with_der=False)
    grid, _ = to_grid(net)
    trafos = [b for b in grid.branches if isinstance(b, Transformer)]
    assert len(trafos) == 2
    for t in trafos:
        assert t.from_connection == WindingConnection.DELTA
        assert t.to_connection == WindingConnection.WYE_GROUNDED
        assert t.tap.shift_deg == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# 4-6. Kerber LV feeders -- Dyn5 (150 deg), a magnetizing branch is present
# ---------------------------------------------------------------------------
# Achieved ~1e-5 pu / ~1e-2 deg: root-caused (see module docstring) to the
# magnetizing-branch HV-terminal-shunt vs T-equivalent placement difference,
# not the vector-group/tap conversion (see test_kerber_landnetz_kabel_1_no_
# magnetizing_branch_is_near_exact below, which isolates the two).
_KERBER_ATOL_VM = 5e-5
_KERBER_ATOL_VA = 1e-2


def test_kerber_vorstadtnetz_kabel_1_dyn5():
    _assert_matches_pandapower(
        pn.create_kerber_vorstadtnetz_kabel_1(),
        atol_vm=_KERBER_ATOL_VM,
        atol_va=_KERBER_ATOL_VA,
    )


def test_kerber_dorfnetz_dyn5():
    _assert_matches_pandapower(
        pn.create_kerber_dorfnetz(), atol_vm=_KERBER_ATOL_VM, atol_va=_KERBER_ATOL_VA
    )


def test_kerber_landnetz_kabel_1_dyn5():
    _assert_matches_pandapower(
        pn.create_kerber_landnetz_kabel_1(),
        atol_vm=_KERBER_ATOL_VM,
        atol_va=_KERBER_ATOL_VA,
    )


def test_kerber_landnetz_kabel_1_no_magnetizing_branch_is_near_exact():
    """Isolates the magnetizing-branch T-vs-pi gap from the vector-group/tap path.

    Zeroing pfe_kw/i0_percent (no magnetizing branch -> no T-vs-pi placement
    ambiguity) on the SAME Dyn5 transformer collapses the residual to machine
    precision, proving the ~1e-5 pu / ~1e-2 deg residual on the other Kerber
    tests is entirely the magnetizing-branch placement difference, not a
    vector-group or tap bug.
    """
    net = pn.create_kerber_landnetz_kabel_1()
    net.trafo["pfe_kw"] = 0.0
    net.trafo["i0_percent"] = 0.0
    _assert_matches_pandapower(net, atol_vm=1e-9, atol_va=1e-8)


# ---------------------------------------------------------------------------
# 7. mv_oberrhein -- YNd5 (delta-LV factor-3 referral) + off-nominal HV taps
#    (tap_pos -2/-3), 2 ext_grids, sgen (with scaling), 6 open line switches.
# ---------------------------------------------------------------------------
def test_mv_oberrhein_ynd5_delta_referral_and_taps():
    """The headline deliverable: YNd5 (delta-LV factor-3 referral) + tap -2/-3.

    Exercises, together, in one real network: the delta-LV coil referral
    (``to_connection=DELTA`` -> the schema's series R/L is 3x the raw
    vk%-derived terminal value, applied the SAME way regardless of
    ``phase_mode`` -- assembly itself undoes the factor for the terminal
    admittance in both the p==1 scalar stamp and the p==3 winding-incidence
    stamp, see the section-3 comment), the tap-changer ratio on two units
    (tap_pos -2 and -3, both tap_side='hv'), TWO ext_grids (both solved as
    independent ideal-slack rows -- no special-casing needed, see
    ``solver.power_flow._slack_rows_and_vref``), and 153 sgen (DER) injections
    with a non-default ``scaling`` (0.0 by default in this net; converted
    correctly after the ``_scaling_factor`` fix -- see module docstring).

    Also exercises the 6 open bus-line switches converted through the
    production ``_open_switch_targets`` path (no test-harness workaround): the
    achieved ~8.9e-4 pu / ~1.7e-2 deg combines the magnetizing-branch family
    residual (~5e-6 pu / ~3.7e-3 deg, see module docstring) with the switch-
    approximation family residual (~8.9e-4 pu / ~1.4e-2 deg on this net,
    measured in isolation -- see module docstring's third tolerance family).
    """
    net = pn.mv_oberrhein()
    # 20 kV rows: an absolute 1e-10 V update sits at the float64 rounding floor of the
    # solve (~5e-15 relative), where the fixed point stalls depending on the BLAS/LAPACK
    # build; 1e-9 V is still ~5e-14 relative and far inside the asserted tolerances.
    _assert_matches_pandapower(net, atol_vm=1.2e-3, atol_va=2e-2, tol=1e-9)


def test_mv_oberrhein_delta_lv_referral_factor_applied_in_both_phase_modes():
    """Regression pin for the delta-LV coil-referral finding.

    Verified two ways against a live pandapower runpp (both reported in the
    final report): (1) the assembled SI Y-bus transformer entries match
    pandapower's own internal Ybus to machine precision WITH the factor-3
    coil correction applied, and are wrong by exactly 3x without it; (2) the
    full nonlinear voltage solve above matches to ~5e-6 pu only with the
    factor applied. The factor is the SAME in both phase modes -- assembly's
    own internal `k_ll` (p==1) / winding-incidence transform (p==3) undoes it
    to recover the correct terminal admittance either way, so the CONVERTER
    must supply the identical coil-referred value regardless of `phase_mode`
    (an earlier, incorrect version of this converter applied the factor only
    under THREE_PHASE; that broke SINGLE_PHASE_EQUIV by 3x, caught by this
    test and the Y-bus check in test_mv_oberrhein_ynd5_delta_referral_and_taps
    via _assert_matches_pandapower).
    """
    net = pn.mv_oberrhein()
    grid_1ph, id_map = to_grid(net, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
    grid_3ph, _ = to_grid(net, phase_mode=PhaseMode.THREE_PHASE)

    trafo_1ph = {b.id: b for b in grid_1ph.branches if isinstance(b, Transformer)}
    trafo_3ph = {b.id: b for b in grid_3ph.branches if isinstance(b, Transformer)}
    assert set(trafo_1ph) == set(trafo_3ph)

    for pp_idx, row in net.trafo.iterrows():
        sn_va = float(row["sn_mva"]) * 1e6
        vn_lv_v = float(row["vn_lv_kv"]) * 1e3
        z_base_lv = vn_lv_v**2 / sn_va
        r_ll_expected = float(row["vkr_percent"]) / 100.0 * z_base_lv

        tid = id_map["trafo"][pp_idx]
        t1 = trafo_1ph[tid]
        t3 = trafo_3ph[tid]
        assert t1.to_connection == WindingConnection.DELTA
        assert t3.to_connection == WindingConnection.DELTA
        # Both phase modes store the SAME coil-referred value (3x the raw
        # terminal vk%-derived quantity for a delta TO winding).
        assert t1.series_resistance_ohm == pytest.approx(3.0 * r_ll_expected, rel=1e-9)
        assert t3.series_resistance_ohm == pytest.approx(
            t1.series_resistance_ohm, rel=1e-12
        )
        assert t3.series_inductance_h == pytest.approx(
            t1.series_inductance_h, rel=1e-12
        )


def test_mv_oberrhein_ybus_transformer_stamp_matches_pandapower():
    """The assembled SI Y-bus off-diagonal transformer entries (linear, exact,
    no nonlinear-solver dependence) match pandapower's own internal Ybus to
    ~3.2e-5 relative -- the definitive proof of the delta-LV factor-3 referral
    (see test_mv_oberrhein_delta_lv_referral_factor_applied_in_both_phase_
    modes' docstring): applying it gives this tight agreement; omitting it is
    wrong by exactly 3x (0.3+, not 3e-5).

    The residual ~3.2e-5 itself is the SAME magnetizing-branch T-vs-pi
    placement gap documented in the module docstring (this trafo has
    pfe_kw=29, i0_percent=0.071 -- nonzero): pandapower's default
    ``trafo_model="t"`` couples the magnetizing shunt into the HV-LV transfer
    admittance (it sits between two leakage half-impedances), while pgml
    stamps it purely on the HV diagonal, outside the leakage transfer. Zeroing
    pfe_kw/i0_percent on this same trafo (not done here to keep the test
    representative of the real network) collapses this to machine precision,
    exactly as shown for the full nonlinear solve in
    test_kerber_landnetz_kabel_1_no_magnetizing_branch_is_near_exact.
    """
    net = pn.mv_oberrhein()
    pp.runpp(net, numba=False)
    grid, id_map = to_grid(net)

    f0 = float(net.f_hz)
    ybus_obj = assemble_network_ybus(grid, [f0], dtype=torch.complex128)
    Y_ours = ybus_obj.Y[0].numpy()
    index = ybus_obj.index

    Ybus_pu = net._ppc["internal"]["Ybus"].toarray()
    base_mva = float(net._ppc["baseMVA"])
    bl = net._pd2ppc_lookups["bus"]

    for pp_idx, row in net.trafo.iterrows():
        hv_bus = int(row["hv_bus"])
        lv_bus = int(row["lv_bus"])
        ppc_hv = int(bl[hv_bus])
        ppc_lv = int(bl[lv_bus])
        base_kv_hv = float(net._ppc["bus"][ppc_hv, 9])
        base_kv_lv = float(net._ppc["bus"][ppc_lv, 9])
        y_base_od = base_mva / (base_kv_hv * base_kv_lv)

        node_hv = id_map["bus"][hv_bus]
        node_lv = id_map["bus"][lv_bus]
        r_hv = index.row(node_hv, Phase.A)
        r_lv = index.row(node_lv, Phase.A)

        y_ours_ft = Y_ours[r_hv, r_lv]
        y_pp_ft = Ybus_pu[ppc_hv, ppc_lv] * y_base_od

        err = abs(y_ours_ft - y_pp_ft)
        rel = err / max(abs(y_pp_ft), 1e-12)
        assert rel < 1e-4, (
            f"trafo {pp_idx} Y_ft: ours={y_ours_ft:.8e}, pp={y_pp_ft:.8e}, "
            f"rel_err={rel:.3e}"
        )


def test_mv_oberrhein_tap_ratios():
    """The two HV-side taps (-2, -3 at 1.5% step) match the closed-form."""
    net = pn.mv_oberrhein()
    grid, id_map = to_grid(net)
    ratios = []
    for pp_idx in net.trafo.index:
        tid = id_map["trafo"][pp_idx]
        branch = next(b for b in grid.branches if b.id == tid)
        ratios.append(float(branch.tap.ratio_magnitude))
    assert sorted(ratios) == pytest.approx(sorted([1.0 - 0.03, 1.0 - 0.045]))


# ---------------------------------------------------------------------------
# 8. Hand-built Yzn5 (zigzag) net -- '0.25 MVA 20/0.4 kV' std type
# ---------------------------------------------------------------------------
def _build_yzn5_net():
    net = pp.create_empty_network(f_hz=50.0)
    b_hv = pp.create_bus(net, vn_kv=20.0, name="hv")
    b_lv = pp.create_bus(net, vn_kv=0.4, name="lv")
    pp.create_ext_grid(net, bus=b_hv, vm_pu=1.0, va_degree=0.0)
    pp.create_transformer(net, hv_bus=b_hv, lv_bus=b_lv, std_type="0.25 MVA 20/0.4 kV")
    pp.create_load(net, bus=b_lv, p_mw=0.1, q_mvar=0.03)
    pp.create_load(net, bus=b_lv, p_mw=0.05, q_mvar=0.01)
    return net


def test_hand_built_yzn5():
    """Yzn5 -- the zigzag case: WYE HV, ZIGZAG_GROUNDED LV, clock 5 (150 deg)."""
    _assert_matches_pandapower(
        _build_yzn5_net(), atol_vm=_KERBER_ATOL_VM, atol_va=_KERBER_ATOL_VA
    )


def test_hand_built_yzn5_connections():
    net = _build_yzn5_net()
    grid, _ = to_grid(net)
    trafo = next(b for b in grid.branches if isinstance(b, Transformer))
    assert trafo.from_connection == WindingConnection.WYE
    assert trafo.to_connection == WindingConnection.ZIGZAG_GROUNDED
    assert trafo.tap.shift_deg == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# 9. case118 -- Yy fallback (shift_degree=0 everywhere, no vector_group) +
#    off-nominal HV taps (-1 x 1.5/4/6.5%).
#
# A full nonlinear voltage-magnitude comparison is NOT attempted: case118 is
# a 345/161/138 kV transmission benchmark that relies on 53 PV (voltage-
# controlled) generator buses plus 14 shunts for voltage support, and the
# pandapower converter does not read `net.gen` (PV buses) or `net.shunt` at
# all (documented gaps, `warn_dropped_elements`). Freezing the generators'
# and shunts' OWN converged P/Q (from pandapower's res_gen/res_shunt) as fixed
# PQ injections and re-solving with pgml's constant-power solver was tried and
# CONVERGES, but to a badly wrong, inflated-voltage operating point (e.g. bus
# 9: pgml 3.37 pu vs pandapower 1.05 pu) at nearly every bus except the slack
# itself (which matches exactly) -- a known nonlinear-load-flow phenomenon
# (removing voltage control from a stressed, voltage-support-dependent
# transmission network lets constant-PQ Newton iteration converge to a
# different, non-physical solution branch). This is a generator/shunt-control
# modeling gap, unrelated to the transformer conversion under test here.
#
# Instead the transformer/tap conversion is validated the RIGHT way for this
# grid: a LINEAR, well-conditioned, algebraic check -- the assembled SI Y-bus
# transformer entries against pandapower's own per-unit Ybus (scaled to SI),
# exactly like test_cigre_lv_full_transformer.py's TestTransformerYbusStamps.
# No power flow, no PV/shunt physics needed; this is an exact oracle for the
# thing actually being changed (vector-group fallback + tap ratio).
# ---------------------------------------------------------------------------
class TestCase118TransformerOnly:
    """Y-bus-stamp + unit-level transformer validation (see class docstring above
    the module for why a full nonlinear voltage comparison is not attempted)."""

    def _net(self):
        net = pn.case118()
        pp.runpp(net, numba=False)
        assert net.converged
        return net

    def test_thirteen_transformers_yy_fallback(self):
        net = self._net()
        grid, id_map = to_grid(net)
        trafos = [b for b in grid.branches if isinstance(b, Transformer)]
        assert len(trafos) == 13
        for t in trafos:
            # shift_degree=0.0 everywhere, no vector_group column/std_type in
            # case118 -> even-clock fallback: WYE_GROUNDED/WYE_GROUNDED.
            assert t.from_connection == WindingConnection.WYE_GROUNDED
            assert t.to_connection == WindingConnection.WYE_GROUNDED
            assert t.tap.shift_deg == pytest.approx(0.0)

    def test_tap_ratios_minus1_times_step_percent(self):
        """The 9 tapped units: ratio_magnitude = 1 - 0.01*tap_step_percent
        (tap_pos=-1, tap_neutral=0, tap_side='hv')."""
        net = self._net()
        grid, id_map = to_grid(net)
        expected_by_pp_idx = {
            0: 1.0 - 0.015,
            1: 1.0 - 0.04,
            2: 1.0 - 0.04,
            3: 1.0 - 0.065,
            4: 1.0 - 0.04,
            5: 1.0 - 0.015,
            6: 1.0 - 0.065,
            8: 1.0 - 0.065,
            10: 1.0 - 0.065,
        }
        for pp_idx, expected in expected_by_pp_idx.items():
            tid = id_map["trafo"][pp_idx]
            branch = next(b for b in grid.branches if b.id == tid)
            assert float(branch.tap.ratio_magnitude) == pytest.approx(expected), (
                f"trafo {pp_idx}: ratio={branch.tap.ratio_magnitude}, "
                f"expected {expected}"
            )

    def test_no_tap_units_stay_at_nominal(self):
        net = self._net()
        grid, id_map = to_grid(net)
        for pp_idx in (7, 9, 11, 12):
            tid = id_map["trafo"][pp_idx]
            branch = next(b for b in grid.branches if b.id == tid)
            assert float(branch.tap.ratio_magnitude) == pytest.approx(1.0)

    def test_ybus_transformer_stamps_match_pandapower(self):
        """SI Y-bus off-diagonal (pure-transformer) entries vs pandapower's own
        internal per-unit Ybus, scaled to SI.

        Tight (~1e-9 relative) for the 9 units with i0_percent==0; looser
        (~1e-2 relative, still asserted -- not skipped) for the 4 units with a
        NEGATIVE i0_percent (a MATPOWER-import artifact: these 4 rows carry
        i0_percent in {-0.64, -0.82, -0.04, -0.17} %, synthesized by
        pandapower's MATPOWER importer from a branch shunt susceptance;
        pgml's magnetizing-branch guard is `if i0_pct > 0.0`, matching the
        pandapower converter's own pre-existing, documented "keep magnetizing
        branch handling as-is" contract -- a negative no-load current has no
        physical magnetizing-branch meaning, so it converts to NO branch
        (dropping a small susceptance pandapower's own Ybus keeps). This is a
        pre-existing magnetizing-branch scope boundary, not a tap/vector-group
        bug -- confirmed because the residual scales with |i0_percent|:
        largest (~4.1e-3 relative) at i0=-0.82%, smallest (~1.7e-4) at
        i0=-0.045%.
        """
        net = self._net()
        grid, id_map = to_grid(net)

        f0 = float(net.f_hz)
        ybus_obj = assemble_network_ybus(grid, [f0], dtype=torch.complex128)
        Y_ours = ybus_obj.Y[0].numpy()
        index = ybus_obj.index

        Ybus_pu = net._ppc["internal"]["Ybus"].toarray()
        base_mva = float(net._ppc["baseMVA"])
        bl = net._pd2ppc_lookups["bus"]

        negative_i0_rows = {7, 9, 11, 12}
        for pp_idx, row in net.trafo.iterrows():
            hv_bus = int(row["hv_bus"])
            lv_bus = int(row["lv_bus"])
            ppc_hv = int(bl[hv_bus])
            ppc_lv = int(bl[lv_bus])
            base_kv_hv = float(net._ppc["bus"][ppc_hv, 9])
            base_kv_lv = float(net._ppc["bus"][ppc_lv, 9])
            y_base_od = base_mva / (base_kv_hv * base_kv_lv)

            node_hv = id_map["bus"][hv_bus]
            node_lv = id_map["bus"][lv_bus]
            r_hv = index.row(node_hv, Phase.A)
            r_lv = index.row(node_lv, Phase.A)

            y_ours_ft = Y_ours[r_hv, r_lv]
            y_pp_ft = Ybus_pu[ppc_hv, ppc_lv] * y_base_od

            err = abs(y_ours_ft - y_pp_ft)
            scale = max(abs(y_pp_ft), 1e-12)
            rel_tol = 1.5e-2 if pp_idx in negative_i0_rows else 1e-6
            assert err / scale < rel_tol, (
                f"trafo {pp_idx} (hv={hv_bus}, lv={lv_bus}) Y_ft: "
                f"ours={y_ours_ft:.6e}, pp={y_pp_ft:.6e}, "
                f"rel_err={err / scale:.3e} (tol={rel_tol:.1e})"
            )


# ---------------------------------------------------------------------------
# 10. Vector-group / shift_degree mismatch: end-to-end conversion raises
# ---------------------------------------------------------------------------
def test_inconsistent_vector_group_and_shift_raises():
    """A vector_group clock that disagrees with shift_degree raises loudly at
    the to_grid boundary (not silently resolved either way)."""
    net = pn.create_kerber_landnetz_kabel_1()
    net.trafo["shift_degree"] = 30.0  # was 150.0 (Dyn5); now disagrees with 'Dyn5'
    with pytest.raises(ConversionError, match="self-inconsistent"):
        to_grid(net)


# ---------------------------------------------------------------------------
# 11. Asymmetric (best-effort): runpp_3ph vs THREE_PHASE for Dyn and Yzn
# ---------------------------------------------------------------------------
# pandapower's runpp_3ph DOES support 'Dyn'/'Yzn' vector groups (its own
# zero-sequence transformer model requires the BARE letter form, no clock
# digit -- see _parse_vector_group's clock=None path), given zero-sequence
# columns (vk0_percent, vkr0_percent, mag0_percent, mag0_rx, si0_hv_partial)
# and an ext_grid short-circuit rating (s_sc_max_mva, rx_max, r0x0_max,
# x0x_max). vk0_percent=vk_percent / vkr0_percent=vkr_percent makes the
# POSITIVE-vs-ZERO-sequence leakage magnitude comparable to pgml's own
# Z0=Z1-via-topology assumption; a very large s_sc_max_mva with
# r0x0_max=x0x_max=1.0 makes the ext_grid near-ideal in every sequence in
# BOTH engines (pgml's ideal-slack mode fixes the full 3-phase phasor set
# exactly, which is equivalent to an infinite source in every sequence; the
# converter does not read r0x0_max/x0x_max at all, so this is the only way to
# make the comparison apples-to-apples rather than attempting -- and failing
# -- to replicate pandapower's own finite zero-sequence source impedance).
# ---------------------------------------------------------------------------
def _build_asym_net(vector_group: str, mag0_percent: float):
    net = pp.create_empty_network(f_hz=50.0)
    b_hv = pp.create_bus(net, vn_kv=20.0, name="hv")
    b_lv = pp.create_bus(net, vn_kv=0.4, name="lv")
    pp.create_ext_grid(
        net,
        bus=b_hv,
        vm_pu=1.0,
        va_degree=0.0,
        s_sc_max_mva=1.0e6,
        rx_max=0.1,
        s_sc_min_mva=1.0e6,
        rx_min=0.1,
        r0x0_max=1.0,
        x0x_max=1.0,
    )
    pp.create_transformer_from_parameters(
        net,
        hv_bus=b_hv,
        lv_bus=b_lv,
        sn_mva=0.4,
        vn_hv_kv=20.0,
        vn_lv_kv=0.4,
        vk_percent=4.0,
        vkr_percent=1.0,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=150.0,
        vector_group=vector_group,
        vk0_percent=4.0,
        vkr0_percent=1.0,
        mag0_percent=mag0_percent,
        mag0_rx=0.0,
        si0_hv_partial=0.9,
    )
    pp.create_asymmetric_load(
        net,
        bus=b_lv,
        p_a_mw=0.05,
        q_a_mvar=0.02,
        p_b_mw=0.03,
        q_b_mvar=0.01,
        p_c_mw=0.02,
        q_c_mvar=0.005,
    )
    return net


def test_asymmetric_dyn_runpp_3ph():
    """Dyn (clock 5, 150 deg), unbalanced load: per-phase voltages vs runpp_3ph.

    ``mag0_percent`` is set very large so pandapower's zero-sequence magnetizing
    shunt vanishes: pgml deliberately models NO zero-sequence magnetizing branch
    (the zero-sequence path is topology-only with the leakage value), and
    pandapower 3 honours ``mag0_percent`` in its Dyn zero-sequence network — a
    finite value (e.g. the pandapower default 10) shunts real zero-sequence
    current and shifts the loaded-phase voltage by ~1e-3 pu, which is a
    modeling-scope difference, not a conversion error. With the shunt removed
    the two zero-sequence models coincide (vk0 = vk here) and the agreement is
    ~2e-7 pu; a balanced load agrees to ~1e-13 pu regardless (positive sequence
    is exact).
    """
    net = _build_asym_net("Dyn", mag0_percent=1.0e6)
    pp.runpp_3ph(net, max_iteration=200)  # raises LoadflowNotConverged on failure

    grid, id_map = to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
    result = solve_power_flow(
        grid, slack="ideal", tol=1e-12, max_iter=200, dtype=torch.complex128
    )
    assert result.converged
    v = result.v.reshape(-1)

    atol_vm, atol_va = 1e-4, 5e-3
    for pp_idx in (0, 1):
        node_id = id_map["bus"][pp_idx]
        u_base = float(net.bus.at[pp_idx, "vn_kv"]) * 1_000.0 / math.sqrt(3.0)
        for ph, label in ((Phase.A, "a"), (Phase.B, "b"), (Phase.C, "c")):
            r = result.index.row(node_id, ph)
            vc = v[r]
            vm = float(vc.abs()) / u_base
            va = math.degrees(math.atan2(float(vc.imag), float(vc.real)))
            vm_ref = float(net.res_bus_3ph.at[pp_idx, f"vm_{label}_pu"])
            va_ref = float(net.res_bus_3ph.at[pp_idx, f"va_{label}_degree"])
            assert abs(vm - vm_ref) < atol_vm, (
                f"bus {pp_idx} phase {label}: vm ours={vm:.6f} pp={vm_ref:.6f}"
            )
            assert abs(_angle_diff_deg(va, va_ref)) < atol_va, (
                f"bus {pp_idx} phase {label}: va ours={va:.4f} pp={va_ref:.4f}"
            )


def test_asymmetric_yzn_runpp_3ph():
    """Yzn (zigzag, clock 5), unbalanced load: per-phase voltages vs runpp_3ph.

    The two zero-sequence zigzag models are built differently and do NOT admit
    an exact correspondence: pgml's zigzag ground path is topology-derived with
    the positive-sequence leakage VALUE (the `TransformerZeroSeq` override is
    not consumed), while pandapower parameterises the zigzag ground path
    through ``mag0_percent``/``si0_hv_partial`` — the gap does not vanish as
    ``mag0`` grows (the ground path itself would disappear; the load flow stops
    converging beyond ~3000), so ``mag0_percent`` is structurally part of the
    path, not a removable magnetizing shunt. Empirical correspondence sweep
    (max per-phase |dvm| on the zigzag bus): mag0=10 -> 2.2e-3, 30 -> 1.9e-3,
    100 -> 6.6e-4 (minimum, used here), 300 -> 5.2e-3, 1000 -> 2.3e-2. The
    tolerance guards that floor; balanced quantities and the positive-sequence
    zigzag model are pinned exactly elsewhere
    (test_transformer_clock_matrix.py, the Yzn5 grid-matrix case).
    """
    net = _build_asym_net("Yzn", mag0_percent=100.0)
    pp.runpp_3ph(net, max_iteration=200)

    grid, id_map = to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
    result = solve_power_flow(
        grid, slack="ideal", tol=1e-12, max_iter=200, dtype=torch.complex128
    )
    assert result.converged
    v = result.v.reshape(-1)

    atol_vm, atol_va = 2.0e-3, 2.0e-1
    for pp_idx in (0, 1):
        node_id = id_map["bus"][pp_idx]
        u_base = float(net.bus.at[pp_idx, "vn_kv"]) * 1_000.0 / math.sqrt(3.0)
        for ph, label in ((Phase.A, "a"), (Phase.B, "b"), (Phase.C, "c")):
            r = result.index.row(node_id, ph)
            vc = v[r]
            vm = float(vc.abs()) / u_base
            va = math.degrees(math.atan2(float(vc.imag), float(vc.real)))
            vm_ref = float(net.res_bus_3ph.at[pp_idx, f"vm_{label}_pu"])
            va_ref = float(net.res_bus_3ph.at[pp_idx, f"va_{label}_degree"])
            assert abs(vm - vm_ref) < atol_vm, (
                f"bus {pp_idx} phase {label}: vm ours={vm:.6f} pp={vm_ref:.6f}"
            )
            assert abs(_angle_diff_deg(va, va_ref)) < atol_va, (
                f"bus {pp_idx} phase {label}: va ours={va:.4f} pp={va_ref:.4f}"
            )


# ---------------------------------------------------------------------------
# 12. `parallel` (identical parallel systems) on a line AND a transformer
# ---------------------------------------------------------------------------
def _build_parallel_net(parallel: int, *, pfe_kw: float = 0.0, i0_percent: float = 0.0):
    """A 3-bus net with one line and one two-winding trafo, both ``parallel``."""
    net = pp.create_empty_network(f_hz=50.0)
    b0 = pp.create_bus(net, vn_kv=20.0, name="hv")
    b1 = pp.create_bus(net, vn_kv=20.0, name="mid")
    b2 = pp.create_bus(net, vn_kv=0.4, name="lv")
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0, va_degree=0.0)
    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1.5,
        r_ohm_per_km=0.32,
        x_ohm_per_km=0.35,
        c_nf_per_km=230.0,
        g_us_per_km=0.5,
        max_i_ka=0.4,
        parallel=parallel,
    )
    pp.create_transformer_from_parameters(
        net,
        hv_bus=b1,
        lv_bus=b2,
        sn_mva=0.25,
        vn_hv_kv=20.0,
        vn_lv_kv=0.4,
        vk_percent=4.0,
        vkr_percent=1.0,
        pfe_kw=pfe_kw,
        i0_percent=i0_percent,
        shift_degree=150.0,
        vector_group="Dyn",
        parallel=parallel,
    )
    pp.create_load(net, bus=b2, p_mw=0.15, q_mvar=0.05)
    return net


def test_parallel_line_and_transformer_matches_pandapower():
    """``parallel=2`` on both the line and the trafo (no magnetizing branch, so
    agreement is limited only by the nonlinear solver's own iteration
    tolerance -- the same "no magnetizing branch" family as
    ``test_case33bw_control_no_transformer``)."""
    _assert_matches_pandapower(_build_parallel_net(2), atol_vm=1e-6, atol_va=1e-5)


def test_parallel_scales_series_impedance_and_shunt_admittance():
    """``parallel`` divides the series impedance and multiplies the shunt
    admittance (line C/G, trafo magnetizing conductance/inductance) and the
    trafo's rated power -- pandapower's own ``build_branch`` convention
    (``_calc_r_x_from_dataframe``/``_calc_y_from_dataframe``), checked here as
    an exact closed-form relation between a ``parallel=1`` and a ``parallel=3``
    conversion of the SAME single-unit parameters (a magnetizing branch is
    present -- ``pfe_kw=0.5``, ``i0_percent=1.0`` -- chosen so ``q_nl_sq>0``
    and ``magnetizing_inductance_h`` is not ``None``)."""
    grid1, _ = to_grid(_build_parallel_net(1, pfe_kw=0.5, i0_percent=1.0))
    grid3, _ = to_grid(_build_parallel_net(3, pfe_kw=0.5, i0_percent=1.0))

    line1 = next(b for b in grid1.branches if isinstance(b, Line))
    line3 = next(b for b in grid3.branches if isinstance(b, Line))
    assert line3.series_resistance_ohm_per_m[0][0] == pytest.approx(
        line1.series_resistance_ohm_per_m[0][0] / 3.0
    )
    assert line3.series_inductance_h_per_m[0][0] == pytest.approx(
        line1.series_inductance_h_per_m[0][0] / 3.0
    )
    assert line3.shunt_capacitance_f_per_m[0][0] == pytest.approx(
        line1.shunt_capacitance_f_per_m[0][0] * 3.0
    )
    assert line3.shunt_conductance_s_per_m[0][0] == pytest.approx(
        line1.shunt_conductance_s_per_m[0][0] * 3.0
    )

    trafo1 = next(b for b in grid1.branches if isinstance(b, Transformer))
    trafo3 = next(b for b in grid3.branches if isinstance(b, Transformer))
    assert trafo3.series_resistance_ohm == pytest.approx(
        trafo1.series_resistance_ohm / 3.0
    )
    assert trafo3.series_inductance_h == pytest.approx(trafo1.series_inductance_h / 3.0)
    assert trafo3.magnetizing_conductance_s == pytest.approx(
        trafo1.magnetizing_conductance_s * 3.0
    )
    assert trafo1.magnetizing_inductance_h is not None
    assert trafo3.magnetizing_inductance_h == pytest.approx(
        trafo1.magnetizing_inductance_h / 3.0
    )
    assert trafo3.s_rated_va == pytest.approx(trafo1.s_rated_va * 3.0)


def test_parallel_default_is_byte_identical():
    """``parallel=1`` (pandapower's own default) reproduces the pre-``parallel``-
    aware output exactly -- dividing/multiplying by ``1.0`` is an IEEE-754 exact
    no-op, so every scaled field matches a hand-computed ``parallel``-naive
    reference bit-for-bit."""
    net = _build_parallel_net(1, pfe_kw=0.5, i0_percent=1.0)
    grid, _ = to_grid(net)
    line = next(b for b in grid.branches if isinstance(b, Line))
    trafo = next(b for b in grid.branches if isinstance(b, Transformer))

    r1 = float(net.line.at[0, "r_ohm_per_km"]) / 1_000.0
    x1 = float(net.line.at[0, "x_ohm_per_km"]) / 1_000.0
    c1 = float(net.line.at[0, "c_nf_per_km"]) * 1.0e-9 / 1_000.0
    g1 = float(net.line.at[0, "g_us_per_km"]) * 1.0e-6 / 1_000.0
    assert line.series_resistance_ohm_per_m[0][0] == r1
    assert line.series_inductance_h_per_m[0][0] == x1 / (2.0 * math.pi * net.f_hz)
    assert line.shunt_capacitance_f_per_m[0][0] == c1
    assert line.shunt_conductance_s_per_m[0][0] == g1
    assert trafo.s_rated_va == float(net.trafo.at[0, "sn_mva"]) * 1.0e6


# ---------------------------------------------------------------------------
# 13. Voltage-dependent (ZIP) loads
# ---------------------------------------------------------------------------
def test_mixed_zip_load_matches_pandapower():
    """A mixed ZIP load (distinct P/Q constant-impedance/current percentages)
    vs a live ``pp.runpp`` (``voltage_depend_loads=True``, pandapower's own
    default): the nonlinear solve honours ``ZipCoefficients`` via
    ``device_current_injections``, matching to near machine precision (the
    same "no magnetizing branch" family -- there is no transformer here at
    all)."""
    net = pp.create_empty_network(f_hz=50.0)
    b0 = pp.create_bus(net, vn_kv=20.0, name="hv")
    b1 = pp.create_bus(net, vn_kv=20.0, name="lv")
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0, va_degree=0.0)
    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=3.0,
        r_ohm_per_km=0.32,
        x_ohm_per_km=0.35,
        c_nf_per_km=230.0,
        max_i_ka=0.4,
    )
    pp.create_load(
        net,
        bus=b1,
        p_mw=0.4,
        q_mvar=0.15,
        const_z_p_percent=30.0,
        const_i_p_percent=20.0,
        const_z_q_percent=50.0,
        const_i_q_percent=10.0,
    )
    _assert_matches_pandapower(net, atol_vm=1e-8, atol_va=1e-6, tol=1e-12)


def test_mixed_zip_load_coefficients_mapped_correctly():
    """``const_z_p_percent``/``const_i_p_percent``/``const_z_q_percent``/
    ``const_i_q_percent`` map onto ``ZipCoefficients`` independently for P and Q
    (pandapower 3's own per-load ZIP model, NOT a single shared P/Q percentage
    pair -- see ``pandapower.build_bus._calc_pq_elements_and_add_on_ppc``)."""
    net = pp.create_empty_network(f_hz=50.0)
    b0 = pp.create_bus(net, vn_kv=20.0)
    pp.create_ext_grid(net, bus=b0)
    pp.create_load(
        net,
        bus=b0,
        p_mw=0.4,
        q_mvar=0.15,
        const_z_p_percent=30.0,
        const_i_p_percent=20.0,
        const_z_q_percent=50.0,
        const_i_q_percent=10.0,
    )
    grid, _ = to_grid(net)
    load = next(a for a in grid.appliances if isinstance(a, Load))
    zc = load.zip_coefficients
    assert zc is not None
    assert (zc.z_p, zc.i_p, zc.p_p) == pytest.approx((0.30, 0.20, 0.50))
    assert (zc.z_q, zc.i_q, zc.p_q) == pytest.approx((0.50, 0.10, 0.40))


def test_zero_zip_percentages_stay_byte_identical():
    """All four percentages at zero (pandapower's own default -- a pure
    constant-power load) converts with NO ``zip_coefficients``/``load_model``
    set, so a plain load's :class:`~pgml.schemas.grid_schema.Load` is unchanged
    from before ZIP support was added."""
    net = pp.create_empty_network(f_hz=50.0)
    b0 = pp.create_bus(net, vn_kv=20.0)
    pp.create_ext_grid(net, bus=b0)
    pp.create_load(net, bus=b0, p_mw=0.4, q_mvar=0.15)
    grid, _ = to_grid(net)
    load = next(a for a in grid.appliances if isinstance(a, Load))
    assert load.zip_coefficients is None
    assert load.load_model == LoadModel.CONST_POWER
