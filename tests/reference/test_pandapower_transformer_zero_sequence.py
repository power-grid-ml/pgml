"""Oracle test: transformer zero-sequence leakage vs pandapower ``runpp_3ph``.

pandapower is the only one of the three reference libraries with a zero-sequence
LEAKAGE input for a two-winding transformer (``vk0_percent`` / ``vkr0_percent``;
OpenDSS and power-grid-model both derive the zero sequence from the winding topology
with the positive-sequence value). Its unbalanced power flow uses that value through
``pandapower.pd2ppc_zero._add_trafo_sc_impedance_zero``, which places it per vector
group: a series transfer impedance plus terminal shunts for ``YNyn``, a pure LV shunt
for ``Dyn``, nothing at all for ``Yy``/``Yd``/``Dy``/``Dd``.

pgml reaches the same model from the other direction: one per-phase leakage MATRIX
(the symmetric-component split of Z1 and Z0) carried through the winding-incidence
transform ``Nᵀ Y_winding N``, so the TOPOLOGY comes from the connections and the VALUE
from ``Transformer.zero_sequence``. The two agree when pandapower's extra
zero-sequence refinements are switched off, which this test does deliberately:

- ``mag0_percent`` is set very large, so pandapower's zero-sequence MAGNETIZING branch
  (the three-limb-core path through tank and air, which pgml does not model) is open.
  With it open, pandapower's T model collapses to the series leakage ``z0_k`` and the
  ``si0_hv_partial`` split has no effect.
- ``pfe_kw = i0_percent = 0``, so the positive-sequence magnetizing branch (which pgml
  places on the external HV terminal instead of inside the leakage T) cannot
  contribute either.
- the ext_grid is very stiff (``s_sc_max_mva = 1e7``), so its own zero-sequence shunt
  is negligible and pgml's ``slack="ideal"`` matches pandapower's slack treatment.

Measured (this environment, float64/complex128, CPU, pandapower 3.5.4) on a 20/0.4 kV
400 kVA unit with ``vk = 4 %``, ``vkr = 1 %`` and an 80 kW single-phase LV load:

============  ===================  ===============  =====================
vector group  vk0/vk               max |dV| (Z0)    max |dV| (Z0 := Z1)
============  ===================  ===============  =====================
YNyn          0.3                  2.0e-4 V         1.31 V (5.7e-3 rel)
YNyn          0.5                  2.0e-4 V         0.93 V (4.1e-3 rel)
YNyn          2.0                  2.0e-4 V         1.88 V (8.2e-3 rel)
Dyn           0.5                  1.0e-4 V         0.93 V (4.1e-3 rel)
Dyn           2.0                  1.0e-4 V         1.88 V (8.2e-3 rel)
============  ===================  ===============  =====================

Tolerance: ``atol = 2e-3 V`` (10x the observed residual, which is pandapower's own
outer-loop convergence tolerance of 3e-8 MVA, and 400x below the Z0 = Z1 error).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pandapower = pytest.importorskip("pandapower")
pp = pandapower

from pgml.convert.pandapower import PhaseMode, to_grid  # noqa: E402
from pgml.schemas.grid_schema import Phase, Transformer  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402

_F0 = 50.0
_ABC = (Phase.A, Phase.B, Phase.C)
_KV_HV, _KV_LV = 20.0, 0.4
_SN_MVA = 0.4
_VK, _VKR = 4.0, 1.0
_LOAD_MW = 0.08  # single phase only -> a strong residual (zero-sequence) current
_ATOL_V = 2.0e-3


def _build_net(vector_group: str, vk0_factor: float):
    """HV ext_grid -> two-winding transformer -> single-phase LV load."""
    net = pp.create_empty_network(sn_mva=1.0, f_hz=_F0)
    hv = pp.create_bus(net, vn_kv=_KV_HV, name="hv")
    lv = pp.create_bus(net, vn_kv=_KV_LV, name="lv")
    # A very stiff feed: the ext_grid's own zero-sequence shunt is ~0, so both engines
    # see an effectively rigid HV boundary and the transformer dominates.
    pp.create_ext_grid(
        net,
        hv,
        vm_pu=1.0,
        va_degree=0.0,
        s_sc_max_mva=1.0e7,
        rx_max=0.1,
        x0x_max=1.0,
        r0x0_max=0.1,
    )
    pp.create_transformer_from_parameters(
        net,
        hv,
        lv,
        sn_mva=_SN_MVA,
        vn_hv_kv=_KV_HV,
        vn_lv_kv=_KV_LV,
        vk_percent=_VK,
        vkr_percent=_VKR,
        pfe_kw=0.0,
        i0_percent=0.0,
        # runpp_3ph requires the clock-less vector-group string; the clock itself
        # travels in shift_degree (Dyn11 = -30 deg).
        shift_degree=-30.0 if vector_group.lower() == "dyn" else 0.0,
        vector_group=vector_group,
        vk0_percent=_VK * vk0_factor,
        vkr0_percent=_VKR * vk0_factor,
        mag0_percent=1.0e8,  # open the zero-sequence magnetizing branch
        mag0_rx=0.0,
        si0_hv_partial=0.9,  # no effect once the magnetizing branch is open
        parallel=1,
    )
    pp.create_asymmetric_load(
        net,
        lv,
        p_a_mw=_LOAD_MW,
        p_b_mw=0.0,
        p_c_mw=0.0,
        q_a_mvar=0.0,
        q_b_mvar=0.0,
        q_c_mvar=0.0,
        type="wye",
    )
    return net, hv, lv


def _reference_voltages(net, buses_kv) -> dict[int, list[complex]]:
    res = net.res_bus_3ph
    out = {}
    for bus, vn_kv in buses_kv.items():
        u_ln = vn_kv * 1.0e3 / math.sqrt(3.0)
        out[bus] = [
            res.at[bus, f"vm_{p}_pu"]
            * u_ln
            * np.exp(1j * np.deg2rad(res.at[bus, f"va_{p}_degree"]))
            for p in ("a", "b", "c")
        ]
    return out


def _max_deviation(grid, id_map, v_ref) -> float:
    res = solve_power_flow(
        grid, slack="ideal", tol=1e-13, max_iter=300, dtype=torch.complex128
    )
    assert res.converged
    worst = 0.0
    for pp_bus, node_id in id_map["bus"].items():
        for k, phase in enumerate(_ABC):
            v_pgml = complex(res.v[res.index.row(node_id, phase)])
            worst = max(worst, abs(v_pgml - v_ref[pp_bus][k]))
    return worst


class TestZeroSequenceLeakageConversion:
    """``vk0_percent``/``vkr0_percent`` -> ``Transformer.zero_sequence``."""

    def test_value_is_the_lv_coil_referred_impedance(self) -> None:
        net, _, _ = _build_net("YNyn", 0.5)
        grid, _ = to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
        xfmr = next(b for b in grid.branches if isinstance(b, Transformer))
        z_base_lv = (_KV_LV * 1.0e3) ** 2 / (_SN_MVA * 1.0e6)
        z0 = 0.5 * _VK / 100.0 * z_base_lv
        r0 = 0.5 * _VKR / 100.0 * z_base_lv
        assert xfmr.zero_sequence is not None
        assert float(xfmr.zero_sequence.r0_ohm) == pytest.approx(r0, rel=1e-12)
        assert float(xfmr.zero_sequence.x0_ohm) == pytest.approx(
            math.sqrt(z0**2 - r0**2), rel=1e-12
        )

    def test_absent_vk0_warns_once_with_the_count_and_ids(self, caplog) -> None:
        """The fallback is reported ONCE per conversion, naming every transformer."""
        net, _, _ = _build_net("YNyn", 0.5)
        net.trafo.loc[0, "vk0_percent"] = float("nan")
        # A second, identical unit: the warning must still be a single record.
        net.trafo.loc[1] = net.trafo.loc[0]
        with caplog.at_level("WARNING"):
            grid, _ = to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
        for branch in grid.branches:
            if isinstance(branch, Transformer):
                assert branch.zero_sequence is None
        notices = [r for r in caplog.records if "vk0_percent" in r.message]
        assert len(notices) == 1
        assert "2 pandapower trafo(s) 0, 1" in notices[0].message

    def test_present_vk0_does_not_warn(self, caplog) -> None:
        net, _, _ = _build_net("YNyn", 0.5)
        with caplog.at_level("WARNING"):
            to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
        assert not [r for r in caplog.records if "vk0_percent" in r.message]

    def test_unmodelled_zero_sequence_refinements_warn(self, caplog) -> None:
        """A finite zero-sequence magnetizing branch is named, not silently dropped."""
        net, _, _ = _build_net("YNyn", 0.5)
        net.trafo.loc[0, "mag0_percent"] = 100.0
        net.trafo.loc[0, "xn_ohm"] = 5.0
        with caplog.at_level("WARNING"):
            to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
        messages = " ".join(r.message for r in caplog.records)
        assert "mag0_percent" in messages
        assert "xn_ohm" in messages


class TestRunpp3phParity:
    """Full unbalanced solve vs ``runpp_3ph``, which uses ``vk0_percent`` itself."""

    @pytest.mark.parametrize(
        ("vector_group", "vk0_factor"),
        [("YNyn", 0.3), ("YNyn", 0.5), ("YNyn", 2.0), ("Dyn", 0.5), ("Dyn", 2.0)],
    )
    def test_unbalanced_voltages_match(self, vector_group, vk0_factor) -> None:
        net, hv, lv = _build_net(vector_group, vk0_factor)
        pp.runpp_3ph(net)
        v_ref = _reference_voltages(net, {hv: _KV_HV, lv: _KV_LV})

        grid, id_map = to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
        xfmr = next(b for b in grid.branches if isinstance(b, Transformer))
        assert xfmr.zero_sequence is not None
        dev = _max_deviation(grid, id_map, v_ref)
        assert dev < _ATOL_V, f"{vector_group} vk0/vk={vk0_factor}: {dev:.3e} V"

    @pytest.mark.parametrize(
        ("vector_group", "vk0_factor"), [("YNyn", 0.3), ("Dyn", 2.0)]
    )
    def test_dropping_the_value_is_visible(self, vector_group, vk0_factor) -> None:
        """Without the zero-sequence value the solve is 100x the tolerance off."""
        net, hv, lv = _build_net(vector_group, vk0_factor)
        pp.runpp_3ph(net)
        v_ref = _reference_voltages(net, {hv: _KV_HV, lv: _KV_LV})

        grid, id_map = to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
        for branch in grid.branches:
            if isinstance(branch, Transformer):
                branch.zero_sequence = None
        dev = _max_deviation(grid, id_map, v_ref)
        assert dev > 100.0 * _ATOL_V, f"{dev:.3e} V"
