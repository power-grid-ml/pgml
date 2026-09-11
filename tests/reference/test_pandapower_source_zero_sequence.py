"""Oracle test: ext_grid zero-sequence impedance vs pandapower's own unbalanced model.

pandapower models an external grid in its unbalanced power flow (``runpp_3ph``) as

1. an IDEAL positive-sequence slack (``vm_pu``/``va_degree`` pinned; the
   positive-sequence impedance does NOT appear in the seq-1 network), and
2. a zero-sequence SHUNT admittance at the slack bus, built in
   ``pandapower.pd2ppc_zero._add_ext_grid_sc_impedance_zero`` from::

       X1 = (U_LL^2 / S_sc) / sqrt(1 + rx_max^2)
       X0 = x0x_max * X1        R0 = r0x0_max * X0
       Z0_pandapower = c * (R0 + j*X0),  c = 1.1 even in power-flow mode

pgml's pandapower converter mirrors that model: the positive-sequence Thevenin stays
near-ideal (1e-6 Ohm, the same ideal slack pandapower's own power flow uses) and the
ZERO-sequence impedance is carried as an absolute value derived from the four
short-circuit columns, without pandapower's IEC voltage factor ``c`` (pgml stores the
physical impedance). The two therefore differ by exactly 1.1 in the zero sequence,
which this file asserts explicitly rather than hiding.

What is validated
-----------------
``TestZeroSequenceModel`` compares the converted per-phase Thevenin matrix's
zero-sequence eigenvalue against pandapower's own internal zero-sequence shunt
(read from ``net._ppc0``'s ``GS``/``BS`` and converted to Ohm on pandapower's
``baseR = U_LL^2 / (3 * sn_mva)`` per-phase base) to 1e-12 relative. That is an exact
model comparison, independent of any solve.

``TestRunpp3phParity`` solves a 2-bus unbalanced case in both engines and compares
per-phase voltages. Residual model differences, both quantified in the assertions:

- pandapower's ``c = 1.1`` (compensated through ``param_overrides`` scaling the
  source matrix, so the comparison isolates the sequence split).
- pandapower's negative-sequence ext-grid shunt equals its POSITIVE-sequence
  short-circuit impedance while its positive sequence is an ideal slack; pgml's source
  is one physical Thevenin with ``Z2 = Z1``. The case therefore uses a deliberately
  stiff ``s_sc_max_mva`` (``Z1 = 1.6e-4`` Ohm against a 0.1 Ohm line) so that
  difference stays at the 1e-6 pu level, and a large ``x0x_max`` so the zero-sequence
  impedance is still comparable to the line's.

Measured (this environment, float64/complex128, CPU, pandapower 3.5.4):

- zero-sequence value: exact to 1.1e-16 relative.
- full solve, ``c`` compensated: max ``|dV|`` = 1.2e-3 V = 5.4e-6 pu.
- full solve, ``c`` NOT compensated: 2.3e-1 V = 9.8e-4 pu (the ``c`` factor alone).
- pre-fix model (zero sequence forced to the near-ideal positive-sequence value):
  2.5 V = 1.1e-2 pu.

Tolerance: ``atol = 5e-3 V`` on the compensated solve (4x the measured residual, 500x
below the pre-fix error).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pandapower = pytest.importorskip("pandapower")
pp = pandapower

from pandapower.pypower.idx_bus import BS, GS  # noqa: E402

from pgml.convert.pandapower import PhaseMode, to_grid  # noqa: E402
from pgml.schemas.grid_schema import Phase, Source  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402

_U_KV = 0.4
_F0 = 50.0
_C_FACTOR = 1.1  # pandapower's IEC voltage factor, applied even in power-flow mode
_ABC = (Phase.A, Phase.B, Phase.C)

# Stiff positive sequence + large X0/X1 so the zero sequence dominates the residual
# (see the module docstring).
_S_SC_MVA = 1000.0
_RX_MAX = 0.1
_X0X_MAX = 2000.0
_R0X0_MAX = 0.25


def _build_net(
    *,
    s_sc_mva: float = _S_SC_MVA,
    rx_max: float = _RX_MAX,
    x0x_max: float = _X0X_MAX,
    r0x0_max: float = _R0X0_MAX,
):
    """2-bus LV net: ext_grid -> 250 m line -> unbalanced wye load."""
    net = pp.create_empty_network(sn_mva=1.0, f_hz=_F0)
    b1 = pp.create_bus(net, vn_kv=_U_KV, name="slack")
    b2 = pp.create_bus(net, vn_kv=_U_KV, name="load")
    pp.create_ext_grid(
        net,
        b1,
        vm_pu=1.0,
        va_degree=0.0,
        s_sc_max_mva=s_sc_mva,
        rx_max=rx_max,
        x0x_max=x0x_max,
        r0x0_max=r0x0_max,
    )
    pp.create_line_from_parameters(
        net,
        b1,
        b2,
        length_km=0.25,
        r_ohm_per_km=0.4,
        x_ohm_per_km=0.2,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        r0_ohm_per_km=1.2,
        x0_ohm_per_km=0.6,
        c0_nf_per_km=0.0,
    )
    pp.create_asymmetric_load(
        net,
        b2,
        p_a_mw=0.006,
        p_b_mw=0.002,
        p_c_mw=0.001,
        q_a_mvar=0.0,
        q_b_mvar=0.0,
        q_c_mvar=0.0,
        type="wye",
    )
    return net, b1, b2


def _sequence_impedances(src: Source, f0: float) -> tuple[complex, complex]:
    """``(Z1, Z0)`` eigenvalues of a 3-phase source's Thevenin matrix [Ohm]."""
    w0 = 2.0 * math.pi * f0
    z = np.array(
        [
            [
                complex(
                    float(src.resistance_ohm[i][j]), w0 * float(src.inductance_h[i][j])
                )
                for j in range(3)
            ]
            for i in range(3)
        ]
    )
    ones = np.ones(3)
    a = np.exp(2j * np.pi / 3.0)
    pos = np.array([1.0, a**2, a])
    return complex(np.conj(pos) @ z @ pos / 3.0), complex(ones @ z @ ones / 3.0)


def _pandapower_zero_sequence_ohm(net, bus: int) -> complex:
    """pandapower's internal zero-sequence ext_grid impedance at ``bus`` [Ohm]."""
    ppc0 = net._ppc0
    y0_pu = complex(ppc0["bus"][bus, GS], ppc0["bus"][bus, BS]) / ppc0["baseMVA"]
    # pf_3ph per-phase base: baseR = U_LL^2 / (3 * sn_mva) (the same base pandapower
    # uses for the zero-sequence line impedance in this mode).
    base_r = (_U_KV * 1.0e3) ** 2 / (3.0 * net.sn_mva * 1.0e6)
    return (1.0 / y0_pu) * base_r


class TestZeroSequenceModel:
    """Exact model comparison: converted Z0 vs pandapower's own zero-sequence shunt."""

    @pytest.mark.parametrize(
        ("s_sc_mva", "rx_max", "x0x_max", "r0x0_max"),
        [(2.0, 0.3, 3.0, 0.3), (50.0, 0.1, 1.0, 0.1), (_S_SC_MVA, _RX_MAX, 10.0, 0.25)],
    )
    def test_converted_zero_sequence_is_pandapower_value_over_c(
        self, s_sc_mva, rx_max, x0x_max, r0x0_max
    ) -> None:
        net, b1, _ = _build_net(
            s_sc_mva=s_sc_mva, rx_max=rx_max, x0x_max=x0x_max, r0x0_max=r0x0_max
        )
        pp.add_zero_impedance_parameters(net)
        pp.runpp_3ph(net)
        z0_pandapower = _pandapower_zero_sequence_ohm(net, b1)

        grid, _ = to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
        src = next(a for a in grid.appliances if isinstance(a, Source))
        _, z0_pgml = _sequence_impedances(src, _F0)

        assert z0_pgml * _C_FACTOR == pytest.approx(z0_pandapower, rel=1e-12), (
            f"pgml Z0 {z0_pgml} vs pandapower {z0_pandapower} (c = {_C_FACTOR})"
        )

    def test_positive_sequence_stays_near_ideal(self) -> None:
        """The converter keeps pandapower's ideal positive-sequence slack behaviour."""
        net, _, _ = _build_net()
        grid, _ = to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
        src = next(a for a in grid.appliances if isinstance(a, Source))
        z1, z0 = _sequence_impedances(src, _F0)
        assert abs(z1) < 1e-5, (
            f"positive-sequence Thevenin should stay near-ideal: {z1}"
        )
        assert abs(z0) > 0.1, f"zero-sequence Thevenin should be finite: {z0}"

    def test_single_phase_equivalent_has_no_zero_sequence(self) -> None:
        """A positive-sequence-equivalent conversion keeps the 1x1 near-ideal stamp."""
        net, _, _ = _build_net()
        grid, _ = to_grid(net, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
        src = next(a for a in grid.appliances if isinstance(a, Source))
        assert src.phases == (Phase.A,)
        assert float(src.resistance_ohm[0][0]) < 1e-5

    def test_missing_short_circuit_data_warns_and_falls_back(self, caplog) -> None:
        """Zero-sequence ratios without s_sc data name the element in a WARNING."""
        net = pp.create_empty_network(sn_mva=1.0, f_hz=_F0)
        b1 = pp.create_bus(net, vn_kv=_U_KV)
        b2 = pp.create_bus(net, vn_kv=_U_KV)
        pp.create_ext_grid(net, b1, vm_pu=1.0, x0x_max=3.0, r0x0_max=0.3)
        pp.create_line_from_parameters(
            net,
            b1,
            b2,
            length_km=0.1,
            r_ohm_per_km=0.4,
            x_ohm_per_km=0.2,
            c_nf_per_km=0.0,
            max_i_ka=1.0,
        )
        pp.create_load(net, b2, p_mw=0.005)
        with caplog.at_level("WARNING"):
            to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
        assert any(
            "ext_grid 0" in r.message and "zero-sequence" in r.message
            for r in caplog.records
        )


class TestRunpp3phParity:
    """Full unbalanced solve vs ``runpp_3ph``."""

    ATOL_V = 5.0e-3

    @staticmethod
    def _reference_voltages(net, buses) -> dict[int, list[complex]]:
        u_ln = _U_KV * 1.0e3 / math.sqrt(3.0)
        res = net.res_bus_3ph
        return {
            bus: [
                res.at[bus, f"vm_{p}_pu"]
                * u_ln
                * np.exp(1j * np.deg2rad(res.at[bus, f"va_{p}_degree"]))
                for p in ("a", "b", "c")
            ]
            for bus in buses
        }

    @staticmethod
    def _max_deviation(res, id_map, v_ref) -> float:
        worst = 0.0
        for pp_bus, node_id in id_map["bus"].items():
            for k, phase in enumerate(_ABC):
                v_pgml = complex(res.v[res.index.row(node_id, phase)])
                worst = max(worst, abs(v_pgml - v_ref[pp_bus][k]))
        return worst

    def test_unbalanced_voltages_match(self) -> None:
        net, b1, b2 = _build_net()
        pp.add_zero_impedance_parameters(net)
        pp.runpp_3ph(net)
        v_ref = self._reference_voltages(net, (b1, b2))

        grid, id_map = to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
        src = next(a for a in grid.appliances if isinstance(a, Source))
        # Compensate pandapower's IEC voltage factor c = 1.1 (a uniform scaling of the
        # whole Thevenin matrix scales Z1 and Z0 alike; only Z0 matters here).
        overrides = {
            ("source", src.id, "resistance_ohm"): torch.tensor(
                src.resistance_ohm, dtype=torch.float64
            )
            * _C_FACTOR,
            ("source", src.id, "inductance_h"): torch.tensor(
                src.inductance_h, dtype=torch.float64
            )
            * _C_FACTOR,
        }
        res = solve_power_flow(
            grid,
            slack="norton",
            tol=1e-13,
            max_iter=300,
            dtype=torch.complex128,
            param_overrides=overrides,
        )
        assert res.converged
        dev = self._max_deviation(res, id_map, v_ref)
        assert dev < self.ATOL_V, f"max |dV| = {dev:.3e} V"

    def test_zero_sequence_assumption_error_is_visible(self) -> None:
        """Forcing Z0 to the near-ideal Z1 (the pre-fix model) is 100x the tolerance."""
        net, b1, b2 = _build_net()
        pp.add_zero_impedance_parameters(net)
        pp.runpp_3ph(net)
        v_ref = self._reference_voltages(net, (b1, b2))

        grid, id_map = to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
        for appliance in grid.appliances:
            if isinstance(appliance, Source):
                appliance.resistance_ohm = [
                    [1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)
                ]
                appliance.inductance_h = [
                    [1e-12 if i == j else 0.0 for j in range(3)] for i in range(3)
                ]
        res = solve_power_flow(
            grid, slack="norton", tol=1e-13, max_iter=300, dtype=torch.complex128
        )
        dev = self._max_deviation(res, id_map, v_ref)
        assert dev > 100.0 * self.ATOL_V, f"max |dV| = {dev:.3e} V"
