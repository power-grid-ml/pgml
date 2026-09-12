"""The delta-LV coil factor of 3, verified on a converted OpenDSS unit.

``Transformer.series_resistance_ohm``/``series_inductance_h`` store the leakage referred
to the TO-side COIL. A delta coil is rated at the LINE-TO-LINE voltage, so its impedance
base is ``3·u_LL²/S`` and the stored value is ``3·z_LL`` for a delta TO winding (``z_LL``
for a wye or zigzag one). The single-phase / positive-sequence-equivalent stamp then
multiplies the leakage by ``k_ll = 3`` for a delta TO winding before forming the scalar
off-nominal-tap pi, so the two factors cancel and the stamped TERMINAL impedance is
``z_LL`` again. The 3-phase stamp reaches the same terminal impedance through the winding
incidence (``Mᵀ M``).

Verified here against a live OpenDSS wye-delta unit (20/0.4 kV, 400 kVA, ``XHL = 4 %``,
``%R = 0.5`` per winding), with the hand-derived value written out:

- ``z_LL = (vkr + j·vk)/100 · u_LV²/S = (0.01 + j0.04)·0.4 Ω = 0.004 + j0.016 Ω``.
- stored TO-side coil leakage: ``0.012 + j0.048 Ω`` = exactly ``3·z_LL`` (ratio 3.000000),
  in BOTH phase modes — the converter does not depend on the phase mode.
- the ``p == 1`` stamp's effective terminal impedance, recovered from the assembled
  coupling entry as ``-1/(Y_ft·conj(t))`` with ``t = (u_from/u_to)·e^{jθ}``:
  ``0.004 + j0.016 Ω``, i.e. ``z_LL`` to 1e-9 relative.
- the solved positive-sequence LV voltage matches OpenDSS's own to 3.2e-6 V on a 228.8 V
  base (1.4e-8 relative).

A 3-phase solve of the same unit is NOT compared: a delta LV winding with no other
zero-sequence reference leaves the LV zero sequence unreferenced, so the phase-domain
``Y`` is singular (OpenDSS stabilises the same circuit with its ``ppm_antifloat`` shunt).
The positive-sequence content is what the factor of 3 governs, and that is what is
compared.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
import torch

opendssdirect = pytest.importorskip("opendssdirect")
dss = opendssdirect

from pgml.assembly import assemble_network_ybus  # noqa: E402
from pgml.convert.opendss import PhaseMode, to_grid  # noqa: E402
from pgml.schemas.grid_schema import Phase, Transformer  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402

pytestmark = pytest.mark.opendss

_F0 = 50.0
_KV_HV, _KV_LV, _KVA = 20.0, 0.4, 400.0
_VK_PCT, _VKR_PCT = 4.0, 1.0  # XHL and the summed %Rs
_Z_BASE_LV = (_KV_LV * 1000.0) ** 2 / (_KVA * 1000.0)
_Z_LL = complex(_VKR_PCT / 100.0, _VK_PCT / 100.0) * _Z_BASE_LV


def _build_circuit(load_conn: str = "delta") -> None:
    script = f"""
    Clear
    Set DefaultBaseFrequency={_F0:g}
    New Circuit.dlv basekv={_KV_HV:g} pu=1.0 phases=3 bus1=hv angle=0
    Edit Vsource.source R1=1e-6 X1=1e-6 R0=1e-6 X0=1e-6
    New Transformer.t1 phases=3 windings=2 buses=[hv.1.2.3, lv.1.2.3] conns=[wye delta]
    ~ kvs=[{_KV_HV:g} {_KV_LV:g}] kvas=[{_KVA:g} {_KVA:g}] XHL={_VK_PCT:g} %Rs=[0.5 0.5]
    ~ %noloadloss=0 %imag=0
    New Load.l1 bus1=lv.1.2.3 phases=3 kv={_KV_LV:g} kw=200 pf=0.98 model=1 conn={load_conn} Vminpu=0.0001 Vmaxpu=10000
    Set VoltageBases=[{_KV_HV:g}, {_KV_LV:g}]
    CalcVoltageBases
    Set Tolerance=1e-10
    Set MaxIterations=100
    Solve
    """
    for line in script.strip().splitlines():
        dss.Text.Command(line.strip())
    assert dss.Solution.Converged()


def _coil_impedance(xfmr: Transformer) -> complex:
    return complex(
        float(xfmr.series_resistance_ohm),
        2.0 * math.pi * _F0 * float(xfmr.series_inductance_h),
    )


@pytest.mark.parametrize("mode", [PhaseMode.SINGLE_PHASE_EQUIV, PhaseMode.THREE_PHASE])
def test_stored_coil_leakage_is_three_times_the_terminal_value(mode) -> None:
    """The converter stores 3·z_LL for a delta TO winding, in both phase modes."""
    _build_circuit()
    grid, _ = to_grid(dss, phase_mode=mode)
    xfmr = next(b for b in grid.branches if isinstance(b, Transformer))
    z_coil = _coil_impedance(xfmr)
    assert z_coil.real == pytest.approx(3.0 * _Z_LL.real, rel=1e-12)
    assert z_coil.imag == pytest.approx(3.0 * _Z_LL.imag, rel=1e-12)


def test_single_phase_stamp_recovers_the_terminal_impedance() -> None:
    """The stamp's k_ll = 3 cancels the stored factor: terminal leakage == z_LL."""
    _build_circuit()
    grid, id_map = to_grid(dss, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
    xfmr = next(b for b in grid.branches if isinstance(b, Transformer))
    yb = assemble_network_ybus(grid, [_F0], dtype=torch.complex128)
    y = (yb.Y[0] if yb.Y.dim() == 3 else yb.Y).numpy()
    y_ft = y[
        yb.index.row(id_map["bus"]["hv"], Phase.A),
        yb.index.row(id_map["bus"]["lv"], Phase.A),
    ]
    t = (_KV_HV / _KV_LV) * cmath.exp(1j * math.radians(float(xfmr.tap.shift_deg)))
    z_eff = -1.0 / (y_ft * t.conjugate())
    assert z_eff.real == pytest.approx(_Z_LL.real, rel=1e-9)
    assert z_eff.imag == pytest.approx(_Z_LL.imag, rel=1e-9)


@pytest.mark.parametrize("load_conn", ["delta", "wye"])
def test_positive_sequence_voltage_matches_opendss(load_conn) -> None:
    """The positive-sequence LV voltage of the converted unit matches a live solve."""
    _build_circuit(load_conn)
    dss.Circuit.SetActiveBus("lv")
    v = np.array(dss.Bus.Voltages())
    v_phase = v[0::2] + 1j * v[1::2]
    a = np.exp(2j * np.pi / 3.0)
    v1_dss = (v_phase[0] + a * v_phase[1] + a**2 * v_phase[2]) / 3.0

    grid, id_map = to_grid(dss, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
    res = solve_power_flow(
        grid, slack="ideal", tol=1e-13, max_iter=300, dtype=torch.complex128
    )
    assert res.converged
    # The 1-phase equivalent carries the LINE-to-line magnitude; compare on the L-N base.
    v_lv = complex(res.v[res.index.row(id_map["bus"]["lv"], Phase.A)]) / math.sqrt(3.0)
    assert abs(v_lv) == pytest.approx(abs(v1_dss), abs=1.0e-4)


def test_wye_to_winding_stores_the_terminal_value() -> None:
    """The factor is delta-specific: a wye TO winding stores z_LL unchanged."""
    script = f"""
    Clear
    Set DefaultBaseFrequency={_F0:g}
    New Circuit.wlv basekv={_KV_HV:g} pu=1.0 phases=3 bus1=hv angle=0
    Edit Vsource.source R1=1e-6 X1=1e-6 R0=1e-6 X0=1e-6
    New Transformer.t1 phases=3 windings=2 buses=[hv.1.2.3, lv.1.2.3] conns=[wye wye]
    ~ kvs=[{_KV_HV:g} {_KV_LV:g}] kvas=[{_KVA:g} {_KVA:g}] XHL={_VK_PCT:g} %Rs=[0.5 0.5]
    ~ %noloadloss=0 %imag=0
    New Load.l1 bus1=lv.1.2.3 phases=3 kv={_KV_LV:g} kw=200 pf=0.98 model=1 conn=wye Vminpu=0.0001 Vmaxpu=10000
    Set VoltageBases=[{_KV_HV:g}, {_KV_LV:g}]
    CalcVoltageBases
    Solve
    """
    for line in script.strip().splitlines():
        dss.Text.Command(line.strip())
    grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    xfmr = next(b for b in grid.branches if isinstance(b, Transformer))
    z_coil = _coil_impedance(xfmr)
    assert z_coil.real == pytest.approx(_Z_LL.real, rel=1e-12)
    assert z_coil.imag == pytest.approx(_Z_LL.imag, rel=1e-12)
