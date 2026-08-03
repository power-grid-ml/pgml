"""Oracle test: OpenDSS ``Loads.Model`` -> pgml ``LoadModel``/``ZipCoefficients``.

Validates the load-model conversion (``pgml.convert.opendss.to_grid`` ->
``_resolve_load_model``) against a LIVE OpenDSS nonlinear solve for:

- Model=1 (constant P, Q) -- the default, `LoadModel.CONST_POWER`.
- Model=2 (constant impedance) -- `LoadModel.CONST_IMPEDANCE`.
- Model=5 (constant current magnitude) -- `LoadModel.CONST_CURRENT`.
- Model=8 (ZIPV, custom coefficients) -- `ZipCoefficients`.

Each is solved with pgml's NONLINEAR ``solve_power_flow`` (which is the only
solve path that reads ``load_model``/``zip_coefficients`` -- the linear
const-Z assembler always uses the base P/Q) and compared against a live
OpenDSS ``Solve`` on the same circuit. A tightened DSS solve tolerance
(``Set Tolerance=1e-13``) isolates the model conversion from DSS's own
(much looser, ``1e-4``) default convergence residual.

Tolerance targets
------------------
- Voltage magnitude: atol = 1e-6 pu.
- Voltage angle:      atol = 1e-4 deg.
"""

from __future__ import annotations

import math

import pytest
import torch

import opendssdirect as dss  # noqa: E402

from pgml.convert.opendss import PhaseMode, to_grid  # noqa: E402
from pgml.schemas.grid_schema import Load, LoadModel, Phase, ZipCoefficients  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402

_F0 = 60.0
_BASEKV = 4.16  # kV, line-to-line

ATOL_VM_PU = 1.0e-6
ATOL_VA_DEG = 1.0e-4


def _dss_clear() -> None:
    dss.Text.Command("Clear")
    dss.Text.Command(f"set DefaultBaseFrequency={int(_F0)}")


def _angle_diff_deg(a: float, b: float) -> float:
    diff = (a - b) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


def _build_circuit(load_line: str) -> None:
    _dss_clear()
    dss.Text.Command(
        f"New Circuit.load_model_test basekv={_BASEKV} pu=1.0 phases=3 bus1=src "
        f"angle=0.0 frequency={_F0} r1=1e-9 x1=1e-9 r0=1e-9 x0=1e-9"
    )
    dss.Text.Command(
        "New Line.l1 phases=3 bus1=src bus2=b1 r1=0.3 x1=0.6 length=1 units=km"
    )
    dss.Text.Command(load_line)
    dss.Text.Command(f"Set voltagebases=[{_BASEKV}]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Set Tolerance=1e-13")
    dss.Text.Command("Set maxiterations=1000")
    dss.Text.Command("Solve")
    assert dss.Solution.Converged(), "DSS load-model circuit did not converge"


def _solve_and_compare() -> tuple:
    grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    result = solve_power_flow(
        grid, slack="ideal", tol=1e-12, max_iter=300, dtype=torch.complex128
    )
    assert result.converged, (
        f"solve_power_flow did not converge (residual={float(result.residual):.3e})"
    )
    for bus_name, node_id in id_map["bus"].items():
        dss.Circuit.SetActiveBus(bus_name)
        dss_va = dss.Bus.puVmagAngle()
        kvbase_ln_kv = dss.Bus.kVBase()
        for k, phase in enumerate(("a", "b", "c")):
            row = result.index.row(node_id, Phase(phase))
            v_val = complex(result.v.reshape(-1)[row].item())
            vm_pu_ours = abs(v_val) / (kvbase_ln_kv * 1_000.0)
            va_deg_ours = math.degrees(math.atan2(v_val.imag, v_val.real))
            vm_pu_ref = dss_va[2 * k]
            va_deg_ref = dss_va[2 * k + 1]
            assert abs(vm_pu_ours - vm_pu_ref) < ATOL_VM_PU, (
                f"bus {bus_name} phase {phase}: |V| mismatch ours={vm_pu_ours:.8f} "
                f"dss={vm_pu_ref:.8f}"
            )
            assert abs(_angle_diff_deg(va_deg_ours, va_deg_ref)) < ATOL_VA_DEG, (
                f"bus {bus_name} phase {phase}: angle mismatch "
                f"ours={va_deg_ours:.6f} dss={va_deg_ref:.6f}"
            )
    return grid, id_map


class TestConstantImpedanceModel:
    """Model=2 (constant impedance) -> LoadModel.CONST_IMPEDANCE."""

    def test_field_conversion_and_voltage_parity(self) -> None:
        _build_circuit(
            f"New Load.z1 phases=3 bus1=b1 kv={_BASEKV} kw=300 kvar=100 model=2"
        )
        grid, _ = _solve_and_compare()
        load = next(a for a in grid.appliances if isinstance(a, Load))
        assert load.load_model == LoadModel.CONST_IMPEDANCE
        assert load.zip_coefficients is None


class TestConstantCurrentModel:
    """Model=5 (constant current magnitude) -> LoadModel.CONST_CURRENT."""

    def test_field_conversion_and_voltage_parity(self) -> None:
        _build_circuit(
            f"New Load.i1 phases=3 bus1=b1 kv={_BASEKV} kw=300 kvar=100 model=5"
        )
        grid, _ = _solve_and_compare()
        load = next(a for a in grid.appliances if isinstance(a, Load))
        assert load.load_model == LoadModel.CONST_CURRENT
        assert load.zip_coefficients is None


class TestZipvModel:
    """Model=8 (ZIPV custom coefficients) -> ZipCoefficients."""

    def test_field_conversion_and_voltage_parity(self) -> None:
        _build_circuit(
            f"New Load.zv1 phases=3 bus1=b1 kv={_BASEKV} kw=300 kvar=100 model=8 "
            "ZIPV=[0.3, 0.1, 0.6, 0.2, 0.1, 0.7, 0.0]"
        )
        grid, _ = _solve_and_compare()
        load = next(a for a in grid.appliances if isinstance(a, Load))
        assert load.load_model == LoadModel.ZIP
        assert load.zip_coefficients == ZipCoefficients(
            z_p=0.3, i_p=0.1, p_p=0.6, z_q=0.2, i_q=0.1, p_q=0.7
        )

    def test_nonzero_cutoff_warns(self, caplog) -> None:
        _dss_clear()
        dss.Text.Command(
            f"New Circuit.zipv_cutoff basekv={_BASEKV} pu=1.0 phases=3 bus1=src "
            f"angle=0.0 frequency={_F0} r1=1e-9 x1=1e-9"
        )
        dss.Text.Command(
            "New Line.l1 phases=3 bus1=src bus2=b1 r1=0.3 x1=0.6 length=1 units=km"
        )
        dss.Text.Command(
            f"New Load.zv2 phases=3 bus1=b1 kv={_BASEKV} kw=300 kvar=100 model=8 "
            "ZIPV=[0.3, 0.1, 0.6, 0.2, 0.1, 0.7, 0.5]"
        )
        dss.Text.Command(f"Set voltagebases=[{_BASEKV}]")
        dss.Text.Command("Calcvoltagebases")
        dss.Text.Command("Solve")
        with caplog.at_level("WARNING", logger="pgml"):
            to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        assert any("cutoff" in r.message.lower() for r in caplog.records), (
            f"expected a ZIPV cutoff warning; got {[r.message for r in caplog.records]}"
        )


class TestUnsupportedModelFallback:
    """Model=3/4/6/7 have no faithful ZIP equivalent -> CONST_POWER + warning."""

    @pytest.mark.parametrize("model_code", [3, 4, 6, 7])
    def test_falls_back_to_const_power_with_warning(self, model_code, caplog) -> None:
        _dss_clear()
        dss.Text.Command(
            f"New Circuit.model_fallback basekv={_BASEKV} pu=1.0 phases=3 bus1=src "
            f"angle=0.0 frequency={_F0} r1=1e-9 x1=1e-9"
        )
        dss.Text.Command(
            "New Line.l1 phases=3 bus1=src bus2=b1 r1=0.3 x1=0.6 length=1 units=km"
        )
        dss.Text.Command(
            f"New Load.mf phases=3 bus1=b1 kv={_BASEKV} kw=300 kvar=100 "
            f"model={model_code}"
        )
        dss.Text.Command(f"Set voltagebases=[{_BASEKV}]")
        dss.Text.Command("Calcvoltagebases")
        dss.Text.Command("Solve")
        with caplog.at_level("WARNING", logger="pgml"):
            grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        load = next(a for a in grid.appliances if isinstance(a, Load))
        assert load.load_model == LoadModel.CONST_POWER
        assert any(
            f"model={model_code}" in r.message.lower() for r in caplog.records
        ), (
            f"expected a Model={model_code} fallback warning; got "
            f"{[r.message for r in caplog.records]}"
        )
