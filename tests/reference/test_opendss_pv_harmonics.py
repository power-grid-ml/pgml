"""Live native OpenDSS model=3 initialization from solved, not configured, Q."""

import numpy as np
import pytest
import torch

from pgml import defaults
from pgml.convert.opendss import PhaseMode, to_grid
from pgml.solver import solve_harmonic_flow
from tests.reference.test_opendss_pv_bus import _build
from tests.reference.test_native_der_harmonics import native_voltages

dss = pytest.importorskip("opendssdirect")
pytestmark = pytest.mark.opendss


@pytest.mark.parametrize("mode", ["upper", "lower", "regulating"])
def test_native_pv_harmonics_use_solved_q(mode, tmp_path):
    dss.Command("set defaultbasefrequency=50")
    _build(1.02 if mode != "lower" else 0.98, 500.0)
    dss.Command(f"set datapath={tmp_path}")
    dss.Command("new spectrum.clean numharm=1 harmonic=[1] %mag=[100] angle=[0]")
    dss.Command("edit vsource.source spectrum=clean")
    dss.Command("edit load.ld spectrum=clean")
    dss.Command(
        "new spectrum.emission numharm=3 harmonic=[1 5 7] %mag=[100 8 5] angle=[0 17 -9]"
    )
    dss.Command("edit generator.g1 spectrum=emission")
    # Align frequency laws so this checks source initialization, not earth-return approximations.
    dss.Command("edit line.l1 rg=0 xg=0")
    dss.Command("edit generator.g1 pvfactor=0.05")
    dss.Command("set tolerance=1e-13 maxiterations=10000")
    if mode == "regulating":
        dss.Command("edit generator.g1 model=1 kvar=120")
        dss.Command("solve")
        dss.Circuit.SetActiveBus("b1")
        setpoint = float(np.mean(dss.Bus.puVmagAngle()[::2]))
        dss.Command(
            f"edit generator.g1 model=3 vpu={setpoint:.16g} maxkvar=500 minkvar=-500"
        )
    dss.Command("solve")
    assert dss.Solution.Converged()
    dss.Circuit.SetActiveElement("Generator.g1")
    q_ref = -1000.0 * np.asarray(dss.CktElement.Powers())[1:6:2].sum()
    refs = {1: native_voltages()}
    grid, mapping = to_grid(
        dss, phase_mode=PhaseMode.THREE_PHASE, harmonic_line_model="naive"
    )
    gid = mapping["generator"]["g1"]
    with defaults.use_preset("opendss"):
        result = solve_harmonic_flow(
            grid,
            [1, 5, 7],
            slack="norton",
            method="newton",
            load_shunt="none",
            tol=1e-11,
            max_iter=100,
        )
    assert result.converged
    assert float(result.pf.regulation.q_var[gid]) == pytest.approx(q_ref, abs=0.02)
    assert bool(result.pf.regulation.regulating[gid]) == (mode == "regulating")
    dss.Command("set neglectloady=yes")
    dss.Command("set mode=harmonic")
    for h in [5, 7]:
        dss.Command(f"set harmonics=[{h}]")
        dss.Command("solve")
        refs[h] = native_voltages()
    inverse = {nid: bus for bus, nid in mapping["bus"].items()}
    nums = {"a": 1, "b": 2, "c": 3}
    keys = [
        f"{inverse[result.index.node_id_of(i)]}.{nums[result.index.phase_of(i).value]}"
        for i in range(result.index.size)
    ]
    reference = torch.tensor(
        [[refs[h][key] for key in keys] for h in [1, 5, 7]], dtype=torch.complex128
    )
    torch.testing.assert_close(result.v, reference, atol=2e-5, rtol=2e-7)
