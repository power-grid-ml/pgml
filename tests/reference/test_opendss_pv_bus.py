"""Oracle test: OpenDSS ``Generator model=3`` (constant P, constant |V|) vs pgml.

``model=3`` is OpenDSS's PV bus. The converter maps it onto a
:class:`~pgml.schemas.grid_schema.Generator` with a
:class:`~pgml.schemas.grid_schema.VoltageRegulation` block (``Vpu`` -> setpoint,
re-referred from the machine's ``kV`` rating to the bus base; ``Maxkvar``/``Minkvar``
-> reactive limits), and the solver replaces that terminal's reactive power-balance
row with ``|V|**2 - V_set**2``.

Two kinds of check:

- the MAPPING (deterministic, no solve): the setpoint and the limits the converter
  reads out of the DSS properties, including the ``kV``-vs-bus-base re-referral and
  the refusal of a delta-connected ``model=3`` machine;
- the PHYSICS against a live OpenDSS solve, on the two configurations whose OpenDSS
  solution is numerically robust: a machine pinned at its upper and at its lower
  reactive limit. OpenDSS reaches ``model=3`` by its own iterative reactive update,
  which on this circuit is sensitive to the engine's warm state (a cold solve can hit
  ``MaxIter`` where a repeated one converges in ~100 iterations) and even to the
  process floating-point flags, so the REGULATING configuration is measured in the
  validation record rather than pinned here: ``agent-reports`` 13_pv_bus (agreement
  6.6e-10 pu in voltage and 4.2e-4 kvar in reactive power, pgml 4 Newton iterations
  against OpenDSS's 108). The two limit cases below converge in 8 iterations from
  cold and agree to 1e-13 pu / 1e-8 kvar.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import opendssdirect as dss  # noqa: E402

from pgml.convert._common import PhaseMode  # noqa: E402
from pgml.convert.opendss import to_grid  # noqa: E402
from pgml.errors import ConversionError  # noqa: E402
from pgml.schemas.grid_schema import Generator, Phase  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402

CDT = torch.complex128
_BASEKV = 20.0  # kV line-to-line
_PH3 = (Phase.A, Phase.B, Phase.C)


def _commands(vpu, maxkvar, *, gen_kv=_BASEKV, conn="wye", kw=500.0, load_kw=1000.0):
    """A two-bus 20 kV feeder with one ``model=3`` machine at the far bus.

    The machine is created as a PQ injection and switched to ``model=3`` after the
    first solve: OpenDSS's ``model=3`` reactive update starts from the previous
    solution, so bringing it up this way is what its own manual recommends.
    """
    return [
        "Clear",
        f"New Circuit.pvbus basekv={_BASEKV} pu=1.0 phases=3 bus1=sourcebus "
        "MVAsc3=10000 MVAsc1=10000",
        "Set DefaultBaseFrequency=50",
        "New Line.l1 bus1=sourcebus bus2=b1 phases=3 r1=0.4 x1=0.3 r0=0.4 x0=0.3 "
        "c1=0 c0=0 length=2.0 units=km",
        f"New Load.ld bus1=b1 phases=3 conn=wye kV={_BASEKV} kW={load_kw} kvar=300 "
        "model=1 Vminpu=0.1 Vmaxpu=2",
        f"New Generator.g1 bus1=b1 phases=3 conn={conn} kV={gen_kv} kW={kw} model=1 "
        "kvar=0 Vminpu=0.1 Vmaxpu=2",
        f"Set VoltageBases=[{_BASEKV}]",
        "Calcvoltagebases",
        "Set Mode=Snapshot",
        "Set Tolerance=1e-10",
        "Set MaxIter=2000",
        "Solve",
        f"Edit Generator.g1 model=3 Vpu={vpu} maxkvar={maxkvar} minkvar={-maxkvar}",
        "Solve",
    ]


def _build(vpu, maxkvar, **kw) -> None:
    for c in _commands(vpu, maxkvar, **kw):
        dss.Command(c)


def _dss_solution():
    """``(|V| pu at b1, injected kvar, iterations)`` of the live OpenDSS solve."""
    assert dss.Solution.Converged(), "OpenDSS did not converge"
    dss.Circuit.SetActiveBus("b1")
    vm = float(np.mean(np.array(dss.Bus.puVmagAngle())[0::2]))
    dss.Circuit.SetActiveElement("Generator.g1")
    q_kvar = -float(np.array(dss.CktElement.Powers())[1:6:2].sum())
    return vm, q_kvar, dss.Solution.Iterations()


def _pgml_solution(grid, id_map):
    """``(|V| pu at b1, injected kvar, regulating, iterations)`` from pgml."""
    res = solve_power_flow(
        grid,
        slack="norton",  # the OpenDSS Vsource convention
        method="newton",
        tol=1e-7,
        max_iter=60,
        dtype=CDT,
        criticality="never",
    )
    assert res.converged, f"pgml did not converge ({float(res.residual):.3e})"
    rows = [res.index.row(id_map["bus"]["b1"], ph) for ph in _PH3]
    base = _BASEKV * 1_000.0 / math.sqrt(3.0)
    vm = float(res.v[rows].abs().mean()) / base
    gid = id_map["generator"]["g1"]
    return (
        vm,
        float(res.regulation.q_var[gid]) / 1.0e3,
        bool(res.regulation.regulating[gid]),
        res.iterations,
    )


# ---------------------------------------------------------------------------
# 1. The mapping
# ---------------------------------------------------------------------------
@pytest.mark.opendss
class TestMapping:
    def test_model_3_becomes_a_regulating_generator(self):
        _build(1.02, 750.0)
        grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        gen = next(
            a
            for a in grid.appliances
            if isinstance(a, Generator) and a.id == id_map["generator"]["g1"]
        )
        reg = gen.voltage_regulation
        assert reg is not None
        assert reg.v_set_pu == pytest.approx(1.02)
        assert reg.q_max_var == pytest.approx(750.0e3)
        assert reg.q_min_var == pytest.approx(-750.0e3)
        # A regulating machine's reactive nameplate is never read.
        assert gen.q_nom_var == 0.0
        assert gen.p_nom_w == pytest.approx(500.0e3)

    def test_model_1_stays_a_pq_injection(self):
        _build(1.02, 750.0)
        dss.Command("Edit Generator.g1 model=1 kvar=120")
        grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        gen = next(
            a
            for a in grid.appliances
            if isinstance(a, Generator) and a.id == id_map["generator"]["g1"]
        )
        assert gen.voltage_regulation is None
        assert gen.q_nom_var == pytest.approx(120.0e3)

    def test_setpoint_is_re_referred_from_the_machine_rating(self):
        """``Vpu`` is per unit of the machine's own ``kV``; the schema's setpoint is
        per unit of the BUS rating, so a machine rated 10 % above its bus carries a
        setpoint 10 % above its ``Vpu``."""
        _build(1.0, 750.0, gen_kv=_BASEKV * 1.1)
        grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        gen = next(
            a
            for a in grid.appliances
            if isinstance(a, Generator) and a.id == id_map["generator"]["g1"]
        )
        assert gen.voltage_regulation.v_set_pu == pytest.approx(1.1)

    def test_delta_model_3_raises(self):
        _build(1.02, 750.0, conn="delta")
        with pytest.raises(ConversionError, match="delta"):
            to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)


# ---------------------------------------------------------------------------
# 2. The physics, against the live solve
# ---------------------------------------------------------------------------
@pytest.mark.opendss
class TestAgainstOpenDSS:
    @pytest.mark.parametrize(
        "vpu,maxkvar,sign", [(1.02, 500.0, +1.0), (0.98, 500.0, -1.0)]
    )
    def test_machine_pinned_at_a_reactive_limit(self, vpu, maxkvar, sign):
        """The setpoint is out of reach, so both tools pin the machine at the limit
        and must land on the same voltage."""
        _build(vpu, maxkvar)
        vm_dss, q_dss, _it_dss = _dss_solution()
        grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        vm_pgml, q_pgml, regulating, _it = _pgml_solution(grid, id_map)
        assert q_dss == pytest.approx(sign * maxkvar, rel=1e-6)  # OpenDSS pinned it
        assert not regulating  # and so did pgml
        assert q_pgml == pytest.approx(sign * maxkvar, rel=1e-9)
        assert abs(vm_pgml - vm_dss) < 1.0e-11, (
            f"|V| {vm_pgml:.12f} vs OpenDSS {vm_dss:.12f}"
        )
        assert abs(q_pgml - q_dss) < 1.0e-6  # kvar
