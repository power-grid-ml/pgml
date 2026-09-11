"""Oracle test: source zero-sequence impedance vs a LIVE OpenDSS solve.

A converted :class:`~pgml.schemas.grid_schema.Source` carries a per-phase Thevenin
MATRIX built from the positive- AND zero-sequence impedance pair through the
symmetric-component identity ``Z_self = (Z0 + 2*Z1)/3``, ``Z_mutual = (Z0 - Z1)/3``
(``pgml.convert._common.build_source``). An OpenDSS ``Vsource`` builds its own Yprim
from exactly that identity out of its ``R1``/``X1``/``R0``/``X0`` properties, so the
two models are the same object and the comparison is a genuine parity test, not an
approximation.

Circuit (4-wire LV feeder fed DIRECTLY by the source, no transformer in between --
the configuration where the source's zero-sequence impedance is NOT masked by a
delta winding)::

    Vsource (0.4 kV, R1/X1 = 0.01/0.04, R0/X0 = 0.05/0.30)
      b1 .1 .2 .3 .4  --[ 4-conductor Line, 150 m, explicit R/X matrices ]--  b2
      b1.4 -- grounding Reactor (0.01 Ohm) -- ground
      b2: three single-phase L-N loads of 6 / 2 / 1 kW (unbalanced -> residual
          current, so zero-sequence current flows through the source)

The loads carry a harmonic spectrum with a strong triplen (h = 3) component, so the
harmonic comparison exercises the zero-sequence path at every order.

pgml solves with ``slack="norton"``: the Vsource is a finite-impedance Thevenin, and
only the Norton slack lets that impedance load the bus the way OpenDSS's own Vsource
does (the default ``slack="ideal"`` would pin all three phase voltages and
short-circuit the zero sequence at the feeding bus).

Measured (this environment, float64/complex128, CPU):

==========  ======================  ==============================
order       max |dV| after the fix  max |dV| with Z0 forced to Z1
==========  ======================  ==============================
1           7.8e-8 V (3.3e-10 pu)   6.7 V     (2.9e-2 pu)
3           3.6e-9 V                3.8 V     (107 % of the order)
5           4.3e-9 V                4.1 V     (226 % of the order)
7           5.6e-9 V                3.7 V     (221 % of the order)
9           5.3e-9 V                1.9 V     (114 % of the order)
==========  ======================  ==============================

Tolerances: ``atol = 1e-6 V`` on every order. That is ~20x the observed residual
(7.8e-8 V at the fundamental, set by OpenDSS's own snap-solve tolerance of 1e-10 pu
on a 231 V base) and four orders of magnitude tighter than the error the Z0 = Z1
assumption produces, so the test fails loudly if the sequence split regresses.

``Rg=0 Xg=0`` on the line: OpenDSS adds its own (imperial-calibrated) earth-return
term to a matrix-specified line by default, which pgml's explicit-matrix line model
does not carry. Zeroing it makes both engines solve the same line.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

opendssdirect = pytest.importorskip("opendssdirect")
dss = opendssdirect

from pgml.convert.opendss import PhaseMode, to_grid  # noqa: E402
from pgml.evaluation.oracles.opendss_scenario_oracle import (  # noqa: E402
    _scratch_datapath,
)
from pgml.schemas.grid_schema import (  # noqa: E402
    HarmonicComponent,
    Load,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
)
from pgml.solver import solve_harmonic_flow, solve_power_flow  # noqa: E402

pytestmark = pytest.mark.opendss

_F0 = 50.0
_R1, _X1 = 0.010, 0.040  # Ohm, positive-sequence source Thevenin
_R0, _X0 = 0.050, 0.300  # Ohm, zero-sequence source Thevenin (deliberately != Z1)
_ORDERS = (1, 3, 5, 7, 9)
_SPECTRUM_MAG_PCT = (100.0, 25.0, 12.0, 8.0, 4.0)
_SPECTRUM_ANG_DEG = (0.0, -30.0, 45.0, 90.0, 0.0)
_ATOL_V = 1.0e-6

_NODE_PHASE = {1: Phase.A, 2: Phase.B, 3: Phase.C, 4: Phase.N}


def _build_circuit() -> None:
    """Define and compile the 4-wire LV feeder in the live OpenDSS engine."""
    harm = " ".join(str(h) for h in _ORDERS)
    mags = " ".join(f"{m:g}" for m in _SPECTRUM_MAG_PCT)
    angs = " ".join(f"{a:g}" for a in _SPECTRUM_ANG_DEG)
    script = f"""
    Clear
    Set DefaultBaseFrequency={_F0:g}
    New Circuit.lv4w basekv=0.4 pu=1.0 phases=3 bus1=b1 angle=0
    Edit Vsource.source R1={_R1:g} X1={_X1:g} R0={_R0:g} X0={_X0:g}
    New Spectrum.dev NumHarm={len(_ORDERS)} harmonic=[{harm}] %mag=[{mags}] angle=[{angs}]
    New Linecode.lv4 nphases=4 units=km Rg=0 Xg=0
    ~ rmatrix=[0.32 | 0.04 0.32 | 0.04 0.04 0.32 | 0.04 0.04 0.04 0.32]
    ~ xmatrix=[0.08 | 0.03 0.08 | 0.03 0.03 0.08 | 0.03 0.03 0.03 0.08]
    ~ cmatrix=[0 | 0 0 | 0 0 0 | 0 0 0 0]
    New Line.l1 bus1=b1.1.2.3.4 bus2=b2.1.2.3.4 linecode=lv4 length=0.15 units=km Rg=0 Xg=0
    New Reactor.ng phases=1 bus1=b1.4 R=0.01 X=0
    New Load.la bus1=b2.1.4 phases=1 kv=0.231 kw=6 pf=1 model=1 conn=wye Vminpu=0.0001 Vmaxpu=10000 spectrum=dev
    New Load.lb bus1=b2.2.4 phases=1 kv=0.231 kw=2 pf=1 model=1 conn=wye Vminpu=0.0001 Vmaxpu=10000 spectrum=dev
    New Load.lc bus1=b2.3.4 phases=1 kv=0.231 kw=1 pf=1 model=1 conn=wye Vminpu=0.0001 Vmaxpu=10000 spectrum=dev
    Set VoltageBases=[0.4]
    CalcVoltageBases
    Set NeglectLoadY=yes
    Set Tolerance=1e-10
    Set MaxIterations=100
    """
    for line in script.strip().splitlines():
        dss.Text.Command(line.strip())


def _bus_voltages() -> dict[str, dict[int, complex]]:
    """Per-bus ``{dss node number: complex voltage [V]}`` of the current solution."""
    out: dict[str, dict[int, complex]] = {}
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        nodes = list(dss.Bus.Nodes())
        v = np.array(dss.Bus.Voltages())
        out[bus] = dict(zip(nodes, v[0::2] + 1j * v[1::2]))
    return out


def _solve_opendss():
    """Solve the live circuit, converting the grid while the engine is at ``f0``.

    ``to_grid`` takes its base frequency from the engine's CURRENT solution
    frequency, so the conversion happens right after the fundamental snap solve --
    converting while the engine sits at a harmonic frequency would stamp that
    frequency as the grid's ``base_frequency_hz``.
    """
    _build_circuit()
    per_order: dict[int, dict[str, dict[int, complex]]] = {}
    with _scratch_datapath():
        dss.Text.Command("Set Mode=Snap")
        dss.Text.Command("Solve")
        assert dss.Solution.Converged(), "OpenDSS snap solve did not converge"
        per_order[1] = _bus_voltages()
        grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        dss.Text.Command("Set Mode=Harmonics")
        for h in _ORDERS[1:]:
            dss.Text.Command(f"Set Harmonic={h}")
            dss.Text.Command("Solve")
            assert dss.Solution.Converged(), f"OpenDSS harmonic solve failed at h={h}"
            per_order[h] = _bus_voltages()
    return per_order, grid, id_map


def _attach_spectrum(grid) -> None:
    spec = StaticSpectrum(
        spectrum=SpectrumPoint(
            components=[
                HarmonicComponent(order=o, magnitude_pu=m / 100.0, phase_deg=a)
                for o, m, a in zip(_ORDERS, _SPECTRUM_MAG_PCT, _SPECTRUM_ANG_DEG)
            ]
        )
    )
    for appliance in grid.appliances:
        if isinstance(appliance, Load):
            appliance.spectrum = spec


def _force_z0_equal_z1(grid):
    """Return a copy of ``grid`` whose source carries the pre-fix diagonal Thevenin."""
    pre_fix = grid.model_copy(deep=True)
    for appliance in pre_fix.appliances:
        if isinstance(appliance, Source):
            n = len(appliance.phases)
            appliance.resistance_ohm = [
                [_R1 if i == j else 0.0 for j in range(n)] for i in range(n)
            ]
            appliance.inductance_h = [
                [_X1 / (2.0 * math.pi * _F0) if i == j else 0.0 for j in range(n)]
                for i in range(n)
            ]
    return pre_fix


def _max_abs_deviation(v_rows, index, id_map, v_dss) -> float:
    worst = 0.0
    for bus, per_node in v_dss.items():
        node_id = id_map["bus"][bus]
        for dss_node, v_ref in per_node.items():
            row = index.row(node_id, _NODE_PHASE[dss_node])
            worst = max(worst, abs(complex(v_rows[row]) - v_ref))
    return worst


@pytest.fixture(scope="module")
def live_case():
    """``(v_dss_per_order, grid, id_map)`` for the shared 4-wire feeder."""
    v_dss, grid, id_map = _solve_opendss()
    assert float(grid.base_frequency_hz) == pytest.approx(_F0)
    _attach_spectrum(grid)
    return v_dss, grid, id_map


class TestSourceThevininMatrix:
    """The converted Thevenin matrix is the symmetric-component split of (Z1, Z0)."""

    def test_matrix_matches_sequence_identity(self, live_case) -> None:
        _, grid, _ = live_case
        sources = [a for a in grid.appliances if isinstance(a, Source)]
        assert len(sources) == 1
        src = sources[0]
        assert src.phases == (Phase.A, Phase.B, Phase.C)
        w0 = 2.0 * math.pi * _F0
        r_self, r_mut = (_R0 + 2.0 * _R1) / 3.0, (_R0 - _R1) / 3.0
        x_self, x_mut = (_X0 + 2.0 * _X1) / 3.0, (_X0 - _X1) / 3.0
        for i in range(3):
            for j in range(3):
                assert src.resistance_ohm[i][j] == pytest.approx(
                    r_self if i == j else r_mut, abs=1e-12
                )
                assert src.inductance_h[i][j] * w0 == pytest.approx(
                    x_self if i == j else x_mut, abs=1e-12
                )

    def test_matrix_eigenvalues_are_z1_and_z0(self, live_case) -> None:
        """Sequence eigenvalues of the matrix reproduce the DSS Vsource pair."""
        _, grid, _ = live_case
        src = [a for a in grid.appliances if isinstance(a, Source)][0]
        w0 = 2.0 * math.pi * _F0
        z = np.array(
            [
                [
                    complex(src.resistance_ohm[i][j], w0 * src.inductance_h[i][j])
                    for j in range(3)
                ]
                for i in range(3)
            ]
        )
        ones = np.ones(3)
        z0 = complex(ones @ z @ ones / 3.0)
        a = np.exp(2j * np.pi / 3.0)
        pos = np.array([1.0, a**2, a])
        z1 = complex(np.conj(pos) @ z @ pos / 3.0)
        assert z0.real == pytest.approx(_R0, abs=1e-12)
        assert z0.imag == pytest.approx(_X0, abs=1e-12)
        assert z1.real == pytest.approx(_R1, abs=1e-12)
        assert z1.imag == pytest.approx(_X1, abs=1e-12)


class TestLiveOpenDSSParity:
    """Full-solve parity against the live OpenDSS circuit."""

    def test_fundamental_unbalanced_voltages(self, live_case) -> None:
        v_dss, grid, id_map = live_case
        res = solve_power_flow(
            grid, slack="norton", tol=1e-12, max_iter=200, dtype=torch.complex128
        )
        assert res.converged
        dev = _max_abs_deviation(res.v, res.index, id_map, v_dss[1])
        assert dev < _ATOL_V, f"fundamental deviation {dev:.3e} V"

    def test_harmonic_voltages_every_order(self, live_case) -> None:
        v_dss, grid, id_map = live_case
        res = solve_harmonic_flow(
            grid,
            list(_ORDERS),
            slack="norton",
            tol=1e-12,
            max_iter=300,
            dtype=torch.complex128,
        )
        assert res.converged
        for k, h in enumerate(_ORDERS):
            dev = _max_abs_deviation(res.v[k], res.index, id_map, v_dss[h])
            assert dev < _ATOL_V, f"order {h} deviation {dev:.3e} V"

    def test_neutral_voltage_is_reproduced(self, live_case) -> None:
        """The 4-wire neutral rows (grounded through a Reactor) match as well."""
        v_dss, grid, id_map = live_case
        res = solve_power_flow(
            grid, slack="norton", tol=1e-12, max_iter=200, dtype=torch.complex128
        )
        for bus in ("b1", "b2"):
            row = res.index.row(id_map["bus"][bus], Phase.N)
            v_ref = v_dss[1][bus][4]
            assert abs(v_ref) > 1e-3, "the case must carry a nonzero neutral voltage"
            assert complex(res.v[row]) == pytest.approx(v_ref, abs=_ATOL_V)

    def test_z0_equal_z1_assumption_is_wrong_by_more_than_the_tolerance(
        self, live_case
    ) -> None:
        """The pre-fix model (Z0 := Z1) misses the triplen voltage by >100 %."""
        v_dss, grid, id_map = live_case
        pre_fix = _force_z0_equal_z1(grid)
        res = solve_harmonic_flow(
            pre_fix,
            list(_ORDERS),
            slack="norton",
            tol=1e-12,
            max_iter=300,
            dtype=torch.complex128,
        )
        dev_h3 = _max_abs_deviation(res.v[1], res.index, id_map, v_dss[3])
        scale_h3 = max(abs(v) for d in v_dss[3].values() for v in d.values())
        assert dev_h3 > 0.5 * scale_h3
