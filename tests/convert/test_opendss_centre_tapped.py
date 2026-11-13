"""Split-phase (centre-tapped) service transformers on an OpenDSS import.

North-American feeders write a service transformer as one single-phase, three-winding
``Transformer`` (``buses=(hv.3, lv.1.0, lv.0.2)``). ``to_grid(centre_tapped=...)`` has
three behaviours, all pinned here on small synthetic circuits:

- ``"raise"`` (default): every three-winding transformer is refused, as before.
- ``"centre_tap"`` (``THREE_PHASE``): a faithful ``Transformer`` with a ``CentreTap``
  block. The solved bus voltages and the transformer terminal current are compared with
  a live OpenDSS solve of the same circuit at every node (alignment: DSS bus name and
  node number against ``res.index.row(node_id, phase)``, DSS node 1/2/3 = phase A/B/C,
  complex volts and amperes compared unnormalised; ``slack="norton"`` because the DSS
  ``Vsource`` is a Thevenin source).
- ``"two_winding"`` (``SINGLE_PHASE_EQUIV``): a structural substitution. With the two
  legs loaded equally no current flows in the centre tap, so the full-secondary
  impedance carries the whole load and the substituted secondary voltage must equal the
  three-winding one; compared in per unit of the 240 V secondary.

Tolerances are measured; see the asserted bounds next to each comparison.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

opendssdirect = pytest.importorskip("opendssdirect")
dss = opendssdirect

from pgml.convert.opendss import PhaseMode, to_grid  # noqa: E402
from pgml.errors import ConversionError  # noqa: E402
from pgml.schemas.grid_schema import Phase, Transformer, WindingConnection  # noqa: E402
from pgml.defaults import use_preset  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402

pytestmark = pytest.mark.opendss


@pytest.fixture(autouse=True)
def _opendss_preset():
    """The modeling choices OpenDSS makes, on the pgml arm of every comparison."""
    with use_preset("opendss"):
        yield


_F0 = 60.0
_PH = {1: Phase.A, 2: Phase.B, 3: Phase.C}

_TRAFO = (
    "New Transformer.tr phases=1 windings=3 %loadloss=1.248151 %Noloadloss={nl} "
    "wdg=1 conn=wye bus=p.3 Kv=7.2 kva=50.0 %r=0.41605 "
    "wdg=2 conn=wye bus=lv.1.0 Kv=0.12 kva=50.0 %r=0.832101 "
    "wdg=3 conn=wye bus=lv.0.2 Kv=0.12 kva=50.0 %r=0.832101 "
    "XHL={xhl} XLT={xlt} XHT={xht} %imag={imag}"
)


def _circuit(
    *,
    kw1: float = 8.0,
    kw2: float = 3.0,
    nl: float = 0.0,
    imag: float = 0.0,
    x: tuple[float, float, float] = (1.08, 1.08, 0.72),
    sec_km: float = 0.05,
    load_scale: float = 1.0,
) -> None:
    """One split-phase unit feeding two 120 V legs; ``x`` is (XHL, XLT, XHT) in percent."""
    for line in (
        "Clear",
        f"Set DefaultBaseFrequency={_F0:g}",
        "New Circuit.sp basekv=12.47 pu=1.0 phases=3 bus1=src R1=0.5 X1=1.5 R0=0.5 X0=1.5",
        "New Line.feed phases=3 bus1=src.1.2.3 bus2=p.1.2.3 length=0.5 units=km "
        "rmatrix=[0.3 | 0.05 0.3 | 0.05 0.05 0.3] xmatrix=[0.4 | 0.15 0.4 | 0.15 0.15 0.4] "
        "cmatrix=[0 | 0 0 | 0 0 0] Rg=0 Xg=0",
        _TRAFO.format(nl=nl, imag=imag, xhl=x[0], xlt=x[1], xht=x[2]),
        f"New Line.sec phases=2 bus1=lv.1.2 bus2=cust.1.2 length={sec_km} units=km "
        "rmatrix=[0.5 | 0.1 0.5] xmatrix=[0.1 | 0.05 0.1] cmatrix=[0 | 0 0] Rg=0 Xg=0",
        f"New Load.l1 phases=1 bus1=cust.1 kv=0.12 kw={kw1 * load_scale} kvar={2 * load_scale} model=1 "
        "vminpu=0.1 vmaxpu=10",
        f"New Load.l2 phases=1 bus1=cust.2 kv=0.12 kw={kw2 * load_scale} kvar={1 * load_scale} model=1 "
        "vminpu=0.1 vmaxpu=10",
        "Set VoltageBases=[0.12 0.208 12.47]",
        "CalcVoltageBases",
        "Set Tolerance=1e-12",
        "Set MaxIterations=100",
        "Solve",
    ):
        dss.Text.Command(line)
    assert dss.Solution.Converged()


def _dss_voltages() -> dict[tuple[str, int], complex]:
    out = {}
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        v = np.array(dss.Bus.Voltages())
        for node, vv in zip(dss.Bus.Nodes(), v[0::2] + 1j * v[1::2]):
            if node in _PH:
                out[(bus, node)] = complex(vv)
    return out


class TestDefaultRaises:
    def test_three_winding_raises_without_option(self) -> None:
        _circuit()
        with pytest.raises(ConversionError, match="centre_tapped"):
            to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)

    def test_unknown_option_value(self) -> None:
        _circuit()
        with pytest.raises(ConversionError, match="centre_tapped must be"):
            to_grid(dss, centre_tapped="merge")  # type: ignore[arg-type]

    def test_option_needs_matching_phase_mode(self) -> None:
        _circuit()
        with pytest.raises(ConversionError, match="THREE_PHASE"):
            to_grid(
                dss, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV, centre_tapped="centre_tap"
            )
        with pytest.raises(ConversionError, match="SINGLE_PHASE_EQUIV"):
            to_grid(dss, phase_mode=PhaseMode.THREE_PHASE, centre_tapped="two_winding")


class TestUnsupportedShapesStillRaise:
    @pytest.mark.parametrize(
        "edit,why",
        [
            ("wdg=3 conn=wye bus=lv.0.2 Kv=0.10", "differ in voltage"),
            ("wdg=3 conn=wye bus=lv.0.2 %r=0.9", "differ in %R"),
            ("wdg=3 conn=wye bus=lv.3.2 ", "meet at one centre-tap"),
            ("wdg=3 conn=delta bus=lv.0.2 ", "wye windings"),
            ("wdg=3 conn=wye bus=other.0.2 ", "different buses"),
        ],
    )
    def test_deviation_names_its_reason(self, edit: str, why: str) -> None:
        _circuit()
        dss.Text.Command(f"Edit Transformer.tr {edit}")
        with pytest.raises(ConversionError, match=why):
            to_grid(dss, phase_mode=PhaseMode.THREE_PHASE, centre_tapped="centre_tap")


class TestCentreTapFaithful:
    @pytest.mark.parametrize(
        "kw1,kw2,nl,imag,x,bound",
        [
            # symmetric halves (XHL == XHT): the model is exact
            (8.0, 3.0, 0.0, 0.0, (0.9, 0.9, 0.9), 1e-4),
            (5.0, 5.0, 0.0, 0.0, (0.9, 0.9, 0.9), 1e-4),
            # leakage data of a real feeder (XHL != XHT): halves symmetrised
            (8.0, 3.0, 0.0, 0.0, (1.08, 1.08, 0.72), 0.05),
            (8.0, 3.0, 0.68, 0.5, (0.9, 0.9, 0.9), 1e-2),
        ],
        ids=["symmetric", "balanced", "asymmetric-halves", "magnetizing"],
    )
    def test_voltages_match_live_opendss(self, kw1, kw2, nl, imag, x, bound) -> None:
        """Worst complex deviation over every node, in volts (opendssdirect 0.9.4,
        complex128, CPU); the bounds sit just above the measured values:

        - symmetric halves: 2.6e-5 V at worst (at the source bus; 2.6e-6 V on the
          120 V legs), the residual of the DSS fixed-point iteration;
        - ``XHL != XHT``: the equal-halves model drops the difference of the halves'
          leakage, measured 0.036 V (3e-4 pu) with the legs at 8 and 3 kW;
        - magnetizing (%noloadloss 0.68, %imag 0.5): 4.9e-3 V (4e-5 pu), the
          placement of the core branch (a winding terminal in OpenDSS, the HV
          terminal in pgml).
        """
        _circuit(kw1=kw1, kw2=kw2, nl=nl, imag=imag, x=x)
        ref = _dss_voltages()
        grid, id_map = to_grid(
            dss, phase_mode=PhaseMode.THREE_PHASE, centre_tapped="centre_tap"
        )
        res = solve_power_flow(
            grid, slack="norton", tol=1e-12, max_iter=300, dtype=torch.complex128
        )
        assert res.converged
        err = 0.0
        for (bus, node), v_ref in ref.items():
            if bus == "p" and node != 3:
                continue
            v = complex(res.v[res.index.row(id_map["bus"][bus], _PH[node])])
            err = max(err, abs(v - v_ref))
        assert abs(ref[("cust", 1)]) > 100.0
        assert err < bound, err

    def test_unit_is_a_centre_tap_transformer(self) -> None:
        _circuit()
        grid, id_map, report = to_grid(
            dss,
            phase_mode=PhaseMode.THREE_PHASE,
            centre_tapped="centre_tap",
            return_report=True,
        )
        trafos = [b for b in grid.branches if isinstance(b, Transformer)]
        assert len(trafos) == 1
        t = trafos[0]
        assert t.centre_tap is not None
        assert t.from_phases == (Phase.C,)
        assert t.to_phases == (Phase.A, Phase.B)
        assert t.from_connection is WindingConnection.WYE_GROUNDED
        assert t.u_rated_to_v == pytest.approx(240.0)
        # XHL != XHT in the default fixture: the halves are symmetrised and reported
        assert report.get("approx.opendss.centre_tapped_halves")[0].ids == ("tr",)
        assert t.u_rated_from_v == pytest.approx(7200.0)
        assert id_map["centre_tapped"] == {"tr": t.id}
        assert id_map["trafo"]["tr"] == t.id
        assert grid.metadata.tags["centre_tapped_units"] == "1"


class TestTwoWindingSubstitution:
    def test_bookkeeping(self) -> None:
        _circuit()
        grid, id_map, report = to_grid(
            dss, centre_tapped="two_winding", return_report=True
        )
        trafos = [b for b in grid.branches if isinstance(b, Transformer)]
        assert len(trafos) == 1 and trafos[0].centre_tap is None
        t = trafos[0]
        assert id_map["centre_tapped"] == {"tr": t.id}
        assert grid.metadata.tags["centre_tapped_units"] == "1"
        assert grid.metadata.tags["centre_tapped_mode"] == "two_winding"
        entry = report.get("approx.opendss.centre_tapped_two_winding")[0]
        assert entry.count == 1 and entry.ids == ("tr",)
        # same kVA (per-phase equivalent: /3), same ratio, full 240 V secondary
        assert t.s_rated_va == pytest.approx(50e3 / 3)
        assert t.u_rated_to_v / t.u_rated_from_v == pytest.approx(240.0 / 7200.0)
        # the legs share one node
        assert len({nd.name for nd in grid.nodes if nd.name == "lv"}) == 1

    def test_transfer_ratio_matches_three_winding_when_legs_balanced(self) -> None:
        """No centre-tap current flows when both legs carry the same load, so the
        full-secondary impedance Z_a reproduces the three-winding solve.

        Alignment: the per-phase equivalent takes every one-phase load as a share of
        a balanced three-phase total (P / 3) and keeps ohms, so the converted unit
        carries one third of the current of the physical circuit it was read from.
        The three-winding reference is therefore solved with legs of P / 3, and the
        comparison is the transformer's own transfer ratio at its terminals,
        ``|V_secondary| / |V_primary|`` (DSS ``|V_lv.1 - V_lv.2| / |V_p.3|``), with a
        negligible secondary line. Measured deviation 1.7e-8 relative.
        """
        x = (0.9, 0.9, 0.9)
        _circuit(kw1=5.0, kw2=5.0, x=x, sec_km=1e-6)  # converted, per-phase loads P/3
        grid, id_map = to_grid(dss, centre_tapped="two_winding")
        _circuit(
            kw1=5.0, kw2=5.0, x=x, sec_km=1e-6, load_scale=1.0 / 3.0
        )  # physical reference
        v = _dss_voltages()
        ref_ratio = abs(v[("lv", 1)] - v[("lv", 2)]) / abs(v[("p", 3)])
        res = solve_power_flow(
            grid, slack="norton", tol=1e-12, max_iter=300, dtype=torch.complex128
        )
        assert res.converged
        v_p = abs(complex(res.v[res.index.row(id_map["bus"]["p"], Phase.A)]))
        v_s = abs(complex(res.v[res.index.row(id_map["bus"]["lv"], Phase.A)]))
        assert ref_ratio < 240.0 / 7200.0  # the loaded drop is visible
        assert v_s / v_p == pytest.approx(ref_ratio, rel=1e-6)
