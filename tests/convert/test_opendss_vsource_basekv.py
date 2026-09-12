"""Verify OpenDSS ``Vsource.basekv`` semantics vs phase count, and that the
converter's ``Source.u_ref_v`` / ``Node.u_rated_v`` formulas track them.

OpenDSS's general documentation describes ``Vsource.basekv`` as the
line-to-line nominal voltage. That is true for a 3-phase (or any >=3-phase)
Vsource, but NOT for a single-phase (``phases=1``) one: OpenDSS uses
``basekv`` directly, unscaled, as the magnitude of the single conductor-pair
EMF -- there is no internal ``sqrt(3)`` multiplication or division for a
1-phase source. This is verified empirically below (Solved bus voltage
magnitude == ``BasekV * pu`` exactly, for TWO different single-phase
``basekv`` styles):

- the historical positive-sequence-equivalent style, where ``basekv`` is set
  to the ORIGINAL 3-phase system's line-to-line nominal (e.g. IEEE 33-bus,
  ``phases=1, basekv=12.66``) -- a deliberate pgml/pandapower-matching
  convention, not a "real" line-to-neutral reading;
- a genuine single-phase source, where ``basekv`` is set to the true
  line-to-neutral EMF of one physical phase (e.g. ``basekv=7.2``, one leg of
  a 12.47 kV system).

The pgml converter's ``u_ref_v = BasekV * pu * 1000`` and
``u_rated_v = kVBase() * sqrt(3) * 1000`` formulas are UNCONDITIONALLY
correct across both styles and both phase counts -- no phase-count branch is
needed in the converter itself -- because they mirror OpenDSS's own internal
bookkeeping exactly: ``Bus.kVBase()`` (via ``Calcvoltagebases``) always
divides the actual solved per-conductor voltage by ``sqrt(3)`` regardless of
phase count (to match it against a declared L-L ``voltagebases`` class), so
multiplying back by ``sqrt(3)`` exactly recovers the solved voltage --
independent of whether that solved voltage is "genuinely" L-L (3-phase) or
L-N (1-phase). See ``docs/pgml/modeling/references/opendss/index.md`` and
``docs/pgml/modeling/conventions.md`` sec. 6 for the documented convention.
"""

from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------------------
# Optional opendssdirect guard (matches existing reference test conventions)
# ---------------------------------------------------------------------------
try:
    import opendssdirect as dss

    _OPENDSS_AVAILABLE = True
except ImportError:
    _OPENDSS_AVAILABLE = False

if not _OPENDSS_AVAILABLE:
    pytest.skip("opendssdirect not installed", allow_module_level=True)

pytestmark = pytest.mark.opendss

from pgml.convert._common import PhaseMode  # noqa: E402
from pgml.convert.opendss import to_grid  # noqa: E402
from pgml.schemas.grid_schema import Phase, Source  # noqa: E402

_F0 = 60.0


def _dss_clear() -> None:
    dss.Text.Command("Clear")
    dss.Text.Command(f"set DefaultBaseFrequency={int(_F0)}")


# ---------------------------------------------------------------------------
# Empirical ground truth: what does OpenDSS do with `basekv`?
# ---------------------------------------------------------------------------


class TestOpenDSSBasekvSemantics:
    """Pin OpenDSS's own (undocumented-for-1-phase) basekv -> solved-voltage rule."""

    def test_three_phase_basekv_is_line_to_line(self) -> None:
        """3-phase: solved |V_LN| == basekv/sqrt(3); Bus.kVBase() is L-N."""
        _dss_clear()
        dss.Text.Command(
            "New Circuit.c3 phases=3 basekv=12.66 bus1=src.1.2.3 pu=1.0 angle=0.0 "
            f"frequency={_F0} r1=1e-9 x1=1e-9 r0=1e-9 x0=1e-9"
        )
        dss.Text.Command("Set voltagebases=[12.66]")
        dss.Text.Command("Calcvoltagebases")
        dss.Text.Command("Solve")
        dss.Circuit.SetActiveBus("src")
        volts = dss.Bus.Voltages()
        vmag = math.hypot(volts[0], volts[1])
        assert vmag == pytest.approx(12660.0 / math.sqrt(3.0))
        assert dss.Bus.kVBase() == pytest.approx(12.66 / math.sqrt(3.0))

    def test_single_phase_basekv_used_directly_ln_style(self) -> None:
        """1-phase, basekv given as a genuine L-N value: solved |V| == basekv exactly."""
        _dss_clear()
        dss.Text.Command(
            "New Circuit.c1ln phases=1 basekv=7.2 bus1=src.1 pu=1.0 angle=0.0 "
            f"frequency={_F0} r1=1e-9 x1=1e-9 r0=1e-9 x0=1e-9"
        )
        dss.Text.Command("Set voltagebases=[7.2]")
        dss.Text.Command("Calcvoltagebases")
        dss.Text.Command("Solve")
        dss.Circuit.SetActiveBus("src")
        volts = dss.Bus.Voltages()
        vmag = math.hypot(volts[0], volts[1])
        assert vmag == pytest.approx(7200.0)  # NOT 7200/sqrt(3) and NOT 7200*sqrt(3)

    def test_single_phase_basekv_used_directly_ll_style(self) -> None:
        """1-phase, basekv given as the parent system's L-L value (legacy positive-
        sequence-equivalent convention, e.g. IEEE 33-bus): solved |V| == basekv too."""
        _dss_clear()
        dss.Text.Command(
            "New Circuit.c1ll phases=1 basekv=12.66 bus1=src.1 pu=1.0 angle=0.0 "
            f"frequency={_F0} r1=1e-9 x1=1e-9 r0=1e-9 x0=1e-9"
        )
        dss.Text.Command("Set voltagebases=[12.66]")
        dss.Text.Command("Calcvoltagebases")
        dss.Text.Command("Solve")
        dss.Circuit.SetActiveBus("src")
        volts = dss.Bus.Voltages()
        vmag = math.hypot(volts[0], volts[1])
        assert vmag == pytest.approx(12660.0)  # basekv used as-is, no sqrt(3) anywhere


# ---------------------------------------------------------------------------
# Converter: u_ref_v / u_rated_v track the above unconditionally
# ---------------------------------------------------------------------------


class TestConverterSourceVoltageTracksOpenDSS:
    def test_genuine_single_phase_ln_source_converts_correctly(self) -> None:
        """A real single-phase (L-N) Vsource: u_ref_v matches the solved L-N EMF."""
        _dss_clear()
        dss.Text.Command(
            "New Circuit.cgen phases=1 basekv=7.2 bus1=src.1 pu=1.0 angle=0.0 "
            f"frequency={_F0} r1=1e-9 x1=1e-9 r0=1e-9 x0=1e-9"
        )
        dss.Text.Command("New Load.ld1 phases=1 bus1=src.1 kv=7.2 kw=10 kvar=3 model=1")
        dss.Text.Command("Set voltagebases=[7.2]")
        dss.Text.Command("Calcvoltagebases")
        dss.Text.Command("Solve")

        grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        src = next(a for a in grid.appliances if isinstance(a, Source))
        assert src.phases == (Phase.A,)
        # n<3 in build_source -> u_ref_v kept unchanged (no sqrt(3) division).
        assert src.u_ref_v[0] == pytest.approx(7200.0, rel=1e-6)

    def test_ieee33_style_single_phase_ll_source_converts_correctly(self) -> None:
        """A positive-sequence-equivalent 1-phase Vsource (basekv=parent L-L):
        u_ref_v recovers the same L-L number (matches pandapower's vn_kv*1000
        convention for the same circuit, per the existing IEEE-33 oracle)."""
        _dss_clear()
        dss.Text.Command(
            "New Circuit.cieee phases=1 basekv=12.66 bus1=src.1 pu=1.0 angle=0.0 "
            f"frequency={_F0} r1=1e-9 x1=1e-9 r0=1e-9 x0=1e-9"
        )
        dss.Text.Command(
            "New Load.ld1 phases=1 bus1=src.1 kv=12.66 kw=10 kvar=3 model=1"
        )
        dss.Text.Command("Set voltagebases=[12.66]")
        dss.Text.Command("Calcvoltagebases")
        dss.Text.Command("Solve")

        grid, id_map = to_grid(dss)  # default SINGLE_PHASE_EQUIV
        src = next(a for a in grid.appliances if isinstance(a, Source))
        assert src.u_ref_v[0] == pytest.approx(12660.0, rel=1e-6)
        node = grid.nodes[0]
        assert node.u_rated_v == pytest.approx(12660.0, rel=1e-6)

    def test_three_phase_source_divides_by_sqrt3_for_ln_emf(self) -> None:
        """A 3-phase Vsource: converted per-phase u_ref_v is the L-N EMF
        (u_ref_v = basekv/sqrt(3)*1000), matching the solved bus voltage."""
        _dss_clear()
        dss.Text.Command(
            "New Circuit.c3conv phases=3 basekv=6.6 bus1=src.1.2.3 pu=1.0 angle=0.0 "
            f"frequency={_F0} r1=1e-9 x1=1e-9 r0=1e-9 x0=1e-9"
        )
        dss.Text.Command(
            "New Load.ld1 phases=3 bus1=src.1.2.3 kv=6.6 kw=30 kvar=9 conn=wye model=1"
        )
        dss.Text.Command("Set voltagebases=[6.6]")
        dss.Text.Command("Calcvoltagebases")
        dss.Text.Command("Solve")

        grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        src = next(a for a in grid.appliances if isinstance(a, Source))
        assert src.phases == (Phase.A, Phase.B, Phase.C)
        expected_ln_v = 6600.0 / math.sqrt(3.0)
        for u in src.u_ref_v:
            assert u == pytest.approx(expected_ln_v, rel=1e-6)

        dss.Circuit.SetActiveBus("src")
        volts = dss.Bus.Voltages()
        vmag_ln = math.hypot(volts[0], volts[1])
        assert vmag_ln == pytest.approx(expected_ln_v, rel=1e-4)
