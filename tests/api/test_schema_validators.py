"""Schema construction validators: mis-sized matrices fail LOUD, at build time.

A wrong-dimension per-phase matrix must never survive to assembly (where it
would surface as an opaque shape error, or worse, a silent broadcast).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pgml.schemas import SCHEMA_VERSION
from pgml.schemas.grid_schema import (
    ComplexTap,
    GenericBranch,
    Phase,
    ShuntReactor,
    Source,
)

ABC = (Phase.A, Phase.B, Phase.C)


def test_schema_version_is_current():
    assert SCHEMA_VERSION == "0.2.0"


def test_shunt_reactor_rejects_mis_sized_matrix():
    with pytest.raises(ValidationError, match="3x3"):
        ShuntReactor(
            id=1,
            from_node=1,
            to_node=1,
            from_phases=ABC,
            to_phases=ABC,
            conductance_s=[[1.0e-4, 0.0], [0.0, 1.0e-4]],  # 2x2 on a 3-phase shunt
            capacitance_f=[
                [1.0e-7 if i == j else 0.0 for j in range(3)] for i in range(3)
            ],
        )


def test_generic_branch_rejects_mis_sized_matrix():
    ok = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    with pytest.raises(ValidationError, match="3x3"):
        GenericBranch(
            id=1,
            from_node=1,
            to_node=2,
            from_phases=ABC,
            to_phases=ABC,
            series_resistance_ohm=ok,
            series_inductance_h=[[1.0e-4]],  # 1x1 on a 3-phase branch
        )


def test_generic_branch_rejects_phase_length_mismatch():
    ok = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    with pytest.raises(ValidationError, match="equal length"):
        GenericBranch(
            id=1,
            from_node=1,
            to_node=2,
            from_phases=ABC,
            to_phases=(Phase.A,),
            series_resistance_ohm=ok,
            series_inductance_h=ok,
        )


def test_complex_tap_shift_deg_is_plain_float():
    """The clock selector is not tensor-capable: whatever goes in, the STORED
    value is a plain python float (pydantic coerces at construction), so no
    autograd graph can reach the discrete clock selection downstream."""
    torch = pytest.importorskip("torch")
    tap = ComplexTap(ratio_magnitude=1.0, shift_deg=30)
    assert isinstance(tap.shift_deg, float)
    coerced = ComplexTap(ratio_magnitude=1.0, shift_deg=torch.tensor(30.0))
    assert isinstance(coerced.shift_deg, float) and coerced.shift_deg == 30.0


# =============================================================================
# MeasurementDevice: instrumentation metadata validated at build / attach time
# =============================================================================
def test_measurement_device_interval_must_be_supported():
    from pgml.schemas.grid_schema import MeasurementDevice

    with pytest.raises(ValidationError, match="supported intervals"):
        MeasurementDevice(
            id=1,
            node=1,
            supported_averaging_intervals_s=[1.0, 600.0],
            averaging_interval_s=10.0,
        )
    dev = MeasurementDevice(
        id=1,
        node=1,
        supported_averaging_intervals_s=[1.0, 600.0],
        averaging_interval_s=600.0,
    )
    assert dev.averaging_interval_s == 600.0


def test_measurement_device_channel_capacity_and_quantities():
    from pgml.schemas.grid_schema import CurrentChannel, MeasurementDevice

    with pytest.raises(ValidationError, match="exceed"):
        MeasurementDevice(
            id=1,
            node=1,
            measured_quantities=("voltage", "current"),
            max_current_channels=1,
            current_channels=[CurrentChannel(branch=20), CurrentChannel(branch=21)],
        )
    with pytest.raises(ValidationError, match="measured_quantities"):
        # a current channel without 'current' among the measured quantities
        MeasurementDevice(id=1, node=1, current_channels=[CurrentChannel(branch=20)])
    with pytest.raises(ValidationError, match="distinct"):
        MeasurementDevice(
            id=1,
            node=1,
            measured_quantities=("current",),
            current_channels=[CurrentChannel(branch=20), CurrentChannel(branch=20)],
        )


def test_grid_validates_device_references():
    from pgml.schemas.grid_schema import CurrentChannel, MeasurementDevice

    from tests.fixtures.tiny_grids import single_phase_chain

    grid = single_phase_chain()
    # node 1 is incident to branch 20 only; branch 21 joins nodes 2-3
    with pytest.raises(ValidationError, match="not incident"):
        grid.attach_measurement_devices(
            [
                MeasurementDevice(
                    id=1,
                    node=1,
                    measured_quantities=("voltage", "current"),
                    current_channels=[CurrentChannel(branch=21)],
                )
            ]
        )
    with pytest.raises(ValidationError, match="missing branch"):
        grid.attach_measurement_devices(
            [
                MeasurementDevice(
                    id=1,
                    node=1,
                    measured_quantities=("current",),
                    current_channels=[CurrentChannel(branch=99)],
                )
            ]
        )
    with pytest.raises(ValidationError, match="missing node"):
        grid.attach_measurement_devices([MeasurementDevice(id=1, node=99)])
    with pytest.raises(ValidationError, match="'from' terminal"):
        grid.attach_measurement_devices(
            [
                MeasurementDevice(
                    id=1,
                    node=2,
                    measured_quantities=("current",),
                    # branch 20 is 1 -> 2: its 'from' terminal is at node 1, not 2
                    current_channels=[CurrentChannel(branch=20, terminal="from")],
                )
            ]
        )


def test_grid_attach_is_atomic_and_roundtrips():
    from pgml.schemas.grid_schema import CurrentChannel, Grid, MeasurementDevice

    from tests.fixtures.tiny_grids import single_phase_chain

    grid = single_phase_chain()
    ok = MeasurementDevice(
        id=1,
        node=2,
        measured_quantities=("voltage", "current"),
        current_channels=[CurrentChannel(branch=20), CurrentChannel(branch=21)],
        manufacturer="Janitza",
        model="UMG 604",
        accuracy_class="0.5S",
        max_harmonic_order=50,
        connection={"kind": "modbus_tcp", "host": "10.0.0.5", "port": 502},
    )
    grid.attach_measurement_devices([ok])
    assert len(grid.measurement_devices) == 1

    # a failing attach raises AND leaves the previous device list untouched
    with pytest.raises(ValidationError):
        grid.attach_measurement_devices([MeasurementDevice(id=2, node=99)])
    assert [d.id for d in grid.measurement_devices] == [1]

    # devices survive the JSON round-trip (dashboards / dataset provenance)
    restored = Grid.model_validate_json(grid.model_dump_json())
    dev = restored.measurement_devices[0]
    assert dev.connection == {"kind": "modbus_tcp", "host": "10.0.0.5", "port": 502}
    assert dev.accuracy_class == "0.5S"
    with pytest.raises(ValidationError, match="unique"):
        grid.attach_measurement_devices([ok])  # duplicate id 1


# ---------------------------------------------------------------------------
# Source: the legacy `spectrum` key
# ---------------------------------------------------------------------------
def _legacy_source_payload(**extra) -> dict:
    return {
        "component": "source",
        "id": 7,
        "node": 1,
        "phases": ["a"],
        "u_ref_v": [400.0],
        "u_angle_deg": [0.0],
        "resistance_ohm": [[1.0e-6]],
        "inductance_h": [[1.0e-12]],
        **extra,
    }


def test_source_has_no_spectrum_field():
    """Upstream distortion is an operating point (`NodeHarmonicSource`), not grid data."""
    assert "spectrum" not in Source.model_fields


def test_source_accepts_and_drops_a_null_legacy_spectrum():
    """Every grid persisted under an earlier revision still validates."""
    src = Source.model_validate(_legacy_source_payload(spectrum=None))
    assert "spectrum" not in src.model_dump()


def test_source_warns_when_dropping_a_populated_legacy_spectrum(caplog):
    payload = _legacy_source_payload(
        spectrum={
            "kind": "static",
            "spectrum": {
                "components": [{"order": 5, "magnitude_pu": 0.05, "phase_deg": 0.0}]
            },
        }
    )
    with caplog.at_level("WARNING"):
        src = Source.model_validate(payload)
    assert "spectrum" not in src.model_dump()
    assert any(
        "Source 7" in r.message and "NodeHarmonicSource" in r.message
        for r in caplog.records
    )


def test_source_still_rejects_an_unknown_field():
    """The migration is narrow: any OTHER unknown field is still a loud error."""
    with pytest.raises(ValidationError):
        Source.model_validate(_legacy_source_payload(not_a_field=1.0))


def test_source_accepts_keyword_construction_with_the_legacy_argument():
    """The migration also covers `Source(..., spectrum=None)` in code, not just JSON.

    Downstream loaders that still pass the removed argument keep working (the
    before-validator sees the keyword dict), so the removal needs no lockstep release.
    """
    src = Source(
        component="source",
        id=9,
        node=1,
        phases=("a",),
        u_ref_v=(400.0,),
        u_angle_deg=(0.0,),
        resistance_ohm=[[1.0e-6]],
        inductance_h=[[1.0e-12]],
        spectrum=None,
    )
    assert "spectrum" not in src.model_dump()
