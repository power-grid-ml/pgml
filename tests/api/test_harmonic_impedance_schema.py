"""The DER impedance contract preserves physical data and tensor identities."""

import pytest
import torch
from pydantic import ValidationError
from pgml.schemas import Generator, HarmonicImpedance, Phase, Storage


@pytest.mark.parametrize("kind", [Generator, Storage])
def test_optional_impedance_preserves_older_device_json(kind):
    device = kind(id=1, node=1, phases=(Phase.A,), p_nom_w=100)
    payload = device.model_dump(mode="json")
    payload.pop("harmonic_impedance")
    assert kind.model_validate(payload).harmonic_impedance is None


@pytest.mark.parametrize(
    "values",
    [
        {"resistance_ohm": -1},
        {"inductance_h": float("nan")},
        {"resistance_ohm": float("inf")},
        {"resistance_ohm": [1, -2]},
        {"resistance_ohm": []},
        {},
        {"resistance_ohm": [1, 0], "inductance_h": [0, 0]},
        {"resistance_ohm": [1, 2], "inductance_h": [1, 2, 3]},
    ],
)
def test_invalid_passive_impedance_is_rejected(values):
    with pytest.raises(ValidationError):
        HarmonicImpedance(**values)


def test_scalar_and_connection_element_values_round_trip():
    impedance = HarmonicImpedance(
        resistance_ohm=[1, 2, 3],
        inductance_h=0.001,
        spectrum_reference="opendss_voltage",
        frequency_model="opendss_admittance",
    )
    restored = HarmonicImpedance.model_validate_json(impedance.model_dump_json())
    assert restored == impedance


def test_tensor_identity_and_gradient_survive_schema():
    r = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    impedance = HarmonicImpedance(resistance_ohm=r, inductance_h=0.001)
    assert impedance.resistance_ohm is r
    impedance.resistance_ohm.square().sum().backward()
    torch.testing.assert_close(r.grad, 2 * r)
