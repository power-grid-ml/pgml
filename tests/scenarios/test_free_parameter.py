"""The free harmonic parameter ``field="h_param"``.

A generator whose emission model needs more per-device, per-order quantities than a
magnitude and a phase declares them as ``h_param`` specs. They are drawn in the batch's
own cube and recorded, and the sampler writes them nowhere, so the model that owns them
can apply them to the sampled injection.
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from pgml.scenarios import (
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    sample,
)

ORDERS = [3, 5]


def _spec(name, field, low, high, **kw):
    kw.setdefault("mode", "absolute")
    return ParameterSpec(
        name=name,
        selector=Selector(component="load"),
        distribution=Uniform(low=low, high=high),
        field=field,
        orders=ORDERS,
        **kw,
    )


def _config(extra=(), seed=3):
    params = [_spec("hm", "h_mag", 0.0, 0.2), _spec("hp", "h_phase", -30.0, 30.0)]
    return ScenarioConfig(n_samples=16, seed=seed, parameters=[*params, *extra])


def test_a_free_parameter_is_recorded_and_written_nowhere(grid3):
    base = sample(grid3, _config())
    free = [_spec("shape", "h_param", 0.0, 1.0), _spec("tilt", "h_param", -5.0, 5.0)]
    s = sample(grid3, _config(free))
    # two free specs cover the same devices and orders without colliding
    for name, low, high in (("shape", 0.0, 1.0), ("tilt", -5.0, 5.0)):
        draw = s.samples[name]
        assert draw.shape == (16, 2, len(ORDERS))
        assert float(draw.min()) >= low and float(draw.max()) <= high
        assert f"{name}_mag" not in s.samples
        assert f"{name}_device_ids" not in s.shared_samples
    assert not torch.equal(s.samples["shape"], s.samples["tilt"] / 10.0 + 0.5)
    # the injection holds exactly the magnitude and phase draws
    assert set(s.harmonic_injection) == set(base.harmonic_injection)
    for cid, per_order in s.harmonic_injection.items():
        for order, (mag, phase) in per_order.items():
            j = [10, 11].index(cid)
            k = ORDERS.index(order)
            assert torch.equal(mag, s.samples["hm"][:, j, k])
            assert torch.equal(phase, s.samples["hp"][:, j, k])
    assert torch.equal(s.samples["hm_mag"], s.samples["hm"])


def test_a_free_parameter_shares_the_cube(grid3):
    """It takes sampling dimensions like any per-scenario harmonic draw; held draws
    (``per="fixed"`` / ``"class"``) take none and leave every other draw in place."""
    base = sample(grid3, _config())
    held = sample(grid3, _config([_spec("shape", "h_param", 0.0, 1.0, per="fixed")]))
    assert torch.equal(held.samples["hm"], base.samples["hm"])
    assert torch.equal(held.samples["hp"], base.samples["hp"])
    shape = held.samples["shape"]
    assert torch.equal(shape, shape[:1].expand_as(shape))
    cls = sample(grid3, _config([_spec("shape", "h_param", 0.0, 1.0, per="class")]))
    other = sample(
        grid3, _config([_spec("shape", "h_param", 0.0, 1.0, per="class")], seed=4)
    )
    assert cls.samples["shape"].shape == (16, 1, len(ORDERS))
    assert torch.equal(cls.samples["shape"], other.samples["shape"])


def test_a_free_parameter_is_validated():
    with pytest.raises(ValidationError, match="requires mode='absolute'"):
        _spec("shape", "h_param", 0.0, 1.0, mode="scale")
    with pytest.raises(ValidationError, match="harmonic_reference"):
        _spec("shape", "h_param", 0.0, 1.0, harmonic_reference="en50160")
    with pytest.raises(ValidationError):
        ParameterSpec(
            name="shape",
            selector=Selector(component="load"),
            distribution=Uniform(low=0.0, high=1.0),
            field="h_param",
            mode="absolute",
        )
    ok = _spec("shape", "h_param", 0.0, 1.0)
    assert ok.is_harmonic and ok.is_free_parameter
    assert not _spec("hm", "h_mag", 0.0, 1.0).is_free_parameter
