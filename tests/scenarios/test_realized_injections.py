"""Realized harmonic injections are persisted, not just the draw that produced them.

The recorded draw of a randomized ``h_mag`` spec is a FRACTION of a per-device emission
reference, so the magnitude that reaches the solver only exists once that reference has
been applied. These tests pin the realized ``<spec>_mag`` / ``<spec>_phase`` columns, the
device-id axis they sit on, their post-reference and post-law values, and their dataset
round-trip.
"""

from __future__ import annotations

import math
import tempfile

import torch

from pgml.scenarios import (
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    en50160_limit,
    iec61000_3_2_device_caps,
    read_dataset,
    run_scenarios,
    sample,
    write_dataset,
)

W = 2.0 * math.pi * 50.0


# =============================================================================
# Randomized specs (`pgml.scenarios.sampler`)
# =============================================================================
def _hcfg(n=16, **kw) -> ScenarioConfig:
    kw.setdefault("name", "hm")
    kw.setdefault("selector", Selector(component="load"))
    kw.setdefault("distribution", Uniform(low=2.0, high=2.0))
    kw.setdefault("field", "h_mag")
    kw.setdefault("mode", "absolute")
    kw.setdefault("orders", [3, 5])
    return ScenarioConfig(n_samples=n, seed=0, parameters=[ParameterSpec(**kw)])


def test_realized_magnitude_is_recorded_post_reference(grid3):
    """The draw is a fraction of the reference; the realized column is what was injected."""
    s = sample(grid3, _hcfg(harmonic_reference="en50160"))

    assert s.samples["hm"].shape == (16, 2, 2)  # the raw draw, unchanged
    assert torch.allclose(s.samples["hm"], torch.full_like(s.samples["hm"], 2.0))
    mag = s.samples["hm_mag"]
    assert mag.shape == (16, 2, 2)  # [B, n_dev, n_ord]
    assert s.all_samples["hm_device_ids"].tolist() == [10, 11]
    for j, cid in enumerate([10, 11]):
        for o, order in enumerate([3, 5]):
            realized = 2.0 * en50160_limit(order)
            column = mag[:, j, o]
            assert torch.allclose(column, torch.full_like(column, realized))
            # and it IS what the solver receives
            torch.testing.assert_close(
                mag[:, j, o], s.harmonic_injection[cid][order][0]
            )


def test_realized_magnitude_is_per_device_under_a_shared_draw(grid3):
    """One shared draw, two devices: the IEC reference is per device, so are the values."""
    s = sample(grid3, _hcfg(per="shared", harmonic_reference="iec61000-3-2"))

    assert s.samples["hm"].shape == (16, 1, 2)  # ONE draw is recorded
    mag = s.samples["hm_mag"]
    assert mag.shape == (16, 2, 2)  # realized per device
    caps = iec61000_3_2_device_caps(grid3, [10, 11], [3, 5], emission_class="auto")
    for j, cid in enumerate([10, 11]):
        for o, order in enumerate([3, 5]):
            column = mag[:, j, o]
            assert torch.allclose(
                column, torch.full_like(column, 2.0 * caps[cid][order])
            )
    # the household (2 kW) and the EV charger (3 kW) do NOT share a realized magnitude
    assert not torch.allclose(mag[:, 0, :], mag[:, 1, :])


def test_realized_phase_follows_the_phase_spec(grid3):
    """The recorded phase is the one the injection carries, whoever wrote it."""
    config = ScenarioConfig(
        n_samples=8,
        seed=0,
        parameters=[
            ParameterSpec(
                name="hm",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.0, high=1.0),
                field="h_mag",
                mode="absolute",
                orders=[3],
            ),
            ParameterSpec(
                name="hp",
                selector=Selector(component="load"),
                distribution=Uniform(low=-30.0, high=30.0),
                field="h_phase",
                mode="absolute",
                orders=[3],
            ),
        ],
    )

    s = sample(grid3, config)

    torch.testing.assert_close(s.samples["hm_phase"][:, :, 0], s.samples["hp"][:, :, 0])


def test_realized_phase_falls_back_to_the_stored_spectrum(grid_spectrum):
    """An order the config gives no phase keeps the device's stored angle, per scenario."""
    s = sample(grid_spectrum, _hcfg(n=8, orders=[5], mode="scale"))

    phase = s.samples["hm_phase"]
    assert phase.shape == (8, 1, 1)
    assert torch.allclose(phase, torch.full_like(phase, 10.0))  # stored h5 angle [deg]


def test_recording_keeps_the_coefficient_on_the_autograd_tape():
    """Recording must not detach: an injection coefficient may be a tracked tensor."""
    from pgml.scenarios.sampler import _coefficient_column

    like = torch.zeros(4, dtype=torch.float64)
    per_scenario = torch.tensor(
        [0.1, 0.2, 0.3, 0.4], dtype=torch.float64, requires_grad=True
    )
    scalar = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

    _coefficient_column(per_scenario, 4, like).sum().backward()
    broadcast = _coefficient_column(scalar, 4, like)
    broadcast.sum().backward()

    assert broadcast.shape == (4,)  # a scalar coefficient spans the batch
    torch.testing.assert_close(per_scenario.grad, torch.ones(4, dtype=torch.float64))
    assert float(scalar.grad) == 4.0


def test_realized_columns_survive_a_dataset_round_trip(grid3):
    """The audit trail must be on disk, not only in memory."""
    result = run_scenarios(
        grid3,
        _hcfg(n=8, harmonic_reference="en50160"),
        calculation="harmonic",
        harmonic_orders=[1, 3, 5],
        dtype=torch.complex128,
    )
    with tempfile.TemporaryDirectory() as tmp:
        loaded = read_dataset(write_dataset(result, tmp))

    for key in ("hm", "hm_mag", "hm_phase", "hm_device_ids"):
        torch.testing.assert_close(loaded.samples[key], result.sampled.all_samples[key])
