"""Realized harmonic injections are persisted, not just the draw that produced them.

The recorded draw of a randomized ``h_mag`` spec is a FRACTION of a per-device emission
reference, and a composed aggregate spectrum used to leave no sample column at all — so a
written dataset could not be audited without re-deriving the reference table or
reconstructing ``I(h) = Y(h)*V(h)`` from the state. These tests pin the realized columns
of all three generating paths (randomized specs, coherent fingerprint, device
composition), their post-reference / post-cap values, and their dataset round-trip.
"""

from __future__ import annotations

import math
import tempfile

import torch

from pgml.schemas.grid_schema import (
    Grid,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
)
from pgml.scenarios import (
    ClassCount,
    CoherentSpectrumConfig,
    CompositionConfig,
    ConsumerComposition,
    DeviceClassSpec,
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    en50160_limit,
    iec61000_3_2_device_caps,
    read_dataset,
    run_scenarios,
    sample,
    sample_coherent_spectra,
    sample_device_composition,
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
    assert s.samples["hm_device_ids"].tolist() == [10, 11]
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
        torch.testing.assert_close(loaded.samples[key], result.sampled.samples[key])


# =============================================================================
# Device composition (`pgml.scenarios.composition`)
# =============================================================================
def _line(bid, a, b):
    return Line(
        id=bid,
        from_node=a,
        to_node=b,
        from_phases=(Phase.A,),
        to_phases=(Phase.A,),
        length_m=1.0,
        series_resistance_ohm_per_m=[[0.5]],
        series_inductance_h_per_m=[[0.5 / W]],
        shunt_capacitance_f_per_m=[[0.0]],
    )


def _one_load_grid(p_nom=3000.0) -> Grid:
    """Single-phase 2-bus grid: source@1, one aggregated load@2 (id 10)."""
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=230.0, phases=(Phase.A,)),
            Node(id=2, u_rated_v=230.0, phases=(Phase.A,)),
        ],
        branches=[_line(1, 1, 2)],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=(Phase.A,),
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[0.1 / W]],
            ),
            Load(
                id=10,
                node=2,
                phases=(Phase.A,),
                p_nom_w=p_nom,
                q_nom_var=0.0,
                load_model=LoadModel.CONST_POWER,
            ),
        ],
    )


def _steady_class(name, *, mag, phase_deg=0.0, rated=1000.0, sign=1) -> DeviceClassSpec:
    """An always-on, fully-loaded member: no activity, loading or law dynamics left."""
    return DeviceClassSpec(
        name=name,
        sign=sign,
        emission_class="D",
        rated_power_w=(rated, rated),
        harmonic_magnitude={5: (mag, mag)},
        harmonic_phase_deg={5: (phase_deg, phase_deg)},
        gamma=(0.0, 0.0),
        activity_preset="flat",
        discrete_activity=False,
        loading_min=1.0,
        loading_mean=(1.0, 1.0),
        loading_jitter=0.0,
    )


def _composed_cfg(classes, **kw) -> CoherentSpectrumConfig:
    comp = CompositionConfig(
        classes=classes,
        compositions=[
            ConsumerComposition(
                classes=[ClassCount(class_name=c.name) for c in classes]
            )
        ],
        scale_to_nominal=False,
        behavioral_coupling=0.0,
        cloud_coupling=0.0,
        **kw,
    )
    return CoherentSpectrumConfig(
        selector=Selector(component="load"),
        orders=[5],
        n_steps=4,
        n_scenarios=3,
        seed=0,
        step_size_s=3600.0,
        composition=comp,
        start_time="2024-06-21T12:00:00",
    )


def test_composed_aggregate_spectrum_is_recorded():
    """One member, no dynamics: the aggregate IS that member's drawn ratio and angle."""
    grid = _one_load_grid()
    config = _composed_cfg([_steady_class("c", mag=0.3, phase_deg=25.0)])

    draw = sample_device_composition(grid, config)

    mag = draw.samples["harmonics_composed_mag"]
    phase = draw.samples["harmonics_composed_phase"]
    assert mag.shape == (3, 1, 1, 4)  # [B, n_agg, n_ord, T]
    assert draw.samples["harmonics_agg_ids"].tolist() == [10]
    torch.testing.assert_close(mag, torch.full_like(mag, 0.3))
    torch.testing.assert_close(phase, torch.full_like(phase, 25.0))
    # and the columns are exactly the injection the solver is handed
    torch.testing.assert_close(mag[:, 0, 0, :], draw.harmonic_injection[10][5][0])
    torch.testing.assert_close(phase[:, 0, 0, :], draw.harmonic_injection[10][5][1])


def test_composed_aggregate_is_the_power_weighted_member_sum():
    """Two in-phase members: the aggregate ratio is their power-weighted mean."""
    grid = _one_load_grid()
    config = _composed_cfg(
        [
            _steady_class("strong", mag=0.4, rated=1000.0),
            _steady_class("weak", mag=0.1, rated=3000.0),
        ]
    )

    draw = sample_device_composition(grid, config)

    power = draw.samples["harmonics_class_p_w"][:, 0, :, :]  # [B, n_class, T]
    expected = (0.4 * power[:, 0, :] + 0.1 * power[:, 1, :]) / power.sum(dim=1)
    torch.testing.assert_close(
        draw.samples["harmonics_composed_mag"][:, 0, 0, :], expected
    )
    # 1 kW at 40 % + 3 kW at 10 % -> 17.5 % of the 4 kW aggregate
    torch.testing.assert_close(expected, torch.full_like(expected, 0.175))


def test_composed_records_the_capped_magnitude():
    """A near-cancelling PV member drives the cap; the CAPPED value is recorded."""
    grid = _one_load_grid()
    config = _composed_cfg(
        [
            _steady_class("load", mag=0.2, rated=3000.0),
            _steady_class("pv", mag=0.0, rated=2850.0, sign=-1),
        ],
        max_injection_pu=3.0,
    )

    draw = sample_device_composition(grid, config)

    binding = draw.samples["harmonics_cap_binding"]
    mag = draw.samples["harmonics_composed_mag"]
    assert float(binding.max()) == 1.0
    assert float(mag.max()) <= 3.0
    bound = mag[binding > 0.0]
    torch.testing.assert_close(bound, torch.full_like(bound, 3.0))


def test_composed_columns_survive_a_dataset_round_trip():
    grid = _one_load_grid()
    config = _composed_cfg([_steady_class("c", mag=0.3, phase_deg=25.0)])
    result = run_scenarios(grid, config, dtype=torch.complex128)

    with tempfile.TemporaryDirectory() as tmp:
        loaded = read_dataset(write_dataset(result, tmp))

    for key in ("harmonics_composed_mag", "harmonics_composed_phase"):
        torch.testing.assert_close(loaded.samples[key], result.sampled.samples[key])


# =============================================================================
# Coherent fingerprint (`pgml.scenarios.harmonics`) — one naming convention
# =============================================================================
def test_every_path_records_a_mag_phase_pair_on_a_device_axis():
    """A consumer reads one convention: <key>_mag / <key>_phase + the id column."""
    grid = _one_load_grid()

    fingerprint = sample_coherent_spectra(
        grid,
        CoherentSpectrumConfig(
            selector=Selector(component="load"),
            orders=[5],
            n_steps=4,
            n_scenarios=3,
            seed=0,
        ),
    )
    composed = sample_device_composition(
        grid, _composed_cfg([_steady_class("c", mag=0.3)])
    )

    for samples, key, ids in (
        (fingerprint.samples, "harmonics", "harmonics_device_ids"),
        (composed.samples, "harmonics_composed", "harmonics_agg_ids"),
    ):
        mag, phase = samples[f"{key}_mag"], samples[f"{key}_phase"]
        assert mag.shape == phase.shape
        assert mag.shape[1] == samples[ids].numel()  # [B, n_dev, n_ord, T]
        assert mag.shape[2] == 1 and mag.shape[-1] == 4
