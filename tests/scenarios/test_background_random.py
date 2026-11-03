"""The upstream harmonic background on a snapshot batch.

``BackgroundHarmonicConfig`` is the one way to express supply-side distortion: it is
realized as a ``NodeHarmonicSource`` at each source node, shared by every device on the
feeder. A snapshot batch has no step axis to walk, so every scenario draws its own level,
the spectrum tensors line up with the per-device ``[B]`` injections, and an unconfigured
background leaves the batch byte-identical. The drift either moves every order together
(``drift_mode="shared"``, the default, unchanged in values and serialized form) or gives
each order's magnitude and angle its own series with the same marginal law
(``"independent"``).
"""

import pytest
import torch

from pgml.scenarios import (
    BackgroundHarmonicConfig,
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    build_background_sources,
    run_scenarios,
    sample,
)

CDT = torch.complex128


def _cfg(n, seed, background=None, *, device_emission=False):
    """A plain snapshot batch varying load power, optionally with an upstream background.

    ``device_emission`` adds a per-device harmonic draw at order 5, so a run has harmonic
    content of its own to compare the upstream background against.
    """
    parameters = [
        ParameterSpec(
            name="load_pq",
            selector=Selector(component="load"),
            distribution=Uniform(low=0.5, high=1.5),
            field="pq",
            mode="scale",
        )
    ]
    if device_emission:
        parameters.append(
            ParameterSpec(
                name="load_h5",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.02, high=0.06),
                field="h_mag",
                mode="absolute",
                orders=[5],
            )
        )
    return ScenarioConfig(
        n_samples=n, seed=seed, parameters=parameters, background=background
    )


def test_background_is_batched_per_scenario_and_inert_when_unset(grid3):
    plain = sample(grid3, _cfg(6, 4))
    assert plain.node_sources == []

    empty = sample(grid3, _cfg(6, 4, BackgroundHarmonicConfig()))
    assert empty.node_sources == [], "no configured order -> no source"
    assert torch.equal(empty.samples["load_pq"], plain.samples["load_pq"]), (
        "an empty background consumes no randomness"
    )

    bg = BackgroundHarmonicConfig(magnitude_pu={3: 0.02, 5: 0.01}, drift_std=0.5)
    got = sample(grid3, _cfg(6, 4, bg))
    assert len(got.node_sources) == 1, "one source per in-service Source node"
    src = got.node_sources[0]
    assert set(src.spectrum) == {3, 5}
    for mag, ang in src.spectrum.values():
        assert tuple(mag.shape) == (6,) and tuple(ang.shape) == (6,), (
            "snapshot batches carry a plain [B] spectrum, no step axis"
        )
    mag3 = src.spectrum[3][0]
    assert mag3.std() > 0, "each scenario draws its own background level"
    again = sample(grid3, _cfg(6, 4, bg)).node_sources[0]
    assert torch.equal(again.spectrum[3][0], mag3), "seeded -> reproducible"


def test_background_reaches_the_solve_and_keeps_the_fundamental(grid3):
    cfg = _cfg(5, 9, device_emission=True)
    orders = [1, 3, 5]
    plain = run_scenarios(
        grid3, cfg, calculation="harmonic", harmonic_orders=orders, dtype=CDT
    )
    bg = _cfg(
        5,
        9,
        BackgroundHarmonicConfig(magnitude_pu={3: 0.03}, drift_std=0.3),
        device_emission=True,
    )
    got = run_scenarios(
        grid3, bg, calculation="harmonic", harmonic_orders=orders, dtype=CDT
    )
    assert torch.allclose(got.v[:, 0], plain.v[:, 0], atol=1e-9), (
        "the background acts only at h > 1"
    )
    d3 = (got.v[:, 1] - plain.v[:, 1]).abs()
    assert d3.min() > 0, (
        "every node sees the upstream background at the configured order"
    )
    # An order WITHOUT a configured level still sees the source's stiffness: the Thevenin
    # background is the upstream network, which holds the coupling point near its own
    # (here zero) level at that order rather than leaving it to float.
    assert (got.v[:, 2, 0].abs() < plain.v[:, 2, 0].abs()).all(), (
        "the stiff upstream pins the source node at an order with no background level"
    )


def test_build_background_sources_takes_the_requested_batch_shape(grid3):
    """The builder is the public entry point for a generator that owns its own step axis."""
    bg = BackgroundHarmonicConfig(
        magnitude_pu={3: 0.02}, phase_deg={3: 30.0}, drift_std=0.4, drift_rho=0.9
    )
    gen = torch.Generator().manual_seed(17)
    sources = build_background_sources(grid3, bg, (3, 5), gen)
    assert len(sources) == 1
    mag, ang = sources[0].spectrum[3]
    assert tuple(mag.shape) == (3, 5) and tuple(ang.shape) == (3, 5)
    # the drift walks ALONG the step axis, so neighbouring steps are closer than the span
    steps = mag[0]
    assert (steps[1:] - steps[:-1]).abs().max() < (steps.max() - steps.min()) * 1.01
    assert torch.equal(
        build_background_sources(grid3, bg, (3, 5), torch.Generator().manual_seed(17))[
            0
        ].spectrum[3][0],
        mag,
    ), "seeded -> reproducible"


def test_an_empty_background_builds_no_source(grid3):
    assert (
        build_background_sources(
            grid3, BackgroundHarmonicConfig(), (2, 1), torch.Generator().manual_seed(0)
        )
        == []
    )


def test_background_drift_is_the_public_ar1_process():
    """The drift is ``ar1_noise`` on the caller's generator, not a second recurrence."""
    from pgml.grids import synthetic_feeder
    from pgml.scenarios import ar1_noise

    grid = synthetic_feeder(4)
    bg = BackgroundHarmonicConfig(
        magnitude_pu={5: 0.01}, drift_std=0.3, drift_rho=0.9, drift_phase_deg=10.0
    )
    sources = build_background_sources(
        grid, bg, (3, 6), torch.Generator().manual_seed(11)
    )
    drift = ar1_noise((3, 6), 0.9, torch.Generator().manual_seed(11))
    mag, phase = sources[0].spectrum[5]
    assert torch.equal(mag, 0.01 * torch.exp(0.3 * drift))
    assert torch.equal(phase, 0.0 + 10.0 * drift)


# --- drift_mode -----------------------------------------------------------------------
_ORDERS = {3: 0.02, 5: 0.01, 7: 0.005}


def _unit_drifts(sources, bg: BackgroundHarmonicConfig) -> tuple[torch.Tensor, ...]:
    """The unit drift series ``([H, *shape], [H, *shape])`` of magnitude and angle."""
    spectrum = sources[0].spectrum
    orders = sorted(spectrum)
    mag = torch.stack(
        [torch.log(spectrum[o][0] / bg.magnitude_pu[o]) / bg.drift_std for o in orders]
    )
    ang = torch.stack(
        [
            (spectrum[o][1] - bg.phase_deg.get(o, 0.0)) / bg.drift_phase_deg
            for o in orders
        ]
    )
    return mag, ang


def test_shared_drift_mode_is_the_default_and_moves_every_order_together():
    """The default realizes the same tensors as before the mode existed."""
    from pgml.grids import synthetic_feeder
    from pgml.scenarios import ar1_noise

    grid = synthetic_feeder(4)
    kw = dict(
        magnitude_pu=_ORDERS,
        phase_deg={5: -40.0},
        drift_std=0.3,
        drift_rho=0.9,
        drift_phase_deg=10.0,
    )
    default = build_background_sources(
        grid, BackgroundHarmonicConfig(**kw), (3, 6), torch.Generator().manual_seed(5)
    )
    shared = build_background_sources(
        grid,
        BackgroundHarmonicConfig(**kw, drift_mode="shared"),
        (3, 6),
        torch.Generator().manual_seed(5),
    )
    drift = ar1_noise((3, 6), 0.9, torch.Generator().manual_seed(5))
    for order, level in _ORDERS.items():
        mag, ang = default[0].spectrum[order]
        assert torch.equal(mag, level * torch.exp(0.3 * drift))
        assert torch.equal(ang, {5: -40.0}.get(order, 0.0) + 10.0 * drift)
        assert torch.equal(mag, shared[0].spectrum[order][0])
        assert torch.equal(ang, shared[0].spectrum[order][1])


def test_shared_drift_mode_keeps_the_serialized_config_and_its_hash():
    from pgml.scenarios import config_hash

    bg = BackgroundHarmonicConfig(magnitude_pu={5: 0.01}, drift_std=0.2)
    assert "drift_mode" not in bg.model_dump_json()
    assert "drift_mode" not in bg.model_dump()
    explicit = BackgroundHarmonicConfig(
        magnitude_pu={5: 0.01}, drift_std=0.2, drift_mode="shared"
    )
    assert config_hash(explicit) == config_hash(bg)
    independent = bg.model_copy(update={"drift_mode": "independent"})
    payload = independent.model_dump_json()
    assert '"drift_mode":"independent"' in payload
    assert BackgroundHarmonicConfig.model_validate_json(payload) == independent
    assert config_hash(independent) != config_hash(bg)


def test_independent_drift_is_one_public_ar1_draw_per_order_and_quantity():
    from pgml.grids import synthetic_feeder
    from pgml.scenarios import ar1_noise

    bg = BackgroundHarmonicConfig(
        magnitude_pu=_ORDERS,
        phase_deg={3: 25.0},
        drift_std=0.3,
        drift_rho=0.8,
        drift_phase_deg=12.0,
        drift_mode="independent",
    )
    sources = build_background_sources(
        synthetic_feeder(4), bg, (2, 7), torch.Generator().manual_seed(23)
    )
    noise = ar1_noise((2, 3, 2, 7), 0.8, torch.Generator().manual_seed(23))
    for k, (order, level) in enumerate(sorted(_ORDERS.items())):
        mag, ang = sources[0].spectrum[order]
        assert tuple(mag.shape) == tuple(ang.shape) == (2, 7)
        assert torch.equal(mag, level * torch.exp(0.3 * noise[0, k]))
        assert torch.equal(ang, {3: 25.0}.get(order, 0.0) + 12.0 * noise[1, k])


def test_independent_drift_decouples_orders_and_angle_but_keeps_the_marginal():
    """Cross-order and magnitude-angle correlation vanish; the per-order law does not move.

    8000 sequences of 20 steps at ``rho = 0.9``: two independent AR(1) series of that
    length correlate with a standard error of about 0.011 over the pooled samples, and a
    series' pooled mean has one of about 0.015. Measured at this seed: largest absolute
    cross-correlation 0.013, means within 0.02 of 0, stds 0.993-1.007, lag-1
    autocorrelations 0.899-0.902 (shared mode: 1.003 and 0.900).
    """
    from pgml.grids import synthetic_feeder

    grid = synthetic_feeder(4)
    rho = 0.9
    kw = dict(magnitude_pu=_ORDERS, drift_std=0.2, drift_phase_deg=10.0, drift_rho=rho)
    shape = (8000, 20)

    def draw(mode):
        bg = BackgroundHarmonicConfig(**kw, drift_mode=mode)
        sources = build_background_sources(
            grid, bg, shape, torch.Generator().manual_seed(31)
        )
        return _unit_drifts(sources, bg)

    def corr(x, y):
        return float(torch.corrcoef(torch.stack([x.reshape(-1), y.reshape(-1)]))[0, 1])

    def lag1(x):
        return corr(x[..., 1:], x[..., :-1])

    mag, ang = draw("independent")
    n = mag.shape[0]
    for i in range(n):
        assert abs(corr(mag[i], ang[i])) < 0.05, "magnitude and angle are decoupled"
        for j in range(i + 1, n):
            assert abs(corr(mag[i], mag[j])) < 0.05, "orders drift independently"
            assert abs(corr(ang[i], ang[j])) < 0.05
            assert abs(corr(mag[i], ang[j])) < 0.05

    shared_mag, shared_ang = draw("shared")
    assert corr(shared_mag[0], shared_mag[2]) == pytest.approx(1.0)
    assert corr(shared_mag[0], shared_ang[1]) == pytest.approx(1.0)
    for series in (*mag, *ang, shared_mag[0]):
        assert float(series.mean()) == pytest.approx(0.0, abs=0.05)
        assert float(series.std()) == pytest.approx(1.0, abs=0.04)
        assert lag1(series) == pytest.approx(rho, abs=0.03)


def test_independent_drift_on_a_snapshot_batch_is_per_scenario(grid3):
    bg = BackgroundHarmonicConfig(
        magnitude_pu={3: 0.02, 5: 0.01},
        drift_std=0.5,
        drift_phase_deg=5.0,
        drift_mode="independent",
    )
    src = sample(grid3, _cfg(6, 4, bg)).node_sources[0]
    for mag, ang in src.spectrum.values():
        assert tuple(mag.shape) == tuple(ang.shape) == (6,)
    assert not torch.allclose(
        torch.log(src.spectrum[3][0] / 0.02), torch.log(src.spectrum[5][0] / 0.01)
    ), "the orders no longer share one level path"
