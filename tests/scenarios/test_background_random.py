"""The upstream harmonic background on a snapshot batch.

``BackgroundHarmonicConfig`` is the one way to express supply-side distortion: it is
realized as a ``NodeHarmonicSource`` at each source node, shared by every device on the
feeder. A snapshot batch has no step axis to walk, so every scenario draws its own level,
the spectrum tensors line up with the per-device ``[B]`` injections, and an unconfigured
background leaves the batch byte-identical.
"""

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
