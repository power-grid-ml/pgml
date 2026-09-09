"""The upstream harmonic background on the RANDOM (snapshot) recipe.

The coherent recipe carries ``BackgroundHarmonicConfig`` already; the snapshot recipe has
to realise the same source so a Task-A dataset can be generated with the upstream
distortion a real feeder sits behind. Every scenario draws its own level (no step axis to
walk), the spectrum tensors line up with the per-device ``[B]`` injections, and an
unconfigured background leaves the batch byte-identical.
"""

import torch

from pgml.scenarios import (
    BackgroundHarmonicConfig,
    run_scenarios,
    sample,
    se_random_scenario_config,
)

CDT = torch.complex128


def _cfg(grid, n, seed, background=None):
    return se_random_scenario_config(
        grid, orders=[3, 5], n_samples=n, seed=seed, background=background
    )


def test_random_background_is_batched_per_scenario_and_inert_when_unset(grid3):
    plain = sample(grid3, _cfg(grid3, 6, 4))
    assert plain.node_sources == []

    empty = sample(grid3, _cfg(grid3, 6, 4, BackgroundHarmonicConfig()))
    assert empty.node_sources == [], "no configured order -> no source"

    bg = BackgroundHarmonicConfig(magnitude_pu={3: 0.02, 5: 0.01}, drift_std=0.5)
    got = sample(grid3, _cfg(grid3, 6, 4, bg))
    assert len(got.node_sources) == 1, "one source per in-service Source node"
    src = got.node_sources[0]
    assert set(src.spectrum) == {3, 5}
    for mag, ang in src.spectrum.values():
        assert tuple(mag.shape) == (6,) and tuple(ang.shape) == (6,), (
            "snapshot batches carry a plain [B] spectrum, no step axis"
        )
    mag3 = src.spectrum[3][0]
    assert mag3.std() > 0, "each scenario draws its own background level"
    again = sample(grid3, _cfg(grid3, 6, 4, bg)).node_sources[0]
    assert torch.equal(again.spectrum[3][0], mag3), "seeded -> reproducible"


def test_random_background_reaches_the_solve_and_keeps_the_fundamental(grid3):
    cfg = _cfg(grid3, 5, 9)
    orders = [1, 3, 5]
    plain = run_scenarios(
        grid3, cfg, calculation="harmonic", harmonic_orders=orders, dtype=CDT
    )
    bg = _cfg(
        grid3, 5, 9, BackgroundHarmonicConfig(magnitude_pu={3: 0.03}, drift_std=0.3)
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
