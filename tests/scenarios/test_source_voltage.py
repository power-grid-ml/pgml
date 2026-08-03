"""Fundamental operating-point variation: the source ``u_ref`` scale + coherent parameters.

Covers the slack-voltage scenario spec (``field="u_ref"``, ``component="source"``) reaching
the solved fundamental as a batched ideal-slack boundary, the coherent generator's
per-scenario ``parameters`` (drawn once per scenario, constant across the T steps), the RNG
stream separation that keeps the harmonic fingerprint reproducible, and the persistence
round-trip of the new fields.
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from pgml.assembly import node_phase_index
from pgml.schemas.grid_schema import Phase
from pgml.scenarios import (
    CoherentSpectrumConfig,
    Constant,
    Normal,
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    read_dataset,
    run_scenarios,
    sample_coherent_spectra,
    write_dataset,
)


# =============================================================================
# config validation
# =============================================================================
def test_source_selector_resolves_source(grid3):
    assert Selector(component="source").resolve(grid3) == [1]
    # a consumer_type filter matches no source (a source carries none)
    assert Selector(component="source", consumer_type="pv").resolve(grid3) == []


def test_uref_spec_requires_source_scale_balanced():
    ok = ParameterSpec(
        name="s",
        selector=Selector(component="source"),
        distribution=Normal(loc=1.0, scale=0.03),
        field="u_ref",
        mode="scale",
        per="shared",
    )
    assert ok.is_source_voltage and not ok.is_harmonic

    # u_ref requires component='source'
    with pytest.raises(ValidationError):
        ParameterSpec(
            name="s",
            selector=Selector(component="load"),
            distribution=Uniform(low=0.9, high=1.1),
            field="u_ref",
        )
    # u_ref rejects mode='absolute'
    with pytest.raises(ValidationError):
        ParameterSpec(
            name="s",
            selector=Selector(component="source"),
            distribution=Uniform(low=0.9, high=1.1),
            field="u_ref",
            mode="absolute",
        )
    # u_ref rejects per-phase symmetry + harmonic options
    with pytest.raises(ValidationError):
        ParameterSpec(
            name="s",
            selector=Selector(component="source"),
            distribution=Uniform(low=0.9, high=1.1),
            field="u_ref",
            symmetry="independent",
        )
    with pytest.raises(ValidationError):
        ParameterSpec(
            name="s",
            selector=Selector(component="source"),
            distribution=Uniform(low=0.9, high=1.1),
            field="u_ref",
            orders=[3],
        )
    # a source selector may ONLY vary u_ref (not power)
    with pytest.raises(ValidationError):
        ParameterSpec(
            name="p",
            selector=Selector(component="source"),
            distribution=Uniform(low=0.9, high=1.1),
            field="pq",
        )


def test_coherent_rejects_harmonic_parameters():
    with pytest.raises(ValidationError):
        CoherentSpectrumConfig(
            selector=Selector(component="load"),
            orders=[3],
            n_steps=2,
            parameters=[
                ParameterSpec(
                    name="h",
                    selector=Selector(component="load"),
                    distribution=Uniform(low=0.0, high=1.0),
                    field="h_mag",
                    orders=[3],
                )
            ],
        )


# =============================================================================
# (a) the source u_ref scale reaches the solved slack + non-slack voltages
# =============================================================================
def test_source_uref_scale_batched_slack(grid3):
    cfg = ScenarioConfig(
        n_samples=5,
        seed=1,
        method="independent",
        parameters=[
            ParameterSpec(
                name="src",
                selector=Selector(component="source"),
                distribution=Uniform(low=0.9, high=1.1),
                field="u_ref",
                mode="scale",
                per="shared",
            )
        ],
    )
    res = run_scenarios(
        grid3, cfg, calculation="power_flow", slack="ideal", dtype=torch.complex128
    )
    idx = node_phase_index(grid3)
    slack_row = idx.row(1, Phase.A)  # source at node 1
    scale = res.sampled.samples["src"].reshape(-1)  # [B]
    v_slack = res.v[:, slack_row].abs()
    # the slack row voltage is EXACTLY u_ref_v (230) times the per-scenario scale
    assert torch.allclose(v_slack, 230.0 * scale, atol=1e-8)
    # two scenarios with different draws -> different slack voltages
    assert v_slack.std() > 1e-3
    # non-slack voltages shift with the slack (not pinned)
    load_row = idx.row(3, Phase.A)
    assert res.v[:, load_row].abs().std() > 1e-3
    # the operating point carries the per-source u_ref_scale entry
    assert "u_ref_scale" in res.sampled.operating_point[1]


def test_source_uref_scale_constant_reproduces_nominal(grid3):
    """A Constant(1.0) u_ref scale reproduces the un-scaled ideal-slack solve exactly."""
    base = run_scenarios(
        grid3,
        ScenarioConfig(
            n_samples=1,
            parameters=[
                ParameterSpec(
                    name="load",
                    selector=Selector(component="load"),
                    distribution=Constant(value=1.0),
                    field="pq",
                )
            ],
        ),
        calculation="power_flow",
        dtype=torch.complex128,
    )
    scaled = run_scenarios(
        grid3,
        ScenarioConfig(
            n_samples=1,
            parameters=[
                ParameterSpec(
                    name="load",
                    selector=Selector(component="load"),
                    distribution=Constant(value=1.0),
                    field="pq",
                ),
                ParameterSpec(
                    name="src",
                    selector=Selector(component="source"),
                    distribution=Constant(value=1.0),
                    field="u_ref",
                    per="shared",
                ),
            ],
        ),
        calculation="power_flow",
        dtype=torch.complex128,
    )
    assert torch.allclose(base.v, scaled.v, atol=1e-9)


# =============================================================================
# (b) coherent parameters vary the fundamental; empty reproduces fixed-op
# =============================================================================
def _coherent(grid, parameters):
    return CoherentSpectrumConfig(
        name="c",
        selector=Selector(component="load"),
        orders=[3, 5],
        n_steps=3,
        n_scenarios=8,
        n_modes=2,
        seed=0,
        parameters=parameters,
    )


def test_coherent_parameters_vary_fundamental(grid3):
    params = [
        ParameterSpec(
            name="load_scale",
            selector=Selector(component="load"),
            distribution=Uniform(low=0.3, high=1.0),
            field="pq",
            mode="scale",
            per="each",
        ),
        ParameterSpec(
            name="source_scale",
            selector=Selector(component="source"),
            distribution=Normal(loc=1.0, scale=0.0333),
            field="u_ref",
            mode="scale",
            per="shared",
        ),
    ]
    with_params = run_scenarios(
        grid3,
        _coherent(grid3, params),
        harmonic_orders=[1, 3, 5],
        dtype=torch.complex128,
    )
    without = run_scenarios(
        grid3, _coherent(grid3, []), harmonic_orders=[1, 3, 5], dtype=torch.complex128
    )
    assert with_params.v.shape == without.v.shape  # [B, T, H, N]

    def band(v):  # per-row |V1| spread across scenarios+steps, averaged
        v1 = v[..., 0, :].abs().reshape(-1, v.shape[-1])
        return float(((v1.max(0).values - v1.min(0).values) / v1.mean(0)).mean())

    assert band(with_params.v) > 1e-2  # the fundamental now varies across scenarios
    assert band(without.v) < 1e-9  # the fixed-op fingerprint keeps |V1| constant

    # the coherent operating point + samples record the per-scenario draws ([B], not [B,T])
    assert set(with_params.sampled.operating_point[1]) == {"u_ref_scale"}
    assert with_params.sampled.samples["source_scale"].shape[0] == 8
    assert not without.sampled.operating_point


# =============================================================================
# (c) reproducibility + RNG stream separation
# =============================================================================
def test_same_config_seed_identical(grid3):
    cfg = _coherent(
        grid3,
        [
            ParameterSpec(
                name="load_scale",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.3, high=1.0),
                field="pq",
                per="each",
            )
        ],
    )
    a = run_scenarios(grid3, cfg, harmonic_orders=[1, 3, 5], dtype=torch.complex128)
    b = run_scenarios(grid3, cfg, harmonic_orders=[1, 3, 5], dtype=torch.complex128)
    assert torch.equal(a.v, b.v)
    for cid in a.sampled.operating_point:
        for k, v in a.sampled.operating_point[cid].items():
            assert torch.equal(v, b.sampled.operating_point[cid][k])


def test_parameters_do_not_perturb_fingerprint_stream(grid3):
    """The fingerprint RNG is a stream distinct from the operating-point cube: adding
    parameters leaves the realized harmonic_injection byte-identical."""
    empty = sample_coherent_spectra(grid3, _coherent(grid3, []))
    params = [
        ParameterSpec(
            name="load_scale",
            selector=Selector(component="load"),
            distribution=Uniform(low=0.3, high=1.0),
            field="pq",
            per="each",
        ),
        ParameterSpec(
            name="source_scale",
            selector=Selector(component="source"),
            distribution=Normal(loc=1.0, scale=0.0333),
            field="u_ref",
            per="shared",
        ),
    ]
    with_params = sample_coherent_spectra(grid3, _coherent(grid3, params))
    assert set(empty.harmonic_injection) == set(with_params.harmonic_injection)
    for cid, orders in empty.harmonic_injection.items():
        for o, (mag, phase) in orders.items():
            mag2, phase2 = with_params.harmonic_injection[cid][o]
            assert torch.equal(mag, mag2) and torch.equal(phase, phase2)
    # the fingerprint labels (mode path / base magnitudes) are also untouched
    assert torch.equal(empty.samples["c_mode"], with_params.samples["c_mode"])
    assert torch.equal(empty.samples["c_mag"], with_params.samples["c_mag"])


# =============================================================================
# (d) persistence round-trip with the new fields
# =============================================================================
def test_dataset_roundtrip_source_spec(grid3, tmp_path):
    cfg = ScenarioConfig(
        n_samples=4,
        seed=3,
        parameters=[
            ParameterSpec(
                name="load_scale",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.5, high=1.0),
                field="pq",
            ),
            ParameterSpec(
                name="source_scale",
                selector=Selector(component="source"),
                distribution=Normal(loc=1.0, scale=0.03),
                field="u_ref",
                per="shared",
            ),
        ],
    )
    res = run_scenarios(
        grid3,
        cfg,
        calculation="harmonic",
        harmonic_orders=[1, 3],
        dtype=torch.complex128,
    )
    path = write_dataset(res, tmp_path / "ds", layout="wide")
    loaded = read_dataset(path)
    # voltages round-trip exactly
    assert torch.allclose(loaded.v.to(res.v.dtype), res.v.cpu(), atol=1e-10)
    # the config (incl. the source spec) round-trips through the meta sidecar
    src = [p for p in loaded.config.parameters if p.field == "u_ref"]
    assert len(src) == 1 and src[0].selector.component == "source"
    # the source_scale draws are persisted in samples
    assert "source_scale" in loaded.samples
    assert loaded.samples["source_scale"].shape[0] == 4
