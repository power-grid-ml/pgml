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
    Constant,
    Normal,
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    read_dataset,
    run_scenarios,
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
