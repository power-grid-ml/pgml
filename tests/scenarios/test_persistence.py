"""Section 5: parquet persistence (both layouts) — round-trip + provenance."""

from __future__ import annotations

import os

import pytest
import torch

from pgml.scenarios import (
    ParameterSpec,
    Perturbation,
    ScenarioConfig,
    Selector,
    Uniform,
    perturbation_sweep,
    read_dataset,
    run_scenarios,
    write_dataset,
)


def _cfg(n=16, seed=7):
    return ScenarioConfig(
        n_samples=n,
        seed=seed,
        parameters=[
            ParameterSpec(
                name="load_pq",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.5, high=1.5),
            )
        ],
    )


@pytest.mark.parametrize("layout", ["wide", "long"])
def test_roundtrip_power_flow(grid3, tmp_path, layout):
    res = run_scenarios(grid3, _cfg())
    write_dataset(res, tmp_path, layout=layout)
    L = read_dataset(tmp_path)
    assert L.v.shape == res.v.shape
    torch.testing.assert_close(L.v, res.v)
    # node-phase layout + sampled inputs restored
    torch.testing.assert_close(L.node_ids, res.index.node_ids)
    torch.testing.assert_close(L.samples["load_pq"], res.sampled.samples["load_pq"])
    # provenance: config reconstructed with the same seed; power flow -> no frequencies
    assert isinstance(L.config, ScenarioConfig) and L.config.seed == 7
    assert L.frequencies_hz is None


def test_layouts_agree(grid3, tmp_path):
    res = run_scenarios(
        grid3, _cfg(), calculation="harmonic", harmonic_orders=[1, 5, 7]
    )
    wide, long = tmp_path / "w", tmp_path / "l"
    write_dataset(res, wide, layout="wide")
    write_dataset(res, long, layout="long")
    vw, vl = read_dataset(wide).v, read_dataset(long).v
    torch.testing.assert_close(vw, vl)
    torch.testing.assert_close(vw, res.v)
    # frequencies restored for a harmonic dataset
    torch.testing.assert_close(read_dataset(wide).frequencies_hz, res.frequencies_hz)


@pytest.mark.parametrize("layout", ["wide", "long"])
def test_roundtrip_sequence_preserves_dtypes(grid3, tmp_path, layout):
    """A sequence batch round-trips its ``[B, T, H, N]`` voltages and both record kinds."""
    from pgml.scenarios import batch_from_values

    b, t, ids = 8, 6, [10, 11]
    mode = torch.randint(0, 2, (b, len(ids), t), dtype=torch.int64)
    batch = batch_from_values(
        grid3,
        n_samples=b,
        n_steps=t,
        p_w={cid: torch.full((b, t), 1.2e3, dtype=torch.float64) for cid in ids},
        harmonic_injection={
            cid: {
                order: (
                    torch.full((b, t), 0.04, dtype=torch.float64),
                    torch.zeros((b, t), dtype=torch.float64),
                )
                for order in (3, 5)
            }
            for cid in ids
        },
        samples={"mode": mode},
        shared_samples={
            "time_s": torch.arange(t, dtype=torch.float64) * 900.0,
            "device_ids": torch.tensor(ids, dtype=torch.long),
        },
    )
    res = run_scenarios(grid3, batch, calculation="harmonic", harmonic_orders=[1, 3, 5])
    write_dataset(res, tmp_path, layout=layout)
    L = read_dataset(tmp_path)
    assert L.v.shape == res.v.shape == (b, t, 3, 3)
    torch.testing.assert_close(L.v, res.v)
    # int64 per-scenario record preserved exactly (B-leading, multi-dimensional)
    assert L.samples["mode"].dtype == mode.dtype
    torch.testing.assert_close(L.samples["mode"], mode)
    # declared batch-shared records restored from the sidecar
    torch.testing.assert_close(L.samples["time_s"], batch.shared_samples["time_s"])
    torch.testing.assert_close(
        L.samples["device_ids"], batch.shared_samples["device_ids"]
    )
    assert L.meta["n_steps"] == t


def test_roundtrip_perturbation_records(grid3, tmp_path):
    res = run_scenarios(
        grid3,
        perturbation_sweep(
            grid3,
            Selector(component="load"),
            Perturbation(field="p", mode="scale", value=1.5),
        ),
    )
    write_dataset(res, tmp_path, layout="wide")
    L = read_dataset(tmp_path)
    torch.testing.assert_close(L.v, res.v)
    assert len(L.perturbations) == 2  # one per scenario (ground truth as dicts)
    assert {p["component_id"] for p in L.perturbations} == {10, 11}
    assert isinstance(L.config, Perturbation)


def test_long_table_shape(grid3, tmp_path):
    import polars as pl

    res = run_scenarios(
        grid3, _cfg(n=4), calculation="harmonic", harmonic_orders=[1, 5]
    )
    write_dataset(res, tmp_path, layout="long")
    df = pl.read_parquet(tmp_path / "voltages.parquet")
    # one row per scenario x step(1) x freq(2) x node-phase(3) = 4*1*2*3 = 24
    assert df.height == 4 * 1 * 2 * 3
    assert {"scenario", "freq_idx", "frequency_hz", "node_id", "v_re", "v_im"} <= set(
        df.columns
    )


def test_wide_not_larger_than_long(grid3, tmp_path):
    res = run_scenarios(
        grid3, _cfg(n=64), calculation="harmonic", harmonic_orders=[1, 3, 5, 7, 11, 13]
    )
    wide, long = tmp_path / "w", tmp_path / "l"
    write_dataset(res, wide, layout="wide")
    write_dataset(res, long, layout="long")
    sz = lambda d: os.path.getsize(d / "voltages.parquet")  # noqa: E731
    assert sz(wide) <= sz(long)


def test_invalid_layout_raises(grid3, tmp_path):
    res = run_scenarios(grid3, _cfg(n=2))
    with pytest.raises(ValueError, match="layout"):
        write_dataset(res, tmp_path, layout="tall")


def test_schema_version_and_provenance(grid3, tmp_path):
    """meta stamps the schema version + environment fingerprint; read validates it."""
    import json

    from pgml.errors import InputError
    from pgml.schemas import SCHEMA_VERSION

    write_dataset(run_scenarios(grid3, _cfg(n=2)), tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["torch_version"] == torch.__version__ and meta["numpy_version"]
    read_dataset(tmp_path)  # matching version: clean read

    def _rewrite(version):
        m = json.loads((tmp_path / "meta.json").read_text())
        m["schema_version"] = version
        (tmp_path / "meta.json").write_text(json.dumps(m))

    # a MAJOR-version mismatch is incompatible -> raises
    _rewrite("9.0.0")
    with pytest.raises(InputError, match="schema_version"):
        read_dataset(tmp_path)
    # a minor/patch drift is read best-effort (no raise)
    _rewrite("0.0.999")
    read_dataset(tmp_path)


def test_generation_provenance_round_trips(grid3, tmp_path):
    """What produced the data: the config fingerprint, the code, the standards tables."""
    import json

    from pgml.scenarios import config_hash

    cfg = _cfg(n=2)
    write_dataset(
        run_scenarios(grid3, cfg), tmp_path, provenance={"recipe_version": "7"}
    )
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["config_hash"] == config_hash(cfg)
    # a generator's own versioned inputs travel in extra_provenance
    assert meta["extra_provenance"] == {"recipe_version": "7"}
    assert meta["provenance"]["pgml_version"] and meta["provenance"]["git_sha"]
    for table in ("en50160", "iec61000_3_2"):
        assert meta["standards"][table]["sha256"]
        assert meta["standards"][table]["override"] is False
    # the same stamps survive the read (they live in the sidecar the loader passes through)
    assert read_dataset(tmp_path).meta["config_hash"] == meta["config_hash"]


def test_config_hash_separates_every_knob(grid3):
    """A fingerprint that ignored the seed or the size would license a stale reuse."""
    from pgml.scenarios import config_hash

    base = config_hash(_cfg(n=8, seed=1))
    assert base == config_hash(_cfg(n=8, seed=1))
    assert base != config_hash(_cfg(n=8, seed=2))
    assert base != config_hash(_cfg(n=9, seed=1))
    # a serialized config hashes the same as the model it came from
    assert base == config_hash(_cfg(n=8, seed=1).model_dump_json())


def test_a_dataset_without_provenance_still_reads(grid3, tmp_path):
    """An older dataset simply lacks the keys; nothing in the read path requires them."""
    import json

    write_dataset(run_scenarios(grid3, _cfg(n=2)), tmp_path)
    meta_path = tmp_path / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("config_hash", "provenance", "extra_provenance", "standards"):
        del meta[key]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    loaded = read_dataset(tmp_path)
    assert loaded.meta.get("config_hash") is None
    assert loaded.v.shape[0] == 2


def test_standards_provenance_marks_an_environment_override(tmp_path, monkeypatch):
    """A PGML_* override silently replaces the emission reference — it must be recorded."""
    from pgml.scenarios import en50160_provenance

    packaged = en50160_provenance()
    table = tmp_path / "en50160.yaml"
    table.write_text("max_harmonic_values: {1: 1.0, 3: 0.05}\n", encoding="utf-8")
    monkeypatch.setenv("PGML_EN50160", str(table))
    overridden = en50160_provenance()
    assert overridden["override"] is True and overridden["source"] == str(table)
    assert overridden["sha256"] != packaged["sha256"]


def test_complex_sample_column_raises(tmp_path):
    """A complex sample column is rejected explicitly: a silent float64 cast
    would drop the imaginary part, and the shared-sample JSON path cannot
    encode complex values at all."""
    from pgml.errors import InputError
    from pgml.scenarios.persistence import _write_samples

    with pytest.raises(InputError, match="complex"):
        _write_samples(
            {"z": torch.ones(4, dtype=torch.complex128)}, {}, 4, tmp_path, "zstd"
        )


def test_convergence_metadata_round_trips(grid3, tmp_path):
    """Failed scenarios must stay identifiable after persistence."""
    from dataclasses import replace

    res = run_scenarios(grid3, _cfg(n=4))
    tainted = replace(res, converged=False, failed_states=(1, 3))
    write_dataset(tainted, tmp_path)
    L = read_dataset(tmp_path)
    assert L.converged is False
    assert L.failed_scenarios == (1, 3)
    assert L.meta["failed_scenarios"] == [1, 3]


def test_legacy_dataset_without_convergence_metadata(grid3, tmp_path):
    """A meta.json lacking convergence keys reads as unknown, not as valid."""
    import json

    res = run_scenarios(grid3, _cfg(n=4))
    write_dataset(res, tmp_path)
    meta_path = tmp_path / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["converged"], meta["failed_scenarios"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    L = read_dataset(tmp_path)
    assert L.converged is None
    assert L.failed_scenarios == ()


def test_read_reconstructs_a_foreign_config_only_when_told(grid3, tmp_path):
    """A config class this package does not define needs ``config_types`` to come back typed.

    Reading a dataset must not import a module named by the file, so reconstruction is the
    caller's explicit choice; without it the config reads back as the raw dict.
    """
    from pydantic import BaseModel

    class ExternalRecipe(BaseModel):
        """A config owned by a downstream generator."""

        n_samples: int
        seed: int = 3

    cfg = ExternalRecipe(n_samples=2)
    res = run_scenarios(grid3, _cfg(n=2))
    res = type(res)(
        v=res.v,
        index=res.index,
        sampled=type(res.sampled)(
            operating_point=res.sampled.operating_point,
            samples=res.sampled.samples,
            n_samples=res.sampled.n_samples,
            config=cfg,
        ),
        frequencies_hz=res.frequencies_hz,
    )
    write_dataset(res, tmp_path)

    untyped = read_dataset(tmp_path)
    assert isinstance(untyped.config, dict) and untyped.config["n_samples"] == 2
    typed = read_dataset(tmp_path, config_types={"ExternalRecipe": ExternalRecipe})
    assert isinstance(typed.config, ExternalRecipe) and typed.config.seed == 3
    # the sidecar names the owner so the class is identifiable without importing it
    assert typed.meta["config_class"].endswith("ExternalRecipe")
    assert typed.meta["config_module"] == ExternalRecipe.__module__


def test_config_hash_is_independent_of_where_the_class_lives(grid3):
    """The fingerprint covers the serialized FIELDS, so relocating a config preserves it.

    This is what lets a config class move between packages without invalidating the
    hashes recorded in existing dataset manifests.
    """
    from pydantic import BaseModel

    from pgml.scenarios import config_hash

    class Recipe(BaseModel):
        n_samples: int
        seed: int

    class RelocatedRecipe(BaseModel):
        n_samples: int
        seed: int

    assert config_hash(Recipe(n_samples=8, seed=1)) == config_hash(
        RelocatedRecipe(n_samples=8, seed=1)
    )


def test_the_sidecar_carries_the_time_axis(grid3, tmp_path):
    """``n_steps`` / ``step_size_s`` / ``t0_unix_s`` are data, not something to re-derive."""
    import json

    from pgml.scenarios import batch_from_values

    b, t, step = 2, 3, 900.0
    t0 = 1_700_000_000.0
    batch = batch_from_values(
        grid3,
        n_samples=b,
        n_steps=t,
        p_w={10: torch.full((b, t), 1.0e3, dtype=torch.float64)},
        shared_samples={
            "time_unix_s": t0 + torch.arange(t, dtype=torch.float64) * step
        },
    )
    write_dataset(run_scenarios(grid3, batch), tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_steps"] == t and meta["dims"]["T"] == t
    assert meta["t0_unix_s"] == t0
    assert meta["step_size_s"] is None  # no config declared one


def test_a_config_its_class_no_longer_accepts_reads_back_as_a_dict(
    grid3, tmp_path, caplog
):
    """A dataset outlives the config class that wrote it: a field value the class has
    dropped since must not make the voltages unreadable."""
    import json

    res = run_scenarios(grid3, _cfg(n=4))
    write_dataset(res, tmp_path)
    meta_path = tmp_path / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stored = json.loads(meta["config_json"])
    stored["parameters"][0]["field"] = "a_field_of_an_older_release"
    meta["config_json"] = json.dumps(stored)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with caplog.at_level("WARNING", logger="pgml"):
        L = read_dataset(tmp_path)
    assert isinstance(L.config, dict)
    assert L.config["parameters"][0]["field"] == "a_field_of_an_older_release"
    assert torch.equal(L.v, res.v)
    assert "no longer validates" in caplog.text
