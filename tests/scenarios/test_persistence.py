"""Section 5: parquet persistence (both layouts) — round-trip + provenance."""

from __future__ import annotations

import os

import pytest
import torch

from pgml.scenarios import (
    CoherentSpectrumConfig,
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
def test_roundtrip_coherent_preserves_dtypes(grid3, tmp_path, layout):
    res = run_scenarios(
        grid3,
        CoherentSpectrumConfig(
            selector=Selector(component="load"), orders=[3, 5], n_steps=6, n_scenarios=8
        ),
    )
    write_dataset(res, tmp_path, layout=layout)
    L = read_dataset(tmp_path)
    assert L.v.shape == res.v.shape == (8, 6, 3, 3)
    torch.testing.assert_close(L.v, res.v)
    # int64 mode path preserved exactly (B-leading multi-dim sample)
    mode = res.sampled.samples["harmonics_mode"]
    assert L.samples["harmonics_mode"].dtype == mode.dtype
    torch.testing.assert_close(L.samples["harmonics_mode"], mode)
    # shared (non-batched) samples restored from the sidecar
    torch.testing.assert_close(L.samples["time_s"], res.sampled.samples["time_s"])
    torch.testing.assert_close(
        L.samples["harmonics_device_ids"], res.sampled.samples["harmonics_device_ids"]
    )
    assert isinstance(L.config, CoherentSpectrumConfig)


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


def test_complex_sample_column_raises(tmp_path):
    """A complex sample column is rejected explicitly: a silent float64 cast
    would drop the imaginary part, and the shared-sample JSON path cannot
    encode complex values at all."""
    from pgml.errors import InputError
    from pgml.scenarios.persistence import _write_samples

    with pytest.raises(InputError, match="complex"):
        _write_samples(
            {"z": torch.ones(4, dtype=torch.complex128)}, 4, tmp_path, "zstd"
        )
