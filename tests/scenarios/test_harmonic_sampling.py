"""Section 4: harmonic-spectrum sampling (4a random + 4b node-coherent fingerprints)."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from pgml.scenarios import (
    CoherentSpectrumConfig,
    Correlation,
    NodeInjectionSweepConfig,
    ParameterSpec,
    ScenarioConfig,
    Selector,
    SpectrumSweepConfig,
    Uniform,
    en50160_limit,
    en50160_limits,
    run_node_injection_sweep,
    run_scenarios,
    sample,
    sample_coherent_spectra,
    spectrum_sweep,
)


# --- EN 50160 loader --------------------------------------------------------
def test_en50160_table():
    t = en50160_limits()
    assert t[5] == 0.06 and t[3] == 0.05 and t[7] == 0.05
    assert en50160_limit(11) == 0.035
    with pytest.raises(KeyError):
        en50160_limit(999)


# --- 4a: random harmonic spectrum sampling ----------------------------------
def _hcfg(n=64, **kw) -> ScenarioConfig:
    kw.setdefault("name", "hm")
    kw.setdefault("selector", Selector(component="load"))
    kw.setdefault("distribution", Uniform(low=0.0, high=1.0))
    kw.setdefault("field", "h_mag")
    kw.setdefault("mode", "absolute")
    kw.setdefault("orders", [3, 5, 7])
    return ScenarioConfig(n_samples=n, seed=0, parameters=[ParameterSpec(**kw)])


def test_hmag_en50160_bounded(grid3):
    s = sample(grid3, _hcfg(harmonic_reference="en50160"))
    assert s.samples["hm"].shape == (64, 2, 3)  # [B, n_comp, n_orders]
    for cid in (10, 11):
        inj = s.harmonic_injection[cid]
        assert sorted(inj) == [3, 5, 7]
        for order in (3, 5, 7):
            mag, _phase = inj[order]
            assert mag.shape == (64,)
            assert float(mag.max()) <= en50160_limit(order) + 1e-9
            assert float(mag.min()) >= 0.0


def test_hmag_absolute_without_reference(grid3):
    s = sample(grid3, _hcfg(distribution=Uniform(low=0.0, high=0.2)))
    mag, _ = s.harmonic_injection[10][5]
    assert float(mag.max()) <= 0.2 and float(mag.min()) >= 0.0


def test_hmag_scale_seeds_from_stored_spectrum(grid_spectrum):
    # mode="scale" multiplies the device's stored per-order magnitude (0.1 at h5).
    s = sample(
        grid_spectrum,
        _hcfg(mode="scale", orders=[5], distribution=Uniform(low=1.0, high=2.0)),
    )
    mag, phase = s.harmonic_injection[10][5]
    assert float(mag.min()) >= 0.1 - 1e-9 and float(mag.max()) <= 0.2 + 1e-9
    # the stored order 7 (not sampled) survives the override, as a float
    assert 7 in s.harmonic_injection[10]
    assert s.harmonic_injection[10][7] == (0.05, -20.0)


def test_hmag_scale_without_stored_raises(grid3):
    with pytest.raises(ValueError, match="no stored spectrum"):
        sample(
            grid3,
            _hcfg(mode="scale", orders=[5], distribution=Uniform(low=1.0, high=2.0)),
        )


def test_hphase_sets_phase(grid3):
    s = sample(
        grid3,
        _hcfg(
            field="h_phase",
            mode="absolute",
            distribution=Uniform(low=-180.0, high=180.0),
        ),
    )
    _mag, phase = s.harmonic_injection[10][5]
    assert phase.shape == (64,)
    assert float(phase.min()) >= -180.0 and float(phase.max()) <= 180.0


def test_hmag_shared_is_identical_across_devices(grid3):
    s = sample(grid3, _hcfg(per="shared", harmonic_reference="en50160"))
    m10, _ = s.harmonic_injection[10][5]
    m11, _ = s.harmonic_injection[11][5]
    torch.testing.assert_close(m10, m11)


def test_harmonic_validators():
    base = dict(
        name="h",
        selector=Selector(component="load"),
        distribution=Uniform(low=0.0, high=1.0),
    )
    with pytest.raises(ValidationError):  # harmonic field needs orders
        ParameterSpec(**base, field="h_mag")
    with pytest.raises(ValidationError):  # orders >= 2
        ParameterSpec(**base, field="h_mag", orders=[1, 5])
    with pytest.raises(ValidationError):  # h_phase needs absolute
        ParameterSpec(**base, field="h_phase", orders=[5], mode="scale")
    with pytest.raises(ValidationError):  # correlation not allowed on harmonic
        ParameterSpec(
            **base,
            field="h_mag",
            orders=[5],
            correlation=Correlation(factor="f", rho=0.5),
        )
    with pytest.raises(ValidationError):  # orders only for harmonic fields
        ParameterSpec(**base, field="p", orders=[5])
    with pytest.raises(ValidationError):  # reference only for h_mag
        ParameterSpec(
            **base,
            field="h_phase",
            orders=[5],
            mode="absolute",
            harmonic_reference="en50160",
        )


def test_run_harmonic_with_sampled_spectrum(grid3):
    res = run_scenarios(
        grid3,
        _hcfg(n=8, harmonic_reference="en50160"),
        calculation="harmonic",
        harmonic_orders=[1, 3, 5, 7],
    )
    assert res.v.shape[0] == 8 and res.v.shape[1] == 4  # [B, H, N]
    assert res.frequencies_hz.tolist() == [50.0, 150.0, 250.0, 350.0]


def test_harmonic_reproducible(grid3):
    a = sample(grid3, _hcfg(harmonic_reference="en50160"))
    b = sample(grid3, _hcfg(harmonic_reference="en50160"))
    torch.testing.assert_close(
        a.harmonic_injection[10][5][0], b.harmonic_injection[10][5][0]
    )


# --- 4b: node-coherent harmonic fingerprints --------------------------------
def _ccfg(**kw) -> CoherentSpectrumConfig:
    kw.setdefault("selector", Selector(component="load"))
    kw.setdefault("orders", [3, 5, 7])
    kw.setdefault("n_steps", 24)
    kw.setdefault("n_scenarios", 8)
    kw.setdefault("n_modes", 2)
    kw.setdefault("seed", 0)
    return CoherentSpectrumConfig(**kw)


def test_coherent_shapes_and_timestamps(grid3):
    s = sample_coherent_spectra(grid3, _ccfg(step_size_s=900.0))
    assert s.samples["harmonics_mode"].shape == (8, 2, 24)  # [B, n_dev, T]
    assert s.samples["harmonics_mag"].shape == (8, 2, 3, 24)  # [B, n_dev, n_ord, T]
    assert s.samples["harmonics_device_ids"].tolist() == [10, 11]
    torch.testing.assert_close(
        s.samples["time_s"][:3], torch.tensor([0.0, 900.0, 1800.0], dtype=torch.float64)
    )
    mag, phase = s.harmonic_injection[10][5]
    assert mag.shape == (8, 24)  # [B, T]


def test_coherent_stickiness_matches_dwell(grid3):
    s = sample_coherent_spectra(grid3, _ccfg(n_scenarios=24, n_steps=60, dwell=0.9))
    mp = s.samples["harmonics_mode"]
    stay = (mp[..., 1:] == mp[..., :-1]).float().mean()
    assert abs(float(stay) - 0.9) < 0.05


def test_coherent_en50160_clamp(grid3):
    s = sample_coherent_spectra(
        grid3, _ccfg(jitter_mag=0.5)
    )  # large jitter exercises clamp
    for order in (3, 5, 7):
        mag, _ = s.harmonic_injection[10][order]
        assert float(mag.max()) <= en50160_limit(order) + 1e-9
        assert float(mag.min()) >= 0.0


def test_coherent_single_mode_runs(grid3):
    s = sample_coherent_spectra(grid3, _ccfg(n_modes=1))
    assert int(s.samples["harmonics_mode"].abs().sum()) == 0  # all mode 0


def test_coherent_reproducible(grid3):
    a = sample_coherent_spectra(grid3, _ccfg())
    b = sample_coherent_spectra(grid3, _ccfg())
    torch.testing.assert_close(
        a.harmonic_injection[10][5][0], b.harmonic_injection[10][5][0]
    )
    torch.testing.assert_close(a.samples["harmonics_mode"], b.samples["harmonics_mode"])


def test_run_coherent_time_axis(grid3):
    res = run_scenarios(grid3, _ccfg(n_scenarios=4, n_steps=10))
    assert res.v.shape == (4, 10, 4, 3)  # [B, T, H, N]; H=[1,3,5,7], N=3 nodes
    assert res.frequencies_hz.tolist() == [50.0, 150.0, 250.0, 350.0]


# --- spectrum_sweep: per-target diagonal harmonic injection -----------------
def test_spectrum_sweep_is_diagonal(grid3):
    cfg = SpectrumSweepConfig.from_spectrum(
        Selector(component="load"), {11: (0.08, 0.0)}, name="inj"
    )
    s = spectrum_sweep(grid3, cfg)
    assert s.n_samples == 2  # one scenario per load (10, 11)
    assert s.samples["inj_id"].tolist() == [10, 11]
    # device 10 injects ONLY in scenario 0 (one-hot), device 11 ONLY in scenario 1
    torch.testing.assert_close(
        s.harmonic_injection[10][11][0], torch.tensor([0.08, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        s.harmonic_injection[11][11][0], torch.tensor([0.0, 0.08], dtype=torch.float64)
    )


def test_spectrum_sweep_from_spectrum_drops_fundamental():
    cfg = SpectrumSweepConfig.from_spectrum(
        Selector(component="load"), {1: (1.0, 0.0), 5: (0.1, 10.0), 11: (0.08, -5.0)}
    )
    assert cfg.orders == [5, 11]  # order 1 dropped, sorted
    assert cfg.magnitudes_pu == [0.1, 0.08]


def test_spectrum_sweep_h_voltage_localized(grid3):
    # injecting at one node produces a nonzero h11 field; the spread is recoverable.
    cfg = SpectrumSweepConfig.from_spectrum(
        Selector(component="load"), {11: (0.1, 0.0)}
    )
    res = run_scenarios(grid3, cfg, harmonic_orders=[1, 11])
    assert res.v.shape == (2, 2, 3)  # [B=2 injections, H=(1,11), N=3]
    h11 = res.v[:, 1, :]  # [B, N] complex h11 field
    assert h11.abs().max() > 0  # injection produced harmonic voltage


def test_spectrum_sweep_selector_subset(grid3):
    cfg = SpectrumSweepConfig.from_spectrum(
        Selector(component="load", ids=[11]), {11: (0.1, 0.0)}
    )
    s = spectrum_sweep(grid3, cfg)
    assert s.n_samples == 1 and set(s.harmonic_injection) == {11}


# --- node_injection_sweep: per-node Thévenin/Norton "error" source ----------
def test_node_injection_sweep_all_nodes(grid3):
    cfg = NodeInjectionSweepConfig.from_spectrum(
        {11: (0.05, 0.0)}, source_power_va=1e5, kind="voltage"
    )
    res = run_node_injection_sweep(grid3, cfg)
    assert res.v.shape == (3, 2, 3)  # [B=3 nodes, H=(1,11), N=3]
    assert res.frequencies_hz.tolist() == [50.0, 550.0]
    assert res.sampled.samples["injection_node_id"].tolist() == [1, 2, 3]
    # injecting produces a nonzero h11 field
    assert res.v[:, 1, :].abs().max() > 0


def test_node_injection_sweep_kinds_and_subset(grid3):
    base = dict(source_power_va=1e5)
    v = run_node_injection_sweep(
        grid3,
        NodeInjectionSweepConfig.from_spectrum(
            {11: (0.05, 0.0)}, kind="voltage", node_ids=[2], **base
        ),
    )
    c = run_node_injection_sweep(
        grid3,
        NodeInjectionSweepConfig.from_spectrum(
            {11: (0.05, 0.0)}, kind="current", node_ids=[2], **base
        ),
    )
    assert v.v.shape == (1, 2, 3) and c.v.shape == (1, 2, 3)
    assert v.v[:, 1, :].abs().max() > 0 and c.v[:, 1, :].abs().max() > 0


def test_node_injection_from_spectrum_drops_fundamental():
    cfg = NodeInjectionSweepConfig.from_spectrum(
        {1: (1.0, 0.0), 5: (0.1, 0.0), 11: (0.05, 0.0)}, source_power_va=1e5
    )
    assert cfg.orders == [5, 11] and cfg.source_power_va == 1e5
