"""IEC 61000-3-2 harmonic current-emission limits: table, fractions, and consumers.

The IEC current-emission standard is the default reference for device current
fingerprints (as opposed to the DIN EN 50160 voltage-compatibility levels, which remain
available for background-distortion-shaped experiments).
"""

from __future__ import annotations

import math

import pytest
import torch
from pydantic import ValidationError

from pgml.schemas.grid_schema import (
    ConsumerType,
    Grid,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
)
from pgml.scenarios import (
    CoherentSpectrumConfig,
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    iec61000_3_2_device_caps,
    iec61000_3_2_fraction,
    iec61000_3_2_limits,
    resolve_emission_class,
    sample,
    sample_coherent_spectra,
)

_W = 2.0 * math.pi * 50.0


# --- table load -------------------------------------------------------------
def test_iec_table_values():
    t = iec61000_3_2_limits()
    assert sorted(t) == ["A", "B", "C", "D"]
    # Class A absolute amperes (Table 1): explicit low orders + the 0.15*15/n tail.
    assert t["A"]["unit"] == "A"
    assert t["A"]["limits"][3] == 2.30 and t["A"]["limits"][5] == 1.14
    assert t["A"]["limits"][7] == 0.77 and t["A"]["limits"][9] == 0.40
    assert t["A"]["limits"][2] == 1.08 and t["A"]["limits"][4] == 0.43
    assert (
        t["A"]["limits"][15] == 0.15
        and abs(t["A"]["limits"][21] - 0.15 * 15 / 21) < 1e-3
    )
    # Class B = 1.5 * Class A.
    assert abs(t["B"]["limits"][3] - 1.5 * 2.30) < 1e-9
    # Class C percentage-of-fundamental; h3 flagged power-factor scaled.
    assert t["C"]["unit"] == "percent"
    assert t["C"]["limits"][3] == 30.0 and t["C"]["limits"][5] == 10.0
    assert t["C"]["limits"][7] == 7.0 and t["C"]["limits"][9] == 5.0
    assert t["C"]["limits"][11] == 3.0 and t["C"]["limits"][2] == 2.0
    assert t["C"]["power_factor_scaled_orders"] == (3,)
    # Class D mA/W (Table 3): explicit 3..11 then 3.85/n.
    assert t["D"]["unit"] == "mA_per_W"
    assert t["D"]["limits"][3] == 3.4 and t["D"]["limits"][5] == 1.9
    assert t["D"]["limits"][7] == 1.0 and t["D"]["limits"][11] == 0.35
    assert abs(t["D"]["limits"][13] - 3.85 / 13) < 1e-3


def test_iec_limits_single_class_copy():
    a = iec61000_3_2_limits("a")  # case-insensitive
    assert a["unit"] == "A" and a["limits"][3] == 2.30
    a["limits"][3] = 0.0  # mutating the copy must not poison the cache
    assert iec61000_3_2_limits("A")["limits"][3] == 2.30


# --- fraction conversion ----------------------------------------------------
def test_fraction_class_a_and_b():
    i1 = 2000.0 / 230.0  # ~8.70 A
    assert iec61000_3_2_fraction(
        3, emission_class="A", p_w=2000.0, u_ln_v=230.0
    ) == pytest.approx(2.30 / i1)
    # Class B is 1.5x the Class A fraction.
    assert iec61000_3_2_fraction(
        3, emission_class="B", p_w=2000.0, u_ln_v=230.0
    ) == pytest.approx(1.5 * 2.30 / i1)


def test_fraction_class_c_percent_and_power_factor():
    # Class C is a percentage of the fundamental (independent of I1); h3 scales by lambda.
    assert iec61000_3_2_fraction(
        5, emission_class="C", p_w=2000.0, u_ln_v=230.0
    ) == pytest.approx(0.10)
    assert iec61000_3_2_fraction(
        3, emission_class="C", p_w=2000.0, u_ln_v=230.0, power_factor=1.0
    ) == pytest.approx(0.30)
    assert iec61000_3_2_fraction(
        3, emission_class="C", p_w=2000.0, u_ln_v=230.0, power_factor=0.7
    ) == pytest.approx(0.30 * 0.7)


def test_fraction_class_d_power_cancels():
    # Class D fraction = mA_per_W * u_ln * pf / 1000 -- independent of power.
    f_small = iec61000_3_2_fraction(3, emission_class="D", p_w=100.0, u_ln_v=230.0)
    f_large = iec61000_3_2_fraction(3, emission_class="D", p_w=500.0, u_ln_v=230.0)
    assert f_small == pytest.approx(f_large)
    assert f_small == pytest.approx(3.4 * 230.0 / 1000.0)


def test_fraction_clamps_and_absent_orders():
    # An emission above the fundamental (tiny / zero current) clamps to 1.0.
    assert iec61000_3_2_fraction(3, emission_class="A", p_w=1.0, u_ln_v=230.0) == 1.0
    assert iec61000_3_2_fraction(3, emission_class="A", p_w=0.0, u_ln_v=230.0) == 1.0
    # Orders absent from a class table return 0.0 (even orders have no Class C/D limit).
    assert iec61000_3_2_fraction(2, emission_class="D", p_w=200.0, u_ln_v=230.0) == 0.0
    assert iec61000_3_2_fraction(99, emission_class="A", p_w=200.0, u_ln_v=230.0) == 0.0


def test_fraction_rejects_auto_and_unknown_class():
    from pgml.errors import InputError

    with pytest.raises(InputError):
        iec61000_3_2_fraction(3, emission_class="auto", p_w=200.0, u_ln_v=230.0)
    with pytest.raises(InputError):
        iec61000_3_2_fraction(3, emission_class="Z", p_w=200.0, u_ln_v=230.0)


# --- auto emission-class resolution -----------------------------------------
def test_resolve_emission_class_mapping():
    # household / EV / PV / unspecified -> the general Class A.
    assert resolve_emission_class("household", 2000.0) == "A"
    assert resolve_emission_class(ConsumerType.EV_CHARGING, 11000.0) == "A"
    assert resolve_emission_class(ConsumerType.PV, 5000.0) == "A"
    assert resolve_emission_class(None, 2000.0) == "A"
    # office (IT / electronics) -> Class D under the 600 W window, else Class A.
    assert resolve_emission_class(ConsumerType.OFFICE, 500.0) == "D"
    assert resolve_emission_class("office", 800.0) == "A"


# --- per-device caps --------------------------------------------------------
def test_device_caps_are_per_device(grid3):
    # household (id 10) P=2000 W and EV (id 11) P=3000 W at 230 V, both Class A.
    caps = iec61000_3_2_device_caps(grid3, [10, 11], [3, 5, 7], emission_class="auto")
    assert caps[10][3] == pytest.approx(2.30 / (2000.0 / 230.0))
    assert caps[11][3] == pytest.approx(2.30 / (3000.0 / 230.0))
    # a bigger load draws more fundamental current -> a smaller emission fraction.
    assert caps[11][3] < caps[10][3]


def test_device_caps_explicit_class(grid3):
    caps = iec61000_3_2_device_caps(grid3, [10], [3], emission_class="C")
    assert caps[10][3] == pytest.approx(0.30)  # Class C h3, pf=1


# --- config validation ------------------------------------------------------
def test_emission_class_requires_iec_reference():
    base = dict(
        name="hm",
        selector=Selector(component="load"),
        distribution=Uniform(low=0.0, high=1.0),
        field="h_mag",
        mode="absolute",
        orders=[3, 5, 7],
    )
    # emission_class only valid with the IEC reference.
    with pytest.raises(ValidationError):
        ParameterSpec(**base, harmonic_reference="en50160", emission_class="C")
    with pytest.raises(ValidationError):
        ParameterSpec(**base, emission_class="A")  # no reference at all
    # valid: IEC reference + explicit class.
    ok = ParameterSpec(**base, harmonic_reference="iec61000-3-2", emission_class="C")
    assert ok.emission_class == "C"


def test_coherent_default_reference_is_iec():
    cfg = CoherentSpectrumConfig(
        selector=Selector(component="load"), orders=[3, 5, 7], n_steps=4
    )
    assert cfg.harmonic_reference == "iec61000-3-2" and cfg.emission_class == "auto"
    with pytest.raises(ValidationError):
        CoherentSpectrumConfig(
            selector=Selector(component="load"),
            orders=[3],
            n_steps=4,
            harmonic_reference="en50160",
            emission_class="B",
        )


# --- consumers: sampler + coherent bounded by the per-device cap ------------
def _hcfg(**kw) -> ScenarioConfig:
    kw.setdefault("name", "hm")
    kw.setdefault("selector", Selector(component="load"))
    kw.setdefault("distribution", Uniform(low=0.0, high=1.0))
    kw.setdefault("field", "h_mag")
    kw.setdefault("mode", "absolute")
    kw.setdefault("orders", [3, 5, 7])
    kw.setdefault("harmonic_reference", "iec61000-3-2")
    return ScenarioConfig(n_samples=64, seed=0, parameters=[ParameterSpec(**kw)])


def test_sampler_iec_bounded_per_device(grid3):
    caps = iec61000_3_2_device_caps(grid3, [10, 11], [3, 5, 7])
    s = sample(grid3, _hcfg())
    for cid in (10, 11):
        for order in (3, 5, 7):
            mag, _ = s.harmonic_injection[cid][order]
            assert float(mag.max()) <= caps[cid][order] + 1e-9
            assert float(mag.min()) >= 0.0


def test_coherent_iec_bounded_per_device(grid3):
    caps = iec61000_3_2_device_caps(grid3, [10, 11], [3, 5, 7])
    s = sample_coherent_spectra(
        grid3,
        CoherentSpectrumConfig(
            selector=Selector(component="load"),
            orders=[3, 5, 7],
            n_steps=24,
            n_scenarios=8,
            jitter_mag=0.5,  # large jitter exercises the clamp
        ),
    )
    for cid in (10, 11):
        for order in (3, 5, 7):
            mag, _ = s.harmonic_injection[cid][order]
            assert float(mag.max()) <= caps[cid][order] + 1e-9


# --- mode_bank_seed: reproducible fingerprint bank --------------------------
def _ccfg(**kw) -> CoherentSpectrumConfig:
    kw.setdefault("selector", Selector(component="load"))
    kw.setdefault("orders", [3, 5, 7])
    kw.setdefault("n_steps", 24)
    kw.setdefault("n_scenarios", 8)
    kw.setdefault("seed", 0)
    return CoherentSpectrumConfig(**kw)


def test_mode_bank_seed_none_is_byte_identical(grid3):
    # The default (mode_bank_seed absent) draws the bank from the `seed` stream; an
    # explicit None must reproduce it exactly (byte-identical) -- no behavior change.
    a = sample_coherent_spectra(grid3, _ccfg())
    b = sample_coherent_spectra(grid3, _ccfg(mode_bank_seed=None))
    torch.testing.assert_close(
        a.samples["harmonics_mode_base_mag"], b.samples["harmonics_mode_base_mag"]
    )
    torch.testing.assert_close(a.samples["harmonics_mode"], b.samples["harmonics_mode"])
    torch.testing.assert_close(
        a.harmonic_injection[10][5][0], b.harmonic_injection[10][5][0]
    )


def test_mode_bank_seed_pins_distinct_bank(grid3):
    base = sample_coherent_spectra(grid3, _ccfg())
    held_out = sample_coherent_spectra(grid3, _ccfg(mode_bank_seed=999))
    # A distinct fingerprint bank while every other setting is shared.
    assert (
        base.samples["harmonics_mode_base_mag"]
        - held_out.samples["harmonics_mode_base_mag"]
    ).abs().max() > 1e-6
    # and it is itself reproducible.
    again = sample_coherent_spectra(grid3, _ccfg(mode_bank_seed=999))
    torch.testing.assert_close(
        held_out.samples["harmonics_mode_base_mag"],
        again.samples["harmonics_mode_base_mag"],
    )


# --- env override -----------------------------------------------------------
def test_env_override(tmp_path, monkeypatch):
    custom = tmp_path / "iec.yaml"
    custom.write_text(
        "class_a:\n  unit: A\n  limits:\n    3: 9.99\n"
        "class_b:\n  unit: A\n  multiplier_of_class_a: 2.0\n"
        "class_c:\n  unit: percent\n  power_factor_scaled_orders: [3]\n"
        "  limits:\n    3: 30.0\n"
        "class_d:\n  unit: mA_per_W\n  limits:\n    3: 3.4\n"
    )
    monkeypatch.setenv("PGML_IEC61000_3_2", str(custom))
    # An explicit path overrides the env; the env overrides the packaged file.
    assert iec61000_3_2_limits("A", path=str(custom))["limits"][3] == 9.99


def _office_grid(p_w: float) -> Grid:
    """Single-phase 2-bus grid with one OFFICE load of the given nominal power."""
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=230.0, phases=(Phase.A,)),
            Node(id=2, u_rated_v=230.0, phases=(Phase.A,)),
        ],
        branches=[
            Line(
                id=1,
                from_node=1,
                to_node=2,
                from_phases=(Phase.A,),
                to_phases=(Phase.A,),
                length_m=1.0,
                series_resistance_ohm_per_m=[[0.5]],
                series_inductance_h_per_m=[[0.5 / _W]],
                shunt_capacitance_f_per_m=[[0.0]],
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=(Phase.A,),
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[0.1 / _W]],
            ),
            Load(
                id=10,
                node=2,
                phases=(Phase.A,),
                p_nom_w=p_w,
                q_nom_var=0.0,
                load_model=LoadModel.CONST_POWER,
                consumer_type="office",
            ),
        ],
    )


def test_auto_office_resolves_class_d_under_window():
    # A <=600 W office load auto-resolves to Class D; Class D h3 is the mA/W fraction.
    caps = iec61000_3_2_device_caps(
        _office_grid(500.0), [10], [3], emission_class="auto"
    )
    assert caps[10][3] == pytest.approx(3.4 * 230.0 / 1000.0)
    # A larger office load falls back to the general Class A.
    caps_a = iec61000_3_2_device_caps(
        _office_grid(5000.0), [10], [3], emission_class="auto"
    )
    assert caps_a[10][3] == pytest.approx(2.30 / (5000.0 / 230.0))
