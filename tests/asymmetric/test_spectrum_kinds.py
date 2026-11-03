"""Stored spectrum kinds: what the solve, the numpy oracle and the sampler read.

``StaticSpectrum`` is used as stored. ``LoadVaryingSpectrum`` is evaluated at the device's
solved active power over its rating, interpolating each order as a complex phasor between
the stored levels (``pgml.assembly._spectrum`` is the one reader all consumers share).
``TimeVaryingSpectrum`` raises ``ModelingError`` naming the device instead of injecting
nothing. ``spectrum_per_phase`` reaches the numpy oracle per element exactly as it reaches
the solve, and the sampler refuses to override a device whose spectrum it cannot seed.
"""

from __future__ import annotations

import cmath
import logging
import math

import numpy as np
import pytest
import torch

from pgml.assembly._spectrum import evaluate_load_varying, spectrum_rating
from pgml.errors import InputError, ModelingError
from pgml.evaluation.oracles.numpy_oracle import numpy_harmonic_voltages
from pgml.scenarios import ParameterSpec, ScenarioConfig, Selector, Uniform, sample
from pgml.schemas.grid_schema import (
    ExtrapolationMethod,
    Generator,
    InterpolationMethod,
    Load,
    LoadModel,
    LoadVaryingSpectrum,
    Phase,
    TimeVaryingSpectrum,
)
from pgml.solver import solve_harmonic_flow

from .test_harmonic_per_phase import ABC, _grid, _spec

ORDERS = [1, 5, 7]
P_NOM = 4500.0


def _point(comps):
    return _spec(comps).spectrum


LOW = _point([(1, 1.0, 0.0), (5, 0.10, 170.0), (7, 0.05, -20.0)])
HIGH = _point([(1, 1.0, 0.0), (5, 0.30, -170.0)])  # h7 absent: zero at this level


def _lv(points=(LOW, HIGH), levels=(0.2, 1.0), **kw):
    return LoadVaryingSpectrum(
        loading_levels_pu=list(levels), spectra=list(points), **kw
    )


def _load(**spectrum_kw):
    return Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=P_NOM,
        q_nom_var=900.0,
        load_model=LoadModel.CONST_POWER,
        **spectrum_kw,
    )


def _solve(spectrum_kw, p_w=None, **kw):
    op = None if p_w is None else {30: {"p_w": p_w, "q_var": 900.0}}
    res = solve_harmonic_flow(
        _grid(_load(**spectrum_kw), ABC),
        ORDERS,
        slack="norton",
        load_shunt="none",
        operating_point=op,
        dtype=torch.complex128,
        **kw,
    )
    return res.v


def _static_at(loading, levels=(0.2, 1.0), points=(LOW, HIGH)):
    """Hand-computed complex interpolation, as a static spectrum (ground truth)."""
    x = min(max(loading, levels[0]), levels[-1])
    w = (x - levels[0]) / (levels[1] - levels[0])
    comps = []
    for order in (1, 5, 7):
        c = []
        for pt in points:
            hit = [p for p in pt.components if p.order == order]
            c.append(
                cmath.rect(hit[0].magnitude_pu, math.radians(hit[0].phase_deg))
                if hit
                else 0j
            )
        z = c[0] * (1 - w) + c[1] * w
        comps.append((order, abs(z), math.degrees(cmath.phase(z))))
    return _spec(comps)


def test_time_varying_raises_in_solve_and_oracle():
    tv = TimeVaryingSpectrum(schedule={0: LOW})
    for field, value in (("spectrum", tv), ("spectrum_per_phase", {Phase.B: tv})):
        grid = _grid(_load(**{field: value}), ABC)
        with pytest.raises(
            ModelingError, match=r"Device 30: spectrum kind 'time_varying'"
        ):
            solve_harmonic_flow(grid, ORDERS, slack="norton")
        with pytest.raises(
            ModelingError, match=r"Device 30: spectrum kind 'time_varying'"
        ):
            numpy_harmonic_voltages(grid, None, ORDERS, slack="norton")


@pytest.mark.parametrize("per_phase", [False, True])
def test_equal_levels_reproduce_the_static_spectrum(per_phase):
    lv, st = (
        _lv(points=(LOW, LOW)),
        _spec([(c.order, c.magnitude_pu, c.phase_deg) for c in LOW.components]),
    )
    wrap = (
        (lambda s: {"spectrum_per_phase": {ph: s for ph in ABC}})
        if per_phase
        else (lambda s: {"spectrum": s})
    )
    v_lv, v_st = _solve(wrap(lv)), _solve(wrap(st))
    assert v_st[1:].abs().max() > 1e-3
    torch.testing.assert_close(v_lv, v_st, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("loading", [0.6, 0.35])
def test_interpolation_is_on_the_complex_phasor(loading):
    """h5 turns through ±180° between the levels; h7 fades to zero at the top level."""
    v_lv = _solve({"spectrum": _lv()}, p_w=loading * P_NOM)
    v_ref = _solve({"spectrum": _static_at(loading)}, p_w=loading * P_NOM)
    torch.testing.assert_close(v_lv, v_ref, rtol=1e-12, atol=1e-12)


def test_batched_loading_evaluates_per_scenario_and_extrapolation_modes():
    loadings = [0.1, 0.6, 1.5]
    p = torch.tensor([x * P_NOM for x in loadings], dtype=torch.float64)
    v = _solve({"spectrum": _lv()}, p_w=p)  # [B, H, N]
    for b, x in enumerate(loadings):  # constant extrapolation clamps to the end levels
        torch.testing.assert_close(
            v[b],
            _solve({"spectrum": _static_at(x)}, p_w=x * P_NOM),
            rtol=1e-11,
            atol=1e-11,
        )
    lv_extend = {"spectrum": _lv(extrapolation=ExtrapolationMethod.EXTEND)}
    v_ext = _solve(lv_extend, p_w=1.5 * P_NOM)
    assert not torch.allclose(v_ext, v[2])  # the end segment continues past the level
    with pytest.raises(InputError, match="extrapolation='error'"):
        _solve({"spectrum": _lv(extrapolation=ExtrapolationMethod.ERROR)}, p_w=p)


def test_nearest_picks_the_closest_level():
    lv = {"spectrum": _lv(interpolation=InterpolationMethod.NEAREST)}
    st = {
        "spectrum": _spec(
            [(c.order, c.magnitude_pu, c.phase_deg) for c in HIGH.components]
        )
    }
    torch.testing.assert_close(
        _solve(lv, p_w=0.7 * P_NOM), _solve(st, p_w=0.7 * P_NOM), rtol=1e-12, atol=1e-12
    )
    with pytest.raises(ModelingError, match="'cubic' is not supported"):
        _solve({"spectrum": _lv(interpolation=InterpolationMethod.CUBIC)})


def test_rating_base(caplog):
    load = _load(spectrum=_lv())
    assert spectrum_rating(load) == (P_NOM, "p_nom_w")
    gen = Generator(id=40, node=2, phases=ABC, p_nom_w=3000.0, spectrum=_lv())
    with caplog.at_level(logging.WARNING, logger="pgml"):
        assert spectrum_rating(gen) == (3000.0, "p_nom_w")
    assert any(
        "Generator 40" in r.message and "no rating" in r.message for r in caplog.records
    )


def test_oracle_matches_the_solve_for_a_load_varying_spectrum():
    for spectrum_kw in (
        {"spectrum": _lv(levels=(0.5, 1.5))},
        {"spectrum_per_phase": {Phase.A: _lv(levels=(0.5, 1.5)), Phase.C: _lv()}},
    ):
        v = _solve(spectrum_kw).detach().numpy()
        grid = _grid(_load(**spectrum_kw), ABC)
        vo = numpy_harmonic_voltages(
            grid, None, ORDERS, slack="norton", v1=v[0], load_shunt="none"
        )
        assert np.abs(v[1]).max() > 1e-3
        np.testing.assert_allclose(vo, v, rtol=0, atol=1e-10 * np.abs(v).max())


def test_gradient_through_the_loading():
    def fn(p):
        return _solve({"spectrum": _lv()}, p_w=p)[1:].abs().reshape(-1)

    p = torch.tensor(0.55 * P_NOM, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (p,), eps=1e-3, atol=1e-8, rtol=1e-4)


def test_evaluate_load_varying_handles_a_zero_phasor_gradient():
    x = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    mag, ang = evaluate_load_varying(_lv(), x, 30)[7]  # h7 is zero at the top level
    (mag + ang).backward()
    assert torch.isfinite(x.grad)


def test_oracle_reads_spectrum_per_phase_like_the_solve():
    """Asymmetric per-phase static spectra (A and C differ, B none) agree solve vs oracle."""
    spp = {
        Phase.A: _spec([(1, 1.0, 0.0), (5, 0.3, 10.0), (7, 0.1, 5.0)]),
        Phase.C: _spec([(1, 1.0, 0.0), (5, 0.05, -40.0)]),
    }
    v = _solve({"spectrum_per_phase": spp}).detach().numpy()
    vo = numpy_harmonic_voltages(
        _grid(_load(spectrum_per_phase=spp), ABC),
        None,
        ORDERS,
        slack="norton",
        v1=v[0],
        load_shunt="none",
    )
    assert np.abs(v[1]).max() > 1e-3
    np.testing.assert_allclose(vo, v, rtol=0, atol=1e-10 * np.abs(v).max())


def _hcfg(**kw) -> ScenarioConfig:
    return ScenarioConfig(
        n_samples=4,
        seed=0,
        parameters=[
            ParameterSpec(
                name="hm",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.5, high=1.5),
                field="h_mag",
                orders=[5],
                **kw,
            )
        ],
    )


def test_sampler_refuses_devices_it_cannot_seed():
    per_phase = _grid(_load(spectrum_per_phase={Phase.A: _spec([(5, 0.2, 0.0)])}), ABC)
    with pytest.raises(ModelingError, match="spectrum_per_phase"):
        sample(per_phase, _hcfg(mode="absolute"))
    varying = _grid(_load(spectrum=_lv()), ABC)
    with pytest.raises(
        ModelingError, match="'load_varying' spectrum depends on the solved"
    ):
        sample(varying, _hcfg(mode="scale"))
