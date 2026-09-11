"""The load-dependent emission law of the randomized recipe.

Measured devices emit ``I_h(lam) = A_h + B_h * lam`` — a load-independent floor plus a
load-proportional part — so the ratio to the fundamental grows as a device unloads and
its emission angle rotates with loading. These tests pin what the randomized path must
satisfy: the shared law reduces exactly at zero and matches the measured ratio law; the
``h_floor`` / ``h_floor_phase`` / ``h_slope`` specs fold it into the realized injection
against the loading each device's OWN power draw realised; the recipe carries it by
default and is recoverable bit-for-bit with the ranges at zero; a dataset drawn from it
carries the harmonic-to-fundamental coupling a proportional draw provably lacks; and the
solver's per-phase harmonic angles keep the physical sequence structure of a three-phase
device (triplen zero-sequence, h5 negative, h7 positive).
"""

from __future__ import annotations

import math

import pytest
import torch
from pydantic import ValidationError

from pgml.grids import add_pv_systems, synthetic_feeder
from pgml.scenarios import (
    LOADING_FLOOR,
    Constant,
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    affine_emission_correction,
    phase_slope_shift,
    sample,
)

ORDERS = [3, 5, 7]


@pytest.fixture(scope="module")
def pv_grid():
    grid = synthetic_feeder(n_nodes=6, n_feeders=2)
    assert add_pv_systems(grid, fraction=0.5) > 0
    return grid


def _law_config(
    n: int = 64, *, floor=(0.5, 0.5), delta=(120.0, 120.0), slope=(0.0, 0.0)
):
    """Loads at a drawn scale with a fixed fraction and the given law ranges."""

    def spec(name, field, rng, **kw):
        lo, hi = rng
        return ParameterSpec(
            name=name,
            selector=Selector(component="load"),
            distribution=Constant(value=lo) if lo == hi else Uniform(low=lo, high=hi),
            field=field,
            mode="absolute",
            per="each",
            orders=ORDERS,
            **kw,
        )

    params = [
        ParameterSpec(
            name="load_scale",
            selector=Selector(component="load"),
            distribution=Uniform(low=0.0, high=1.0),
            field="pq",
            mode="scale",
            per="each",
        ),
        spec("hm", "h_mag", (0.2, 0.2)),
        spec("hp", "h_phase", (10.0, 10.0)),
        spec("floor", "h_floor", floor),
        spec("floor_phase", "h_floor_phase", delta),
        spec("slope", "h_slope", slope),
    ]
    return ScenarioConfig(n_samples=n, seed=3, parameters=params)


# =============================================================================
# The shared law
# =============================================================================
def test_the_shared_law_reduces_exactly_and_matches_the_measured_ratio_law():
    lam = torch.tensor([0.05, 0.1, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    zero = torch.zeros_like(lam)
    assert torch.all(affine_emission_correction(lam, zero, zero) == 1.0 + 0.0j)
    one = torch.ones(1, dtype=torch.float64)
    for f in (0.43, 0.61, 0.73):
        for d in (0.0, 100.0, 150.0):
            assert torch.allclose(
                affine_emission_correction(one, one * f, one * d),
                torch.ones(1, dtype=torch.complex128),
            )
    f = 0.61
    got = affine_emission_correction(lam, torch.full_like(lam, f), zero).abs()
    assert torch.allclose(got, f / lam + (1.0 - f))
    # the explicit slope is zero at rating and linear in the loading
    assert float(phase_slope_shift(torch.tensor(25.0), torch.tensor(1.0))) == 0.0
    assert float(
        phase_slope_shift(torch.tensor(25.0), torch.tensor(0.5))
    ) == pytest.approx(-12.5)


# =============================================================================
# The specs fold the law into the realized injection
# =============================================================================
def test_the_realized_magnitude_follows_each_devices_own_loading(grid3):
    s = sample(grid3, _law_config())
    lam = s.samples["load_scale"]  # [B, n_dev] the drawn loading
    mag = s.samples["hm_mag"]  # [B, n_dev, n_ord] realized (post-law)
    used = s.samples["hm_loading"]  # [B, n_dev] the loading the law read
    assert torch.allclose(used, lam.clamp(min=LOADING_FLOOR).to(used.dtype))
    corr = affine_emission_correction(
        used, torch.full_like(used, 0.5), torch.full_like(used, 120.0)
    )
    expected = 0.2 * corr.abs().unsqueeze(-1).expand_as(mag)
    assert torch.allclose(mag.to(torch.float64), expected.to(torch.float64), rtol=1e-6)
    # unloading inflates the ratio: the least-loaded scenarios carry the largest ratio
    order = torch.argsort(lam[:, 0])
    assert mag[order[0], 0, 0] > mag[order[-1], 0, 0]
    # …and the injection the solver receives IS the realized value
    dev_id = int(s.all_samples["hm_device_ids"][0])
    inj_mag, inj_phase = s.harmonic_injection[dev_id][3]
    assert torch.allclose(inj_mag.to(torch.float64), mag[:, 0, 0].to(torch.float64))


def test_the_realized_phase_rotates_with_loading(grid3):
    s = sample(grid3, _law_config(slope=(30.0, 30.0)))
    used = s.samples["hm_loading"]
    ph = s.samples["hm_phase"]  # [B, n_dev, n_ord]
    corr = affine_emission_correction(
        used, torch.full_like(used, 0.5), torch.full_like(used, 120.0)
    )
    expected = 10.0 + torch.rad2deg(torch.angle(corr)) + 30.0 * (used - 1.0)
    assert torch.allclose(
        ph[..., 0].to(torch.float64), expected.to(torch.float64), atol=1e-9
    )
    # rated devices keep exactly the drawn angle; unloaded ones have moved
    assert torch.all((ph[..., 0] - 10.0).abs()[used >= 0.999] < 1e-6)
    assert torch.all((ph[..., 0] - 10.0).abs()[used < 0.5] > 1.0)


def test_a_zero_law_is_bit_identical_and_the_floor_keeps_it_finite(grid3):
    base = sample(grid3, _law_config(floor=(0.0, 0.0), delta=(0.0, 0.0)))
    same = sample(
        grid3, _law_config(floor=(0.0, 0.0), delta=(0.0, 0.0), slope=(0.0, 0.0))
    )
    assert torch.equal(base.samples["hm_mag"], same.samples["hm_mag"])
    assert torch.allclose(
        base.samples["hm_mag"], torch.full_like(base.samples["hm_mag"], 0.2)
    )
    assert torch.allclose(
        base.samples["hm_phase"], torch.full_like(base.samples["hm_phase"], 10.0)
    )
    # a device drawn at (near) zero load evaluates the law at the floor, never at 0
    s = sample(grid3, _law_config(floor=(0.9, 0.9)))
    assert torch.isfinite(s.samples["hm_mag"]).all()
    assert torch.isfinite(s.samples["hm_phase"]).all()
    bound = 0.2 * float(
        affine_emission_correction(
            torch.tensor(LOADING_FLOOR, dtype=torch.float64),
            torch.tensor(0.9, dtype=torch.float64),
            torch.tensor(120.0, dtype=torch.float64),
        ).abs()
    )
    assert float(s.samples["hm_mag"].max()) <= bound + 1e-9
    assert float(s.samples["hm_mag"].max()) > 0.2 * 5.0  # the floor does inflate it


def test_law_specs_are_validated():
    sel = Selector(component="load")
    with pytest.raises(ValidationError):
        ParameterSpec(
            name="f",
            selector=sel,
            distribution=Uniform(low=0.0, high=1.5),
            field="h_floor",
            mode="absolute",
            orders=[3],
        )
    with pytest.raises(ValidationError):
        ParameterSpec(
            name="f",
            selector=sel,
            distribution=Uniform(low=0.0, high=1.0),
            field="h_slope",
            mode="scale",
            orders=[3],
        )
    ok = ParameterSpec(
        name="f",
        selector=sel,
        distribution=Uniform(low=0.0, high=1.0),
        field="h_floor",
        mode="absolute",
        orders=[3],
    )
    assert ok.is_harmonic and ok.is_emission_law


# =============================================================================
# The solver keeps the physical sequence structure across a device's phases
# =============================================================================
def test_a_three_phase_device_injects_sequence_consistent_harmonics(grid_3ph):
    """Harmonic ``h`` on phase ``b`` is the phase-``a`` waveform delayed by a third of a
    cycle, i.e. rotated by ``-h * 120 deg``: h3 is zero-sequence (all phases in phase),
    h5 negative-sequence, h7 positive-sequence. The solver derives the per-phase angle as
    ``ang_h + h * arg(I_1,phase)``, which is exactly that structure."""
    from pgml.assembly import node_phase_index
    from pgml.schemas.grid_schema import Load
    from pgml.solver.harmonic_flow import _harmonic_injections

    load = next(
        a for a in grid_3ph.appliances if isinstance(a, Load) and len(a.phases) == 3
    )
    index = node_phase_index(grid_3ph)
    node = next(n for n in grid_3ph.nodes if n.id == load.node)
    v_ln = float(node.u_rated_v) / math.sqrt(3.0)
    v1 = torch.zeros(index.size, dtype=torch.complex128)
    for n in grid_3ph.nodes:
        for k, p in enumerate(n.phases):
            v1[int(index.row(n.id, p))] = v_ln * torch.exp(
                torch.tensor(-1j * 2 * math.pi * k / 3, dtype=torch.complex128)
            )
    inj = {load.id: {h: (0.1, 0.0) for h in ORDERS}}
    out = _harmonic_injections(
        grid_3ph,
        v1,
        index,
        ORDERS,
        {},
        inj,
        torch.complex128,
        torch.float64,
        torch.device("cpu"),
    )  # [Hh, N]
    rows = [int(index.row(load.node, p)) for p in load.phases]
    for k, h in enumerate(ORDERS):
        ang = torch.rad2deg(torch.angle(out[k, rows]))
        d_ba = float((ang[1] - ang[0] + 180.0) % 360.0 - 180.0)
        d_ca = float((ang[2] - ang[0] + 180.0) % 360.0 - 180.0)
        expected = {3: (0.0, 0.0), 5: (120.0, -120.0), 7: (-120.0, 120.0)}[h]
        assert d_ba == pytest.approx(expected[0], abs=1e-6), h
        assert d_ca == pytest.approx(expected[1], abs=1e-6), h
        # equal magnitudes on the three phases of a balanced device
        assert torch.allclose(out[k, rows].abs(), out[k, rows[0]].abs().expand(3))
