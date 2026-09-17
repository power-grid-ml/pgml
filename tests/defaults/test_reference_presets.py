"""Reference presets select physics without changing explicit component data."""

import pytest
import torch

from pgml import defaults
from pgml.assembly import assemble_network_ybus
from pgml.errors import ConfigurationError
from pgml.geometry.carson import line_constants, series_impedance
from pgml.geometry.sequence import zero_sequence_harmonic_z
from pgml.multigrid import merge_grids
from pgml.schemas import Grid, LineGeometry
from tests.differentiability.test_carson_gradcheck import _geom_grid


def test_preset_fallback_nesting_and_exception_restore():
    baseline = defaults.defaults()
    with defaults.use_preset("opendss"):
        assert defaults.get("transformer.magnetizing_placement") == "to_terminal"
        assert (
            defaults.get("solver.ift.jacobian_budget_mb")
            == baseline["solver"]["ift"]["jacobian_budget_mb"]["value"]
        )
        assert (
            defaults.resolve("transformer.magnetizing_placement", explicit="split")
            == "split"
        )
        with pytest.raises(RuntimeError), defaults.use_preset("pgml"):
            assert defaults.get("transformer.magnetizing_placement") == "split"
            raise RuntimeError("restore context")
        assert defaults.get("transformer.magnetizing_placement") == "to_terminal"
    assert defaults.defaults() is baseline
    with pytest.raises(ConfigurationError), defaults.use_preset("typo"):
        pass


def test_earth_return_law_default_guard_and_runtime_preset():
    """Shipped law is linear; the OpenDSS preset selects the unguarded sub-linear law."""
    f = torch.tensor([50.0, 250.0, 1250.0], dtype=torch.float64)
    x0 = 0.1662e-3  # a cable-like X0, far below the deep-earth reactance
    args = (0.648e-3, x0, 50.0, f)
    shipped = zero_sequence_harmonic_z(*args, skin=False)
    guarded = zero_sequence_harmonic_z(
        *args, skin=False, x0_frequency="carson_sublinear"
    )
    with defaults.use_preset("opendss"):
        assert defaults.get("line.earth_return.x0_frequency") == "carson_sublinear"
        unguarded = zero_sequence_harmonic_z(*args, skin=False)
    assert defaults.get("line.earth_return.x0_frequency") == "linear"
    assert defaults.get("line.earth_return.x0_nonnegative") is True
    # Linear law: X0(h) = h * X0 at every order.
    torch.testing.assert_close(shipped.imag, x0 * f / 50.0, rtol=1e-14, atol=0.0)
    assert guarded[0] == unguarded[0] == shipped[0]
    assert unguarded[-1].imag < 0
    assert guarded[-1].imag == 0
    torch.testing.assert_close(guarded.real, unguarded.real)
    torch.testing.assert_close(shipped.real, unguarded.real)


def test_public_carson_helpers_resolve_runtime_preset_and_preserve_explicit_args():
    """Omitted helper arguments follow the context; explicit model and band still win."""
    rdt = torch.float64
    x = torch.tensor([0.0], dtype=rdt)
    y = torch.tensor([10.0], dtype=rdt)
    gmr = torch.tensor([0.008], dtype=rdt)
    rdc = torch.tensor([2.0e-4], dtype=rdt)
    radius = torch.tensor([0.012], dtype=rdt)
    freqs = torch.tensor([1250.0], dtype=rdt)
    args = (x, y, gmr, rdc, 100.0, freqs)

    z_gmr = series_impedance(*args, radius=radius, internal_inductance="gmr")
    z_reference = series_impedance(
        *args, radius=radius, internal_inductance="gmr_power_frequency"
    )
    assert not torch.equal(z_gmr, z_reference)

    with defaults.use_preset("opendss"):
        z_series = series_impedance(*args, radius=radius)
        z_line, _ = line_constants(x, y, gmr, rdc, radius, 100.0, freqs, 1)
        z_explicit = series_impedance(*args, radius=radius, internal_inductance="gmr")
        z_explicit_band = series_impedance(
            *args,
            radius=radius,
            internal_inductance="gmr_power_frequency",
            power_frequency_band_hz=(40.0, 2000.0),
        )

    assert torch.equal(z_series, z_reference)
    assert torch.equal(z_line, z_reference)
    assert torch.equal(z_explicit, z_gmr)
    assert torch.equal(z_explicit_band, z_gmr)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mixed_geometry_models_assemble_separately_and_roundtrip(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    grids = []
    for model in ("gmr", "gmr_skin", "bessel", "gmr_power_frequency"):
        grid = _geom_grid(1.2e-4)
        grid.branches[0].conductor_geometry.internal_inductance = model
        grids.append(Grid.model_validate_json(grid.model_dump_json()))
    f = torch.tensor([50.0, 1250.0], dtype=torch.float64)
    with defaults.use_preset("opendss"):
        parts = [assemble_network_ybus(g, f, dtype=torch.complex128).Y for g in grids]
        merged = merge_grids(grids)
        actual = assemble_network_ybus(
            merged.grid, f, dtype=torch.complex128, device=device
        ).Y.cpu()
    expected = torch.stack([torch.block_diag(*(p[h] for p in parts)) for h in range(2)])
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(parts[0], parts[2])
    with pytest.raises(ValueError):
        LineGeometry(conductors=[], internal_inductance="unknown")


def test_prepared_system_rejects_different_preset():
    from pgml.solver import prepare_power_flow, solve_power_flow
    from pgml.errors import InputError

    grid = _geom_grid(1.2e-4)
    system = prepare_power_flow(grid)
    with (
        defaults.use_preset("opendss"),
        pytest.raises(InputError, match="modeling defaults"),
    ):
        solve_power_flow(grid, system=system)
