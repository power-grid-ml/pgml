"""A backward pass preserves the physical model chosen for its forward solve."""

import yaml
import torch

from pgml import defaults
from pgml.defaults import use_preset
from pgml.schemas import Phase
from pgml.solver import solve_power_flow
from tests.differentiability.test_transformer_gradcheck import _dyn_grid


def test_backward_after_preset_exit_matches_backward_inside():
    def solve():
        g = _dyn_grid((Phase.A, Phase.B, Phase.C))
        loss = torch.tensor(1e-4, dtype=torch.float64, requires_grad=True)
        g.branches[0].magnetizing_conductance_s = loss
        return loss, solve_power_flow(
            g, slack="ideal", dtype=torch.complex128
        ).v.abs().sum()

    with use_preset("opendss"):
        leaf, value = solve()
    outside = torch.autograd.grad(value, leaf)[0]
    with use_preset("opendss"):
        leaf, value = solve()
        inside = torch.autograd.grad(value, leaf)[0]
    torch.testing.assert_close(outside, inside, rtol=1e-12, atol=1e-12)


def test_backward_after_defaults_reload_uses_forward_snapshot(tmp_path, monkeypatch):
    """A delayed adjoint is independent of a process-wide defaults-source change."""

    def solve():
        grid = _dyn_grid((Phase.A, Phase.B, Phase.C))
        loss = torch.tensor(1e-4, dtype=torch.float64, requires_grad=True)
        grid.branches[0].magnetizing_conductance_s = loss
        value = (
            solve_power_flow(grid, slack="ideal", dtype=torch.complex128).v.abs().sum()
        )
        return loss, value

    expected_leaf, expected_value = solve()
    expected = torch.autograd.grad(expected_value, expected_leaf)[0]
    delayed_leaf, delayed_value = solve()

    changed = yaml.safe_load(yaml.safe_dump(defaults.defaults()))
    changed["transformer"]["magnetizing_placement"]["value"] = "to_terminal"
    path = tmp_path / "changed-defaults.yaml"
    path.write_text(yaml.safe_dump(changed))
    monkeypatch.setenv("PGML_DEFAULTS", str(path))
    actual = None
    try:
        defaults.reload(str(path))
        assert defaults.get("transformer.magnetizing_placement") == "to_terminal"
        actual = torch.autograd.grad(delayed_value, delayed_leaf)[0]
    finally:
        monkeypatch.delenv("PGML_DEFAULTS", raising=False)
        defaults.reload()

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
