"""The explicit-value batch builder: ``batch_from_values``.

Covers what the builder promises — absent component / absent field keeps the nominal,
the declared step axis drives the rank of the solved result, ids are resolved against the
grid, and the values pass through to the solver on the autograd tape.
"""

from __future__ import annotations

import pytest
import torch

from pgml.errors import InputError
from pgml.scenarios import (
    batch_from_values,
    read_dataset,
    run_scenarios,
    write_dataset,
)
from pgml.solver import solve_power_flow

CDT = torch.complex128


def test_absent_field_keeps_the_nominal(grid3):
    """A field no mapping names is not written at all, so the solver keeps the nameplate."""
    batch = batch_from_values(
        grid3, n_samples=3, p_w={10: torch.tensor([1.0e3, 2.0e3, 3.0e3])}
    )
    assert set(batch.operating_point) == {10}
    assert set(batch.operating_point[10]) == {"p_w"}
    assert batch.batch_shape == (3,)

    res = run_scenarios(grid3, batch, calculation="power_flow", dtype=CDT)
    nominal = solve_power_flow(grid3, dtype=CDT)
    # scenario whose p_w equals the nameplate reproduces the unbatched solve
    batch_at_nominal = batch_from_values(
        grid3, n_samples=1, p_w={10: torch.tensor([float(grid3.appliances[1].p_nom_w)])}
    )
    res_at_nominal = run_scenarios(
        grid3, batch_at_nominal, calculation="power_flow", dtype=CDT
    )
    torch.testing.assert_close(
        res_at_nominal.v.reshape(-1), nominal.v.reshape(-1), rtol=1e-9, atol=1e-9
    )
    assert res.v.shape[0] == 3


def test_unknown_component_id_raises(grid3):
    """A typo in a component id must not be silently ignored by the solver."""
    with pytest.raises(InputError, match="not an in-service"):
        batch_from_values(grid3, n_samples=2, p_w={9999: torch.zeros(2)})


def test_declared_step_axis_yields_a_four_dimensional_result(grid3):
    """``n_steps`` alone declares the sequence rank: v is ``[B, T, H, N]``."""
    b, t = 2, 4
    mag = torch.full((b, t), 0.1, dtype=torch.float64)
    phase = torch.zeros((b, t), dtype=torch.float64)
    batch = batch_from_values(
        grid3,
        n_samples=b,
        n_steps=t,
        p_w={10: torch.full((b, t), 1.5e3, dtype=torch.float64)},
        harmonic_injection={10: {5: (mag, phase)}},
        shared_samples={"time_s": torch.arange(t, dtype=torch.float64) * 900.0},
    )
    assert batch.batch_shape == (b, t)
    res = run_scenarios(
        grid3, batch, calculation="harmonic", harmonic_orders=[1, 5], dtype=CDT
    )
    assert res.v.shape == (b, t, 2, res.v.shape[-1])
    assert res.converged


def test_a_batch_shape_mismatch_raises(grid3):
    """A tensor that cannot broadcast against the declared batch shape is rejected."""
    with pytest.raises(InputError, match="does not broadcast"):
        batch_from_values(grid3, n_samples=4, p_w={10: torch.zeros(3)})
    with pytest.raises(InputError, match="does not broadcast"):
        batch_from_values(grid3, n_samples=2, n_steps=3, p_w={10: torch.zeros((2, 5))})


def test_a_record_cannot_be_both_per_scenario_and_shared(grid3):
    """The per-scenario / shared split is a declaration, so it must be unambiguous."""
    with pytest.raises(InputError, match="both per-scenario"):
        batch_from_values(
            grid3,
            n_samples=2,
            samples={"x": torch.zeros(2)},
            shared_samples={"x": torch.zeros(2)},
        )


def test_shared_records_survive_a_round_trip_at_batch_length(grid3, tmp_path):
    """A shared record whose length equals B reads back with its own shape.

    Classified by shape alone such a record would become a per-scenario column and come
    back as B scalars; the declaration is what keeps it intact.
    """
    b = 2
    ids = torch.tensor([10, 11], dtype=torch.long)  # numel == B on purpose
    batch = batch_from_values(
        grid3,
        n_samples=b,
        p_w={10: torch.tensor([1.0e3, 2.0e3])},
        samples={"load_p": torch.tensor([1.0e3, 2.0e3])},
        shared_samples={"load_ids": ids},
    )
    res = run_scenarios(grid3, batch, calculation="power_flow", dtype=CDT)
    loaded = read_dataset(write_dataset(res, tmp_path))
    assert torch.equal(loaded.samples["load_ids"], ids)
    assert loaded.samples["load_p"].shape == (b,)
    assert loaded.config is None  # no config supplied, and none invented


def test_values_stay_on_the_autograd_tape(grid3):
    """Gradients flow from the supplied tensors through the batched solve."""
    p = torch.tensor([1.0e3, 2.0e3], dtype=torch.float64, requires_grad=True)
    batch = batch_from_values(grid3, n_samples=2, p_w={10: p})
    res = run_scenarios(grid3, batch, calculation="power_flow", dtype=CDT)
    res.v.abs().sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0


def test_a_prebuilt_batch_is_validated_against_the_grid_it_runs_on(grid3, grid_3ph):
    """Running a batch on a different grid fails before the solver sees it."""
    batch = batch_from_values(grid3, n_samples=2, p_w={10: torch.zeros(2)})
    with pytest.raises(InputError, match="not in-service appliances"):
        run_scenarios(grid_3ph, batch, calculation="power_flow", dtype=CDT)
