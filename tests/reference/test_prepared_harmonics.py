"""Prepared harmonic solves must use current values, never input identity."""

from dataclasses import replace

import pytest
import torch

from pgml import defaults
from pgml.solver import (
    HarmonicFlowSystem,
    NodeHarmonicSource,
    assemble_harmonic_system,
    prepare_harmonic_flow,
    solve_harmonic_flow,
)
from pgml.schemas.grid_schema import LoadModel

from tests.reference.test_harmonic_flow import _grid, _numpy_harmonic_v_ld
from tests.reference.test_harmonic_shunt_woodbury import (
    _sparse_load_grid,
    _operating_point,
)


def _solve(grid, cache=None, orders=(1, 5, 7), **kwargs):
    return solve_harmonic_flow(
        grid, orders, system=cache, slack="norton", criticality="never", **kwargs
    )


def test_source_only_reuses_assembly_and_factors_and_matches_numpy():
    grid = _grid()
    cache = prepare_harmonic_flow(grid, [1, 5, 7], slack="norton")
    result = _solve(grid, cache)
    oracle = _numpy_harmonic_v_ld(complex(result.v[0, 1]), [5, 7])
    for k, h in enumerate([5, 7], 1):
        assert complex(result.v[k, 1]) == pytest.approx(oracle[h], rel=1e-7)
    inj = {2: {1: (1.0, 0.0), 5: (0.3, 23.0), 7: (0.2, -11.0)}}
    changed = _solve(grid, cache, harmonic_injection=inj)
    torch.testing.assert_close(changed.v, _solve(grid, harmonic_injection=inj).v)
    assert not torch.equal(changed.v, result.v)
    assert cache.stats["harmonic_network_misses"] == 1
    assert cache.stats["harmonic_factors_misses"] == 1
    assert cache.stats["harmonic_factors_hits"] == 2
    assert cache.stats["fundamental_hits"] == 2


@pytest.mark.parametrize("edit", ["replace", "in_place", "remove", "add"])
@pytest.mark.parametrize("argument", ["param_overrides", "branch_states"])
def test_changed_override_or_state_rebuilds(edit, argument):
    grid, cache = _grid(), HarmonicFlowSystem()
    key = (
        ("line", 1, "series_resistance_ohm_per_m")
        if argument == "param_overrides"
        else 1
    )
    initial = (
        torch.tensor([[0.4]], dtype=torch.float64)
        if argument == "param_overrides"
        else torch.tensor(0.8)
    )
    values = {} if edit == "add" else {key: initial}
    before = _solve(grid, cache, **{argument: values})
    if edit == "in_place":
        initial.mul_(0.5)
    elif edit == "remove":
        values.pop(key)
    else:
        values[key] = initial * 0.5
    changed = _solve(grid, cache, **{argument: values})
    torch.testing.assert_close(changed.v, _solve(grid, **{argument: values}).v)
    assert not torch.equal(changed.v, before.v)
    assert cache.stats["fundamental_misses"] == 2
    assert cache.stats["harmonic_network_misses"] == 2
    assert cache.stats["harmonic_factors_misses"] == 2


@pytest.mark.parametrize(
    "basis,model,refactor",
    [
        ("operating_point", "opendss", True),
        ("nameplate", "opendss", False),
        ("operating_point", "none", False),
    ],
)
def test_load_power_rechecks_admittance(basis, model, refactor):
    grid, cache = _grid(), HarmonicFlowSystem()
    p = torch.tensor(1800.0, dtype=torch.float64)
    kwargs = dict(
        load_shunt=model, load_shunt_basis=basis, operating_point={2: {"p_w": p}}
    )
    _solve(grid, cache, **kwargs)
    p.add_(300)
    result = _solve(grid, cache, **kwargs)
    torch.testing.assert_close(result.v, _solve(grid, **kwargs).v)
    assert cache.stats["harmonic_network_hits"] == 1
    assert cache.stats["harmonic_factors_misses"] == 1 + refactor


@pytest.mark.parametrize(
    "change",
    ["frequency", "orders", "source_z", "rated_voltage", "composition", "connection"],
)
def test_grid_fields_and_frequency_are_not_the_fundamental_fingerprint(change):
    grid, cache = _grid(), HarmonicFlowSystem()
    _solve(grid, cache)
    orders = [1, 5, 7]
    if change == "frequency":
        grid.base_frequency_hz = 60.0
    elif change == "orders":
        orders = [1, 7, 11]
    elif change == "source_z":
        grid.appliances[0].resistance_ohm = [[0.3]]
    elif change == "rated_voltage":
        grid.nodes[1].u_rated_v = 240.0
    elif change == "composition":
        grid.appliances[1].load_model = LoadModel.CONST_IMPEDANCE
    elif change == "connection":
        grid.appliances[1].in_service = False
    result = _solve(grid, cache, orders)
    torch.testing.assert_close(result.v, _solve(grid, orders=orders).v)
    assert cache.stats["harmonic_network_misses"] == 2
    assert cache.stats["harmonic_factors_misses"] == 2


@pytest.mark.parametrize("kind", ["current", "voltage"])
def test_node_source_spectrum_reuses_but_voltage_source_strength_refactors(kind):
    grid, cache = _grid(), HarmonicFlowSystem()
    strength = torch.tensor(1000.0, dtype=torch.float64)
    source = NodeHarmonicSource(
        2, spectrum={5: (0.02, 0)}, source_power_va=strength, kind=kind
    )
    _solve(grid, cache, node_sources=[source])
    source = replace(source, spectrum={5: (0.03, 25)})
    result = _solve(grid, cache, node_sources=[source])
    torch.testing.assert_close(result.v, _solve(grid, node_sources=[source]).v)
    assert cache.stats["harmonic_factors_hits"] == 1
    strength.mul_(1.5)
    result = _solve(grid, cache, node_sources=[source])
    torch.testing.assert_close(result.v, _solve(grid, node_sources=[source]).v)
    assert cache.stats["harmonic_factors_misses"] == (2 if kind == "voltage" else 1)


@pytest.mark.parametrize("backend", ["dense", "sparse", "block"])
@pytest.mark.parametrize("precision", ["full", "mixed"])
def test_backends_and_numerical_options(backend, precision):
    grid, cache = _grid(), HarmonicFlowSystem()
    kwargs = dict(linear_solver=backend, precision=precision)
    if backend == "block":
        kwargs["block_rows"] = [torch.arange(2)]
    _solve(grid, cache, **kwargs)
    result = _solve(grid, cache, **kwargs)
    torch.testing.assert_close(result.v, _solve(grid, **kwargs).v)
    assert cache.stats["harmonic_factors_hits"] == 1
    kwargs["equilibrate"] = "off"
    result = _solve(grid, cache, **kwargs)
    torch.testing.assert_close(result.v, _solve(grid, **kwargs).v)
    assert cache.stats["harmonic_factors_misses"] == 2


@pytest.mark.parametrize("target", ["network", "power", "rhs"])
def test_current_autograd_graph_after_nondifferentiable_warmup(target):
    grid, cache = _grid(), HarmonicFlowSystem()
    _solve(grid, cache)

    def run(leaf, preparation):
        if target == "network":
            kw = {
                "param_overrides": {
                    ("line", 1, "series_resistance_ohm_per_m"): leaf.reshape(1, 1)
                }
            }
        elif target == "power":
            kw = {"operating_point": {2: {"p_w": leaf}}}
        else:
            kw = {"harmonic_injection": {2: {5: (leaf, 0)}}}
        return _solve(grid, preparation, **kw).v[1:].abs().square().sum()

    initial = {"network": 0.5, "power": 2000.0, "rhs": 0.2}[target]
    for _ in range(2):
        a = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        actual, expected = run(a, cache), run(b, None)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, a)[0], torch.autograd.grad(expected, b)[0]
        )
    if target == "rhs":
        assert cache.stats["harmonic_factors_hits"] == 2
    else:
        assert cache.stats["harmonic_factors_bypasses"] == 2


def test_inference_warmup_allows_later_rhs_backward():
    grid, cache = _grid(), HarmonicFlowSystem()
    with torch.inference_mode():
        _solve(grid, cache)
    magnitude = torch.tensor(0.2, requires_grad=True, dtype=torch.float64)
    result = _solve(grid, cache, harmonic_injection={2: {5: (magnitude, 0)}})
    assert torch.isfinite(torch.autograd.grad(result.v.abs().sum(), magnitude)[0])
    assert cache.stats["harmonic_factors_hits"] == 1


def test_lowrank_updates_rebuilt_while_base_reused():
    grid, cache = _sparse_load_grid(), HarmonicFlowSystem()
    p = torch.tensor([1500.0, 2000.0, 2500.0], dtype=torch.float64)
    _solve(grid, cache, operating_point=_operating_point(p))
    p.add_(200)
    result = _solve(grid, cache, operating_point=_operating_point(p))
    torch.testing.assert_close(
        result.v, _solve(grid, operating_point=_operating_point(p)).v
    )
    assert cache.stats["harmonic_factors_hits"] == 1
    assert cache.stats["harmonic_factors_misses"] == 1


def test_retained_bytes_grow_with_entries_and_are_released():
    grid, cache = _grid(), HarmonicFlowSystem()
    assert cache.nbytes() == 0
    _solve(grid, cache, orders=(1, 5, 7))
    two_orders = cache.nbytes()
    # A matrix, its factorization and the key snapshot the validity check keeps.
    y, _, _ = assemble_harmonic_system(grid, [5, 7], _solve(grid).pf.v)
    assert two_orders >= 2 * y.numel() * y.element_size()
    _solve(grid, cache, orders=(1, 3, 5, 7, 9))
    assert cache.nbytes() > two_orders
    cache.clear()
    assert cache.nbytes() == 0


def test_assembly_result_cannot_mutate_cache():
    grid, cache = _grid(), HarmonicFlowSystem()
    result = _solve(grid, cache)
    y, _, _ = assemble_harmonic_system(grid, [5, 7], result.pf.v, system=cache)
    y.zero_()
    torch.testing.assert_close(_solve(grid, cache).v, result.v)
    cache.clear()
    assert cache.stats == {}


def test_changed_defaults_invalidate_preparation():
    grid, cache = _grid(), HarmonicFlowSystem()
    _solve(grid, cache)
    with defaults.use_preset("opendss"):
        result = _solve(grid, cache)
        torch.testing.assert_close(result.v, _solve(grid).v)
    assert cache.stats["harmonic_network_misses"] == 2
    assert cache.stats["harmonic_factors_misses"] == 2


@pytest.mark.parametrize(
    "change", ["resistance", "frequency_model", "zero_output", "disconnect", "setpoint"]
)
def test_der_fields_and_resolved_pv_output(change):
    from tests.reference.test_pv_harmonic_handoff import harmonic_grid, solve
    from pgml.schemas import VoltageRegulation

    grid, cache = (
        harmonic_grid("internal_voltage", regulation=VoltageRegulation(v_set_pu=1.0)),
        HarmonicFlowSystem(),
    )
    solve(grid, system=cache, load_shunt="none")
    gen = grid.appliances[2]
    kwargs = dict(load_shunt="none")
    if change == "resistance":
        gen.harmonic_impedance.resistance_ohm *= 2
    elif change == "frequency_model":
        gen.harmonic_impedance.frequency_model = "opendss_admittance"
    elif change == "zero_output":
        kwargs["operating_point"] = {22: {"p_w": 0.0}}
    elif change == "disconnect":
        gen.in_service = False
    else:
        gen.voltage_regulation.v_set_pu = 1.005
    result = solve(grid, system=cache, **kwargs)
    torch.testing.assert_close(result.v, solve(grid, **kwargs).v)
    pq = grid.model_copy(deep=True)
    pq.appliances[2].voltage_regulation = None
    kwargs["operating_point"] = result.pf.resolved_operating_point(
        kwargs.get("operating_point")
    )
    torch.testing.assert_close(result.v, solve(pq, **kwargs).v, atol=2e-8, rtol=2e-10)
    assert cache.stats["harmonic_factors_misses"] == (
        1 if change in ("zero_output", "setpoint") else 2
    )


def test_fusion_and_zeroed_islands():
    from tests.reference.test_harmonic_shunt_fusion import _grid as fused_grid

    cache = HarmonicFlowSystem()
    for fused in (False, True, False):
        grid = fused_grid(LoadModel.CONST_CURRENT, fused=fused)
        torch.testing.assert_close(_solve(grid, cache).v, _solve(grid).v)
    grid.branches[-1].in_service = False
    actual = _solve(grid, cache, on_disconnected="zero")
    torch.testing.assert_close(actual.v, _solve(grid, on_disconnected="zero").v)
    assert torch.count_nonzero(actual.v[..., -1]) == 0


def test_simulation_observes_inplace_override_changes():
    from pgml import SimulationConfig, simulate

    grid, cache = _grid(), HarmonicFlowSystem()
    r = torch.tensor([[0.5]], dtype=torch.float64)
    kwargs = dict(param_overrides={("line", 1, "series_resistance_ohm_per_m"): r})
    config = SimulationConfig(harmonic_orders=[1, 5, 7], slack="norton")
    simulate(grid, config, harmonic_system=cache, **kwargs)
    r.mul_(2)
    actual = simulate(grid, config, harmonic_system=cache, **kwargs)
    expected = simulate(grid, config, **kwargs)
    torch.testing.assert_close(actual.v, expected.v)
    for a, b in zip(actual.branch_currents(), expected.branch_currents()):
        torch.testing.assert_close(a.i_from, b.i_from)
        torch.testing.assert_close(a.i_to, b.i_to)


def test_numpy_network_parameter_inplace_change():
    import numpy as np

    grid, cache = _grid(), HarmonicFlowSystem()
    grid.branches[0].series_resistance_ohm_per_m = np.array([[0.5]])
    _solve(grid, cache)
    grid.branches[0].series_resistance_ohm_per_m *= 2
    torch.testing.assert_close(_solve(grid, cache).v, _solve(grid).v)
    assert cache.stats["harmonic_factors_misses"] == 2


def test_prepared_parameter_gradcheck():
    grid, cache = _grid(), HarmonicFlowSystem()
    _solve(grid, cache)
    resistance = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda r: _solve(
            grid, cache, param_overrides={("line", 1, "series_resistance_ohm_per_m"): r}
        ).v[1:],
        (resistance,),
        atol=2e-5,
        rtol=2e-4,
    )


def test_streaming_batches_do_not_retain_scenario_factors(monkeypatch):
    import pgml.solver.harmonic_flow as harmonic

    monkeypatch.setattr(harmonic, "_harmonic_system_budget_bytes", lambda: 256)
    grid, cache = _grid(), HarmonicFlowSystem(cache_batched_factors=False)
    for powers in ([1800.0, 2000.0, 2200.0], [1900.0, 2100.0, 2300.0]):
        op = {2: {"p_w": torch.tensor(powers, dtype=torch.float64)}}
        actual = _solve(grid, cache, operating_point=op)
        torch.testing.assert_close(actual.v, _solve(grid, operating_point=op).v)
    assert cache.stats["harmonic_factors_bypasses"] == 6
    assert "harmonic_factors_misses" not in cache.stats
    assert cache.stats["harmonic_network_misses"] == 1


def test_inplace_block_partition_change_and_tensor_dtype_change():
    grid, cache = _grid(), HarmonicFlowSystem()
    rows = torch.arange(2)
    _solve(grid, cache, linear_solver="block", block_rows=[rows])
    rows.copy_(torch.tensor([1, 0]))
    actual = _solve(grid, cache, linear_solver="block", block_rows=[rows])
    torch.testing.assert_close(actual.v, _solve(grid).v)
    assert cache.stats["harmonic_factors_misses"] == 2
    actual = _solve(grid, cache, dtype=torch.complex64)
    torch.testing.assert_close(actual.v, _solve(grid, dtype=torch.complex64).v)
    assert cache.stats["harmonic_factors_misses"] == 3


def test_failed_rebuild_does_not_install_partial_preparation():
    cache = HarmonicFlowSystem()
    cache._get("harmonic_factors", 1, lambda: "old")

    def fail():
        assert "harmonic_factors" not in cache._entries
        raise RuntimeError("factorization failed")

    with pytest.raises(RuntimeError, match="factorization failed"):
        cache._get("harmonic_factors", 2, fail)
    assert cache._get("harmonic_factors", 2, lambda: "new") == "new"
    assert cache.stats["harmonic_factors_misses"] == 3


def test_worker_local_preparations_are_independent():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    ready = Barrier(2)

    def worker(resistance):
        grid, cache = _grid(), HarmonicFlowSystem()
        grid.branches[0].series_resistance_ohm_per_m = [[resistance]]
        ready.wait(timeout=10)
        first = _solve(grid, cache)
        repeated = _solve(grid, cache)
        torch.testing.assert_close(repeated.v, first.v)
        assert cache.stats["harmonic_factors_hits"] == 1
        grid.branches[0].series_resistance_ohm_per_m = [[2 * resistance]]
        changed = _solve(grid, cache)
        torch.testing.assert_close(changed.v, _solve(grid).v)
        assert cache.stats["harmonic_factors_misses"] == 2
        return changed.v

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, [0.25, 0.75]))
    assert not torch.equal(*results)


@pytest.mark.gpu
def test_prepared_cpu_cuda_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    results = []
    for device in ("cpu", "cuda"):
        cache = HarmonicFlowSystem()
        grid = _grid()
        _solve(grid, cache, device=torch.device(device))
        magnitude = torch.tensor(
            0.2, dtype=torch.float64, device=device, requires_grad=True
        )
        result = _solve(
            grid,
            cache,
            device=torch.device(device),
            harmonic_injection={2: {5: (magnitude, 0)}},
        )
        gradient = torch.autograd.grad(result.v.abs().sum(), magnitude)[0]
        results.append((result.v.detach().cpu(), gradient.cpu()))
        assert cache.stats["harmonic_factors_hits"] == 1
    for cpu, cuda in zip(*results):
        torch.testing.assert_close(cpu, cuda, rtol=2e-8, atol=2e-8)
