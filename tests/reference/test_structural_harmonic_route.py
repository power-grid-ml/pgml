"""The harmonic orders on the sparse backend never form a dense ``Y(h)``.

Where the factorization backend resolves to sparse, ``solve_harmonic_flow`` assembles
each order's STRUCTURAL entries over the topology's sparsity pattern and factors them
directly (``lu_factor_values``). The same stamps run in the same order, so the solved
voltages must equal the dense-assembled route's BIT FOR BIT — pinned here on benchmark
feeders with the per-scenario device shunt and on a grid with four-wire terminals, delta
loads, a vector-group transformer, bus fusion and batched branch states.

Both routes are compared at the SAME scenario chunking. The right-hand side ``I(h)``
depends on the chunk size in its last bit (the CPU complex multiply rounds differently
in its vectorized body and its scalar remainder, so an element's result depends on
where it falls in the tensor), and the two routes charge the memory budget differently.

Every case the structural route cannot serve takes the dense route; each is asserted
here by observing which factorization entry point ran.
"""

from __future__ import annotations

import pytest
import torch

import pgml.solver.harmonic_flow as hf
from pgml.grids import synthetic_feeder
from pgml.schemas.grid_schema import (
    ComplexTap,
    Generator,
    Grid,
    HarmonicComponent,
    HarmonicImpedance,
    Line,
    Load,
    Node,
    Phase,
    ShuntAppliance,
    Source,
    SpectrumPoint,
    StaticSpectrum,
    Switch,
    Transformer,
    WindingConnection,
)
from pgml.solver import (
    HarmonicFlowSystem,
    NodeHarmonicSource,
    assemble_harmonic_system,
    lu_factor_system,
    solve_factored,
    solve_harmonic_flow,
)

CDT = torch.complex128
ABC = (Phase.A, Phase.B, Phase.C)
ABCN = (Phase.A, Phase.B, Phase.C, Phase.N)
ORDERS = [1, 3, 5, 7, 11, 13]
SPECTRUM = StaticSpectrum(
    spectrum=SpectrumPoint(
        components=[
            HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
            HarmonicComponent(order=3, magnitude_pu=0.3, phase_deg=20.0),
            HarmonicComponent(order=5, magnitude_pu=0.2, phase_deg=-40.0),
            HarmonicComponent(order=7, magnitude_pu=0.12, phase_deg=75.0),
            HarmonicComponent(order=11, magnitude_pu=0.06, phase_deg=10.0),
            HarmonicComponent(order=13, magnitude_pu=0.04, phase_deg=-5.0),
        ]
    )
)


# ---------------------------------------------------------------------------
# grids
# ---------------------------------------------------------------------------
def _matrix(n, value):
    return [[(value if i == j else 0.0) for j in range(n)] for i in range(n)]


def _line(bid, u, v, phases=ABC, length=250.0):
    n = len(phases)
    return Line(
        id=bid,
        from_node=u,
        to_node=v,
        from_phases=phases,
        to_phases=phases,
        length_m=length,
        series_resistance_ohm_per_m=_matrix(n, 2.0e-4),
        series_inductance_h_per_m=_matrix(n, 8.0e-7),
        shunt_capacitance_f_per_m=_matrix(n, 1.0e-10),
    )


def _mixed_grid() -> Grid:
    """Four-wire terminals, delta loads, a Dyn transformer, an ideal switch, a DER.

    MV source -> Dyn5 transformer -> four-wire LV bus -> four-wire line -> ideal
    switch (fused) -> three-wire line to a delta load. A parallel line (branch 17) is
    the one ``branch_states`` sweeps; a generator carries an explicit harmonic
    impedance; a delta capacitor bank sits beside the delta load.
    """
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=20_000.0, phases=ABC),
            Node(id=2, u_rated_v=400.0, phases=ABCN),
            Node(id=3, u_rated_v=400.0, phases=ABCN),
            Node(id=4, u_rated_v=400.0, phases=ABC),
            Node(id=5, u_rated_v=400.0, phases=ABC),
        ],
        branches=[
            Transformer(
                id=10,
                from_node=1,
                to_node=2,
                from_phases=ABC,
                to_phases=ABC,
                u_rated_from_v=20_000.0,
                u_rated_to_v=400.0,
                s_rated_va=630_000.0,
                series_resistance_ohm=2.0e-3,
                series_inductance_h=2.0e-5,
                from_connection=WindingConnection.DELTA,
                to_connection=WindingConnection.WYE_GROUNDED,
                tap=ComplexTap(ratio_magnitude=1.0, shift_deg=150.0),
            ),
            _line(11, 2, 3, phases=ABCN),
            Switch(
                id=12,
                from_node=3,
                to_node=4,
                from_phases=ABC,
                to_phases=ABC,
                closed=True,
                resistance_ohm=0.0,
            ),
            _line(13, 4, 5),
            _line(17, 2, 5, length=400.0),
        ],
        appliances=[
            Source(
                id=100,
                node=1,
                phases=ABC,
                u_ref_v=(20_000.0,) * 3,
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=_matrix(3, 0.05),
                inductance_h=_matrix(3, 5.0e-4),
            ),
            Load(
                id=200,
                node=3,
                phases=ABC,
                p_nom_w=12.0e3,
                q_nom_var=3.0e3,
                spectrum=SPECTRUM,
            ),
            Load(
                id=201,
                node=5,
                phases=ABC,
                p_nom_w=9.0e3,
                q_nom_var=2.0e3,
                connection=WindingConnection.DELTA,
                spectrum=SPECTRUM,
            ),
            ShuntAppliance(
                id=202,
                node=5,
                phases=ABC,
                conductance_s=(0.0,) * 3,
                capacitance_f=(2.0e-5,) * 3,
                connection=WindingConnection.DELTA,
            ),
            Generator(
                id=203,
                node=4,
                phases=ABC,
                p_nom_w=4.0e3,
                q_nom_var=0.0,
                spectrum=SPECTRUM,
                harmonic_impedance=HarmonicImpedance(
                    resistance_ohm=0.4, inductance_h=1.5e-3
                ),
            ),
        ],
    )


def _loads(grid):
    return [a for a in grid.appliances if isinstance(a, Load)]


def _with_spectra(grid):
    for i, load in enumerate(_loads(grid)):
        if i % 3 == 0:
            load.spectrum = SPECTRUM
    return grid


def _scenarios(grid, b, seed=0):
    """A per-scenario operating point (0.8 ... 1.2 of nameplate) for every load."""
    gen = torch.Generator().manual_seed(seed)
    scale = 0.8 + 0.4 * torch.rand(
        (len(_loads(grid)), b), generator=gen, dtype=torch.float64
    )
    return {
        load.id: {
            "p_w": float(load.p_nom_w) * scale[k],
            "q_var": float(load.q_nom_var or 0.0) * scale[k],
        }
        for k, load in enumerate(_loads(grid))
    }


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
class _Routes:
    """Counts the factorization entry points the harmonic orders reach."""

    def __init__(self, monkeypatch):
        self.structural = 0
        self.dense = 0
        values, system = hf.lu_factor_values, hf.lu_factor_system

        def spy_values(*args, **kwargs):
            self.structural += 1
            return values(*args, **kwargs)

        def spy_system(*args, **kwargs):
            self.dense += 1
            return system(*args, **kwargs)

        monkeypatch.setattr(hf, "lu_factor_values", spy_values)
        monkeypatch.setattr(hf, "lu_factor_system", spy_system)

    def reset(self):
        self.structural = self.dense = 0


def _solve(grid, orders=ORDERS, **kwargs):
    kwargs.setdefault("slack", "ideal")
    kwargs.setdefault("criticality", "never")
    return solve_harmonic_flow(grid, orders, dtype=CDT, **kwargs).v


def _both_routes(monkeypatch, grid, orders=ORDERS, chunk=None, **kwargs):
    """``(structural, dense)`` voltages of one study; the dense run is forced."""
    if chunk is not None:
        monkeypatch.setattr(hf, "_harmonic_chunk", lambda *a: chunk)
        monkeypatch.setattr(hf, "_pattern_harmonic_chunk", lambda *a: chunk)
    routes = _Routes(monkeypatch)
    structural = _solve(grid, orders, **kwargs)
    assert routes.structural > 0 and routes.dense == 0
    with monkeypatch.context() as m:
        m.setattr(hf, "_structural_route", lambda *a: False)
        routes.reset()
        dense = _solve(grid, orders, **kwargs)
        assert routes.structural == 0 and routes.dense > 0
    return structural, dense


def _assert_matches(v, reference):
    """Agreement to the fundamental's convergence tolerance, relative to the voltages.

    A different backend or precision takes the fundamental fixed point to a different
    iterate inside its tolerance, and the harmonic sources scale with it.
    """
    scale = reference.abs().amax()
    torch.testing.assert_close(v, reference, rtol=0.0, atol=1e-8 * float(scale))


def _assert_identical(a, b):
    assert a.shape == b.shape
    assert torch.equal(a, b), f"max |dV| = {(a - b).abs().max().item():.3e}"


# ---------------------------------------------------------------------------
# bit identity
# ---------------------------------------------------------------------------
def test_ieee33_per_scenario_shunt_is_bit_identical(monkeypatch):
    pytest.importorskip("pandapower")
    from pgml.grids import ieee33_geometry_grid

    grid, _ = ieee33_geometry_grid()
    a, b = _both_routes(
        monkeypatch,
        grid,
        linear_solver="sparse",
        operating_point=_scenarios(grid, 16),
    )
    _assert_identical(a, b)


@pytest.mark.parametrize(("n_nodes", "batch", "chunk"), [(98, 8, 3), (392, 2, 1)])
def test_feeders_per_scenario_shunt_are_bit_identical(
    monkeypatch, n_nodes, batch, chunk
):
    """294 and 1,176 rows, chunked at the same scenario count on both routes."""
    grid = _with_spectra(synthetic_feeder(n_nodes))
    a, b = _both_routes(
        monkeypatch,
        grid,
        chunk=chunk,
        linear_solver="sparse",
        operating_point=_scenarios(grid, batch),
    )
    _assert_identical(a, b)
    assert a.shape[0] == batch and torch.isfinite(a).all()


def test_mixed_grid_with_fusion_and_batched_states_is_bit_identical(monkeypatch):
    grid = _mixed_grid()
    states = {17: torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)}
    a, b = _both_routes(
        monkeypatch,
        grid,
        linear_solver="sparse",
        operating_point=_scenarios(grid, 3),
        branch_states=states,
    )
    _assert_identical(a, b)
    assert torch.isfinite(a).all() and a[:, 1:].abs().amax() > 0


@pytest.mark.parametrize("basis", ["operating_point", "nameplate"])
@pytest.mark.parametrize("shunt", ["opendss", "none"])
def test_mixed_grid_every_shunt_configuration_is_bit_identical(
    monkeypatch, basis, shunt
):
    grid = _mixed_grid()
    a, b = _both_routes(
        monkeypatch,
        grid,
        linear_solver="sparse",
        slack="norton",
        operating_point=_scenarios(grid, 4),
        load_shunt=shunt,
        load_shunt_basis=basis,
    )
    _assert_identical(a, b)


def test_voltage_node_source_is_served_bit_identically(monkeypatch):
    """A Thevenin background source adds to the diagonal, which the pattern holds."""
    grid = _mixed_grid()
    source = NodeHarmonicSource(
        node_id=2,
        kind="voltage",
        source_power_va=torch.tensor([2.0e6, 4.0e6, 8.0e6], dtype=torch.float64),
        spectrum={5: (0.02, 10.0), 7: (0.01, -30.0)},
    )
    a, b = _both_routes(
        monkeypatch,
        grid,
        linear_solver="sparse",
        operating_point=_scenarios(grid, 3),
        node_sources=[source],
    )
    _assert_identical(a, b)


def test_voltage_node_source_at_a_fused_node_is_served_bit_identically(monkeypatch):
    """The diagonal add lands at the reduced row when the source sits on a fused node.

    Switch 12 fuses nodes 3 and 4 in ``_mixed_grid``; the source is placed on node 4.
    """
    grid = _mixed_grid()
    source = NodeHarmonicSource(
        node_id=4,
        kind="voltage",
        source_power_va=torch.tensor([2.0e6, 4.0e6, 8.0e6], dtype=torch.float64),
        spectrum={5: (0.02, 10.0), 7: (0.01, -30.0)},
    )
    a, b = _both_routes(
        monkeypatch,
        grid,
        linear_solver="sparse",
        operating_point=_scenarios(grid, 3),
        node_sources=[source],
    )
    _assert_identical(a, b)


def test_deeper_than_flat_injection_batch_is_served_bit_identically(monkeypatch):
    """A per-scenario Y(h) padded to a node-coherent ``[B, T]`` injection batch."""
    grid = _mixed_grid()
    magnitude = torch.linspace(0.05, 0.3, 6, dtype=torch.float64).reshape(3, 2)
    # Every emitting device carries the step axis; the orders are the overridden ones.
    injection = {
        200: {5: (magnitude, 0.0), 7: (0.5 * magnitude, 15.0)},
        201: {5: (0.8 * magnitude, 30.0), 7: (0.3 * magnitude, -20.0)},
        203: {5: (0.2 * magnitude, 0.0), 7: (0.1 * magnitude, 0.0)},
    }
    a, b = _both_routes(
        monkeypatch,
        grid,
        [1, 5, 7],
        linear_solver="sparse",
        operating_point=_scenarios(grid, 3),
        harmonic_injection=injection,
    )
    assert a.shape[:2] == (3, 2)
    _assert_identical(a, b)


def test_structural_chunking_equals_the_whole_batch(monkeypatch):
    """A budget that splits the batch changes nothing but the chunk count."""
    grid = _with_spectra(synthetic_feeder(40))
    op = _scenarios(grid, 5)
    whole = _solve(grid, linear_solver="sparse", operating_point=op)
    monkeypatch.setattr(hf, "_harmonic_system_budget_bytes", lambda: 1)
    chunked = _solve(grid, linear_solver="sparse", operating_point=op)
    torch.testing.assert_close(chunked, whole, rtol=1e-12, atol=1e-12)


def test_matrix_callers_get_the_dense_system_of_the_same_entries():
    """``assemble_harmonic_system`` returns the dense Y(h) the orders were solved from."""
    grid = _mixed_grid()
    op = _scenarios(grid, 3)
    result = solve_harmonic_flow(
        grid, ORDERS, dtype=CDT, linear_solver="sparse", operating_point=op
    )
    y, i, index = assemble_harmonic_system(
        grid,
        ORDERS[1:],
        result.pf.v,
        operating_point=result.pf.resolved_operating_point(op),
    )
    assert y.shape[-2:] == (index.size, index.size)
    from pgml.assembly import ybus_structure

    fac = lu_factor_system(y, backend="sparse", pattern=ybus_structure(grid, index))
    v = result.fusion.prolong(solve_factored(fac, i))
    _assert_identical(v, result.v[:, 1:])


# ---------------------------------------------------------------------------
# the dense route: every case the structural route does not serve
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("backend", "extra"),
    [("dense", {}), ("block", {"block_rows": "all"}), ("auto", {})],
)
def test_dense_and_block_backends_assemble_the_matrix(monkeypatch, backend, extra):
    """``auto`` below the sparse backend's row threshold resolves to dense."""
    grid = _mixed_grid()
    op = _scenarios(grid, 3)
    reference = _solve(grid, linear_solver="sparse", operating_point=op)
    if extra.get("block_rows") == "all":
        from pgml.assembly import node_phase_index

        extra = {"block_rows": [torch.arange(node_phase_index(grid).size)]}
    routes = _Routes(monkeypatch)
    v = _solve(grid, linear_solver=backend, operating_point=op, **extra)
    assert routes.structural == 0 and routes.dense > 0
    _assert_matches(v, reference)


def test_mixed_precision_assembles_the_matrix(monkeypatch):
    grid = _with_spectra(synthetic_feeder(20))
    op = _scenarios(grid, 3)
    reference = _solve(grid, linear_solver="sparse", operating_point=op)
    routes = _Routes(monkeypatch)
    v = _solve(grid, linear_solver="sparse", precision="mixed", operating_point=op)
    assert routes.structural == 0 and routes.dense > 0
    _assert_matches(v, reference)


def test_low_rank_device_shunt_update_assembles_the_matrix(monkeypatch):
    """The Woodbury path reads the shared shunt-free network as a matrix."""
    from tests.reference.test_harmonic_shunt_woodbury import (
        _operating_point,
        _sparse_load_grid,
    )

    grid = _sparse_load_grid()
    op = _operating_point(torch.tensor([1500.0, 2000.0, 2500.0], dtype=torch.float64))
    lowrank = []
    update = hf.low_rank_update
    monkeypatch.setattr(
        hf, "low_rank_update", lambda *a, **k: lowrank.append(1) or update(*a, **k)
    )
    routes = _Routes(monkeypatch)
    v = _solve(
        grid, [1, 5, 7], slack="norton", linear_solver="sparse", operating_point=op
    )
    assert lowrank and routes.structural == 0 and routes.dense == 1
    # The same study without the low-rank update (a rank limit of zero rejects it).
    monkeypatch.setattr(
        hf,
        "_harmonic_shunt_lowrank_terms",
        lambda *a, **k: (None, None, 1),
    )
    routes.reset()
    direct = _solve(
        grid, [1, 5, 7], slack="norton", linear_solver="sparse", operating_point=op
    )
    assert routes.structural > 0
    torch.testing.assert_close(v, direct, rtol=1e-11, atol=1e-11)


def test_cuda_keeps_the_dense_route():
    """The decision is taken before assembly; no pattern ever reaches a CUDA system."""
    cuda = torch.device("cuda")
    assert not hf._structural_route("sparse", cuda, "full")
    assert not hf._structural_route("dense", torch.device("cpu"), "full")
    assert not hf._structural_route("block", torch.device("cpu"), "full")
    assert not hf._structural_route("sparse", torch.device("cpu"), "mixed")
    assert hf._structural_route("sparse", torch.device("cpu"), "full")
    assert hf._select_backend("auto", 4096, cuda) == "dense"
    assert hf._select_backend("auto", 4096, torch.device("cpu")) == "sparse"
    assert hf._select_backend("auto", 511, torch.device("cpu")) == "dense"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_study_matches_the_structural_cpu_study():
    grid = _mixed_grid()
    op = _scenarios(grid, 3)
    cpu = _solve(grid, linear_solver="sparse", operating_point=op)
    op_gpu = {
        k: {f: t.to("cuda") for f, t in fields.items()} for k, fields in op.items()
    }
    gpu = _solve(grid, device=torch.device("cuda"), operating_point=op_gpu)
    _assert_matches(gpu.cpu(), cpu)


# ---------------------------------------------------------------------------
# the preparation cache on the structural route
# ---------------------------------------------------------------------------
def test_prepared_structural_factors_reuse_and_reject_stale_entries(monkeypatch):
    grid = _mixed_grid()
    op = _scenarios(grid, 3)
    # Per-scenario factors are retained only on request: replaying one batch is the case.
    cache = HarmonicFlowSystem(cache_batched_factors=True)
    routes = _Routes(monkeypatch)
    first = _solve(grid, linear_solver="sparse", operating_point=op, system=cache)
    again = _solve(grid, linear_solver="sparse", operating_point=op, system=cache)
    _assert_identical(again, first)
    assert cache.stats["harmonic_factors_hits"] == 1
    assert routes.structural == 1 and routes.dense == 0
    # The retained network is its structural entries, not an N x N matrix per order.
    network = cache._entries["harmonic_network"][1]
    assert network.values.ndim == 2
    # An in-place change of one load's power moves the per-scenario shunt entries: the
    # value-keyed factors must be rebuilt, never served stale.
    op[200]["p_w"].mul_(1.1)
    changed = _solve(grid, linear_solver="sparse", operating_point=op, system=cache)
    assert cache.stats["harmonic_factors_misses"] == 2
    _assert_identical(changed, _solve(grid, linear_solver="sparse", operating_point=op))
    assert not torch.equal(changed, first)
    # The dense route on the same preparation keys its own representation.
    with monkeypatch.context() as m:
        m.setattr(hf, "_structural_route", lambda *a: False)
        dense = _solve(grid, linear_solver="sparse", operating_point=op, system=cache)
    _assert_identical(dense, changed)
    assert cache.stats["harmonic_network_misses"] == 2
