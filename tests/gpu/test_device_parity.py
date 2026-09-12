"""GPU gate: CPU-vs-CUDA parity of assemble + solve (skips cleanly without CUDA)."""

from __future__ import annotations

import pytest
import torch

from pgml.assembly import (
    assemble_network_ybus,
    assemble_ybus,
    build_injections,
    node_phase_index,
)
from pgml.geometry.carson import INTERNAL_INDUCTANCE_MODELS, line_constants
from pgml.geometry.sequence import positive_sequence_z, sequence_aware_phase_z
from pgml.geometry.synthesis import (
    apply_positive_sequence_harmonic_model,
    apply_sequence_aware_harmonic_model,
)
from pgml.schemas.grid_schema import (
    ComplexTap,
    ConductorPlacement,
    Grid,
    Line,
    LineGeometry,
    Load,
    Node,
    Phase,
    Source,
    Transformer,
    TransformerZeroSeq,
    WindingConnection,
)
from pgml.solver import solve_harmonic, solve_power_flow

from tests.fixtures.tiny_grids import single_phase_chain, three_phase_two_bus

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

GRIDS = [single_phase_chain, three_phase_two_bus]


@pytest.mark.parametrize("grid_fn", GRIDS)
@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_cpu_cuda_parity(grid_fn, dtype):
    grid = grid_fn()
    freqs = [50.0, 250.0]

    # CPU path.
    idx_cpu = node_phase_index(grid)
    f_cpu = torch.tensor(
        freqs, dtype=torch.float64 if dtype == torch.complex128 else torch.float32
    )
    yb_cpu = assemble_ybus(grid, f_cpu, dtype=dtype, device=torch.device("cpu"))
    i_cpu = build_injections(
        grid, f_cpu, idx_cpu, dtype=dtype, device=torch.device("cpu")
    )
    v_cpu = solve_harmonic(yb_cpu.Y, i_cpu)

    # CUDA path.
    dev = torch.device("cuda")
    f_cuda = f_cpu.to(dev)
    idx_cuda = node_phase_index(grid)
    yb_cuda = assemble_ybus(grid, f_cuda, dtype=dtype, device=dev)
    i_cuda = build_injections(grid, f_cuda, idx_cuda, dtype=dtype, device=dev)
    v_cuda = solve_harmonic(yb_cuda.Y, i_cuda)

    assert yb_cuda.Y.device.type == "cuda"
    assert v_cuda.device.type == "cuda"
    assert yb_cuda.Y.dtype == dtype

    tol = 1e-9 if dtype == torch.complex128 else 1e-3
    torch.testing.assert_close(yb_cuda.Y.cpu(), yb_cpu.Y, rtol=tol, atol=tol)
    torch.testing.assert_close(v_cuda.cpu(), v_cpu, rtol=tol, atol=tol)


def test_cuda_ideal_slack_parity():
    grid = single_phase_chain()
    dev = torch.device("cuda")
    f = [50.0]
    idx = node_phase_index(grid)
    slack_row = idx.row(1, grid.nodes[0].phases[0])

    yb_cpu = assemble_ybus(grid, f, dtype=torch.complex128, device=torch.device("cpu"))
    i_cpu = build_injections(
        grid, f, idx, dtype=torch.complex128, device=torch.device("cpu")
    )
    fixed = torch.tensor([slack_row], dtype=torch.int64)
    vfix = torch.tensor([230.0 + 0.0j], dtype=torch.complex128)
    v_cpu = solve_harmonic(yb_cpu.Y, i_cpu, fixed_rows=fixed, v_fixed=vfix)

    yb_cuda = assemble_ybus(
        grid, torch.tensor(f, device=dev), dtype=torch.complex128, device=dev
    )
    i_cuda = build_injections(
        grid, torch.tensor(f, device=dev), idx, dtype=torch.complex128, device=dev
    )
    v_cuda = solve_harmonic(
        yb_cuda.Y, i_cuda, fixed_rows=fixed.to(dev), v_fixed=vfix.to(dev)
    )

    torch.testing.assert_close(v_cuda.cpu(), v_cpu, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_cpu_cuda_positive_sequence_z_parity(dtype):
    """The positive-sequence harmonic line model runs identically on CPU and CUDA."""
    freqs = [50.0, 250.0, 650.0]
    f_cpu = torch.tensor(freqs, dtype=dtype)
    z_cpu = positive_sequence_z(3.6e-4, 3.0e-4, 50.0, f_cpu)
    z_cuda = positive_sequence_z(3.6e-4, 3.0e-4, 50.0, f_cpu.to("cuda"))
    assert z_cuda.device.type == "cuda"
    tol = 1e-9 if dtype == torch.float64 else 1e-4
    torch.testing.assert_close(z_cuda.cpu(), z_cpu, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_cpu_cuda_positive_sequence_assembly_parity(dtype):
    """Assembling a feeder with the positive-sequence skin law matches on CPU/CUDA."""
    grid = single_phase_chain()
    apply_positive_sequence_harmonic_model(grid)
    rdt = torch.float64 if dtype == torch.complex128 else torch.float32
    f = torch.tensor([50.0, 550.0], dtype=rdt)
    yb_cpu = assemble_network_ybus(grid, f, dtype=dtype, device=torch.device("cpu"))
    yb_cuda = assemble_network_ybus(grid, f.to("cuda"), dtype=dtype, device="cuda")
    assert yb_cuda.Y.device.type == "cuda"
    tol = 1e-9 if dtype == torch.complex128 else 1e-3
    torch.testing.assert_close(yb_cuda.Y.cpu(), yb_cpu.Y, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_cpu_cuda_sequence_aware_z_parity(dtype):
    """The sequence-aware phase matrix Z_abc(h) runs identically on CPU and CUDA."""
    f = torch.tensor([50.0, 250.0, 650.0], dtype=dtype)
    z_cpu = sequence_aware_phase_z(0.21e-3, 0.08e-3, 0.82e-3, 0.32e-3, 50.0, f)
    z_cuda = sequence_aware_phase_z(
        0.21e-3, 0.08e-3, 0.82e-3, 0.32e-3, 50.0, f.to("cuda")
    )
    assert z_cuda.device.type == "cuda"
    tol = 1e-9 if dtype == torch.float64 else 1e-4
    torch.testing.assert_close(z_cuda.cpu(), z_cpu, rtol=tol, atol=tol)


def _sequence_aware_grid() -> Grid:
    """Tiny 3-phase grid whose line uses the sequence-aware (Z1/Z0) harmonic model."""
    import math

    f0 = 50.0
    z1c, z0c = complex(0.3e-3, 0.3e-3), complex(0.6e-3, 1.2e-3)
    zs, zm = (z0c + 2 * z1c) / 3, (z0c - z1c) / 3
    w = 2 * math.pi * f0
    ph = (Phase.A, Phase.B, Phase.C)
    grid = Grid(
        base_frequency_hz=f0,
        nodes=[
            Node(id=1, u_rated_v=400.0, phases=ph),
            Node(id=2, u_rated_v=400.0, phases=ph),
        ],
        branches=[
            Line(
                id=10,
                from_node=1,
                to_node=2,
                from_phases=ph,
                to_phases=ph,
                length_m=100.0,
                series_resistance_ohm_per_m=[
                    [zs.real if i == j else zm.real for j in range(3)] for i in range(3)
                ],
                series_inductance_h_per_m=[
                    [(zs.imag if i == j else zm.imag) / w for j in range(3)]
                    for i in range(3)
                ],
                shunt_capacitance_f_per_m=[[0.0] * 3 for _ in range(3)],
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ph,
                u_ref_v=(230.0, 230.0, 230.0),
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=[
                    [1e-3 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                inductance_h=[
                    [1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
            )
        ],
    )
    return apply_sequence_aware_harmonic_model(grid)


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_cpu_cuda_sequence_aware_assembly_parity(dtype):
    """The sequence-aware (Z1/Z0) assembly path is CPU/CUDA identical."""
    grid = _sequence_aware_grid()
    rdt = torch.float64 if dtype == torch.complex128 else torch.float32
    f = torch.tensor([50.0, 350.0, 650.0], dtype=rdt)
    yb_cpu = assemble_network_ybus(grid, f, dtype=dtype, device=torch.device("cpu"))
    yb_cuda = assemble_network_ybus(grid, f.to("cuda"), dtype=dtype, device="cuda")
    assert yb_cuda.Y.device.type == "cuda"
    tol = 1e-9 if dtype == torch.complex128 else 1e-3
    torch.testing.assert_close(yb_cuda.Y.cpu(), yb_cpu.Y, rtol=tol, atol=tol)


def _carson_geometry_grid() -> Grid:
    """Tiny single-phase grid whose line gets Z(h) from Carson conductor geometry."""
    geom = LineGeometry(
        conductors=[
            ConductorPlacement(
                phase=Phase.A,
                x_m=0.0,
                y_m=10.0,
                gmr_m=0.0078,
                radius_m=0.0102,
                r_dc_ohm_per_m=1.2e-4,
            )
        ]
    )
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=12660.0, phases=(Phase.A,)),
            Node(id=2, u_rated_v=12660.0, phases=(Phase.A,)),
        ],
        branches=[
            Line(
                id=20,
                from_node=1,
                to_node=2,
                from_phases=(Phase.A,),
                to_phases=(Phase.A,),
                length_m=1000.0,
                conductor_geometry=geom,
            )
        ],
        appliances=[
            Source(
                id=10,
                node=1,
                phases=(Phase.A,),
                u_ref_v=(12660.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[1e-3]],
            ),
            Load(id=30, node=2, phases=(Phase.A,), p_nom_w=2e5, q_nom_var=5e4),
        ],
    )


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_cpu_cuda_carson_geometry_assembly_parity(dtype):
    """The Carson/Deri geometry assembly path is CPU/CUDA identical."""
    grid = _carson_geometry_grid()
    rdt = torch.float64 if dtype == torch.complex128 else torch.float32
    f = torch.tensor([50.0, 250.0, 750.0], dtype=rdt)
    yb_cpu = assemble_network_ybus(grid, f, dtype=dtype, device=torch.device("cpu"))
    yb_cuda = assemble_network_ybus(grid, f.to("cuda"), dtype=dtype, device="cuda")
    assert yb_cuda.Y.device.type == "cuda"
    tol = 1e-9 if dtype == torch.complex128 else 1e-3
    torch.testing.assert_close(yb_cuda.Y.cpu(), yb_cpu.Y, rtol=tol, atol=tol)


@pytest.mark.parametrize("model", INTERNAL_INDUCTANCE_MODELS)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_cpu_cuda_internal_inductance_parity(model, dtype):
    """Every conductor internal-inductance model is CPU/CUDA identical.

    The frequencies straddle the power-frequency band, so the masked model
    (``"gmr_power_frequency"``) exercises both of its branches on both devices.
    """
    f = torch.tensor([50.0, 250.0, 1250.0, 2500.0], dtype=dtype)
    args = [
        torch.tensor([-1.0, 0.0, 1.0], dtype=dtype),
        torch.tensor([10.0, 10.0, 10.0], dtype=dtype),
        torch.tensor([0.0078, 0.0078, 0.0078], dtype=dtype),
        torch.tensor([1e-4, 1e-4, 1e-4], dtype=dtype),
        torch.tensor([0.0102, 0.0102, 0.0102], dtype=dtype),
    ]
    z_cpu, _ = line_constants(*args, 100.0, f, 3, internal_inductance=model)
    z_cuda, _ = line_constants(
        *[a.to("cuda") for a in args],
        100.0,
        f.to("cuda"),
        3,
        internal_inductance=model,
    )
    assert z_cuda.device.type == "cuda"
    tol = 1e-9 if dtype == torch.float64 else 1e-4
    torch.testing.assert_close(z_cuda.cpu(), z_cpu, rtol=tol, atol=tol)


@pytest.mark.parametrize("grid_fn", GRIDS)
@pytest.mark.parametrize("slack", ["ideal", "norton"])
def test_cpu_cuda_power_flow_parity(grid_fn, slack):
    """Nonlinear const-power power flow gives identical V on CPU and CUDA."""
    grid = grid_fn()
    res_cpu = solve_power_flow(
        grid, slack=slack, dtype=torch.complex128, device=torch.device("cpu")
    )
    res_cuda = solve_power_flow(
        grid, slack=slack, dtype=torch.complex128, device=torch.device("cuda")
    )
    assert res_cuda.v.device.type == "cuda"
    torch.testing.assert_close(res_cuda.v.cpu(), res_cpu.v, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_cpu_cuda_inductive_shunt_parity(dtype):
    """The inductive shunt term ``1/(j*2*pi*f*L)`` is CPU/CUDA identical."""
    from pgml.schemas.grid_schema import ShuntAppliance, ShuntReactor

    ph = (Phase.A, Phase.B, Phase.C)
    grid = Grid(
        base_frequency_hz=50.0,
        nodes=[Node(id=1, u_rated_v=400.0, phases=ph)],
        branches=[
            ShuntReactor(
                id=5,
                from_node=1,
                to_node=1,
                from_phases=ph,
                to_phases=ph,
                conductance_s=[
                    [1e-4 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                capacitance_f=[
                    [1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                inductance_h=[
                    [0.4 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ph,
                u_ref_v=(230.0,) * 3,
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=[
                    [1.0 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                inductance_h=[
                    [1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
            ),
            ShuntAppliance(
                id=9,
                node=1,
                phases=ph,
                conductance_s=[0.0] * 3,
                capacitance_f=[0.0] * 3,
                inductance_h=[0.25] * 3,
            ),
        ],
    )
    rdt = torch.float64 if dtype == torch.complex128 else torch.float32
    f = torch.tensor([50.0, 250.0, 650.0], dtype=rdt)
    yb_cpu = assemble_network_ybus(grid, f, dtype=dtype, device=torch.device("cpu"))
    yb_cuda = assemble_network_ybus(grid, f.to("cuda"), dtype=dtype, device="cuda")
    assert yb_cuda.Y.device.type == "cuda"
    tol = 1e-9 if dtype == torch.complex128 else 1e-3
    torch.testing.assert_close(yb_cuda.Y.cpu(), yb_cpu.Y, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_cpu_cuda_per_line_earth_return_parity(dtype):
    """A per-line earth-return override (tensor coefficients) is CPU/CUDA identical."""
    from pgml.schemas.grid_schema import EarthReturnModel

    grid = _sequence_aware_grid()
    rdt = torch.float64 if dtype == torch.complex128 else torch.float32
    for b in grid.branches:
        if isinstance(b, Line):
            b.earth_return = EarthReturnModel(
                resistance_coeff_ohm_per_m_per_hz=torch.tensor(9.8696e-7, dtype=rdt),
                x0_frequency="carson_sublinear",
            )
    f = torch.tensor([50.0, 250.0, 650.0], dtype=rdt)
    yb_cpu = assemble_network_ybus(grid, f, dtype=dtype, device=torch.device("cpu"))
    grid_cuda = _sequence_aware_grid()
    for b in grid_cuda.branches:
        if isinstance(b, Line):
            b.earth_return = EarthReturnModel(
                resistance_coeff_ohm_per_m_per_hz=torch.tensor(
                    9.8696e-7, dtype=rdt, device="cuda"
                ),
                x0_frequency="carson_sublinear",
            )
    yb_cuda = assemble_network_ybus(grid_cuda, f.to("cuda"), dtype=dtype, device="cuda")
    assert yb_cuda.Y.device.type == "cuda"
    tol = 1e-9 if dtype == torch.complex128 else 1e-3
    torch.testing.assert_close(yb_cuda.Y.cpu(), yb_cpu.Y, rtol=tol, atol=tol)


def _zero_sequence_grid() -> Grid:
    """HV source (coupled Thevenin) -> YNyn transformer (Z0 != Z1) -> unbalanced load.

    Exercises both per-phase MATRIX paths added for the zero sequence: the source's
    symmetric-component Thevenin (off-diagonal R/L) and the transformer's per-phase
    leakage matrix (a batched matrix inverse inside the winding primitive).
    """
    abc = (Phase.A, Phase.B, Phase.C)
    r_self, r_mut = 0.0233, 0.0133
    l_self, l_mut = 4.03e-4, 2.76e-4
    src = Source(
        id=1,
        node=1,
        phases=abc,
        u_ref_v=(20_000.0 / 3**0.5,) * 3,
        u_angle_deg=(0.0, -120.0, 120.0),
        resistance_ohm=[
            [r_self if i == j else r_mut for j in range(3)] for i in range(3)
        ],
        inductance_h=[
            [l_self if i == j else l_mut for j in range(3)] for i in range(3)
        ],
    )
    xfmr = Transformer(
        id=2,
        from_node=1,
        to_node=2,
        from_phases=abc,
        to_phases=abc,
        s_rated_va=4.0e5,
        u_rated_from_v=20_000.0,
        u_rated_to_v=400.0,
        from_connection=WindingConnection.WYE_GROUNDED,
        to_connection=WindingConnection.WYE_GROUNDED,
        series_resistance_ohm=0.004,
        series_inductance_h=4.93e-5,
        tap=ComplexTap(ratio_magnitude=1.0, shift_deg=0.0),
        zero_sequence=TransformerZeroSeq(r0_ohm=0.002, x0_ohm=0.0077),
    )
    load = Load(
        id=3,
        node=2,
        phases=abc,
        p_nom_w=3.0e4,
        q_nom_var=0.0,
        p_nom_per_phase_w=(2.0e4, 7.0e3, 3.0e3),
        q_nom_per_phase_var=(0.0, 0.0, 0.0),
    )
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=20_000.0, phases=abc),
            Node(id=2, u_rated_v=400.0, phases=abc),
        ],
        branches=[xfmr],
        appliances=[src, load],
    )


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_cpu_cuda_zero_sequence_assembly_parity(dtype):
    """Coupled source Thevenin + per-phase transformer leakage assemble identically."""
    grid = _zero_sequence_grid()
    rdt = torch.float64 if dtype == torch.complex128 else torch.float32
    f = torch.tensor([50.0, 150.0, 250.0], dtype=rdt)
    yb_cpu = assemble_ybus(grid, f, dtype=dtype, device=torch.device("cpu"))
    yb_cuda = assemble_ybus(grid, f.to("cuda"), dtype=dtype, device="cuda")
    assert yb_cuda.Y.device.type == "cuda"
    tol = 1e-9 if dtype == torch.complex128 else 1e-3
    torch.testing.assert_close(yb_cuda.Y.cpu(), yb_cpu.Y, rtol=tol, atol=tol)


@pytest.mark.parametrize("slack", ["ideal", "norton"])
def test_cpu_cuda_zero_sequence_power_flow_parity(slack):
    """The unbalanced solve over both matrix paths matches on CPU and CUDA."""
    grid = _zero_sequence_grid()
    res_cpu = solve_power_flow(
        grid, slack=slack, dtype=torch.complex128, device=torch.device("cpu")
    )
    res_cuda = solve_power_flow(
        grid, slack=slack, dtype=torch.complex128, device=torch.device("cuda")
    )
    assert res_cuda.v.device.type == "cuda"
    torch.testing.assert_close(res_cuda.v.cpu(), res_cpu.v, rtol=1e-9, atol=1e-9)
