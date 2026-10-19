"""Impedance scan: closed-form RLC networks, sequence impedances, layout and contracts.

Every closed form here is the exact phasor expression of the network that the grid
describes, so the comparison tolerance is set by floating-point rounding of a linear solve
(measured worst relative error on these cases: 4e-14 in complex128, tolerance 1e-12).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import warnings

import numpy as np
import pytest
import torch

from pgml import defaults
from pgml.assembly import assemble_ybus
from pgml.errors import InputError
from pgml.geometry import apply_default_harmonic_model
from pgml.grids import synthetic_feeder
from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    Line,
    Load,
    Node,
    Phase,
    ShuntAppliance,
    Source,
    Switch,
)
from pgml.simulation import SimulationConfig
from pgml.studies import ImpedanceScan, impedance_scan
from pgml.studies import impedance as impedance_module
from pgml.studies.impedance import DEFAULT_MAX_ORDER, DEFAULT_ORDER_STEP

from tests.fixtures.tiny_grids import three_phase_two_bus

F0 = 50.0
ABC = (Phase.A, Phase.B, Phase.C)
RTOL = 1e-12


def _sym(diag: float, off: float) -> list[list[float]]:
    return [[diag if i == j else off for j in range(3)] for i in range(3)]


def _source(node: int, r, ind, phases=(Phase.A,)) -> Source:
    n = len(phases)
    u = 230.0 if n == 1 else 400.0 / math.sqrt(3)
    return Source(
        id=100,
        node=node,
        phases=phases,
        u_ref_v=(u,) * n,
        u_angle_deg=(0.0, -120.0, 120.0)[:n],
        resistance_ohm=r,
        inductance_h=ind,
    )


def _node(nid: int, phases=(Phase.A,)) -> Node:
    return Node(id=nid, u_rated_v=230.0 if len(phases) == 1 else 400.0, phases=phases)


# --------------------------------------------------------------------------- closed forms
R_S, L_S, C_1 = 0.05, 1.0e-3, 100.0e-6


def _parallel_rlc_grid() -> Grid:
    """Source R_S + jωL_S in parallel with a capacitor C_1 at the source bus."""
    return Grid(
        base_frequency_hz=F0,
        nodes=[_node(1)],
        branches=[],
        appliances=[
            _source(1, [[R_S]], [[L_S]]),
            ShuntAppliance(
                id=200,
                node=1,
                phases=(Phase.A,),
                conductance_s=(0.0,),
                capacitance_f=(C_1,),
            ),
        ],
    )


def _z_parallel(f: np.ndarray) -> np.ndarray:
    w = 2 * np.pi * f
    return 1.0 / (1.0 / (R_S + 1j * w * L_S) + 1j * w * C_1)


def test_parallel_resonance_matches_closed_form():
    """Whole curve against the closed form; the peak against the analytic maximum.

    Measured (complex128, default 5 Hz grid): curve relative error 9e-15. Refined peak
    0.016 Hz from the true maximum of the closed form (tolerance 0.1 Hz), its ``|Z|``
    0.23 % low (tolerance 1 %). The true maximum sits 2e-5 Hz from
    ``1/(2*pi*sqrt(L*C))`` = 503.29 Hz (Q = 63).
    """
    scan = impedance_scan(_parallel_rlc_grid())
    f = scan.frequencies_hz.numpy()
    z = scan.impedance(1).numpy()
    np.testing.assert_allclose(z, _z_parallel(f), rtol=RTOL, atol=0)

    res = scan.resonances(1)
    assert [r.kind for r in res] == ["parallel"]
    fine = np.linspace(400.0, 600.0, 2_000_001)
    f_true = fine[np.argmax(np.abs(_z_parallel(fine)))]
    f_lc = 1.0 / (2 * np.pi * math.sqrt(L_S * C_1))
    assert abs(f_true - f_lc) < 0.01
    assert abs(res[0].frequency_hz - f_true) < 0.1
    assert res[0].order == pytest.approx(res[0].frequency_hz / F0, rel=1e-15)
    assert res[0].z_ohm == pytest.approx(abs(_z_parallel(np.array(f_true))), rel=1e-2)


L_L, R_L, C_2 = 0.5e-3, 0.02, 40.0e-6


def _series_branch_grid() -> Grid:
    """Source L_S at bus 1, line R_L + jωL_L to bus 2, capacitor C_2 at bus 2."""
    line = Line(
        id=10,
        from_node=1,
        to_node=2,
        from_phases=(Phase.A,),
        to_phases=(Phase.A,),
        length_m=1.0,
        series_resistance_ohm_per_m=[[R_L]],
        series_inductance_h_per_m=[[L_L]],
        shunt_capacitance_f_per_m=[[0.0]],
        harmonic_line_model="naive",
    )
    return Grid(
        base_frequency_hz=F0,
        nodes=[_node(1), _node(2)],
        branches=[line],
        appliances=[
            _source(1, [[R_S]], [[L_S]]),
            ShuntAppliance(
                id=200,
                node=2,
                phases=(Phase.A,),
                conductance_s=(0.0,),
                capacitance_f=(C_2,),
            ),
        ],
    )


def test_series_and_parallel_resonance_of_a_tuned_branch():
    """``Z_s || (Z_line + 1/(jωC))``: a parallel peak at ``1/(2π√((L_s+L_l)C))``, then a
    series dip at ``1/(2π√(L_l C))``. Transfer and far-end impedances in closed form too.

    Measured (default 5 Hz grid): curves 4e-14 relative; refined peak 0.019 Hz and dip
    0.033 Hz from the closed-form extrema (tolerance 0.1 Hz), which themselves sit
    0.021 Hz and 0.009 Hz from the LC formulas (tolerance 0.05 Hz).
    """
    scan = impedance_scan(_series_branch_grid(), transfer=True)
    f = scan.frequencies_hz.numpy()
    w = 2 * np.pi * f
    zs = R_S + 1j * w * L_S
    zb = R_L + 1j * w * L_L
    zc = 1.0 / (1j * w * C_2)
    z11 = zs * (zb + zc) / (zs + zb + zc)
    z22 = zc * (zs + zb) / (zs + zb + zc)
    z12 = zs * zc / (zs + zb + zc)
    np.testing.assert_allclose(scan.impedance(1).numpy(), z11, rtol=RTOL)
    np.testing.assert_allclose(scan.impedance(2).numpy(), z22, rtol=RTOL)
    zt = scan.z_transfer_ohm.numpy()
    np.testing.assert_allclose(zt[:, 0, 1], z12, rtol=RTOL)
    np.testing.assert_allclose(zt[:, 1, 0], z12, rtol=RTOL)

    res = scan.resonances(1)
    assert [r.kind for r in res] == ["parallel", "series"]
    fine = np.linspace(F0, 50 * F0, 4_000_001)
    wf = 2 * np.pi * fine
    zsf, zbf, zcf = R_S + 1j * wf * L_S, R_L + 1j * wf * L_L, 1 / (1j * wf * C_2)
    mag = np.abs(zsf * (zbf + zcf) / (zsf + zbf + zcf))
    f_peak = fine[np.argmax(mag)]
    f_dip = fine[np.argmin(mag)]
    assert abs(f_peak - 1 / (2 * np.pi * math.sqrt((L_S + L_L) * C_2))) < 0.05
    assert abs(f_dip - 1 / (2 * np.pi * math.sqrt(L_L * C_2))) < 0.05
    assert abs(res[0].frequency_hz - f_peak) < 0.1
    assert abs(res[1].frequency_hz - f_dip) < 0.1


Z1_R, Z1_L, Z0_R, Z0_L, C_3 = 0.03, 0.8e-3, 0.09, 2.4e-3, 60.0e-6


def _three_phase_bank_grid() -> Grid:
    """Sequence-aware source (Z1 != Z0) with a grounded-wye capacitor bank at its bus."""
    r = _sym((Z0_R + 2 * Z1_R) / 3, (Z0_R - Z1_R) / 3)
    ind = _sym((Z0_L + 2 * Z1_L) / 3, (Z0_L - Z1_L) / 3)
    return Grid(
        base_frequency_hz=F0,
        nodes=[_node(1, ABC)],
        branches=[],
        appliances=[
            _source(1, r, ind, phases=ABC),
            ShuntAppliance(
                id=200,
                node=1,
                phases=ABC,
                conductance_s=(0.0,) * 3,
                capacitance_f=(C_3,) * 3,
            ),
        ],
    )


def test_sequence_impedances_of_a_capacitor_bank():
    """``Z1 = Z_s1 || 1/(jωC)`` and ``Z0 = Z_s0 || 1/(jωC)`` (grounded wye), resonating at
    different frequencies; the phase matrix is ``A diag(Z0, Z1, Z1) A^-1``.

    Measured: 4e-14 relative on both sequences; refined resonances 0.013 Hz (positive)
    and 0.029 Hz (zero) from the LC formulas (tolerance 0.1 Hz; R/X is small).
    """
    scan = impedance_scan(_three_phase_bank_grid())
    w = 2 * np.pi * scan.frequencies_hz.numpy()
    yc = 1j * w * C_3
    z1 = 1 / (1 / (Z1_R + 1j * w * Z1_L) + yc)
    z0 = 1 / (1 / (Z0_R + 1j * w * Z0_L) + yc)
    np.testing.assert_allclose(scan.impedance(1, "positive").numpy(), z1, rtol=RTOL)
    np.testing.assert_allclose(scan.impedance(1, "zero").numpy(), z0, rtol=RTOL)
    a = np.exp(2j * np.pi / 3)
    amat = np.array([[1, 1, 1], [1, a * a, a], [1, a, a * a]])
    zseq = np.zeros((w.size, 3, 3), dtype=complex)
    zseq[:, 0, 0], zseq[:, 1, 1], zseq[:, 2, 2] = z0, z1, z1
    expected = amat @ zseq @ np.linalg.inv(amat)
    np.testing.assert_allclose(
        scan.driving_point(1).numpy(), expected, rtol=RTOL, atol=1e-15
    )

    f1 = scan.resonances(1, "positive")[0].frequency_hz
    f0_ = scan.resonances(1, "zero")[0].frequency_hz
    assert f1 == pytest.approx(1 / (2 * np.pi * math.sqrt(Z1_L * C_3)), abs=0.1)
    assert f0_ == pytest.approx(1 / (2 * np.pi * math.sqrt(Z0_L * C_3)), abs=0.1)
    # default component of a three-phase node is the positive sequence
    assert scan.resonances(1)[0].frequency_hz == f1


def test_constant_impedance_load_fold():
    """``load_shunt="const_z"``: a load P + jQ (Q > 0) at the source bus folds to
    ``P/V² - jQ/(V² h)``.

    Measured: rounding-level agreement (tolerance 1e-12 relative). The operating point
    overrides the nameplate, and the explicit argument wins over
    ``config.operating_point``.
    """
    p, q, v = 4.0e3, 1.5e3, 230.0
    grid = Grid(
        base_frequency_hz=F0,
        nodes=[_node(1)],
        branches=[],
        appliances=[
            _source(1, [[R_S]], [[L_S]]),
            Load(id=300, node=1, phases=(Phase.A,), p_nom_w=1.0, q_nom_var=0.0),
        ],
    )
    op = {300: {"p_w": p, "q_var": q}}
    f = np.array([50.0, 150.0, 1234.5])
    h = f / F0
    y_load = p / v**2 - 1j * q / (v**2 * h)
    expected = 1 / (1 / (R_S + 2j * np.pi * f * L_S) + y_load)

    scan = impedance_scan(
        grid, frequencies_hz=f, operating_point=op, load_shunt="const_z"
    )
    np.testing.assert_allclose(scan.impedance(1).numpy(), expected, rtol=RTOL)
    via_cfg = impedance_scan(
        grid,
        frequencies_hz=f,
        config=SimulationConfig(operating_point=op),
        load_shunt="const_z",
    )
    np.testing.assert_allclose(via_cfg.impedance(1).numpy(), expected, rtol=RTOL)
    wins = impedance_scan(
        grid,
        frequencies_hz=f,
        operating_point=op,
        config=SimulationConfig(operating_point={300: {"p_w": 1.0, "q_var": 0.0}}),
        load_shunt="const_z",
    )
    np.testing.assert_allclose(wins.impedance(1).numpy(), expected, rtol=RTOL)


def _single_load_grid(p: float, q: float) -> Grid:
    return Grid(
        base_frequency_hz=F0,
        nodes=[_node(1)],
        branches=[],
        appliances=[
            _source(1, [[R_S]], [[L_S]]),
            Load(id=300, node=1, phases=(Phase.A,), p_nom_w=p, q_nom_var=q),
        ],
    )


def test_default_load_model_is_the_harmonic_flow_shunt():
    """The default load model is ``appliance.harmonic_shunt.model`` (``"opendss"``): the
    admittance ``Y = (P - jQ)/V²`` split into a parallel branch ``(1-s)(G + jB/h)`` and a
    series branch ``1/(R + jhX)`` with ``R + jX = 1/(s Y)``, ``s = 0.5``. Equal to the
    const-Z fold at the fundamental only. ``config.load_shunt`` is honoured and the
    explicit argument wins over it.

    Measured: rounding-level agreement (tolerance 1e-12 relative). At 1234.5 Hz the
    const-Z fold differs from the default model by 24.9 % (``|Z_cz - Z| / |Z|``) on
    this bus, 7.4 % in ``|Z|``.
    """
    p, q, v, s = 4.0e3, 1.5e3, 230.0, 0.5
    grid = _single_load_grid(p, q)
    f = np.array([50.0, 150.0, 1234.5])
    h = f / F0
    y_eq = (p - 1j * q) / v**2
    g, b = y_eq.real, y_eq.imag
    y_par = (1 - s) * (g + 1j * b / h)
    z_ser = 1.0 / (s * y_eq)
    y_ser = 1.0 / (z_ser.real + 1j * h * z_ser.imag)
    z_src = R_S + 2j * np.pi * f * L_S
    expected = 1 / (1 / z_src + y_par + y_ser)

    scan = impedance_scan(grid, frequencies_hz=f)
    np.testing.assert_allclose(scan.impedance(1).numpy(), expected, rtol=RTOL)
    const_z = impedance_scan(grid, frequencies_hz=f, load_shunt="const_z")
    np.testing.assert_allclose(const_z.impedance(1).numpy()[0], expected[0], rtol=RTOL)
    rel = np.abs(const_z.impedance(1).numpy() - expected) / np.abs(expected)
    assert rel[2] == pytest.approx(0.249, abs=1e-3)

    none_cfg = SimulationConfig(load_shunt="none")
    bare = impedance_scan(grid, frequencies_hz=f, config=none_cfg)
    np.testing.assert_allclose(bare.impedance(1).numpy(), z_src, rtol=RTOL)
    wins = impedance_scan(grid, frequencies_hz=f, config=none_cfg, load_shunt="opendss")
    np.testing.assert_allclose(wins.impedance(1).numpy(), expected, rtol=RTOL)


def test_nameplate_basis_ignores_the_operating_point():
    grid = _single_load_grid(4.0e3, 1.5e3)
    f = [150.0, 650.0]
    op = {300: {"p_w": 1.0e3, "q_var": 0.0}}
    plate = impedance_scan(grid, frequencies_hz=f)
    with_op = impedance_scan(grid, frequencies_hz=f, operating_point=op)
    ignored = impedance_scan(
        grid, frequencies_hz=f, operating_point=op, load_shunt_basis="nameplate"
    )
    torch.testing.assert_close(ignored.z_phase_ohm, plate.z_phase_ohm, rtol=0, atol=0)
    assert (with_op.z_phase_ohm[0] - plate.z_phase_ohm[0]).abs().max() > 1e-3


# --------------------------------------------------------------------------- behaviour
def _inverse_per_frequency(y: torch.Tensor) -> torch.Tensor:
    """Dense reference inverse of ``[F, N, N]``, one matrix at a time: a batched CPU
    ``torch.linalg.inv`` hits the multi-threaded batched-LU defect of some LAPACK builds
    (the one ``pgml.solver`` guards), which would corrupt the reference, not the scan."""
    return torch.stack([torch.linalg.inv(y[k]) for k in range(y.shape[0])])


def _with_generator(grid: Grid) -> Grid:
    g = copy.deepcopy(grid)
    g.appliances.append(
        Generator(id=400, node=2, phases=ABC, p_nom_w=20.0e3, q_nom_var=0.0)
    )
    return g


def test_generation_is_open_unless_folded():
    base = three_phase_two_bus()
    with_gen = _with_generator(base)
    f = [50.0, 350.0, 1100.0]
    z_base = impedance_scan(base, frequencies_hz=f).z_positive_ohm
    z_open = impedance_scan(with_gen, frequencies_hz=f).z_positive_ohm
    torch.testing.assert_close(z_open, z_base, rtol=0, atol=0)
    z_base_cz = impedance_scan(
        base, frequencies_hz=f, load_shunt="const_z"
    ).z_positive_ohm
    z_open_cz = impedance_scan(
        with_gen, frequencies_hz=f, load_shunt="const_z"
    ).z_positive_ohm
    z_fold = impedance_scan(
        with_gen, frequencies_hz=f, load_shunt="const_z", fold_generation=True
    ).z_positive_ohm
    torch.testing.assert_close(z_open_cz, z_base_cz, rtol=0, atol=0)
    assert (z_fold - z_base_cz).abs().max() > 1e-6
    with pytest.raises(InputError, match="fold_generation"):
        impedance_scan(with_gen, frequencies_hz=f, fold_generation=True)


def test_inputs_are_not_mutated():
    grid = _with_generator(three_phase_two_bus())
    before = grid.model_dump(mode="json")
    op = {30: {"p_w": 1.0e3, "q_var": 2.0e2}}
    op_before = copy.deepcopy(op)
    cfg = SimulationConfig(operating_point={30: {"p_w": 5.0, "q_var": 0.0}})
    cfg_before = cfg.model_dump()
    impedance_scan(grid, frequencies_hz=[50.0, 250.0], operating_point=op, config=cfg)
    impedance_scan(grid, frequencies_hz=[50.0, 250.0], config=cfg)
    assert grid.model_dump(mode="json") == before
    assert op == op_before
    assert cfg.model_dump() == cfg_before


def test_matches_dense_inverse_of_the_assembled_admittance():
    """With ``load_shunt="const_z"`` the column solve equals the dense inverse of
    ``assemble_ybus`` (with generation left at the nameplate on this load-only grid).
    Measured: identical (0 difference) on this 6-row grid; reciprocity ``Z = Z^T``
    holds to 3e-15 relative."""
    grid = three_phase_two_bus()
    f = [50.0, 450.0, 2500.0]
    scan = impedance_scan(grid, frequencies_hz=f, transfer=True, load_shunt="const_z")
    y = assemble_ybus(grid, f, dtype=torch.complex128).Y
    z_inv = _inverse_per_frequency(y)
    torch.testing.assert_close(scan.z_transfer_ohm, z_inv, rtol=1e-12, atol=0)
    torch.testing.assert_close(
        scan.driving_point(2), z_inv[:, 3:, 3:], rtol=1e-12, atol=0
    )
    # reciprocity: the transfer matrix of a passive network is symmetric
    torch.testing.assert_close(
        scan.z_transfer_ohm, scan.z_transfer_ohm.transpose(1, 2), rtol=1e-12, atol=0
    )


def test_fused_switch_nodes_share_one_impedance():
    """An ideal closed switch fuses its two buses: both report the impedance of the
    fused row, equal to the scan of the grid with the switch's buses merged."""
    grid = three_phase_two_bus()
    grid.nodes.append(_node(3, ABC))
    grid.branches.append(
        Switch(
            id=50,
            from_node=2,
            to_node=3,
            from_phases=ABC,
            to_phases=ABC,
            default_closed=True,
        )
    )
    f = [50.0, 700.0]
    scan = impedance_scan(grid, nodes=[2, 3], frequencies_hz=f)
    torch.testing.assert_close(
        scan.driving_point(2), scan.driving_point(3), rtol=0, atol=0
    )
    plain = impedance_scan(three_phase_two_bus(), nodes=[2], frequencies_hz=f)
    torch.testing.assert_close(
        scan.driving_point(2), plain.driving_point(2), rtol=1e-12, atol=0
    )


def test_frequency_chunks_reproduce_one_batch(monkeypatch):
    """Chunking over frequencies (forced by a tiny memory budget) is invisible.

    Measured: identical (0 difference) with 40 one-frequency chunks."""
    grid = three_phase_two_bus()
    f = np.linspace(50.0, 2500.0, 40)
    whole = impedance_scan(grid, frequencies_hz=f, transfer=True)
    real_get = defaults.get

    def tiny_budget(key, *args, **kwargs):
        if key == "solver.harmonic.system_budget_mb":
            return 1.0e-4  # ~100 bytes: one frequency per chunk
        return real_get(key, *args, **kwargs)

    monkeypatch.setattr(impedance_module.defaults, "get", tiny_budget)
    chunked = impedance_scan(grid, frequencies_hz=f, transfer=True)
    torch.testing.assert_close(
        chunked.z_transfer_ohm, whole.z_transfer_ohm, rtol=1e-14, atol=0
    )


def test_sparse_backend_on_a_larger_feeder():
    """A 200-node feeder (600 rows) takes the sparse factorization on CPU; it agrees with
    the dense inverse of the assembled matrix, norm-wise (largest entry difference over
    the largest entry). Measured: 9e-14 at 1 to 16 threads; tolerance 1e-11. Entry-wise the small transfer
    impedances between distant buses differ by up to 1.2e-9 relative (both solves round
    differently on an ill-conditioned Y)."""
    grid = apply_default_harmonic_model(synthetic_feeder(200))
    f = [50.0, 650.0]
    nodes = [0, 57, 199]
    scan = impedance_scan(
        grid, nodes=nodes, frequencies_hz=f, transfer=True, load_shunt="const_z"
    )
    y = assemble_ybus(grid, f, dtype=torch.complex128)
    z_inv = _inverse_per_frequency(y.Y)
    rows = [r for n in nodes for r in y.index.rows(n)]
    ref = z_inv[:, rows][:, :, rows]
    err = (scan.z_transfer_ohm - ref).abs().amax(dim=(1, 2)) / ref.abs().amax(
        dim=(1, 2)
    )
    assert err.max() < 1e-11


def test_complex64_agrees_with_complex128():
    """Measured worst relative difference 2e-6 on this grid (tolerance 1e-5)."""
    grid = three_phase_two_bus()
    z128 = impedance_scan(grid).z_positive_ohm
    s64 = impedance_scan(grid, dtype=torch.complex64)
    assert s64.z_positive_ohm.dtype == torch.complex64
    assert s64.frequencies_hz.dtype == torch.float32
    rel = (s64.z_positive_ohm.to(torch.complex128) - z128).abs() / z128.abs()
    assert rel.max() < 1e-5
    by_name = impedance_scan(grid, dtype="complex64")
    torch.testing.assert_close(
        by_name.z_positive_ohm, s64.z_positive_ohm, rtol=0.0, atol=0.0
    )


def test_default_frequency_grid():
    scan = impedance_scan(_parallel_rlc_grid())
    f = scan.frequencies_hz
    n = int(round((DEFAULT_MAX_ORDER - 1) / DEFAULT_ORDER_STEP)) + 1
    assert f.shape == (n,) == (491,)
    assert f[0].item() == F0 and f[-1].item() == pytest.approx(DEFAULT_MAX_ORDER * F0)
    np.testing.assert_allclose(np.diff(f.numpy()), DEFAULT_ORDER_STEP * F0, rtol=1e-9)
    torch.testing.assert_close(scan.orders, f / F0)


# --------------------------------------------------------------------------- contracts
def test_to_dict_is_json_ready_and_round_trips():
    grid = three_phase_two_bus()
    grid.nodes.append(_node(5))
    grid.branches.append(
        Line(
            id=60,
            from_node=2,
            to_node=5,
            from_phases=(Phase.A,),
            to_phases=(Phase.A,),
            length_m=50.0,
            series_resistance_ohm_per_m=[[1e-3]],
            series_inductance_h_per_m=[[1e-6]],
            shunt_capacitance_f_per_m=[[0.0]],
            harmonic_line_model="naive",
        )
    )
    scan = impedance_scan(grid, frequencies_hz=[50.0, 300.0, 900.0], transfer=True)
    d = scan.to_dict()
    text = json.dumps(d, allow_nan=False)
    assert d["study"] == "impedance_scan"
    back = ImpedanceScan.from_dict(json.loads(text))
    assert back.node_ids == scan.node_ids == (1, 2, 5)
    assert back.phases == scan.phases
    assert d["nodes"][2]["z_positive_re_ohm"] is None
    assert d["nodes"][2]["phases"] == ["a"]
    assert d["transfer"]["rows"][-1] == {"node_id": 5, "phase": "a"}
    torch.testing.assert_close(back.frequencies_hz, scan.frequencies_hz, rtol=0, atol=0)
    for k in range(3):
        torch.testing.assert_close(
            back.z_phase_ohm[k], scan.z_phase_ohm[k], rtol=0, atol=0
        )
    torch.testing.assert_close(
        back.z_positive_ohm[:2], scan.z_positive_ohm[:2], rtol=0, atol=0
    )
    assert (
        torch.isnan(back.z_positive_ohm[2]).all()
        and torch.isnan(scan.z_zero_ohm[2]).all()
    )
    torch.testing.assert_close(back.z_transfer_ohm, scan.z_transfer_ohm, rtol=0, atol=0)
    assert back.to_dict() == d
    without = impedance_scan(grid, frequencies_hz=[50.0]).to_dict()
    assert without["transfer"] is None
    d["study"] = "hosting_capacity"
    with pytest.raises(
        InputError, match="expected a 'impedance_scan'.*found.*'hosting_capacity'"
    ):
        ImpedanceScan.from_dict(d)


def test_impedance_component_selection_and_errors():
    grid = three_phase_two_bus()
    grid.nodes.append(_node(5))
    grid.branches.append(
        Line(
            id=60,
            from_node=2,
            to_node=5,
            from_phases=(Phase.A,),
            to_phases=(Phase.A,),
            length_m=50.0,
            series_resistance_ohm_per_m=[[1e-3]],
            series_inductance_h_per_m=[[1e-6]],
            shunt_capacitance_f_per_m=[[0.0]],
            harmonic_line_model="naive",
        )
    )
    scan = impedance_scan(grid, frequencies_hz=[50.0, 300.0])
    torch.testing.assert_close(scan.impedance(5), scan.driving_point(5)[:, 0, 0])
    torch.testing.assert_close(scan.impedance(2, "b"), scan.driving_point(2)[:, 1, 1])
    with pytest.raises(InputError, match="sequence"):
        scan.impedance(5, "positive")
    with pytest.raises(InputError, match="no phase"):
        scan.impedance(5, "c")
    with pytest.raises(InputError, match="component"):
        scan.impedance(2, "negative")
    with pytest.raises(InputError, match="not part"):
        scan.driving_point(99)
    assert scan.resonances(5) == []


def test_invalid_inputs_raise():
    grid = three_phase_two_bus()
    with pytest.raises(InputError, match="unknown"):
        impedance_scan(grid, nodes=[1, 7])
    with pytest.raises(InputError, match="repeat"):
        impedance_scan(grid, nodes=[1, 1])
    with pytest.raises(InputError, match="increasing"):
        impedance_scan(grid, frequencies_hz=[100.0, 50.0])
    with pytest.raises(InputError, match="positive"):
        impedance_scan(grid, frequencies_hz=[0.0, 50.0])
    with pytest.raises(InputError, match="dtype"):
        impedance_scan(grid, dtype=torch.float64)
    with pytest.raises(InputError, match="dtype"):
        impedance_scan(grid, dtype="float32")
    ideal = copy.deepcopy(grid)
    ideal.appliances[0] = ideal.appliances[0].model_copy(
        update={"resistance_ohm": _sym(0.0, 0.0), "inductance_h": _sym(0.0, 0.0)}
    )
    with pytest.raises(InputError, match="zero impedance"):
        impedance_scan(ideal)
    no_source = copy.deepcopy(grid)
    no_source.appliances = [
        a for a in no_source.appliances if not isinstance(a, Source)
    ]
    with pytest.raises(InputError, match="Source"):
        impedance_scan(no_source)


# --------------------------------------------------------------------------- resonances
def _curve_scan(f: np.ndarray, z: np.ndarray) -> ImpedanceScan:
    """A one-node, one-phase scan holding the curve ``z(f)``."""
    zt = torch.as_tensor(z, dtype=torch.complex128)
    nan = torch.full((1, f.size), complex(math.nan, math.nan), dtype=torch.complex128)
    return ImpedanceScan(
        frequencies_hz=torch.as_tensor(f, dtype=torch.float64),
        base_frequency_hz=F0,
        node_ids=(1,),
        phases=((Phase.A,),),
        z_phase_ohm=(zt.reshape(-1, 1, 1),),
        z_positive_ohm=nan,
        z_zero_ohm=nan,
    )


def test_flat_noisy_curve_has_no_resonance():
    """Rounding noise on a flat curve (1e-15 relative, as complex128 leaves it, and
    1e-7, as complex64 does) is not a resonance; ``min_prominence=0`` reports every
    strict extremum of the noise."""
    rng = np.random.default_rng(0)
    f = np.linspace(50.0, 2500.0, 100)
    for noise in (1e-15, 1e-7):
        z = 3.0 * (1.0 + noise * rng.standard_normal(f.size)) + 0j
        scan = _curve_scan(f, z)
        assert scan.resonances(1) == []
        assert len(scan.resonances(1, min_prominence=0.0)) > 10


def test_noise_on_the_flank_neither_hides_nor_splits_a_resonance():
    """Prominence is measured against the surrounding curve, so a resonance with noisy
    flanks is still one resonance at the right place."""
    rng = np.random.default_rng(1)
    f = np.linspace(400.0, 600.0, 201)
    z = _z_parallel(f) * (1.0 + 1e-7 * rng.standard_normal(f.size))
    res = _curve_scan(f, z).resonances(1)
    assert [r.kind for r in res] == ["parallel"]
    assert abs(res[0].frequency_hz - 503.29) < 1.0
    assert 0.9 < res[0].prominence <= 1.0


def test_plateau_is_one_extremum_at_its_middle():
    f = np.arange(10, dtype=float) * 10.0 + 50.0
    mag = np.array([1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 3.0, 2.0, 1.0, 0.5])
    res = _curve_scan(f, mag + 0j).resonances(1)
    assert [r.kind for r in res] == ["parallel"]
    assert res[0].frequency_hz == f[4]
    assert res[0].z_ohm == 5.0
    assert res[0].prominence == pytest.approx(0.8)  # base: the higher side minimum, 1


def test_prominence_threshold_and_values():
    """The peak (10) stands over its base, the higher of the two side minima (8 on the
    left, 1 on the right): 0.2. The dip (1) stands under its base, the lower of the two
    side maxima (10 on the left, 2.5 at the right end): 0.6."""
    f = np.arange(7, dtype=float) * 10.0 + 50.0
    mag = np.array([8.0, 9.0, 10.0, 9.0, 1.0, 2.0, 2.5])
    res = _curve_scan(f, mag + 0j).resonances(1)
    assert [r.kind for r in res] == ["parallel", "series"]
    assert res[0].prominence == pytest.approx(0.2)
    assert res[1].prominence == pytest.approx(0.6)
    assert [
        r.kind for r in _curve_scan(f, mag + 0j).resonances(1, min_prominence=0.3)
    ] == ["series"]
    with pytest.raises(InputError, match="min_prominence"):
        _curve_scan(f, mag + 0j).resonances(1, min_prominence=1.5)


def test_lossless_series_dip_is_reported_without_a_warning():
    """``|Z| = 0`` at a sample: prominence 1, no divide-by-zero warning."""
    f = np.arange(5, dtype=float) * 10.0 + 50.0
    mag = np.array([4.0, 2.0, 0.0, 2.0, 4.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = _curve_scan(f, mag + 0j).resonances(1)
    assert [r.kind for r in res] == ["series"]
    assert res[0].prominence == 1.0
    assert res[0].z_ohm == pytest.approx(0.0, abs=1e-12)
    assert res[0].frequency_hz == pytest.approx(70.0)


def test_complex64_scan_finds_the_same_resonances():
    """The tuned branch scanned in complex64: the same two resonances as complex128,
    no spurious ones from single-precision noise. Measured: frequencies agree to
    0.01 Hz (tolerance 0.1 Hz)."""
    grid = _series_branch_grid()
    r128 = impedance_scan(grid).resonances(1)
    r64 = impedance_scan(grid, dtype=torch.complex64).resonances(1)
    assert [r.kind for r in r64] == [r.kind for r in r128] == ["parallel", "series"]
    for a, b in zip(r64, r128):
        assert abs(a.frequency_hz - b.frequency_hz) < 0.1


def test_nearly_singular_admittance_is_reported(caplog):
    """Two buses joined by a line and held to ground only by a 1e-20 S shunt, with no
    path to the source: ``Y`` is singular to working precision, the solve stays finite,
    and the scan names the frequencies in a WARNING."""
    line = Line(
        id=10,
        from_node=2,
        to_node=3,
        from_phases=(Phase.A,),
        to_phases=(Phase.A,),
        length_m=1.0,
        series_resistance_ohm_per_m=[[R_L]],
        series_inductance_h_per_m=[[L_L]],
        shunt_capacitance_f_per_m=[[0.0]],
        harmonic_line_model="naive",
    )
    grid = Grid(
        base_frequency_hz=F0,
        nodes=[_node(1), _node(2), _node(3)],
        branches=[line],
        appliances=[
            _source(1, [[R_S]], [[L_S]]),
            ShuntAppliance(
                id=200,
                node=3,
                phases=(Phase.A,),
                conductance_s=(1.0e-20,),
                capacitance_f=(0.0,),
            ),
        ],
    )
    with caplog.at_level(logging.WARNING, logger="pgml"):
        scan = impedance_scan(grid, frequencies_hz=[50.0, 250.0])
    assert torch.isfinite(torch.view_as_real(scan.z_phase_ohm[1])).all()
    assert "numerically singular" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pgml"):
        impedance_scan(_series_branch_grid(), frequencies_hz=[50.0, 250.0])
    assert "numerically singular" not in caplog.text
