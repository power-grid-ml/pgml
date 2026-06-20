"""Increment 1: calculation-symmetry threading through the power flow.

A grid with genuine per-phase load data:
- ``symmetry="symmetric"`` IGNORES the per-phase split and distributes the total
  equally over the phases (power-grid-model rule);
- ``symmetry="asymmetric"`` honors the per-phase data;
- the two solves differ; the symmetric solve equals an equal-split baseline grid.
"""

from __future__ import annotations

import torch

from pgml.schemas.grid_schema import Grid, Line, Load, Node, Phase, Source
from pgml.solver import solve_power_flow

ABC = (Phase.A, Phase.B, Phase.C)
CDT = torch.complex128


def _grid(per_phase_p, per_phase_q):
    nodes = [
        Node(id=1, u_rated_v=400.0, phases=ABC),
        Node(id=2, u_rated_v=400.0, phases=ABC),
    ]
    source = Source(
        id=10,
        node=1,
        phases=ABC,
        u_ref_v=(231.0, 231.0, 231.0),
        u_angle_deg=(0.0, -120.0, 120.0),
        resistance_ohm=[[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
        inductance_h=[[1e-5, 0, 0], [0, 1e-5, 0], [0, 0, 1e-5]],
    )
    line = Line(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=ABC,
        to_phases=ABC,
        length_m=100.0,
        series_resistance_ohm_per_m=[[1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1e-3]],
        series_inductance_h_per_m=[[1e-6, 0, 0], [0, 1e-6, 0], [0, 0, 1e-6]],
        shunt_capacitance_f_per_m=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    )
    total_p = sum(per_phase_p)
    total_q = sum(per_phase_q)
    load = Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=total_p,
        q_nom_var=total_q,
        p_nom_per_phase_w=per_phase_p,
        q_nom_per_phase_var=per_phase_q,
    )
    return Grid(nodes=nodes, branches=[line], appliances=[source, load])


def _balanced_grid(total_p, total_q):
    """Same grid but with the totals only (no per-phase split = equal split)."""
    nodes = [
        Node(id=1, u_rated_v=400.0, phases=ABC),
        Node(id=2, u_rated_v=400.0, phases=ABC),
    ]
    source = Source(
        id=10,
        node=1,
        phases=ABC,
        u_ref_v=(231.0, 231.0, 231.0),
        u_angle_deg=(0.0, -120.0, 120.0),
        resistance_ohm=[[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
        inductance_h=[[1e-5, 0, 0], [0, 1e-5, 0], [0, 0, 1e-5]],
    )
    line = Line(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=ABC,
        to_phases=ABC,
        length_m=100.0,
        series_resistance_ohm_per_m=[[1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1e-3]],
        series_inductance_h_per_m=[[1e-6, 0, 0], [0, 1e-6, 0], [0, 0, 1e-6]],
        shunt_capacitance_f_per_m=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    )
    load = Load(id=30, node=2, phases=ABC, p_nom_w=total_p, q_nom_var=total_q)
    return Grid(nodes=nodes, branches=[line], appliances=[source, load])


def test_symmetric_ignores_per_phase_and_equals_equal_split():
    p = (3000.0, 1000.0, 500.0)
    q = (600.0, 200.0, 100.0)
    grid = _grid(p, q)

    res_asym = solve_power_flow(grid, symmetry="asymmetric", dtype=CDT)
    res_sym = solve_power_flow(grid, symmetry="symmetric", dtype=CDT)
    assert res_asym.converged and res_sym.converged

    # The two solves must DIFFER (the per-phase imbalance is real).
    assert not torch.allclose(res_asym.v, res_sym.v, atol=1e-6)

    # The symmetric solve equals the equal-split baseline grid (total/3 per phase).
    base = solve_power_flow(
        _balanced_grid(sum(p), sum(q)), symmetry="asymmetric", dtype=CDT
    )
    assert torch.allclose(res_sym.v, base.v, atol=1e-12)


def test_auto_promotes_to_asymmetric_when_per_phase_present():
    p = (3000.0, 1000.0, 500.0)
    q = (600.0, 200.0, 100.0)
    grid = _grid(p, q)
    # auto sees per-phase data -> asymmetric; must equal the forced asymmetric solve.
    res_auto = solve_power_flow(grid, symmetry="auto", dtype=CDT)
    res_asym = solve_power_flow(grid, symmetry="asymmetric", dtype=CDT)
    assert torch.allclose(res_auto.v, res_asym.v, atol=1e-12)
