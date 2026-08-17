"""Independent plain-numpy Y-bus + solve oracle for the tiny fixture grids.

This deliberately recomputes the nodal admittance and node voltages WITHOUT using
any pgml.assembly / pgml.solver code, so the forward-correctness tests compare two
independent implementations. It only supports the small fixtures' element set.
"""

from __future__ import annotations

import numpy as np

from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    Line,
    Load,
    ShuntAppliance,
    Source,
)


def _phase_index(grid: Grid):
    """Compact (node_id, phase) -> row map, matching assembly.node_phase_index."""
    row_of = {}
    r = 0
    for node in grid.nodes:
        for ph in node.phases:
            row_of[(node.id, ph)] = r
            r += 1
    return row_of, r


def _series_admittance(r_mat, l_mat, f) -> np.ndarray:
    z = np.asarray(r_mat, float) + 1j * (2 * np.pi * f) * np.asarray(l_mat, float)
    return np.linalg.inv(z)


def _u_ln(node) -> float:
    if len(node.phases) >= 3:
        return node.u_rated_v / np.sqrt(3.0)
    return node.u_rated_v


def build_y_and_i(grid: Grid, f: float):
    """Return (Y, I, row_of, N) for a single absolute frequency ``f`` (numpy)."""
    row_of, n = _phase_index(grid)
    y = np.zeros((n, n), dtype=np.complex128)
    i = np.zeros(n, dtype=np.complex128)
    nodes_by_id = {nd.id: nd for nd in grid.nodes}

    for b in grid.branches:
        if isinstance(b, Line) and b.in_service:
            length = b.length_m
            r = np.asarray(b.series_resistance_ohm_per_m, float) * length
            ind = np.asarray(b.series_inductance_h_per_m, float) * length
            c = np.asarray(b.shunt_capacitance_f_per_m, float) * length
            g = (
                np.asarray(b.shunt_conductance_s_per_m, float) * length
                if b.shunt_conductance_s_per_m is not None
                else np.zeros_like(r)
            )
            ys = _series_admittance(r, ind, f)
            y_sh = g + 1j * (2 * np.pi * f) * c
            fr = [row_of[(b.from_node, ph)] for ph in b.from_phases]
            to = [row_of[(b.to_node, ph)] for ph in b.to_phases]
            idx = fr + to
            p = len(fr)
            prim = np.block([[ys, -ys], [-ys, ys]])
            prim[:p, :p] += 0.5 * y_sh
            prim[p:, p:] += 0.5 * y_sh
            for a, ga in enumerate(idx):
                for bb, gb in enumerate(idx):
                    y[ga, gb] += prim[a, bb]

    for a in grid.appliances:
        if isinstance(a, Source) and a.in_service:
            r = np.asarray(a.resistance_ohm, float)
            ind = np.asarray(a.inductance_h, float)
            ys = _series_admittance(r, ind, f)
            rows = [row_of[(a.node, ph)] for ph in a.phases]
            for ai, ga in enumerate(rows):
                for bi, gb in enumerate(rows):
                    y[ga, gb] += ys[ai, bi]
            u_ref = np.asarray(a.u_ref_v, float)
            ang = np.deg2rad(np.asarray(a.u_angle_deg, float))
            vth = u_ref * np.exp(1j * ang)
            i_s = ys @ vth
            for ai, ga in enumerate(rows):
                i[ga] += i_s[ai]
        elif isinstance(a, ShuntAppliance) and a.in_service:
            g = np.asarray(a.conductance_s, float)
            c = np.asarray(a.capacitance_f, float)
            ysh = g + 1j * (2 * np.pi * f) * c
            rows = [row_of[(a.node, ph)] for ph in a.phases]
            for ai, ga in enumerate(rows):
                y[ga, ga] += ysh[ai]
        elif isinstance(a, (Load, Generator)) and a.in_service:
            node = nodes_by_id[a.node]
            u_ln = _u_ln(node)
            sign = 1.0 if isinstance(a, Load) else -1.0
            nph = len(a.phases)
            if a.p_nom_per_phase_w is not None:
                p_pp = np.asarray(a.p_nom_per_phase_w, float)
            else:
                p_pp = np.full(nph, a.p_nom_w / nph)
            if a.q_nom_per_phase_var is not None:
                q_pp = np.asarray(a.q_nom_per_phase_var, float)
            else:
                q_pp = np.full(nph, a.q_nom_var / nph)
            ypp = np.conj(sign * p_pp + 1j * sign * q_pp) / (u_ln**2)
            rows = [row_of[(a.node, ph)] for ph in a.phases]
            for ai, ga in enumerate(rows):
                y[ga, ga] += ypp[ai]

    return y, i, row_of, n


def solve_norton(grid: Grid, f: float):
    """numpy Norton solve V = Y^-1 I."""
    y, i, row_of, n = build_y_and_i(grid, f)
    v = np.linalg.solve(y, i)
    return v, y, i, row_of


__all__ = ["build_y_and_i", "solve_norton"]
