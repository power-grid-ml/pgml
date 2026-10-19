"""Differentiability of the impedance scan (float64 gradcheck).

The scanned impedances are differentiable with respect to tensor-valued grid parameters
through ``assemble_ybus`` and the linear-solve adjoint of the factored column solve. Checked
on the phase-domain driving-point matrix and both sequence impedances, w.r.t. line R/L/C,
source R/L and the operating-point power of the load, under the default harmonic device
shunt and the const-Z fold, plus a finite-difference spot check of one parameter.
"""

from __future__ import annotations

import pytest
import torch

from pgml.schemas.grid_schema import Grid, Line, Load, Node, Phase, Source
from pgml.studies import impedance_scan

RDT = torch.float64
ABC = (Phase.A, Phase.B, Phase.C)
FREQUENCIES_HZ = [50.0, 350.0, 1150.0]


def _sym(diag: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    """Symmetric 3x3 matrix ``diag`` on the diagonal, ``ratio * diag`` off it."""
    eye = torch.eye(3, dtype=RDT)
    return diag * (eye + ratio * (1.0 - eye))


def _grid(r_line, l_line, c_line, r_src, l_src, p_load) -> Grid:
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=400.0, phases=ABC),
            Node(id=2, u_rated_v=400.0, phases=ABC),
        ],
        branches=[
            Line(
                id=10,
                from_node=1,
                to_node=2,
                from_phases=ABC,
                to_phases=ABC,
                length_m=200.0,
                series_resistance_ohm_per_m=_sym(r_line),
                series_inductance_h_per_m=_sym(l_line),
                shunt_capacitance_f_per_m=_sym(c_line, 0.2),
                harmonic_line_model="naive",
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ABC,
                u_ref_v=(230.94,) * 3,
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=_sym(r_src, 0.3),
                inductance_h=_sym(l_src, 0.3),
            ),
            Load(id=2, node=2, phases=ABC, p_nom_w=p_load, q_nom_var=3.0e3),
        ],
    )


#: line R [Ω/m], L [H/m], C [F/m], source R [Ω], L [H], load P [W]
VALUES = (2.0e-4, 8.0e-7, 2.0e-7, 0.05, 4.0e-4, 1.2e4)


def _leaves():
    return tuple(torch.tensor(v, dtype=RDT, requires_grad=True) for v in VALUES)


def _outputs(*leaves, load_shunt=None) -> torch.Tensor:
    scan = impedance_scan(
        _grid(*leaves), frequencies_hz=FREQUENCIES_HZ, load_shunt=load_shunt
    )
    return torch.cat(
        [
            torch.view_as_real(scan.driving_point(2)).reshape(-1),
            torch.view_as_real(scan.z_positive_ohm).reshape(-1),
            torch.view_as_real(scan.z_zero_ohm).reshape(-1),
        ]
    )


@pytest.mark.parametrize("load_shunt", [None, "const_z"])
def test_gradcheck_impedance_scan(load_shunt):
    """Every output w.r.t. line R/L/C, source R/L and load P (relative step scaling)."""

    def fn(*unit):
        return _outputs(*(u * s for u, s in zip(unit, VALUES)), load_shunt=load_shunt)

    unit = tuple(torch.ones((), dtype=RDT, requires_grad=True) for _ in VALUES)
    assert torch.autograd.gradcheck(fn, unit, eps=1e-6, atol=1e-8, rtol=1e-5)


def test_finite_difference_spot_check_source_inductance():
    """d|Z1(f)|/dL_src at the source bus by central differences (relative error < 1e-6)."""
    leaves = _leaves()
    scan = impedance_scan(_grid(*leaves), nodes=[1], frequencies_hz=FREQUENCIES_HZ)
    mag = scan.z_positive_ohm.abs().sum()
    (grad,) = torch.autograd.grad(mag, leaves[4])

    vals = list(VALUES)
    h = 1e-6 * vals[4]

    def mag_at(l_src: float) -> float:
        args = [torch.tensor(v, dtype=RDT) for v in vals]
        args[4] = torch.tensor(l_src, dtype=RDT)
        s = impedance_scan(_grid(*args), nodes=[1], frequencies_hz=FREQUENCIES_HZ)
        return float(s.z_positive_ohm.abs().sum())

    fd = (mag_at(vals[4] + h) - mag_at(vals[4] - h)) / (2.0 * h)
    assert abs(float(grad) - fd) <= 1e-6 * abs(fd)
