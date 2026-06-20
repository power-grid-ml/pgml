"""Equation registry: forward values, evaluate/solve_for, latex, and gradcheck."""

from __future__ import annotations

import math

import torch

from pgml.equations import registry


def test_reactance_and_susceptance_values():
    f, ind, c = 50.0, 1.0e-3, 1.0e-9
    x = registry.torch_fn("reactance_from_inductance")(
        X=torch.tensor(0.0, dtype=torch.float64),
        f=torch.tensor(f, dtype=torch.float64),
        L=torch.tensor(ind, dtype=torch.float64),
    )
    # residual = X - 2*pi*f*L; with X=0 residual = -2*pi*f*L
    assert math.isclose(float(x), -2 * math.pi * f * ind, rel_tol=1e-12)

    # solve_for X should give the reactance directly.
    x_fn = registry.solve_for("reactance_from_inductance", "X")
    xval = x_fn(
        f=torch.tensor(f, dtype=torch.float64), L=torch.tensor(ind, dtype=torch.float64)
    )
    assert math.isclose(float(xval), 2 * math.pi * f * ind, rel_tol=1e-12)

    b_fn = registry.solve_for("susceptance_from_capacitance", "B")
    bval = b_fn(
        f=torch.tensor(f, dtype=torch.float64), C=torch.tensor(c, dtype=torch.float64)
    )
    assert math.isclose(float(bval), 2 * math.pi * f * c, rel_tol=1e-12)


def test_series_and_shunt_admittance_solve_for():
    y_fn = registry.solve_for("series_admittance_scalar", "y")
    r, x = 1.0, 2.0
    yv = y_fn(
        R=torch.tensor(r, dtype=torch.complex128),
        X=torch.tensor(x, dtype=torch.complex128),
    )
    expected = 1.0 / (r + 1j * x)
    assert abs(complex(yv) - expected) < 1e-12

    ysh_fn = registry.solve_for("shunt_admittance_scalar", "y")
    g, b = 0.5, 1.5
    yv = ysh_fn(
        G=torch.tensor(g, dtype=torch.complex128),
        B=torch.tensor(b, dtype=torch.complex128),
    )
    assert abs(complex(yv) - (g + 1j * b)) < 1e-12


def test_seq_to_phase():
    self_fn = registry.solve_for("seq_to_phase_self", "Z_self")
    mut_fn = registry.solve_for("seq_to_phase_mutual", "Z_mutual")
    z0, z1 = 3.0 + 1j, 1.0 + 0.5j
    s = self_fn(
        Z0=torch.tensor(z0, dtype=torch.complex128),
        Z1=torch.tensor(z1, dtype=torch.complex128),
    )
    m = mut_fn(
        Z0=torch.tensor(z0, dtype=torch.complex128),
        Z1=torch.tensor(z1, dtype=torch.complex128),
    )
    assert abs(complex(s) - (z0 + 2 * z1) / 3) < 1e-12
    assert abs(complex(m) - (z0 - z1) / 3) < 1e-12


def test_evaluate_broadcasts_and_normalizes():
    f = torch.tensor([50.0, 100.0], dtype=torch.float64)
    ind = torch.tensor(1.0e-3, dtype=torch.float64)
    x = torch.tensor([0.3, 0.6], dtype=torch.float64)
    res = registry.evaluate("reactance_from_inductance", {"X": x, "f": f, "L": ind})
    assert res.shape == (2,)
    # Residual is the closed form X - 2*pi*f*L, evaluated per broadcast element.
    expected = x - 2 * math.pi * f * ind
    torch.testing.assert_close(res, expected, rtol=0, atol=1e-12)

    rel = registry.evaluate(
        "reactance_from_inductance", {"X": x, "f": f, "L": ind}, normalize_by="X"
    )
    assert rel.shape == (2,)
    # Normalization divides the residual by the named field (here X).
    torch.testing.assert_close(rel, expected / x, rtol=0, atol=1e-12)


def test_latex_is_string():
    s = registry.latex("series_admittance_scalar")
    assert isinstance(s, str) and "y" in s


def test_gradcheck_reactance():
    def fn(L):
        return registry.solve_for("reactance_from_inductance", "X")(
            f=torch.tensor(50.0, dtype=torch.float64), L=L
        )

    L = torch.tensor(1.0e-3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (L,), eps=1e-9, atol=1e-7)


def test_gradcheck_series_admittance():
    def fn(R, X):
        return registry.solve_for("series_admittance_scalar", "y")(R=R, X=X)

    R = torch.tensor(1.0, dtype=torch.complex128, requires_grad=True)
    X = torch.tensor(2.0, dtype=torch.complex128, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (R, X), eps=1e-6, atol=1e-6)
