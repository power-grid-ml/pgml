"""Gradients through the harmonic device shunt (float64 gradcheck).

The shunt ``Y_eq = conj(S)/V_rated**2`` puts the device's operating point into ``Y(h)``,
so a harmonic voltage now depends on P and Q through TWO paths: the current injection
(its fundamental current) and the matrix itself. For a voltage-dependent device (ZIP or
an inverter control law) the realised power also depends on the converged fundamental
voltage, so the shunt sits behind the implicit-function gradient of the fundamental
solve. All of it has to stay on the tape.

Checked per shunt model (``opendss``, ``motor``): gradcheck w.r.t. P and Q of a load,
w.r.t. a line resistance with the shunt present, the same for a ZIP load (the
``v1``-dependent path), a finite-difference spot check, and that the gradient really
differs from the no-shunt model (the shunt is not a constant w.r.t. P).
"""

from __future__ import annotations

import torch

from pgml.schemas.grid_schema import (
    Grid,
    HarmonicComponent,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
    ZipCoefficients,
)
from pgml.solver import solve_harmonic_flow
from pgml.solver.harmonic_flow import assemble_harmonic_system

CDT = torch.complex128
A = (Phase.A,)

_SPECTRUM = StaticSpectrum(
    spectrum=SpectrumPoint(
        components=[
            HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
            HarmonicComponent(order=5, magnitude_pu=0.2, phase_deg=0.0),
            HarmonicComponent(order=7, magnitude_pu=0.1, phase_deg=0.0),
        ]
    )
)


def _two_bus(*, zip_load: bool = False) -> Grid:
    """Single-phase two-bus grid; every varied parameter comes from an override."""
    load_kwargs = {}
    if zip_load:
        load_kwargs = dict(
            load_model=LoadModel.ZIP,
            zip_coefficients=ZipCoefficients(
                z_p=0.4, i_p=0.3, p_p=0.3, z_q=0.5, i_q=0.2, p_q=0.3
            ),
        )
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=230.0, phases=A),
            Node(id=2, u_rated_v=230.0, phases=A),
        ],
        branches=[
            Line(
                id=20,
                from_node=1,
                to_node=2,
                from_phases=A,
                to_phases=A,
                length_m=100.0,
                series_resistance_ohm_per_m=[[1.0e-3]],
                series_inductance_h_per_m=[[1.0e-6]],
                shunt_capacitance_f_per_m=[[1.0e-9]],
            )
        ],
        appliances=[
            Source(
                id=10,
                node=1,
                phases=A,
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[1.0e-3]],
            ),
            Load(
                id=30,
                node=2,
                phases=A,
                p_nom_w=2000.0,
                q_nom_var=500.0,
                spectrum=_SPECTRUM,
                **load_kwargs,
            ),
        ],
    )


def _solve(p=None, q=None, r=None, *, shunt="opendss", zip_load=False):
    overrides = {}
    if p is not None:
        overrides[("load", 30, "p_nom_per_phase_w")] = p
    if q is not None:
        overrides[("load", 30, "q_nom_per_phase_var")] = q
    if r is not None:
        overrides[("line", 20, "series_resistance_ohm_per_m")] = r
    return solve_harmonic_flow(
        _two_bus(zip_load=zip_load),
        [1, 5, 7],
        dtype=CDT,
        load_shunt=shunt,
        param_overrides=overrides,
    ).v.reshape(-1)


def test_gradcheck_active_power_through_the_shunt():
    """``dV(h)/dP`` with the shunt in ``Y(h)``: injection AND matrix depend on P."""
    p = torch.tensor([2000.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda p: _solve(p=p), (p,), eps=1e-2, atol=1e-4, rtol=1e-3
    )


def test_gradcheck_reactive_power_through_the_shunt():
    """``dV(h)/dQ``: the shunt susceptance is ``-Q/V_rated**2``, scaled per order."""
    q = torch.tensor([500.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda q: _solve(q=q), (q,), eps=1e-2, atol=1e-4, rtol=1e-3
    )


def test_gradcheck_motor_model():
    """The motor branch carries gradients too (its reactance scales with the kVA)."""
    p = torch.tensor([2000.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda p: _solve(p=p, shunt="motor"), (p,), eps=1e-2, atol=1e-4, rtol=1e-3
    )


def test_gradcheck_line_resistance_with_the_shunt_present():
    """The network parameters stay differentiable with the extra stamp in place."""
    r = torch.tensor([[1.0e-3]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda r: _solve(r=r), (r,), eps=1e-6, atol=1e-5, rtol=1e-3
    )


def test_gradcheck_zip_load_shunt_depends_on_the_fundamental_solution():
    """A ZIP device's shunt reads the CONVERGED fundamental voltage.

    ``S_eff = S0*(z*r^2 + i*r + p)`` at ``r = |V_term|/V0``, so the gradient runs through
    the implicit-function adjoint of the nonlinear fundamental solve and back into
    ``Y(h)``.
    """
    p = torch.tensor([2000.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda p: _solve(p=p, zip_load=True),
        (p,),
        eps=1e-2,
        atol=1e-4,
        rtol=1e-3,
    )


def test_finite_difference_spot_check():
    """An independent central difference backs up the autograd gradient of ``|V(7)|``."""
    p0 = 2000.0

    def mag(p_value):
        p = torch.tensor([p_value], dtype=torch.float64)
        return _solve(p=p).abs()[-1]

    p = torch.tensor([p0], dtype=torch.float64, requires_grad=True)
    analytic = torch.autograd.grad(_solve(p=p).abs()[-1], p)[0].item()
    dp = 1.0
    numeric = (mag(p0 + dp) - mag(p0 - dp)).item() / (2.0 * dp)
    assert abs(analytic - numeric) <= 1e-6 * max(abs(numeric), 1e-12) + 1e-12


def test_the_shunt_changes_the_gradient():
    """With and without the shunt the SENSITIVITY differs, not just the value."""
    grads = {}
    for shunt in ("none", "opendss"):
        p = torch.tensor([2000.0], dtype=torch.float64, requires_grad=True)
        _solve(p=p, shunt=shunt).abs().sum().backward()
        grads[shunt] = p.grad.item()
    assert abs(grads["none"] - grads["opendss"]) > 1e-9 * abs(grads["none"])


def test_assembled_system_carries_the_shunt_gradient():
    """``assemble_harmonic_system`` (the residual hook) keeps P in ``Y(h)``'s graph."""
    grid = _two_bus()
    v1 = solve_harmonic_flow(grid, [1], dtype=CDT).v[..., 0, :]
    p = torch.tensor([2000.0], dtype=torch.float64, requires_grad=True)
    y, _, _ = assemble_harmonic_system(
        grid,
        [5, 7],
        v1,
        dtype=CDT,
        load_shunt="opendss",
        param_overrides={("load", 30, "p_nom_per_phase_w"): p},
    )
    y.abs().sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert float(p.grad.abs().sum()) > 0.0


def _solve_batched(p, *, budget_bytes=None, basis=None):
    """A 3-scenario batched harmonic study, differentiable in the batched load power."""
    import pgml.solver.harmonic_flow as hf

    grid = _two_bus()
    op = {30: {"p_w": p, "q_var": torch.full_like(p, 500.0)}}
    kw = {} if basis is None else {"load_shunt_basis": basis}
    if budget_bytes is not None:
        orig = hf._harmonic_system_budget_bytes
        hf._harmonic_system_budget_bytes = lambda: budget_bytes
        try:
            return solve_harmonic_flow(
                grid, [1, 5, 7], dtype=CDT, operating_point=op, **kw
            ).v.reshape(-1)
        finally:
            hf._harmonic_system_budget_bytes = orig
    return solve_harmonic_flow(
        grid, [1, 5, 7], dtype=CDT, operating_point=op, **kw
    ).v.reshape(-1)


def _solve_sparse_shunt_batch(p):
    """Five rows with one shunt row select the automatic low-rank path."""
    grid = _two_bus()
    nodes = list(grid.nodes) + [
        Node(id=i, u_rated_v=230.0, phases=A) for i in range(3, 6)
    ]
    branches = list(grid.branches) + [
        Line(
            id=20 + i,
            from_node=i - 1,
            to_node=i,
            from_phases=A,
            to_phases=A,
            length_m=100.0,
            series_resistance_ohm_per_m=[[1.0e-3]],
            series_inductance_h_per_m=[[1.0e-6]],
            shunt_capacitance_f_per_m=[[1.0e-9]],
        )
        for i in range(3, 6)
    ]
    sparse_grid = grid.model_copy(update={"nodes": nodes, "branches": branches})
    op = {30: {"p_w": p, "q_var": torch.full_like(p, 500.0)}}
    return solve_harmonic_flow(
        sparse_grid, [1, 5, 7], dtype=CDT, operating_point=op
    ).v.reshape(-1)


def test_gradcheck_through_the_chunked_scenario_batch():
    """A batch whose per-scenario ``Y(h)`` is assembled in chunks stays differentiable.

    The chunked path concatenates per-chunk solves, so the gradient has to flow through
    the concatenation into every chunk's own assembly and factorization.
    """
    p = torch.tensor([1500.0, 2000.0, 2500.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: _solve_batched(x, budget_bytes=1),
        (p,),
        eps=1e-4,
        atol=1e-6,
        rtol=1e-4,
    )


def test_gradcheck_through_sparse_shunt_woodbury_update():
    """The compact update retains every scenario's operating-point gradient."""
    p = torch.tensor([1500.0, 2000.0, 2500.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        _solve_sparse_shunt_batch,
        (p,),
        eps=1e-4,
        atol=1e-6,
        rtol=1e-4,
    )


def test_chunked_and_whole_batch_gradients_agree():
    """Chunking must not change the gradient, only the peak allocation."""

    def grad(budget):
        p = torch.tensor(
            [1500.0, 2000.0, 2500.0], dtype=torch.float64, requires_grad=True
        )
        _solve_batched(p, budget_bytes=budget).abs().sum().backward()
        return p.grad.detach().clone()

    g_whole = grad(None)
    g_chunked = grad(1)  # one scenario per chunk
    assert torch.allclose(g_whole, g_chunked, rtol=1e-12, atol=0.0)


def test_gradcheck_on_the_nameplate_basis():
    """The nameplate basis keeps the scenario power on the tape through the injection.

    ``Y(h)`` no longer depends on the scenario, but the harmonic current injection still
    does, so the gradient w.r.t. a batched load power must stay exact.
    """
    p = torch.tensor([1500.0, 2000.0, 2500.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: _solve_batched(x, basis="nameplate"),
        (p,),
        eps=1e-4,
        atol=1e-6,
        rtol=1e-4,
    )
