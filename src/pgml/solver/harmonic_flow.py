"""Harmonic power flow: nonlinear fundamental + linear per-harmonic solves.

Public API
----------
- ``solve_harmonic_flow(grid, harmonic_orders, *, slack, operating_point,
  harmonic_injection, include_load_shunt, tol, max_iter, dtype, device)
  -> HarmonicFlowResult``

Model (matches OpenDSS ``Solve mode=harmonics`` — see
``references/opendss/harmonics.md``):

1. Solve the nonlinear fundamental power flow (:func:`solve_power_flow`). Order 1 of
   the result is this solution.
2. From the converged fundamental voltage, each Load/Generator's per-phase
   FUNDAMENTAL current is ``I1 = sign * conj(S0) / conj(Vt)`` (load convention,
   ``sign`` +1 load / -1 gen, consistent with the const-Z stamp).
3. For each harmonic order ``h > 1`` (batched over all orders): the network is
   LINEAR. ``Y(h) = assemble_network_ybus(grid, [h*f0]) + source Norton shunt``.
   Each device injects (load-terminal convention) a harmonic current with
   ``|I_h| = (mag_h/mag_1) * |I1|`` and ``arg(I_h) = ang_h + h*(arg(I1) - ang_1)``
   from its ``Spectrum`` (or the ``harmonic_injection`` override). The NODAL
   injection is ``-I_h`` (the device draws it, same sense as the fundamental). The
   source is a Norton shunt held at zero harmonic voltage (no ideal slack here),
   so ``V(h) = solve_harmonic(Y(h), I(h))`` in Norton mode.

Differentiability + GPU: the fundamental carries IFT gradients (via
:func:`solve_power_flow`); the harmonics are linear and differentiable. Gradients
flow to network params, load P/Q, and the harmonic injections. Batched over leading
scenario dims. No ``.item()/.detach()/.numpy()`` on the tape; honors device/dtype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from pgml.assembly import NodePhaseIndex, assemble_network_ybus, node_phase_index
from pgml.assembly._params import resolve_operating_power
from pgml.assembly._stamps import _cdtype, _rdtype
from pgml.assembly.ybus import _stamp_sources
from pgml.schemas.grid_schema import Generator, Grid, Load, StaticSpectrum

from .harmonic import solve_harmonic
from .power_flow import PowerFlowResult, solve_power_flow


@dataclass(frozen=True)
class HarmonicFlowResult:
    """Per-order harmonic power-flow solution.

    Attributes
    ----------
    v:
        Complex node voltages ``[*batch, H, N]`` — one slice per requested order, in
        the order of ``harmonic_orders``. Order 1 (if requested) is the nonlinear
        :func:`solve_power_flow` solution; other orders are the linear per-harmonic
        solves. DIFFERENTIABLE.
    frequencies_hz:
        Real tensor ``[H]`` = ``order * f0`` for each requested order.
    index:
        The compact :class:`NodePhaseIndex` describing the row layout of ``v``.
    pf:
        The fundamental :class:`PowerFlowResult` (convergence info + order-1 V).
    """

    v: Tensor
    frequencies_hz: Tensor
    index: NodePhaseIndex
    pf: PowerFlowResult


def solve_harmonic_flow(
    grid: Grid,
    harmonic_orders,
    *,
    slack: str = "ideal",
    operating_point: Optional[dict] = None,
    harmonic_injection: Optional[dict] = None,
    include_load_shunt: bool = False,
    tol: float = 1e-10,
    max_iter: int = 100,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
) -> HarmonicFlowResult:
    """Solve the harmonic power flow (nonlinear fundamental + linear harmonics).

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.schemas.grid_schema.Grid`.
    harmonic_orders:
        Iterable of integer orders to solve (e.g. ``[1, 5, 7]``; 1 = fundamental).
    slack, operating_point, tol, max_iter, dtype, device:
        Passed to the fundamental :func:`solve_power_flow`.
    harmonic_injection:
        Optional SCENARIO override of per-device spectra, tensor-friendly so
        scenarios can vary harmonic injections differentiably. Format:
        ``{appliance_id: {order:int -> (magnitude_pu, phase_deg)}}`` where the values
        may be python floats OR 0-d tensors. Overrides the device's stored Spectrum.
    include_load_shunt:
        ``False`` (default) = OpenDSS ``NeglectLoadY`` pure current-source model.
        ``True`` (load Norton shunt from ``HarmonicShuntModel``) is NOT yet
        implemented (the exact OpenDSS shunt split is unpinned) and raises.

    Returns
    -------
    HarmonicFlowResult
        ``v`` complex ``[*batch, H, N]`` per requested order, frequencies, index, pf.
    """
    if include_load_shunt:
        raise NotImplementedError(
            "include_load_shunt=True (harmonic load Norton shunt) is not yet "
            "implemented; the exact OpenDSS shunt split is unpinned. Use False "
            "(pure current-source / NeglectLoadY model)."
        )

    orders = [int(h) for h in harmonic_orders]
    if not orders:
        raise ValueError("harmonic_orders must be non-empty.")

    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)
    f0 = float(grid.base_frequency_hz)
    index = node_phase_index(grid)

    # 1. Fundamental nonlinear power flow (order 1).
    pf = solve_power_flow(
        grid,
        slack=slack,
        operating_point=operating_point,
        tol=tol,
        max_iter=max_iter,
        dtype=dtype,
        device=device,
    )
    v1 = pf.v  # [*batch, N] complex
    if device is None:
        device = v1.device

    # 2./3. Harmonic orders (> 1), batched.
    harm = [h for h in orders if h != 1]
    v_by_order: dict[int, Tensor] = {1: v1}
    if harm:
        freqs = [h * f0 for h in harm]
        fvec = torch.as_tensor(freqs, dtype=rdt, device=device)
        yh = assemble_network_ybus(grid, freqs, dtype=dtype, device=device).Y
        if yh.ndim == 2:  # single harmonic returned [N, N] -> [1, N, N]
            yh = yh.unsqueeze(0)
        yh = _stamp_sources(grid, fvec, yh, index, cdt, rdt, device, None)  # [Hh, N, N]
        ih = _harmonic_injections(
            grid, v1, index, harm, operating_point, harmonic_injection, cdt, rdt, device
        )  # [*batch, Hh, N]
        vh = solve_harmonic(yh, ih)  # Norton mode -> [*batch, Hh, N]
        for k, h in enumerate(harm):
            v_by_order[h] = vh[..., k, :]

    n = index.size
    cols = [v_by_order[h] for h in orders]
    bshape = torch.broadcast_shapes(*[c.shape[:-1] for c in cols])
    cols = [c.broadcast_to(*bshape, n) for c in cols]
    v = torch.stack(cols, dim=-2)  # [*batch, H, N]
    frequencies_hz = torch.as_tensor([h * f0 for h in orders], dtype=rdt, device=device)
    return HarmonicFlowResult(v=v, frequencies_hz=frequencies_hz, index=index, pf=pf)


# ---------------------------------------------------------------------------
# harmonic current injection from spectra
# ---------------------------------------------------------------------------
def _resolve_spectrum(appliance, harmonic_injection: Optional[dict]) -> Optional[dict]:
    """Return ``{order: (magnitude_pu, phase_deg)}`` for a device, or ``None``.

    Override (``harmonic_injection[id]``) wins over the stored ``StaticSpectrum``.
    Values may be floats or 0-d tensors (kept for the differentiable path).
    """
    if harmonic_injection is not None and appliance.id in harmonic_injection:
        return dict(harmonic_injection[appliance.id])
    spec = getattr(appliance, "spectrum", None)
    if isinstance(spec, StaticSpectrum):
        return {c.order: (c.magnitude_pu, c.phase_deg) for c in spec.spectrum.components}
    return None


def _as_rt(x, rdt, device) -> Tensor:
    """Coerce a float or (0-d) tensor to a real tensor, preserving grad for tensors."""
    if isinstance(x, Tensor):
        return x.to(dtype=rdt, device=device)
    return torch.as_tensor(x, dtype=rdt, device=device)


def _broadcast_last(x: Tensor) -> Tensor:
    """Append a trailing phase axis to a scalar/batched per-order coefficient."""
    return x.reshape(*x.shape, 1) if x.ndim > 0 else x


def _harmonic_injections(
    grid, v1, index, harm_orders, operating_point, harmonic_injection, cdt, rdt, device
) -> Tensor:
    """Per-device harmonic nodal current injection ``[*batch, Hh, N]``.

    For each Load/Generator with a spectrum: ``I1 = sign*conj(S0)/conj(Vt)`` from the
    fundamental voltage; ``I_h = (mag_h/mag_1)|I1| ∠ (ang_h + h*(arg(I1)-ang_1))`` is
    the device's drawn harmonic current; the NODAL injection is ``-I_h``. Loops over
    devices and orders (python, both small/fixed); fully broadcasting so per-order
    magnitudes/phases may carry a SCENARIO batch dim (differentiable override).
    """
    n = index.size

    # Precompute per-device fundamental current info.
    devs = []
    for a in grid.appliances:
        if not (isinstance(a, (Load, Generator)) and a.in_service):
            continue
        spec = _resolve_spectrum(a, harmonic_injection)
        if spec is None:
            continue
        sign = 1.0 if isinstance(a, Load) else -1.0
        p_list, q_list = resolve_operating_power(a, operating_point)
        p_t = torch.stack([_as_rt(x, rdt, device) for x in p_list])  # [P]
        q_t = torch.stack([_as_rt(x, rdt, device) for x in q_list])  # [P]
        s0 = torch.complex(sign * p_t, sign * q_t).to(cdt)  # [P]
        rows = index.rows_for_terminal(a.node, a.phases, device=device)  # [P] int64
        vt = v1.index_select(-1, rows).to(cdt)  # [*vbatch, P]
        i1 = torch.conj(s0) / torch.conj(vt)  # [*vbatch, P]
        mag1 = _as_rt(spec.get(1, (1.0, 0.0))[0], rdt, device)
        ang1 = _as_rt(spec.get(1, (1.0, 0.0))[1], rdt, device) * (math.pi / 180.0)
        devs.append((spec, torch.abs(i1), torch.angle(i1), rows, mag1, ang1))

    cols: list[Tensor] = []
    for h in harm_orders:
        contribs = []  # (rows, i_h [*batch, P])
        for spec, i1_mag, i1_ang, rows, mag1, ang1 in devs:
            ratio = _broadcast_last(_as_rt(spec.get(h, (0.0, 0.0))[0], rdt, device) / mag1)
            ang_h = _broadcast_last(
                _as_rt(spec.get(h, (0.0, 0.0))[1], rdt, device) * (math.pi / 180.0)
            )
            mag = ratio * i1_mag  # [*batch, P]
            phase = ang_h + float(h) * (i1_ang - ang1)  # [*batch, P]
            contribs.append((rows, torch.polar(mag, phase)))
        bshape = torch.broadcast_shapes(*[c.shape[:-1] for _, c in contribs]) if contribs else ()
        col = torch.zeros((*bshape, n), dtype=cdt, device=device)
        for rows, i_h in contribs:
            p = rows.shape[0]
            i_h_b = i_h.broadcast_to(*bshape, p)
            idx = rows.view(*([1] * len(bshape)), p).expand(*bshape, p)
            col = col.scatter_add(-1, idx, -i_h_b)  # nodal injection = -I_drawn
        cols.append(col)

    bshape = torch.broadcast_shapes(*[c.shape[:-1] for c in cols])
    cols = [c.broadcast_to(*bshape, n) for c in cols]
    return torch.stack(cols, dim=-2)  # [*batch, Hh, N]


__all__ = ["solve_harmonic_flow", "HarmonicFlowResult"]
