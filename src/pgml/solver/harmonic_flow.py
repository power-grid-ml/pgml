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
2. From the converged fundamental voltage, each Load/Generator's per-ELEMENT
   FUNDAMENTAL current is ``I1_elem = sign * conj(S0_elem) / conj(V_term)`` (load
   convention, ``sign`` +1 load / -1 gen, consistent with the const-Z stamp). The
   ELEMENT (terminal) voltage ``V_term = M @ V_used`` uses the SAME connection
   incidence ``M`` the load flow uses (``pgml.assembly._incidence``): WYE-ground
   ``M = I``, WYE-neutral ``[I|-1]`` (terminal = ``V_phase - V_N``), DELTA-3
   circulant (terminal = L-L difference).
3. For each harmonic order ``h > 1`` (batched over all orders): the network is
   LINEAR. ``Y(h) = assemble_network_ybus(grid, [h*f0]) + source Norton shunt``.
   Each device injects (load-terminal convention) a per-element harmonic current
   ``|I_h^e| = (mag_h^e/mag_1^e) * |I1_elem|`` and
   ``arg(I_h^e) = ang_h^e + h*(arg(I1_elem) - ang_1^e)`` from its per-ELEMENT
   spectrum coefficients (device ``spectrum`` broadcast on all elements,
   ``spectrum_per_phase`` per phase/branch, or the ``harmonic_injection`` override).
   The NODAL injection is ``-(M^T @ i_h_elem)`` (the device draws it, same sense as
   the fundamental). The source is a Norton shunt held at zero harmonic voltage (no
   ideal slack here), so ``V(h) = solve_harmonic(Y(h), I(h))`` in Norton mode.

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
from pgml.assembly._incidence import build_incidence, group_appliances, used_rows
from pgml.assembly._params import resolve_operating_power
from pgml.assembly._stamps import _cdtype, _rdtype
from pgml.assembly._symmetry import resolve_asymmetric
from pgml.assembly.ybus import _stamp_sources
from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    Load,
    StaticSpectrum,
)

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
    symmetry: Optional[str] = None,
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
        ``{appliance_id: {order:int -> (magnitude_pu, phase_deg)}}``. Each
        ``magnitude_pu`` / ``phase_deg`` value follows an UNAMBIGUOUS convention:

        - a python ``list``/``tuple`` is PER-ELEMENT — it MUST have length ``n_elem``
          (aligned to the device's elements: WYE phase ``k`` / DELTA branch ``k``);
          each entry may itself be a python float or a 0-d / ``[*batch]`` tensor. A
          list/tuple of any other length is an error.
        - a SCALAR or bare tensor (python float, 0-d tensor, or a ``[*batch]`` tensor
          carrying ONLY leading SCENARIO batch dims, NO element axis) is BROADCAST
          identically to every element (the backward-compatible path).

        The override wins over the device's stored ``spectrum`` / ``spectrum_per_phase``.
    include_load_shunt:
        ``False`` (default) = OpenDSS ``NeglectLoadY`` pure current-source model.
        ``True`` (load Norton shunt from ``HarmonicShuntModel``) is NOT yet
        implemented (the exact OpenDSS shunt split is unpinned) and raises.
    symmetry:
        Calculation-symmetry mode ``None`` / ``"auto"`` / ``"symmetric"`` /
        ``"asymmetric"`` (``None`` -> config). Resolved ONCE here and threaded into
        the fundamental :func:`solve_power_flow` (single log) and the harmonic
        injection power resolution.

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

    # Resolve calculation symmetry ONCE; thread the canonical string into the
    # fundamental PF (which emits the single modeling-summary log).
    asymmetric = resolve_asymmetric(grid, operating_point, mode=symmetry)
    sym_resolved = "asymmetric" if asymmetric else "symmetric"

    # 1. Fundamental nonlinear power flow (order 1).
    pf = solve_power_flow(
        grid,
        slack=slack,
        operating_point=operating_point,
        tol=tol,
        max_iter=max_iter,
        dtype=dtype,
        device=device,
        symmetry=sym_resolved,
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
            grid,
            v1,
            index,
            harm,
            operating_point,
            harmonic_injection,
            cdt,
            rdt,
            device,
            asymmetric,
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
# harmonic current injection from spectra (connection-aware, per-element)
# ---------------------------------------------------------------------------
def _as_rt(x, rdt, device) -> Tensor:
    """Coerce a float or tensor to a real tensor, preserving grad for tensors."""
    if isinstance(x, Tensor):
        return x.to(dtype=rdt, device=device)
    return torch.as_tensor(x, dtype=rdt, device=device)


def _static_spectrum_dict(spec_obj) -> Optional[dict]:
    """``{order: (magnitude_pu, phase_deg)}`` (floats) from a ``StaticSpectrum``, or None."""
    if isinstance(spec_obj, StaticSpectrum):
        return {
            c.order: (c.magnitude_pu, c.phase_deg) for c in spec_obj.spectrum.components
        }
    return None


def _element_coeff(value, n_elem: int, rdt, device) -> Tensor:
    """Per-element coefficient tensor with trailing dim ``n_elem``.

    UNAMBIGUOUS convention (no trailing-dim sniffing, which would silently misread a
    length-``n_elem`` SCENARIO batch as per-element):

    - a python ``list``/``tuple`` is PER-ELEMENT: it MUST have length ``n_elem`` (each
      entry may itself be a python float or a 0-d / ``[*batch]`` tensor, stacked on a
      new trailing element axis, preserving per-entry grad). Any other length is an
      error.
    - anything else (python float, 0-d tensor, or a ``[*batch]`` tensor carrying ONLY
      leading SCENARIO batch dims) is SCALAR/BATCH: it is BROADCAST identically to all
      ``n_elem`` elements (append + ``expand`` the element axis — NOT by matching a
      trailing dim to ``n_elem``).

    Returns a real tensor whose trailing dim is ``n_elem`` (with any leading batch dims
    preserved).
    """
    if isinstance(value, (list, tuple)):
        if len(value) != n_elem:
            raise ValueError(
                f"per-element harmonic coefficient (list/tuple) has length "
                f"{len(value)} != n_elem {n_elem}. A list/tuple is ALWAYS per-element; "
                f"use a scalar / bare tensor to broadcast across elements."
            )
        return torch.stack([_as_rt(x, rdt, device) for x in value], dim=-1)
    # scalar / leading-batch-only tensor -> broadcast across the element axis.
    t = _as_rt(value, rdt, device)
    return t.unsqueeze(-1).expand(*t.shape, n_elem)


def _device_element_spectra(
    appliance, harmonic_injection, n_elem, rdt, device
) -> Optional[dict]:
    """Per-ELEMENT per-order ``{order: (mag_elem, phase_deg_elem)}`` for a device.

    Each value is a real tensor whose trailing dim is ``n_elem`` (broadcastable with
    leading SCENARIO batch dims). Three sources, override winning:

    - ``harmonic_injection[id]`` (runtime override): ``{order: (mag, phase_deg)}``
      where a python list/tuple of length ``n_elem`` applies PER ELEMENT and a scalar
      or bare tensor BROADCASTS to all elements (see :func:`_element_coeff`).
    - ``spectrum_per_phase`` (schema): element ``k`` uses the spectrum stored at
      ``phases[k]`` (for DELTA-3 the delta branch ``k`` keys off ``phases[k]``); an
      element whose phase has no entry injects NO harmonics (all orders 0).
    - ``spectrum`` (schema, ``StaticSpectrum``): the SAME spectrum on every element.

    Returns ``None`` if the device injects no harmonics at all.
    """
    phases = appliance.phases  # element k <-> phases[k] (WYE phase / DELTA branch)

    override = (
        harmonic_injection.get(appliance.id) if harmonic_injection is not None else None
    )
    if override is not None:
        return {
            int(o): (
                _element_coeff(mag, n_elem, rdt, device),
                _element_coeff(ph, n_elem, rdt, device),
            )
            for o, (mag, ph) in override.items()
        }

    spp = getattr(appliance, "spectrum_per_phase", None)
    if spp is not None:
        # Gather the set of orders any element carries; element k uses phases[k].
        per_elem_dicts = [
            _static_spectrum_dict(spp.get(phases[k])) for k in range(n_elem)
        ]
        all_orders: set[int] = set()
        for d in per_elem_dicts:
            if d is not None:
                all_orders.update(d.keys())
        if not all_orders:
            return None
        out: dict[int, tuple] = {}
        for o in all_orders:
            mags = [
                (d[o][0] if (d is not None and o in d) else 0.0) for d in per_elem_dicts
            ]
            phs = [
                (d[o][1] if (d is not None and o in d) else 0.0) for d in per_elem_dicts
            ]
            out[o] = (
                torch.stack([_as_rt(m, rdt, device) for m in mags], dim=-1),
                torch.stack([_as_rt(p, rdt, device) for p in phs], dim=-1),
            )
        return out

    sd = _static_spectrum_dict(getattr(appliance, "spectrum", None))
    if sd is None:
        return None
    return {
        o: (
            _as_rt(mag, rdt, device).reshape(()).expand(n_elem),
            _as_rt(ph, rdt, device).reshape(()).expand(n_elem),
        )
        for o, (mag, ph) in sd.items()
    }


def _harmonic_injections(
    grid,
    v1,
    index,
    harm_orders,
    operating_point,
    harmonic_injection,
    cdt,
    rdt,
    device,
    asymmetric=True,
) -> Tensor:
    """Per-device, connection-aware harmonic nodal current injection ``[*batch, Hh, N]``.

    Mirrors :func:`pgml.assembly.ybus.device_current_injections`: each injecting
    Load/Generator has a terminal incidence ``M`` ``[n_elem, n_used]``
    (``V_term = M @ V_used``); WYE-ground ``M = I``, WYE-neutral ``[I|-1]``, DELTA-3
    circulant. The per-ELEMENT fundamental current is
    ``I1_elem = sign*conj(S0_elem)/conj(V_term)`` (TERMINAL voltage, not the phase
    row), then per element and order

        ``|I_h^e| = (mag_h^e/mag_1^e)*|I1_elem|``
        ``arg(I_h^e) = ang_h^e + h*(arg(I1_elem) - ang_1^e)``

    with PER-ELEMENT spectrum coefficients (device ``spectrum`` broadcast on all
    elements, ``spectrum_per_phase`` per phase/branch, or the ``harmonic_injection``
    override). In the override, a coefficient given as a python ``list``/``tuple`` of
    length ``n_elem`` is PER-ELEMENT, while a scalar or bare tensor is BROADCAST to all
    elements (see :func:`_element_coeff`). The NODAL injection (drawn) is
    ``I_used = -(M^T @ i_h_elem)`` scattered into the device's ``used_rows``.
    WYE-to-ground reduces EXACTLY to the historical per-phase form (bit-exact
    regression). Fully broadcasting so per-element / per-order coefficients may carry
    leading SCENARIO batch dims (differentiable override).
    """
    n = index.size
    node_map = {nd.id: nd for nd in grid.nodes}

    loads = [
        a for a in grid.appliances if isinstance(a, (Load, Generator)) and a.in_service
    ]

    # Per device that injects: (group, rows[n_used], i1_mag[*b,n_elem],
    # i1_ang[*b,n_elem], {order:(mag_elem,ang1_elem?)} via element spectra).
    # We collect the per-element fundamental current and the element spectra, grouped
    # so the incidence M (a topology constant) is shared.
    entries = []  # (m_c, rows, i1_mag, i1_ang, elem_spectra, n_elem)
    for grp in group_appliances(loads, node_map):
        n_elem = grp.n_elem
        m = build_incidence(grp, rdt, device)  # [n_elem, n_used] real
        m_c = m.to(cdt)
        rows_grp = used_rows(grp, index, device)  # [K, n_used] int64
        for ki, a in enumerate(grp.appliances):
            elem_spectra = _device_element_spectra(
                a, harmonic_injection, n_elem, rdt, device
            )
            if elem_spectra is None:
                continue  # device injects no harmonics.
            sign = 1.0 if isinstance(a, Load) else -1.0
            p_list, q_list = resolve_operating_power(
                a, operating_point, asymmetric=asymmetric
            )
            p_t = torch.stack(
                [_as_rt(x, rdt, device) for x in p_list], dim=-1
            )  # [*b,n_elem]
            q_t = torch.stack([_as_rt(x, rdt, device) for x in q_list], dim=-1)
            s0 = torch.complex(sign * p_t, sign * q_t).to(cdt)  # [*b, n_elem]

            rows = rows_grp[ki]  # [n_used]
            v_used = v1.index_select(-1, rows).to(cdt)  # [*vbatch, n_used]
            # V_term[..., e] = sum_u M[e,u] V_used[..., u].
            vt = torch.einsum("eu,...u->...e", m_c, v_used)  # [*vbatch, n_elem]
            # Guard the conj(vt) divide for a dead/disconnected terminal (vt == 0)
            # or a gradcheck perturbation toward zero: mask the DENOMINATOR before
            # dividing (so conj(s0)/0 never enters the graph), then mask the RESULT
            # to 0 afterwards. Same two-step pattern as the mag1 == 0 ratio guard.
            vtc = torch.conj(vt)
            safe_vtc = torch.where(vtc.abs() < 1e-300, torch.ones_like(vtc), vtc)
            i1 = torch.where(
                vtc.abs() < 1e-300, torch.zeros_like(s0), torch.conj(s0) / safe_vtc
            )  # [*batch, n_elem]

            mag1_e, ang1_e = elem_spectra.get(
                1,
                (
                    torch.ones(n_elem, dtype=rdt, device=device),
                    torch.zeros(n_elem, dtype=rdt, device=device),
                ),
            )
            ang1_e = ang1_e * (math.pi / 180.0)
            entries.append(
                (
                    m_c,
                    rows,
                    torch.abs(i1),
                    torch.angle(i1),
                    mag1_e,
                    ang1_e,
                    elem_spectra,
                )
            )

    cols: list[Tensor] = []
    for h in harm_orders:
        contribs = []  # (m_c, rows, i_h_elem [*batch, n_elem])
        for m_c, rows, i1_mag, i1_ang, mag1_e, ang1_e, elem_spectra in entries:
            n_elem = mag1_e.shape[-1]
            mag_h_e, ph_h_e = elem_spectra.get(
                h,
                (
                    torch.zeros(n_elem, dtype=rdt, device=device),
                    torch.zeros(n_elem, dtype=rdt, device=device),
                ),
            )
            # Safe ratio mag_h/mag1: an element with NO spectrum has mag1 == 0 (and
            # mag_h == 0) -> it injects nothing; avoid the 0/0 NaN with a guarded
            # divide (the where keeps the gradient finite on the live elements).
            safe1 = torch.where(mag1_e == 0, torch.ones_like(mag1_e), mag1_e)
            ratio = torch.where(
                mag1_e == 0, torch.zeros_like(mag_h_e), mag_h_e / safe1
            )  # [*batch, n_elem]
            ang_h = ph_h_e * (math.pi / 180.0)
            mag = ratio * i1_mag  # [*batch, n_elem]
            phase = ang_h + float(h) * (i1_ang - ang1_e)  # [*batch, n_elem]
            contribs.append((m_c, rows, torch.polar(mag, phase)))

        bshape = (
            torch.broadcast_shapes(*[c.shape[:-1] for _, _, c in contribs])
            if contribs
            else ()
        )
        col = torch.zeros((*bshape, n), dtype=cdt, device=device)
        for m_c, rows, i_h_elem in contribs:
            # Nodal current at the used rows: I_used = -(M^T @ i_elem) (drawn).
            i_used = torch.einsum("eu,...e->...u", m_c, i_h_elem)  # [*batch, n_used]
            # `broadcast_to` returns a VIEW; a non-contiguous complex tensor can fail
            # the CUDA index_add backend, so materialise it (defensive, matches
            # ybus._scatter_injection). `.contiguous()` is autograd-safe.
            i_used = i_used.broadcast_to(*bshape, rows.shape[0]).contiguous()
            # Out-of-place index_add (GPU-safe for COMPLEX, unlike scatter_add).
            col = col.index_add(-1, rows, -i_used)
        cols.append(col)

    bshape = torch.broadcast_shapes(*[c.shape[:-1] for c in cols])
    cols = [c.broadcast_to(*bshape, n) for c in cols]
    return torch.stack(cols, dim=-2)  # [*batch, Hh, N]


__all__ = ["solve_harmonic_flow", "HarmonicFlowResult"]
