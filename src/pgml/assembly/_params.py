"""Differentiable parameter extraction from materialised schema objects.

These helpers turn schema model fields (Python lists / floats / tuples) into
batched torch tensors that enter the differentiable Y-bus path. The conversion
from python numbers to a tensor is the START of the autograd tape; downstream
math is torch and gradients flow back to these tensors (a test wishing to
gradcheck w.r.t. R/L/C/Z replaces the relevant tensor with a leaf — see the
``param_overrides`` hook on the public assembly API).

No differentiable in-place ops; loops here are over the (small, fixed) python
collection of components, building lists that are stacked ONCE into a tensor.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence

import torch
from torch import Tensor


class _SolvedPVOperatingPoint(dict):
    """Solver-produced device readout, not a prescribed per-phase Q override.

    Created by PowerFlowResult.resolved_operating_point. The distinct entry type
    prevents ordinary input dictionaries from bypassing symmetric power splitting.
    It carries no additional state and must be preserved when slicing a batch.
    """


def _preserve_solved_pv_q(appliance, operating_point) -> bool:
    """Identify solved Q readout; warn if a raw PV phase split will be averaged."""
    if getattr(appliance, "voltage_regulation", None) is None:
        return False
    entry = (operating_point or {}).get(appliance.id, {})
    if "q_per_phase_var" not in entry:
        return False
    if isinstance(entry, _SolvedPVOperatingPoint):
        return True
    logging.getLogger("pgml").warning(
        "generator %s: q_per_phase_var phase allocation is ignored for a "
        "symmetric calculation; the total is split equally. Only solver-produced "
        "PV reactive-power readout retains its solved phase allocation.",
        appliance.id,
    )
    return False


def phase_voltage_magnitude(
    u_rated_v: float, n_phases: int, *, line_to_line: bool = False
) -> float:
    """Nominal element voltage magnitude ``V0`` for the const-Z / ZIP load model.

    The schema stores ``Node.u_rated_v`` as line-to-line for 3-phase nodes and
    line-to-neutral for 1-phase nodes.

    - ``line_to_line=False`` (WYE, the default): a line-to-NEUTRAL element voltage —
      divide by sqrt(3) when 3 (or more) phases are present (``u_rated`` for 1-phase
      nodes is already L-N).
    - ``line_to_line=True`` (DELTA): the element sees the full line-to-LINE voltage,
      so return ``u_rated_v`` directly for a 3-phase node (no sqrt(3) division).

    Connection-aware per ``docs/pgml/modeling/asymmetric.md`` section 2 (OpenDSS
    ``VBase``: wye 3-phase uses ``kVLL/sqrt(3)``; delta uses the supplied kV).
    """
    if line_to_line:
        return u_rated_v
    # ``n_phases >= 3`` intentionally also covers a 4-wire ABCN node (4 phases):
    # ``u_rated`` is the line-to-line value for ANY node with >= 3 phases, so a WYE
    # element there sees ``u_rated / sqrt(3)`` regardless of whether a neutral row
    # is modeled.
    if n_phases >= 3:
        return u_rated_v / math.sqrt(3.0)
    return u_rated_v


def const_z_shunt_admittance(
    p_per_phase: Sequence[float],
    q_per_phase: Sequence[float],
    u_ln_v: float,
    sign: float,
    dtype: torch.dtype,
    device,
) -> Tensor:
    """Per-phase complex const-Z shunt admittance ``y = conj(P + jQ)/|U|^2``.

    Parameters
    ----------
    p_per_phase, q_per_phase:
        Per-phase active/reactive operating-point power (W / var).
    u_ln_v:
        Line-to-neutral voltage magnitude.
    sign:
        ``+1`` for a load (consumes), ``-1`` for a generator (injects), applied to
        both P and Q before forming the admittance.
    Returns
    -------
    Tensor
        Complex ``[P]`` per-phase shunt admittance (diagonal stamp values).
    """
    rdt = torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32

    def _stack(per_phase) -> Tensor:
        # A per-phase entry may be a python float OR a (possibly scenario-batched)
        # tensor; stack along the trailing phase axis, broadcasting any leading
        # batch dims to a common shape (graph-preserving for tensor leaves).
        ts = [
            x.to(dtype=rdt, device=device)
            if isinstance(x, Tensor)
            else torch.as_tensor(x, dtype=rdt, device=device)
            for x in per_phase
        ]
        lead = torch.broadcast_shapes(*[t.shape for t in ts])
        return torch.stack([t.broadcast_to(lead) for t in ts], dim=-1)

    p = sign * _stack(p_per_phase)
    q = sign * _stack(q_per_phase)
    u2 = u_ln_v * u_ln_v
    # y = conj(P + jQ) / |U|^2 = (P - jQ) / |U|^2
    cdt = torch.complex128 if rdt == torch.float64 else torch.complex64
    s = torch.complex(p, -q).to(cdt)
    return s / u2


def _tensor_sum(per_phase: Sequence):
    """Autograd-safe sum of a per-phase sequence (mixed python floats / tensors).

    Mirrors :func:`pgml.assembly.ybus._tensor_sum` but does NOT coerce dtype/device:
    a pure-float input stays a python float (so the const-Z reference path is
    numerically unchanged), while a tensor anywhere in the sequence makes the total a
    graph-preserving tensor (gradients flow back to each per-phase leaf). Uses plain
    ``+`` accumulation, which is autograd-safe and never an in-place op.
    """
    total = None
    for x in per_phase:
        total = x if total is None else total + x
    return total


def resolve_operating_power(
    appliance,
    operating_point: Optional[dict],
    *,
    asymmetric: bool = True,
) -> tuple[list[float], list[float]]:
    """Per-phase (P, Q) operating point for a Load/Generator, length == phases.

    Resolution order (when ``asymmetric=True``):
    1. ``operating_point[appliance.id]`` if given, a dict with ``p_w`` / ``q_var``
       (totals) and/or ``p_per_phase_w`` / ``q_per_phase_var``.
    2. The appliance's ``*_per_phase_*`` nameplate split if present.
    3. The total nameplate ``p_nom_w`` / ``q_nom_var`` split equally across phases.

    When ``asymmetric=False`` (a SYMMETRIC calculation), every per-phase value is
    IGNORED and the total is split equally over the phases (the power-grid-model
    rule: a symmetric calculation averages an asymmetric load —
    ``docs/pgml/modeling/asymmetric.md`` section 1). The total is taken from a
    total operating-point override if given, else from the per-phase override / the
    per-phase nameplate (summed), else from the total nameplate.

    Ordinary per-phase Q overrides are not exempt for voltage-regulating generators:
    their phase split is ignored with a warning in symmetric calculations. The
    fundamental PV solver ignores prescribed Q altogether and replaces it with
    constraint/limit values before evaluating its residual.

    Solver-produced entries from ``PowerFlowResult.resolved_operating_point`` are
    READOUT, not prescribed inputs: their solved Q allocation is preserved for
    harmonic initialization and device-current recovery. Symmetry balances input
    powers, not the phase-domain network or its solved outputs. Averaging solved Q
    on an unbalanced network would invalidate the converged nodal balance.
    """
    n = len(appliance.phases)
    p_total = appliance.p_nom_w
    q_total = appliance.q_nom_var
    p_per = appliance.p_nom_per_phase_w
    q_per = appliance.q_nom_per_phase_var

    if operating_point is not None and appliance.id in operating_point:
        op = operating_point[appliance.id]
        if "p_per_phase_w" in op:
            p_per = op["p_per_phase_w"]
        elif "p_w" in op:
            p_total = op["p_w"]
            p_per = None
        if "q_per_phase_var" in op:
            q_per = op["q_per_phase_var"]
        elif "q_var" in op:
            q_total = op["q_var"]
            q_per = None

    if not asymmetric:
        # Symmetric calc: ignore per-phase data, split the total equally. Prefer an
        # explicit total; otherwise sum a per-phase spec back to a total with the
        # autograd-safe reduction (``_tensor_sum`` keeps the graph when the per-phase
        # entries are tensor leaves; a plain ``sum`` over tensors is graph-preserving
        # too but goes through python ``+``, which we make explicit here).
        p_t = _tensor_sum(p_per) if p_per is not None else p_total
        q_t = _tensor_sum(q_per) if q_per is not None else q_total
        # Build n INDEPENDENT entries (each ``/ n`` is a fresh autograd node). A
        # ``[x] * n`` literal would alias ONE object into every slot, so a per-phase
        # gradient would wrongly perturb all phases under tensor duality.
        # PV input Q is replaced by the fundamental solver. A per-phase Q
        # override on its solved operating point is an OUTPUT allocation, which
        # must survive even when the requested input powers were symmetric.
        solved_q = _preserve_solved_pv_q(appliance, operating_point)
        return [p_t / n for _ in range(n)], (
            list(q_per) if solved_q else [q_t / n for _ in range(n)]
        )

    # Same independent-entries rule as the symmetric branch above: each ``/ n``
    # is a fresh autograd node, never one object aliased into every slot.
    p_list = list(p_per) if p_per is not None else [p_total / n for _ in range(n)]
    q_list = list(q_per) if q_per is not None else [q_total / n for _ in range(n)]
    return p_list, q_list


__all__ = [
    "phase_voltage_magnitude",
    "const_z_shunt_admittance",
    "resolve_operating_power",
]
