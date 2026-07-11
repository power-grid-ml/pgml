"""Storage dispatch and state-of-charge integration (time-series, off the autodiff tape).

A :class:`~pgml.schemas.grid_schema.Storage` element is a signed ``(P, Q)`` injection at
any single power-flow snapshot (``p_nom_w > 0`` = discharging / injecting). What couples
timesteps is the state of charge and the dispatch decision; following pandapower and
OpenDSS, these are resolved HERE — outside the per-snapshot solve — into a realized
per-step active-power sequence the solver consumes as an ``operating_point``.

The dispatch RULE (a profile, a threshold, a price signal, a sampled human-behaviour
trace) is ordinary Python control flow and carries no gradient: the caller supplies the
REQUESTED power sequence. :func:`integrate_soc` then realizes it under the energy-state
constraints with torch, so a gradient w.r.t. the realized setpoint value (and, where the
power is not clamped by a SoC/rating limit, through the SoC recurrence) is available — but
never through the decision logic. See ``docs/pgml/modeling/der-pv-storage.md`` section 4.4.

State-of-charge energy update per step (discharge-positive convention, OpenDSS
``Storage`` semantics): discharging draws ``P*dt / eff_discharge`` from the store;
charging adds ``|P|*eff_charge*dt``::

    E[t+1] = E[t] - ( P[t]*dt/eff_discharge   if P[t] >= 0   (discharge)
                      P[t]*eff_charge*dt       if P[t] <  0   (charge) )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from pgml.errors import InputError
from pgml.schemas.grid_schema import Storage


@dataclass(frozen=True)
class StorageDispatchResult:
    """Realized storage dispatch over a horizon.

    Attributes
    ----------
    realized_power_w:
        The active power actually delivered/absorbed each step ``[*batch, T]`` (signed,
        ``> 0`` = discharge). Equals the requested power except where a SoC reserve / cap
        or the power rating clamped it.
    soc:
        State of charge as a fraction ``[*batch, T + 1]`` (``soc[..., 0]`` is the initial
        state); ``None`` if no ``energy_capacity_wh`` was given (no state tracked).
    energy_wh:
        Stored energy ``[*batch, T + 1]`` in watt-hours, or ``None`` (as ``soc``).
    """

    realized_power_w: Tensor
    soc: Optional[Tensor]
    energy_wh: Optional[Tensor]


def integrate_soc(
    requested_power_w,
    dt_s: float,
    *,
    energy_capacity_wh: Optional[float] = None,
    soc0: float = 0.5,
    soc_min: float = 0.0,
    soc_max: float = 1.0,
    efficiency_charge: float = 1.0,
    efficiency_discharge: float = 1.0,
    p_rated_w: Optional[float] = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> StorageDispatchResult:
    """Realize a requested storage power sequence under SoC + rating limits.

    Only ``requested_power_w`` is differentiable. Every LIMIT parameter
    (``energy_capacity_wh``, ``soc*``, ``efficiency_*``, ``p_rated_w``) is read
    as a plain float — a tensor passed here (e.g. straight off a tensor-valued
    ``Storage`` schema field) is deliberately treated as a constant, per the
    module's off-tape decision-logic design. Learning battery SIZING through
    this function is not supported.

    Parameters
    ----------
    requested_power_w:
        Requested active power per step, shape ``[*batch, T]`` (signed, ``> 0`` =
        discharge). A python list / array-like or a tensor (gradients preserved).
    dt_s:
        Time-step length in seconds.
    energy_capacity_wh:
        Usable energy capacity. ``None`` disables SoC tracking (only the power rating, if
        any, clamps the request) and returns ``soc = energy_wh = None``.
    soc0, soc_min, soc_max:
        Initial / minimum / maximum state of charge (fractions in ``[0, 1]``).
    efficiency_charge, efficiency_discharge:
        One-way charge / discharge efficiencies in ``(0, 1]``.
    p_rated_w:
        Inverter power rating bounding ``|P|``; ``None`` = unbounded.
    dtype, device:
        Real dtype / device of the computation.

    Returns
    -------
    StorageDispatchResult
        The realized power and (if a capacity was given) the SoC / energy trajectories.
    """
    p_req = (
        requested_power_w.to(dtype=dtype, device=device)
        if isinstance(requested_power_w, Tensor)
        else torch.as_tensor(requested_power_w, dtype=dtype, device=device)
    )
    if p_req.ndim == 0:
        raise InputError("requested_power_w must have a trailing time axis [*, T].")
    t = p_req.shape[-1]
    batch = p_req.shape[:-1]
    p_rated = (
        torch.as_tensor(float(p_rated_w), dtype=dtype, device=device)
        if p_rated_w is not None
        else None
    )

    if energy_capacity_wh is None:
        realized = p_req if p_rated is None else torch.clamp(p_req, -p_rated, p_rated)
        return StorageDispatchResult(
            realized_power_w=realized, soc=None, energy_wh=None
        )

    cap = float(energy_capacity_wh)
    dt_h = dt_s / 3600.0  # energy in watt-hours
    e = torch.full(batch, soc0 * cap, dtype=dtype, device=device)
    e_floor = soc_min * cap
    e_ceil = soc_max * cap

    realized_steps, energy_steps = [], [e]
    for k in range(t):
        p_k = p_req[..., k]
        # Power bounds from the present energy: discharge cannot take the store below the
        # reserve; charging cannot exceed the cap (efficiency-adjusted), and |P| <= rating.
        p_max_dis = (e - e_floor).clamp_min(0.0) * efficiency_discharge / dt_h
        p_max_chg = (e_ceil - e).clamp_min(0.0) / (efficiency_charge * dt_h)
        if p_rated is not None:
            p_max_dis = torch.minimum(p_max_dis, p_rated)
            p_max_chg = torch.minimum(p_max_chg, p_rated)
        p_real = torch.clamp(p_k, -p_max_chg, p_max_dis)
        # Energy removed from the store this step (negative when charging).
        draw = torch.where(
            p_real >= 0,
            p_real * dt_h / efficiency_discharge,
            p_real * efficiency_charge * dt_h,
        )
        e = e - draw
        realized_steps.append(p_real)
        energy_steps.append(e)

    realized = torch.stack(realized_steps, dim=-1)  # [*batch, T]
    energy = torch.stack(energy_steps, dim=-1)  # [*batch, T+1]
    return StorageDispatchResult(
        realized_power_w=realized, soc=energy / cap, energy_wh=energy
    )


def dispatch_storage(
    storage: Storage,
    requested_power_w,
    dt_s: float,
    *,
    soc0: Optional[float] = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> StorageDispatchResult:
    """Realize a dispatch for a :class:`Storage` element using its own state parameters.

    Reads ``energy_capacity_wh``, ``soc`` (initial, overridable via ``soc0``),
    ``soc_min`` / ``soc_max``, the charge/discharge efficiencies, and ``p_rated_w`` from
    the element; delegates to :func:`integrate_soc`.
    """
    s0 = soc0 if soc0 is not None else (storage.soc if storage.soc is not None else 0.5)
    return integrate_soc(
        requested_power_w,
        dt_s,
        energy_capacity_wh=storage.energy_capacity_wh,
        soc0=s0,
        soc_min=storage.soc_min,
        soc_max=storage.soc_max,
        efficiency_charge=storage.efficiency_charge,
        efficiency_discharge=storage.efficiency_discharge,
        p_rated_w=storage.p_rated_w,
        dtype=dtype,
        device=device,
    )


def storage_operating_point(power_by_id: dict, q_by_id: Optional[dict] = None) -> dict:
    """Build a solver ``operating_point`` for one timestep from realized storage powers.

    ``power_by_id`` maps a :class:`Storage` id to its realized active power for the step
    (a python float or a ``[*batch]`` tensor, signed, ``> 0`` = discharge/inject — the
    same convention the assembler consumes via ``Storage.p_nom_w``). ``q_by_id``
    optionally supplies the reactive setpoint per id (default 0). The result merges into
    any other ``operating_point`` passed to :func:`~pgml.solver.solve_power_flow` /
    :func:`~pgml.solver.solve_harmonic_flow`.
    """
    q_by_id = q_by_id or {}
    return {
        sid: {"p_w": p, "q_var": q_by_id.get(sid, 0.0)}
        for sid, p in power_by_id.items()
    }


__all__ = [
    "StorageDispatchResult",
    "integrate_soc",
    "dispatch_storage",
    "storage_operating_point",
]
