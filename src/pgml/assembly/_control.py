"""Differentiable inverter / DER control laws for the operating-point injection.

An inverter control law maps the local terminal-voltage magnitude (per unit of the
element nominal) and the available active power to the active and reactive power the
inverter actually injects, bounded by the apparent-power capability circle. It is
evaluated inside :func:`pgml.assembly.ybus.device_current_injections`, so the voltage
dependence enters the nonlinear power-flow residual ``I_device(V)`` and is differentiated
by the same implicit-function-theorem backward as a const-P/ZIP load — no separate
adjoint. See ``references/der_pv_storage_modeling.md`` sections 4.2-4.3.

All quantities are in the appliance's NATIVE authoring convention (positive active power
= the device's nominal direction, e.g. generation for a :class:`Generator`; positive
reactive power = overexcited / injecting). The caller applies the
consume/inject ``sign`` afterwards, exactly as for the plain ZIP injection.

Non-smooth pieces (the capability clamp, curve breakpoints) are made C\\ :sup:`1` for
gradient-based use by a soft saturation of half-width ``smoothing`` (``0`` recovers the
exact hard clamp / piecewise-linear curve, which matches OpenDSS ``InvControl`` /
pandapower ``CharacteristicControl`` at the operating point). Everything is vectorized
(no Python loop over elements), honours the input device/dtype, and is GPU-ready.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from pgml.errors import ModelingError
from pgml.schemas.grid_schema import (
    Characteristic,
    ConstantPowerFactorControl,
    ConstantReactivePowerControl,
    ExtrapolationMethod,
    InterpolationMethod,
    PowerFactorWattControl,
    QReference,
    VoltVarControl,
    VoltVarVoltWattControl,
    VoltWattControl,
)


def _as_rt(x, rdt: torch.dtype, device) -> Tensor:
    """Real tensor view of a python scalar/sequence or an array-like (tensor) leaf."""
    if isinstance(x, Tensor):
        return x.to(dtype=rdt, device=device)
    return torch.as_tensor(x, dtype=rdt, device=device)


def _softplus(x: Tensor, beta: float) -> Tensor:
    """Numerically stable ``log(1 + exp(beta*x)) / beta`` (-> ``relu`` as beta->inf)."""
    return torch.nn.functional.softplus(x, beta=beta)


def smooth_min(x: Tensor, ceiling: Tensor, beta: float) -> Tensor:
    """Smooth ``min(x, ceiling)``; the transition half-width is ``1/beta``.

    ``smooth_min(x, c) = c - softplus(c - x)``. As ``beta -> inf`` it converges to the
    exact ``min``; finite ``beta`` keeps the map C\\ :sup:`1` (well-defined gradient at
    the corner). ``beta = inf`` (i.e. ``smoothing == 0`` upstream) uses the exact min.
    """
    if not math.isfinite(beta):
        return torch.minimum(x, ceiling)
    return ceiling - _softplus(ceiling - x, beta)


def smooth_max(x: Tensor, floor: Tensor, beta: float) -> Tensor:
    """Smooth ``max(x, floor)`` (mirror of :func:`smooth_min`)."""
    if not math.isfinite(beta):
        return torch.maximum(x, floor)
    return floor + _softplus(x - floor, beta)


def smooth_clamp(x: Tensor, lo: Tensor, hi: Tensor, beta: float) -> Tensor:
    """Smooth clamp of ``x`` to ``[lo, hi]`` (soft saturation; beta = inf -> hard)."""
    return smooth_min(smooth_max(x, lo, beta), hi, beta)


def _beta_from_smoothing(smoothing: float) -> float:
    """Map the schema ``smoothing`` half-width to the softplus ``beta`` (inf if 0)."""
    if smoothing is None or smoothing <= 0.0:
        return math.inf
    return 1.0 / float(smoothing)


def evaluate_characteristic(
    x: Tensor, char: Characteristic, rdt: torch.dtype, device
) -> Tensor:
    """Evaluate ``y = f(x)`` of a :class:`Characteristic`, differentiable in x and curve.

    ``linear`` (default) reproduces the OpenDSS XYcurve / pandapower ``Characteristic``
    piecewise-linear lookup; ``cubic`` is a smooth Catmull-Rom/Hermite interpolant for a
    C\\ :sup:`1` curve. Extrapolation is ``constant`` (hold the endpoint, the default),
    ``extend`` (continue the end segment's slope), or ``error`` (treated as ``constant``
    here — a tensor bound cannot branch on the differentiable path). ``nearest`` /
    ``log_log`` are not meaningful for a control curve and raise.

    Gradients flow to the query ``x`` AND to the curve ``x_values`` / ``y_values`` (which
    may be tensor leaves), so a curve slope is recoverable through the solve.
    """
    xp = _as_rt(char.x_values, rdt, device)  # [K]
    fp = _as_rt(char.y_values, rdt, device)  # [K]
    k = xp.shape[0]
    if char.interpolation not in (
        InterpolationMethod.LINEAR,
        InterpolationMethod.CUBIC,
    ):
        raise ModelingError(
            f"control characteristic interpolation {char.interpolation.value!r} is not "
            "supported (use 'linear' or 'cubic')."
        )

    # Interval index: idx in [1, k-1] so [idx-1, idx] is a valid segment everywhere.
    idx = torch.searchsorted(xp, x.detach(), right=True).clamp(1, k - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    dx = x1 - x0
    t = (x - x0) / dx  # local coordinate (can leave [0,1] under extrapolation)

    if char.interpolation == InterpolationMethod.CUBIC and k >= 2:
        # Hermite with Catmull-Rom interior tangents (one-sided at the ends), scaled to
        # the local segment. Smooth (C^1) and differentiable in the curve values.
        ii = idx
        xm = xp[(ii - 2).clamp(min=0)]
        xpp = xp[(ii + 1).clamp(max=k - 1)]
        ym = fp[(ii - 2).clamp(min=0)]
        ypp = fp[(ii + 1).clamp(max=k - 1)]
        # Secant-based tangents (guard the one-sided ends where the neighbour coincides).
        left_dx = x0 - xm
        right_dx = xpp - x1
        m0 = torch.where(
            left_dx.abs() > 0, (y1 - ym) / (x1 - xm).clamp_min(1e-30), (y1 - y0) / dx
        )
        m1 = torch.where(
            right_dx.abs() > 0, (ypp - y0) / (xpp - x0).clamp_min(1e-30), (y1 - y0) / dx
        )
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        y = h00 * y0 + h10 * dx * m0 + h01 * y1 + h11 * dx * m1
    else:
        y = y0 + t * (y1 - y0)  # linear

    if char.extrapolation == ExtrapolationMethod.EXTEND:
        return y  # the clamped-index formula already extends the end segments
    # CONSTANT (and ERROR, which cannot branch on tensor values here): hold endpoints.
    y = torch.where(x <= xp[0], fp[0], y)
    y = torch.where(x >= xp[-1], fp[-1], y)
    return y


def resolve_injection_power(
    control,
    p_avail: Tensor,
    v_pu: Tensor,
    *,
    rdt: torch.dtype,
    device,
) -> tuple[Tensor, Tensor]:
    """Active / reactive power of a controlled inverter, in the native convention.

    Parameters
    ----------
    control:
        An :data:`pgml.schemas.grid_schema.InverterControl` instance.
    p_avail:
        Available active power per element ``[*batch, n_elem]`` (the device's MPP /
        setpoint magnitude, native sign).
    v_pu:
        Terminal-voltage magnitude per element ``[*batch, H, n_elem]`` in per unit of
        the element nominal (``|V_term| / V0``).
    rdt, device:
        Real dtype / target device.

    Returns
    -------
    (p_eff, q_eff):
        Each ``[*batch, H, n_elem]`` real, in the appliance's native convention
        (positive = nominal direction / overexcited), bounded by the capability circle.
    """
    p_avail = p_avail.to(dtype=rdt, device=device)
    v_pu = v_pu.to(dtype=rdt, device=device)
    p = p_avail.unsqueeze(-2)  # [*batch, 1, n_elem] -> broadcast over H
    p = torch.broadcast_to(p, v_pu.shape)
    beta = _beta_from_smoothing(getattr(control, "smoothing", 0.0))
    s_rated = control.s_rated_va

    # --- active power: Volt-Watt curtails it; others pass it through ----------
    if isinstance(control, VoltWattControl):
        frac = evaluate_characteristic(v_pu, control.characteristic, rdt, device)
        p_eff = p * frac
    elif isinstance(control, VoltVarVoltWattControl):
        frac = evaluate_characteristic(v_pu, control.volt_watt, rdt, device)
        p_eff = p * frac
    else:
        p_eff = p

    # --- reactive power per mode ---------------------------------------------
    if isinstance(control, ConstantReactivePowerControl):
        q_eff = torch.full_like(p_eff, 0.0) + _as_rt(control.q_var, rdt, device)
    elif isinstance(control, ConstantPowerFactorControl):
        pf = float(control.power_factor)
        tan_phi = math.sqrt(max(1.0 - pf * pf, 0.0)) / pf
        s = 1.0 if control.overexcited else -1.0
        q_eff = s * p_eff.abs() * tan_phi
    elif isinstance(control, PowerFactorWattControl):
        p_ref = control.p_ref_w
        p_ref_t = (
            _as_rt(p_ref, rdt, device) if p_ref is not None else p_avail.abs().amax()
        )
        p_ref_t = torch.clamp(p_ref_t, min=1e-30)
        x = p_eff.abs() / p_ref_t
        pf_signed = evaluate_characteristic(x, control.characteristic, rdt, device)
        pf_mag = torch.clamp(pf_signed.abs(), min=1e-6, max=1.0)
        tan_phi = torch.sqrt(torch.clamp(1.0 - pf_mag * pf_mag, min=0.0)) / pf_mag
        q_eff = torch.sign(pf_signed) * p_eff.abs() * tan_phi
    elif isinstance(control, (VoltVarControl, VoltVarVoltWattControl)):
        curve = (
            control.characteristic
            if isinstance(control, VoltVarControl)
            else control.volt_var
        )
        q_pu = evaluate_characteristic(v_pu, curve, rdt, device)
        q_ref = getattr(control, "q_reference", QReference.RATED)
        if q_ref == QReference.RATED:
            # Validated to require s_rated; this branch is only reached when set.
            q_base = _as_rt(s_rated, rdt, device)
        else:  # AVAILABLE: vars left under the capability circle at the present P.
            s_t = _as_rt(s_rated, rdt, device) if s_rated is not None else None
            if s_t is None:
                q_base = p_eff.abs()  # no rating -> reference the active power
            else:
                q_base = torch.sqrt(torch.clamp(s_t * s_t - p_eff * p_eff, min=0.0))
        q_eff = q_pu * q_base
    else:  # VoltWattControl: pure active curtailment, no reactive control.
        q_eff = torch.zeros_like(p_eff)

    # --- capability clamp: bound |Q| to the apparent-power circle -------------
    if s_rated is not None:
        s_t = _as_rt(s_rated, rdt, device)
        q_max = torch.sqrt(torch.clamp(s_t * s_t - p_eff * p_eff, min=0.0))
        q_eff = smooth_clamp(q_eff, -q_max, q_max, beta)

    return p_eff, q_eff


__all__ = [
    "evaluate_characteristic",
    "smooth_clamp",
    "smooth_min",
    "smooth_max",
    "resolve_injection_power",
]
