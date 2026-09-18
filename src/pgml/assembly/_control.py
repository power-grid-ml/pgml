"""Differentiable inverter / DER control laws for the operating-point injection.

An inverter control law maps the local terminal-voltage magnitude (per unit of the
element nominal) and the available active power to the active and reactive power the
inverter actually injects, bounded by the apparent-power capability circle. It is
evaluated inside :func:`pgml.assembly.ybus.device_current_injections`, so the voltage
dependence enters the nonlinear power-flow residual ``I_device(V)`` and is differentiated
by the same implicit-function-theorem backward as a const-P/ZIP load — no separate
adjoint. See ``docs/pgml/modeling/der-pv-storage.md`` sections 4.2-4.3.

All quantities are in the appliance's NATIVE authoring convention (positive active power
= the device's nominal direction, e.g. generation for a :class:`Generator`; positive
reactive power = overexcited / injecting). The caller applies the
consume/inject ``sign`` afterwards, exactly as for the plain ZIP injection.

Ratings are device totals. The power quantities of a control block (``s_rated_va``,
``q_var``, ``p_ref_w``) describe the WHOLE device, like ``p_nom_w``. The laws are
evaluated per connection element: one per connected phase for a WYE device, one per
phase pair for a (three-phase) DELTA device, so ``n_elem`` equals the number of phases
the device connects. Every element works with an equal ``1 / n_elem`` share of each
device-level quantity: its rating is ``s_rated_va / n_elem``, its fixed
reactive power ``q_var / n_elem``, its ``cosphi(P)`` base ``p_ref_w / n_elem``. The
bases that derive from the rating follow (the rated Volt-VAr base, the capability
circle, the width of the soft clamp). A balanced three-phase device therefore injects
the same totals as its single-phase positive-sequence equivalent, and a single-phase
device (``n_elem = 1``) sees the values as written. The split is equal whatever the
per-phase distribution of the active power; an unbalanced device does not shift
rating between its elements.

Each element evaluates its voltage-dependent curve at ITS OWN terminal voltage
``|V_term| / V0`` (phase-to-neutral or phase-to-ground for WYE, phase-to-phase for
DELTA), not at a positive-sequence or phase-average magnitude. On a balanced network
the two coincide; on an unbalanced one the elements of one device respond
individually, where OpenDSS ``InvControl`` and pandapower apply one device-level
response.

The capability clamp is made C\\ :sup:`1` for gradient-based use by a soft saturation
whose half-width is ``smoothing`` times the rating ``s_rated_va`` (``smoothing`` is a
fraction of the rating; ``0`` recovers the exact hard clamp, which matches OpenDSS
``InvControl`` / pandapower ``CharacteristicControl`` at the operating point). The soft
clamp is ONE function, used by the forward solve and by its gradient alike, so a
positive ``smoothing`` also moves the solved operating point near the limit, by about
``0.7 * smoothing * s_rated_va`` at the corner (for the device as a whole; each
element carries its share) and exponentially less away from it.
Curve breakpoints are not blended: a ``linear`` characteristic is exactly piecewise
linear (one-sided gradients at a breakpoint), and ``cubic`` interpolation is the
C\\ :sup:`1` alternative. Everything is vectorized (no Python loop over elements),
honours the input device/dtype, and is GPU-ready.
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


def _element_share(x, n_elem: int, rdt: torch.dtype, device) -> Tensor:
    """Equal share of a device-level power quantity carried by one of ``n_elem`` elements.

    ``n_elem`` is a tensor SHAPE (the number of connection elements), not a tensor
    value, so the division stays on the differentiable path of ``x``.
    """
    return _as_rt(x, rdt, device) / n_elem


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
    """Map the schema ``smoothing`` half-width to the softplus ``beta`` (inf if 0).

    ``smoothing`` is a fraction of the inverter rating, so the resulting ``beta``
    applies to quantities expressed in per unit of that rating
    (:func:`_clamp_to_rating`).
    """
    if smoothing is None or smoothing <= 0.0:
        return math.inf
    return 1.0 / float(smoothing)


def _clamp_to_rating(x: Tensor, limit: Tensor, s_rated: Tensor, beta: float) -> Tensor:
    """Soft clamp of a power ``x`` to ``[-limit, limit]``, all in W / var / VA.

    The saturation is evaluated in per unit of the rating ``s_rated``, which is what
    gives the transition the declared half-width of ``smoothing * s_rated``. The hard
    clamp (``beta = inf``) needs no scaling.
    """
    if not math.isfinite(beta):
        return smooth_clamp(x, -limit, limit, beta)
    lim_pu = limit / s_rated
    return s_rated * smooth_clamp(x / s_rated, -lim_pu, lim_pu, beta)


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
    # The .detach() feeds only the integer segment SELECTION (searchsorted has no
    # gradient); the interpolation below re-reads the tracked x/xp/fp, so no
    # gradient path is severed — this is not on the differentiable value path.
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
        An :data:`pgml.schemas.grid_schema.InverterControl` instance. Its power
        quantities (``s_rated_va``, ``q_var``, ``p_ref_w``) are DEVICE totals; each of
        the ``n_elem`` elements works with a ``1 / n_elem`` share (module docstring).
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
        (positive = nominal direction / overexcited), per element, each bounded by
        its share of the capability circle. The device totals are the sums over the
        last axis.
    """
    p_avail = p_avail.to(dtype=rdt, device=device)
    v_pu = v_pu.to(dtype=rdt, device=device)
    p = p_avail.unsqueeze(-2)  # [*batch, 1, n_elem] -> broadcast over H
    p = torch.broadcast_to(p, v_pu.shape)
    beta = _beta_from_smoothing(getattr(control, "smoothing", 0.0))
    # Device-level ratings -> the equal share one element works with. ``s_t`` is the
    # element's capability circle, its rated Volt-VAr base and (through
    # ``_clamp_to_rating``) the base of the soft-clamp width.
    n_elem = v_pu.shape[-1]
    s_rated = control.s_rated_va
    s_t = _element_share(s_rated, n_elem, rdt, device) if s_rated is not None else None

    # --- active power: Volt-Watt curtails it; others pass it through ----------
    if isinstance(control, VoltWattControl):
        frac = evaluate_characteristic(v_pu, control.characteristic, rdt, device)
        p_eff = p * frac
    elif isinstance(control, VoltVarVoltWattControl):
        frac = evaluate_characteristic(v_pu, control.volt_watt, rdt, device)
        p_eff = p * frac
    else:
        p_eff = p

    # --- capability clamp, watt priority: P itself cannot exceed the rating ---
    # An oversized source (available P above the inverter VA rating) is clipped
    # to the circle before any reactive-power law sees it, so P² + Q² <= S²
    # holds for the pair actually injected (OpenDSS PVSystem kVA semantics).
    if s_t is not None:
        p_eff = _clamp_to_rating(p_eff, s_t, s_t, beta)

    # --- reactive power per mode ---------------------------------------------
    if isinstance(control, ConstantReactivePowerControl):
        q_eff = torch.zeros_like(p_eff) + _element_share(
            control.q_var, n_elem, rdt, device
        )
    elif isinstance(control, ConstantPowerFactorControl):
        pf = float(control.power_factor)
        tan_phi = math.sqrt(max(1.0 - pf * pf, 0.0)) / pf
        s = 1.0 if control.overexcited else -1.0
        q_eff = s * p_eff.abs() * tan_phi
    elif isinstance(control, PowerFactorWattControl):
        p_ref = control.p_ref_w
        # No reference given: the largest element share of the available power,
        # which is |p_nom_w| / n_elem for an equally split device.
        p_ref_t = (
            _element_share(p_ref, n_elem, rdt, device)
            if p_ref is not None
            else p_avail.abs().amax()
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
            q_base = s_t
        else:  # AVAILABLE: vars left under the capability circle at the present P.
            if s_t is None:
                q_base = p_eff.abs()  # no rating -> reference the active power
            else:
                q_base = torch.sqrt(torch.clamp(s_t * s_t - p_eff * p_eff, min=0.0))
        q_eff = q_pu * q_base
    else:  # VoltWattControl: pure active curtailment, no reactive control.
        q_eff = torch.zeros_like(p_eff)

    # --- capability clamp: bound |Q| to the apparent-power circle -------------
    if s_t is not None:
        q_max = torch.sqrt(torch.clamp(s_t * s_t - p_eff * p_eff, min=0.0))
        q_eff = _clamp_to_rating(q_eff, q_max, s_t, beta)

    return p_eff, q_eff


__all__ = [
    "evaluate_characteristic",
    "smooth_clamp",
    "smooth_min",
    "smooth_max",
    "resolve_injection_power",
]
