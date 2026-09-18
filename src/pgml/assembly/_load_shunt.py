"""Harmonic Norton shunt of an injection appliance (internal).

At orders ``h > 1`` a Load / Generator / Storage is a harmonic current source (its
``spectrum``) in PARALLEL with a shunt admittance derived from the device's
fundamental operating point. This module owns the model selection and the per-ELEMENT
admittance math; the nodal stamp (incidence, scatter) lives with the harmonic assembly
in :mod:`pgml.solver.harmonic_flow`, which is where the fundamental solution the
operating point comes from is available.

Model (the series / parallel split of OpenDSS ``Load.pas``
``TLoadObj.CalcYPrimMatrix``), per element, with ``s = series_rl_fraction`` and
``Y_eq = conj(S)/V_rated**2 = G + jB``::

    Y_par(h) = (1 - s)*G + j*(1 - s)*B_h                        parallel branch
    Z_ser    = 1/(s*Y_eq) = R + jX,   Z_ser(h) = R + j*X_h       series branch
    Y(h)     = Y_par(h) + 1/Z_ser(h)

``V_rated`` is the element's RATED voltage (line-to-neutral for WYE, line-to-line for
DELTA) and ``S`` the power the device draws at the converged fundamental solution.

How the reactive part scales with the order is the documented default
``appliance.harmonic_shunt.reactive_element``. ``sign_aware`` (shipped) scales it as
the element it is at the operating point: a lagging device (``Q > 0``, ``B < 0``,
``X > 0``) is an inductance, ``B_h = B/h`` and ``X_h = h*X``; a leading device
(``Q < 0``, ``B > 0``, ``X < 0``) is a capacitance, ``B_h = h*B`` and ``X_h = X/h``.
That is the same law the const-Z fold of :func:`pgml.assembly.assemble_ybus` applies,
so both load representations agree for either sign of ``Q``. ``inductive`` applies
``B/h`` and ``h*X`` whatever the sign, which is what OpenDSS computes (verified on a
live ``CktElement.YPrim`` of a load with negative kvar); for a leading device that is
a negative inductance, an element whose susceptance has the sign of a capacitor and
the frequency slope of an inductor. The ``opendss`` preset selects it. Both laws are
identical for ``Q >= 0`` and exact at ``h = 1``.

The ``motor`` model keeps the parallel branch and replaces the derived series impedance
by a fixed blocked-rotor reactance (OpenDSS ``puXharm`` / ``XRharm``)::

    X        = V_rated**2/(S_base*s) * x_pu,  Z_ser(h) = X/xr + j*h*X

``S_base`` is the apparent power of the OpenDSS element the device maps to: the
per-element power for a WYE device (one single-phase element per phase) and the device
total for a DELTA device (one multi-phase element).

A GENERATION-sign device (Generator / Storage) carries NO shunt by default: ``S`` is
negative for it, so ``Y_eq`` has a negative conductance and the expression would make an
inverter feed harmonic energy into the network instead of damping it. The policy is the
documented default ``appliance.harmonic_shunt.generation_model``
(:func:`generation_shunt_is_neglected`), whose ``load_style`` value applies the load
expression anyway — what OpenDSS computes for the negative-kW ``Load`` idiom, sign
included. A real inverter's harmonic output impedance is its filter impedance and a
machine's its subtransient reactance (OpenDSS ``%R``/``%X``, ``Xdpp``). A device
states it through its ``harmonic_impedance`` block
(:class:`~pgml.schemas.grid_schema.HarmonicImpedance`), which the harmonic solve stamps
instead of a P,Q-derived shunt.

All math is autograd-safe torch: gradients flow to the operating-point power (hence to
the fundamental solution) and the degenerate cases (zero power, ``s = 0``) are handled
by masking a safe denominator, never by Python control flow on a tensor value. The
model parameters themselves are plain python floats (schema
:class:`~pgml.schemas.grid_schema.HarmonicShuntModel`), so masks built from them are
constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from pgml.defaults import get as _cfg
from pgml.errors import InputError

#: The selectable harmonic shunt models (``appliance.harmonic_shunt.model``).
HARMONIC_SHUNT_MODELS = ("none", "opendss", "motor")

#: What a generation-sign device carries (``appliance.harmonic_shunt.generation_model``).
GENERATION_SHUNT_MODELS = ("none", "load_style")

#: How the reactive part of the P,Q-derived shunt scales with the order
#: (``appliance.harmonic_shunt.reactive_element``).
REACTIVE_ELEMENT_LAWS = ("sign_aware", "inductive")

#: Which power / terminal voltage the shunt is built from
#: (``appliance.harmonic_shunt.basis``).
SHUNT_BASES = ("operating_point", "nameplate")


@dataclass(frozen=True)
class ResolvedHarmonicShunt:
    """One device's resolved harmonic shunt parameters.

    Attributes
    ----------
    kind:
        ``"none"`` (pure current source), ``"opendss"`` (the P,Q-derived series /
        parallel split) or ``"motor"`` (fixed blocked-rotor series reactance).
    series_rl_fraction:
        ``s`` — the fraction of the shunt carried by the series R-L branch.
    motor_x_harm_pu:
        Blocked-rotor reactance in per unit of the device's apparent power
        (``"motor"`` only, else ``0.0``).
    motor_xr_harm:
        X/R of that reactance at the fundamental (``"motor"`` only, else ``1.0``).
    """

    kind: str
    series_rl_fraction: float
    motor_x_harm_pu: float
    motor_xr_harm: float


def resolve_shunt_model_name(model: Optional[str]) -> str:
    """Validate a run-level shunt model name (``None`` -> the documented default).

    ``None`` resolves the modeling default ``appliance.harmonic_shunt.model``; any
    other value must be one of :data:`HARMONIC_SHUNT_MODELS` (an unknown name raises
    instead of silently selecting a model).
    """
    name = _cfg("appliance.harmonic_shunt.model") if model is None else model
    if name not in HARMONIC_SHUNT_MODELS:
        raise InputError(
            f"unknown harmonic load-shunt model {name!r}; use one of "
            f"{', '.join(repr(m) for m in HARMONIC_SHUNT_MODELS)}."
        )
    return str(name)


def resolve_shunt_basis(basis: Optional[str]) -> str:
    """Validate the shunt's power basis (``None`` -> the documented default).

    ``"operating_point"`` builds the shunt from the power the device draws in this
    scenario at the solved fundamental terminal voltage, which is what OpenDSS's own
    ``YPrim`` does with its Load's specified kW/kvar. ``"nameplate"`` builds it from the
    device's stored P, Q at its rated terminal voltage, which makes ``Y(h)`` independent
    of the scenario — one factorisation per order for a whole batch instead of one per
    scenario and order — at the price of a shunt that does not follow the loading.
    ``None`` resolves ``appliance.harmonic_shunt.basis``; an unknown name raises.
    """
    name = _cfg("appliance.harmonic_shunt.basis") if basis is None else basis
    if name not in SHUNT_BASES:
        raise InputError(
            f"unknown harmonic shunt basis {name!r}; use one of "
            f"{', '.join(repr(m) for m in SHUNT_BASES)}."
        )
    return str(name)


def resolve_reactive_element_law(law: Optional[str] = None) -> str:
    """Validate the reactive-element law (``None`` -> the documented default).

    ``"sign_aware"`` scales a leading device's reactive part as a capacitance and a
    lagging device's as an inductance; ``"inductive"`` treats both as an inductance
    (OpenDSS). ``None`` resolves ``appliance.harmonic_shunt.reactive_element``; an
    unknown name raises.
    """
    name = _cfg("appliance.harmonic_shunt.reactive_element") if law is None else law
    if name not in REACTIVE_ELEMENT_LAWS:
        raise InputError(
            f"unknown harmonic shunt reactive-element law {name!r}; use one of "
            f"{', '.join(repr(m) for m in REACTIVE_ELEMENT_LAWS)}."
        )
    return str(name)


def generation_shunt_is_neglected(appliance) -> bool:
    """Does the documented generation policy leave ``appliance`` a pure current source?

    ``True`` for an in-service GENERATION-sign device (Generator / Storage) while
    ``appliance.harmonic_shunt.generation_model`` is ``"none"`` (the shipped value) and
    the device's own ``harmonic_model`` does not name the ``motor`` model. The reason is
    in that default's documentation: the load expression ``conj(S)/V_rated**2`` has a
    negative conductance for a device that injects power, so applying it to an inverter
    would make it feed harmonic energy into the network.
    """
    from pgml.schemas.grid_schema import Load

    if isinstance(appliance, Load):
        return False
    override = getattr(appliance, "harmonic_model", None)
    if override is not None and override.motor_x_harm_pu is not None:
        return False
    return _generation_model() == "none"


def _generation_model() -> str:
    """The validated ``appliance.harmonic_shunt.generation_model`` default."""
    name = _cfg("appliance.harmonic_shunt.generation_model")
    if name not in GENERATION_SHUNT_MODELS:
        raise InputError(
            f"unknown harmonic generation-shunt model {name!r}; use one of "
            f"{', '.join(repr(m) for m in GENERATION_SHUNT_MODELS)}."
        )
    return str(name)


def resolve_harmonic_shunt(appliance, model: str) -> ResolvedHarmonicShunt:
    """Resolve one appliance's harmonic shunt: device override > run model > defaults.

    ``model`` is the already-validated run-level name
    (:func:`resolve_shunt_model_name`). ``model == "none"`` wins over every device
    (the run carries no device shunt at all, OpenDSS ``Set NeglectLoadY=Yes``).
    Otherwise the appliance's own ``harmonic_model``
    (:class:`~pgml.schemas.grid_schema.HarmonicShuntModel`) decides, falling back to
    the documented ``appliance.harmonic_shunt.*`` values for anything it leaves open:
    ``neglect_shunt=True`` -> ``"none"``, a set ``motor_x_harm_pu`` -> ``"motor"``,
    else the run-level model.

    A GENERATION-sign device (Generator / Storage) is additionally governed by
    ``appliance.harmonic_shunt.generation_model``, whose shipped value leaves it a pure
    current source (:func:`generation_shunt_is_neglected`): the load expression's
    conductance is negative for an injecting device, which would damp nothing and feed
    harmonic energy instead. A stored ``harmonic_model`` block does NOT override that —
    every grid written before the shunt was consumed carries the former default block on
    every device, so honouring it would silently re-introduce the negative conductance.
    Naming the ``motor`` model (``motor_x_harm_pu``) does, and so does setting the
    default to ``load_style``.
    """
    if model == "none":
        return ResolvedHarmonicShunt("none", 0.0, 0.0, 1.0)
    if generation_shunt_is_neglected(appliance):
        return ResolvedHarmonicShunt("none", 0.0, 0.0, 1.0)
    override = getattr(appliance, "harmonic_model", None)
    s = (
        float(override.series_rl_fraction)
        if override is not None
        else float(_cfg("appliance.harmonic_shunt.series_rl_fraction"))
    )
    if override is not None and override.neglect_shunt:
        return ResolvedHarmonicShunt("none", 0.0, 0.0, 1.0)
    x_pu = None if override is None else override.motor_x_harm_pu
    kind = "motor" if (model == "motor" or x_pu is not None) else "opendss"
    if kind != "motor":
        return ResolvedHarmonicShunt("opendss", s, 0.0, 1.0)
    if x_pu is None:
        x_pu = float(_cfg("appliance.harmonic_shunt.motor_x_harm_pu"))
    xr = (
        float(override.motor_xr_harm)
        if override is not None
        else float(_cfg("appliance.harmonic_shunt.motor_xr_harm"))
    )
    if float(x_pu) <= 0.0 or xr <= 0.0:
        raise InputError(
            f"appliance {appliance.id}: the motor harmonic shunt needs "
            f"motor_x_harm_pu > 0 and motor_xr_harm > 0 (got {x_pu!r}, {xr!r})."
        )
    if s == 0.0:
        # OpenDSS evaluates puXharm inside its `%SeriesRL <> 0` branch, so a motor
        # device with no series fraction keeps the parallel branch only.
        return ResolvedHarmonicShunt("motor", 0.0, float(x_pu), xr)
    return ResolvedHarmonicShunt("motor", s, float(x_pu), xr)


def harmonic_shunt_element_admittance(
    s_elem: Tensor,
    v_rated: Tensor,
    h: Tensor,
    series_rl: Tensor,
    *,
    motor_x_pu: Tensor,
    motor_xr: Tensor,
    motor_s_base: Tensor,
    cdtype: torch.dtype,
    reactive_element: Optional[str] = None,
) -> Tensor:
    """Per-element harmonic shunt admittance ``[*batch, H, K, E]`` (complex).

    Parameters
    ----------
    s_elem:
        Complex ``[*batch, K, E]`` per-element apparent power at the fundamental
        operating point, in the LOAD convention (``+`` drawn, so a generator's entry is
        negative). ``K`` devices of one incidence group, ``E`` elements each.
    v_rated:
        Real ``[K, 1]`` element rated voltage (line-to-neutral WYE / line-to-line
        DELTA), in volts.
    h:
        Real ``[H]`` harmonic orders (``> 1``).
    series_rl:
        Real ``[K, 1]`` series fraction ``s`` in ``[0, 1]``. ``s = 0`` drops the series
        branch entirely (the most damped model), ``s = 1`` drops the parallel branch.
    motor_x_pu:
        Real ``[K, 1]`` blocked-rotor reactance in per unit of ``motor_s_base``; ``0``
        selects the P,Q-derived series impedance for that device.
    motor_xr:
        Real ``[K, 1]`` X/R of ``motor_x_pu`` at the fundamental (``1`` where unused).
    motor_s_base:
        Real ``[*batch, K, E]`` (broadcastable) apparent-power base of the motor
        reactance: the mapped OpenDSS element's kVA — the element's own power for WYE,
        the device total for DELTA.
    cdtype:
        Complex dtype of the result.
    reactive_element:
        ``"sign_aware"`` or ``"inductive"`` (:func:`resolve_reactive_element_law`);
        ``None`` resolves ``appliance.harmonic_shunt.reactive_element``.

    Returns
    -------
    Tensor
        Complex ``[*batch, H, K, E]`` element admittance, ``0`` where a device has no
        power (``Y_eq = 0``) and exact at ``h = 1`` (``Y_par + Y_ser = Y_eq``) for the
        derived split.
    """
    rdt = torch.float64 if cdtype == torch.complex128 else torch.float32
    y_eq = torch.conj(s_elem) / (v_rated * v_rated)  # [*b, K, E]
    g = y_eq.real.unsqueeze(-3).to(rdt)  # [*b, 1, K, E]
    b = y_eq.imag.unsqueeze(-3).to(rdt)
    hb = h.to(rdt).reshape(-1, 1, 1)  # [H, 1, 1]
    s = series_rl.to(rdt)  # [K, 1]

    # Reactive part over the order. An inductance has B/h in the parallel branch and
    # h*X in the series one; a capacitance (B > 0, sign-aware law only) has h*B and X/h.
    # The sign split uses clamp, so it stays differentiable and exact at h = 1.
    if resolve_reactive_element_law(reactive_element) == "sign_aware":
        b_ind, b_cap = torch.clamp(b, max=0.0), torch.clamp(b, min=0.0)
    else:
        b_ind, b_cap = b, torch.zeros_like(b)
    b_par = b_ind / hb + b_cap * hb  # parallel susceptance over h
    b_ser = b_ind * hb + b_cap / hb  # -X_h*s*|Y_eq|**2 of the series branch

    # Parallel branch: the conductance keeps its full value at every order.
    y_par = _complex((1.0 - s) * g, (1.0 - s) * b_par)

    # Derived series branch: with Z_ser = 1/(s*Y_eq) = (G - jB)/(s*|Y_eq|**2),
    # Y_ser(h) = s*|Y_eq|**2/(G - j*B_ser(h)) — one division instead of three, and the
    # |Y_eq| = 0 case (a device at zero power) masks to 0 instead of dividing by 0.
    mag2 = g * g + b * b
    num = _complex(s * mag2, torch.zeros_like(mag2))
    den = _complex(g + 0.0 * hb, -b_ser)
    zero = mag2 == 0.0
    safe_den = torch.where(zero, torch.ones_like(den), den)
    y_ser_derived = torch.where(zero, torch.zeros_like(den), num / safe_den)

    # Motor series branch: Z_ser(h) = X/xr + j*h*X with X = V**2/(S*s)*x_pu. Carried as
    # the conductance 1/X = s*S/(V**2*x_pu) so a zero-power device gives 0, not inf.
    x_pu = motor_x_pu.to(rdt)
    safe_x_pu = torch.where(x_pu == 0.0, torch.ones_like(x_pu), x_pu)
    inv_x = (s * motor_s_base.to(rdt).unsqueeze(-3)) / (
        v_rated * v_rated * safe_x_pu
    )  # [*b, 1, K, E]
    y_ser_motor = _complex(inv_x, torch.zeros_like(inv_x)) / _complex(
        (1.0 / motor_xr.to(rdt)) + 0.0 * hb, hb + 0.0 * motor_xr.to(rdt)
    )

    is_motor = (x_pu > 0.0).unsqueeze(-3)  # [1, K, 1] constant
    y_ser = torch.where(is_motor, y_ser_motor, y_ser_derived)
    has_series = (s > 0.0).unsqueeze(-3)  # [1, K, 1] constant
    y = y_par + torch.where(has_series, y_ser, torch.zeros_like(y_ser))
    return y.to(cdtype)


def _complex(re: Tensor, im: Tensor) -> Tensor:
    """``torch.complex`` on two real tensors of DIFFERENT broadcastable shapes."""
    re_b, im_b = torch.broadcast_tensors(re, im)
    return torch.complex(re_b.contiguous(), im_b.contiguous())


__all__ = [
    "GENERATION_SHUNT_MODELS",
    "HARMONIC_SHUNT_MODELS",
    "REACTIVE_ELEMENT_LAWS",
    "SHUNT_BASES",
    "ResolvedHarmonicShunt",
    "generation_shunt_is_neglected",
    "resolve_shunt_model_name",
    "resolve_shunt_basis",
    "resolve_reactive_element_law",
    "resolve_harmonic_shunt",
    "harmonic_shunt_element_admittance",
]
