"""Reading and evaluating a device's stored harmonic spectrum.

One reader for every consumer of ``Load``/``Generator``/``Storage`` spectra (the harmonic
solve, the scenario sampler, the numpy oracles), so they agree on what a device injects.

- :class:`~pgml.schemas.StaticSpectrum`: the stored components.
- :class:`~pgml.schemas.LoadVaryingSpectrum`: evaluated at the device's loading, its
  solved fundamental active power over its rating (:func:`spectrum_rating`). Each order
  is interpolated as the complex phasor ``m * exp(j*phi)`` between the stored levels
  (:func:`evaluate_load_varying`), batched and differentiable in the loading.
- :class:`~pgml.schemas.TimeVaryingSpectrum` needs a time index the solve does not take;
  it raises :class:`~pgml.errors.ModelingError`.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Union

import torch
from torch import Tensor

from pgml.errors import InputError, ModelingError
from pgml.schemas.grid_schema import (
    ExtrapolationMethod,
    InterpolationMethod,
    LoadVaryingSpectrum,
    StaticSpectrum,
)

_log = logging.getLogger("pgml")

SpectrumDict = dict[int, tuple[Any, Any]]
"""``{order: (magnitude_pu, phase_deg)}`` with the schema's values (floats or tensors)."""

StoredSpectrum = Union[SpectrumDict, LoadVaryingSpectrum]
"""What one element stores: a static spectrum dict, or a spectrum to evaluate."""


def _unsupported(spectrum, device_id) -> ModelingError:
    kind = getattr(spectrum, "kind", type(spectrum).__name__)
    return ModelingError(
        f"Device {device_id}: spectrum kind {kind!r} is not evaluated by the harmonic "
        "solve (only 'static' and 'load_varying' are). Store one of those, or evaluate "
        "the spectrum yourself and pass it as `harmonic_injection`."
    )


def static_spectrum_dict(spectrum, device_id) -> Optional[SpectrumDict]:
    """``{order: (magnitude_pu, phase_deg)}`` of a static spectrum, or ``None`` if unset.

    Raises
    ------
    ModelingError
        For any other kind (a load-varying spectrum has no value before the solve).
    """
    if spectrum is None:
        return None
    if isinstance(spectrum, StaticSpectrum):
        return {
            c.order: (c.magnitude_pu, c.phase_deg) for c in spectrum.spectrum.components
        }
    if isinstance(spectrum, LoadVaryingSpectrum):
        raise ModelingError(
            f"Device {device_id}: a 'load_varying' spectrum depends on the solved "
            "operating point and has no single stored value here."
        )
    raise _unsupported(spectrum, device_id)


def _stored(spectrum, device_id) -> Optional[StoredSpectrum]:
    if isinstance(spectrum, LoadVaryingSpectrum):
        return spectrum
    if spectrum is not None and not isinstance(spectrum, StaticSpectrum):
        raise _unsupported(spectrum, device_id)
    return static_spectrum_dict(spectrum, device_id)


def element_stored_spectra(appliance) -> Optional[list[Optional[StoredSpectrum]]]:
    """What each element ``k`` of a device (``phases[k]``) stores, or ``None``.

    ``spectrum_per_phase`` gives element ``k`` the spectrum stored at ``phases[k]`` (a
    phase without an entry injects nothing: ``None``); ``spectrum`` gives every element
    the same one. Returns ``None`` when the device stores neither. Static spectra come
    back as dicts, load-varying ones as the schema object (see
    :func:`evaluate_load_varying`); any other kind raises ``ModelingError``.
    """
    phases = appliance.phases
    spp = getattr(appliance, "spectrum_per_phase", None)
    if spp is not None:
        return [_stored(spp.get(ph), appliance.id) for ph in phases]
    shared = _stored(getattr(appliance, "spectrum", None), appliance.id)
    if shared is None:
        return None
    out: list[Optional[StoredSpectrum]] = [shared] * len(phases)
    return out


def spectrum_rating(appliance) -> tuple[Any, str]:
    """Rated power a load-varying spectrum's levels refer to, and where it came from.

    The inverter rating ``control.s_rated_va`` when the device has one, else
    ``Storage.p_rated_w``, else ``|p_nom_w|`` (the nameplate of a load; for a
    generator the available power of the scenario, which is why it is the last resort
    and logs a warning for a generator or storage). Returns ``(rating, source)``.
    """
    control = getattr(appliance, "control", None)
    s_rated = getattr(control, "s_rated_va", None) if control is not None else None
    if s_rated is not None:
        return s_rated, "control.s_rated_va"
    p_rated = getattr(appliance, "p_rated_w", None)
    if p_rated is not None:
        return p_rated, "p_rated_w"
    if type(appliance).__name__ != "Load":
        _log.warning(
            "pgml: %s %s has a 'load_varying' spectrum but no rating "
            "(control.s_rated_va / p_rated_w); its loading is taken relative to "
            "|p_nom_w|, the available power of the scenario.",
            type(appliance).__name__,
            appliance.id,
        )
    p_nom = appliance.p_nom_w
    return (p_nom.abs() if isinstance(p_nom, Tensor) else abs(p_nom)), "p_nom_w"


def _level_phasors(spec: LoadVaryingSpectrum, rdt, device) -> dict[int, Tensor]:
    """``{order: complex [L]}`` phasors ``m * exp(j*phi)`` at the stored levels."""
    orders = sorted({c.order for point in spec.spectra for c in point.components})
    cdt = torch.complex128 if rdt == torch.float64 else torch.complex64
    out: dict[int, Tensor] = {}
    for order in orders:
        values = []
        for point in spec.spectra:
            comp = next((c for c in point.components if c.order == order), None)
            if comp is None:
                values.append(0.0 + 0.0j)
            else:
                values.append(
                    complex(
                        float(comp.magnitude_pu)
                        * math.cos(math.radians(float(comp.phase_deg))),
                        float(comp.magnitude_pu)
                        * math.sin(math.radians(float(comp.phase_deg))),
                    )
                )
        out[order] = torch.tensor(values, dtype=cdt, device=device)
    return out


def evaluate_load_varying(
    spec: LoadVaryingSpectrum, loading: Tensor, device_id
) -> dict[int, tuple[Tensor, Tensor]]:
    """``{order: (magnitude_pu, phase_deg)}`` of a load-varying spectrum at ``loading``.

    ``loading`` is a real tensor of any (batch) shape, in per unit of the rating; the
    returned tensors have its shape. Linear interpolation acts on the complex phasor, so
    magnitude and phase come from one smooth path; ``nearest`` picks the closest level.
    Outside the levels ``extrapolation`` holds the end value (``constant``), continues
    the end segment (``extend``) or raises (``error``). Differentiable in ``loading``
    (except at the level breakpoints and, for the phase, where the phasor is zero; the
    phase gradient is set to zero there).
    """
    rdt = loading.dtype
    levels = torch.tensor(spec.loading_levels_pu, dtype=rdt, device=loading.device)
    lo, hi = levels[0], levels[-1]
    if spec.extrapolation == ExtrapolationMethod.ERROR:
        outside = (loading < lo) | (loading > hi)
        if bool(outside.any()):
            raise InputError(
                f"Device {device_id}: loading {loading.detach().min().item():.3g}.."
                f"{loading.detach().max().item():.3g} pu leaves the load-varying "
                f"spectrum's levels [{float(lo):.3g}, {float(hi):.3g}] "
                "(extrapolation='error')."
            )
    x = loading
    if spec.extrapolation == ExtrapolationMethod.CONSTANT:
        x = torch.minimum(torch.maximum(loading, lo), hi)

    phasors = _level_phasors(spec, rdt, loading.device)
    n_levels = levels.shape[0]
    if spec.interpolation == InterpolationMethod.NEAREST:
        idx = torch.argmin((x.detach().unsqueeze(-1) - levels).abs(), dim=-1)
        values = {o: c[idx] for o, c in phasors.items()}
    elif spec.interpolation == InterpolationMethod.LINEAR:
        # Segment [i-1, i] holding x; the end segments carry 'extend' beyond the levels.
        i = torch.searchsorted(levels, x.detach().contiguous()).clamp(1, n_levels - 1)
        x0, x1 = levels[i - 1], levels[i]
        w = ((x - x0) / (x1 - x0)).to(next(iter(phasors.values())).dtype)
        values = {o: c[i - 1] * (1.0 - w) + c[i] * w for o, c in phasors.items()}
    else:
        raise ModelingError(
            f"Device {device_id}: load-varying interpolation "
            f"{spec.interpolation.value!r} is not supported (use 'linear' or 'nearest')."
        )

    out: dict[int, tuple[Tensor, Tensor]] = {}
    for order, c in values.items():
        zero = c.abs() < 1e-300
        safe = torch.where(zero, torch.ones_like(c), c)
        mag = torch.where(zero, torch.zeros_like(c.real), safe.abs())
        ang = torch.where(
            zero, torch.zeros_like(c.real), torch.angle(safe) * (180.0 / math.pi)
        )
        out[order] = (mag, ang)
    return out
