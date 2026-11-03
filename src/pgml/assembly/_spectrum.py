"""Reading a device's stored harmonic spectrum.

One reader for every consumer of ``Load``/``Generator``/... spectra (the harmonic solve,
the scenario sampler, the numpy oracles), so they agree on what a device injects.

Only :class:`~pgml.schemas.StaticSpectrum` has evaluation semantics today. The other
schema kinds (``load_varying``, ``time_varying``, ``random``, ``distribution``) need an
operating point, a time index, a random draw or an external fitted distribution that the
solve does not have; they raise :class:`~pgml.errors.ModelingError` instead of injecting
nothing.
"""

from __future__ import annotations

from typing import Any, Optional

from pgml.errors import ModelingError
from pgml.schemas.grid_schema import StaticSpectrum

SpectrumDict = dict[int, tuple[Any, Any]]
"""``{order: (magnitude_pu, phase_deg)}`` with the schema's values (floats or tensors)."""


def static_spectrum_dict(spectrum, device_id) -> Optional[SpectrumDict]:
    """``{order: (magnitude_pu, phase_deg)}`` of a stored spectrum, or ``None`` if unset.

    Raises
    ------
    ModelingError
        For a spectrum kind other than ``static``; the message names the device and the
        kind and how to inject the emission instead.
    """
    if spectrum is None:
        return None
    if isinstance(spectrum, StaticSpectrum):
        return {
            c.order: (c.magnitude_pu, c.phase_deg) for c in spectrum.spectrum.components
        }
    kind = getattr(spectrum, "kind", type(spectrum).__name__)
    raise ModelingError(
        f"Device {device_id}: spectrum kind {kind!r} is not evaluated by the harmonic "
        "solve (only 'static' is). Store a StaticSpectrum, or evaluate the spectrum "
        "yourself and pass it as `harmonic_injection`."
    )


def element_spectrum_dicts(appliance) -> Optional[list[Optional[SpectrumDict]]]:
    """Stored spectrum of each element ``k`` of a device (``phases[k]``), or ``None``.

    ``spectrum_per_phase`` gives element ``k`` the spectrum stored at ``phases[k]`` (a
    phase without an entry injects nothing: ``None``); ``spectrum`` gives every element
    the same one. Returns ``None`` when the device stores neither.
    """
    phases = appliance.phases
    spp = getattr(appliance, "spectrum_per_phase", None)
    if spp is not None:
        return [static_spectrum_dict(spp.get(ph), appliance.id) for ph in phases]
    shared = static_spectrum_dict(getattr(appliance, "spectrum", None), appliance.id)
    if shared is None:
        return None
    out: list[Optional[SpectrumDict]] = [shared] * len(phases)
    return out
