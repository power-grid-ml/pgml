"""Load-dependent harmonic emission laws shared by every scenario generator.

Measured devices do not emit a harmonic current proportional to their fundamental. Across
certified PV inverters and laboratory device racks the ratio ``|I_h| / |I_1|`` at 10 %
loading is 6-10x its value at rating, because part of the emission is present whenever the
device is on and does not scale with how hard it is driven. The law that fits (and
transfers to held-out devices) is COMPLEX AFFINE::

    I_h(lam) = A_h + B_h * lam        lam = |I_1| / |I_1 at rating|

with ``A_h`` the load-independent floor and ``B_h`` the load-proportional part. Two
further measured effects follow from the same two parameters: the emission ANGLE rotates
with loading (the phasor turns from ``arg A_h`` at low load toward ``arg B_h`` at rating),
and ``|I_h|(lam)`` has an interior cancellation null where the two parts are near
anti-phase. A separate, explicit phase slope models the rotation a device shows beyond the
affine geometry.

The functions here express those laws as CORRECTIONS on the proportional emission
``I_h = r_h * I_1`` the generators draw, so the rated operating point is untouched and a
zero floor reproduces the proportional law bit-for-bit. The sampler applies them through
the ``h_floor`` / ``h_floor_phase`` / ``h_slope``
:class:`~pgml.scenarios.ParameterSpec` fields (:mod:`pgml.scenarios.sampler`), and a
generator that composes an aggregate out of member devices applies the same functions per
member, so one definition serves both.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "LOADING_FLOOR",
    "affine_emission_correction",
    "phase_slope_shift",
]

#: The loading below which the randomized recipe evaluates the affine law at this value
#: instead of the drawn loading. The correction is expressed as a RATIO to the device's
#: fundamental current, which the solver then multiplies by the actual ``|I_1|``; the ratio
#: grows as ``1 / lam`` and the fundamental shrinks as ``lam``, so the harmonic current
#: itself stays finite — but at ``lam = 0`` the product is ``0 * inf``. Below the floor the
#: harmonic current therefore falls linearly with the fundamental to zero (a device that
#: draws nothing emits nothing), instead of holding the floor current of a device that is
#: on but idle.
LOADING_FLOOR = 0.05


def affine_emission_correction(lam: Tensor, floor: Tensor, delta_deg: Tensor) -> Tensor:
    """The affine emission law as a complex correction on the proportional phasor.

    Returns ``z(lam) / (lam * z(1))`` with ``z(lam) = floor * exp(j * delta) +
    (1 - floor) * lam``: multiplying the proportional harmonic phasor ``r_h * I_1(lam)`` by
    it gives the affine one, normalised so that the RATED (``lam = 1``) emission is
    unchanged for any floor. ``floor`` is ``|A_h|`` as a share of ``|A_h| + |B_h|`` and
    ``delta_deg`` is ``arg(A_h) - arg(B_h)``; ``0.61`` reproduces the ratio inflation
    measured across certified inverters, ``ratio(lam) = ratio_rated * (0.39 + 0.61 / lam)``.

    At ``floor = 0`` the result is exactly ``1 + 0j`` (``lam / lam`` is ``1.0`` in IEEE
    arithmetic for any finite non-zero ``lam``), so a configuration that does not ask for the
    law is unaffected bit-for-bit. Broadcasts over any leading axes.
    """
    a = floor * torch.exp(1j * torch.deg2rad(delta_deg))
    z = a + (1.0 - floor) * lam
    z_rated = a + (1.0 - floor)
    return z / (lam.clamp(min=1e-9) * z_rated)


def phase_slope_shift(slope_deg: Tensor, lam: Tensor) -> Tensor:
    """The explicit loading-dependent phase shift ``s_h * (lam - 1)`` [deg].

    Zero at rating by construction, so the rated emission angle stays the drawn one; a
    negative slope rotates the phasor one way as the device unloads, a positive one the
    other way. Independent of the affine law's own rotation (which is set by the floor
    angle) and added to it.
    """
    return slope_deg * (lam - 1.0)
