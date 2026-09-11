"""Harmonic excitation primitives: an upstream background and a per-target sweep.

Two ways to put harmonic content into a batch without a per-device emission model:

- :func:`build_background_sources` realizes a
  :class:`~pgml.scenarios.BackgroundHarmonicConfig` as batched
  :class:`~pgml.solver.NodeHarmonicSource` entries at the grid's supply nodes — one
  upstream network state every device on the feeder sees, drifting along the step axis.
- :func:`spectrum_sweep` injects one spectrum at one target per scenario (a diagonal
  enumeration), for mapping how a single source spreads through the network.

Both use seeded RNG off the autograd tape where they draw at all; the realized tensors
are plain torch and feed the solver, so no ``.item()`` / ``.detach()`` is involved.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from pgml.errors import InputError
from pgml.schemas.grid_schema import Grid

from .config import (
    BackgroundHarmonicConfig,
    Selector,
    SpectrumSweepConfig,
)
from .sampler import SampledScenarios

_F64 = torch.float64


def _ar1(shape: tuple, rho: float, gen: torch.Generator) -> Tensor:
    """Standard-normal stationary AR(1) noise along the LAST axis.

    ``e_t = rho*e_{t-1} + sqrt(1-rho^2)*eta_t`` with standard-normal ``eta``, so the
    drift is marginally standard normal whatever ``rho`` is and the caller scales it into
    the quantity it perturbs.
    """
    eta = torch.randn(shape, generator=gen, dtype=_F64)
    e = torch.empty_like(eta)
    e[..., 0] = eta[..., 0]
    c = math.sqrt(1.0 - rho * rho)
    for step in range(1, shape[-1]):
        e[..., step] = rho * e[..., step - 1] + c * eta[..., step]
    return e


def build_background_sources(
    grid: Grid,
    config: BackgroundHarmonicConfig,
    shape: tuple,
    gen: torch.Generator,
) -> list:
    """Realize the upstream background as batched ``NodeHarmonicSource`` entries.

    ``shape`` is the batch shape the spectrum tensors take — ``(B, T)``, matching the
    per-device injection tensors (``T = 1`` for snapshot recipes). The AR(1) drift runs
    along the STEP axis; a snapshot batch has no step to walk, so its scenarios sample
    the drift's stationary distribution independently — scenarios are far apart relative
    to any correlation time, never neighbours on one drift path. One drift series is
    shared by every order and every device on the feeder, which is the point: what the
    background contributes is common, not private to a device.

    The background is ONE upstream network state seen through every point of common
    coupling, so each in-service ``Source`` node receives a ``NodeHarmonicSource``
    carrying the SAME realized spectrum (an explicit ``config.node_id`` narrows the
    injection to that single node instead).

    Returns an empty list when no order is configured, so a caller can pass the result
    through unconditionally.
    """
    from pgml.solver import NodeHarmonicSource
    from pgml.topology import slack_node_ids

    if not config.magnitude_pu:
        return []
    if config.node_id is not None:
        node_ids = [int(config.node_id)]
    else:
        try:
            node_ids = slack_node_ids(grid)
        except ValueError:
            raise InputError(
                "BackgroundHarmonicConfig.node_id is None and the grid has no "
                "in-service Source to resolve it from; set node_id explicitly."
            ) from None

    if config.drift_std > 0.0 or config.drift_phase_deg > 0.0:
        drift = _ar1(shape, config.drift_rho, gen)
    else:
        drift = torch.zeros(shape, dtype=_F64)
    spectrum = {}
    for order, mag in sorted(config.magnitude_pu.items()):
        ang0 = float(config.phase_deg.get(order, 0.0))
        spectrum[int(order)] = (
            float(mag) * torch.exp(config.drift_std * drift),
            ang0 + config.drift_phase_deg * drift,
        )
    return [
        NodeHarmonicSource(
            node_id=int(node_id),
            phases=None,
            spectrum=spectrum,
            source_power_va=float(config.source_power_va),
            kind="voltage",
        )
        for node_id in node_ids
    ]


def spectrum_sweep(
    grid: Grid,
    selector: Selector | SpectrumSweepConfig,
    spectrum: dict | None = None,
    *,
    name: str = "injection",
) -> SampledScenarios:
    """Per-target harmonic-injection sweep (diagonal one-hot over matched devices).

    Scenario ``i`` injects the spectrum at target ``i`` ONLY (every other device
    silent); ``B = #targets``. The harmonic analogue of
    :func:`~pgml.scenarios.perturbation.perturbation_sweep` — for "inject one spectrum
    at each node, measure how it spreads".

    Parameters
    ----------
    grid:
        The reference grid; the selector's matched Load/Generator devices are the
        injection targets (injection is a Norton current at a device terminal).
    selector:
        A :class:`Selector` (then ``spectrum`` is required) or a fully-built
        :class:`SpectrumSweepConfig`.
    spectrum:
        ``{order: (magnitude_pu, phase_deg)}`` relative to the fundamental (order 1 is
        the implicit reference, dropped). Ignored when a config is passed.
    name:
        Label for the recorded target-id sample column (``"<name>_id"``).

    Returns
    -------
    SampledScenarios
        Empty ``operating_point`` (fundamental nominal); ``harmonic_injection`` is the
        diagonal ``{device_id: {order: (mag[B], phase[B])}}`` (mag is the spectrum value
        at the device's own scenario index, 0 elsewhere); ``samples["<name>_id"]`` is
        the injected device id per scenario.
    """
    config = (
        selector
        if isinstance(selector, SpectrumSweepConfig)
        else SpectrumSweepConfig.from_spectrum(selector, spectrum or {}, name=name)
    )
    ids = config.selector.resolve(grid)
    if not ids:
        raise InputError("spectrum_sweep selector matched no in-service components.")
    b = len(ids)

    harmonic_injection: dict = {}
    for j, tid in enumerate(ids):
        inj: dict = {}
        for order, mag, phase in zip(
            config.orders, config.magnitudes_pu, config.phases_deg
        ):
            magvec = torch.zeros(b, dtype=_F64)
            magvec[j] = mag  # one-hot: only scenario j injects at device tid
            inj[order] = (magvec, torch.full((b,), phase, dtype=_F64))
        harmonic_injection[tid] = inj

    samples = {f"{config.name}_id": torch.tensor(ids, dtype=torch.long)}
    return SampledScenarios(
        operating_point={},
        samples=samples,
        n_samples=b,
        config=config,
        harmonic_injection=harmonic_injection,
    )


__all__ = ["build_background_sources", "spectrum_sweep"]
