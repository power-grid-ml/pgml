"""Node-coherent harmonic "fingerprint" sampling over a step sequence.

Builds a ``[B, T]`` batch of harmonic injections in which every device keeps a stable,
recognisable per-node signature that varies realistically step to step — the training
signal a harmonic state estimator needs to attribute a pattern to a node. Each device
draws ``n_modes`` base spectra (operating "states"); over ``T`` steps it sticks to a
mode (Markov dwell) and wanders around it (AR(1) jitter), clamped to DIN EN 50160.

The output is a :class:`~pgml.scenarios.sampler.SampledScenarios` whose
``harmonic_injection`` carries ``[B, T]``-shaped per-(device, order) tensors; feeding it
through :func:`~pgml.scenarios.run.run_scenarios` (calculation="harmonic") yields node
voltages ``[B, T, H, N]`` (the explicit time axis). Per-step timestamps and the
ground-truth active mode per device/step are recorded in ``samples`` for ML labels.

Sampling uses seeded RNG (not the QMC cube): the Markov mode path and the AR(1) jitter
are inherently sequential. Deterministic for a fixed seed; all torch ops, no
``.item()``/``.detach()``, honors broadcasting so the realized tensors feed the solver.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from pgml.schemas.grid_schema import Grid

from .config import CoherentSpectrumConfig, Selector, SpectrumSweepConfig
from .en50160 import en50160_limit
from .sampler import SampledScenarios

_F64 = torch.float64


def _markov_path(
    b: int, n_dev: int, t: int, n_modes: int, dwell: float, gen: torch.Generator
) -> Tensor:
    """``[B, n_dev, T]`` int64 mode path: stay w.p. ``dwell``, else switch uniformly."""
    path = torch.empty((b, n_dev, t), dtype=torch.long)
    cur = torch.randint(0, n_modes, (b, n_dev), generator=gen)
    path[..., 0] = cur
    for step in range(1, t):
        if n_modes > 1:
            stay = torch.rand((b, n_dev), generator=gen) < dwell
            offset = torch.randint(1, n_modes, (b, n_dev), generator=gen)
            cur = torch.where(stay, cur, (cur + offset) % n_modes)
        path[..., step] = cur
    return path


def _ar1(shape: tuple, rho: float, gen: torch.Generator) -> Tensor:
    """AR(1) noise along the LAST axis: ``e_t = rho*e_{t-1} + sqrt(1-rho^2)*eta_t``."""
    eta = torch.randn(shape, generator=gen, dtype=_F64)
    e = torch.empty_like(eta)
    e[..., 0] = eta[..., 0]
    c = math.sqrt(1.0 - rho * rho)
    for step in range(1, shape[-1]):
        e[..., step] = rho * e[..., step - 1] + c * eta[..., step]
    return e


def sample_coherent_spectra(
    grid: Grid, config: CoherentSpectrumConfig
) -> SampledScenarios:
    """Sample node-coherent harmonic injection sequences (reproducible).

    Implements the Markov-mode + AR(1) jitter fingerprint model described in
    :class:`~pgml.scenarios.CoherentSpectrumConfig`.  The call is deterministic for a
    fixed ``config.seed``.

    Parameters
    ----------
    grid : Grid
        The reference grid.  Only the appliances matched by ``config.selector`` are
        varied; all others keep their nominal spectra.
    config : CoherentSpectrumConfig
        Fully specified sampling config (mode count, temporal structure, EN 50160
        clamping, etc.).  Validated on construction.

    Returns
    -------
    SampledScenarios
        ``operating_point`` is empty (fundamental P/Q stays nominal).
        ``harmonic_injection`` maps
        ``{device_id: {order: (mag[B, T], phase[B, T])}}`` — pass directly to
        ``solve_harmonic_flow``.
        ``samples`` contains the following keys (``name`` = ``config.name``,
        default ``"harmonics"``):

        - ``"<name>_mode"`` ``[B, n_dev, T]`` — active mode index per device per
          step; the ground-truth attribution label for ML training.
        - ``"<name>_mag"`` / ``"<name>_phase"`` ``[B, n_dev, n_ord, T]`` — realized
          injection magnitudes [pu] and phases [deg] after AR(1) jitter.
        - ``"<name>_mode_base_mag"`` — per-device mode fingerprint magnitudes
          (before jitter; shape ``[n_dev, n_ord, n_modes]`` or
          ``[B, n_dev, n_ord, n_modes]`` when ``resample_modes_per_scenario=True``).
        - ``"<name>_device_ids"`` ``[n_dev]`` — matched device IDs in selector order.
        - ``"time_s"`` ``[T]`` — step timestamps in seconds.

        Pass this :class:`SampledScenarios` to :func:`~pgml.scenarios.run_scenarios`
        (or directly to ``solve_harmonic_flow``) to obtain ``v[B, T, H, N]``.

    Raises
    ------
    ValueError
        If the selector matches no in-service components.

    Notes
    -----
    All intermediate tensors use ``float64``.  No ``.item()`` / ``.detach()`` calls
    are made; the realized tensors can feed the solver's autograd tape.
    """
    ids = config.selector.resolve(grid)
    if not ids:
        raise ValueError(
            "CoherentSpectrumConfig selector matched no in-service components."
        )
    nm = config.name
    n_dev, n_ord = len(ids), len(config.orders)
    n_modes, b, t = config.n_modes, config.n_scenarios, config.n_steps
    gen = torch.Generator().manual_seed(config.seed)

    # 1. per-order EN 50160 limits (also the upper clamp), or 1.0 (absolute pu).
    en50160 = config.harmonic_reference == "en50160"
    limits = torch.tensor(
        [en50160_limit(o) if en50160 else 1.0 for o in config.orders], dtype=_F64
    )  # [n_ord]

    # 2. base mode spectra (the fingerprints): magnitude (fraction of limit) + phase.
    mode_shape = (
        (b, n_dev, n_ord, n_modes)
        if config.resample_modes_per_scenario
        else (n_dev, n_ord, n_modes)
    )
    base_mag = config.mag_distribution.icdf(
        torch.rand(mode_shape, generator=gen, dtype=_F64)
    ) * limits.reshape(-1, 1)  # [..., n_ord, n_modes], broadcast limit over modes
    base_phase = config.phase_distribution.icdf(
        torch.rand(mode_shape, generator=gen, dtype=_F64)
    )

    # 3. Markov mode path + gather each step's base spectrum -> [B, n_dev, n_ord, T].
    path = _markov_path(b, n_dev, t, n_modes, config.dwell, gen)
    idx = path.unsqueeze(2).expand(b, n_dev, n_ord, t)  # [B, n_dev, n_ord, T]
    bm = base_mag if config.resample_modes_per_scenario else base_mag.unsqueeze(0)
    bp = base_phase if config.resample_modes_per_scenario else base_phase.unsqueeze(0)
    bm = bm.expand(b, n_dev, n_ord, n_modes).contiguous()
    bp = bp.expand(b, n_dev, n_ord, n_modes).contiguous()
    sel_mag = torch.gather(bm, -1, idx)  # [B, n_dev, n_ord, T]
    sel_phase = torch.gather(bp, -1, idx)

    # 4. AR(1) jitter around the selected base, clamped to [0, limit] / wrapped phase.
    e_mag = _ar1((b, n_dev, n_ord, t), config.ar1_rho, gen)
    e_phase = _ar1((b, n_dev, n_ord, t), config.ar1_rho, gen)
    mag = (sel_mag * (1.0 + config.jitter_mag * e_mag)).clamp(min=0.0)
    if en50160:
        mag = torch.minimum(mag, limits.reshape(1, 1, n_ord, 1))
    phase = sel_phase + config.jitter_phase_deg * e_phase  # degrees

    # 5. assemble harmonic_injection {id: {order: (mag[B,T], phase[B,T])}}.
    harmonic_injection: dict = {}
    for d, cid in enumerate(ids):
        harmonic_injection[cid] = {
            order: (mag[:, d, o, :], phase[:, d, o, :])
            for o, order in enumerate(config.orders)
        }

    samples = {
        f"{nm}_mode": path,  # [B, n_dev, T]
        f"{nm}_mag": mag,  # [B, n_dev, n_ord, T]
        f"{nm}_phase": phase,  # [B, n_dev, n_ord, T]
        f"{nm}_mode_base_mag": base_mag,  # fingerprints
        f"{nm}_device_ids": torch.tensor(ids, dtype=torch.long),
        "time_s": torch.arange(t, dtype=_F64) * config.step_size_s,  # [T]
    }
    return SampledScenarios(
        operating_point={},
        samples=samples,
        n_samples=b,
        config=config,
        harmonic_injection=harmonic_injection,
    )


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
        raise ValueError("spectrum_sweep selector matched no in-service components.")
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


__all__ = ["sample_coherent_spectra", "spectrum_sweep"]
