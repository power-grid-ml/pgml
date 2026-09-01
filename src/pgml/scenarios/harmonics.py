"""Node-coherent harmonic "fingerprint" sampling over a step sequence.

Builds a ``[B, T]`` batch of harmonic injections in which every device keeps a stable,
recognisable per-node signature that varies realistically step to step — the training
signal a harmonic state estimator needs to attribute a pattern to a node. Each device
draws ``n_modes`` base spectra (operating "states"); over ``T`` steps it sticks to a
mode (Markov dwell) and wanders around it (AR(1) jitter), clamped to the per-order
emission reference (IEC 61000-3-2 by default; DIN EN 50160 or none optionally).

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

from pgml.errors import InputError
from pgml.schemas.grid_schema import Grid

from .composition import (
    lift_operating_point_to_bt,
    resolve_composed_ids,
    sample_device_composition,
)
from .config import (
    BackgroundHarmonicConfig,
    CoherentSpectrumConfig,
    ScenarioConfig,
    Selector,
    SpectrumSweepConfig,
)
from .en50160 import en50160_limit
from .iec61000_3_2 import iec61000_3_2_device_caps
from .profiles import apply_load_profiles
from .sampler import SampledScenarios, sample

_F64 = torch.float64

# The coherent operating-point unit cube is drawn on a stream DISTINCT from the
# fingerprint RNG (which is seeded from ``config.seed``): a fixed derived offset keeps
# the two independent, so the realized ``harmonic_injection`` is byte-identical with and
# without ``config.parameters`` (reproducible reconstruction from config + seed).
_OP_CUBE_SEED_OFFSET = 0x9E3779B9  # 2654435769; golden-ratio mix


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


def _sample_operating_specs(
    grid: Grid, config: CoherentSpectrumConfig, b: int
) -> tuple[dict, dict]:
    """Draw the coherent config's fundamental operating-point specs, once per scenario.

    Reuses the :class:`ScenarioConfig` sampler (Sobol unit cube, copula correlation,
    per-phase symmetry, and the source ``u_ref`` scale) with ``n_samples = b`` on a seed
    DERIVED from ``config.seed`` (a stream distinct from the fingerprint RNG, so the
    harmonic injection is unchanged). Returns ``(operating_point, samples)`` — the ``[B]``
    per-scenario operating point (constant across the ``T`` steps; the harmonic solve
    aligns the fundamental voltage against the ``[B, T]`` injection) and the raw ``[B, ...]``
    draws recorded in ``samples`` exactly as :class:`ScenarioConfig` records them. Empty
    when the config has no ``parameters``.
    """
    if not config.parameters:
        return {}, {}
    op_seed = (int(config.seed) + _OP_CUBE_SEED_OFFSET) & 0x7FFFFFFF
    inner = ScenarioConfig(
        n_samples=b,
        seed=op_seed,
        method="sobol",
        parameters=list(config.parameters),
        factors=list(config.factors),
    )
    drawn = sample(grid, inner)
    return drawn.operating_point, drawn.samples


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
        Fully specified sampling config (mode count, temporal structure, emission
        reference + clamping, etc.).  Validated on construction.

    Returns
    -------
    SampledScenarios
        ``operating_point`` is empty when ``config.parameters`` is empty (fundamental P/Q
        stays nominal, source at ``u_ref_v``); otherwise it carries the per-scenario
        fundamental draws as ``[B]`` tensors (constant across the ``T`` steps — the
        harmonic solve broadcasts the ``[B]`` fundamental against the ``[B, T]`` injection),
        incl. a per-source ``u_ref_scale`` for the slack. When ``config.profile`` is set
        the P/Q totals instead carry the step axis (``[B, T]``), a TIME-VARYING
        fundamental aligned with the ``[B, T]`` injection.
        ``harmonic_injection`` maps
        ``{device_id: {order: (mag[B, T], phase[B, T])}}`` — pass directly to
        ``solve_harmonic_flow``.
        ``samples`` contains the following keys (``name`` = ``config.name``,
        default ``"harmonics"``):

        - ``"<name>_mode"`` ``[B, n_dev, T]`` — active mode index per device per
          step; the ground-truth attribution label for ML training.
        - ``"<name>_mag"`` / ``"<name>_phase"`` ``[B, n_dev, n_ord, T]`` — the REALIZED
          injection per device and order after AR(1) jitter and the emission clamp:
          magnitude in per unit of the device's own fundamental current, phase in
          degrees. The device axis is ``"<name>_device_ids"``.
        - ``"<name>_mode_base_mag"`` — per-device mode fingerprint magnitudes
          (before jitter; shape ``[n_dev, n_ord, n_modes]`` or
          ``[B, n_dev, n_ord, n_modes]`` when ``resample_modes_per_scenario=True``).
        - ``"<name>_device_ids"`` ``[n_dev]`` — matched device IDs in selector order.
        - ``"time_s"`` ``[T]`` — step timestamps in seconds (relative to the start).

        When ``config.profile`` is set the fundamental P/Q is TIME-VARYING (per step),
        so ``operating_point`` carries the step axis (``[B, T]``) and ``samples`` also
        records the ground-truth profile:

        - ``"<name>_profile_factor"`` ``[B, n_dev, T]`` — the realized multiplicative
          profile factor per profiled device per step (ML ground truth).
        - ``"<name>_profile_device_ids"`` ``[n_dev]`` — the profiled device IDs.
        - ``"time_unix_s"`` ``[T]`` — absolute per-step timestamps (epoch seconds,
          derived from ``config.start_time`` + ``k * step_size_s``).

        When ``config.composition`` is set the covered loads leave the fingerprint device
        set, and their class attribution plus their REALIZED aggregate spectrum
        (``"<name>_composed_mag"`` / ``"<name>_composed_phase"`` ``[B, n_agg, n_ord, T]``
        on the ``"<name>_agg_ids"`` device axis, same units as ``"<name>_mag"``) are
        recorded instead — see
        :func:`~pgml.scenarios.composition.sample_device_composition`.

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
    nm = config.name
    n_ord = len(config.orders)
    n_modes, b, t = config.n_modes, config.n_scenarios, config.n_steps

    # A statistical device composition (if set) SUPERSEDES the mode-bank fingerprint for
    # the loads it covers: those ids are dropped from the fingerprint device set and
    # filled from the composition instead. Composition-free runs keep `fp_ids == ids`
    # (byte-identical).
    comp = config.composition
    composed_ids = resolve_composed_ids(grid, comp) if comp is not None else []
    composed_set = set(composed_ids)
    fp_ids = [i for i in ids if i not in composed_set]
    if not fp_ids and not composed_ids:
        raise InputError(
            "CoherentSpectrumConfig selector matched no in-service components."
        )

    samples: dict = {"time_s": torch.arange(t, dtype=_F64) * config.step_size_s}  # [T]
    harmonic_injection: dict = {}

    if fp_ids:
        n_dev = len(fp_ids)
        gen = torch.Generator().manual_seed(config.seed)
        # The per-device fingerprint (mode) bank draws from its own generator when
        # `mode_bank_seed` is set (a distinct signature bank for a held-out test set);
        # `None` draws it from `gen`, byte-identical to a config without the field -- the
        # bank is the first consumption of the `seed` stream (the Markov path + AR(1)
        # jitter follow). With `mode_bank_seed` set, the path + jitter no longer consume
        # the bank draws, so they are a distinct (statistically identical) realization.
        bank_gen = (
            gen
            if config.mode_bank_seed is None
            else torch.Generator().manual_seed(int(config.mode_bank_seed))
        )

        # 1. per-order emission caps (also the upper clamp). IEC 61000-3-2 is PER DEVICE
        #    (each device's nominal P + node voltage): shape [n_dev, n_ord, 1]. EN 50160
        #    is a global per-order voltage-compatibility level; None is absolute pu (cap
        #    1.0): shape [n_ord, 1]. The trailing 1 broadcasts over the mode axis (base
        #    multiply) and the T axis (clamp).
        ref = config.harmonic_reference
        clamp_to_cap = ref is not None
        if ref == "iec61000-3-2":
            cap_map = iec61000_3_2_device_caps(
                grid, fp_ids, config.orders, emission_class=config.emission_class
            )
            caps = torch.tensor(
                [[cap_map[cid][o] for o in config.orders] for cid in fp_ids],
                dtype=_F64,
            ).reshape(n_dev, n_ord, 1)
        else:
            caps = torch.tensor(
                [en50160_limit(o) if ref == "en50160" else 1.0 for o in config.orders],
                dtype=_F64,
            ).reshape(n_ord, 1)

        # 2. base mode spectra (the fingerprints): magnitude (fraction of cap) + phase.
        mode_shape = (
            (b, n_dev, n_ord, n_modes)
            if config.resample_modes_per_scenario
            else (n_dev, n_ord, n_modes)
        )
        base_mag = (
            config.mag_distribution.icdf(
                torch.rand(mode_shape, generator=bank_gen, dtype=_F64)
            )
            * caps
        )  # [..., n_ord, n_modes], cap broadcasts over modes (+ devices for [n_ord, 1])
        base_phase = config.phase_distribution.icdf(
            torch.rand(mode_shape, generator=bank_gen, dtype=_F64)
        )

        # 3. Markov mode path + gather each step's base spectrum -> [B, n_dev, n_ord, T].
        path = _markov_path(b, n_dev, t, n_modes, config.dwell, gen)
        idx = path.unsqueeze(2).expand(b, n_dev, n_ord, t)  # [B, n_dev, n_ord, T]
        bm = base_mag if config.resample_modes_per_scenario else base_mag.unsqueeze(0)
        bp = (
            base_phase
            if config.resample_modes_per_scenario
            else base_phase.unsqueeze(0)
        )
        bm = bm.expand(b, n_dev, n_ord, n_modes).contiguous()
        bp = bp.expand(b, n_dev, n_ord, n_modes).contiguous()
        sel_mag = torch.gather(bm, -1, idx)  # [B, n_dev, n_ord, T]
        sel_phase = torch.gather(bp, -1, idx)

        # 4. AR(1) jitter around the selected base, clamped to [0, cap] / wrapped phase.
        e_mag = _ar1((b, n_dev, n_ord, t), config.ar1_rho, gen)
        e_phase = _ar1((b, n_dev, n_ord, t), config.ar1_rho, gen)
        mag = (sel_mag * (1.0 + config.jitter_mag * e_mag)).clamp(min=0.0)
        if clamp_to_cap:
            mag = torch.minimum(mag, caps)  # caps' trailing 1 broadcasts over T
        phase = sel_phase + config.jitter_phase_deg * e_phase  # degrees

        # 5. assemble harmonic_injection {id: {order: (mag[B,T], phase[B,T])}}.
        for d, cid in enumerate(fp_ids):
            harmonic_injection[cid] = {
                order: (mag[:, d, o, :], phase[:, d, o, :])
                for o, order in enumerate(config.orders)
            }

        samples.update(
            {
                f"{nm}_mode": path,  # [B, n_dev, T]
                f"{nm}_mag": mag,  # [B, n_dev, n_ord, T]
                f"{nm}_phase": phase,  # [B, n_dev, n_ord, T]
                f"{nm}_mode_base_mag": base_mag,  # fingerprints
                f"{nm}_device_ids": torch.tensor(fp_ids, dtype=torch.long),
            }
        )

    # Optional per-scenario fundamental operating point (load / PV / slack-voltage specs).
    # Drawn on a separate stream (above) so the harmonic fingerprint is untouched.
    operating_point, op_samples = _sample_operating_specs(grid, config, b)
    samples.update(op_samples)

    # Optional TIME-VARYING fundamental profile: lift the per-scenario operating point
    # to a per-step [B, T] one (seasonal / weekly / daily / short-term, class-aware).
    # Drawn on yet another distinct stream, so the harmonic fingerprint + the raw
    # parameter draws stay byte-identical to a run without a profile. The harmonic
    # injection magnitude is RELATIVE to each device's fundamental current, so a
    # profile-scaled fundamental already scales the absolute harmonic current at the
    # solve — no extra coupling here.
    if config.profile is not None:
        operating_point, profile_samples = apply_load_profiles(
            grid, config, operating_point
        )
        samples.update(profile_samples)

    # Statistical device composition: OVERRIDE the composed loads' fundamental (a per-step
    # [B, T] aggregate) and harmonic injection (the summed member spectrum). Drawn on
    # streams distinct from the fingerprint / op-cube / profile. The mixed [B]/[B, T]
    # operating point is unified to [B, T] so every device shares one leading batch.
    if comp is not None:
        draw = sample_device_composition(grid, config)
        if draw.operating_point:
            operating_point = lift_operating_point_to_bt(operating_point, b, t)
            operating_point.update(draw.operating_point)
            harmonic_injection.update(draw.harmonic_injection)
            samples.update(draw.samples)
        if "time_unix_s" not in samples:
            # A composed sequence is anchored to absolute time (activity presets are
            # clock-driven); stamp the step axis like the profile path does, so the
            # dataset carries its own time axis instead of consumers re-deriving it.
            from .profiles import _time_axis

            samples["time_unix_s"] = _time_axis(
                config.start_time, config.step_size_s, t
            )[0]

    node_sources = (
        build_background_sources(
            grid,
            config.background,
            (b, t),
            torch.Generator().manual_seed(config.seed + 8117),
        )
        if config.background is not None
        else []
    )
    return SampledScenarios(
        operating_point=operating_point,
        samples=samples,
        n_samples=b,
        config=config,
        harmonic_injection=harmonic_injection,
        node_sources=node_sources,
    )


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


__all__ = ["sample_coherent_spectra", "spectrum_sweep"]
