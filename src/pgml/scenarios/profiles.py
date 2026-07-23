"""Multi-scale synthetic load profiles for node-coherent scenario sequences.

Turns the constant per-scenario fundamental of a :class:`CoherentSpectrumConfig`
into a per-STEP time series. Each matched Load/Generator gets a multiplicative
factor composed on four time scales,

    ``factor(t) = f_seasonal(t) * f_weekly(t) * f_daily(t) * f_short(t)``,

applied to the device's per-scenario base ``P`` and ``Q`` together (constant power
factor). The daily shape is class-aware, keyed by each device's ``consumer_type``:

- ``household`` / ``heat_pump`` — low overnight, a small morning peak and a dominant
  evening peak;
- ``office`` / ``workshop`` — a smooth business-hours plateau;
- ``restaurant`` — twin lunch + dinner peaks;
- ``ev_charging`` — a late-evening charging peak;
- ``industrial_drive`` — a broad, shallow daytime plateau (near-flat base load);
- ``pv`` — a solar bell that is ZERO at night, with a daylight window and amplitude
  that widen in summer;
- everything else / unset — a neutral, near-flat default with a soft evening rise.

The shapes are smooth parametric curves (circular Gaussian bumps and sigmoid
plateaus), so they are continuous across midnight and differentiable in time. Every
consumption shape is zero-centred and scaled to a per-class daily depth, so the
composed factor oscillates around the device's base level; ``pv`` is a peak-1 bell
scaled by the seasonal amplitude.

Correlation model
-----------------
Per scenario, two SHARED latents make devices co-vary: a ``behavioral`` latent scales
the daily amplitude of every non-``pv`` device together, and a ``cloudiness`` latent
scales every ``pv`` device's output together. On top of them each device draws
idiosyncratic per-scenario values (overall level, amplitude multiplier, daily phase
offset) and a per-step AR(1) short-term term. See :class:`LoadProfileConfig`.

Reproducibility + tape hygiene
------------------------------
All draws come from a seeded RNG DERIVED from ``CoherentSpectrumConfig.seed`` with its
own offset — distinct from the fingerprint bank, the Markov path, the AR(1) jitter and
the operating-point cube — so enabling a profile leaves the harmonic fingerprint and
the raw parameter draws byte-identical. The factor is a plain ``float64`` tensor with
no ``requires_grad``; multiplying it onto a (possibly tensor-valued) base P/Q keeps the
solve differentiable w.r.t. that base while the sampling itself stays off the tape. All
sampling runs on CPU (like the rest of ``pgml.scenarios``); the tensors are promoted to
the solve device by :func:`~pgml.scenarios.run.run_scenarios`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
from torch import Tensor

from pgml.schemas.grid_schema import Generator, Grid, Load

from .config import CoherentSpectrumConfig, LoadProfileConfig

_F64 = torch.float64

# Derived-stream offset for the profile RNG: distinct from the fingerprint RNG
# (seeded from ``config.seed``) and the operating-point cube (``_OP_CUBE_SEED_OFFSET``
# in ``harmonics``), so the profile draws never disturb either stream.
_PROFILE_SEED_OFFSET = 0x85EBCA6B  # 2246822507

_SECONDS_PER_DAY = 86400.0
_SECONDS_PER_YEAR = 365.25 * _SECONDS_PER_DAY
# 1970-01-01 (epoch day 0) was a Thursday; Python weekday() has Monday=0, so adding
# 3 maps epoch day 0 to weekday index 3 (Thursday). Saturday=5, Sunday=6.
_EPOCH_WEEKDAY_OFFSET = 3.0

# Per-consumer-type daily preset key. Absent / unlisted types fall back to "neutral";
# "pv" is handled by the dedicated solar-bell path (never a consumption shape).
_PRESET_BY_TYPE = {
    "household": "household",
    "heat_pump": "household",
    "ev_charging": "ev",
    "office": "office",
    "workshop": "office",
    "restaurant": "restaurant",
    "industrial_drive": "industrial",
    "pv": "pv",
}

# Per-class daily DEPTH: the peak deviation of the (zero-centred, unit-peak) shape,
# i.e. how strongly the class swings over a day relative to its base level.
_TYPE_DEPTH = {
    "household": 0.60,
    "ev": 0.90,
    "office": 0.70,
    "restaurant": 0.70,
    "industrial": 0.25,
    "neutral": 0.10,
}

# Dense reference grid (hours in [0, 24)) used to zero-centre + unit-normalise each
# raw daily shape once, independent of the actual step grid.
_HOUR_REF = torch.linspace(0.0, 24.0, 241, dtype=_F64)[:-1]


def _cbump(hour: Tensor, mu: float, sigma: float) -> Tensor:
    """Circular Gaussian bump on the 24 h clock (continuous across midnight)."""
    d = torch.remainder(hour - mu + 12.0, 24.0) - 12.0
    return torch.exp(-0.5 * (d / sigma) ** 2)


def _sig(x: Tensor, w: float) -> Tensor:
    """Logistic ``sigmoid(x / w)`` — the building block of a smooth plateau."""
    return torch.sigmoid(x / w)


def _raw_daily(key: str, hour: Tensor) -> Tensor:
    """Raw (un-normalised) daily shape for a consumption preset key."""
    if key == "household":
        return 1.05 * _cbump(hour, 19.5, 2.2) + 0.45 * _cbump(hour, 7.0, 1.6)
    if key == "ev":
        return _cbump(hour, 22.0, 2.3)
    if key == "office":
        return _sig(hour - 8.0, 1.0) - _sig(hour - 18.0, 1.0)
    if key == "restaurant":
        return 0.7 * _cbump(hour, 13.0, 1.4) + 1.0 * _cbump(hour, 20.0, 2.0)
    if key == "industrial":
        return _sig(hour - 6.0, 1.5) - _sig(hour - 22.0, 1.5)
    # neutral: a soft evening rise around an otherwise flat day.
    return _cbump(hour, 18.0, 4.0)


def _daily_shape(key: str, hour: Tensor) -> Tensor:
    """Zero-centred daily shape scaled to the class depth (peak deviation).

    Normalises the raw shape on the dense reference grid to zero mean and unit peak
    deviation, then scales by the per-class depth, so ``1 + amplitude * shape`` swings
    by roughly ``+/- (amplitude * depth)`` around the base level over a day.
    """
    ref = _raw_daily(key, _HOUR_REF)
    m = ref.mean()
    peak = (ref - m).abs().max().clamp(min=1e-12)
    depth = _TYPE_DEPTH.get(key, _TYPE_DEPTH["neutral"])
    return depth * (_raw_daily(key, hour) - m) / peak


def _pv_bell(hour: Tensor, doy: Tensor, cfg: LoadProfileConfig) -> Tensor:
    """Solar bell: peak 1 at solar noon, zero at night, seasonally widening window.

    ``hour`` is ``[*batch, T]`` (device time-of-day, possibly phase-shifted); ``doy`` is
    ``[T]`` (day-of-year). The daylight half-width widens in summer; outside the window
    the bell is exactly 0.
    """
    daylen = cfg.pv_daylight_hours + cfg.pv_daylight_swing * torch.cos(
        2.0 * math.pi * (doy - cfg.pv_seasonal_peak_doy) / 365.25
    )
    half_width = (daylen / 2.0).clamp(min=0.5)  # [T]
    x = (hour - 12.0) / half_width  # solar noon at 12:00
    return torch.cos(0.5 * math.pi * x).clamp(min=0.0) ** 1.3


def _time_axis(
    start_time: str, step_size_s: float, t: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Absolute time axis for the profile phases.

    Returns ``(time_unix_s, hour_of_day, day_of_year, day_of_week)`` as ``float64``
    ``[T]`` tensors. A naive (timezone-less) ``start_time`` is interpreted as UTC so the
    epoch is reproducible across machines; the daily / weekly / seasonal fields are
    recovered from that same UTC frame by modular arithmetic (a wall-clock ``12:00``
    anchor therefore lands the solar bell at hour 12).
    """
    dt = datetime.fromisoformat(start_time)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    base_epoch = dt.timestamp()
    time_unix_s = base_epoch + torch.arange(t, dtype=_F64) * float(step_size_s)
    hour = torch.remainder(time_unix_s, _SECONDS_PER_DAY) / 3600.0
    doy = torch.remainder(time_unix_s, _SECONDS_PER_YEAR) / _SECONDS_PER_DAY
    dow = torch.remainder(time_unix_s / _SECONDS_PER_DAY + _EPOCH_WEEKDAY_OFFSET, 7.0)
    return time_unix_s, hour, doy, dow


def _weekly_factor(dow: Tensor, contrast: float) -> Tensor:
    """Smooth weekday/weekend factor: ``~1`` on weekdays, ``1 - contrast`` on weekends.

    ``dow`` is a continuous day-of-week index (Monday 0 .. Sunday 6, fractional over the
    day). A circular Gaussian centred on the Sat/Sun midpoint gives a smooth weekend
    dip whose depth is ``contrast`` (a single aggregate contrast; positive means lower
    weekend consumption).
    """
    d = torch.remainder(dow - 5.5 + 3.5, 7.0) - 3.5  # circular distance to Sat/Sun mid
    weekend = torch.exp(-0.5 * (d / 0.75) ** 2)
    return 1.0 - contrast * weekend


@dataclass(frozen=True)
class ProfileDraw:
    """Realized profile factors for a coherent batch.

    Attributes
    ----------
    factor:
        ``[B, n_dev, T]`` float64 multiplicative factor per profiled device per step.
    device_ids:
        int64 ``[n_dev]`` profiled device ids (Load/Generator), in selector order.
    time_unix_s:
        float64 ``[T]`` absolute epoch seconds per step (``start_time`` + ``k*dt``).
    """

    factor: Tensor
    device_ids: Tensor
    time_unix_s: Tensor


def _profiled_ids(grid: Grid, cfg: LoadProfileConfig) -> list[int]:
    """Ids of the devices the profile applies to (in appliance order)."""
    if cfg.selector is not None:
        return cfg.selector.resolve(grid)
    return [
        a.id
        for a in grid.appliances
        if isinstance(a, (Load, Generator)) and a.in_service
    ]


def load_profile_factors(grid: Grid, config: CoherentSpectrumConfig) -> ProfileDraw:
    """Draw the per-device, per-step profile factors for a coherent config.

    Parameters
    ----------
    grid:
        The reference grid; only the profiled Load/Generator devices are used.
    config:
        A :class:`CoherentSpectrumConfig` whose ``profile`` and ``start_time`` are set.

    Returns
    -------
    ProfileDraw
        ``factor`` ``[B, n_dev, T]`` (float64, non-negative), ``device_ids`` ``[n_dev]``,
        ``time_unix_s`` ``[T]``.
    """
    cfg = config.profile
    if cfg is None:  # pragma: no cover - guarded by the caller
        raise ValueError("load_profile_factors requires config.profile to be set.")
    if config.start_time is None:  # pragma: no cover - validated on config
        raise ValueError("load_profile_factors requires config.start_time.")

    ids = _profiled_ids(grid, cfg)
    by_id = {a.id: a for a in grid.appliances}
    b, t = int(config.n_scenarios), int(config.n_steps)
    n_dev = len(ids)

    time_unix_s, hour, doy, dow = _time_axis(config.start_time, config.step_size_s, t)

    if n_dev == 0:  # nothing matched: an empty factor bank (still shape-correct).
        return ProfileDraw(
            factor=torch.zeros((b, 0, t), dtype=_F64),
            device_ids=torch.zeros((0,), dtype=torch.long),
            time_unix_s=time_unix_s,
        )

    seed = (int(config.seed) + _PROFILE_SEED_OFFSET) & 0x7FFFFFFF
    gen = torch.Generator().manual_seed(seed)

    # Per-scenario shared latents (standard normal), clamped so the co-variation
    # scaling stays non-negative.
    z_behavior = torch.randn((b, 1), generator=gen, dtype=_F64)
    z_cloud = torch.randn((b, 1), generator=gen, dtype=_F64)
    behavioral_scale = (1.0 + cfg.behavioral_coupling * z_behavior).clamp(min=0.0)
    cloud_scale = (1.0 + cfg.cloud_coupling * z_cloud).clamp(min=0.0)

    # Per-scenario, per-device idiosyncratic draws.
    def _uniform(lo: float, hi: float) -> Tensor:
        return lo + (hi - lo) * torch.rand((b, n_dev), generator=gen, dtype=_F64)

    level = _uniform(cfg.level_min, cfg.level_max)
    amp_jitter = _uniform(cfg.amplitude_jitter_min, cfg.amplitude_jitter_max)
    shift = _uniform(-cfg.phase_offset_hours, cfg.phase_offset_hours)

    # Effective per-device daily amplitude / pv output scale (shared latent x jitter).
    amp_daily = (cfg.daily_amplitude * behavioral_scale * amp_jitter).clamp(min=0.0)
    pv_scale = (cloud_scale * amp_jitter).clamp(min=0.0)

    # Per-device per-step AR(1) short-term term.
    e_short = _ar1((b, n_dev, t), cfg.short_rho, gen)  # [B, n_dev, T]

    # Slow (device-independent) modulators, [T].
    f_seasonal = 1.0 + cfg.seasonal_amplitude * torch.cos(
        2.0 * math.pi * (doy - cfg.seasonal_peak_doy) / 365.25
    )
    f_seasonal_pv = 1.0 + cfg.pv_seasonal_amplitude * torch.cos(
        2.0 * math.pi * (doy - cfg.pv_seasonal_peak_doy) / 365.25
    )
    f_weekly = _weekly_factor(dow, cfg.weekend_contrast)

    factors = []
    for d, cid in enumerate(ids):
        ctype = getattr(by_id[cid], "consumer_type", None)
        # ``consumer_type`` is a ConsumerType (str) enum or a plain string; key on its
        # string value (``ConsumerType.PV.value == "pv"``) so both forms resolve.
        ctype_key = getattr(ctype, "value", ctype)
        key = _PRESET_BY_TYPE.get(ctype_key, "neutral")
        hour_d = hour.unsqueeze(0) - shift[:, d : d + 1]  # [B, T]
        short_d = 1.0 + cfg.short_sigma * e_short[:, d, :]  # [B, T]
        if key == "pv":
            bell = _pv_bell(hour_d, doy.unsqueeze(0), cfg)  # [B, T]
            fac = pv_scale[:, d : d + 1] * f_seasonal_pv.unsqueeze(0) * bell * short_d
        else:
            shape = _daily_shape(key, hour_d)  # [B, T]
            daily = 1.0 + amp_daily[:, d : d + 1] * shape
            fac = (
                level[:, d : d + 1]
                * f_seasonal.unsqueeze(0)
                * f_weekly.unsqueeze(0)
                * daily
                * short_d
            )
        factors.append(fac.clamp(min=0.0))

    factor = torch.stack(factors, dim=1)  # [B, n_dev, T]
    return ProfileDraw(
        factor=factor,
        device_ids=torch.as_tensor(ids, dtype=torch.long),
        time_unix_s=time_unix_s,
    )


def _ar1(shape: tuple, rho: float, gen: torch.Generator) -> Tensor:
    """AR(1) noise along the LAST axis: ``e_t = rho*e_{t-1} + sqrt(1-rho^2)*eta_t``.

    Mirrors :func:`pgml.scenarios.harmonics._ar1` (a standard-normal stationary AR(1));
    kept local so the profile stream is self-contained.
    """
    eta = torch.randn(shape, generator=gen, dtype=_F64)
    e = torch.empty_like(eta)
    e[..., 0] = eta[..., 0]
    c = math.sqrt(1.0 - rho * rho)
    for step in range(1, shape[-1]):
        e[..., step] = rho * e[..., step - 1] + c * eta[..., step]
    return e


def _mul(base, fac: Tensor) -> Tensor:
    """Multiply a per-scenario / scalar base P or Q by a ``[B, T]`` factor.

    ``base`` is a python float, a 0-d/``[B]`` tensor (a per-scenario draw), or a
    (tensor) nominal. Its batch is right-padded with a step axis so a ``[B]`` base
    broadcasts against ``[B, T]``; grad to a tensor base is preserved.
    """
    if isinstance(base, Tensor):
        b = base
        while b.ndim < fac.ndim:
            b = b.unsqueeze(-1)
        return b * fac
    return base * fac


def apply_load_profiles(
    grid: Grid, config: CoherentSpectrumConfig, operating_point: dict
) -> tuple[dict, dict]:
    """Lift a per-scenario operating point to a per-step ``[B, T]`` one via the profile.

    Each profiled device's base P/Q (its ``operating_point`` entry if present, else its
    nominal ``p_nom_w`` / ``q_nom_var``) is multiplied by the device's profile factor,
    so the totals gain the step axis (``p_w`` / ``q_var`` become ``[B, T]``; a per-phase
    base keeps its structure, each phase scaled by the same factor). Non-profiled
    entries (e.g. a Source ``u_ref_scale``) pass through untouched.

    Parameters
    ----------
    grid, config:
        The reference grid and the coherent config (``config.profile`` set).
    operating_point:
        The per-scenario base operating point (``[B]`` draws or empty). NOT mutated —
        a new dict is returned.

    Returns
    -------
    (operating_point, samples)
        The lifted operating point and the profile ``samples`` — ``"<name>_profile_factor"``
        ``[B, n_dev, T]``, ``"<name>_profile_device_ids"`` ``[n_dev]``, and
        ``"time_unix_s"`` ``[T]``.
    """
    draw = load_profile_factors(grid, config)
    by_id = {a.id: a for a in grid.appliances}
    nm = config.name

    op = {cid: dict(entry) for cid, entry in operating_point.items()}
    for d, cid in enumerate(draw.device_ids.tolist()):
        fac = draw.factor[:, d, :]  # [B, T]
        entry = op.get(cid, {})
        appliance = by_id[cid]
        new: dict = {}
        if "p_per_phase_w" in entry or "q_per_phase_var" in entry:
            if "p_per_phase_w" in entry:
                new["p_per_phase_w"] = [_mul(x, fac) for x in entry["p_per_phase_w"]]
            else:
                new["p_w"] = _mul(entry.get("p_w", appliance.p_nom_w), fac)
            if "q_per_phase_var" in entry:
                new["q_per_phase_var"] = [
                    _mul(x, fac) for x in entry["q_per_phase_var"]
                ]
            else:
                new["q_var"] = _mul(entry.get("q_var", appliance.q_nom_var), fac)
        else:
            new["p_w"] = _mul(entry.get("p_w", appliance.p_nom_w), fac)
            new["q_var"] = _mul(entry.get("q_var", appliance.q_nom_var), fac)
        op[cid] = new

    samples = {
        f"{nm}_profile_factor": draw.factor,  # [B, n_dev, T]
        f"{nm}_profile_device_ids": draw.device_ids,  # [n_dev]
        "time_unix_s": draw.time_unix_s,  # [T]
    }
    return op, samples


__all__ = ["ProfileDraw", "load_profile_factors", "apply_load_profiles"]
