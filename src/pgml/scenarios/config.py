"""Scenario configuration: serializable, reproducible sampling spec.

A `ScenarioConfig` (+ its `seed`) fully determines a batch of realized operating
points — saving the config reproduces the dataset exactly (reproducibility is
paramount for ML experiment tracking). The config is plain pydantic (floats), so it
serialises to YAML/JSON; the sampler turns it into batched torch tensors.

Distributions expose a closed-form inverse CDF `icdf(u)` mapping unit-cube samples
``u in [0,1]`` to values, so a single code path serves BOTH independent sampling
(``u`` from a seeded RNG) and quasi-Monte-Carlo / hyperspace sampling (``u`` from a
Sobol or Latin-hypercube engine — better coverage of the parameter space for
training data).
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from pgml.schemas.grid_schema import Phase

_U_EPS = 1e-7  # clamp unit samples off {0,1} so Gaussian-tail icdf stays finite.


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Distributions (closed-form icdf -> QMC-ready)
# =============================================================================
class Uniform(_Base):
    kind: Literal["uniform"] = "uniform"
    low: float
    high: float

    def icdf(self, u: Tensor) -> Tensor:
        return self.low + u * (self.high - self.low)


class Normal(_Base):
    kind: Literal["normal"] = "normal"
    loc: float
    scale: float = Field(gt=0.0)

    def icdf(self, u: Tensor) -> Tensor:
        u = u.clamp(_U_EPS, 1.0 - _U_EPS)
        return self.loc + self.scale * math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


class LogNormal(_Base):
    kind: Literal["lognormal"] = "lognormal"
    loc: float  # mean of the underlying normal (in log space)
    scale: float = Field(gt=0.0)

    def icdf(self, u: Tensor) -> Tensor:
        u = u.clamp(_U_EPS, 1.0 - _U_EPS)
        z = self.loc + self.scale * math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
        return torch.exp(z)


class LogUniform(_Base):
    """Uniform in log-space — for parameters spanning orders of magnitude."""

    kind: Literal["loguniform"] = "loguniform"
    low: float = Field(gt=0.0)
    high: float = Field(gt=0.0)

    def icdf(self, u: Tensor) -> Tensor:
        ll, lh = math.log(self.low), math.log(self.high)
        return torch.exp(ll + u * (lh - ll))


class Constant(_Base):
    kind: Literal["constant"] = "constant"
    value: float

    def icdf(self, u: Tensor) -> Tensor:
        return torch.full_like(u, self.value)


Distribution = Annotated[
    Union[Uniform, Normal, LogNormal, LogUniform, Constant],
    Field(discriminator="kind"),
]


# =============================================================================
# Targeting + parameter spec
# =============================================================================
class Selector(_Base):
    """Selects which appliances a parameter varies.

    ``ids`` (specific), ``consumer_type`` (e.g. ``"pv"``), both None = ALL of the
    given component kind. Filters combine (AND). ``component="source"`` targets the
    slack :class:`~pgml.schemas.grid_schema.Source` appliances (for the ``"u_ref"``
    field); a source has no ``consumer_type`` (setting it matches nothing).
    ``component="storage"`` varies the SIGNED storage setpoint (positive =
    discharging/injecting, negative = charging — the schema's generator-consistent
    convention), so a ``[low, high]`` band spanning zero sweeps charge and discharge.
    """

    component: Literal["load", "generator", "storage", "source"] = "load"
    ids: Optional[list[int]] = None
    consumer_type: Optional[str] = None

    def resolve(self, grid) -> list[int]:
        from pgml.schemas.grid_schema import Generator, Load, Source, Storage

        cls = {
            "load": Load,
            "generator": Generator,
            "storage": Storage,
            "source": Source,
        }[self.component]
        out: list[int] = []
        for a in grid.appliances:
            if not isinstance(a, cls) or not a.in_service:
                continue
            if self.ids is not None and a.id not in self.ids:
                continue
            # A Source carries no consumer_type; a consumer_type filter matches none.
            if self.consumer_type is not None and (
                getattr(a, "consumer_type", None) != self.consumer_type
            ):
                continue
            out.append(a.id)
        return out


class LatentFactor(_Base):
    """A shared latent driver coupling several :class:`ParameterSpec`.

    Any spec whose ``correlation.factor`` names this factor is coupled to it through a
    single-factor Gaussian copula, so their realized values rise and fall together
    (e.g. all PV generators share a ``"solar"`` factor — the sun shines on the whole
    grid). The factor is a standard-normal latent: it consumes ONE sampling dimension
    and does NOT change any spec's marginal distribution, only the dependence between
    the specs/components that reference it.
    """

    name: str


class Correlation(_Base):
    """Couple a :class:`ParameterSpec` to a shared :class:`LatentFactor`.

    Single-factor Gaussian copula: every matched component ``i`` draws a latent score
    ``Z_i = sqrt(rho)*Z_factor + sqrt(1-rho)*eps_i`` (``eps_i`` idiosyncratic), which
    is mapped back through the standard-normal CDF and the spec's ``icdf`` — so each
    component keeps the spec's marginal distribution while the pairwise correlation
    between two components on the same factor equals ``rho``. ``rho=0`` reproduces
    ``per="each"`` (independent); ``rho=1`` reproduces ``per="shared"`` (identical).
    When set, ``correlation`` supersedes ``per``.
    """

    factor: str
    rho: float = Field(ge=0.0, le=1.0)


class ParameterSpec(_Base):
    """One varied quantity.

    - ``field``: a POWER field — ``"p"`` / ``"q"`` (one) or ``"pq"`` (both, same
      factor — vary apparent power at constant power factor; ``pq`` requires
      ``mode="scale"``) — the SLACK-VOLTAGE field ``"u_ref"`` (a per-scenario scale on
      the :class:`~pgml.schemas.grid_schema.Source` reference voltage
      ``u_ref_v``; requires ``selector.component="source"`` and ``mode="scale"``, and
      supports neither per-phase ``symmetry`` nor harmonic options) — or a HARMONIC
      field — ``"h_mag"`` (per-order injection magnitude relative to the fundamental) /
      ``"h_phase"`` (per-order phase in degrees). Harmonic fields require ``orders`` and
      feed ``solve_harmonic_flow(harmonic_injection=...)`` instead of an operating point.
      ``u_ref`` writes a per-source ``u_ref_scale`` operating-point entry that the
      ideal-slack solve multiplies onto ``u_ref_v`` (a batched fundamental boundary).
    - ``mode``: ``"scale"`` (multiply the nominal P/Q or the stored per-order spectrum
      magnitude) or ``"absolute"`` (the sampled value IS the W / var / pu / degrees).
    - ``per``: ``"each"`` (every matched component varies independently — one sampling
      dimension per component) or ``"shared"`` (one sample applied to all matched).
      Ignored when ``correlation`` is set.
    - ``correlation``: optional :class:`Correlation` coupling matched components
      through a shared :class:`LatentFactor` (power fields only).
    - ``symmetry`` (per-phase, power fields only): ``"balanced"`` (one value per
      component applied to all phases — writes a scalar total, equally split
      downstream), ``"independent"`` (each phase drawn independently), or
      ``"small_imbalance"`` (a balanced base plus a small per-phase perturbation of
      fractional std ``imbalance``). The latter two write per-phase ``p_per_phase_w`` /
      ``q_per_phase_var`` overrides, which promote the solve to ASYMMETRIC
      automatically (``symmetry="auto"`` resolution).
    - ``imbalance``: fractional std of the per-phase perturbation; required (> 0) iff
      ``symmetry="small_imbalance"``.
    - ``orders``: harmonic orders varied by a harmonic field (e.g. ``[3, 5, 7]``).
    - ``harmonic_reference``: reference standard that turns an ``h_mag`` distribution
      into a FRACTION of a per-order limit (so use a ``[0, 1]`` distribution).

      * ``"iec61000-3-2"`` -- the IEC 61000-3-2 appliance harmonic-CURRENT emission
        limits: the physically correct per-device reference for a device current
        fingerprint (the limit is per-device, from its nominal power, node voltage and
        ``emission_class``).
      * ``"en50160"`` -- a DIN EN 50160 voltage-compatibility-level-SHAPED spectrum.
        These are supply VOLTAGE compatibility levels, NOT an appliance emission model;
        kept for background-distortion-shaped experiments and backward compatibility.
      * ``None`` -- the sampled value is an absolute pu magnitude (or a ``scale`` of the
        stored spectrum).
    - ``emission_class``: IEC 61000-3-2 equipment class ``"A"``/``"B"``/``"C"``/``"D"``,
      or ``"auto"`` (default) to resolve it per device from its ``consumer_type`` and
      nominal power. Valid only with ``harmonic_reference="iec61000-3-2"``.
    """

    name: str
    selector: Selector
    distribution: Distribution
    field: Literal["p", "q", "pq", "u_ref", "h_mag", "h_phase"] = "pq"
    mode: Literal["scale", "absolute"] = "scale"
    per: Literal["each", "shared"] = "each"
    correlation: Optional[Correlation] = None
    symmetry: Literal["balanced", "independent", "small_imbalance"] = "balanced"
    imbalance: float = Field(default=0.0, ge=0.0)
    orders: Optional[list[int]] = None
    harmonic_reference: Optional[Literal["en50160", "iec61000-3-2"]] = None
    emission_class: Literal["A", "B", "C", "D", "auto"] = "auto"

    @property
    def is_harmonic(self) -> bool:
        return self.field in ("h_mag", "h_phase")

    @property
    def is_source_voltage(self) -> bool:
        return self.field == "u_ref"

    @model_validator(mode="after")
    def _check(self) -> "ParameterSpec":
        # The slack-voltage field targets a Source only; it multiplies u_ref_v, so it is
        # scale-only and carries no per-phase / harmonic structure. Every other field
        # targets an injecting Load/Generator, so it may NOT select a source.
        if self.is_source_voltage:
            if self.selector.component != "source":
                raise ValueError("field='u_ref' requires selector.component='source'.")
            if self.mode != "scale":
                raise ValueError(
                    "field='u_ref' requires mode='scale' (it multiplies u_ref_v)."
                )
            if self.symmetry != "balanced":
                raise ValueError(
                    "field='u_ref' does not support per-phase symmetry "
                    "(the source reference scales all phases together)."
                )
            if self.orders is not None or self.harmonic_reference is not None:
                raise ValueError(
                    "field='u_ref' takes no `orders` / `harmonic_reference`."
                )
        elif self.selector.component == "source":
            raise ValueError("selector.component='source' supports only field='u_ref'.")
        if self.field == "pq" and self.mode != "scale":
            raise ValueError(
                "field='pq' requires mode='scale' (constant power factor)."
            )
        if self.correlation is not None and self.symmetry == "independent":
            raise ValueError(
                "correlation is incompatible with symmetry='independent' (there is no "
                "component-level value to correlate)."
            )
        if self.symmetry == "small_imbalance" and self.imbalance <= 0.0:
            raise ValueError("symmetry='small_imbalance' requires imbalance > 0.")
        if self.symmetry != "small_imbalance" and self.imbalance != 0.0:
            raise ValueError("imbalance is only used with symmetry='small_imbalance'.")
        if self.is_harmonic:
            if not self.orders:
                raise ValueError(f"field={self.field!r} requires a non-empty `orders`.")
            if any(o < 2 for o in self.orders):
                raise ValueError(
                    "harmonic `orders` must all be >= 2 (1 = fundamental)."
                )
            if self.symmetry != "balanced" or self.correlation is not None:
                raise ValueError(
                    "harmonic fields support neither per-phase `symmetry` nor "
                    "`correlation` (use the grid `spectrum_per_phase` for per-phase "
                    "distortion)."
                )
            if self.field == "h_phase" and self.mode != "absolute":
                raise ValueError("field='h_phase' requires mode='absolute'.")
            if self.harmonic_reference is not None and self.field != "h_mag":
                raise ValueError("harmonic_reference applies to field='h_mag' only.")
        else:
            if self.orders is not None or self.harmonic_reference is not None:
                raise ValueError(
                    "`orders` / `harmonic_reference` are only valid for harmonic fields."
                )
        if self.emission_class != "auto" and self.harmonic_reference != "iec61000-3-2":
            raise ValueError(
                "emission_class is only valid with harmonic_reference='iec61000-3-2'."
            )
        return self


class ScenarioConfig(_Base):
    """A full, reproducible RANDOM/QMC batch specification.

    ``method``: ``"sobol"`` (QMC, low-discrepancy — recommended for training data),
    ``"lhs"`` (Latin hypercube), or ``"independent"`` (plain seeded RNG).
    ``factors`` declares the shared :class:`LatentFactor` drivers referenced by any
    spec's ``correlation`` (each consumes one sampling dimension).
    """

    n_samples: int = Field(gt=0)
    seed: int = 0
    method: Literal["sobol", "lhs", "independent"] = "sobol"
    parameters: list[ParameterSpec]
    factors: list[LatentFactor] = Field(default_factory=list)


# =============================================================================
# Cartesian-product (grid-sweep) batches
# =============================================================================
class CartesianAxis(_Base):
    """One axis of a cartesian-product sweep: explicit discrete levels.

    Each level is applied (``scale``/``absolute``, like :class:`ParameterSpec`) to
    ALL components the selector matches. The batch is the cartesian product of all
    axes' levels (``B = prod(len(axis.values))``).
    """

    name: str
    selector: Selector
    values: list[float] = Field(min_length=1)
    field: Literal["p", "q", "pq"] = "pq"
    mode: Literal["scale", "absolute"] = "scale"


class CartesianConfig(_Base):
    """A reproducible (deterministic, no RNG) cartesian-product batch (pgm-style)."""

    axes: list[CartesianAxis] = Field(min_length=1)


# =============================================================================
# Per-target structured perturbation sweep (inject one error per node)
# =============================================================================
class Perturbation(_Base):
    """One injected error swept across targets by :func:`perturbation_sweep`.

    The sweep builds ``B = #targets`` scenarios, each perturbing exactly ONE selected
    target's operating point (all other targets nominal) — the "inject an error at each
    node and measure how it spreads" use case. The ground truth is recorded as
    :class:`~pgml.schemas.scenario_schema.ParameterPerturbation` rows.

    - ``field``: the operating-point quantity perturbed — ``"p"`` / ``"q"`` (one) or
      ``"pq"`` (both at constant power factor; requires ``mode="scale"``).
    - ``mode``: ``"scale"`` (× ``value``), ``"delta"`` (+ ``value``, an absolute Δ in
      W / var), or ``"set"`` (= ``value``).
    - ``value``: the perturbation magnitude.
    """

    name: str = "perturbation"
    field: Literal["p", "q", "pq"] = "pq"
    mode: Literal["scale", "delta", "set"] = "scale"
    value: float

    @model_validator(mode="after")
    def _check(self) -> "Perturbation":
        if self.field == "pq" and self.mode != "scale":
            raise ValueError("Perturbation field='pq' requires mode='scale'.")
        return self


class SpectrumSweepConfig(_Base):
    """Per-target harmonic-injection sweep: scenario *i* injects ``spectrum`` at target
    *i* only (all others silent). The harmonic analogue of :class:`Perturbation` /
    ``perturbation_sweep`` — a diagonal one-hot enumeration over the selector's matched
    devices (``B = #targets``), for mapping how a single injected spectrum spreads.

    The spectrum is stored as parallel ``orders`` / ``magnitudes_pu`` / ``phases_deg``
    lists (serializable). Magnitudes are RELATIVE to the device's fundamental injection
    (the ``harmonic_injection`` convention); order 1 is the implicit reference and must
    NOT be listed. Build one ergonomically from a ``{order: (mag_pu, phase_deg)}`` dict
    via :meth:`from_spectrum`.
    """

    name: str = "injection"
    selector: Selector
    orders: list[int] = Field(min_length=1)
    magnitudes_pu: list[float]
    phases_deg: list[float]

    @classmethod
    def from_spectrum(
        cls, selector: Selector, spectrum: dict, *, name: str = "injection"
    ) -> "SpectrumSweepConfig":
        """Build from a ``{order: (magnitude_pu, phase_deg)}`` dict (order 1 dropped)."""
        orders = sorted(int(o) for o in spectrum if int(o) >= 2)
        return cls(
            name=name,
            selector=selector,
            orders=orders,
            magnitudes_pu=[float(spectrum[o][0]) for o in orders],
            phases_deg=[float(spectrum[o][1]) for o in orders],
        )

    @model_validator(mode="after")
    def _check(self) -> "SpectrumSweepConfig":
        if not (len(self.orders) == len(self.magnitudes_pu) == len(self.phases_deg)):
            raise ValueError(
                "orders / magnitudes_pu / phases_deg must have equal length."
            )
        if any(o < 2 for o in self.orders):
            raise ValueError("harmonic `orders` must all be >= 2 (1 = fundamental).")
        return self


class NodeInjectionSweepConfig(_Base):
    """Per-node harmonic "error" SOURCE sweep: inject a transient harmonic source at one
    node at a time (scenario ``i`` → node ``i``; ``B = #nodes``). Unlike
    :class:`SpectrumSweepConfig` (a device Norton current scaled by a load's fundamental),
    this is the per-node Thévenin/Norton source of ``docs/pgml/modeling/error-injection.md`` —
    injectable at ANY node, of a user-set STRENGTH ``source_power_va`` (S_sc), applied
    only at h>1 (fundamental exact).

    Spectrum stored as parallel ``orders``/``magnitudes_pu``/``phases_deg`` (order 1 is
    the implicit reference); build from a ``{order: (mag_pu, phase_deg)}`` dict via
    :meth:`from_spectrum`.
    """

    name: str = "injection"
    node_ids: Optional[list[int]] = None  # None = every node in the grid
    phases: Optional[list[Phase]] = None  # None = all phases of each node
    orders: list[int] = Field(min_length=1)
    magnitudes_pu: list[float]
    phases_deg: list[float]
    source_power_va: float = Field(gt=0.0)
    kind: Literal["voltage", "current"] = "voltage"

    @classmethod
    def from_spectrum(
        cls,
        spectrum: dict,
        *,
        source_power_va: float,
        kind: str = "voltage",
        node_ids: Optional[list[int]] = None,
        phases: Optional[list[Phase]] = None,
        name: str = "injection",
    ) -> "NodeInjectionSweepConfig":
        """Build from a ``{order: (magnitude_pu, phase_deg)}`` dict (order 1 dropped)."""
        orders = sorted(int(o) for o in spectrum if int(o) >= 2)
        return cls(
            name=name,
            node_ids=node_ids,
            phases=phases,
            orders=orders,
            magnitudes_pu=[float(spectrum[o][0]) for o in orders],
            phases_deg=[float(spectrum[o][1]) for o in orders],
            source_power_va=source_power_va,
            kind=kind,
        )

    @model_validator(mode="after")
    def _check(self) -> "NodeInjectionSweepConfig":
        if not (len(self.orders) == len(self.magnitudes_pu) == len(self.phases_deg)):
            raise ValueError(
                "orders / magnitudes_pu / phases_deg must have equal length."
            )
        if any(o < 2 for o in self.orders):
            raise ValueError("harmonic `orders` must all be >= 2 (1 = fundamental).")
        return self


# =============================================================================
# Time-varying fundamental load profiles (multi-scale synthetic recurrence)
# =============================================================================
class LoadProfileConfig(_Base):
    """Multi-scale synthetic load-profile generator for a coherent step sequence.

    Turns the constant per-scenario fundamental of a
    :class:`CoherentSpectrumConfig` into a per-STEP time series: every matched
    device gets a multiplicative factor composed on four time scales,

        ``factor(t) = f_seasonal(t) * f_weekly(t) * f_daily(t) * f_short(t)``,

    applied to the device's per-scenario base ``P`` and ``Q`` TOGETHER (constant
    power factor, the ``field="pq"`` semantics). The base is the device's sampled
    per-scenario operating point (from ``CoherentSpectrumConfig.parameters``) if it
    has one, else its nominal ``p_nom_w`` / ``q_nom_var``. Gradients still flow to a
    tensor-valued base (the factor is a plain, off-tape multiplier).

    The daily SHAPE is class-aware, keyed by each device's ``consumer_type``
    (household evening peak, office/commercial business hours, EV evening charging,
    a flat industrial plateau, and a neutral default). ``pv`` is special: a solar
    bell that is ZERO at night, with a daylight window and amplitude that widen in
    summer (the seasonal modulation folds into the daily bell, not ``f_seasonal``).
    The concrete shapes live in :mod:`pgml.scenarios.profiles`; this config sets
    their amplitudes and the stochastic ranges.

    Correlation model (per-scenario co-variation)
    ---------------------------------------------
    Two per-scenario SHARED latents make devices co-vary within a scenario:

    - a ``behavioral`` latent scales the DAILY AMPLITUDE of every non-``pv`` device
      (a busy day lifts everyone's daily swing together), coupling strength
      ``behavioral_coupling``;
    - a ``cloudiness`` latent scales every ``pv`` device's output together
      (an overcast day dims all panels), coupling strength ``cloud_coupling``.

    On top of the shared latents each device draws IDIOSYNCRATIC per-scenario values:
    an overall level (``level_min`` .. ``level_max``), an amplitude multiplier
    (``amplitude_jitter_min`` .. ``amplitude_jitter_max``), a daily phase offset
    (``+/- phase_offset_hours``, so devices do not all peak at the same instant), and
    a per-step AR(1) short-term term (``short_rho`` stickiness, ``short_sigma``
    std). The composed factor is clamped to be non-negative.

    The generator draws on an RNG stream DERIVED from ``CoherentSpectrumConfig.seed``
    with its own offset, DISTINCT from the fingerprint / Markov / AR(1) jitter and the
    operating-point cube — so enabling a profile leaves the harmonic fingerprint and
    the raw parameter draws byte-identical.
    """

    # Devices to profile: None = every in-service Load and Generator in the grid.
    selector: Optional[Selector] = None

    # Overall daily-shape depth (per-class base depths live in ``profiles``).
    daily_amplitude: float = Field(default=1.0, ge=0.0)

    # Per-scenario, per-device idiosyncratic draws.
    level_min: float = Field(default=0.85, gt=0.0)
    level_max: float = Field(default=1.15, gt=0.0)
    amplitude_jitter_min: float = Field(default=0.8, ge=0.0)
    amplitude_jitter_max: float = Field(default=1.2, ge=0.0)
    phase_offset_hours: float = Field(default=1.0, ge=0.0)

    # Per-scenario SHARED latents (co-variation across devices).
    behavioral_coupling: float = Field(default=0.3, ge=0.0)  # non-pv daily amplitude
    cloud_coupling: float = Field(default=0.5, ge=0.0)  # pv output

    # Weekly weekday/weekend contrast (consumption; pv is unaffected).
    weekend_contrast: float = Field(default=0.15)

    # Seasonal modulation (annual sinusoid). ``*_peak_doy`` is a day-of-year phase.
    seasonal_amplitude: float = Field(default=0.15, ge=0.0)  # consumption swing
    seasonal_peak_doy: float = Field(default=15.0)  # consumption peaks in winter
    pv_seasonal_amplitude: float = Field(default=0.4, ge=0.0)  # pv output swing
    pv_seasonal_peak_doy: float = Field(default=172.0)  # summer solstice
    pv_daylight_hours: float = Field(default=12.0, gt=0.0)  # mean day length
    pv_daylight_swing: float = Field(default=4.0, ge=0.0)  # +/- seasonal day-length

    # Short-term stochastic term (AR(1), per device per step).
    short_rho: float = Field(default=0.9, ge=0.0, le=1.0)
    short_sigma: float = Field(default=0.05, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> "LoadProfileConfig":
        if self.level_max < self.level_min:
            raise ValueError("level_max must be >= level_min.")
        if self.amplitude_jitter_max < self.amplitude_jitter_min:
            raise ValueError("amplitude_jitter_max must be >= amplitude_jitter_min.")
        return self


# =============================================================================
# Statistical device-class composition of aggregated loads
# =============================================================================
#: Diurnal activity-rate presets a device class may follow. They REFERENCE the
#: class-aware daily shapes in :mod:`pgml.scenarios.profiles` (household evening,
#: office hours, EV evening, restaurant lunch/dinner, industrial plateau, the ``pv``
#: solar bell) plus a ``"flat"`` constant availability.
ActivityPreset = Literal[
    "household", "office", "ev", "restaurant", "industrial", "pv", "flat"
]


class DeviceState(_Base):
    """One operating state of a MULTI-STATE device class.

    Captures the washing-machine / white-goods pattern where the SAME appliance draws
    very different current AND injects a very different spectrum depending on its cycle
    phase — e.g. a high-power, near-linear resistive HEATING state versus a low-power,
    harmonic-rich INVERTER-driven spinning state. The device jumps between its states on
    a Markov chain (dwell ``DeviceClassSpec.state_dwell``); the stationary probability of
    a state is proportional to its ``weight``.

    - ``power_fraction`` — the state's loading as a fraction of the device's rated power
      (drives BOTH the drawn power and, through the ``gamma`` law, the harmonic
      magnitude).
    - ``spectrum_scale`` — an ADDITIONAL multiplier on the device's per-order harmonic
      magnitudes in this state, capturing the qualitative spectral change that is NOT a
      function of power magnitude alone (heating ≈ clean, inverter ≈ rich).
    """

    name: str = "state"
    power_fraction: float = Field(gt=0.0, le=1.0)
    spectrum_scale: float = Field(default=1.0, ge=0.0)
    weight: float = Field(default=1.0, gt=0.0)


class DeviceClassSpec(_Base):
    """A STATISTICAL device class — a distribution over one appliance's behaviour.

    The goal is NOT an accurate appliance model but plausible statistical coverage: an
    aggregated load is composed of several members drawn from these classes (see
    :class:`ConsumerComposition`), and a state estimator learns to attribute an observed
    aggregate spectrum to a class mix. Every range below is drawn PER MEMBER (once, at
    roster build); the temporal draws (activity, loading, state) then evolve per step.

    The per-order harmonic magnitude ranges are FRACTIONS of the device's OWN fundamental
    current at rated load and are only LOOSELY bounded by the IEC 61000-3-2 emission
    shape (odd-dominated, decreasing with order) — they are a modelling convenience for
    generating diverse training data, NOT the standard's per-appliance limits.

    Load dependence (the measured reality that a device's harmonic signature depends on
    how hard it is driven):

    - magnitude ``mag_h(lam) = mag_h_rated * lam ** gamma_h`` with ``gamma_h`` drawn per
      member per order from ``gamma`` (``gamma = 0`` → constant ratio; ``gamma < 0`` →
      the THD FRACTION falls as load rises while the absolute harmonic current still
      grows — the measured EV-charger / PV-inverter behaviour);
    - phase ``ang_h(lam) = ang_h0 + s_h * (lam - 1)`` with ``s_h`` [deg] drawn per member
      per order from ``phase_slope_deg``.

    Activity: ``activity_preset`` is a diurnal availability shape (see
    :data:`ActivityPreset`). ``discrete_activity`` selects how it is realised — a
    switching appliance (``True``, the default) follows a two-state on/off Markov chain
    whose target occupancy tracks the diurnal rate (stickiness drawn from
    ``on_off_dwell``); a continuously-modulated device (``False``, e.g. a PV inverter
    tracking irradiance, a background base load tracking occupancy) uses the diurnal rate
    itself as a fractional availability in ``[0, 1]``.
    """

    name: str
    #: ``+1`` consuming (Load-like), ``-1`` injecting (PV-like).
    sign: Literal[1, -1] = 1
    #: Rated active power per instance ``[low, high]`` in watts (magnitude).
    rated_power_w: tuple[float, float]
    #: Fundamental displacement power factor ``lambda`` (``Q = P * tan(acos(pf))``).
    power_factor: float = Field(default=1.0, gt=0.0, le=1.0)

    #: Per-order harmonic magnitude ranges at RATED load, as a FRACTION of the device's
    #: own fundamental current: ``{order: [low, high]}`` (orders >= 2; absent = no
    #: emission at that order). Bounded loosely by the IEC 61000-3-2 shape.
    harmonic_magnitude: dict[int, tuple[float, float]] = Field(default_factory=dict)
    #: Per-order harmonic phase ranges [deg] at rated load: ``{order: [low, high]}``.
    harmonic_phase_deg: dict[int, tuple[float, float]] = Field(default_factory=dict)
    #: Range for the per-order magnitude load-dependence exponent ``gamma_h``.
    gamma: tuple[float, float] = (0.0, 0.0)
    #: Range for the per-order phase load-dependence slope ``s_h`` [deg per unit load].
    phase_slope_deg: tuple[float, float] = (0.0, 0.0)

    #: Diurnal availability shape.
    activity_preset: ActivityPreset = "flat"
    #: ``True`` → a switching appliance (on/off Markov); ``False`` → continuous
    #: availability equal to the diurnal rate (PV irradiance, background base load).
    discrete_activity: bool = True
    #: Range for the on/off Markov stickiness (per-step probability of persisting).
    on_off_dwell: tuple[float, float] = (0.85, 0.98)

    #: Loading floor ``lam_min`` (the per-step loading is clamped to ``[lam_min, 1]``).
    loading_min: float = Field(default=0.2, gt=0.0, le=1.0)
    #: Range for the per-member mean loading (single-state classes).
    loading_mean: tuple[float, float] = (0.6, 1.0)
    #: Fractional std of the AR(1) per-step loading jitter.
    loading_jitter: float = Field(default=0.05, ge=0.0)
    #: AR(1) stickiness of the per-step loading jitter.
    loading_rho: float = Field(default=0.85, ge=0.0, le=1.0)

    #: Optional multi-state operation (empty = single-state, continuous loading).
    states: list[DeviceState] = Field(default_factory=list)
    #: Range for the multi-state Markov dwell (per-step probability of staying in state).
    state_dwell: tuple[float, float] = (0.85, 0.97)

    @model_validator(mode="after")
    def _check(self) -> "DeviceClassSpec":
        lo, hi = self.rated_power_w
        if lo <= 0.0 or hi < lo:
            raise ValueError(
                f"class {self.name!r} rated_power_w must be 0 < low <= high."
            )
        for field_name in ("harmonic_magnitude", "harmonic_phase_deg"):
            for order in getattr(self, field_name):
                if order < 2:
                    raise ValueError(
                        f"class {self.name!r} {field_name} orders must be >= 2."
                    )
        if not (
            self.loading_min <= self.loading_mean[0] <= self.loading_mean[1] <= 1.0
        ):
            raise ValueError(
                f"class {self.name!r} requires loading_min <= loading_mean[0] "
                "<= loading_mean[1] <= 1."
            )
        for lo_, hi_ in (
            self.gamma,
            self.phase_slope_deg,
            self.on_off_dwell,
            self.state_dwell,
        ):
            if hi_ < lo_:
                raise ValueError(f"class {self.name!r} has a range with high < low.")
        return self


class ClassCount(_Base):
    """How many instances of a device class an aggregated load contains.

    ``count`` is an inclusive ``[min, max]`` integer range; ``power_share`` weights this
    class's instances when :class:`CompositionConfig` scales the roster to the load's
    nominal power (a larger share claims a larger slice of the nameplate).
    """

    class_name: str
    count: tuple[int, int] = (1, 1)
    power_share: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> "ClassCount":
        lo, hi = self.count
        if lo < 0 or hi < lo:
            raise ValueError(
                f"ClassCount {self.class_name!r} count must be 0 <= min <= max."
            )
        return self


class ConsumerComposition(_Base):
    """A composition rule: the device-class roster of one consumer character.

    A rule matches an aggregated load by explicit ``load_ids`` (a per-appliance override
    — e.g. a public charging station is just an EV charger) if given, else by
    ``consumer_type`` (matched against the load's own ``consumer_type``); a rule with
    both ``load_ids`` and ``consumer_type`` unset is the fallback for any load no other
    rule claims. ``classes`` lists the device classes and their instance counts.
    """

    consumer_type: Optional[str] = None
    load_ids: Optional[list[int]] = None
    classes: list[ClassCount] = Field(min_length=1)


def default_device_classes() -> list[DeviceClassSpec]:
    """The built-in statistical device-class library (every range user-overridable).

    Six plausible LV device characters: a harmonic-free linear base load, an
    SMPS-electronics class (3rd/5th-dominated), an EV charger and a PV inverter (both
    with the measured falling-THD-fraction-with-load behaviour, ``gamma < 0``; PV
    injecting), a multi-state inverter drive (white goods / heat pump: a near-linear
    heating state vs a harmonic-rich inverter state), and a thermostatic resistive
    heater. Magnitudes are loosely IEC 61000-3-2-shaped plausibility, NOT appliance
    models.
    """
    return [
        DeviceClassSpec(
            name="base_linear",
            sign=1,
            rated_power_w=(150.0, 1500.0),
            activity_preset="household",
            discrete_activity=False,
            loading_min=0.3,
            loading_mean=(0.6, 1.0),
        ),
        DeviceClassSpec(
            name="electronics_smps",
            sign=1,
            rated_power_w=(20.0, 400.0),
            harmonic_magnitude={
                3: (0.5, 0.85),
                5: (0.25, 0.6),
                7: (0.1, 0.4),
                9: (0.05, 0.25),
                11: (0.03, 0.15),
                13: (0.02, 0.1),
            },
            harmonic_phase_deg={
                3: (-30.0, 30.0),
                5: (-60.0, 60.0),
                7: (-90.0, 90.0),
                9: (-120.0, 120.0),
                11: (-150.0, 150.0),
                13: (-180.0, 180.0),
            },
            gamma=(-0.2, 0.1),
            phase_slope_deg=(-20.0, 20.0),
            activity_preset="household",
            on_off_dwell=(0.5, 0.8),
            loading_min=0.1,
            loading_mean=(0.3, 0.9),
        ),
        DeviceClassSpec(
            name="ev_charger",
            sign=1,
            rated_power_w=(3700.0, 11000.0),
            harmonic_magnitude={
                3: (0.01, 0.05),
                5: (0.02, 0.08),
                7: (0.01, 0.05),
                9: (0.005, 0.03),
                11: (0.005, 0.02),
            },
            harmonic_phase_deg={
                3: (-40.0, 40.0),
                5: (-40.0, 40.0),
                7: (-60.0, 60.0),
                9: (-90.0, 90.0),
                11: (-120.0, 120.0),
            },
            gamma=(-1.6, -0.7),
            phase_slope_deg=(-15.0, 15.0),
            activity_preset="ev",
            on_off_dwell=(0.6, 0.82),
            loading_min=0.2,
            loading_mean=(0.7, 1.0),
        ),
        DeviceClassSpec(
            name="pv_inverter",
            sign=-1,
            rated_power_w=(1000.0, 8000.0),
            harmonic_magnitude={
                3: (0.01, 0.04),
                5: (0.02, 0.07),
                7: (0.01, 0.05),
                9: (0.005, 0.03),
            },
            harmonic_phase_deg={
                3: (-60.0, 60.0),
                5: (-60.0, 60.0),
                7: (-90.0, 90.0),
                9: (-120.0, 120.0),
            },
            gamma=(-1.3, -0.6),
            phase_slope_deg=(-25.0, 25.0),
            activity_preset="pv",
            discrete_activity=False,
            loading_min=0.05,
            loading_mean=(0.5, 1.0),
        ),
        DeviceClassSpec(
            name="inverter_drive",
            sign=1,
            rated_power_w=(500.0, 3000.0),
            harmonic_magnitude={
                3: (0.08, 0.25),
                5: (0.15, 0.45),
                7: (0.08, 0.3),
                9: (0.03, 0.15),
                11: (0.02, 0.1),
            },
            harmonic_phase_deg={
                3: (-45.0, 45.0),
                5: (-60.0, 60.0),
                7: (-90.0, 90.0),
                9: (-120.0, 120.0),
                11: (-150.0, 150.0),
            },
            gamma=(-0.4, 0.2),
            phase_slope_deg=(-30.0, 30.0),
            activity_preset="household",
            on_off_dwell=(0.6, 0.85),
            loading_min=0.15,
            loading_mean=(0.4, 1.0),
            states=[
                DeviceState(
                    name="heating", power_fraction=0.95, spectrum_scale=0.2, weight=0.5
                ),
                DeviceState(
                    name="inverter", power_fraction=0.35, spectrum_scale=1.6, weight=0.5
                ),
            ],
            state_dwell=(0.85, 0.97),
        ),
        DeviceClassSpec(
            name="resistive_heating",
            sign=1,
            rated_power_w=(500.0, 3000.0),
            activity_preset="flat",
            on_off_dwell=(0.5, 0.75),
            loading_min=0.9,
            loading_mean=(0.95, 1.0),
            loading_jitter=0.02,
        ),
    ]


def default_compositions() -> list[ConsumerComposition]:
    """The built-in per-``consumer_type`` composition rules (over the default library).

    Every count / share is user-overridable. The rule with no ``consumer_type`` /
    ``load_ids`` is the fallback for any unmatched load.
    """

    def cc(name: str, lo: int, hi: int, share: float = 1.0) -> ClassCount:
        return ClassCount(class_name=name, count=(lo, hi), power_share=share)

    return [
        ConsumerComposition(
            consumer_type="household",
            classes=[
                cc("base_linear", 1, 1, 3.0),
                cc("electronics_smps", 1, 3, 1.0),
                cc("resistive_heating", 0, 1, 2.0),
                cc("inverter_drive", 0, 1, 1.5),
                cc("ev_charger", 0, 1, 1.0),
                cc("pv_inverter", 0, 1, 1.0),
            ],
        ),
        ConsumerComposition(
            consumer_type="office",
            classes=[
                cc("base_linear", 1, 1, 4.0),
                cc("electronics_smps", 3, 10, 1.5),
                cc("inverter_drive", 0, 2, 2.0),
            ],
        ),
        ConsumerComposition(
            consumer_type="restaurant",
            classes=[
                cc("base_linear", 1, 1, 3.0),
                cc("inverter_drive", 1, 3, 2.0),
                cc("electronics_smps", 1, 4, 1.0),
            ],
        ),
        ConsumerComposition(
            consumer_type="heat_pump",
            classes=[cc("base_linear", 1, 1, 1.0), cc("inverter_drive", 1, 2, 3.0)],
        ),
        ConsumerComposition(
            consumer_type="workshop",
            classes=[
                cc("base_linear", 1, 1, 3.0),
                cc("inverter_drive", 1, 3, 2.5),
                cc("electronics_smps", 1, 4, 1.0),
                cc("resistive_heating", 0, 1, 1.0),
            ],
        ),
        ConsumerComposition(
            consumer_type="ev_charging",
            classes=[cc("base_linear", 0, 1, 0.5), cc("ev_charger", 1, 2, 1.0)],
        ),
        ConsumerComposition(
            consumer_type="pv",
            classes=[cc("pv_inverter", 1, 1, 1.0)],
        ),
        ConsumerComposition(
            classes=[cc("base_linear", 1, 1, 3.0), cc("electronics_smps", 1, 2, 1.0)],
        ),
    ]


class CompositionConfig(_Base):
    """Statistical device-class composition of aggregated loads.

    Set on :class:`CoherentSpectrumConfig`. When present it SUPERSEDES the mode-bank
    fingerprint machinery for the loads it covers: each such load becomes a sum of
    statistical member devices (drawn from ``classes`` per the matching
    :class:`ConsumerComposition` in ``compositions``), whose per-step activity drives
    BOTH the drawn fundamental power AND the injected harmonic spectrum — so the dataset
    carries a consistent load-to-spectrum mapping a model can learn from (and, in the
    best case, use to attribute an observed spectrum to a device mix / an error source).

    A load covered by the composition draws its fundamental P/Q and its harmonic
    injection ENTIRELY from the composition; any ``parameters`` / ``profile`` targeting
    the same load is superseded. Loads NOT covered (no matching rule, or outside
    ``selector``) keep the fingerprint / profile behaviour.

    Roster: per covered load a device roster is drawn once (seeded, persisted). With
    ``scale_to_nominal`` the members' rated powers are rescaled so their share-weighted
    installed capacity equals the load's ``p_nom_w`` (keeping the nameplate meaningful
    across grids); otherwise the class rated ranges are absolute. Cross-device
    correlation reuses the profile latents: one ``behavioral`` latent scales every
    consumption activity together, one ``cloud`` latent scales every PV together.

    Guard: near a net-zero aggregate fundamental (e.g. PV cancelling load) the RELATIVE
    harmonic magnitude explodes (a physically real residual-THD effect); it is capped at
    ``max_injection_pu`` and the binding is recorded in the samples.
    """

    #: Loads to compose. ``None`` = every in-service Load. A load covered here but
    #: without a matching rule in ``compositions`` falls through to the fingerprint.
    selector: Optional[Selector] = None
    #: The device-class library.
    classes: list[DeviceClassSpec] = Field(default_factory=default_device_classes)
    #: The per-consumer-type composition rules.
    compositions: list[ConsumerComposition] = Field(
        default_factory=default_compositions
    )
    #: Rescale each roster's share-weighted installed capacity to the load's ``p_nom_w``.
    scale_to_nominal: bool = True
    #: Cap on the aggregate harmonic magnitude [pu of the aggregate fundamental current].
    max_injection_pu: float = Field(default=3.0, gt=0.0)
    #: Cross-device correlation strengths (shared per-scenario latents).
    behavioral_coupling: float = Field(default=0.3, ge=0.0)
    cloud_coupling: float = Field(default=0.5, ge=0.0)
    #: Optional distinct seed for the roster + temporal draws (a held-out composition
    #: bank). ``None`` derives both streams from ``CoherentSpectrumConfig.seed``.
    roster_seed: Optional[int] = None

    def class_names(self) -> list[str]:
        """Ordered device-class names (the ``n_class`` axis of the recorded samples)."""
        return [c.name for c in self.classes]

    @model_validator(mode="after")
    def _check(self) -> "CompositionConfig":
        names = [c.name for c in self.classes]
        if len(names) != len(set(names)):
            raise ValueError("CompositionConfig class names must be unique.")
        known = set(names)
        for rule in self.compositions:
            for cc in rule.classes:
                if cc.class_name not in known:
                    raise ValueError(
                        f"ConsumerComposition references unknown class "
                        f"{cc.class_name!r} (not in classes)."
                    )
        return self


# =============================================================================
# Node-coherent harmonic "fingerprint" sampling (temporal sequences)
# =============================================================================
class CoherentSpectrumConfig(_Base):
    """Node-coherent harmonic sampling: a stable per-device fingerprint over a sequence.

    Each matched device draws a small set of base spectra (``n_modes`` "modes" — e.g.
    appliance operating states like a washing machine heating vs spinning), drawn once
    (or per scenario). Over ``n_steps`` consecutive steps it STICKS to a mode (Markov
    dwell ``dwell``) and WANDERS around it (AR(1) jitter with stickiness ``ar1_rho``),
    clamped to the per-order emission reference (``harmonic_reference``). This yields a
    ``[B, T]`` batch of harmonic injections in which each node keeps a recognisable
    signature that varies realistically — so a state estimator can attribute the pattern
    to the node.

    The result voltages are ``[B, T, H, N]`` (B = ``n_scenarios`` sequences, T = steps);
    per-step timestamps are recorded as ``samples["time_s"]``.

    ``harmonic_reference`` selects the per-order magnitude reference (and the upper
    clamp): the default ``"iec61000-3-2"`` is the IEC 61000-3-2 appliance harmonic
    CURRENT-emission standard — the physically correct per-device reference for a device
    current fingerprint, keyed by ``emission_class`` (``"auto"`` resolves per device from
    its ``consumer_type`` and nominal power). ``"en50160"`` instead SHAPES the fingerprint
    by the DIN EN 50160 supply-VOLTAGE compatibility levels (kept for
    background-distortion-shaped experiments and backward compatibility; it is NOT an
    appliance emission model). ``None`` treats magnitudes as absolute pu (clamped to 1.0).

    Held-out test set recipe: reproducibility hangs on both ``seed`` (the temporal
    Markov + AR(1) stream) and ``mode_bank_seed`` (the per-device fingerprint bank). For
    an unseen-FINGERPRINT test set, give a DISTINCT ``mode_bank_seed`` (a different
    device signature bank while every other setting is shared); combine it with a
    distinct ``seed`` for an entirely independent temporal realization too. Leaving
    ``mode_bank_seed=None`` draws the bank from the ``seed`` stream (the default,
    byte-identical to a config with no ``mode_bank_seed`` set).

    ``parameters`` / ``factors`` add a FUNDAMENTAL operating-point variation on top of the
    harmonic fingerprint: the same :class:`ParameterSpec` / :class:`LatentFactor` machinery
    as :class:`ScenarioConfig`, but drawn ONCE PER SCENARIO (shape ``[B]``, held constant
    across the ``T`` steps and broadcast in the solve). So a coherent sequence may vary the
    load level, PV output, and the slack voltage (``field="u_ref"``) per scenario while the
    per-step spectrum keeps its device fingerprint. Harmonic ``ParameterSpec`` fields
    (``h_mag`` / ``h_phase``) are REJECTED here — the fingerprint machinery owns the
    harmonics; ``parameters`` shapes only the fundamental. The draws are sampled on a unit
    cube seeded from a stream DISTINCT from the fingerprint RNG, so the realized
    ``harmonic_injection`` is byte-identical with and without ``parameters``. When
    ``parameters`` is empty (the default) the fundamental P/Q stays nominal and the source
    at ``u_ref_v`` — the original fingerprint-only behavior.

    ``profile`` (a :class:`LoadProfileConfig`) makes the fundamental P/Q TIME-VARYING
    across the ``T`` steps instead of constant: each device's per-scenario base P/Q (from
    ``parameters`` if set, else nominal) is multiplied by a multi-scale synthetic
    profile factor (seasonal / weekly / daily / short-term, class-aware by
    ``consumer_type``). The operating point then carries the step axis (``[B, T]``),
    aligned with the ``[B, T]`` harmonic injection, so the solve yields ``[B, T, H, N]``
    with a moving fundamental. ``start_time`` (ISO 8601) is REQUIRED when ``profile`` is
    set — the daily / weekly / seasonal phases need an absolute anchor. Because the
    harmonic injection magnitude is RELATIVE to the device's fundamental current, a
    profile-scaled fundamental already scales the absolute harmonic current; no extra
    coupling is applied. ``profile=None`` (the default) leaves the fundamental constant
    over the sequence — byte-identical to the fingerprint-only behavior. The profile is
    drawn on an RNG stream distinct from the fingerprint, Markov path, AR(1) jitter, and
    the operating-point cube, so enabling it leaves the harmonic fingerprint unchanged.
    """

    name: str = "harmonics"
    selector: Selector
    orders: list[int] = Field(min_length=1)
    n_steps: int = Field(gt=0)  # T
    n_scenarios: int = Field(default=1, gt=0)  # B
    n_modes: int = Field(default=2, ge=1)
    seed: int = 0
    mag_distribution: Distribution = Field(
        default_factory=lambda: Uniform(low=0.0, high=1.0)
    )
    harmonic_reference: Optional[Literal["en50160", "iec61000-3-2"]] = "iec61000-3-2"
    emission_class: Literal["A", "B", "C", "D", "auto"] = "auto"
    phase_distribution: Distribution = Field(
        default_factory=lambda: Uniform(low=-180.0, high=180.0)
    )
    jitter_mag: float = Field(default=0.05, ge=0.0)  # AR(1) fractional std on magnitude
    jitter_phase_deg: float = Field(default=5.0, ge=0.0)  # AR(1) std on phase (deg)
    ar1_rho: float = Field(default=0.8, ge=0.0, le=1.0)  # temporal stickiness of jitter
    dwell: float = Field(default=0.9, ge=0.0, le=1.0)  # P(stay in mode) per step
    step_size_s: float = Field(default=1.0, gt=0.0)
    resample_modes_per_scenario: bool = False
    # Seeds ONLY the per-device fingerprint (mode) bank; None draws it from `seed`.
    mode_bank_seed: Optional[int] = None
    # Per-scenario fundamental operating-point variation (constant across the T steps).
    parameters: list[ParameterSpec] = Field(default_factory=list)
    factors: list[LatentFactor] = Field(default_factory=list)
    # Time-varying fundamental profile (per-step P/Q). Requires ``start_time``.
    profile: Optional[LoadProfileConfig] = None
    # Statistical device-class composition of aggregated loads (supersedes the
    # fingerprint + fundamental for the loads it covers). Requires ``start_time``.
    composition: Optional[CompositionConfig] = None
    # Absolute anchor for the profile's daily / weekly / seasonal phases (ISO 8601).
    # A naive (timezone-less) timestamp is interpreted as UTC.
    start_time: Optional[str] = None

    @model_validator(mode="after")
    def _check(self) -> "CoherentSpectrumConfig":
        if any(o < 2 for o in self.orders):
            raise ValueError("harmonic `orders` must all be >= 2 (1 = fundamental).")
        if self.emission_class != "auto" and self.harmonic_reference != "iec61000-3-2":
            raise ValueError(
                "emission_class is only valid with harmonic_reference='iec61000-3-2'."
            )
        for spec in self.parameters:
            if spec.is_harmonic:
                raise ValueError(
                    f"CoherentSpectrumConfig.parameters spec {spec.name!r} is harmonic "
                    f"(field={spec.field!r}); the fingerprint owns the harmonic spectrum. "
                    "parameters may vary only the fundamental (p/q/pq/u_ref)."
                )
        if self.profile is not None and self.start_time is None:
            raise ValueError(
                "CoherentSpectrumConfig.profile requires start_time (ISO 8601): the "
                "daily / weekly / seasonal phases need an absolute anchor."
            )
        if self.composition is not None and self.start_time is None:
            raise ValueError(
                "CoherentSpectrumConfig.composition requires start_time (ISO 8601): "
                "the device activity model is temporal (diurnal activity rates need an "
                "absolute anchor)."
            )
        if self.start_time is not None:
            from datetime import datetime

            try:
                datetime.fromisoformat(self.start_time)
            except ValueError as exc:
                raise ValueError(
                    f"start_time {self.start_time!r} is not a valid ISO 8601 "
                    "timestamp (e.g. '2024-06-21T00:00:00')."
                ) from exc
        return self


__all__ = [
    "Uniform",
    "Normal",
    "LogNormal",
    "LogUniform",
    "Constant",
    "Distribution",
    "Selector",
    "LatentFactor",
    "Correlation",
    "ParameterSpec",
    "ScenarioConfig",
    "CartesianAxis",
    "CartesianConfig",
    "CoherentSpectrumConfig",
    "LoadProfileConfig",
    "DeviceState",
    "DeviceClassSpec",
    "ClassCount",
    "ConsumerComposition",
    "CompositionConfig",
    "default_device_classes",
    "default_compositions",
    "Perturbation",
    "SpectrumSweepConfig",
    "NodeInjectionSweepConfig",
]


def _example() -> "ScenarioConfig":
    """A representative, valid batch: load P/Q scaling + EN 50160-referenced harmonics."""
    return ScenarioConfig(
        n_samples=256,
        seed=0,
        method="sobol",
        parameters=[
            ParameterSpec(
                name="load_pq",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.5, high=1.5),
                field="pq",
                mode="scale",
            ),
            ParameterSpec(
                name="harmonic_injection",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.0, high=1.0),
                field="h_mag",
                mode="absolute",
                orders=[3, 5, 7],
                harmonic_reference="en50160",
            ),
        ],
    )


if (
    __name__ == "__main__"
):  # `python -m pgml.scenarios.config --json-schema | --example`
    import argparse
    import json
    import sys

    import yaml

    ap = argparse.ArgumentParser(
        prog="python -m pgml.scenarios.config",
        description="Inspect the pgml ScenarioConfig: its JSON Schema or an example YAML.",
    )
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--json-schema", action="store_true", help="Print the config JSON Schema."
    )
    grp.add_argument(
        "--example", action="store_true", help="Print a valid example config as YAML."
    )
    ns = ap.parse_args()
    if ns.json_schema:
        json.dump(ScenarioConfig.model_json_schema(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(yaml.safe_dump(_example().model_dump(), sort_keys=False))
