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
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from pgml.schemas.grid_schema import Phase

if TYPE_CHECKING:  # the sampler imports this module, so the batch type is a forward ref
    from .sampler import SampledScenarios

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

    Under ``symmetry="independent"`` the coupling applies to every PHASE draw (that
    symmetry has no component-level base to couple instead), so a correlated spec keeps
    its per-phase asymmetry and still co-moves with the factor.

    Correlation is what keeps an AGGREGATE quantity varying: with ``rho=0`` the mean over
    ``N`` matched components concentrates as ``1/sqrt(N)``, so a few hundred independent
    loads leave the total demand nearly constant however wide the marginal is.
    """

    factor: str
    rho: float = Field(ge=0.0, le=1.0)


#: The harmonic fields of a :class:`ParameterSpec`: the injected magnitude and phase,
#: and the free per-order parameter that is drawn and recorded but written nowhere.
HARMONIC_FIELDS: tuple[str, ...] = ("h_mag", "h_phase", "h_param")


class ParameterSpec(_Base):
    """One varied quantity.

    - ``field``: a POWER field — ``"p"`` / ``"q"`` (one) or ``"pq"`` (both, same
      factor — vary apparent power at constant power factor; ``pq`` requires
      ``mode="scale"``) — the SLACK-VOLTAGE field ``"u_ref"`` (a per-scenario scale on
      the :class:`~pgml.schemas.grid_schema.Source` reference voltage
      ``u_ref_v``; requires ``selector.component="source"`` and ``mode="scale"``, and
      supports neither per-phase ``symmetry`` nor harmonic options) — or a HARMONIC
      field — ``"h_mag"`` (per-order injection magnitude relative to the fundamental) /
      ``"h_phase"`` (per-order phase in degrees), or the FREE field ``"h_param"`` (a
      per-device, per-order draw that is recorded and written nowhere, below). Harmonic
      fields require ``orders``; the first two feed
      ``solve_harmonic_flow(harmonic_injection=...)`` instead of an operating point.
      ``u_ref`` writes a per-source ``u_ref_scale`` operating-point entry that the
      ideal-slack solve multiplies onto ``u_ref_v`` (a batched fundamental boundary).
    - ``mode``: ``"scale"`` (multiply the nominal P/Q or the stored per-order spectrum
      magnitude) or ``"absolute"`` (the sampled value IS the W / var / pu / degrees).
    - ``per``: ``"each"`` (every matched component varies independently — one sampling
      dimension per component) or ``"shared"`` (one sample applied to all matched).
      Ignored when ``correlation`` is set. Harmonic fields additionally accept
      ``"fixed"``: one draw per matched component held FIXED across every scenario of
      the batch — a device's own signature (its emission fraction and angle),
      drawn once from a stream seeded by the config's ``seed`` and the spec's name, and
      consuming no sampling dimension (every other draw is unchanged). This is what makes a
      device's harmonic a stable function of its own loading across the dataset, the
      relation a learner can exploit; it also ties that relation to THIS population's
      signatures, so a model trained on it must be judged on a population drawn with
      another seed. ``"class"``: ONE draw for all matched components, held across the
      batch AND identical for every seed — seeded by the spec's name alone — a CLASS
      constant: with a selector that names a consumer class, every device of that class
      shares one value in every dataset ever drawn, so what a model learns about the
      class transfers to another population of the same grid and, given the class of a
      node, to another grid.
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
    - FREE PARAMETER ``"h_param"`` (``mode="absolute"`` only): drawn per matched device
      and order exactly like ``h_mag`` / ``h_phase`` — same cube, same ``per`` options —
      and recorded under the spec's name (``samples[<name>]`` ``[B, n_eff, n_orders]``),
      but never written into the injection. It is the hook for an emission model this
      package does not define: a generator that makes a device's spectrum depend on
      further per-device quantities declares them here, so they share the batch's
      space-filling cube, seed and persisted record, and applies them to the sampled
      ``harmonic_injection`` itself. Any number of ``h_param`` specs may cover the same
      device and order.
    """

    name: str
    selector: Selector
    distribution: Distribution
    field: Literal[
        "p",
        "q",
        "pq",
        "u_ref",
        "h_mag",
        "h_phase",
        "h_param",
    ] = "pq"
    mode: Literal["scale", "absolute"] = "scale"
    per: Literal["each", "shared", "fixed", "class"] = "each"
    correlation: Optional[Correlation] = None
    symmetry: Literal["balanced", "independent", "small_imbalance"] = "balanced"
    imbalance: float = Field(default=0.0, ge=0.0)
    orders: Optional[list[int]] = None
    harmonic_reference: Optional[Literal["en50160", "iec61000-3-2"]] = None
    emission_class: Literal["A", "B", "C", "D", "auto"] = "auto"

    @property
    def is_harmonic(self) -> bool:
        return self.field in HARMONIC_FIELDS

    @property
    def is_free_parameter(self) -> bool:
        """Whether this spec draws a recorded value that is written nowhere."""
        return self.field == "h_param"

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
        if self.symmetry == "small_imbalance" and self.imbalance <= 0.0:
            raise ValueError("symmetry='small_imbalance' requires imbalance > 0.")
        if self.symmetry != "small_imbalance" and self.imbalance != 0.0:
            raise ValueError("imbalance is only used with symmetry='small_imbalance'.")
        if self.per in ("fixed", "class") and not self.is_harmonic:
            raise ValueError(
                f"per={self.per!r} (a draw held across the batch) is a harmonic option; a "
                "power field varies per scenario."
            )
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
            if self.field != "h_mag" and self.mode != "absolute":
                raise ValueError(f"field={self.field!r} requires mode='absolute'.")
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
    #: Upstream harmonic background at the source, shared by every device on the feeder
    #: (see :class:`BackgroundHarmonicConfig`). ``None`` (default) = no background.
    background: Optional["BackgroundHarmonicConfig"] = None

    #: No implied calculation: the harmonic orders to solve are the caller's choice, since
    #: a config may vary harmonic magnitudes, fundamental power, or both.
    harmonic_orders: ClassVar[Optional[list[int]]] = None

    def sample(self, grid) -> "SampledScenarios":
        """Draw this config's batch from ``grid`` (see :func:`~pgml.scenarios.sample`)."""
        from .sampler import sample

        return sample(grid, self)


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

    #: A cartesian sweep varies operating points only; the calculation is the caller's.
    harmonic_orders: ClassVar[Optional[list[int]]] = None

    def sample(self, grid) -> "SampledScenarios":
        """Enumerate the product batch (see :func:`~pgml.scenarios.cartesian_sample`)."""
        from .sampler import cartesian_sample

        return cartesian_sample(grid, self)


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

    @property
    def harmonic_orders(self) -> list[int]:
        """The solved order set this sweep implies: the fundamental plus ``orders``."""
        return [1, *self.orders]

    def sample(self, grid) -> "SampledScenarios":
        """Build the diagonal sweep (see :func:`~pgml.scenarios.spectrum_sweep`)."""
        from .harmonics import spectrum_sweep

        return spectrum_sweep(grid, self)

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
# Upstream (supply-side) harmonic background
# =============================================================================
class BackgroundHarmonicConfig(_Base):
    """A slowly-varying UPSTREAM harmonic background, injected at the grid's sources.

    Every device behind a common supply sees the same background distortion, so the part
    of a measured harmonic that its own fundamental does not explain is largely SHARED
    across the devices rather than private to each. In an unpublished in-house measurement
    of six inverter racks on one low-voltage laboratory supply, driven together, 54-93 % of
    that unexplained emission was common to all of them within an acquisition — a share a
    per-device emission model cannot produce, because the common part comes from the
    network upstream rather than from the devices. That observation is why this
    construct exists; it fixes none of the levels below, which are all the caller's.

    Realised as the Thevenin source of ``docs/pgml/modeling/error-injection.md``,
    present in EVERY scenario rather than swept one node at a time as
    :func:`~pgml.scenarios.run_node_injection_sweep` does. It is ONE upstream network
    state seen through every point of common coupling: by default each in-service
    ``Source`` node receives an injection carrying the same realized spectrum (an
    explicit ``node_id`` narrows it to that single node). ``source_power_va`` is
    constant across the batch, so ``Y(h)`` is unchanged scenario-to-scenario and only
    the Norton current varies — the batched solve is preserved.

    The level drifts along the STEP axis as an AR(1) shared by every order, because an
    upstream background moves on the timescale of the supplying network's own load rather
    than with anything local. Note the drift is a per-step process: a recipe whose steps
    are far apart relative to the drift's correlation time samples it as white noise —
    and a snapshot recipe (``n_steps == 1``) has no step axis at all, so its scenarios
    draw the level independently from the drift's stationary distribution.

    ``magnitude_pu`` empty (the default) disables the background entirely, leaving the
    rest of the run unaffected.
    """

    #: Per-order background VOLTAGE distortion at the source, per unit of the fundamental:
    #: ``{order: magnitude_pu}``. Empty (default) = no background.
    magnitude_pu: dict[int, float] = Field(default_factory=dict)
    #: Per-order background phase [deg]; orders absent here default to 0.
    phase_deg: dict[int, float] = Field(default_factory=dict)
    #: Node to inject at. ``None`` (default) injects at EVERY in-service ``Source``
    #: node — one shared upstream state at each external-grid coupling point; an
    #: explicit id narrows the injection to that single node.
    node_id: Optional[int] = None
    #: Source strength ``S_sc`` [VA] per injection — larger is stiffer, so more of the
    #: background appears at the node. Constant across the batch to keep ``Y(h)`` batched.
    source_power_va: float = Field(default=20e6, gt=0.0)
    #: Log-magnitude std of the shared slow drift (``0.0`` = a fixed background level).
    drift_std: float = Field(default=0.0, ge=0.0)
    #: Angle std [deg] of the same drift.
    drift_phase_deg: float = Field(default=0.0, ge=0.0)
    #: AR(1) stickiness of the drift along the step axis.
    drift_rho: float = Field(default=0.99, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> "BackgroundHarmonicConfig":
        bad = [o for o in self.magnitude_pu if o < 2]
        if bad:
            raise ValueError(
                f"BackgroundHarmonicConfig magnitude_pu orders must be >= 2, got {bad}: "
                "the fundamental is set by the source's own voltage, not the background."
            )
        return self


# ``ScenarioConfig`` names the background by forward reference (it is defined above it).
ScenarioConfig.model_rebuild()


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
    "BackgroundHarmonicConfig",
    "Perturbation",
    "SpectrumSweepConfig",
    "NodeInjectionSweepConfig",
]


def _example() -> "ScenarioConfig":
    """A representative, valid batch: load P/Q scaling + IEC 61000-3-2 emission draws."""
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
                harmonic_reference="iec61000-3-2",
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
