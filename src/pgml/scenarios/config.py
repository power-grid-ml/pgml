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
    given component kind. Filters combine (AND).
    """

    component: Literal["load", "generator"] = "load"
    ids: Optional[list[int]] = None
    consumer_type: Optional[str] = None

    def resolve(self, grid) -> list[int]:
        from pgml.schemas.grid_schema import Generator, Load

        cls = Load if self.component == "load" else Generator
        out: list[int] = []
        for a in grid.appliances:
            if not isinstance(a, cls) or not a.in_service:
                continue
            if self.ids is not None and a.id not in self.ids:
                continue
            if self.consumer_type is not None and a.consumer_type != self.consumer_type:
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
      ``mode="scale"``) — or a HARMONIC field — ``"h_mag"`` (per-order injection
      magnitude relative to the fundamental) / ``"h_phase"`` (per-order phase in
      degrees). Harmonic fields require ``orders`` and feed
      ``solve_harmonic_flow(harmonic_injection=...)`` instead of an operating point.
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
    - ``harmonic_reference``: ``"en50160"`` makes an ``h_mag`` distribution a FRACTION
      of the per-order DIN EN 50160 limit (so use a ``[0, 1]`` distribution); ``None``
      treats the sampled value as an absolute pu magnitude (or a ``scale`` of the
      stored spectrum).
    """

    name: str
    selector: Selector
    distribution: Distribution
    field: Literal["p", "q", "pq", "h_mag", "h_phase"] = "pq"
    mode: Literal["scale", "absolute"] = "scale"
    per: Literal["each", "shared"] = "each"
    correlation: Optional[Correlation] = None
    symmetry: Literal["balanced", "independent", "small_imbalance"] = "balanced"
    imbalance: float = Field(default=0.0, ge=0.0)
    orders: Optional[list[int]] = None
    harmonic_reference: Optional[Literal["en50160"]] = None

    @property
    def is_harmonic(self) -> bool:
        return self.field in ("h_mag", "h_phase")

    @model_validator(mode="after")
    def _check(self) -> "ParameterSpec":
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
# Node-coherent harmonic "fingerprint" sampling (temporal sequences)
# =============================================================================
class CoherentSpectrumConfig(_Base):
    """Node-coherent harmonic sampling: a stable per-device fingerprint over a sequence.

    Each matched device draws a small set of base spectra (``n_modes`` "modes" — e.g.
    appliance operating states like a washing machine heating vs spinning), drawn once
    (or per scenario). Over ``n_steps`` consecutive steps it STICKS to a mode (Markov
    dwell ``dwell``) and WANDERS around it (AR(1) jitter with stickiness ``ar1_rho``),
    clamped to the DIN EN 50160 per-order limit. This yields a ``[B, T]`` batch of
    harmonic injections in which each node keeps a recognisable signature that varies
    realistically — so a state estimator can attribute the pattern to the node.

    The result voltages are ``[B, T, H, N]`` (B = ``n_scenarios`` sequences, T = steps);
    per-step timestamps are recorded as ``samples["time_s"]``. Fundamental P/Q stays
    nominal (the fingerprint models the harmonic spectrum, not the fundamental load).
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
    harmonic_reference: Optional[Literal["en50160"]] = "en50160"
    phase_distribution: Distribution = Field(
        default_factory=lambda: Uniform(low=-180.0, high=180.0)
    )
    jitter_mag: float = Field(default=0.05, ge=0.0)  # AR(1) fractional std on magnitude
    jitter_phase_deg: float = Field(default=5.0, ge=0.0)  # AR(1) std on phase (deg)
    ar1_rho: float = Field(default=0.8, ge=0.0, le=1.0)  # temporal stickiness of jitter
    dwell: float = Field(default=0.9, ge=0.0, le=1.0)  # P(stay in mode) per step
    step_size_s: float = Field(default=1.0, gt=0.0)
    resample_modes_per_scenario: bool = False

    @model_validator(mode="after")
    def _check(self) -> "CoherentSpectrumConfig":
        if any(o < 2 for o in self.orders):
            raise ValueError("harmonic `orders` must all be >= 2 (1 = fundamental).")
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
