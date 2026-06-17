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
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

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


class ParameterSpec(_Base):
    """One varied quantity.

    - ``field``: ``"p"`` / ``"q"`` (one) or ``"pq"`` (both, same factor — vary
      apparent power at constant power factor; ``pq`` requires ``mode="scale"``).
    - ``mode``: ``"scale"`` (multiply the nominal P/Q) or ``"absolute"`` (the sampled
      value IS the W / var).
    - ``per``: ``"each"`` (every matched component varies independently — one sampling
      dimension per component) or ``"shared"`` (one sample applied to all matched).
    """

    name: str
    selector: Selector
    distribution: Distribution
    field: Literal["p", "q", "pq"] = "pq"
    mode: Literal["scale", "absolute"] = "scale"
    per: Literal["each", "shared"] = "each"


class ScenarioConfig(_Base):
    """A full, reproducible RANDOM/QMC batch specification.

    ``method``: ``"sobol"`` (QMC, low-discrepancy — recommended for training data),
    ``"lhs"`` (Latin hypercube), or ``"independent"`` (plain seeded RNG).
    """

    n_samples: int = Field(gt=0)
    seed: int = 0
    method: Literal["sobol", "lhs", "independent"] = "sobol"
    parameters: list[ParameterSpec]


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


__all__ = [
    "Uniform",
    "Normal",
    "LogNormal",
    "LogUniform",
    "Constant",
    "Distribution",
    "Selector",
    "ParameterSpec",
    "ScenarioConfig",
    "CartesianAxis",
    "CartesianConfig",
]
