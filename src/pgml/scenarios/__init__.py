"""pgml.scenarios — reproducible, config-driven batched scenario sampling.

A serializable `ScenarioConfig` (+ seed) deterministically defines a batch of
realized operating points; `sample` draws them (independent / Sobol-QMC / LHS) and
`run_scenarios` solves the whole batch at once via the batched solver. The goal is
generating ML training data in controlled distributions, reproducibly.

See `scenarios/CONTEXT.md` for the interface ledger and the deferred roadmap.
"""

from __future__ import annotations

from .config import (
    CartesianAxis,
    CartesianConfig,
    Constant,
    Correlation,
    Distribution,
    LatentFactor,
    LogNormal,
    LogUniform,
    Normal,
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
)
from .run import ScenarioResult, run_scenarios
from .sampler import SampledScenarios, cartesian_sample, sample

# Canonical __module__ for public re-exports (avoids autodoc duplicate warnings).
for _name in [
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
]:
    _obj = locals().get(_name)
    if _obj is not None and hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
SampledScenarios.__module__ = __name__
ScenarioResult.__module__ = __name__

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
    "SampledScenarios",
    "sample",
    "cartesian_sample",
    "ScenarioResult",
    "run_scenarios",
]
