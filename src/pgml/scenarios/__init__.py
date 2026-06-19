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
    CoherentSpectrumConfig,
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
from .en50160 import en50160_limit, en50160_limits
from .harmonics import sample_coherent_spectra
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
    "CoherentSpectrumConfig",
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
    "CoherentSpectrumConfig",
    "SampledScenarios",
    "sample",
    "cartesian_sample",
    "sample_coherent_spectra",
    "en50160_limits",
    "en50160_limit",
    "ScenarioResult",
    "run_scenarios",
]
