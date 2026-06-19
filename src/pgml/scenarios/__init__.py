"""pgml.scenarios — reproducible, config-driven batched scenario sampling.

A serializable `ScenarioConfig` (+ seed) deterministically defines a batch of
realized operating points; `sample` draws them (independent / Sobol-QMC / LHS) and
`run_scenarios` solves the whole batch at once via the batched solver. The goal is
generating ML training data in controlled distributions, reproducibly.

Additional sweep helpers:

- ``NodeInjectionSweepConfig(node_ids, phases, orders, magnitudes_pu, phases_deg,
  source_power_va, kind="voltage")`` — serializable config for a per-node harmonic
  "error"-source sweep (one node per scenario). Build from a spectrum dict via
  :meth:`~NodeInjectionSweepConfig.from_spectrum`.
- ``run_node_injection_sweep(grid, config, *, slack, dtype, device) ->
  ScenarioResult`` — sweeps the per-node ``NodeHarmonicSource`` over
  ``config.node_ids`` (all nodes if ``None``), returns ``v [B, H, N]``.

See ``scenarios/CONTEXT.md`` for the interface ledger and the deferred roadmap.
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
    NodeInjectionSweepConfig,
    Normal,
    ParameterSpec,
    Perturbation,
    ScenarioConfig,
    Selector,
    SpectrumSweepConfig,
    Uniform,
)
from .en50160 import en50160_limit, en50160_limits
from .harmonics import sample_coherent_spectra, spectrum_sweep
from .node_injection import run_node_injection_sweep
from .persistence import LoadedDataset, read_dataset, write_dataset
from .perturbation import perturbation_sweep
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
    "Perturbation",
    "SpectrumSweepConfig",
    "NodeInjectionSweepConfig",
]:
    _obj = locals().get(_name)
    if _obj is not None and hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
SampledScenarios.__module__ = __name__
ScenarioResult.__module__ = __name__
LoadedDataset.__module__ = __name__

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
    "SampledScenarios",
    "sample",
    "cartesian_sample",
    "sample_coherent_spectra",
    "spectrum_sweep",
    "perturbation_sweep",
    "run_node_injection_sweep",
    "en50160_limits",
    "en50160_limit",
    "write_dataset",
    "read_dataset",
    "LoadedDataset",
    "ScenarioResult",
    "run_scenarios",
]
