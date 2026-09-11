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

from .batch import batch_from_values, broadcast_operating_point
from .composition import (
    CompositionDraw,
    resolve_composed_ids,
    sample_device_composition,
)
from .config import (
    DEVICE_LIBRARY_VERSION,
    CartesianAxis,
    CartesianConfig,
    ClassCount,
    CoherentSpectrumConfig,
    BackgroundHarmonicConfig,
    CompositionConfig,
    Constant,
    ConsumerComposition,
    Correlation,
    DeviceClassSpec,
    DeviceState,
    Distribution,
    LatentFactor,
    LoadProfileConfig,
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
    composition_silent_orders,
    default_compositions,
    default_device_classes,
)
from .en50160 import en50160_limit, en50160_limits, en50160_provenance
from .harmonics import (
    build_background_sources,
    sample_coherent_spectra,
    spectrum_sweep,
)
from .profiles import apply_load_profiles, load_profile_factors
from .iec61000_3_2 import (
    iec61000_3_2_device_caps,
    iec61000_3_2_fraction,
    iec61000_3_2_limits,
    iec61000_3_2_provenance,
    resolve_emission_class,
)
from .node_injection import run_node_injection_sweep
from .persistence import (
    SCENARIO_CONFIG_TYPES,
    LoadedDataset,
    config_hash,
    generation_provenance,
    read_dataset,
    write_dataset,
)
from .emission import (
    LOADING_FLOOR,
    affine_emission_correction,
    phase_slope_shift,
)
from .presets import (
    EMISSION_FLOOR,
    EMISSION_FLOOR_PHASE_DEG,
    EMISSION_PHASE_SLOPE_DEG,
    HIGH_ACTIVITY_START_TIME,
    SE_PRESET_VERSION,
    se_coherent_scenario_config,
    se_random_scenario_config,
)
from .perturbation import perturbation_sweep
from .run import ScenarioResult, ScenarioSpec, run_scenarios
from .sampler import (
    NominalPower,
    SampledScenarios,
    cartesian_sample,
    nominal_power,
    sample,
    unit_samples,
)
from .storage import (
    StorageDispatchResult,
    dispatch_storage,
    integrate_soc,
    storage_operating_point,
)

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
    "LoadProfileConfig",
    "DeviceState",
    "DeviceClassSpec",
    "ClassCount",
    "ConsumerComposition",
    "BackgroundHarmonicConfig",
    "CompositionConfig",
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
StorageDispatchResult.__module__ = __name__
CompositionDraw.__module__ = __name__

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
    "BackgroundHarmonicConfig",
    "CompositionConfig",
    "CompositionDraw",
    "DEVICE_LIBRARY_VERSION",
    "composition_silent_orders",
    "default_device_classes",
    "default_compositions",
    "SE_PRESET_VERSION",
    "HIGH_ACTIVITY_START_TIME",
    "EMISSION_FLOOR",
    "EMISSION_FLOOR_PHASE_DEG",
    "EMISSION_PHASE_SLOPE_DEG",
    "LOADING_FLOOR",
    "affine_emission_correction",
    "phase_slope_shift",
    "se_random_scenario_config",
    "se_coherent_scenario_config",
    "Perturbation",
    "SpectrumSweepConfig",
    "NodeInjectionSweepConfig",
    "SampledScenarios",
    "NominalPower",
    "ScenarioSpec",
    "sample",
    "cartesian_sample",
    "batch_from_values",
    "broadcast_operating_point",
    "unit_samples",
    "nominal_power",
    "sample_coherent_spectra",
    "spectrum_sweep",
    "build_background_sources",
    "sample_device_composition",
    "resolve_composed_ids",
    "load_profile_factors",
    "apply_load_profiles",
    "perturbation_sweep",
    "run_node_injection_sweep",
    "en50160_limits",
    "en50160_limit",
    "en50160_provenance",
    "iec61000_3_2_limits",
    "iec61000_3_2_fraction",
    "iec61000_3_2_device_caps",
    "iec61000_3_2_provenance",
    "resolve_emission_class",
    "write_dataset",
    "read_dataset",
    "config_hash",
    "generation_provenance",
    "LoadedDataset",
    "SCENARIO_CONFIG_TYPES",
    "ScenarioResult",
    "run_scenarios",
    "StorageDispatchResult",
    "integrate_soc",
    "dispatch_storage",
    "storage_operating_point",
]
