"""pgml.scenarios — reproducible, config-driven batched scenario sampling.

A batch of scenarios is a set of per-component DELTAS on one grid: what a spec names
varies, everything else keeps the grid's nominal value. This subpackage owns that batch
contract end to end — how a batch is declared, drawn, solved and persisted — and nothing
about which variations a particular study should draw.

Three ways to produce a batch:

- A serializable config plus its seed: :class:`ScenarioConfig` (random / Sobol-QMC draws
  over declared :class:`ParameterSpec` quantities) or :class:`CartesianConfig` (explicit
  product of discrete levels). Saving the config reproduces the batch exactly.
- An excitation primitive: :func:`perturbation_sweep` (one operating-point error per
  target), :func:`spectrum_sweep` (one injected spectrum per target),
  :func:`run_node_injection_sweep` (a per-node disturbance source), or
  :func:`build_background_sources` (an upstream supply-side background).
- Explicit values: :func:`batch_from_values` takes tensors you already have and returns
  the same :class:`SampledScenarios` the samplers do.

:func:`run_scenarios` solves any of them in one batched solve (the solver broadcasts the
leading scenario dimension), and :func:`write_dataset` / :func:`read_dataset` persist the
result as a self-describing parquet dataset. A downstream generator plugs its own recipe
in through the :class:`ScenarioSpec` protocol — an object with ``sample(grid)``.

See ``scenarios/CONTEXT.md`` for the interface ledger and the deferred roadmap.
"""

from __future__ import annotations

from .batch import batch_from_values, broadcast_operating_point
from .config import (
    BackgroundHarmonicConfig,
    CartesianAxis,
    CartesianConfig,
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
from .en50160 import en50160_limit, en50160_limits, en50160_provenance
from .harmonics import build_background_sources, spectrum_sweep
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
from .perturbation import perturbation_sweep
from .random import ar1_noise
from .run import ScenarioResult, ScenarioSpec, run_scenarios
from .sampler import (
    NominalPower,
    SampledScenarios,
    cartesian_sample,
    nominal_power,
    sample,
    unit_samples,
)

#: Study recipes that used to live here: device populations, calibrated emission ranges,
#: load-profile models and the load-dependent emission law. Looked up by the module
#: ``__getattr__`` below so an old import fails with an explanation instead of a bare
#: ``ImportError``.
_RECIPE_NAMES = frozenset(
    (
        "CoherentSpectrumConfig",
        "LoadProfileConfig",
        "CompositionConfig",
        "CompositionDraw",
        "DeviceState",
        "DeviceClassSpec",
        "ClassCount",
        "ConsumerComposition",
        "DEVICE_LIBRARY_VERSION",
        "SE_PRESET_VERSION",
        "HIGH_ACTIVITY_START_TIME",
        "EMISSION_FLOOR",
        "EMISSION_FLOOR_PHASE_DEG",
        "EMISSION_PHASE_SLOPE_DEG",
        "LOADING_FLOOR",
        "affine_emission_correction",
        "phase_slope_shift",
        "composition_silent_orders",
        "default_device_classes",
        "default_compositions",
        "se_random_scenario_config",
        "se_coherent_scenario_config",
        "sample_coherent_spectra",
        "sample_device_composition",
        "resolve_composed_ids",
        "load_profile_factors",
        "apply_load_profiles",
    )
)
#: Names that moved to another pgml module.
_MOVED = dict.fromkeys(
    (
        "StorageDispatchResult",
        "integrate_soc",
        "dispatch_storage",
        "storage_operating_point",
    ),
    "pgml.dispatch",
)


def __getattr__(name: str):
    """Explain an import of a name this package no longer defines.

    Device populations, calibrated emission ranges, load-profile models and the
    load-dependent emission law are modeling choices of a study rather than properties of
    the engine, so the generator that calibrates them defines them and hands the result
    to this package as a :class:`ScenarioSpec` or through :func:`batch_from_values`.
    Storage dispatch moved to :mod:`pgml.dispatch`, where a reader looks for device time
    coupling.
    """
    if name in _MOVED:
        raise ImportError(
            f"{name!r} is no longer part of pgml.scenarios; it now lives in "
            f"{_MOVED[name]}."
        )
    if name in _RECIPE_NAMES:
        raise ImportError(
            f"{name!r} is no longer part of pgml.scenarios. It belongs to a study's "
            "scenario recipe (a device population, a calibrated emission model or a "
            "load profile), which pgml does not define. pgml.scenarios keeps the batch "
            "contract (ScenarioConfig, CartesianConfig, SpectrumSweepConfig, "
            "BackgroundHarmonicConfig, batch_from_values, run_scenarios, write_dataset) "
            "and the standards tables; a recipe plugs in as a ScenarioSpec, an object "
            "with sample(grid)."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "BackgroundHarmonicConfig",
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
    "BackgroundHarmonicConfig",
    "ar1_noise",
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
    "spectrum_sweep",
    "build_background_sources",
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
]
