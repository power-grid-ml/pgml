# schemas — the frozen data contract

`grid_schema.py`, `result_schema.py` and `scenario_schema.py` are the single source of
truth for the grid description, solve results and scenario identity. Every other pgml
subsystem and every downstream package imports them and never redefines them.

Change rule: a schema edit (field, type, default, validator semantics, documented
reference side or unit) needs the maintainer's sign-off and a `SCHEMA_VERSION` bump
(`schemas/__init__.py`). Pre-1.0 the schema MAJOR tracks the library major; bump
minor/patch on any contract change. `SCHEMA_VERSION` is stamped into persisted datasets
(`meta.json`); `read_dataset` raises on a MAJOR mismatch and warns on minor/patch drift
or a missing stamp. Field descriptions surface in `model_json_schema()`, so wording
changes are contract changes too. Change history lives in git, not here.

Signatures: docstrings / API_INDEX.md.

## Codemap

- `grid_schema.py` — input: `Grid`, `Node`, branches (`Line`, `Transformer`, `Switch`,
  `ShuntReactor`, `GenericBranch`), appliances (`Source`, `Load`, `Generator`,
  `Storage`, `ShuntAppliance`; the first three share `InjectionAppliance`), spectra,
  `TypeLibrary`, conductor geometry (`ConductorPlacement`, `LineGeometry`,
  `EarthReturnModel`), inverter control (`InverterControl` union, `Characteristic`),
  `VoltageRegulation`, `HarmonicShuntModel`, `HarmonicImpedance`, measurement metadata
  (`MeasurementDevice`, `CurrentChannel`), the helper value types (`Num`, `PosNum`,
  `NonNegNum`, `Vec`, `PerPhaseMatrix`) and `si_field`. Its module docstring holds the
  conventions (read it first).
- `result_schema.py` — output: `ResultSet`, `SolverDiagnostics`, `NodeResult`,
  `BranchResult`, `InjectionResult`; indexed by `frequency_hz`, phasors as real pairs.
- `scenario_schema.py` — `Scenario` (batch identity + provenance) and
  `ParameterPerturbation` (injected ground truth). Realized operating points and spectra
  are tensors (`pgml.scenarios.SampledScenarios`) and parquet columns, never rows here.

## Invariants every consumer honours

- Phase-domain, SI, no complex numbers: series (R, L), shunt (G, C); reactances and
  susceptances are never stored, they are derived at frequency `h`. `ComplexTap` is the
  only complex-like quantity (a dimensionless ratio).
- Units are structured metadata (`si_field` → `unit` in the JSON schema); consumers read
  the schema, never parse descriptions.
- Per-phase arrays align to the component's `phases` (or `from_phases` / `to_phases`);
  per-phase matrices are validated `n×n` on construction (`ShuntReactor` against
  `from_phases`).
- Branch direction is `from_node → to_node`.
- `*_rated` / `*_nom` are nameplate ratings, not operating points; power totals are over
  all connected phases; control-block powers are DEVICE totals that assembly splits over
  the device's elements. Injection sign: +1 `Load`, −1 `Generator` / `Storage`.
- Transformer zero-sequence and leakage data are referred to the to-side (LV) winding
  coil.
- `type_ref` is an authoring layer: assembly consumes materialised parameters and pgml
  ships no resolver (see the module docstring "Standard types").
- `Generator.voltage_regulation` and `control` are mutually exclusive; `spectrum` and
  `spectrum_per_phase` are mutually exclusive.
- Upstream background distortion is an operating-point quantity
  (`NodeHarmonicSource`, `BackgroundHarmonicConfig`), never grid data.
- `MeasurementDevice` and the rest of `Grid.measurement_devices` are inert metadata:
  plain floats, never on the autograd tape, never read by assembly or the solver.
  `Grid.attach_measurement_devices` is atomic (a rejected attach restores the list).
- Slack designation is Source-based (`pgml.topology.slack_node_ids`).
- `extra="forbid"`: an older reader rejects a grid that uses a newer field (one-way
  compatibility); legacy keys are migrated by `mode="before"` validators with a warning.

## Float / tensor duality

- Physical fields accept plain floats/lists (the serializable default) OR any array-like
  (`torch.Tensor`, `numpy.ndarray`), passed through UNTOUCHED, so gradients flow
  grid → Y-bus → solve → outputs through the same `assemble_ybus(grid)` call.
- The schema imports no compute framework; array-likes are duck-typed
  (`.detach` / `__array__`). JSON dump detaches tensors to lists.
- pydantic `gt`/`ge` constraints do not apply to these `Any`-typed fields; positivity is
  enforced inside validators for floats only.
- Plain-float fields (never gradient leaves): `ComplexTap.shift_deg` (a discrete clock
  selector; the differentiable tap is `ratio_magnitude`), catalog `LineType` /
  `TransformerType`, `ZipCoefficients`, `HarmonicShuntModel`, spectra, converter
  input-convention DTOs, measurement metadata.

## Contract surface (names other packages rely on)

`pgml.schemas` re-exports all three modules' `__all__` plus `SCHEMA_VERSION`. Consumers:
pgml assembly/solver/converters, pghub loaders, pgg generation, pgl datasets, pgd grid
storage and measurement mapping (`MeasurementDevice`, `CurrentChannel`).
