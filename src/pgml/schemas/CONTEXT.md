# Interface ledger: schemas (FROZEN)

These three files are the single source of truth. Import them; never edit them.

`SCHEMA_VERSION` (in `schemas/__init__.py`, currently `"0.2.0"`) stamps the contract version
into persisted datasets (`meta.json`); `read_dataset` validates it (MAJOR mismatch → raise,
minor/patch drift → warn). Pre-1.0 the schema MAJOR tracks the library major (both stay `0.x`
while the library is < 1.0.0); bump the patch/minor on any contract change.

Revision history (what each bump added, and what it means for data written before it):

- Rev 0.0.4 — the dataset sidecar `meta.json` additionally records per-run convergence
  (`converged` + `failed_scenarios`), a persistence-contract addition; the three schema
  modules are unchanged.
- Rev 0.0.5 — `InjectionAppliance.return_path` ("auto"/"neutral"/"ground" WYE
  return-conductor override, default "auto" = the historical node-level rule) and
  `ShuntAppliance.connection` (WYE default, phase-to-ground, or DELTA cyclic
  phase-to-phase bank; zigzag rejected). Both defaults are backward-compatible; assembly
  enforces the semantics.
- Rev 0.0.6 — the harmonic line model, the inductive shunt, the source background
  distortion, the PV terminal, the harmonic device shunt, and four unused realized-value
  models dropped from `scenario_schema`, in detail below.

## Migrating a 0.0.5 grid JSON to 0.0.6

A grid written under 0.0.5 loads under 0.0.6. Five things happen, in order of how much
attention they need:

1. **Nothing at all for the new optional fields.** `Line.harmonic_line_model`,
   `Line.harmonic_skin_effect`, `Line.earth_return`, `ShuntReactor.inductance_h`,
   `ShuntAppliance.inductance_h` and `Generator.voltage_regulation` are optional and
   default to `None`, and every consumer treats `None` as the pre-0.0.6 behaviour: the
   line is assembled from its stored R/X exactly as before, the shunt has no inductive
   path, the generator is a plain PQ injection. No stored value changes and no result
   moves.
2. **Legacy harmonic-line-model tags migrate, with a warning.** A `Line` carrying
   `tags["harmonic_line_model"]`, `tags["seq_skin"]` or `tags["seq_earth_coeff"]` has
   them moved onto the typed fields by a `mode="before"` validator, which emits a
   `UserWarning` naming the line and removes the keys from `tags`. The physics is
   preserved (`"sequence_aware"` + `seq_skin="false"` + a `seq_earth_coeff` float becomes
   `harmonic_line_model="sequence_aware"`, `harmonic_skin_effect=False`,
   `earth_return.resistance_coeff_ohm_per_m_per_hz=<the float>`). An UNRECOGNISED model
   name now raises instead of silently selecting the naive model. Re-persisting the grid
   clears the warning.
3. **`Source.spectrum` is dropped.** The field is REMOVED (upstream / background
   distortion is an operating-point quantity carried by `pgml.solver.NodeHarmonicSource`
   or `pgml.scenarios`' `BackgroundHarmonicConfig`, never by the grid description). A
   persisted `Source` that still carries the key loads: a before-validator drops it, with
   a WARNING when the value is non-null and silently when it is null (every grid written
   by pgml has `"spectrum": null`). Any OTHER unknown field still raises. The same
   before-validator runs on keyword construction, so downstream code that still writes
   `Source(..., spectrum=None)` keeps working; it should drop the argument.
4. **One-way compatibility.** A grid written under 0.0.6 that actually USES a new field
   cannot be read by an older pgml (`extra="forbid"` rejects the unknown key). This is
   the usual direction of this contract.
5. **Four realized-value models are gone from `scenario_schema`.**
   `LoadOperatingPoint`, `GeneratorOperatingPoint`, `SourceOperatingPoint` and
   `RealizedSpectrumPoint` are removed. Nothing constructed, read or persisted them: a
   realized operating point is a tensor in `pgml.scenarios.SampledScenarios` and a column
   in `samples.parquet`, never a row of these types. No stored file contains them, so
   there is nothing to migrate; code that imported the names gets an `ImportError` and
   should read the tensors or the parquet columns instead. `Scenario` and
   `ParameterPerturbation` are unchanged.

Two 0.0.5 fields that existed but were not consumed now ARE consumed, which changes the
RESULT of a grid that set them (no migration is possible or needed — the stored value was
always meant to be used):

- `Transformer.zero_sequence` now carries the zero-sequence leakage VALUE into the stamp
  (it was ignored, so such a unit silently used `Z0 = Z1`). No converter emitted it before
  0.0.6, so no grid in the ecosystem is affected.
- `Transformer.harmonic_xr_constant` and `Transformer.resistance_frequency` now select the
  winding-resistance frequency law (`R(f) = R·m(f)·(f/f0 if the unit holds X/R constant
  else 1)`). The OpenDSS converter has always written `harmonic_xr_constant` from
  `XRConst`, so an OpenDSS-imported grid with `XRConst=Yes` changes its harmonic result.
  The shipped defaults (`False` and a constant multiplier) reproduce the old behaviour.

`Load`/`Generator`/`Storage.harmonic_model` becomes
`Optional[HarmonicShuntModel] = None` and IS consumed: it is now the per-device OVERRIDE
of the harmonic Norton shunt, whose default lives in `appliance.harmonic_shunt.*` and is
selected per run by `solve_harmonic_flow(load_shunt=...)`. A persisted grid is read
unchanged — a stored block (every grid written by pgml carries the former default
`{series_rl_fraction: 0.5, neglect_shunt: false, motor_x_harm_pu: null,
motor_xr_harm: 6.0}`) keeps exactly the same meaning, which is also the shipped default,
so no stored value changes its physics. The RESULT of a harmonic solve DOES change,
because the field was previously ignored: at orders `h > 1` each device now carries
`Y_eq = conj(S)/V_rated²` split between a parallel and a series R-L branch
(`docs/pgml/modeling/references/opendss/harmonics.md`). `load_shunt="none"` reproduces
the former pure current-source model exactly. A `Generator` / `Storage` is the exception:
its stored block does NOT grant it a shunt, because every grid written before the field
was consumed carries the former default block on every device and the load expression's
conductance is negative for an injecting device (the documented default
`appliance.harmonic_shunt.generation_model`, shipped `none`, leaves such a device a pure
current source; `load_style` applies the expression anyway, and a block naming
`motor_x_harm_pu` selects the motor model either way). `HarmonicShuntModel` additionally
rejects two ill-posed combinations (`neglect_shunt=True` together with a `motor_x_harm_pu`, and a
non-positive `motor_x_harm_pu`/`motor_xr_harm`) instead of silently picking a branch.

`TransformerZeroSeq`'s documented reference side is corrected from "HV" to "the to-side
(LV) winding coil" — the side the positive-sequence leakage fields use and the side
assembly consumes (a docstring / unit-metadata change; no field, name or value changes).
The field was not consumed while it was documented as HV-referred, so no result changed,
but a persisted grid that carries HV-referred values is read as to-side-coil values
without a warning and is then too large by the squared turns ratio (times 3 for a delta
to-side coil). The class docstring gives the conversion.

- `grid_schema.py`  — input: Grid, Node, Branch (Line/Transformer/Switch/
  ShuntReactor/GenericBranch), Appliance (Source/Load/Generator/Storage/ShuntAppliance),
  FrequencyParam, Spectrum, TypeLibrary, plus input-convention DTOs and converters'
  target types. Phase-domain, SI, L/C storage, pi-form + ComplexTap, no complex in
  the schema (real pairs), structured unit metadata via `si_field`. Conductor geometry:
  `ConductorPlacement` (x/y/GMR/radius/Rdc, tensor-capable) + `LineGeometry`; a
  `Line.conductor_geometry` makes assembly use the Carson/Deri path (`pgml.geometry`).
  - Harmonic line model (typed, replaces the former free-text `Line.tags` selectors):
    `Line.harmonic_line_model: Optional[Literal["geometry","sequence_aware",
    "positive_sequence","naive"]]` (None = unresolved; the converters and
    `pgml.geometry.apply_default_harmonic_model` resolve it from the modeling default
    `line.harmonic_model.*`, assembly never does), `Line.harmonic_skin_effect:
    Optional[bool]`, and `Line.earth_return: Optional[EarthReturnModel]`
    (`resistance_coeff_ohm_per_m_per_hz`, `reactance_coeff_ohm_per_m_per_hz`,
    `x0_frequency`, `x0_exponent`, `r0_includes_earth_return` — the lumped Carson
    earth-return path of the `sequence_aware` model, every field tensor-capable and on
    the differentiable path). The `Line` validator REJECTS contradictory combinations (a
    lumped model on a geometry line, `sequence_aware` on a non-3-phase line, an
    `earth_return` / `harmonic_skin_effect` / `resistance_frequency` law the selected
    model does not consume). A grid persisted with the old
    `tags["harmonic_line_model"]` / `tags["seq_skin"]` / `tags["seq_earth_coeff"]` is
    MIGRATED to the typed fields by a `mode="before"` validator with a warning, and an
    unknown model name now raises instead of silently selecting the naive model.
  - Inductive shunt: `ShuntReactor.inductance_h: Optional[PerPhaseMatrix]` and
    `ShuntAppliance.inductance_h: Optional[Vec]` (entries > 0) add the
    `1/(j·2πh f0 L)` term to the shunt stamp, so an inductive shunt's susceptance
    magnitude falls as `1/h` instead of rising as `h`. Absent = no inductive path, so
    persisted grids are unchanged.
  - DER / inverter control + storage (`docs/pgml/modeling/der-pv-storage.md`):
    `Load`/`Generator`/`Storage` share the `InjectionAppliance` base (consumers test
    `isinstance(a, InjectionAppliance)`; sign = +1 Load, −1 Generator/Storage). `Generator`
    and `Storage` carry an optional `control: InverterControl` — a discriminated union
    (`kind`) of `ConstantPowerFactorControl`, `ConstantReactivePowerControl`,
    `PowerFactorWattControl`, `VoltVarControl`, `VoltWattControl`, `VoltVarVoltWattControl`,
    each over `InverterControlBase` (`s_rated_va` capability circle, `smoothing` half-width).
    Curves use the generic tensor-capable `Characteristic` (x_values/y_values, linear/cubic).
    `Storage`: signed `p_nom_w` (>0 discharge/inject), plus inert energy-state fields
    (`energy_capacity_wh`, `soc`, `soc_min/max`, `efficiency_charge/discharge`, `p_rated_w`)
    consumed only by `pgml.dispatch`. `consumer_type` is now the closed
    `ConsumerType` enum (str-enum; `"pv"` etc. still validate; ML categorical, no physics).
    Control is honored by the NONLINEAR solve (`device_current_injections`); the linear
    const-Z assembler uses the base P/Q.
  - VOLTAGE REGULATION (a PV terminal): `Generator` additionally carries an optional
    `voltage_regulation: VoltageRegulation` — `v_set_pu` (PosNum, per unit of the HOST
    NODE's rated voltage, default 1.0), `q_min_var`/`q_max_var` (Optional[Num], TOTAL
    over the phases, injection-positive, `None` = unbounded) and `regulated`
    (`RegulatedQuantity`: `positive_sequence` (default) | `per_phase`). MUTUALLY
    EXCLUSIVE with `control` (validated): a control law states Q as a function of the
    voltage, regulation states the voltage and leaves Q implicit. All three numeric
    fields keep the float/tensor duality, so `dV/dv_set` and `dV/dq_limit` flow through
    the IFT. Consumed by `solve_power_flow` (the terminal's reactive power-balance row
    becomes `|V|² − V_set²`, solved by Newton with PV-to-PQ switching for the limits —
    `solver/_pv_bus.py`, `solver/CONTEXT.md`); the linear const-Z assembler and the
    harmonic orders h>1 use the base P/Q as before. Written by the pandapower
    (`net.gen`) and OpenDSS (`Generator model=3`) converters. Backward-compatible: the
    field defaults to `None` and a persisted grid without it is unchanged.
  - Measurement instrumentation (rev 0.0.3): `Grid.measurement_devices:
    list[MeasurementDevice]` — INERT metadata (never on the autograd tape, plain floats
    only; nothing here enters assembly or the solver). A `MeasurementDevice` is
    node-anchored (a meter cabinet at a bus): voltage at `node`/`phases`, currents per
    `CurrentChannel` (branch id + optional phases; terminal inferred from the device's
    node, explicit only for self-loops) on INCIDENT branches, `measured_quantities`
    (`MeasuredQuantity` str-enum voltage/current/power), capability
    (`max_harmonic_order`, `max_current_channels`), identity (`manufacturer`, `model`),
    acquisition (`supported_averaging_intervals_s`, `averaging_interval_s` — membership
    validated), `accuracy_class` (categorical; keys the ML noise model, no numeric
    interpretation in pgml), `connection` (free-form JSON with a `kind` discriminator,
    interpreted by the external acquisition service only). Cross-references (node/branch
    existence, incidence, phase subsets, unique ids) validate in `Grid._integrity`.
    Attach devices to an existing (e.g. converted) grid via
    `Grid.attach_measurement_devices(devices)` — atomic: a rejected attach restores the
    previous list (a plain field assignment would keep the bad value on a model-validator
    failure). Slack designation stays Source-based: `pgml.topology.slack_node_ids` (all
    in-service Sources; `slack_node_id` = first). Metadata for a downstream state-estimation
    sensor model (sensor set + slack coverage) and a future measurement-acquisition service.
- `result_schema.py` — output: ResultSet, SolverDiagnostics, NodeResult (v_re/v_im),
  BranchResult (i_from_*, i_to_*), InjectionResult (`injection_kind` covers
  load/generator/storage/source/shunt); optional per-phase P/Q/S;
  indexed by frequency_hz; phasors as (real, imag).
- `scenario_schema.py` — realized inputs: Scenario (the batch's identity + provenance),
  ParameterPerturbation (injected ground truth). The realized operating points and
  spectra themselves are tensors, not rows: `pgml.scenarios.SampledScenarios` carries
  them and `write_dataset` persists them columnar in `samples.parquet`.

Invariants every consumer must honor:
- Per-phase arrays align to the component's `phases` tuple (and from_/to_phases).
- Branch direction anchored to from_node -> to_node.
- Reactances/susceptances are NEVER stored; derive from L,C at frequency h.
- A Line/Transformer may carry `type_ref` instead of explicit params; it must be
  materialised (resolved against Grid.types) BEFORE assembly.
- Matrix dimensions are validated on construction: Line, Source, ShuntReactor and
  GenericBranch all reject a per-phase matrix that is not `n×n` for their phase
  count (ShuntReactor validates against `from_phases` — it is single-terminal).

## Float / tensor duality (duck-typed, torch-free)
Physical fields accept EITHER plain python floats/lists (the serializable default)
OR any array-like object (torch.Tensor / numpy.ndarray), passed through UNTOUCHED so
autograd gradients flow `grid -> Y-bus -> solve -> outputs` via the same
`assemble_ybus(grid)` call — no separate parameter container, no `param_overrides`
needed. The schema imports NO compute framework; "array-like" is duck-typed
(`.detach`/`__array__`). Helper types in `grid_schema.py`: `Num`, `PosNum`,
`NonNegNum`, `Vec`, and the (now tensor-capable) `PerPhaseMatrix`. Notes:
- pydantic numeric constraints (gt/ge) do NOT apply to these `Any`-typed fields;
  positivity is enforced inside the validators (floats only). Unit metadata still
  appears in `model_json_schema()`; JSON dump detaches tensors to lists.
- Converted (load-flow path): Node.u_rated_v; Line R/L/G/C + length; Source u_ref/
  angle + R/L; Switch; ShuntReactor; GenericBranch; ShuntAppliance; Load/Generator
  P/Q (+per-phase); Transformer R/L/ratings + ComplexTap + grounding/zero-seq.
  CARVE-OUT: `ComplexTap.shift_deg` is a plain `float` (a discrete vector-group
  clock selector realised by the constant winding incidence — never a gradient
  leaf; the continuous differentiable tap is `ratio_magnitude`).
- Asymmetry: `Load.connection`/`Generator.connection` are now
  `Optional[WindingConnection]` defaulting to `None` (= resolve from config
  `appliance.load.{single_phase_,}default_connection`); DELTA needs >=2 phases, zigzag
  rejected on appliances. Whether a run honors per-phase vs splits totals equally is the
  config `calculation.symmetry` decision (`auto`/`symmetric`/`asymmetric`), resolved by
  `pgml.assembly._symmetry`. Cross-tool basis: `docs/pgml/modeling/asymmetric.md`.
- Per-phase harmonics: `Load`/`Generator` gain
  `spectrum_per_phase: Optional[dict[Phase, Spectrum]]` (asymmetric distortion), mutually
  exclusive with the all-phases `spectrum`; keys must be a subset of `phases`. Consumed
  by `solve_harmonic_flow` (connection-aware harmonic injection). The runtime
  `harmonic_injection` override also accepts per-phase magnitudes/phases.
- NOT yet converted (plain float; convert when their differentiable path lands):
  catalog `LineType`/`TransformerType`, `ZipCoefficients`, `HarmonicShuntModel` +
  spectra, and the converter input-convention DTOs. The harmonic shunt is still
  differentiable in the quantities that matter — it is built from the device's P/Q and
  the fundamental solution — but `series_rl_fraction` / `motor_x_harm_pu` themselves are
  not gradient leaves.

## Schema 0.1.0 / library 0.4.0

- `LineGeometry.internal_inductance: Optional[Literal["gmr", "gmr_skin",
  "gmr_power_frequency", "bessel"]] = None`; resolves at assembly time if absent.
- `EarthReturnModel.x0_nonnegative: Optional[bool] = None`; per-line override of the
  guarded sub-linear Carson law.
- Both additions serialize/round-trip and accept old JSON with unset fields. A stored
  grid with unresolved defaults is not a complete historical modeling snapshot.
- `ConductorPlacement`, `LineGeometry`, `EarthReturnModel` are exported by
  `pgml.schemas` as well as `grid_schema`, and appear in the generated API docs.

## Rev0.2.0: passive DER harmonic impedance

`HarmonicImpedance` is public. Generator/Storage add optional `harmonic_impedance`:
scalar or per-connection-element `resistance_ohm`/`inductance_h` retain tensors;
finite nonnegative, nonzero per element. Absent fields in older grids remain
absent; no universal machine/inverter impedance is guessed. New grids using the
block need a0.2-aware reader. Library version0.5.0.

`spectrum_reference`: current preserves terminal-current emissions;
internal_voltage initializes per-element E1=Vt-Z1*I_absorbed; opendss_voltage
reproduces the native first-phase balanced nodal-voltage convention.
`frequency_model`: series_rl is the passive series R/L default;
opendss_admittance explicitly retains Re(1/Z1) and scales Im(1/Z1)/h.
The impedance is harmonic-only and independent of the load-shunt switch.
