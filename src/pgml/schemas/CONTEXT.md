# Interface ledger: schemas (FROZEN)

These three files are the single source of truth. Import them; never edit them.

`SCHEMA_VERSION` (in `schemas/__init__.py`, currently `"0.0.5"`) stamps the contract version
into persisted datasets (`meta.json`); `read_dataset` validates it (MAJOR mismatch → raise,
minor/patch drift → warn). Pre-1.0 the schema MAJOR tracks the library major (both stay `0.x`
while the library is < 1.0.0); bump the patch/minor on any contract change. (Rev 0.0.4: the
dataset sidecar `meta.json` additionally records per-run convergence — `converged` +
`failed_scenarios` — a persistence-contract addition; the three schema modules are unchanged.
Rev 0.0.5: `InjectionAppliance.return_path` — "auto"/"neutral"/"ground" WYE return-conductor
override, default "auto" = the historical node-level rule; `ShuntAppliance.connection` —
WYE (default, phase-to-ground) or DELTA (cyclic phase-to-phase bank), zigzag rejected.
Both defaults are backward-compatible; assembly enforces the semantics.)

Pending contract changes awaiting the next `SCHEMA_VERSION` bump: `Source.spectrum` is
REMOVED (upstream/background distortion is an operating-point quantity carried by
`pgml.solver.NodeHarmonicSource` / `pgml.scenarios`' `BackgroundHarmonicConfig`, never by
the grid description). A persisted `Source` that still carries the key loads unchanged: a
before-validator drops it, with a WARNING when the value is non-null (a null carries no
information and is dropped silently). Any OTHER unknown field still raises. Downstream code
that CONSTRUCTS a `Source` with `spectrum=None` must drop that argument.

- `grid_schema.py`  — input: Grid, Node, Branch (Line/Transformer/Switch/
  ShuntReactor/GenericBranch), Appliance (Source/Load/Generator/Storage/ShuntAppliance),
  FrequencyParam, Spectrum, TypeLibrary, plus input-convention DTOs and converters'
  target types. Phase-domain, SI, L/C storage, pi-form + ComplexTap, no complex in
  the schema (real pairs), structured unit metadata via `si_field`. Conductor geometry:
  `ConductorPlacement` (x/y/GMR/radius/Rdc, tensor-capable) + `LineGeometry`; a
  `Line.conductor_geometry` makes assembly use the Carson/Deri path (`pgml.geometry`).
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
    consumed only by `pgml.scenarios` dispatch. `consumer_type` is now the closed
    `ConsumerType` enum (str-enum; `"pv"` etc. still validate; ML categorical, no physics).
    Control is honored by the NONLINEAR solve (`device_current_injections`); the linear
    const-Z assembler uses the base P/Q.
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
- `scenario_schema.py` — realized inputs: Scenario, *OperatingPoint,
  RealizedSpectrumPoint, ParameterPerturbation.

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
  spectra (Phase 3), and the converter input-convention DTOs.
