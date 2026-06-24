# Interface ledger: schemas (FROZEN — orchestrator-only)

These three files are the single source of truth. Import them; never edit them.

`SCHEMA_VERSION` (in `schemas/__init__.py`, currently `"0.0.1"`) stamps the contract version
into persisted datasets (`meta.json`); `read_dataset` validates it (MAJOR mismatch → raise,
minor/patch drift → warn). Pre-1.0 the schema MAJOR tracks the library major (both stay `0.x`
while the library is < 1.0.0); bump the patch/minor on any contract change.

- `grid_schema.py`  — input: Grid, Node, Branch (Line/Transformer/Switch/
  ShuntReactor/GenericBranch), Appliance (Source/Load/Generator/Storage/ShuntAppliance),
  FrequencyParam, Spectrum, TypeLibrary, plus input-convention DTOs and converters'
  target types. Phase-domain, SI, L/C storage, pi-form + ComplexTap, no complex in
  the schema (real pairs), structured unit metadata via `si_field`. Conductor geometry:
  `ConductorPlacement` (x/y/GMR/radius/Rdc, tensor-capable) + `LineGeometry`; a
  `Line.conductor_geometry` makes assembly use the Carson/Deri path (`pgml.geometry`).
  - DER / inverter control + storage (`references/der_pv_storage_modeling.md`):
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
- `result_schema.py` — output: ResultSet, SolverDiagnostics, NodeResult (v_re/v_im),
  BranchResult (i_from_*, i_to_*), InjectionResult; optional per-phase P/Q/S;
  indexed by frequency_hz; phasors as (real, imag).
- `scenario_schema.py` — realized inputs: Scenario, *OperatingPoint,
  RealizedSpectrumPoint, ParameterPerturbation.

Invariants every consumer must honor:
- Per-phase arrays align to the component's `phases` tuple (and from_/to_phases).
- Branch direction anchored to from_node -> to_node.
- Reactances/susceptances are NEVER stored; derive from L,C at frequency h.
- A Line/Transformer may carry `type_ref` instead of explicit params; it must be
  materialised (resolved against Grid.types) BEFORE assembly.

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
- Asymmetry (rev: Increment 0): `Load.connection`/`Generator.connection` are now
  `Optional[WindingConnection]` defaulting to `None` (= resolve from config
  `appliance.load.{single_phase_,}default_connection`); DELTA needs >=2 phases, zigzag
  rejected on appliances. Whether a run honors per-phase vs splits totals equally is the
  config `calculation.symmetry` decision (`auto`/`symmetric`/`asymmetric`), resolved by
  `pgml.assembly._symmetry`. Cross-tool basis: `references/asymmetric_modeling.md`.
- Per-phase harmonics (rev: Increment 2): `Load`/`Generator` gain
  `spectrum_per_phase: Optional[dict[Phase, Spectrum]]` (asymmetric distortion), mutually
  exclusive with the all-phases `spectrum`; keys must be a subset of `phases`. Consumed
  by `solve_harmonic_flow` (connection-aware harmonic injection). The runtime
  `harmonic_injection` override also accepts per-phase magnitudes/phases.
- NOT yet converted (plain float; convert when their differentiable path lands):
  catalog `LineType`/`TransformerType`, `ZipCoefficients`, `HarmonicShuntModel` +
  spectra (Phase 3), and the converter input-convention DTOs.
