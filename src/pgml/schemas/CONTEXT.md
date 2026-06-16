# Interface ledger: schemas (FROZEN — orchestrator-only)

These three files are the single source of truth. Import them; never edit them.

- `grid_schema.py`  — input: Grid, Node, Branch (Line/Transformer/Switch/
  ShuntReactor/GenericBranch), Appliance (Source/Load/Generator/ShuntAppliance),
  FrequencyParam, Spectrum, TypeLibrary, plus input-convention DTOs and converters'
  target types. Phase-domain, SI, L/C storage, pi-form + ComplexTap, no complex in
  the schema (real pairs), structured unit metadata via `si_field`.
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
- NOT yet converted (plain float; convert when their differentiable path lands):
  catalog `LineType`/`TransformerType`, `ZipCoefficients`, `HarmonicShuntModel` +
  spectra (Phase 3), and the converter input-convention DTOs.
