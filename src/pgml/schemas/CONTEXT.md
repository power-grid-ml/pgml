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
