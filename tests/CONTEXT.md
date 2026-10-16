# Testing conventions

Run the quick loop with `pixi run -e cpu pytest -q -m "not slow"`; CI runs the full CPU
suite. `tests/gpu` needs a CUDA host.

Layout (one directory per concern; mirrors the package layout where it maps 1:1):
- `tests/reference/`        cross-library oracle comparisons (Y-bus + results).
- `tests/differentiability/` gradcheck and finite-difference vs autograd.
- `tests/gpu/`              device/dtype parity (skipped if no CUDA).
- `tests/api/`              the public `simulate`/`simulate_serializable` entry points.
- `tests/asymmetric/`       connection-aware / per-phase load, generator and harmonic modeling.
- `tests/control/`          DER inverter control laws and storage dispatch.
- `tests/convert/`          pandapower / OpenDSS / power-grid-model converters.
- `tests/defaults/`         `pgml.defaults` loader + override resolution.
- `tests/evaluation/`       plotting + evaluation data containers.
- `tests/scenarios/`        batched scenario sampling, presets, persistence.
- `tests/topology/`         slack/connectivity/fingerprint/multigrid bookkeeping.
- `tests/fixtures/`         shared tiny-grid builders used across the suite.

Markers (registered in `pyproject.toml`): `gpu` (requires CUDA), `opendss` (drives a live
`opendssdirect` engine), `slow` (long-running). An optional dependency a module imports at
module level (`opendssdirect`, `power_grid_model`, `pandapower`) must guard that import and
call `pytest.skip(..., allow_module_level=True)` when it is missing or broken, so the suite
degrades cleanly on a minimal install.

Differentiability gate (MANDATORY for assembly + solver):
- Use float64/complex128 and `torch.autograd.gradcheck` on small systems.
- Check grads of node voltages w.r.t. line R/L/C, transformer R/L/tap, source Z.
- A finite-difference spot check backs up gradcheck on at least one parameter.

GPU gate:
- Build a small grid, assemble + solve on CPU, then on CUDA (if available); assert
  results match within tol and dtype/device are honored. Mark cuda tests to skip
  cleanly when unavailable.

Oracle comparisons (load flow vs external reference tools):
- Y-bus vs OpenDSS `Export Y` / `SystemY` (align by YNodeOrder; reconcile SI vs
  the reference's units/pu; tol e.g. rtol=1e-6 on matched entries).
- Node voltages / branch flows vs pandapower `res_bus` / `res_line` (convert pp
  engineering units to SI; align by id_map; tol e.g. atol on pu voltage 1e-4).
- power-grid-model as a second results oracle (asym power flow).
Always align per-phase ordering between our (A,B,C,N) layout and the reference.

Reference-model fixture: `tests/reference/conftest.py::opendss_model_defaults`, opted into
by conformance modules. The default/preset restoration, schema round-trip, mixed-model
assembly and CPU/CUDA checks are in `tests/defaults/test_reference_presets.py`.
The IFT context-retention test is `tests/differentiability/test_modeling_presets.py`.

Environment traps:
- Run tests through `pixi run`. Calling the environment's python directly skips activation,
  so `CONDA_PREFIX` is unset, `import power_grid_model` fails, and the guarded
  power-grid-model modules skip silently (the only symptom is a lower collected count). To
  change `PYTHONPATH`, use `pixi run -e cpu env PYTHONPATH=… python -m pytest …`; a
  `PYTHONPATH` set before `pixi run` is overwritten by the activation.
- `opendssdirect` wraps one process-global engine. `Clear` does not reset
  `DefaultBaseFrequency`, so a module that sets 50 Hz leaks into the next DSS-driven module
  even when that circuit says `frequency=60`. Reassert `set DefaultBaseFrequency=<f0>` right
  after `Clear` (see `_dss_clear` in `tests/reference/test_opendss_transformer.py`).
- Matplotlib runs on the `Agg` backend in tests (set in `tests/evaluation/conftest.py`); a
  module that plots imports it through `pytest.importorskip`.
- A numpy-oracle parity test is only as strict as the line model it exercises: branches
  that carry no harmonic current hide line-model differences. When a parity test degrades
  after adding a term that loads every bus, re-run it with `harmonic_line_model="naive"`
  first.
