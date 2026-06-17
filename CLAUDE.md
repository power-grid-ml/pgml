# power-grid-ml (pgml) — project memory

Differentiable, GPU-ready, vectorized harmonic power-flow + ML for power grids.
Three goals: (1) grid generation, (2) harmonic power-quality simulation, (3) ML
on simulated data (GNN state estimation). The equation system is the core; load
flow is one differentiable output of it.

## TWO HARD CONSTRAINTS (non-negotiable, apply to every line of core code)
1. DIFFERENTIABLE: gradients must flow from grid parameters -> Y-bus -> solve ->
   outputs. No `.item()`, `.detach()`, `.numpy()`, in-place ops on tracked
   tensors, or Python-number control flow on tensor values in the differentiable
   path. Everything that touches parameters is torch and autograd-safe.
2. GPU-READY: every core op runs on CPU and CUDA unchanged. No hard-coded device;
   honor input tensor device/dtype. Complex dtypes (complex64/complex128).
   Vectorized/batched — no Python loops over nodes/branches/harmonics/scenarios.

A change that breaks gradcheck (float64) or the GPU device/dtype test is not done.

## Repo map
- `src/pgml/schemas/`  FROZEN contracts: grid_schema, result_schema, scenario_schema. READ `schemas/CONTEXT.md`.
- `src/pgml/equations/` residual-form (0=a-b) SymPy registry + torch evaluators.
- `src/pgml/assembly/`  per-phase, per-harmonic, batched, differentiable Y-bus.
- `src/pgml/solver/`    complex batched linear solve (Y(h) V(h) = I(h)).
- `src/pgml/convert/`   converters from pandapower / power-grid-model / OpenDSS -> our schema.
- `src/pgml/geometry/`  differentiable Carson/Deri line constants (geometry -> Z(h)/Yc(h), bit-exact vs OpenDSS) + R/X->geometry synthesis. See `geometry/CONTEXT.md`.
- `src/pgml/scenarios/` reproducible config-driven batched sampling (QMC/cartesian). See `scenarios/CONTEXT.md` + `scenarios/ROADMAP.md` (deferred batching options).
- `src/pgml/evaluation/` comparison & evaluation plots (refs vs ours): Y-bus heatmaps, voltage/harmonic profiles, 3D plotly. See `evaluation/CONTEXT.md`; demo `examples/evaluate_ieee33.py`.
- `references/`         distilled library briefs + ARCHITECTURE.md (the big picture).
- `tests/`              reference (oracle) comparison, differentiability, gpu.

## Frozen-contract rule
The three files in `src/pgml/schemas/` are the single source of truth. They are
ORCHESTRATOR-ONLY. Subagents import them and conform to them.

## Core conventions (all defined in the schemas; do not reinvent)
- Phase-domain canonical form; SI base units; reactive elements store L and C
  (X(h)=2*pi*h*f0*L, B(h)=2*pi*h*f0*C); resistance skin-effect is an explicit law.
- Every branch -> pi-form primitive admittance stamp; transformers add a complex tap.
- Results store phasors as (real, imag); inputs in natural forms. Index by frequency_hz.
- Equations are residual form `0 = a - b`; normalization is an eval-time option, never stored.

## Commands
- use pixi for installing packages and running python code `pixi --environment cpu`
- Tests:            `pytest -q`
- Differentiability gate: `pytest -q tests/differentiability`
- GPU gate:         `pytest -q tests/gpu`
- Lint/format:      `ruff check src tests && ruff format src tests`

## Delegation policy
- Delegate heavy, isolatable work to subagents (see `.claude/agents/`). Keep the
  orchestrator context lean.
- Keep nesting shallow (orchestrator -> subagent). Do not fan out the numerically
  coupled core (assembly+solver) — that is one focused agent.
- After a subagent ships a module, record its PUBLIC SIGNATURES in that module's
  `CONTEXT.md`. That file is how the next agent learns the interface.

@references/ARCHITECTURE.md
@src/pgml/schemas/CONTEXT.md
