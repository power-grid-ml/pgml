# power-grid-ml (pgml) — project memory

Differentiable, GPU-ready, vectorized harmonic power-flow + ML for power grids.
Three goals: (1) grid generation, (2) harmonic power-quality simulation, (3) ML on the
simulated data. The equation system is the core; load flow is one differentiable output.

**Architecture, package map, and conventions: `CONTEXT.md` (read it first).**
**Orientation + open work: `HANDOFF.md`.**

## TWO HARD CONSTRAINTS (non-negotiable, every line of core code)
1. DIFFERENTIABLE: gradients must flow grid parameters → Y-bus → solve → outputs. No
   `.item()`, `.detach()`, `.numpy()`, in-place ops on tracked tensors, or Python-number
   control flow on tensor values in the differentiable path. (Sanctioned `.detach()`: the
   IFT adjoint in `solver/power_flow.py`.)
2. GPU-READY: every core op runs on CPU and CUDA unchanged; honor input device/dtype;
   complex dtypes (complex64/128); vectorized/batched (no Python loops over
   nodes/branches/harmonics/scenarios).

A change that breaks gradcheck (float64) or the GPU device/dtype test is not done.

## Code style (production-bound; research-stage, heading to production)
- NO conversational / process references in code, comments, docstrings, CONTEXT files, or
  docs — e.g. "Increment 1", "M1", "the fix above", "as requested", PR/chat phrasing. These
  do not translate to the published documentation. Describe the BEHAVIOUR and the WHY, not
  the development history. Roadmap / open-work notes belong in `HANDOFF.md` (and git history).
- Write code as if it ships: clear names, self-explanatory comments, no dead scaffolding. It
  is fine to leave a capability incomplete in research stage, but what exists reads as
  production code.

## Frozen-contract rule
`src/pgml/schemas/` (grid/result/scenario) is the single source of truth. Subagents IMPORT
and conform to it, never modify it. The orchestrator MAY modify a schema, but only after
asking the user first. (Docstring-only schema edits to keep the docs build RST-clean are
preferred over working around the schemas in `docs/conf.py`.)

## Commands
- Install / run code: `pixi run -e cpu python ...` / `pixi add <pkg>` (use `pixi`, not bare pip).
- Tests: `pixi run -e cpu pytest -q`. Differentiability gate: `pytest -q tests/differentiability`.
  GPU gate: `pytest -q tests/gpu`.
- Lint/format: `ruff check src tests && ruff format src tests`.
- Docs (HTML): `pixi run --environment docs docs` (clean rebuild: `docs-clean`).
  Strict / CI-mirror: `pixi run --environment docs sphinx-build -b html -W --keep-going docs docs/_build/html`.

## Documentation (Read-the-Docs / Sphinx)
Sphinx lives in `docs/` (autodoc + autosummary + napoleon + MyST, furo theme); RTD config
`.readthedocs.yaml`; pip deps `docs/requirements.txt`; local build env is the pixi `docs`
feature. The API reference is generated from each subpackage's `__init__.py` `__all__`, so
**docstrings ARE the docs**. `pandapower` is mocked at autodoc time (NumPy-2 import break);
`pydantic`/`torch` are real. Keep schema docstrings RST-safe IN SOURCE (no build-time
rewriting); `docs/conf.py` keeps only a Python-domain dedup hook for the re-exported schema
types. A docs build with import errors or broken autosummary is NOT done — keep new
docstrings valid reStructuredText (wrap inline math/identifiers containing `*` or trailing
`_` in double backticks; `::` before indented blocks; blank line before bullet lists; no
explicit forward-ref quotes in annotations under `from __future__ import annotations`).

## Delegation policy
- Delegate heavy, isolatable work to subagents (`.claude/agents/`); keep the orchestrator
  context lean. Keep nesting shallow (orchestrator → subagent). Do NOT fan out the
  numerically coupled core (assembly+solver) — that is one focused agent.
- After a subagent ships a module, record its PUBLIC SIGNATURES in that module's `CONTEXT.md`
  — that file is how the next agent learns the interface.
- **rtd-docs-builder** (`.claude/agents/`): owns the Sphinx/RTD pipeline + the authored docs
  under `docs/`. Run it at the END of any API-changing refactor (public signatures, `__all__`
  exports, or module docstrings change, or a module is added/removed): it re-authors the
  affected pages and validates with a clean local build mirroring CI. The docs build is a
  gate — an API change is not done until it passes.

@references/ARCHITECTURE.md
@src/pgml/schemas/CONTEXT.md
