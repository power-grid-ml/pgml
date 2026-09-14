# pgml (power-grid-ml) — project memory

Differentiable, GPU-ready, vectorized harmonic power flow for power grids. The harmonic
system `Y(h)·V(h)=I(h)` is the core; load flow is one differentiable output. pgml is
designed as the base layer of a larger power-grid ecosystem (state estimation, grid
synthesis, dataset and dashboard tooling can build on its public API); pgml itself imports
none of them.

Architecture, package map, and conventions: `CONTEXT.md` (read it first), then
`src/pgml/CONTEXT.md` (subpackage ledgers). Orientation and open work: `src/pgml/STATUS.md`.
Published human docs: `docs/pgml/` (Sphinx, built as a standalone Read-the-Docs site).

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
- NO conversational / process references in code, comments, docstrings, CONTEXT files,
  docs, or commit messages — e.g. "Increment 1", "M1", "the fix above", "as requested",
  PR/chat phrasing. These do not translate to the published documentation. Describe the
  BEHAVIOUR and the WHY, not the development history. Roadmap / open-work notes belong in
  `src/pgml/STATUS.md` (and git history). Commit messages stand alone.
- Write code as if it ships: clear names, self-explanatory comments, no dead scaffolding. It
  is fine to leave a capability incomplete in research stage, but what exists reads as
  production code.

## Frozen-contract rule
`src/pgml/schemas/` (grid/result/scenario) is the single source of truth — for this package
AND for every dependent repository (they pin a `power-grid-ml` version range and read
`SCHEMA_VERSION` from persisted datasets). Subagents IMPORT and conform to it, never modify
it. The orchestrator MAY modify a schema, but only after asking the user first, and bumps
`SCHEMA_VERSION`. (Docstring-only schema edits to keep the docs build RST-clean are
preferred over working around the schemas in `docs/conf.py`.)

## Commands
- Install / run code: `pixi run -e cpu python ...` / `pixi add <pkg>` (use `pixi`, not bare
  pip). `PYTHONPATH=src` is set by the pixi activation, so run from the repository root.
- Tests: `pixi run -e cpu pytest -q` (`-m "not slow"` for the quick loop). Differentiability
  gate: `pytest -q tests/differentiability`. GPU gate: `pytest -q tests/gpu` (CUDA host).
- Lint/format: `ruff check src tests run && ruff format src tests run`.
- Docs (HTML): `pixi run -e docs docs` (clean rebuild: `docs-clean`). Strict / CI-mirror:
  `pixi run -e docs docs-strict`.

## Documentation (Sphinx)
This repository documents its package under `docs/pgml/` (concepts, `modeling/` decisions
with the reference-library briefs, `api/` autodoc, `examples`); `docs/index.md` is the
landing toctree, built and published as a standalone Read-the-Docs site; the strict build
here is the CI gate. The API reference is generated from each subpackage's `__init__.py`
`__all__`, so docstrings ARE the docs.
`pandapower` is mocked at autodoc time; `pydantic`/`torch` are real. Keep schema docstrings
RST-safe IN SOURCE; `docs/conf.py` keeps only the Python-domain dedup hook and the
suite-wide `{doc}` reference resolver. A docs build with import errors or broken
autosummary is NOT done — keep new docstrings valid reStructuredText (wrap inline
math/identifiers containing `*` or trailing `_` in double backticks; `::` before indented
blocks; blank line before bullet lists; no explicit forward-ref quotes in annotations under
`from __future__ import annotations`). Figures are committed SVGs in `docs/_static/figures/`.

## Publishing
Distribution `power-grid-ml` on PyPI, import name `pgml` (unchanged). Version lives in
`src/pgml/__init__.py` (`pixi.toml`/`CITATION.cff` mirror it). A downstream package that
depends on pgml pins a `power-grid-ml>=X.Y,<X.Y+1` range and updates after a release.

## Delegation policy
- Delegate heavy, isolatable work to subagents (`.claude/agents/`); keep the orchestrator
  context lean. Keep nesting shallow (orchestrator → subagent). Do NOT fan out the
  numerically coupled core (assembly+solver) — that is one focused agent.
- After a subagent ships a module, record its PUBLIC SIGNATURES in that module's `CONTEXT.md`
  — that file is how the next agent learns the interface.
- **rtd-docs-builder** (`.claude/agents/`): owns the Sphinx pipeline + the authored docs
  under `docs/`. Run it at the END of any API-changing refactor (public signatures, `__all__`
  exports, or module docstrings change, or a module is added/removed): it re-authors the
  affected pages and validates with a clean local build mirroring CI. The docs build is a
  gate — an API change is not done until it passes.

@CONTEXT.md
@src/pgml/schemas/CONTEXT.md
