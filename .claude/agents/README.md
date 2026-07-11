# Subagents: the development engine

These agents exist to outsource focused, isolatable work so the orchestrator's context stays
lean. They are organised by **capability (a verb), not by package (a noun)**: what to build
and every library-specific fact are supplied at invocation by the orchestrator's task and by
the repo's own `CLAUDE.md` / `CONTEXT.md` / `STATUS.md` / `docs/`, which each agent reads on
entry. Nothing library-specific is pinned into an agent file — that is what keeps a growing
codebase from leaving these descriptions stale.

## The roster

| Agent | Capability | Reads for specifics |
|---|---|---|
| `ml-library-architect` | Design / architecture review (structure, schemas, API surface, portability) | the repo it is pointed at |
| `implementer` | Build a module/feature in the package the orchestrator names | that package's `CONTEXT.md`/`STATUS.md`, `docs/<pkg>/`, `CLAUDE.md` |
| `code-reviewer` | Read-only pre-integration review of a finished change | `CLAUDE.md` + the module `CONTEXT.md` (the standard is learned, not hard-coded) |
| `test-runner` | Run + judge the suite and the project's correctness gates | `tests/CONTEXT.md`, `CLAUDE.md` |
| `reference-integrator` | Converter + oracle test for ONE external tool (OpenDSS / pandapower / power-grid-model / …) | that tool's brief under `docs/pgml/modeling/references/<tool>/` |
| `rtd-docs-builder` | Author + validate the Sphinx/RTD docs | `docs/`, the changed public surface |

## Conventions for editing or adding an agent

- **Description = router, body = method.** The `description:` frontmatter states *when to reach
  for the agent* (the trigger the orchestrator matches on), tersely. The body states *how the
  agent works*. Neither enumerates the current module tree or restates the hard constraints —
  those live in the repo and drift; point to them instead.
- **One capability, parameterised.** Prefer a single agent told which package/tool to act on
  over N package-welded clones. Add a new agent only for a genuinely new *kind* of work.
- **Keep nesting shallow** (orchestrator → subagent) and keep the numerically coupled core
  (assembly + solver) as one focused `implementer` task — do not fan it out.
