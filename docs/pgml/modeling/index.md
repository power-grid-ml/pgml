# Modeling decisions

These pages record *how* pgml models a power grid and *why* — the conventions, the
physics, and the deliberate choices behind each part of the simulation. They are the
reference behind the [concepts](../concepts.md) overview: read concepts first for the
big picture, then come here for the derivations, cross-tool comparisons, and validation
detail.

Everything here is about the **simulation and modeling** layer (`pgml`). The learning
(`pgl`) and generation (`pgg`) packages have their own sections.

## Conventions and representation

- **[Cross-tool conventions](conventions.md)** — pgml's canonical internal definitions
  (base voltage L-L/L-N, transformer referral, power signs, units, earth return) and how
  they differ from pandapower, OpenDSS, and power-grid-model. Read this before touching the
  converters, the slack, or transformers.
- **[Asymmetric / per-phase modeling](asymmetric.md)** — symmetric vs asymmetric
  calculation, WYE/DELTA connections, single-phase loads, the neutral/earth return, and
  per-phase harmonic injection.

## Components and physics

- **[Two-winding transformer (vector groups)](transformer.md)** — the winding-incidence
  primitive `Y = NᵀY_winding N` and why a Dyn delta correctly traps triplen harmonics.
- **[Harmonic line model](harmonic-line-model.md)** — frequency scaling of a line's
  sequence impedance, and why the earth-return term belongs to the zero sequence.
- **[DER: PV, generators, storage](der-pv-storage.md)** — inverter control laws
  (Volt-VAr / Volt-Watt / power factor), storage and state of charge, and how the smooth
  control laws fold into the differentiable solve.
- **[Per-node harmonic disturbance source](error-injection.md)** — injecting a defined
  harmonic "error" at any node, independent of whether a device sits there.
- **[Harmonic emission of a device](harmonic-emission.md)** — the measured complex affine
  law `I_h(λ) = A_h + B_h·λ` (load-independent floor, phase rotation with loading,
  cancellation null) as the correction both scenario recipes apply to the proportional
  draw, and the sequence structure of a device's harmonics across its phases.

## Solver architecture

- **[Performance and structural checks](solver-performance.md)** — why the solve is
  organised the way it is: factor-once/solve-many, the precomputed injection plan, the
  measured dense/sparse crossover (and why CUDA stays dense), threaded back-substitution,
  the pre-solve connectivity check compared to pandapower / power-grid-model / OpenDSS,
  switch states as differentiable admittance scaling, disjoint-union grid ensembles, and
  the single/double-precision policy.

## Reference libraries

- **[Reference-library notes](references/index.md)** — distilled briefs on the tools pgml
  converts from and validates against (OpenDSS, pandapower, power-grid-model), including the
  empirically-verified OpenDSS harmonic and Carson conventions pgml reproduces.

```{toctree}
:hidden:

conventions
asymmetric
transformer
harmonic-line-model
der-pv-storage
error-injection
harmonic-emission
solver-performance
references/index
```
