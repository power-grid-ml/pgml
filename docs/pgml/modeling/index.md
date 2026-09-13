# Modelling decisions

These pages record how a power grid is modelled and why. They sit behind
{doc}`../concepts`, which is the short version. Read them when a number needs explaining,
when you convert a network from another tool, or when you need to know what the model leaves
out.

Each page states its own limits. Nothing here is hidden behind a default.

## Conventions and representation

- {doc}`conventions` pins the internal definitions, base voltage, transformer referral, power
  signs, units and earth return, and records how pandapower, OpenDSS and power-grid-model
  differ. Read it before converting a network.
- {doc}`asymmetric` covers symmetric against asymmetric calculation, wye and delta
  connections, single-phase loads, the neutral and the earth return, and per-phase harmonic
  injection.

## Components and physics

- {doc}`transformer` derives the winding-incidence primitive `Y = NᵀY_windingN` and shows why
  a Dyn delta traps triplen harmonics.
- {doc}`harmonic-line-model` explains how a line's impedance scales with frequency, and why
  the earth-return term belongs to the zero sequence.
- {doc}`der-pv-storage` covers inverter control laws, storage and dispatch, and which parts of
  them carry gradients.
- {doc}`error-injection` describes injecting a defined harmonic disturbance at any node.

## Solver

- {doc}`solver-performance` explains how the solve is organised, factor once and solve many,
  the measured dense and sparse crossover, switch states as differentiable admittance scaling,
  grid ensembles, the structural checks run before any numerics, the per-unit convergence
  criteria, the diagonal equilibration of every factorization, the working precision and what
  a gradient costs.

## Reference tools

- {doc}`presets` selects reference model choices for conformance comparisons.
- {doc}`references/index` holds short briefs on the tools pgml reads and validates against,
  OpenDSS, pandapower and power-grid-model, including the OpenDSS harmonic and Carson
  conventions that pgml reproduces.

```{toctree}
:hidden:

conventions
asymmetric
transformer
harmonic-line-model
der-pv-storage
error-injection
solver-performance
references/index
presets
```
