# Per-node harmonic disturbance source

A mode for injecting a defined harmonic disturbance at any node, whether or not a device
sits there. It answers questions of the form "inject a spectrum at each node in turn and
measure how far it spreads". It is separate from the device spectra of
`solve_harmonic_flow`, which are tied to a load's own fundamental current.

## The model

The disturbance is either a Thévenin source, an internal EMF `E(h)` behind a series
impedance `Z_s`, or a Norton current source `I(h)`. OpenDSS calls these a `VSource` and an
`ISource`. The shape per order comes from a user spectrum, usually a voltage spectrum for the
Thévenin form. The strength is a short-circuit power `S_sc`.

Because each harmonic order is an independent linear system, the source is added only at
orders above the fundamental. The fundamental power flow is therefore preserved exactly. A
tool that solves all frequencies in one circuit needs a fundamental-blocking reactor to
achieve the same thing.

## Parameters

| Parameter | Meaning |
|---|---|
| `node`, `phases` | Where to inject, phase to ground, one or several phases |
| `spectrum` | `{order: (magnitude_pu, phase_deg)}`, relative to order 1 |
| `source_power_va` | Source strength. Larger means stiffer, so more of the spectrum appears |
| `kind` | `"voltage"` for Thévenin, `"current"` for Norton |

## Definitions

Per order `h > 1` and per injected node-phase row, with `V1` the converged fundamental
voltage at that row, `V_base` the node's line-to-neutral base, and `mag_h`, `ang_h` the
spectrum entry:

- Source impedance, resistive and flat in frequency, `Z_s = V_base² / S_sc`, so
  `Y_s = S_sc / V_base²`.
- EMF referenced to the node's own fundamental, `|E_h| = (mag_h/mag_1)·|V1|` and
  `arg(E_h) = ang_h + h·(arg(V1) − ang_1)`. This is the phase convention the device
  injection uses, see [OpenDSS harmonics](references/opendss/harmonics.md).
- Norton current `I_N(h) = E_h · Y_s`.

Stamping follows from that. The Thévenin form adds `Y_s` to the diagonal of `Y(h)` and
`I_N(h)` to the right-hand side, so the node voltage becomes the divider
`V_node(h) = E_h·Z_net(h)/(Z_s+Z_net(h))`. The Norton form adds only `I_N(h)`, which is an
ideal current injection independent of the network.

A stiff source imposes its spectrum, `V_node(h) → E_h`. A weak source behaves like a current
injector and the node barely moves. Since `Z_s` is flat in frequency it adds no roll-off of
its own, so the injected spectrum is shaped only by the network response `Z_net(h)`.

## Gradients and sweeps

`source_power_va` and the spectrum may be tensors, and gradients flow to them as well as to
the grid parameters. Several sources at once are a list. A per-node sweep that places one
source at one node per scenario is the diagonal enumeration of that list.

The same mechanism carries the upstream background. `pgml.scenarios`'
`BackgroundHarmonicConfig` realizes one voltage-kind source per in-service `Source` node from
a config plus a seed, which is how a reproducible study supplies the distortion the upstream
network imposes. A voltage-kind source is an admittance as well as a current, so it cannot sit
on a row that bus fusion collapsed; such a placement is refused by name.

The OpenDSS equivalents for validation are an `ISource` carrying the spectrum for the Norton
form, and a `VSource` with the EMF set to the node's fundamental voltage, `MVAsc1 = S_sc` and
a resistive impedance for the Thévenin form.
