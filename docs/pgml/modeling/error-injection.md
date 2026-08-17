# Error-injection mode: per-node harmonic disturbance source

A modeling mode for injecting a defined harmonic "error" (disturbance) at ANY node of
the grid — independent of whether a load/generator sits there. Used for studies like
"inject a spectrum at each node and measure how it spreads" (state-estimation training,
sensitivity maps). Distinct from the device spectra in `solve_harmonic_flow`
(`harmonic_injection=...`), which are tied to a load's fundamental current.

## The physical model

The disturbance is a **Thévenin harmonic source** at the node: an internal EMF `E(h)`
behind a series impedance `Z_s` (a `VSource` in OpenDSS terms), OR a **Norton current
source** `I(h)` (an `ISource`). The error magnitude per order comes from a user
**spectrum** (the disturbance is typically a VOLTAGE spectrum); the **source strength** is a
short-circuit power `S_sc` (MVAsc).

### Why no damping reactor is needed in pgml
OpenDSS solves all frequencies in one circuit, so a transient `VSource` would perturb
the fundamental; the established workaround places a 50-Hz-only reactor to cancel it
(and that reactor's resistance formula was found to be dimensionally inverted — see
"OpenDSS equivalence"). **pgml solves each harmonic as an independent linear system**
(`Y(h)·V(h)=I(h)`), so the source is added ONLY at `h>1` and the fundamental power flow
is left untouched — the fundamental is preserved EXACTLY, with no reactor and no
approximate cancellation.

## Parameters
- `node` (+ `phases`): where to inject (phase-to-ground, one or more phases).
- `spectrum`: `{order: (magnitude_pu, phase_deg)}`, magnitudes relative to the
  fundamental (order 1 = reference); a VOLTAGE spectrum for the voltage source.
- `source_power_va` (MVAsc): the source STRENGTH. Larger ⇒ stiffer ⇒ more of the
  spectrum actually appears at the node.
- `kind`: `"voltage"` (Thévenin) or `"current"` (Norton).

## Definitions (per harmonic `h > 1`, per injected node-phase row)
Let `V1` = the converged fundamental voltage at that row, `V_base` = the node's
line-to-neutral base, `mag_h/ang_h` the spectrum entry, `mag_1/ang_1` its order-1 entry.

- **Source impedance (resistive / frequency-flat, i.e. `x1r1≈0`):**
  `Z_s = V_base² / S_sc`  (real, constant in `h`);  `Y_s = 1/Z_s = S_sc / V_base²`.
- **EMF (reference = node fundamental voltage):**
  `|E_h| = (mag_h/mag_1)·|V1|`,
  `arg(E_h) = ang_h + h·(arg(V1) − ang_1)`  (same phase convention as the device
  injection in [OpenDSS harmonics](references/opendss/harmonics.md)).
- **Norton current:** `I_N(h) = E_h · Y_s`.

### Stamping into the linear system `Y(h)·V(h)=I(h)`
- `kind="voltage"` (Thévenin): add `Y_s` to the diagonal `Y(h)[row,row]` AND `I_N(h)` to
  `I(h)[row]`. The node harmonic voltage is then the divider
  `V_node(h) = E_h·Z_net(h)/(Z_s+Z_net(h))` — a finite-strength source.
- `kind="current"` (Norton): add `I_N(h)` to `I(h)[row]` only (no shunt) — an ideal
  current injection independent of the network.

Stiff source (large `S_sc` ⇒ small `Z_s`) ⇒ `V_node(h) → E_h` (imposes the spectrum);
weak source ⇒ behaves like a current injector and the node sees little. `Z_s` is flat in
frequency (resistive), so it adds NO frequency-dependent roll-off — the injected spectrum
is shaped only by the grid's own response `Z_net(h)`, which is what the study wants.

## Differentiability / batching
`source_power_va` and the spectrum may be tensors; gradients flow to them and to grid
parameters. Multiple simultaneous sources are a list. A per-node SWEEP (one node per
scenario) is the diagonal enumeration analogue of `spectrum_sweep`/`perturbation_sweep`.

## OpenDSS equivalence (the live oracle)
- `kind="current"` ↔ an OpenDSS `ISource` with the spectrum (clean; injects only the
  specified harmonic current).
- `kind="voltage"` ↔ an OpenDSS `VSource` carrying the spectrum with EMF set to the
  node's fundamental voltage and `MVAsc1 = S_sc`, `x1r1≈0` (resistive). To keep the
  fundamental undisturbed in OpenDSS's all-frequency solve one must cancel the source's
  50-Hz component; the prior approach used a 50-Hz-only reactor whose resistance was
  `1e6·MVAsc/V²` — **dimensionally an admittance used as ohms (the reciprocal of the
  intended `V²/(MVAsc·1e6)`); a real bug** that corrupts the cancellation. The oracle
  avoids it (pgml-style: only harmonics are injected / a fundamental-free spectrum).
