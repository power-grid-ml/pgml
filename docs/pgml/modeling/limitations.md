# Limitations

This page lists what pgml does not model, or models only approximately, so that a result is
never read as more physical than it is. Each entry says what is left out, how large the effect
is where it has been measured, and what to do about it. The modelling pages linked from each
entry carry the derivations.

pgml is a steady-state, frequency-domain tool: it solves the fundamental power flow and the
harmonic system `Y(h)·V(h) = I(h)` at integer harmonic orders. Flicker, transients,
saturation and inrush are out of scope. {func}`pgml.solve_harmonic_flow` raises on a
non-integer order; {func}`pgml.studies.emission_flow` solves arbitrary frequencies
(interharmonics and supraharmonics) with the inverter models of {mod}`pgml.inverter`.

## Transformers

**Frequency dependence of the winding resistance.** The harmonic resistance is
`R(f) = R · m(f)`, times `f/f0` when `harmonic_xr_constant` is set. By default the reactance
grows with the order at constant resistance, as in OpenDSS (`XRConst=No`); with
`harmonic_xr_constant=True` the X/R ratio is held constant instead, which agrees with
OpenDSS's own `Yprim` at orders 1, 5 and 13. `resistance_frequency` accepts a constant, the
Carson skin law or a sampled curve. No eddy-current or stray-loss curve ships as a default, so
unless you supply a measured `resistance_frequency` curve a transformer's winding resistance is
constant with frequency.

**Magnetizing branch placement.** Where the magnetizing shunt attaches is a modelling choice,
`transformer.magnetizing_placement`, because the three reference tools disagree:

- `split` (the default) puts half on each terminal, as power-grid-model does, and reproduces
  power-grid-model to machine precision;
- `to_terminal` attaches the whole branch to the last winding's terminal, which is OpenDSS's
  own placement and agrees with a live OpenDSS solve to 3e-10 to 2e-9 pu;
- `from_terminal` puts it on the HV diagonal, so the core loss does not depend on loading.

Against OpenDSS on a 500 kVA 20/0.4 kV unit (Dyn and YNyn, 0 to 500 kW load) `from_terminal`
deviates by 9.2e-5 pu at a no-load current of 0.1 %, 2.2e-4 pu at 0.5 % and 8.2e-4 pu at
2 %, and `split` by about half of that. The reference presets select each tool's own
placement. No placement has a zero-sequence magnetizing branch. See {doc}`transformer`.

**Construction.** Every two-winding pairing except zigzag-zigzag is supported, at every clock
number of the pairing's parity. Zigzag-zigzag and non-solid neutral grounding raise. The zigzag
incidence is experimental: it reproduces the clock shift, the blocked zero-sequence transfer
and the winding's own zero-sequence path, and agrees with power-grid-model on an unbalanced
solve, but no second reference tool can express the same unit as one two-winding element.
Constructing one logs a warning once per process. Three-winding units are not modelled.

**Zero sequence.** The zero-sequence leakage is `Transformer.zero_sequence` when set (the
pandapower converter reads `vk0_percent`/`vkr0_percent`), otherwise the
`transformer.zero_sequence.*` ratios of the modelling defaults (1.0, that is `Z0 = Z1`, which
is what OpenDSS and power-grid-model imply). Not modelled: a zero-sequence magnetizing branch
(pandapower `mag0_percent`/`mag0_rx`, power-grid-model `i0_zero_sequence`/`p0_zero_sequence`),
the HV/LV split of the zero-sequence leakage inside a T equivalent (pandapower
`si0_hv_partial`), and a neutral earthing impedance (pandapower `xn_ohm`/`rn_ohm`, OpenDSS
`Rneut`/`Xneut`). The converters name each of these in a warning, or refuse the element,
instead of dropping it silently. pgml has no explicit transformer T equivalent; the
differences to a T-model tool are reported by the converters.

**Ungrounded secondaries.** A fully ungrounded secondary island has no absolute zero-sequence
reference. Its line-to-line voltages are correct, but its absolute voltages are undetermined
as the load goes to zero.

**Single-phase equivalent units.** A single-phase (positive-sequence equivalent) transformer
has no zero-sequence topology, so triplen orders pass it instead of being blocked by a delta
winding. pgml warns once when this happens; use the three-phase model to study triplen
blocking.

## Loads and generation

**Harmonic behaviour of loads.** Each load, generator and storage unit is a harmonic current
source. A load carries the OpenDSS device shunt in parallel by default
(`appliance.harmonic_shunt.model = opendss`, OpenDSS's `NeglectLoadY=no` with
`%SeriesRL=50`); `none` gives the pure current-source model and `motor` the blocked-rotor
variant. The shunt is derived from the power the device actually draws at the converged
fundamental, where OpenDSS uses the specified power. The two agree for a constant-power device
and differ by 6e-5 to 1e-4 pu of nominal for a constant-impedance, constant-current or ZIP
device on IEEE 33-bus. pgml has no per-device rated voltage: the shunt's rated voltage comes
from the host node, while OpenDSS uses each load's own `kV`, so an imported circuit whose load
`kV` differs from its node's nominal carries a correspondingly different shunt.

**Harmonic behaviour of generation.** A generator or storage unit carries no load-style shunt
by default (`appliance.harmonic_shunt.generation_model = none`): `conj(S)/V_rated²` would have a
negative conductance for an injecting device, which adds energy instead of damping it. A
passive internal impedance (an inverter's filter, a machine's subtransient reactance) can be
given per device with the `harmonic_impedance` field; leave it absent when it is unknown. See
{doc}`der-pv-storage`.

**Inverter harmonic models.** {func}`pgml.solve_harmonic_flow` treats an inverter as a Norton
current source with a fixed spectrum and an optional passive R-L impedance; it models no
controller-dependent behaviour, no coupling between frequencies (orders are solved
independently), and no harmonic headroom or total-current limit. The passive impedance is
valid only where the controller's influence on the harmonic response can be neglected; pgml
does not check that range. A physical Norton model of a grid-following inverter exists in
{mod}`pgml.inverter` and is used by {func}`pgml.studies.emission_flow` (see
{doc}`inverter-emission`). It still leaves out frequency coupling (the mirror frequency),
common-mode PWM emission through the EMC ground capacitance, regularly sampled PWM and
overmodulation, MPPT patterns other than perturb and observe, and nonlinear saturation.

**Voltage-regulating generators.** The regulated row pair is formed for a wye terminal that
returns to ground. A delta machine, a wye machine returning through its node's neutral row, a
positive-sequence setpoint on a two-phase terminal, a regulating generator on a source's node,
and two regulating generators on one node all raise `ModelingError` (the pandapower converter
merges co-located regulating rows instead). A remote regulated bus is not supported. Reactive
limits bind by switching, so the solution is exact at the limit but the active set is
piecewise constant in the parameters: at a switching boundary the gradient is one-sided.
Regulation is a fundamental-frequency concept. At harmonic orders the machine is a Norton
current source scaled from its solved reactive power. See {doc}`der-pv-storage`.

**ZIP loads next to generation.** pandapower reduces a ZIP load's coefficients onto the bus and
applies them to the bus's net injection, so a constant-impedance load and a generator on one
bus cancel before the voltage dependence is applied. pgml keeps the devices separate, which is
the physical model: a constant-impedance load and a constant-power inverter cancel only at
nominal voltage. On an LV feeder where each inverter mirrors its bus's load the two tools
differ by 3.9e-3 pu at the affected buses, and agree to 5e-8 pu with the generation stopped.
Keep this in mind when pandapower serves as ground truth for such a grid.

**Voltage limits of loads.** The schema has no load voltage-limit model. OpenDSS
`Vminpu`/`Vmaxpu`/`Vlowpu` and CVR settings are not read; the converter warns when a
constant-power or constant-current load carries voltage limits. A piecewise voltage-band
load model is not available.

## Sources

The per-phase Thevenin source is sequence-aware: the converters read the native zero-sequence
data (OpenDSS `Vsource.R0`/`X0`, power-grid-model `source.z01_ratio`, pandapower
`ext_grid.x0x_max`/`r0x0_max` with `s_sc_max_mva`/`rx_max`), and otherwise the
`source.zero_sequence.*` ratios apply (`Z0 = Z1`) with a warning that names the element. The
negative sequence is always `Z2 = Z1`, as for a passive upstream network; a rotating-machine
source with `Z2 ≠ Z1` is not modelled. pandapower's `runpp_3ph` instead pins the positive
sequence, puts the short-circuit impedance into the negative-sequence network and scales its
zero-sequence shunt by the IEC factor `c = 1.1`; `tests/reference/test_pandapower_source_zero_sequence.py`
quantifies both differences.

## Lines, switches and shunts

**Conductor geometry.** The Carson/Deri line model has no conductor temperature dependence and
no sub-conductor bundling, and assumes transposition as documented in
{doc}`harmonic-line-model`. The `gmr_skin` and `bessel` internal-inductance models treat a
stranded conductor as the equivalent solid round conductor; on the published ACSR data of the
OpenDSS line-constants example the spread is under 2 % of `X` up to 2.5 kHz. Lines with a very
low X/R ratio hit the GMR floor when geometry is synthesized (they are flagged
`synth_unphysical`), two-phase lines are skipped by `synthesize_grid_geometry`, and OpenDSS's
`capradius` option is not read.

**Zero-sequence impedance of lumped lines at harmonics.** A line with `conductor_geometry`
computes `Z(h)` from first principles. A line without geometry embeds the earth return at the
fundamental, and extrapolating its `Z0` to higher orders is an assumption in every tool. pgml's
lumped `sequence_aware` model adds the Carson earth-return resistance to `R0` and scales `X0`
with the order; OpenDSS's R/X line does the same with its `Rg`, and additionally bends `X0`
sub-linearly with `Xg`. With matched earth parameters the two agree to about 1e-10 relative,
so the remaining divergence is a choice of parameters, not of formula. The sub-linear law is
available as `line.earth_return.x0_frequency: carson_sublinear`. Supply conductor geometry when
the zero-sequence earth return must be right. See {doc}`harmonic-line-model`.

**Missing zero-sequence line data** is filled with overhead-line ratios (`R0/R1 = 4`,
`X0/X1 = 3`, `C0/C1 = 0.5`, from the modelling defaults), which are weak for cables. The
converters warn once per grid when they use them.

**Switches with a resistance.** The pandapower converter reads a closed bus-bus switch's
`z_ohm` as a pure resistance. pandapower itself splits that value into R and X at its
`switch_rx_ratio` (default 2). On a low-reactance cable feeder a 1 mΩ switch caused a 0.0125°
angle difference from pandapower. Switches with `z_ohm = 0` (every pandapower built-in) are
ideal in both tools and are fused exactly.

**Switch state.** The grid stores each switch's default state (`Switch.default_closed`),
because switch states determine the topology. The state of a calculation is an input of that
calculation: `branch_states` overrides the default per solve without editing the grid. A
branch whose state is swept per scenario (`branch_states`) is never fused, because its state
scales a stamped admittance block, so an ideal zero-impedance switch cannot be swept: it
raises `ModelingError`. Give a swept switch a finite impedance.

**Shunt reactors with a series resistance.** The shunt primitive is the parallel form
`G + 1/(jωL) + jωC`. A lossless reactor, or a pandapower inductive shunt, maps exactly at every
order. An OpenDSS reactor with a series `R > 0` is converted to the equivalent parallel pair at
the fundamental (the converter warns), so its loss term stays flat with frequency where the
series branch decays as `1/h²`.

## Harmonic spectra

EN 50160 gives supply-voltage compatibility levels (orders 26 to 49 are a marked extension).
They describe source or background distortion, not device emission. Device current spectra
default to IEC 61000-3-2 per emission class A to D. There is no lighting consumer type, so
class C is never selected automatically.

## Numerics

**Single precision.** pgml solves in SI units without per-unit normalisation, and the diagonal
equilibration removes most of the resulting badly scaled conditioning, but not all of it. A
low-voltage harmonic system stays near a condition number of 1e7 because that conditioning is
physical, not a scaling artefact. A plain complex64 solve therefore keeps only about
`7 − log10(cond)` digits: measured against complex128 at the fundamental, 2e-6 to 3e-5 pu,
against 1e-14 to 1e-12 pu for `precision="mixed"`. pgml logs a one-time warning with the
estimate, and a solve that stalls at that floor reports `diagnostics.floor_governed`. Solve at
complex128, or with `precision="mixed"`, and store the results as complex64. See
{doc}`solver-performance`.

**Anchored solves.** `solve_anchored` and `AnchoredSystem` factor the unscaled matrix; they do
not apply the equilibration the other solve paths use.

**Criticality analysis.** The Jacobian-based criticality analysis describes the hardest
scenario of a batch, and is skipped for real state sizes above `2N = 4000`.

**Loadability.** `loadability_limit` steps and bisects on Newton feasibility, so its breaking
load factor is a lower bound on the nose of the PV curve (about 4 % below the closed form on a
two-bus feeder). It is not an arc-length continuation.

## Converter coverage

Converted from pandapower: `bus`, `line`, `load`, `asymmetric_load`, `trafo` (vector groups and
taps, honouring `tap_changer_type` as pandapower does), bus-bus and line/trafo `switch`
(including `parallel`), `ext_grid`, `sgen`, `gen` (a voltage-regulating terminal by default),
`storage` and `shunt`. From OpenDSS: `Line`, two-winding `Transformer`, `Vsource`, `Load` (ZIP
models), `Capacitor`, `Reactor`, `Generator` (including `model=3`), `PVSystem` and `Storage`.
From power-grid-model: `node`, `line`, `sym_load`, `asym_load`, `source`, `sym_gen`,
`transformer`, `link` (an ideal closed switch) and `voltage_regulator`.

Not converted; a warning names every non-empty dropped kind:

- pandapower: `trafo3w`, `impedance`, `ward`, `xward`, `dcline`, `motor`, `asymmetric_sgen`,
  `measurement`, and controllers other than the mapped DER controls (for example tap
  controllers and `CharacteristicControl`);
- OpenDSS: three-winding transformers, `RegControl`, `XfmrCode` and frequency-correction
  curves, explicit non-solid neutral nodes, unmapped `InvControl` modes, and `monVoltageCalc`;
- power-grid-model: `three_winding_transformer`, `shunt`, `asym_gen`,
  `transformer_tap_regulator`.

pandapower ideal phase-shifter taps raise, and a tap position whose `tap_changer_type` is unset
is dropped with a warning, as pandapower ignores it too. Neither the pandapower nor the
power-grid-model converter emits a neutral phase; a four-wire grid with an explicit neutral
conductor comes only from the OpenDSS converter's `THREE_PHASE` mode or a hand-built grid.
