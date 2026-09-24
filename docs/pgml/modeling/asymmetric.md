# Per-phase and asymmetric modelling

What "asymmetric" means in power-grid-model, pandapower and OpenDSS, and what pgml does
instead. The five topics are the calculation mode, the load connection, single-phase loads,
the neutral and earth return, and per-phase harmonic injection. Converter-side detail is in
the [reference-library notes](references/index.md).

## Calculation mode

power-grid-model keeps one model and switches per call.
`calculate_power_flow(symmetric=True)` solves a positive-sequence single-phase equivalent,
and `symmetric=False` solves the full abc system, where every output gains a trailing phase
axis and node voltages become line-to-neutral. Symmetry is resolved per component as well. A
`sym_load` in an asymmetric calculation is split equally over the phases, and an `asym_load`
in a symmetric calculation is aggregated to the three-phase total.
([Calculations](https://power-grid-model.readthedocs.io/en/stable/user_manual/calculations.html))

pandapower has a separate entry point, `runpp_3ph`, which solves in the sequence frame. The
positive sequence goes through Newton-Raphson and the zero and negative sequences through
current injection, with earth return. A symmetric load or static generator is split into
thirds unconditionally, per-phase results land in `res_bus_3ph` and `res_line_3ph`, and the
network needs zero-impedance parameters added first.
([Three-phase power flow](https://pandapower.readthedocs.io/en/latest/powerflow/ac_3ph.html))

OpenDSS is always multi-phase in the phase domain and has no switch. A load's `kW` and `kvar`
are the total and are divided equally over its phases. Genuine imbalance is expressed with
separate single-phase load objects.

In pgml the network solve is always phase domain, so the mode is purely how an appliance's
operating point is resolved over its phases. `"symmetric"` splits each total equally, which
matches a `sym_load` in power-grid-model and `net.load` in `runpp_3ph`. `"asymmetric"` honours
per-phase nameplate fields and per-phase operating points, falling back to an equal split for
an appliance that has none. `"auto"`, the default, picks asymmetric as soon as any appliance or
operating point carries per-phase data, so a per-phase entry in the configuration promotes the
whole calculation even on a symmetrically defined grid. Imbalance only becomes visible on
genuinely multi-phase nodes, so a single-phase-equivalent grid from a converter has to be
expanded to abc nodes first.

`"symmetric"` is **not a guarantee of balanced solved voltages or currents**:
unequal phase impedances, single-phase devices and independent per-phase voltage
regulation remain in the phase-domain model. In particular, a PV generator's Q is
solved from its voltage constraints, not prescribed by its input Q fields.
Prescribed Q overrides on such a generator are ignored with a warning by the
fundamental solver, regardless of symmetry. In direct power resolution, an ordinary
per-phase Q override is averaged in symmetric mode, with a warning that its phase
allocation is not accepted.

`PowerFlowResult.resolved_operating_point()` produces distinct solver-readout
entries for downstream harmonic initialization and device-current recovery. Only
these entries preserve the **solved** Q allocation in symmetric mode; a plain
user-supplied dictionary does not bypass symmetric splitting. If symmetric inputs
nevertheless produce unequal solved PV Q, the solver warns explicitly. Averaging
that output after convergence would make its device currents inconsistent with
the solved network. Enforcing a balanced-output model would instead require a
balanced network/compatible constraints or a separate sequence-reduced formulation.
For equal per-phase generator Q, `VoltageRegulation(regulated="positive_sequence")`
(the default) uses one total Q with equal shares and regulates the positive-sequence
voltage magnitude. `regulated="per_phase"` instead regulates each phase magnitude
independently and can require unequal Q. This choice belongs in the fundamental
regulation constraints, not in a post-solve averaging of the readout. Neither choice
makes arbitrary network voltages balanced.

## Load connection

power-grid-model has no connection field on loads or generators. Every load is wye and
injects at the node. Its winding enum applies to transformers only.

pandapower gives `net.asymmetric_load` a `type` of `"wye"` or `"delta"`. Wye uses the
phase-to-earth voltage directly. Delta converts node voltages to line-to-line with the
circulant difference matrix, computes the loop currents, then maps back to line currents. Its
own documentation notes that the phase-to-earth type is called wye because neutral and earth
are treated as the same point.

OpenDSS has `Load.conn` of `wye` or `delta`, defaulting to wye. A wye load has one conductor
more than it has phases and injects between each phase conductor and the neutral conductor. A
delta load injects from conductor `i` into `i+1`, wrapping around. The base voltage differs
too. A wye load with two or more phases uses `kVLL/√3`, while a delta load and a single-phase
wye load use the supplied kV directly.
([Neutral rules](https://opendss.epri.com/NeutralRules.html))

pgml uses one connection-aware terminal incidence `M` per appliance, which is the union of
those two rules. WYE maps each phase terminal to its phase row and returns either into the
node's `Phase.N` row, when the node has one, or to ground. With no neutral row the matrix
reduces to the identity and the stamp becomes the plain diagonal one. DELTA maps terminal `k`
to the pair `(phase_k, phase_{k+1})`, with the loop order taken from the appliance's `phases`
tuple. A constant-impedance shunt is stamped as `Mᵀ·diag(y)·M` and a ZIP current as
`I_node = Mᵀ·i_term` with `V_term = M·V_node`. DELTA needs at least two phases, and zigzag is
a transformer connection that appliances reject.

## Single-phase loads

power-grid-model has no node-phase concept, since a node is always a three-phase busbar. A
phase-A load is written as a per-phase vector with zeros elsewhere. pandapower has no
dedicated single-phase element either, so the unused phases of an asymmetric load are zeroed.

OpenDSS uses `phases=1` with a bus string naming the phase and the return, for example
`Bus.1.0`. The European LV residential default is a single-phase wye element whose `kV` is the
line-to-neutral voltage with no √3 applied. A two-conductor `phases=1` element routes the
return through the neutral wire, which is what the EPRI maintainers recommend over `phases=2`.

In pgml a single-phase load is a one-element `phases` tuple plus a connection. With no
connection given, the configuration default applies, which is wye on the line-to-neutral base.
The return follows the connection rule above.

## Neutral and earth return

None of the three tools solves an explicit neutral conductor in the load flow. power-grid-model
is three-wire and Kron-reduces a four-conductor series matrix to three before solving;
otherwise the zero-sequence parameters carry the earth return. pandapower works in the
sequence domain, where the neutral current is the derived residual `i_a+i_b+i_c = 3·i_0`.
OpenDSS can keep an explicit neutral as a fourth conductor node, but the shorthand bus string
grounds it to node 0, which is excluded from the system admittance and therefore shorts any
neutral impedance.

pgml follows the phase domain. A neutral exists only when a node carries `Phase.N` in its
`phases`, and it is then a solved row that a WYE appliance returns into, the genuine four-wire
case. Without that row a WYE appliance returns to ground, which reproduces the other three
tools exactly. There is no separate neutral solver.

The node-level rule is a default rather than a constraint. A four-wire bus in the field mixes
a solidly grounded element with a neutral-returning one on the same node, which OpenDSS
expresses by writing one element with three conductors and another with an explicit neutral
tie. Every injection appliance therefore carries `return_path`, one of `"auto"`, `"neutral"`
or `"ground"`. `"ground"` pins the return to true ground even on a node that has a neutral
row, and `"neutral"` requires the node to carry one. Two appliances on one node can then stamp
with different incidence matrices, `M = I_n` against `M = [I_n | -1]`, so assembly groups
appliances by connection, phase count and effective return. The OpenDSS reader reproduces each
element's own resolved return conductor rather than applying one rule per bus. The field is
meaningless for DELTA, where a non-default value raises.

### Where a four-wire neutral is grounded

A bus that carries `Phase.N` gets its own matrix row, and nothing ties that row to the ground
reference implicitly. A WYE appliance's return current flows into it, a four-conductor line
carries it as a series conductor, and the transformer stamp only touches the phases it is wired
to. The ground tie has to come from an explicit element. On an OpenDSS import that is a
`Reactor` on the neutral conductor (`bus1=bus.4`), which converts to a phase-to-ground shunt on
`Phase.N`. With that tie the neutral voltages of a three-bus four-wire feeder reproduce a live
OpenDSS solve to 5e-10 V. Without any tie the neutral rows have no reference, the admittance
matrix is rank-deficient, and the solve reports non-convergence rather than an answer, because
pgml has no anti-float stabiliser where OpenDSS adds a `ppm_antifloat` shunt. A transformer
winding wired to an explicit fourth conductor is refused by the converter rather than silently
re-grounded. The pandapower converter never emits `Phase.N`, so four-wire grids come from
OpenDSS or from hand-built input.

## Per-phase harmonic injection

An OpenDSS load carries exactly one spectrum, and one complex multiplier per order applies to
every phase. Per-phase asymmetry enters only through the per-phase fundamental phasors
captured before the harmonic solve. For phase `i` the harmonic current is
`|I_h^i| = (mag_h/mag_1)·|I_1^i|` with `arg = ang_h + h·(a1_i − ang_1)`. A multi-phase load is
therefore inherently balanced in its spectrum, and genuinely different per-phase spectra need
three separate single-phase loads with their own spectra.
([Harmonics load modelling](https://opendss.epri.com/HarmonicsLoadModeling.html))

pgml keeps that convention for a device-level `spectrum`, which is the shorthand for the same
spectrum on every phase, and adds `spectrum_per_phase` plus a phase axis on the
`harmonic_injection` override for a genuinely asymmetric device. `I(h)` is built per phase
from each phase's own fundamental phasor either way. See
[OpenDSS harmonics](references/opendss/harmonics.md) for the order-by-order convention.

## Sources

- power-grid-model documentation, the calculations, components and data-model pages, with the
  enums and dtypes checked against the installed package.
- pandapower documentation for `runpp_3ph` and `asymmetric_load`, with the three-phase load
  mapping read from its source.
- OpenDSS documentation from EPRI, the load, ZIP, neutral-rules and harmonics pages, with the
  load element's own source for the stick-current and harmonic-mode rules, and the EPRI forum
  threads on single-phase neutrals and unbalanced spectra.
