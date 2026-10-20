# Impedance scan

{func}`pgml.studies.impedance_scan` answers the first question of a harmonic planning
study: what impedance does the network present at a bus, over frequency, and where does it
resonate? A harmonic current source at a bus with a parallel resonance near one of its
orders produces a large harmonic voltage there; a series resonance is a low-impedance path
that draws harmonic current from elsewhere in the grid.

```python
from pgml.geometry import apply_default_harmonic_model
from pgml.grids import synthetic_feeder
from pgml.schemas import Phase, ShuntAppliance
from pgml.studies import impedance_scan

grid = apply_default_harmonic_model(synthetic_feeder(40))   # 20 kV feeder, 40 buses
abc = (Phase.A, Phase.B, Phase.C)
grid.appliances.append(                                     # 3 x 20 uF grounded-wye bank
    ShuntAppliance(id=900, node=12, phases=abc,
                   conductance_s=(0.0,) * 3, capacitance_f=(20.0e-6,) * 3)
)

scan = impedance_scan(grid, nodes=[0, 12])                  # orders 1 ... 50, 0.1 steps
z1 = scan.impedance(12)                                     # positive sequence [F], ohm
print(f"|Z1| at 50 Hz  {z1.abs()[0]:7.2f} ohm")
for r in scan.resonances(12):
    print(f"{r.kind:9s} h = {r.order:5.2f}  f = {r.frequency_hz:7.1f} Hz  |Z| = {r.z_ohm:8.2f} ohm")
```

```text
|Z1| at 50 Hz     1.69 ohm
parallel  h =  9.92  f =   496.0 Hz  |Z| =   139.45 ohm
```

The 2.5 Mvar bank tunes the feeder's source and line inductance to a parallel resonance
just below the 10th harmonic, where the positive-sequence impedance is about 80 times its
fundamental value.

## What is computed

For each scan frequency `f` the study assembles the nodal admittance `Y(f)` and solves
`Y(f) X = E` for the unit-current columns `E` of the requested buses' rows. The rows of `X`
at those buses are `Z(f) = Y(f)^-1` restricted to them:

- the phase-domain driving-point matrix of each bus, `[F, P, P]`
  ({attr}`~pgml.studies.ImpedanceScan.z_phase_ohm`): entry `(i, j)` is the voltage at
  phase `i` per ampere injected into phase `j` of the same bus;
- the positive- and zero-sequence driving-point impedance of every bus with phases a, b, c,
  the diagonal of `A^-1 Z_abc A` (Fortescue);
- with `transfer=True`, the transfer matrix between all requested node-phases, `[F, R, R]`.

The full inverse of `Y` is never formed. Frequencies are processed in chunks that fit the
memory budget `solver.harmonic.system_budget_mb`, each chunk as one batched factorization.

## Network model

By default `Y(f)` is the matrix the harmonic power flow solves at its harmonic orders
({func}`pgml.solver.assemble_harmonic_ybus`), evaluated at every scan frequency. At an
integer order the scan's `Y` is the flow's `Y(h)` for the same options, so the scan
describes the network that {func}`pgml.simulate` solves:

- Branches with their frequency-dependent models (`Line.harmonic_line_model`,
  transformer vector groups and winding-resistance laws). Hand-built grids should resolve
  the line model first, as above, with
  {func}`~pgml.geometry.apply_default_harmonic_model`; the converters do it for you.
- The source as its Norton impedance, always. An ideal slack is a short circuit and would
  make the scan zero at the source bus, so `SimulationConfig.slack` is not read. This is the
  source model of the harmonic power flow at every order above the fundamental. A source
  with zero impedance is rejected.
- Each load's harmonic Norton shunt, chosen by `load_shunt` exactly as in the harmonic
  power flow: `config.load_shunt` when a config is given, else the modeling default
  `appliance.harmonic_shunt.model`, shipped as `"opendss"`. That model splits the
  operating-point admittance `(P - jQ)/V²` into a parallel R-L branch and a series R-L
  branch (`series_rl_fraction`, 0.5 by default); `"motor"` replaces the series branch by a
  blocked-rotor reactance and `"none"` removes the derived shunt. A device's own
  `harmonic_model` overrides the choice. `operating_point` (or `config.operating_point`)
  sets the powers; `load_shunt_basis="nameplate"` uses the stored ones instead.
- Generators and storage follow `appliance.harmonic_shunt.generation_model` (shipped
  `"none"`: no shunt derived from their power, so they are open circuits in the scan; the
  load expression would give them a negative conductance). A device with an explicit
  `harmonic_impedance` is scanned with that impedance.
- Ideal switches and other zero-impedance branches are fused: buses joined by one share a
  single row and report the same impedance.

The scan has no fundamental solution, so every device shunt is built at the device's rated
terminal voltage. For a constant-power device that is exactly the harmonic flow's shunt; a
voltage-dependent (ZIP) or inverter-controlled device is named in a warning, because the
flow builds its shunt from the power at the solved voltage.

`load_shunt="const_z"` selects the simpler constant-impedance fold of
{func}`pgml.assembly.assemble_ybus` instead: the conductance `P/V²` flat in frequency and
the susceptance scaled as the equivalent inductance or capacitance, the
`series_rl_fraction = 0` limit of the `"opendss"` shunt and the most heavily damped of the
models. With it, `fold_generation=True` folds generators and storage too. Both models agree
at the fundamental and differ above it, most at a load-end parallel resonance: on the feeder
above the const-Z fold puts the peak at 490 Hz with `|Z| = 81.4 ohm`, the default model at
496 Hz with 139.5 ohm.

### Numerical singularity

A frequency at which `Y` is numerically singular is named in a warning: a lower bound of the
condition number of the equilibrated system, read from the solved columns at no extra cost,
times the unit roundoff exceeds 1, so no digit of the impedances there is guaranteed. The
usual cause is a part of the network without a path to the source that is held to ground
only by a very small shunt. A regular feeder stays far below the thresholds (4.5e15 in
complex128, 8.4e6 in complex64): the bound measures 3.5e4 on a 200-node feeder at orders 1
to 50.

## Frequency grid and resonances

The default grid runs from the fundamental to the 50th order in steps of 0.1 order
(491 points, 5 Hz at 50 Hz). Pass `frequencies_hz` for a different range or a finer grid
around a resonance of interest.

{meth}`~pgml.studies.ImpedanceScan.resonances` reports the interior local maxima of `|Z|`
as parallel resonances and the local minima as series resonances, for the positive sequence
by default (`component="zero"` or a phase letter for the others). An extremum counts only
when its relative prominence reaches `min_prominence` (default 0.05): for a peak, how far
it rises above the higher of the lowest points on either side before the curve climbs
higher again, as a fraction of the peak; for a dip, how far it falls below the lower of the
highest points on either side, as a fraction of that shoulder. Rounding noise on a flat
curve (about 1e-15 relative in complex128, 1e-7 in complex64) therefore produces no
resonance, and noise on the flank of a real one neither splits nor hides it. A run of equal
samples is one extremum at its middle sample; a lossless series resonance (`|Z| = 0`) has
prominence 1. Each {class}`~pgml.studies.Resonance` reports its prominence.

The frequency is refined between the scan points: near a lightly damped resonance `1/|Z|²`
(peak) and `|Z|²` (dip) are quadratic in frequency, so the vertex of a parabola through the
extremum and its neighbours lands within a small fraction of the step. A resonance much
narrower than the step can be missed; scan finer around it.

## Validation

- Closed forms, complex128: a source inductance in parallel with a capacitor, a tuned
  series branch, and a three-phase capacitor bank behind a source with different positive-
  and zero-sequence impedance. The scanned curves agree with the exact phasor expressions to
  4e-14 relative; on the default 5 Hz grid the refined resonance frequencies are within
  0.033 Hz of the closed-form extrema.
- OpenDSS 0.9.4 (opendssdirect.py), on a circuit without loads: the scan equals the dense inverse of OpenDSS's own
  `SystemY` of a three-phase feeder with a capacitor bank at 50 to 2450 Hz to 1.8e-13
  relative, element by element.
- Harmonic power flow: on a four-wire feeder with wye, delta, leading single-phase and
  motor-model loads, a generator with an explicit harmonic impedance and a capacitor bank,
  the driving-point impedance from the scan equals the block of `Y(h)^-1` of the harmonic
  flow's assembled system ({func}`pgml.solver.assemble_harmonic_system` at the solved
  fundamental) at orders 3 to 13, for the default, `"motor"`, `"none"` and both shunt
  bases: identical (0 difference) in complex128.
- Differentiability: a float64 gradcheck of the phase matrix and both sequence impedances
  with respect to line R, L and C, the source R and L and the load power, for the default
  model and the const-Z fold.

## Results and serialization

{class}`~pgml.studies.ImpedanceScan` holds CPU tensors. The impedances stay on the autograd
tape when the grid carried tensor-valued parameters, so a scan can sit inside an
optimisation, for example detuning a filter or sizing a bank against a target `|Z|` band.
{meth}`~pgml.studies.ImpedanceScan.to_dict` returns a JSON-ready dict (complex values split
into `*_re_ohm` / `*_im_ohm` lists, sequence impedances `None` on buses without a, b, c) and
{meth}`~pgml.studies.ImpedanceScan.from_dict` rebuilds the result from it.
