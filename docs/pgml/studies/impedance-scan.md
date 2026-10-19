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
parallel  h =  9.80  f =   490.0 Hz  |Z| =    81.38 ohm
```

The 2.5 Mvar bank tunes the feeder's source and line inductance to a parallel resonance
just below the 10th harmonic, where the positive-sequence impedance is about 50 times its
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

`Y(f)` is {func}`pgml.assembly.assemble_ybus`, the linear model of the grid:

- Branches with their frequency-dependent models (`Line.harmonic_line_model`,
  transformer vector groups and winding-resistance laws). Hand-built grids should resolve
  the line model first, as above, with
  {func}`~pgml.geometry.apply_default_harmonic_model`; the converters do it for you.
- The source as its Norton impedance, always. An ideal slack is a short circuit and would
  make the scan zero at the source bus, so `SimulationConfig.slack` is not read. This is the
  source model of the harmonic power flow at every order above the fundamental. A source
  with zero impedance is rejected.
- Loads folded as constant impedances at the operating point: the conductance `P/V²` is
  flat in frequency and the susceptance scales as the equivalent inductance or capacitance.
  `operating_point` (or `config.operating_point`) sets the powers; the nameplate is the
  default.
- Generators and storage are open circuits unless `fold_generation=True`. Their
  constant-impedance image has a negative conductance, an artefact of the load expression
  rather than a property of the inverter or machine. The harmonic power flow makes the same
  choice for its device shunt.
- Ideal switches and other zero-impedance branches are fused: buses joined by one share a
  single row and report the same impedance.

The device model is the constant-impedance fold of `assemble_ybus`, not the harmonic device
Norton shunt that {func}`~pgml.solver.solve_harmonic_flow` stamps (`load_shunt`). Both
reduce to the same conductance at the fundamental; above it they describe the load's
reactive part differently.

## Frequency grid and resonances

The default grid runs from the fundamental to the 50th order in steps of 0.1 order
(491 points, 5 Hz at 50 Hz). Pass `frequencies_hz` for a different range or a finer grid
around a resonance of interest.

{meth}`~pgml.studies.ImpedanceScan.resonances` reports every interior local maximum of
`|Z|` as a parallel resonance and every local minimum as a series resonance, for the
positive sequence by default (`component="zero"` or a phase letter for the others). The
frequency is refined between the scan points: near a lightly damped resonance `1/|Z|²`
(peak) and `|Z|²` (dip) are quadratic in frequency, so the vertex of a parabola through the
extremum and its neighbours lands within a small fraction of the step. A resonance much
narrower than the step can be missed; scan finer around it.

## Validation

- Closed forms, complex128: a source inductance in parallel with a capacitor, a tuned
  series branch, and a three-phase capacitor bank behind a source with different positive-
  and zero-sequence impedance. The scanned curves agree with the exact phasor expressions to
  4e-14 relative; on the default 5 Hz grid the refined resonance frequencies are within
  0.033 Hz of the closed-form extrema.
- OpenDSS 0.9.4 (opendssdirect.py): the scan equals the dense inverse of OpenDSS's own
  `SystemY` of a three-phase feeder with a capacitor bank at 50 to 2450 Hz to 1.8e-13
  relative, element by element.
- Differentiability: a float64 gradcheck of the phase matrix and both sequence impedances
  with respect to line R, L and C, the source R and L and the load power.

## Results and serialization

{class}`~pgml.studies.ImpedanceScan` holds CPU tensors. The impedances stay on the autograd
tape when the grid carried tensor-valued parameters, so a scan can sit inside an
optimisation, for example detuning a filter or sizing a bank against a target `|Z|` band.
{meth}`~pgml.studies.ImpedanceScan.to_dict` returns a JSON-ready dict (complex values split
into `*_re_ohm` / `*_im_ohm` lists, sequence impedances `None` on buses without a, b, c) and
{meth}`~pgml.studies.ImpedanceScan.from_dict` rebuilds the result from it.
