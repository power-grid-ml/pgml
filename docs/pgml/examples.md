# Examples and validation

The repository ships runnable studies under `run/examples/pgml/`. Each one is self-contained
and writes its figures and tables into an output directory.

```bash
python run/examples/pgml/evaluate_ieee33.py [output_dir]
```

The figures below come from those scripts and are the evidence behind the modelling claims in
{doc}`modeling/index`.

## Against the reference tools

`evaluate_ieee33.py` rebuilds the IEEE 33-bus feeder, assembles the admittance matrix, runs
the load flow and the harmonic flow, and compares all three against pandapower and OpenDSS.

```{figure} ../_static/figures/ybus_heatmaps.svg
:alt: Assembled admittance matrix compared with pandapower and OpenDSS
:width: 95%

The IEEE-33 nodal admittance magnitude assembled by pgml (left), pandapower's network
admittance (centre) and OpenDSS's exported system admittance (right). Look for the same
sparsity pattern and the same colour scale in all three panels. The residual between them is
at floating-point level.
```

```{figure} ../_static/figures/voltage_profile.svg
:alt: Load-flow voltage profile, pgml against pandapower
:width: 85%

Nonlinear load-flow voltage in per unit against distance from the slack. The pgml curve and
the pandapower markers lie on top of each other along the whole feeder, including the
laterals that drop below 0.95 pu.
```

```{figure} ../_static/figures/harmonic_h5.svg
:alt: Fifth-harmonic voltage profile and angle
:width: 85%

Fifth-harmonic voltage magnitude and angle along the feeder, pgml against an independent
NumPy solution of the same nodal system. Look for two curves that coincide in both panels.
The magnitude rises towards the three converter loads at the feeder end.
```

`evaluate_harmonics_carson.py` does the stricter harmonic comparison. It synthesizes
conductor geometry that reproduces each line's impedance at the fundamental, then feeds the
same geometry to pgml and to OpenDSS. With the geometry identical on both sides there is no
earth-model ambiguity left, and the two engines agree to floating-point precision at every
order.

## Harmonic line models

`evaluate_line_sequence_harmonics.py` overlays the analytic harmonic line models and a live
OpenDSS profile on the same feeder. The resulting figures and the physics behind them are in
{doc}`modeling/harmonic-line-model`.

## Batched scenarios

`scenario_randomized.py` and `scenario_node_injection_sweep.py` sample operating points,
solve them in one batch, and summarise how a harmonic disturbance spreads across a feeder.
Sampling is reproducible from a configuration and a seed.

```{figure} ../_static/figures/spread_h11.svg
:alt: Spread of the 11th-harmonic voltage across a scenario batch
:width: 85%

Distribution of the 11th-harmonic node voltage across a sampled batch on the CIGRE LV
network. Look at how the spread widens with distance from the transformer, which is the
quantity a harmonic study cares about rather than any single snapshot.
```

## Solver behaviour near the limit

`loadability_continuation.py` walks the loading parameter up to the point where the load flow
stops having a solution, and `current_injection_convergence.py` compares the two nonlinear
solvers along the way.

```{figure} ../_static/figures/pv_nose.svg
:alt: Loadability nose curve
:width: 80%

Voltage against loading for the critical bus. The nose is where the solution disappears. The
continuation reports the margin, the critical bus and the limiting load instead of returning
a convergence failure.
```

## Performance studies

Three scripts measure rather than plot. `benchmark_speed.py` times the batched solve against
batch size on whichever devices are present. `benchmark_sparse.py` sweeps system size and
compares the dense and sparse factorization backends, which is how the automatic backend
threshold was calibrated. `benchmark_woodbury.py` times a switch-state sweep with and without
the low-rank update, and reports the voltage agreement between the two paths alongside the
speedup. The measured numbers are in {doc}`modeling/solver-performance`.
