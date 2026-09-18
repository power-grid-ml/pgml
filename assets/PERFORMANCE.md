# How the performance figures are measured

This page explains the throughput figure in the README and adds a second one on
grid size. The numbers come from the benchmark scripts of the pgml paper
repository. The figures here are drawn from the recorded results by
`run/readme/render.py`, which runs no experiment.

## What is compared

Each tool solves the same fundamental-frequency power flow for many load
scenarios of one grid. Throughput is the number of scenarios divided by the
wall time of one call. Only the forward solve is measured. Gradients are not
part of this comparison.

| Tool | How it is run |
|---|---|
| pgml on the CPU | one batched call, complex128. The figure shows the faster of the dense and the sparse backend at each point. |
| pgml on the GPU | the same batched call on one GPU, dense backend, complex128 |
| pandapower | `runpp` per scenario with numba, the pi transformer model and recycled network matrices, split over eight worker processes with one thread each |
| power-grid-model | its native batch calculation with the iterative-current method and eight threads |
| OpenDSS | OpenDSSDirect.py, eight worker processes with one thread each. A scenario changes the load powers, solves and reads the voltages. |

pandapower follows its documented pattern for time series. power-grid-model
uses the settings that a comparison in the paper repository found fastest among
its exact methods. OpenDSS solves the circuit that
`pgml.convert.opendss.from_grid` writes from the same pgml grid in matched mode.
Two details make that circuit comparable with the single-phase equivalent the
other tools solve. The transformer clock, which a one-phase OpenDSS transformer
cannot carry and which does not change voltage magnitudes, is removed. The
transformer magnetizing branch is split between both terminals as pgml and
power-grid-model place it, and as pandapower's pi model does.

## Grids and scenarios

The README figure uses the IEEE 33-bus feeder and the Kerber Vorstadtnetz Kabel 1
low-voltage grid with 294 buses, both taken from pandapower and converted to pgml
as single-phase equivalents. The Kerber builder draws cable types at random, so
its seed is fixed.

The size figure uses a family of radial 20 kV feeders with 16 to 4,096 buses.
Every grid has eight feeder branches of identical cable segments and one load
per bus. The grids exist as pandapower networks, so every tool solves the same
case at every size.

A scenario multiplies the active and reactive power of every load by its own
factor, drawn uniformly between 0.8 and 1.2 with a fixed seed. The scenarios are
drawn once for the largest batch. Smaller batches use the first rows, so all
tools and all batch sizes see identical inputs.

## What is timed

One timed call solves one whole batch. Each configuration gets one untimed
warm-up call followed by five timed repetitions, or three for pandapower and
OpenDSS in the batch figure. The figures show the median.

Inside the timed region are the solve itself and, for the process-based tools,
the hand-over of scenarios to the workers and of voltages back. GPU timings
synchronise the device before the clock starts and before it stops.

Outside the timed region are grid conversion, building each tool's model, moving
pgml inputs to the GPU and all correctness checks. For pandapower one full
untimed solve stores the measured options, because a recycled pandapower solve
reuses the options of the solve that built its matrices. For OpenDSS every worker
compiles the circuit once before timing starts.

One difference remains between the two process-based tools. The pandapower
adapter follows pandapower's documented joblib pattern and sends the network to
its workers within every timed call. The OpenDSS workers keep their compiled
circuit between calls. This favours OpenDSS.

## Validity guard

A throughput value is kept only if the solution is correct. Every scenario of
every timed batch is checked. Its voltage magnitudes are compared with a pgml
CPU complex128 reference, for the reference tools on a separate untimed solve of
the same scenarios. The point counts if the tool
reports convergence and the largest difference stays below 1e-6 pu. Points that
fail are recorded without a throughput and cannot reach a figure. The renderer
repeats the check on the recorded values and refuses to draw an invalid point.

The pandapower runs fail at the start if numba cannot be imported, and every
worker confirms that pandapower really used numba and the pi transformer model.

## Results

### Batch size

![Throughput against batch size on two grids](readme/batch_throughput.svg)

pgml's throughput grows almost in proportion to the batch size, because one
batched call shares its fixed cost among all scenarios. At 4,096 scenarios the
GPU solves 267,000 scenarios per second on the 33-bus feeder and 53,000 on the
Kerber grid. power-grid-model reaches 199,000 and 37,000 there, OpenDSS 58,000
and 10,000, pandapower 1,200 and 900.

Small batches show the opposite order. A single scenario takes pgml 8 ms on the
CPU and 15 ms on the GPU for the 33-bus feeder, which is 125 and 68 scenarios per
second. power-grid-model solves 5,500 per second, OpenDSS 2,000 and pandapower
320. The GPU passes pgml's own CPU path at about 256 scenarios per batch and
passes power-grid-model only at the largest batch measured.

### Grid size

![Throughput against grid size at two batch sizes](readme/size_scaling.svg)

Here the batch size is fixed and the grid grows from 16 to 4,096 buses. pgml
factorises a dense matrix on the GPU, and on the CPU the faster of a dense and a
sparse factorisation is shown. The reference tools use sparse solvers that
exploit the radial structure, so they lose less speed as the grid grows.

With a single scenario pgml is the slowest tool at every size, from 240 scenarios
per second at 16 buses down to fewer than 2 at 4,096 buses. With 256 scenarios
per batch OpenDSS is 1.1 to 2.7 times faster than pgml at every size and
power-grid-model about six to eight times. At 4,096 buses pandapower is also
ahead of pgml, with 350 against 235 scenarios per second. pgml's advantage comes from
large batches of small and medium grids, as in the batch figure, and not from
large grids.

<sub>Measured 2026-09-18, Each job had one NVIDIA L40S 48 GB (driver 610.57.04, CUDA 13.0) and eight logical CPUs, which are four cores of an AMD EPYC 9334, on a node shared with other jobs. pgml 0.5.1 at commit afb5ba6, Python 3.13, torch 2.13.0, pandapower 3.5.4 with numba 0.67.0, power-grid-model 1.13.172, OpenDSSDirect.py 0.9.4. All solves in complex128. Convergence tolerance 1e-8 for the reference tools, the library default for pgml. pgml's dense CPU path runs on one thread above 128 rows, its sparse path and power-grid-model on eight threads.</sub>

## Reproduce

The scripts live in the pgml paper repository under `scripts/bench` and
`scripts/cluster`. On a SLURM cluster described by a target file:

```bash
scripts/cluster/sync.sh --pgml <pgml checkout>      # ship the engine and the scripts
ssh <cluster> 'bash <campaign root>/paper/scripts/cluster/env_setup.sh'
scripts/cluster/run_all.sh --stages readme_ieee33 readme_kerber readme_size --tag readme
scripts/cluster/fetch.sh --logs
python scripts/bench/readme_assets.py --out <pgml checkout>/assets/readme
```

Without a cluster the same measurements run directly:

```bash
python scripts/bench/bench_readme.py --grid ieee33
python scripts/bench/bench_readme.py --grid kerber
python scripts/bench/bench_size.py --ladder pandapower --batches 1 256 --dtypes c128 \
    --opendss --guard-all --out bench_size_readme.json
```

Then draw the figures in the pgml checkout:

```bash
pixi run -e cpu python run/readme/render.py
```

`assets/readme/provenance.json` lists the software versions, the
engine revision and the SHA-256 of every result file the figures are drawn from.
