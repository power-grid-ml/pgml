# Performance measurement record

This file records how the measurements on [PERFORMANCE.md](PERFORMANCE.md) were
taken: the machines, the software revisions, the exact configuration of every tool,
what is inside the timed region, how a point is validated, and how the protocol has
changed. PERFORMANCE.md presents the results; this file is where a figure's
conditions are checked. The benchmark scripts live in the
[pgml paper repository](https://github.com/power-grid-ml/pgml-paper) (to be
published soon), and `assets/readme/provenance.json` lists the software versions and
the SHA-256 of every result file the figures are drawn from.

## Machines and revisions

Unless a section below says otherwise, every number was measured on one NVIDIA L40S
48 GB node of an institutional cluster with eight physical cores of an AMD EPYC 9334,
driver 610.57.04, torch 2.13.0, on 2026-09-26 (the harmonic sweep and the harmonic
footprint in a run from 2026-09-26 into 2026-09-27). The reference engines are
pandapower 3.5.4 with numba 0.67.0, power-grid-model 1.13.172 and OpenDSSDirect.py
0.9.4.

The engine measured is pgml 0.5.1 at revision 966df9d of its main branch with the
merged solver branch and the sparse harmonic assembly, recorded by its source digest
242967f7eefb. The harmonic sweep and the harmonic footprint are measured with the
merged cache-defaults branch in addition, source digest c3bc1b3d1049, which touches no
timed code path. The README figure is drawn from the fundamental sweep and so from the
first engine.

Four sections of PERFORMANCE.md are workstation measurements instead:

| Section | Machine | Date |
|---|---|---|
| Solver configuration | one NVIDIA RTX A2000 12 GB, eight CPU threads of a 16-core workstation | 2026-09-24 |
| Filling the machine with scenarios | one NVIDIA RTX A2000 12 GB and sixteen CPUs of one workstation on an idle node | 2026-09-24 |
| Why the large grids go the other way | one NVIDIA RTX A2000 12 GB and sixteen CPUs of one workstation on an idle node | 2026-09-24 |
| Sparse harmonic assembly, before and after | Intel i7-12700, RTX A2000 12 GB, a 58 GB allocation | 2026-09-24 |

The per-scenario against shared shunt comparison was measured on the cluster
allocation on 2026-09-28.

## Tool configuration

| Tool | Hardware | How it is run |
|---|---|---|
| pgml on the GPU | one NVIDIA L40S 48 GB | one batched call, complex128, dense factorisation, in a fresh process |
| pgml on the CPU, 8 workers | 8 physical cores | eight worker processes with one thread each; each solves its share of the batch in one batched call |
| pgml on the CPU, one call | 8 physical cores | one batched call in a fresh process; the sparse backend on eight threads, the dense backend on one thread above 128 rows |
| pandapower | 8 physical cores | `runpp` per scenario with numba, the pi transformer model and recycled network matrices, over eight worker processes with one primed network each; `runpp_3ph` on the three-phase grid |
| power-grid-model | 8 physical cores | its native batch calculation with the iterative-current method and eight threads, in a fresh process |
| OpenDSS | 8 physical cores | OpenDSSDirect.py, eight worker processes with one thread each, each with one compiled circuit |

Eight physical cores means an allocation of sixteen logical cpus on a machine with two
hardware threads per core. Every result records the affinity mask it ran under and the
physical core count derived from the core topology, so the allocation is a measured
property of the run rather than an assumption. The BLAS and torch thread caps are set
to the physical core count for every engine.

### The two pgml CPU configurations

The eight-worker configuration is the like-for-like comparison: it gives pgml exactly
the parallelism pandapower and OpenDSS get, so the CPU columns differ by engine and not
by how the work was spread. It is also the only CPU configuration in which pgml's
batched dense factorisation is trustworthy on this build of PyTorch:
`torch.linalg.lu_factor` on a batch of complex matrices of 200 rows or more returns
invalid pivots when it runs on more than one thread, so a single-process dense solve
above 128 rows is pinned to one thread. Eight single-threaded processes never reach
that path.

The one-call configuration is how a user writes pgml: one `solve_power_flow` on a
batch. It is faster at small batches, where handing scenarios to worker processes costs
more than it saves, and it is what the same code does on a GPU.

## Grids and scenarios

The batch figures use the IEEE 33-bus feeder, the CIGRE low-voltage benchmark solved
in three phases, the Kerber Vorstadtnetz Kabel 1 with 294 buses, and four merged copies
of it with 1,176 buses. All come from pandapower. The Kerber builder draws cable types
at random, so its seed is fixed.

The size figure uses a family of radial 20 kV feeders with 16 to 4,096 buses. Every
grid has eight feeder branches of identical cable segments and one load per bus, and
exists as a pandapower network, so every tool solves the same case at every size. On
this ladder a bus is one node-phase row.

A scenario multiplies the active and reactive power of every load by its own factor,
drawn uniformly between 0.8 and 1.2 with a fixed seed. The scenarios are drawn once
for the largest batch. Smaller batches use the first rows, so all tools and all batch
sizes see identical inputs.

A harmonic study is one nonlinear solve at the fundamental plus one linear solve at
each odd order up to 25, with a six-pulse converter spectrum on a quarter of the loads.
OpenDSS and pgml solve one study definition, the same thirteen orders, the same
spectrum on the same loads and the same device shunt, and run the same batch ladder
from 1 to 16,384 studies in one job.

## What is timed

One timed call solves one whole batch. Each configuration gets one untimed warm-up
call followed by five timed repetitions, for every engine alike. The figures show the
median; the full sample and its interquartile range are in the recorded JSON.

Inside the timed region, for every engine, are the solve itself and everything needed
to have the voltage magnitudes of the whole batch in host memory when the clock stops:
for the process-based tools the hand-over of scenarios to the workers and of results
back, for pgml on the GPU the copy of the result from the device. GPU timings
synchronise the device before the clock starts and before it stops. pgml's admittance
assembly and factorisation are inside as well, because no prepared system is passed
(PERFORMANCE.md, [Does pgml rebuild its matrix on every call?](PERFORMANCE.md#does-pgml-rebuild-its-matrix-on-every-call)).

Outside the timed region are grid conversion, building each tool's model, moving pgml's
inputs to the GPU and all correctness checks. Each worker pool is created once and each
worker builds its model once and then solves once, untimed: pandapower's workers
unpickle and prime their own network and run one `runpp`, OpenDSS's workers compile
their circuit and run one `Solve`, pgml's workers build their grid and solve once. No
timed call lands on a process that has not solved before, for any of the three pools.

The two single-process arms, power-grid-model and pgml's single call, are each timed in
a fresh child process that has never held a worker pool and that inherits the parent's
thread caps. A process in which a pool has existed measures a threaded engine slower,
and isolating the arm removes the question of by how much.

Every arm stops at the first batch whose call exceeds the per-call time budget, 60
seconds in the fundamental sweeps and 120 in the harmonic one; that call is recorded,
and the batches beyond it are recorded as skipped for time rather than left absent. A
harmonic batch is admitted by one memory rule for every engine and every pgml arm: the
planned working set is the batch-shaped `[B, H, N]` vectors plus the per-order system
chunk the engine bounds itself, charged per worker for a pool, against 0.8 of the
allocation's memory on the host and the card's memory on the device. A refused batch is
recorded with the estimate and the budget.

## Validity guard

A throughput value is kept only if the solution is correct. The leading 256 scenarios
of every timed batch are re-solved outside the clock and their voltage magnitudes
compared with a pgml CPU complex128 reference at 1e-6 per unit, and a batch smaller
than that is checked in full; the harmonic sweep compares 64 scenarios per batch, and
its OpenDSS arm is accepted at 1e-4 per unit (see below). Each row records how many
scenarios it was compared over, and `assets/readme/provenance.json` records the largest
depth observed in the run as `validated_scenarios_per_batch`, which for this campaign
is 256. The point counts if the tool reports convergence and the largest difference
stays within the tolerance recorded with it. Points that fail are recorded without a
throughput and cannot reach a figure. The renderer repeats the check on the recorded
values and refuses to draw an invalid point.

The pandapower runs fail at the start if numba cannot be imported, and every worker
confirms that pandapower really used numba and the pi transformer model.

## What is still not identical

- The three process-pool tools (pgml's pooled arm, pandapower, OpenDSS) pay for sending
  scenarios to their workers and results back inside the timed call. The two
  single-process tools (power-grid-model, pgml's single call) have no workers to pay,
  and each of them is timed in a fresh process that has never held a pool. The worker
  hand-over is a genuine cost of running an engine over processes and it is why the
  pooled arms lose at a batch of one. Every engine's timed call, pgml on the GPU
  included, ends with the voltage magnitudes in host memory.
- pgml and OpenDSS build the same harmonic device shunt but not quite the same line
  impedance above the fundamental. The harmonic comparison is therefore accepted at
  1e-4 per unit rather than the 1e-6 the fundamental comparison uses, and each recorded
  point carries the tolerance it was judged at.
- pandapower and power-grid-model have no harmonic power flow, so the harmonic figure
  has one reference tool instead of three.
- The two reference engines that solve the CIGRE LV grid in three phases are fed
  different zero-sequence fills, because its source network carries none. pandapower
  gets pgml's own converter defaults (lines `R0 = 4 R1`, `X0 = 3 X1`, `C0 = 0.5 C1`; the
  external grid `Z0 = Z1`; the transformers `Dyn` with the positive-sequence leakage)
  plus pandapower's own suggested placeholders for the zero-sequence magnetising
  branch, which pgml does not model. power-grid-model gets its converter's customary
  `R0 = 3 R1`, `X0 = 3 X1`, `C0 = 0`. The scenario draw scales every phase by the same
  factor, so no current flows in the zero-sequence network and both arms agree with
  pgml to below 1e-9 per unit (pandapower to 6.7e-10); on an unbalanced draw the fills
  would matter.
- pgml re-assembles and re-factorises its admittance on every timed call, where
  power-grid-model reuses its factorisation across batches and OpenDSS keeps its
  compiled circuit. This costs pgml, not the others.
- The scheduler did not hold the node exclusively for these jobs. A repeated
  measurement on the shared node moves a tool by up to 63 per cent between runs, which
  is why PERFORMANCE.md reads a ratio below 1.5 as level.

## Section-specific conditions

**Per-scenario against shared harmonic shunt.** Cluster allocation of this record,
2026-09-28, pgml 0.5.1 at the revision of this record, complex128, thirteen orders,
medians of five repeats after warm-up, one batch ladder from 1 to 16,384 studies for
every engine and basis, a series stopped at the first call over 120 s. The CPU arm
follows the thread rule above: one thread only for the dense backend above 128 rows.
Agreement is the largest absolute difference in per-unit voltage magnitude against a
live OpenDSS on the matched circuit over the same sixteen scenarios.

**Sparse harmonic assembly, before and after.** Both engine revisions measured with the
same harness on the workstation; the two runs also differ in the harness's memory rule,
which only decides where a series stops.

**Solver configuration.** Library 0.5.1, complex128 unless stated, medians of five to
seven repeats after warm-up, every solution validated by the residual of the system it
claims to have solved.

**Filling the machine with scenarios.** Library 0.5.1, complex128, eight worker
processes or eight threads for every arm that uses them, one engine at a time, every
probe in a fresh interpreter. The batch doubles until the memory ceiling is passed, a
probe runs out of memory, the batch cap of 131,072 is reached or one solve takes longer
than a minute. The ceiling is 10.5 GiB of device memory for the GPU arm and 32 GiB of
host memory for every other arm, measured the way the memory figure measures it, with
nothing subtracted.

**Why the large grids go the other way.** Library 0.5.1, complex128, phases timed by
wrapping the functions the solver calls, a CUDA configuration synchronising inside
every wrapper.

**Memory.** Each point is one fresh interpreter that builds one tool's model, brings up
its worker pool and solves one batch. While the solve runs, the whole process tree is
sampled and the largest total is kept. The figure shows the proportional set size; the
resident sum, which counts a shared page once per process, is in the data as well and
is 10 to 40 per cent higher for the pooled tools. Device memory is the PyTorch
allocator's peak, which excludes the CUDA context. pgml's CPU arms run the backend the
library selects for the grid, and each record names the backend it resolved to.

**Cost.** On-demand, Linux, US-region list prices read on 2026-09-24: one NVIDIA L40S
48 GB with four vCPU at 1.861 USD per hour, sixteen vCPU with eight physical cores at
0.714 USD per hour. A second provider selling the same L40S by the hour lists it at
1.09, and a community tier of the same listing at 0.79. At a price `p` per
instance-hour and a throughput `T` scenarios per second, a million scenarios cost
`1e6 / (T * 3600) * p`.

## Protocol history

The protocol of 2026-09-25, which every cluster measurement above uses, differs from
the one of 2026-09-24 in these points.

- pgml's single-call timed region ends with the voltage magnitudes in host memory, as
  every other engine's always did; before, the result was left on the GPU.
- power-grid-model and pgml's single call are timed in a fresh child process that has
  never held a worker pool; before, they ran in the sweep's own process after its
  pools.
- Every worker pool, pandapower's and OpenDSS's as well as pgml's, solves once untimed
  in every worker before the clock runs; before, only pgml's pool did.
- Every engine gets five timed repetitions; before, pandapower and OpenDSS got three.
- The validity guard re-solves the leading 256 scenarios of every timed batch; before,
  32.
- pandapower's three-phase solver solves the CIGRE LV grid, on zero-sequence data
  written from pgml's own converter defaults; before, that arm was recorded as
  unavailable.
- Each grid's pgml arms and reference arms are measured in one job; before, the two
  smaller grids paired pgml from one job with the reference engines from another.
- OpenDSS and pgml run one harmonic ladder, 1 to 16,384 studies per batch, in one job
  under the same per-call time budget; before, pgml's harmonic ladder was a shorter
  per-grid table and OpenDSS ran in a separate job.
- A harmonic batch is admitted by an engine-agnostic memory rule that plans the
  batch-shaped vectors plus the system chunk the engine itself bounds; before, a
  harness cap on the nominal dense `[B, H, N, N]` size cut pgml's harmonic curves short
  of both the card and the host.
- pgml's single call stops on the same per-call time budget as every other arm.
- The crossover reduction takes pgml's double-precision series only.
- The memory footprint requests the library's own backend choice for pgml's CPU arms
  and records it, and the harmonic footprint gained the 1,176-row network.
- A killed or hung worker is recorded as an error row instead of stopping the stage.

## Reproduce

The scripts live in the pgml paper repository under `scripts/bench` and
`scripts/cluster`. On a SLURM cluster described by a target file:

```bash
scripts/cluster/sync.sh --pgml <pgml checkout>      # ship the engine and the scripts
ssh <cluster> 'bash <campaign root>/paper/scripts/cluster/env_setup.sh'
scripts/cluster/run_all.sh --tag fair --max-queued 4 \
    --stages fair_batch_small fair_batch_large fair_harmonic fair_size \
             footprint footprint_harmonic fair_cost fair_tables
scripts/cluster/fetch.sh --logs
python scripts/bench/readme_assets.py --fair --out <pgml checkout>/assets/readme
```

`fair_batch_small` measures the IEEE 33 and CIGRE LV grids and `fair_batch_large` the
two Kerber networks, each with pgml's arms and the reference engines in one job;
`fair_harmonic` runs OpenDSS and pgml on one ladder in one job.

Without a cluster the same measurements run directly:

```bash
python scripts/bench/run_all.py --stages fair_batch_small fair_batch_large \
    fair_harmonic fair_size footprint footprint_harmonic fair_cost fair_tables
```

Then draw the figures in the pgml checkout:

```bash
pixi run -e cpu python run/readme/render.py
```
