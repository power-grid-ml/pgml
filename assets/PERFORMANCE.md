# How the performance figures are measured

This page explains the throughput figure in the README and adds the rest of the
comparison: grid size, harmonic studies, memory and cost. The numbers come from
the benchmark scripts of the
[pgml paper repository](https://github.com/power-grid-ml/pgml-paper) (to be published soon). The figures here are drawn from the recorded results by
`run/readme/render.py`, which runs no experiment.

Read the summary first if you are deciding whether to use pgml: it says where
pgml wins and where it loses.

## What is compared, and on what

Each tool solves the same problem for many load scenarios of one grid.
Throughput is the number of scenarios divided by the wall time of one call.

The comparison is between tools, on one stated allocation, with the same
parallelism everywhere. One measurement uses either one GPU or eight physical
CPU cores; nothing uses both.

| Tool | Hardware | How it is run |
|---|---|---|
| pgml on the GPU | one NVIDIA L40S 48 GB | one batched call, complex128, dense factorisation |
| pgml on the CPU, 8 workers | 8 physical cores | eight worker processes with one thread each; each solves its share of the batch in one batched call |
| pgml on the CPU, one call | 8 physical cores | one batched call in one process; the sparse backend on eight threads, the dense backend on one thread above 128 rows |
| pandapower | 8 physical cores | `runpp` per scenario with numba, the pi transformer model and recycled network matrices, over eight worker processes with one primed network each |
| power-grid-model | 8 physical cores | its native batch calculation with the iterative-current method and eight threads |
| OpenDSS | 8 physical cores | OpenDSSDirect.py, eight worker processes with one thread each, each with one compiled circuit |

Eight physical cores means an allocation of sixteen logical cpus on a machine
with two hardware threads per core. Every result records the affinity mask it
ran under and the physical core count derived from the core topology, so the
allocation is a measured property of the run rather than an assumption.

### Why pgml appears twice on the CPU

The two CPU rows are the same engine used in two ways, and they answer different
questions.

The eight-worker row is the like-for-like comparison: it gives pgml exactly the
parallelism pandapower and OpenDSS get, so the CPU columns differ by engine and
not by how the work was spread. It is also the only CPU configuration in which
pgml's batched dense factorisation is trustworthy on this build of PyTorch:
`torch.linalg.lu_factor` on a batch of complex matrices of 200 rows or more
returns invalid pivots when it runs on more than one thread, so a single-process
dense solve above 128 rows is pinned to one thread. Eight single-threaded
processes never reach that path.

The one-call row is how a user writes pgml: one `solve_power_flow` on a batch. It
is faster at small batches, where handing scenarios to worker processes costs
more than it saves, and it is what the same code does on a GPU.

### What is still not identical

- The three process-pool tools (pgml's pooled arm, pandapower, OpenDSS) pay for
  sending scenarios to their workers and results back inside the timed call. The
  two in-process tools (power-grid-model, pgml's single call) do not. That is a
  genuine cost of running an engine over processes and it is why the pooled arms
  lose at a batch of one.
- pgml and OpenDSS do not solve quite the same harmonic device model. The
  harmonic comparison is therefore accepted at 1e-4 per unit rather than the 1e-6
  the fundamental comparison uses, and each recorded point carries the tolerance
  it was judged at.
- pandapower and power-grid-model have no harmonic power flow, so the harmonic
  figure has one reference tool instead of three.
- The measurements ran on a node shared with other jobs.

## Grids and scenarios

The batch figures use the IEEE 33-bus feeder, the CIGRE low-voltage benchmark
solved in three phases, the Kerber Vorstadtnetz Kabel 1 with 294 buses, and four
merged copies of it with 1,176 buses. All come from pandapower. The Kerber
builder draws cable types at random, so its seed is fixed.

The size figure uses a family of radial 20 kV feeders with 16 to 4,096 buses.
Every grid has eight feeder branches of identical cable segments and one load per
bus, and exists as a pandapower network, so every tool solves the same case at
every size.

A scenario multiplies the active and reactive power of every load by its own
factor, drawn uniformly between 0.8 and 1.2 with a fixed seed. The scenarios are
drawn once for the largest batch. Smaller batches use the first rows, so all
tools and all batch sizes see identical inputs.

A harmonic study is one nonlinear solve at the fundamental plus one linear solve
at each odd order up to 25, with a six-pulse converter spectrum on a quarter of
the loads. It is a different amount of work from a power flow, so it is counted
and plotted separately.

## What is timed

One timed call solves one whole batch. Each configuration gets one untimed
warm-up call followed by five timed repetitions, or three for the per-scenario
tools. The figures show the median.

Inside the timed region are the solve itself and, for the process-based tools,
the hand-over of scenarios to the workers and of results back.
GPU timings synchronise the device before the clock starts and before it stops.

Outside the timed region are grid conversion, building each tool's model, moving
pgml inputs to the GPU and all correctness checks. Each worker pool is created
once and each worker builds its model once: pandapower's workers unpickle and
prime their own network, OpenDSS's workers compile their circuit, pgml's workers
build their grid and solve once, so no timed call ever lands on a process that
has not solved before.

## Validity guard

A throughput value is kept only if the solution is correct. The leading 32
scenarios of every timed batch are re-solved outside the clock and their voltage
magnitudes compared with a pgml CPU complex128 reference, and a batch smaller
than that is checked in full. The point counts if the
tool reports convergence and the largest difference stays within the tolerance
recorded with it. Points that fail are recorded without a throughput and cannot
reach a figure. The renderer repeats the check on the recorded values and refuses
to draw an invalid point.

The pandapower runs fail at the start if numba cannot be imported, and every
worker confirms that pandapower really used numba and the pi transformer model.

## Results

### Batch size

![Throughput against batch size on four grids](readme/batch_throughput_all.svg)

pgml's throughput grows almost in proportion to the batch size, because one
batched call shares its fixed cost among all the scenarios in it. The reference
tools solve one scenario at a time and level off once their worker pool is busy.

At one scenario per batch the order is reversed and pgml is last. power-grid-model
solves 5,700 scenarios per second on the 33-bus feeder and 2,600 on the Kerber
network where pgml manages 140 and 50. That is the fixed cost of a batched
solver being paid by a batch of one.

Where the curves cross, against each reference tool, taking the fastest pgml arm
on the device:

| Grid | against pandapower | against OpenDSS | against power-grid-model | lead at the largest common batch |
|---|---|---|---|---|
| IEEE 33, 33 rows | 16 | 4,096 | 4,096 to 16,384 | 4.2x power-grid-model at 65,536 |
| CIGRE LV, 132 rows | no arm | 4,096 | 4,096 | 3.1x at 16,384 |
| Kerber, 294 rows | 64 | 1,024 | 16,384 to 65,536 | 1.5x at 65,536 |
| Kerber x4, 1,176 rows | 256 | 1,024 | never | 0.5x at 4,096 |

A crossover is given as a range where the two curves are within 1.5 times of
each other at the batch where they cross, because repeating a measurement on
this shared machine moves a tool by up to half as much again.

The pattern is the point. The batch at which pgml overtakes power-grid-model
rises with the grid, and the lead it reaches afterwards falls: four times on a
33-row feeder, three times at 132 rows, one and a half times at 294 rows, and on
the 1,176-row network it does not overtake it at any batch measured.

The 1,176-row row is measured only to 4,096 scenarios per batch, where
power-grid-model is about twice as fast. The grid-size figure below, where all
four batch sizes were measured at every rung, is the evidence for what happens
above a thousand rows.

pgml on eight CPU worker processes follows the same shape one level down. It
overtakes pandapower at the same batch sizes and OpenDSS on the smallest grid,
and it never overtakes power-grid-model.

pandapower's three-phase solver cannot solve the converted CIGRE LV network, so
that arm is absent rather than slow.

### Grid size

![Throughput against grid size at four batch sizes](readme/size_scaling.svg)

Here the batch is fixed and the grid grows from 16 to 4,096 buses, every grid a
pandapower network every tool solves. pgml factorises a dense matrix on the GPU,
and on the CPU the faster of a dense and a sparse factorisation is shown. The
reference tools use sparse solvers that exploit the radial structure, so they
lose less speed as the grid grows.

At one and at sixteen scenarios per batch pgml is the slowest tool at every size.
It becomes competitive at 256 and leads at 4,096, and only up to a point. Taking
power-grid-model as the tool to beat, pgml on the GPU at 4,096 scenarios per
batch is 1.4 times faster at 256 buses, 1.2 times at 512, and behind from 1,024
buses upward, reaching 0.37 times at 4,096 buses. Against OpenDSS at the same
batch pgml stays ahead at every size, by 7.4 times at 256 buses falling to 1.5
times at 4,096. Against pandapower pgml leads from 256 scenarios per batch at
almost every size.

pgml on eight CPU worker processes never overtakes power-grid-model at any size
or batch measured here.

### Harmonic studies

![Throughput of whole harmonic studies against batch size](readme/harmonic_throughput.svg)

A study is one nonlinear solve at the fundamental plus a linear solve at each of
twelve further odd orders. OpenDSS is the only reference tool that can do it;
pandapower and power-grid-model have no harmonic power flow.

The two engines do not solve quite the same harmonic device model, so this arm is
accepted at 1e-4 per unit rather than 1e-6. The measured difference is 3.1e-5 per
unit on the 33-bus feeder and below 1e-5 on the two larger grids.

On the 33-bus feeder OpenDSS levels off near 8,000 studies per second. pgml
passes it at about 1,000 studies per batch and reaches 39,000 on the GPU and
15,000 on eight CPU workers at 16,384 studies, so 4.9 and 1.9 times OpenDSS.

On the two larger grids pgml loses, and by a wide margin. On the 294-row Kerber
network OpenDSS reaches 1,900 studies per second against 383 for pgml's best
arm, and on the 1,176-row network 390 against 21. pgml never overtakes OpenDSS
on either.

pgml's harmonic batch is bounded by memory in a way its fundamental batch is not.
The device shunt of a load follows the solved operating point, so the per-order
admittance becomes one matrix per scenario rather than one matrix shared by all
of them, and the working set grows as batch times orders times the square of the
grid. On the 1,176-row network that limits a GPU study to about 16 scenarios per
batch on a 48 GB card, and it is why the harmonic curves stop earlier than the
fundamental ones. A device shunt on a nameplate basis removes the dependence and
is a separate configuration of the solver.

## Solver configuration

The comparison above runs the solver's defaults. Three choices inside the solver
change what it costs, measured separately on a workstation card.

![Speed-up of a prepared harmonic study against the grid size, on a CPU and on a GPU](readme/solver_strategy_preparation.svg)

A `HarmonicFlowSystem` passed as `system=` keeps the factored fundamental system, the harmonic
network matrix for the requested orders, and the factorization of the assembled `Y(h)`.
Everything that depends on the operating point is recomputed: the fundamental power flow is
solved again, the device shunts are evaluated from the new solution, and `I(h)` is rebuilt.
Reuse is decided by value, not identity, and an unchanged operating point is neither necessary
nor sufficient for it. Network assembly and an operating-point-independent factorization are
reused across genuinely different scenarios; a `load_shunt_basis="operating_point"` shunt puts
the scenario into `Y(h)`, and then only the network entry is reusable.

Whether the preparation pays depends on the device. On eight CPU threads at complex128 and
orders 1 to 13 it is a loss at 99 rows in every configuration, clears 1.0x between 129 and 600
rows, and returns 1.38x to 1.62x at 900 rows. On one RTX A2000 it pays at every size measured,
1.12x to 1.27x at 99 rows and 1.38x to 2.49x at 900, with the gain growing with the grid.
Retention with an operating-point-independent `Y(h)` is 4 MiB at 99 rows and 247 MiB at 900;
with a per-scenario `Y(h)` it reaches 804 MiB for 32 scenarios of a 300-row feeder, because
the validity check keeps a copy of the matrix beside its factorization.

![Cost of reusing one factorization, of a low-rank correction and of solving each scenario alone](readme/solver_strategy_refactor.svg)

On the linear algebra itself, a dense direct solve factorizes too: solving the assembled
matrix and keeping an explicit factorization cost the same to within one percent at all 24
measured points, so there is nothing to gain by not refactorizing. What pays is reuse. One
factorization of a shared network answers 64 scenarios of a 900-row feeder in 26 ms against
960 ms for one factorization each (37x), and where the change is structured, one factorization
of the shunt-free network plus an exact Woodbury correction reaches each scenario's own matrix
in 62 ms (15.6x) at 6.6e-13 relative agreement. Solving one scenario at a time instead of
batching costs 4.1x to 12.6x. A Jacobi-preconditioned iterative solve that never factorizes is
2x to 5x slower at complex128 and does not reach the tolerance at complex64.

![Iterations and wall time of the fixed point against Newton](readme/solver_strategy_method.svg)

The two fundamental methods converge to the same solution. The current-injection fixed point
takes 3 to 8 iterations at nameplate loading and 32 to 36 near its convergence limit; Newton
takes 2 to 5 throughout. Newton is nonetheless 2.0x to 26x slower per unbatched solve and 41x
to 424x slower on a batch of 32, because every step rebuilds and factors the state Jacobian and
a batch is solved one scenario at a time. Both carry the same implicit-function-theorem
backward at the converged voltage, so the gradients are identical and equally conditioned.

<sub>This section only: measured 2026-09-24, library 0.5.1, complex128 unless stated, one NVIDIA RTX A2000 12 GB and
eight CPU threads of a 16-core workstation, medians of five to seven repeats after warm-up,
every solution validated by the residual of the system it claims to have solved.</sub>

## Memory

![Host memory per tool and device memory for pgml, against batch size](readme/memory_footprint.svg)

Each point is one fresh interpreter that builds one tool's model, brings up its
worker pool and solves one batch. While the solve runs, the whole process tree is
sampled and the largest total is kept. The figure shows the proportional set
size, which splits a page shared between worker processes between them; the
resident sum, which counts such a page once per process, is in the data as well
and is 10 to 40 per cent higher for the pooled tools. Device memory is the
PyTorch allocator's peak, which excludes the CUDA context.

Nothing is subtracted. The tree at rest, measured after the model is built and
the pool is up but before the solve starts, is recorded beside the peak.

Host memory at the largest measured batch, in MiB:

| Tool | IEEE 33 | Kerber, 294 rows | Kerber x4, 1,176 rows |
|---|---|---|---|
| pgml CPU, one call | 383 | 757 | 1,728 |
| power-grid-model, 8 threads | 616 | 703 | 1,235 |
| pgml GPU | 1,372 | 1,266 | 1,324 |
| pgml CPU, 8 workers | 3,354 | 3,415 | 5,210 |
| OpenDSS, 8 workers | 3,761 | 3,467 | 4,146 |
| pandapower, 8 workers | 4,827 | 4,994 | 5,265 |

Three things follow.

A worker pool costs about 3 GB before it solves anything, because eight
interpreters each hold their own copy of the library. That cost barely moves with
the grid or the batch, so it dominates every small job. pgml's pooled arm and
OpenDSS pay it; pandapower pays it and a little more.

A single process is far leaner. pgml in one batched call and power-grid-model in
its native threaded batch both stay under 2 GB at every point measured, and
pgml's single call is the smallest of all on the two smaller grids. pgml on the
GPU needs about 1.3 GB on the host, almost all of it the CUDA runtime, and then
grows on the device instead.

Device memory grows with the batch and with the square of the grid, because the
dense factorisation is what the GPU runs. At 4,096 scenarios it is 43 MiB on the
33-bus feeder, 242 MiB on Kerber and 988 MiB on the 1,176-row network.

### What fits on a 48 GB card

The capacity search doubles the batch until the device runs out of memory.

| Grid | largest batch measured to fit | device memory there | how the search ended |
|---|---|---|---|
| Kerber x4, 1,176 rows | 131,072 | 29.4 GiB | 262,144 ran out of memory |
| Kerber, 294 rows | at least 262,144 | 14.3 GiB | stopped at the search cap |
| IEEE 33 | at least 262,144 | 2.2 GiB | stopped at the search cap |

Only the first row is a capacity. The other two are lower bounds: the search
reached its own cap with the card less than a third full, so the real limit is
higher and was not measured.

## Cost

Throughput alone does not say whether an accelerator is worth using, because one
GPU and eight CPU cores are not the same thing to rent. This converts the
measured throughput into money at published list prices: at a price `p` per
instance-hour and a throughput `T` scenarios per second, a million scenarios cost
`1e6 / (T * 3600) * p`.

This is a price model, not a measurement. Nothing here was paid, and spot or
committed-use rates change the ratios several-fold.

| Priced as | Instance | USD per hour |
|---|---|---|
| the GPU side | one NVIDIA L40S 48 GB with four vCPU | 1.861 |
| the CPU side | sixteen vCPU, eight physical cores | 0.714 |

Both are on-demand, Linux, US-region list prices read on 2026-09-24. A second
provider selling the same L40S by the hour lists it at 1.09, and a community tier
of the same listing at 0.79, so the accelerator side of every ratio below moves
by up to a factor of two depending on where it is bought. The instance price
includes the host, so an idle host on a GPU-bound job is paid for either way.

The ratio that matters is 2.6: the GPU instance costs 2.6 times the CPU
instance. A pgml GPU throughput less than 2.6 times the best CPU throughput
means the CPU allocation is the cheaper way to buy those solves, however much
faster the card is.

![Cost per million solved scenarios against batch size](readme/cost_per_million.svg)

Cost per million solved scenarios, each tool at its own fastest measured batch:

| Tool | IEEE 33 | CIGRE LV, 132 rows | Kerber, 294 rows |
|---|---|---|---|
| pgml GPU | 0.00041 | 0.00242 | 0.0064 |
| power-grid-model | 0.00067 | 0.00292 | 0.0033 |
| OpenDSS | 0.00203 | 0.00343 | 0.0082 |
| pgml CPU, 8 workers | 0.00108 | 0.0046 | 0.0101 |
| pgml CPU, one call | 0.00152 | 0.0108 | 0.0198 |
| pandapower | 0.0906 | no arm | 0.127 |

pgml on the GPU is the cheapest way to buy these solves on the two smaller grids
and 1.9 times power-grid-model's price on the 294-row network. The cost crossover
comes at a smaller grid than the throughput crossover, and that is the point of
pricing at all: pgml is still faster than power-grid-model on the Kerber network
at a large batch, and it is already the more expensive way to get the answer,
because the accelerator costs 2.6 times the eight cores.

pandapower is two orders of magnitude more expensive per scenario than anything
else here.

## Where pgml wins and where it loses

Use pgml when you need many operating points of a small or medium grid, or when
you need gradients through the solve, which no other tool here offers. Use a
dedicated solver when you need a few answers, or when the grid is large.

pgml wins

- on batches of thousands of scenarios on grids of a few hundred node-phase rows
  or fewer, where one batched call amortises the fixed cost of a solve over the
  whole batch;
- against pandapower almost everywhere above a few dozen scenarios per batch;
- against OpenDSS at 4,096 scenarios per batch at every grid size measured;
- on cost per million scenarios on the smallest grid, where it is also the
  cheapest tool in this comparison.

pgml loses

- at a few scenarios, badly. At one scenario per batch it is the slowest tool
  here at every grid size, by one to two orders of magnitude. A batched engine
  pays its fixed cost once whether the batch holds one scenario or a hundred
  thousand.
- against power-grid-model from about a thousand node-phase rows upward, at
  every batch size measured. power-grid-model is the tool to beat in this
  comparison, and on larger grids it is not beaten.
- on harmonic studies of anything but the smallest grid, where OpenDSS is five to
  nineteen times faster.
- on standing memory whenever it is run over worker processes: eight
  interpreters cost about 3 GB before any scenario is solved.

Two further things a reader should take from the numbers rather than from the
headline.

Most of the batching gain is available without a GPU. pgml on the same eight
cores the other tools get captures a large part of it, and the accelerator adds a
further factor that only pays above a few thousand scenarios per batch. The GPU
instance also costs 2.6 times the CPU instance, so a speed-up below 2.6 times is
not a saving.

The dense factorisation pgml runs on the GPU is what makes it lose on large
grids. The reference tools use sparse solvers that exploit the radial structure
of a distribution feeder, and that advantage grows with the grid.

## Reproduce

The scripts live in the [pgml paper repository](https://github.com/power-grid-ml/pgml-paper) (to be published soon) under `scripts/bench` and
`scripts/cluster`. On a SLURM cluster described by a target file:

```bash
scripts/cluster/sync.sh --pgml <pgml checkout>      # ship the engine and the scripts
ssh <cluster> 'bash <campaign root>/paper/scripts/cluster/env_setup.sh'
scripts/cluster/run_all.sh --tag fair --max-queued 4 \
    --stages fair_batch fair_harmonic fair_size footprint footprint_harmonic \
             fair_cost fair_tables
scripts/cluster/fetch.sh --logs
python scripts/bench/readme_assets.py --fair --out <pgml checkout>/assets/readme
```

Without a cluster the same measurements run directly:

```bash
python scripts/bench/run_all.py --stages fair_batch fair_harmonic fair_size \
    footprint footprint_harmonic fair_cost fair_tables
```

Then draw the figures in the pgml checkout:

```bash
pixi run -e cpu python run/readme/render.py
```

`assets/readme/provenance.json` lists the software versions and the SHA-256 of every result file the figures are drawn from.
