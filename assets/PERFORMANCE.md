# How the performance figures are measured

This page explains the throughput figure in the README and adds the rest of the
comparison: grid size, harmonic studies, memory, capacity and cost. The numbers come from
the benchmark scripts of the
[pgml paper repository](https://github.com/power-grid-ml/pgml-paper) (to be published soon). The figures here are drawn from the recorded results by
`run/readme/render.py`, which runs no experiment.

Read the summary first if you are deciding whether to use pgml: it says where
pgml wins and where it loses.

Unless a subsection says otherwise, every number on this page was measured on one
NVIDIA L40S 48 GB node of an institutional cluster, driver 610.57.04,
torch 2.13.0, on 2026-09-26 (the harmonic sweep and the harmonic footprint in a
run from 2026-09-26 into 2026-09-27), with the software versions listed
in `assets/readme/provenance.json`. Four subsections are workstation measurements
and say so in their captions: the harmonic shunt basis, the solver configuration,
filling the machine with scenarios, and the large-grid diagnosis.
The engine measured is pgml 0.5.1 at revision 966df9d of its main branch with the
merged solver branch and the sparse harmonic assembly, recorded by its source
digest 242967f7eefb; the harmonic sweep and the harmonic footprint are measured
with the merged cache-defaults branch in addition, source digest c3bc1b3d1049,
which touches no timed code path. The README figure is drawn from the fundamental
sweep of this page and so from the first engine.
<!-- env.driver, env.torch, env.date: bench_batch_throughput_fair.json environment.driver, environment.versions.torch, environment.date (also provenance.json performance_source.{driver,versions.torch,jobs.*.date}) -->

## Protocol changes 2026-09-25

The protocol of the measurements on this page differs from the one the documents
of 2026-09-24 described, in these points. pgml's single-call timed region now ends
with the voltage magnitudes in host memory, as every other engine's always did,
where before the result was left on the GPU. power-grid-model and pgml's single
call are timed in a fresh child process that has never held a worker pool, where
before they ran in the sweep's own process after its pools. Every worker pool,
pandapower's and OpenDSS's as well as pgml's, now solves once untimed in every
worker before the clock runs, where before only pgml's pool did. Every engine gets
five timed repetitions, where pandapower and OpenDSS got three. The validity
guard re-solves the leading 256 scenarios of every timed batch, where the
recorded runs compared 32. pandapower's three-phase solver now solves the CIGRE
LV grid, on zero-sequence data written from pgml's own converter defaults, where
before that arm was recorded as unavailable. Each grid's pgml arms and reference
arms are measured in one job, where before the two smaller grids paired pgml from
one job with the reference engines from another. OpenDSS and pgml run one
harmonic ladder, 1 to 16,384 studies per batch, in one job under the same
per-call time budget, where before pgml's harmonic ladder was a shorter per-grid
table and OpenDSS ran in a separate job. A harmonic batch is admitted by an
engine-agnostic memory rule that plans the batch-shaped vectors plus the system
chunk the engine itself bounds, where before a harness cap on the nominal dense
`[B, H, N, N]` size cut pgml's harmonic curves short of both the card and the
host. pgml's single call stops on the same per-call time budget as every other
arm. The crossover reduction takes pgml's double-precision series only. The
memory footprint requests the library's own backend choice for pgml's CPU arms
and records it, and the harmonic footprint gained the 1,176-row network. A killed
or hung worker is recorded as an error row instead of stopping the stage.

## What is compared, and on what

Each tool solves the same problem for many load scenarios of one grid.
Throughput is the number of scenarios divided by the wall time of one call.

The comparison is between tools, on one stated allocation, with the same
parallelism everywhere. One measurement uses either one GPU or eight physical
CPU cores; nothing uses both.

| Tool | Hardware | How it is run |
|---|---|---|
| pgml on the GPU | one NVIDIA L40S 48 GB | one batched call, complex128, dense factorisation, in a fresh process |
| pgml on the CPU, 8 workers | 8 physical cores | eight worker processes with one thread each; each solves its share of the batch in one batched call |
| pgml on the CPU, one call | 8 physical cores | one batched call in a fresh process; the sparse backend on eight threads, the dense backend on one thread above 128 rows |
| pandapower | 8 physical cores | `runpp` per scenario with numba, the pi transformer model and recycled network matrices, over eight worker processes with one primed network each; `runpp_3ph` on the three-phase grid |
| power-grid-model | 8 physical cores | its native batch calculation with the iterative-current method and eight threads, in a fresh process |
| OpenDSS | 8 physical cores | OpenDSSDirect.py, eight worker processes with one thread each, each with one compiled circuit |

Eight physical cores means an allocation of sixteen logical cpus on a machine
with two hardware threads per core. Every result records the affinity mask it
ran under and the physical core count derived from the core topology, so the
allocation is a measured property of the run rather than an assumption. The BLAS
and torch thread caps are set to the physical core count for every engine.

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
  two single-process tools (power-grid-model, pgml's single call) have no workers
  to pay, and each of them is timed in a fresh process that has never held a
  pool. The worker hand-over is a genuine cost of running an engine over
  processes and it is why the pooled arms lose at a batch of one. What is not
  asymmetric any more is the result: every engine's timed call, pgml on the GPU
  included, ends with the voltage magnitudes in host memory.
- pgml and OpenDSS build the same harmonic device shunt but not quite the same
  line impedance above the fundamental. The harmonic comparison is therefore
  accepted at 1e-4 per unit rather than the 1e-6 the fundamental comparison uses,
  and each recorded point carries the tolerance it was judged at.
- pandapower and power-grid-model have no harmonic power flow, so the harmonic
  figure has one reference tool instead of three.
- The two reference engines that solve the CIGRE LV grid in three phases are fed
  different zero-sequence fills, because its source network carries none.
  pandapower gets pgml's own converter defaults (lines `R0 = 4 R1`, `X0 = 3 X1`,
  `C0 = 0.5 C1`; the external grid `Z0 = Z1`; the transformers `Dyn` with the
  positive-sequence leakage) plus pandapower's own suggested placeholders for the
  zero-sequence magnetising branch, which pgml does not model. power-grid-model
  gets its converter's customary `R0 = 3 R1`, `X0 = 3 X1`, `C0 = 0`. The scenario
  draw scales every phase by the same factor, so no current flows in the
  zero-sequence network and both arms agree with pgml to below 1e-9 per unit;
  on an unbalanced draw the fills would matter.
- pgml re-assembles and re-factorises its admittance on every timed call, because
  no prepared system is passed, where power-grid-model reuses its factorisation
  across batches and OpenDSS keeps its compiled circuit. That is the
  one-call-solves-one-batch reading and it costs pgml, not the others.
- The scheduler did not hold the node exclusively for these jobs.

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
and plotted separately. OpenDSS and pgml solve one study definition, the same
thirteen orders, the same spectrum on the same loads and the same device shunt,
and run the same batch ladder from 1 to 16,384 studies in one job.

## What is timed

One timed call solves one whole batch. Each configuration gets one untimed
warm-up call followed by five timed repetitions, for every engine alike. The
figures show the median; the full sample and its interquartile range are in the
recorded JSON.

Inside the timed region, for every engine, are the solve itself and everything
needed to have the voltage magnitudes of the whole batch in host memory when the
clock stops: for the process-based tools the hand-over of scenarios to the workers
and of results back, for pgml on the GPU the copy of the result from the device.
GPU timings synchronise the device before the clock starts and before it stops.
pgml's admittance assembly and factorisation are inside as well, because no
prepared system is passed.

Outside the timed region are grid conversion, building each tool's model, moving
pgml's inputs to the GPU and all correctness checks. Each worker pool is created
once and each worker builds its model once and then solves once, untimed:
pandapower's workers unpickle and prime their own network and run one `runpp`,
OpenDSS's workers compile their circuit and run one `Solve`, pgml's workers build
their grid and solve once. No timed call lands on a process that has not solved
before, for any of the three pools.

The two single-process arms, power-grid-model and pgml's single call, are each
timed in a fresh child process that has never held a worker pool and that
inherits the parent's thread caps. A process in which a pool has existed measures
a threaded engine slower, and isolating the arm removes the question of by how
much.

Every arm stops at the first batch whose call exceeds the per-call time budget,
60 seconds in the fundamental sweeps and 120 in the harmonic one; that call is
recorded, and the batches beyond it are recorded as skipped for time rather than
left absent. A harmonic batch is admitted by one memory rule for every engine and
every pgml arm: the planned working set is the batch-shaped `[B, H, N]` vectors
plus the per-order system chunk the engine bounds itself, charged per worker for a
pool, against 0.8 of the allocation's memory on the host and the card's memory on
the device. A refused batch is recorded with the estimate and the budget.

## Validity guard

A throughput value is kept only if the solution is correct. The leading 256
scenarios of every timed batch are re-solved outside the clock and their voltage
magnitudes compared with a pgml CPU complex128 reference at 1e-6 per unit, and a
batch smaller than that is checked in full; the harmonic sweep compares 64
scenarios per batch, and its OpenDSS arm is accepted at 1e-4 per unit for the
reason given above. Each row records how many scenarios it was compared over, and
`assets/readme/provenance.json` records the largest depth observed in the run as
`validated_scenarios_per_batch`, which for this campaign is
256. The point counts if the
tool reports convergence and the largest difference stays within the tolerance
recorded with it. Points that fail are recorded without a throughput and cannot
reach a figure. The renderer repeats the check on the recorded values and refuses
to draw an invalid point.
<!-- guard.validated_scenarios_per_batch: provenance.json performance_source.validated_scenarios_per_batch (max comparison_scenarios over bench_batch_throughput_fair.json reference rows; 256 expected) -->

The pandapower runs fail at the start if numba cannot be imported, and every
worker confirms that pandapower really used numba and the pi transformer model.

## Results

### Batch size

![Throughput against batch size on four grids](readme/batch_throughput_all.svg)

pgml's throughput grows almost in proportion to the batch size, because one
batched call shares its fixed cost among all the scenarios in it. The reference
tools solve one scenario at a time and level off once their worker pool is busy.

At one scenario per batch the order is reversed. pgml is last on the 33-bus
feeder and on both Kerber networks, although on the feeder pandapower is only 1.26
times faster, and on the three-phase CIGRE LV grid pandapower is slower still, 36
scenarios per second against pgml's 111. power-grid-model
solves 5,559 scenarios per second on the 33-bus feeder and
2,635 on the Kerber network where pgml's fastest arm manages
160 and 46. That is the fixed
cost of a batched solver being paid by a batch of one, and it is the size of the
loss: at one scenario pgml is 35 times slower on
the feeder and 57 times slower on the Kerber
network, and the gap closes only as the batch grows.
<!-- batch.<grid>.pgm_b1: bench_batch_throughput_fair.json grids.<grid>.baselines.power_grid_model, batch 1 (fairness_tables.py markdown "fastest at the smallest batch"); batch.<grid>.pgml_b1: same file, grids.<grid>.series.* at batch 1, the fastest series; batch.<grid>.pgm_over_pgml_b1: the quotient of the two, hand-derived -->

Where the curves cross, against each reference tool, taking the fastest pgml arm
on the device:

<!-- crossover table: bench_fairness_tables.json fundamental.grids.<grid>.crossovers.pgml_gpu.{pandapower,opendss,power_grid_model} (fairness_tables.py markdown "Crossovers"); a range is written by hand where the ratio at the crossing batch is below 1.5 -->
<!-- lead column: bench_fairness_tables.json fundamental.grids.<grid>.crossovers.pgml_gpu.power_grid_model.ratio_at_largest and the batch it was taken at -->
| Grid | against pandapower | against OpenDSS | against power-grid-model | lead over power-grid-model at the largest common batch |
|---|---|---|---|---|
| IEEE 33, 33 rows | 64 | 4,096 | 16,384 | 4.49x at 65,536 |
| CIGRE LV, 132 rows | 1 to 16 | 4,096 | 4,096 | 3.14x at 65,536 |
| Kerber, 294 rows | 256 | 4,096 | level from 16,384 | 1.44x at 65,536 |
| Kerber x4, 1,176 rows | 256 to 1,024 | 4,096 to 65,536 | never ahead | 0.69x at 65,536 |

A crossover is given as a range where the two curves are within 1.5 times of
each other at the batch where they cross. The rule exists because a repeated
measurement on a shared node moves a tool by up to
63 per cent between runs, and a ratio inside that
band is not read as a difference. A lead below 1.5 in the last column is read the
same way: as level, not as ahead.
<!-- batch.repeat_spread_pct: bench_batch_throughput_fair.json timing.{iqr,median} of the reference rows, the largest relative spread over the run (hand-derived); drop the sentence if the run logs show the node was exclusive -->

The batch at which pgml overtakes power-grid-model does not rise steadily with
the grid: it is 16,384 scenarios on the 33-row feeder, 4,096 on the 132-row CIGRE
LV grid, 16,384 on the 294-row network, and never on the 1,176-row one. The lead
it reaches at the largest batch does fall with the grid:
almost four and a half times on a 33-row feeder,
just over three times at 132 rows,
less than one and a half times at 294 rows, and on the 1,176-row network
it does not overtake it at any batch measured.
<!-- cross.<grid>.lead_words: the lead column of the table above, rounded in words; cross.kerber_x4.verdict: "it does not overtake it at any batch measured" if bench_fairness_tables.json says never ahead, else the batch and ratio -->

pgml on eight CPU worker processes follows the same shape one level down. It
overtakes pandapower at a batch of 1 on CIGRE LV, 16 on the 33-bus feeder, 64 on Kerber (level there, 5.2 times at 256) and 256 on the 1,176-row network, OpenDSS at
4,096 on the 33-bus feeder only (on Kerber it draws level at 16,384, 1.13 times, and on CIGRE LV and the 1,176-row network it stays behind at every batch, at best 0.98 and 0.91 times), and power-grid-model
at no batch on any of the four grids; at best it reaches 0.85 times power-grid-model's throughput, on CIGRE LV.
<!-- cross_pool.*: bench_fairness_tables.json fundamental.grids.<grid>.crossovers.pgml_cpu_pool.{pandapower,opendss,power_grid_model}, summarised in words across the four grids -->

pandapower's three-phase solver runs on the CIGRE LV grid in this campaign, on
the zero-sequence data described above, and agrees with pgml to
6.7e-10 per unit; its arm is measured,
not absent.
<!-- batch.cigre_lv3.pandapower_deviation_pu: bench_batch_throughput_fair.json grids.cigre_lv3.baselines.pandapower[*].reference_deviation_pu, the largest over the ladder -->

### Grid size

![Throughput against grid size at four batch sizes](readme/size_scaling.svg)

Here the batch is fixed and the grid grows from 16 to 4,096 buses, every grid a
pandapower network every tool solves. pgml factorises a dense matrix on the GPU,
and on the CPU the faster of a dense and a sparse factorisation is shown. The
reference tools use sparse solvers that exploit the radial structure, so they
lose less speed as the grid grows. On this ladder a bus is one node-phase row,
so the sizes below are rows as well as buses.

At one scenario per batch pgml is the slowest tool at every size except 16 rows,
where its single call is 1.43 times as fast as pandapower, and it is level with
pandapower at 32 rows (0.95 times); both are within the 1.5 band. At sixteen
scenarios per batch it is behind power-grid-model and OpenDSS at every size, but
its single call is ahead of pandapower at 16, 32 and 64 rows (3.29, 2.53 and 1.53
times) and level with it at 128 rows (1.12 times); from 256 rows it is the slowest
tool again. At 256 scenarios per batch it leads pandapower up to 1,024 rows and
still trails power-grid-model and OpenDSS at every size. At 4,096 it leads
OpenDSS and pandapower on most of the ladder, and power-grid-model only up to a
point. Taking
power-grid-model as the tool to beat, pgml on the GPU at 4,096 scenarios per
batch is 1.44 times as fast at 256 rows,
1.22 times at 512, 0.92 times at 1,024, and
0.34 times at 4,096 rows, so it is
never 1.5 times as fast at any size: it is ahead only from 64 to 512 rows, by at most 1.44 times, which the 1.5 rule reads as level, and power-grid-model is 1.69 and 2.88 times as fast at 2,048 and 4,096 rows. Against OpenDSS at the same batch pgml is
4.03 times as fast at 256 rows and 1.48
times at 4,096. Against pandapower pgml leads from 256 scenarios per batch at
every size up to 1,024 rows; at 2,048 rows it passes pandapower only at 4,096 scenarios per batch, and at 4,096 rows it is level with it (1.33 times).
<!-- size.<tool>.r<rows>: bench_fairness_tables.json size.rungs.<rows>.crossovers.pgml_gpu.<tool>.ratio_at_largest at batch 4,096 (fairness_tables.py markdown "Grid size"); size.pgm.verdict_words: the rung range where the ratio is at or above 1.5, in words; size.pandapower.lead_words: the rungs where the pandapower ratio is at or above 1.5 -->

pgml on eight CPU worker processes never overtakes power-grid-model at
the sizes and batches measured here.
<!-- size.pool_pgm_words: bench_fairness_tables.json size.rungs.*.crossovers.pgml_cpu_pool.power_grid_model, "never overtakes" if every rung says never ahead -->

### Harmonic studies

![Throughput of whole harmonic studies against batch size](readme/harmonic_throughput.svg)

A study is one nonlinear solve at the fundamental plus a linear solve at each of
twelve further odd orders. OpenDSS is the only reference tool that can do it;
pandapower and power-grid-model have no harmonic power flow.

Above the fundamental a load is a harmonic current source in parallel with a
shunt admittance `conj(P + jQ)/V_rated²`, which OpenDSS splits half and half
between a parallel and a series R-L branch. `P`, `Q` and `V_rated` are the values
declared on the element, not quantities read off the fundamental solution: the
same load returns the same order-5 admittance whether its terminal settles at 224
or at 217 volts. What the fundamental solution carries is the current injection,
whose magnitude and angle follow the element's own fundamental current.

A scenario here multiplies every load's declared power, so every shunt in the
circuit moves with it. OpenDSS's admittance scales exactly with the edited power
and the system matrix is rebuilt whenever an element's is, and again at every
order because branch reactances scale with frequency. Both engines therefore
factorize one matrix per scenario and per order. OpenDSS never holds more than
one of them, because it solves the scenarios one after another. pgml holds a
chunk of the batch at once: on the GPU as dense matrices, on the CPU sparse
backend as their structural entries (the next subsection), and in both cases as
many scenarios per chunk as its system budget admits. That is where its memory
goes.

pgml's default builds the same expression from the power the device draws at the
converged fundamental over the same rated voltage, which for a constant-power
device is the declared power. That is the configuration in the figure above and
the one that reproduces OpenDSS. On sixteen scenarios of the 33-bus feeder the
two engines agree to 3.2e-5 per unit on harmonic voltages of about 1.9e-2 per
unit (a workstation measurement of 2026-09-24, described under the shunt basis
below). The arm is nonetheless accepted at 1e-4 rather than 1e-6, because the two
line models still differ above the fundamental.

On the 33-bus feeder OpenDSS levels off near 8,000
studies per second. pgml's GPU passes it at 1,024 studies per batch, and its
eight-worker pool draws level there (1.38 times) and leads from 4,096. pgml
reaches 38,772 on the GPU and
14,846 on eight CPU workers at
16,384 studies, so 4.84 and
1.85 times OpenDSS.
<!-- harm.ieee33.*: bench_fairness_tables.json harmonic.grids.ieee33 (fairness_tables.py markdown "Harmonic"): the OpenDSS peak, crossovers.pgml_gpu.opendss, the pgml_gpu and pgml_cpu_pool peaks with their batch, and ratio_at_largest of both -->

On the two larger grids the picture is reversed on the GPU and at best level on eight CPU workers. On the
294-row Kerber network OpenDSS reaches 1,992 studies
per second against 1,983 for pgml's best arm
(eight CPU workers, at 16,384 studies per batch; the GPU peaks at 270), and on the 1,176-row network
394 against 323
(eight CPU workers, at 1,024; the GPU peaks at 5.4). pgml draws level with OpenDSS on Kerber only at 16,384 studies per batch on eight CPU workers, by 1.04 times, which the 1.5 rule reads as level, and passes it at no batch on the 1,176-row network; on the GPU it stays 7.4 and 73 times below OpenDSS's peak.
<!-- harm.<grid>.*: bench_fairness_tables.json harmonic.grids.{kerber,kerber_x4}: the OpenDSS peak, the best pgml peak and which series it is, crossovers.*.opendss; harm.large.verdict_words: "the same" or "reversed", from the ratios; harm.large.crossover_words: "never overtakes OpenDSS on either" if both say never ahead, else the batches -->

The harmonic figure measures pgml 0.5.1 at revision 966df9d of its main branch
with the merged solver branch, the sparse harmonic assembly and the merged
cache-defaults branch, source digest c3bc1b3d1049. That revision carries the
structural route of the next subsection, which the CPU sparse backend takes at
full precision; the GPU arm assembles the dense matrix either way.
<!-- env.pgml_git: bench_harmonic_batch_fair.json environment.versions.pgml_git (provenance.json performance_source.versions.pgml_git); state after it whether that revision carries the structural route -->

#### Assembling the harmonic matrices sparsely on the CPU

On the CPU sparse backend, at full precision, pgml assembles each scenario's and
each order's admittance as its structural nonzero entries, `[B, H, nnz]`, and
factors the sparse system from those entries. The dense `[B, H, N, N]` tensor is
never formed on that route. Everything else is unchanged: the same stamps are
accumulated in the same order at the same positions, so the solved voltages are
bit-identical to the dense route at equal scenario chunking, on the IEEE feeder,
both Kerber networks and a mixed grid with fusion, delta loads and batched switch
states. At different chunkings the last bit of the injection vector differs, on
either route, and the difference is at the 1e-14 level.

It is a host-memory change first. Measured on the workstation (Intel i7-12700,
RTX A2000 12 GB, a 58 GB allocation), with the same harness for both engine
revisions, one harmonic study on the 1,176-row network peaks at 1,058 instead of
4,680 MiB of proportional set size at 64 studies per batch and at 1,525 instead of
16,837 MiB at 256, eleven times less; the eight-worker pool peaks at 6,168 instead
of 16,714 MiB at 256. On the 33-bus feeder and the 294-row network the library's
automatic choice is the dense backend on both revisions, and their footprints
agree within noise. On the same 58 GB allocation the previous revision's CPU
sparse study was killed for memory on the 294-row network at 16,384 studies per
batch, where the new one completes. The batch a host holds grows where one
scenario's dense system is large against the chunk budget, which on these grids
is above a thousand rows; on a small grid the batch is bounded by the
`[B, H, N]` injection and solution vectors, which grow on both routes.

The CPU sparse arms also run faster on that workstation. At the largest batch
both revisions completed, one call solves 2.6, 5.7 and 20 times as many studies
per second as the previous revision on the 33-bus feeder, the 294-row and the
1,176-row network (at 16,384, 4,096 and 256 studies per batch), and eight
workers 2.0, 6.5 and 17 times. The dense and GPU arms, which the change does
not touch, give 0.94 to 1.00 times at the same batches and 0.68 to 1.02 times
across all shared batches, which is the spread between two runs on that machine.
The two runs also differ in the harness's memory rule, which only decides where
a series stops. The figure above is from the cluster campaign and uses the new
revision only.

The route does not serve CUDA. A study on the GPU assembles the dense matrix
exactly as before, so device-side capacity and the GPU numbers on this page are
unchanged by construction. The dense and block backends, mixed precision and the
low-rank device-shunt update also keep the dense route, and `auto` below 512 rows
resolves to the dense backend, so a small grid solved with the defaults does not
take it either.

`solver.harmonic.system_budget_mb` keeps its meaning as the bound on one chunk:
on the dense route the dense matrices and their factorisation, on the structural
route the chunk's entries and its sparse factors, each system charged for the
factor storage SuperLU keeps. On neither route does it bound the `[B, H, N]`
vectors, which are what grows with the batch.

The wall-clock effect depends on the machine. Where the dense assembly was a
fifth to a third of a study, removing it is worth well under 1.5x end to end; on
a machine that spent most of the study in that assembly the factor was near
seven. The claim to carry is the memory one; a time factor is quoted only with
its hardware.

#### What the per-scenario matrix costs

![Studies per second for a per-scenario and a shared device shunt, against batch size](readme/harmonic_shunt_basis.svg)

`load_shunt_basis="nameplate"` builds the shunt from each device's stored power
instead and ignores the scenario, so `Y(h)` is one matrix per order shared by the
whole batch and one factorization per order answers all of it. It is the cheaper
model and it is a different one. On the 33-bus feeder it sits 1.6e-4 per unit
from OpenDSS against 3.2e-5 for the default, five times further away and past the
1e-4 this arm is accepted at; on the two Kerber networks, where the loads are
smaller against the source, it is about twice as far and still inside.

Measured on a workstation card (RTX A2000, 2026-09-24; not the cluster campaign),
each basis at its own fastest batch, against OpenDSS on the same scenarios in the
same job:

| Grid | OpenDSS, 8 workers | pgml GPU, per-scenario shunt | pgml GPU, shared shunt | agreement, per-scenario | agreement, shared |
|---|---|---|---|---|---|
| IEEE 33, 33 rows | 11,300 | 12,100 | 76,200 | 3.2e-5 | 1.6e-4 |
| Kerber, 294 rows | 2,470 | 93 | 4,130 | 8.5e-6 | 1.6e-5 |
| Kerber x4, 1,176 rows | 434 | 2.1 | 225 | 9.7e-6 | 1.6e-5 |

Throughput is studies per second, agreement the largest difference in per-unit
voltage magnitude. So the matched model is the one that costs, and on that card
it costs a factor of six on the smallest grid and a hundred on the largest. With
it, pgml matches OpenDSS on the 33-bus feeder and is 27 and 207 times slower on
the two larger ones. Without it, pgml is seven times OpenDSS on the feeder and
1.7 times on the 294-row network, and still half its speed at 1,176 rows. These
are the workstation's factors; the cluster figure above carries its own.

The solver does not hold the whole `[B, H, N, N]` admittance on either route. It
assembles and factors as many scenarios at a time as fit
`solver.harmonic.system_budget_mb` and concatenates the results, so the peak is
bounded by that budget and not by the batch. On the 294-row network the device
peak holds at 1,540 and 1,552 MiB while the nominal admittance grows from 1.07 to
4.29 GiB, and on the 1,176-row network at 1,534 MiB against a nominal 4.29 GiB.
The shared shunt needs 114 and 998 MiB for the same batches, because there is one
matrix per order rather than one per scenario. The harmonic curves in this
workstation figure stop earlier than the fundamental ones because the harness of
that date refused a batch whose nominal dense admittance exceeded a fixed cap,
which was a harness guard rather than a capacity of the card; the cluster
campaign admits batches by the memory rule described under "What is timed" and
stops them by the time budget instead.

An exact alternative exists and does not apply here. The solver can factor the
shunt-free network once and reach each scenario's own matrix through a low-rank
correction, which is selected when three times the number of node-phase rows a
device shunt touches stays below the row count. A distribution feeder carries a
load on most buses, so that number is 32 of 33 rows on the IEEE feeder and 146 of
294 and 584 of 1,176 on the two Kerber networks. At half the rows a low-rank
correction costs more than a fresh factorization, so the per-scenario assembly is
what runs.

<sub>This subsection only: a workstation measurement, not the cluster campaign. Measured 2026-09-24, library 0.5.1, complex128, thirteen orders,
one NVIDIA RTX A2000 12 GB and the CPU of a 16-core workstation, medians of three repeats
after warm-up; the CPU arm runs on one thread wherever the dense backend is selected,
for both bases alike. Agreement is the largest absolute difference in per-unit voltage
magnitude against a live OpenDSS on the matched circuit over the same sixteen scenarios.</sub>

## Solver configuration

The comparison above runs the solver's defaults. Three choices inside the solver
change what it costs, measured separately on a workstation card (RTX A2000,
2026-09-24; not the cluster campaign).

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

<sub>This section only: a workstation measurement, not the cluster campaign. Measured 2026-09-24, library 0.5.1, complex128 unless stated, one NVIDIA RTX A2000 12 GB and
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
PyTorch allocator's peak, which excludes the CUDA context. pgml's CPU arms run
the backend the library selects for the grid, and each record names the backend
it resolved to.

Nothing is subtracted. The tree at rest, measured after the model is built and
the pool is up but before the solve starts, is recorded beside the peak.

Host memory at the largest measured batch (4,096 scenarios),
in MiB:

<!-- host table: bench_footprint.json tree_peak_pss_mib per engine and grid at the largest batch (fairness_tables.py markdown "Memory footprint"); mem.largest_batch: the batch of those rows -->
| Tool | IEEE 33 | Kerber, 294 rows | Kerber x4, 1,176 rows |
|---|---|---|---|
| pgml CPU, one call | 632 | 978 | 1,717 |
| power-grid-model, 8 threads | 823 | 932 | 1,334 |
| pgml GPU | 1,366 | 1,426 | 1,624 |
| pgml CPU, 8 workers | 3,445 | 3,775 | 5,625 |
| OpenDSS, 8 workers | 3,972 | 4,040 | 4,258 |
| pandapower, 8 workers | 5,135 | 5,225 | 5,460 |

Three things follow.

A worker pool costs about 3.5 to 4.4 GB before it solves
anything, because eight interpreters each hold their own copy of the library. That
cost barely moves with the grid or the batch, so it dominates every small job.
pgml's pooled arm and OpenDSS pay it; pandapower pays it and a little more.
<!-- mem.pool_standing_gb: bench_footprint.json tree_baseline_pss_mib of the three pooled engines, rounded to GB -->

A single process is far leaner. pgml in one batched call and power-grid-model in
its native threaded batch both stay under 2 GB at
every point measured, and pgml's single call is the smallest of all on
the 33-bus feeder at every batch measured, on Kerber up to 1,024 scenarios and on the 1,176-row network up to 64. pgml on the GPU needs about
1.2 to 1.7 GB on the host, almost all of it the CUDA runtime, and
then grows on the device instead.
<!-- mem.single_process_cap_gb: the largest tree_peak_pss_mib of pgml_cpu_single and power_grid_model in bench_footprint.json, rounded up; mem.single_smallest_grids: the grids where pgml_cpu_single has the smallest peak; mem.gpu_host_gb: pgml_gpu tree_peak_pss_mib -->

Device memory has two parts. The matrix and its factorisation do not depend on
the batch and grow with the square of the grid, 8.3 MiB on
the 33-bus feeder and 93 MiB on the 1,176-row network.
Everything the iteration carries has a scenario axis and grows with the batch and
linearly with the grid, 7.7 KiB per scenario on the
33-bus feeder, 52.4 KiB on Kerber and
205.6 KiB on the 1,176-row network. At
4,096 scenarios the second part is by far the larger, giving
38.9 MiB, 223.2 MiB and
915.1 MiB in total on the device.
<!-- dev.fixed.<grid>: bench_footprint.json pgml_gpu device_peak_allocated_mib at batch 1; dev.per_scenario.<grid>: (device_peak_allocated_mib at the largest batch minus at batch 1) over the batch difference, hand-derived; dev.total.<grid>: device_peak_allocated_mib at the largest batch -->

### What the memory is made of

The three groups in the table are three different things, and what separates them
is not the engine.

A worker pool holds nine interpreters, one parent and eight workers, and each of
them imports the library, builds its own model of the grid and keeps its own
working arrays. A page mapped by several of them is already split across them by
the proportional set size, so those gigabytes are pages no other process holds.
They are there before the first scenario is solved. Measured at rest, with the
pool up and nothing solving, pgml over eight workers stands at
3.5 to 3.9 GB, OpenDSS at 4.1 to 4.4 and
pandapower at 5.3 to 5.6, and the grid and the batch barely
move those numbers.
<!-- rest.*_range_gb: bench_footprint.json tree_baseline_pss_mib, the range over grids and batches per pooled engine -->

One process holds the same things once. pgml in a single batched call stands at
529 to 687 MiB with the grid loaded and its injection plan
built, power-grid-model at 802 to 1,102 MiB with its model built,
and pgml on the GPU at about 671 to 796 MiB on the host, most of
which is the CUDA runtime. What a pgml process then adds for the work itself is
one admittance matrix, one factorisation of it, and one tensor per quantity the
iteration carries, each of them with a scenario axis.
<!-- rest.single_range_mib, rest.pgm_range_mib, rest.gpu_host_mib: bench_footprint.json tree_baseline_pss_mib of pgml_cpu_single, power_grid_model and pgml_gpu -->

That is why the two kinds of tool grow differently with the batch. On the
1,176-row network, measured between one scenario and 4,096:

<!-- standing column: bench_footprint.json tree_baseline_pss_mib at batch 1 (pgml_gpu: host); added per scenario: (tree_peak_pss_mib at the largest batch minus at batch 1) over the batch difference, for pgml_gpu the same on device_peak_allocated_mib; hand-derived, no script prints it -->
| Tool | standing | added per scenario |
|---|---|---|
| pgml CPU, one call | 552 MiB | 273.7 KiB |
| pgml GPU | 686 MiB on the host | 205.6 KiB on the device |
| power-grid-model, 8 threads | 824 MiB | 127.5 KiB |
| OpenDSS, 8 workers | 4,087 MiB | 42.6 KiB |
| pgml CPU, 8 workers | 3,578 MiB | 507.0 KiB |
| pandapower, 8 workers | 5,197 MiB | 65.6 KiB |

A per-scenario engine needs only the scenario it is solving, so what grows with
the batch is the input matrix and the result array and nothing else. pgml keeps
every scenario of the batch in flight, so it pays about
2.15 times as much per scenario, and a batched
factorisation and a batched back-substitution are what it buys with that.
<!-- slope.pgml_over_pgm_factor: the single-call per-scenario slope over power-grid-model's, from the table above -->

The two costs cross. On the 1,176-row network one pgml process passes
power-grid-model at about 600 (read off the measured points, where it is leaner at 64 and heavier at 1,024; the averaged slopes above would put it near 1,900, but the growth is steeper at small batches) scenarios per batch
and the pools at about 15,700 for OpenDSS and 22,900 for pandapower, extrapolated from the table; pgml's own pool adds more per scenario than one process and is never passed. Below that pgml is the
leanest way to solve the case, and above it the pool's standing cost has been
amortised while pgml is carrying the batch.
<!-- slope.cross_*_scenarios: linear extrapolation of the standing and per-scenario columns above, hand-derived -->

The lean end is worth something concrete. Eight worker processes need about
5.4 GB before they solve anything, so a container
of that size cannot start pandapower's arm at all, while pgml solves the same
grid there in one process with room for about 17,000
scenarios in the batch. The other end is worth something too, and it is the next
section.
<!-- mem.pandapower_standing_gb: bench_footprint.json pandapower tree_baseline_pss_mib on kerber_x4; mem.container_scenarios: (that size minus the single-call standing) over the single-call per-scenario slope, hand-derived -->

### What fits on a 48 GB card

The capacity search doubles the batch until the device runs out of memory.

<!-- frontier table: bench_footprint.json device_frontier per grid (fairness_tables.py "largest batch measured to fit"): largest_fit, device_peak_allocated_mib there, and how the search ended (oom batch or the cap) -->
| Grid | largest batch measured to fit | device memory there | how the search ended |
|---|---|---|---|
| Kerber x4, 1,176 rows | 131,072 | 26.4 GiB | out of memory at 262,144 |
| Kerber, 294 rows | 262,144 | 13.2 GiB | reached the search cap of 262,144 |
| IEEE 33 | 262,144 | 1.9 GiB | reached the search cap of 262,144 |

Only a row whose search ended in an out-of-memory is a capacity. A row that
reached the search cap is a lower bound: the real limit is higher and was not
measured.

## Filling the machine with scenarios

Every figure above fixes the batch and compares speed. The other question is how
many scenarios a machine holds at once, and how fast they are solved when it is
full. It is worth asking because the tools answer it differently. A per-scenario
engine never holds more than the scenario it is working on, so its batch is a
convenience; pgml holds the whole batch and gets its speed from it, so its batch
is a resource.

The measurement fills one workstation, an RTX A2000 with 12 GB beside sixteen
CPUs, on 2026-09-24. That is a much smaller accelerator than the L40S every
figure above uses, so these numbers say what one workstation does and are not a
second reading of the comparison. For each tool and grid the batch doubles until
the memory ceiling is passed, a probe runs out of memory, the batch cap of
131,072 is reached or one solve takes longer than a minute. The ceiling is 10.5
GiB of device memory for the GPU arm and 32 GiB of host memory for every other
arm, measured the way the memory figure measures it, with nothing subtracted.
Every probe runs in its own interpreter, so a failure ends the probe and not the
search. A search that ended at the cap or the time limit is a lower bound and
not a capacity, and the figure draws those with an open marker.

![Scenarios that fit and the throughput reached there, against grid size](readme/capacity.svg)

Scenarios per second at the batch each tool reached:

| Tool | IEEE 33 | Kerber, 294 rows | Kerber x4, 1,176 rows |
|---|---|---|---|
| power-grid-model, 8 threads | 254,800 | 45,500 | 10,300 |
| pgml GPU | 126,900 | 15,400 | 1,500 |
| pgml CPU, 8 workers | 106,600 | 13,200 | 2,600 |
| OpenDSS, 8 workers | 103,200 | 27,900 | 5,700 |
| pgml CPU, one call | 41,600 | 2,100 | 1,000 |
| pandapower, 8 workers | 1,800 | 1,300 | 760 |

Every one of those batches is 131,072, the search cap, with three exceptions. pgml
on the GPU at 1,176 rows ran out of device memory at 65,536 and is reported at
32,768, which is the only measured capacity in the whole table. pgml's single call
and pandapower at 1,176 rows passed the one-minute budget at 65,536 and are
reported there.

What held those batches on the largest grid: power-grid-model 14.5 GiB of host
memory, pgml over eight workers 30.9, pgml's single call 15.8, pandapower 7.3,
OpenDSS 7.1, and pgml on the GPU 7.2 GiB of device memory for a quarter of the
batch.

Three things come out of it.

Memory is rarely what stops a tool. On the two smaller grids every arm reached
the batch cap or its own time limit with the ceiling untouched, and on the
largest grid only the GPU arm was stopped by memory. What limits a study on this
machine is time, and after that price. Memory decides whether a tool fits at all,
which is what the previous section is about, and then stops deciding.

power-grid-model is first at capacity on every grid here, by a factor of two on
the 33-bus feeder and nearly seven on the 1,176-row network. That is a different
order from the fixed-batch figures above, where pgml on the GPU leads it on the
33-bus feeder. The metric is not what changed. The accelerator
is: an RTX A2000 is a workstation card with a fraction of an L40S's
double-precision throughput, and pgml's GPU arm is the one arm that depends on
it. Who wins therefore depends on which accelerator is bought at least as much as
on how the batch is chosen, which is why neither number should be read as a
ranking of engines.

The metric itself rewards a property only one tool has. A per-scenario engine
never needs its scenarios resident, so filling the machine measures pgml's design
and very little about the others, and their throughput at capacity is the plateau
the batch figure already showed. The number is worth reporting because it says
what a single call can hold, not because it settles anything.

<sub>This section only: a workstation measurement, not the cluster campaign. Measured 2026-09-24, library 0.5.1, complex128, one NVIDIA RTX
A2000 12 GB and sixteen CPUs of one workstation on an idle node, eight worker processes or
eight threads for every arm that uses them, one engine at a time, every probe in a fresh
interpreter.</sub>

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
includes the host, so an idle host on a GPU-bound job is paid for either way. The
GPU side is the card that was measured, and the CPU side has the eight physical
cores every CPU engine was measured on; every pgml CPU arm is priced on the same
CPU instance as pandapower, power-grid-model and OpenDSS.

The ratio that matters is 2.6: the GPU instance costs 2.6 times the CPU
instance. A pgml GPU throughput less than 2.6 times the best CPU throughput
means the CPU allocation is the cheaper way to buy those solves, however much
faster the card is.

![Cost per million solved scenarios against batch size](readme/cost_per_million.svg)

Cost per million solved scenarios, each tool at its own fastest measured batch:

<!-- cost table: bench_cost_fair.json cost_named_grids.<grid>.series.<engine>.usd_per_million (fairness_tables.py markdown "Cost per million") -->
| Tool | IEEE 33 | CIGRE LV, 132 rows | Kerber, 294 rows |
|---|---|---|---|
| pgml GPU | 0.000387 | 0.00234 | 0.00616 |
| power-grid-model | 0.000652 | 0.00293 | 0.00326 |
| OpenDSS | 0.00201 | 0.00339 | 0.00765 |
| pgml CPU, 8 workers | 0.000797 | 0.00345 | 0.00693 |
| pgml CPU, one call | 0.00121 | 0.0124 | 0.0207 |
| pandapower | 0.0904 | 0.659 | 0.124 |

The cheapest way to buy these solves is pgml on the GPU on the
33-bus feeder, pgml on the GPU on CIGRE LV and
power-grid-model on the 294-row network, where pgml on the GPU costs
1.89 times power-grid-model's price. The cost
crossover comes at a smaller grid than the throughput crossover, and that is the
point of pricing at all: where pgml is faster than power-grid-model at a large
batch by less than the price ratio, it is already the more expensive way to get
the answer, because the accelerator costs 2.6 times the eight cores.
<!-- cost.cheapest.<grid>: the engine with the smallest usd_per_million in the table above; cost.gpu_over_pgm.kerber: cost.gpu.kerber over cost.pgm.kerber -->

pandapower is 5.9 to 53 times more expensive per scenario
than anything else here.
<!-- cost.pandapower_factor_words: the pandapower row over the next most expensive row, in words ("two orders of magnitude" if the ratio is near 100) -->

## Where pgml wins and where it loses

Use pgml when you need many operating points of a small or medium grid, or when
you need gradients through the solve, which no other tool here offers. Use a
dedicated solver when you need a few answers, or when the grid is large.

pgml wins

- on batches of thousands of scenarios on the 33- and 132-row grids of the batch
  figure, where one batched call amortises the fixed cost of a solve over the
  whole batch: against power-grid-model from 16,384 scenarios per batch on the
  33-row feeder and from 4,096 on CIGRE LV, by 4.49 and 3.14 times at the largest
  common batch. On the 294-row network it passes power-grid-model at 16,384 but
  only draws level (1.44 times at 65,536), and on the size ladder it is never more
  than 1.44 times ahead of it, which the 1.5 rule also reads as level;
- against pandapower from a few scenarios per batch even on eight CPU workers: from 1 on CIGRE LV, 16 on the 33-bus feeder and 256 on both Kerber networks;
- against OpenDSS at 4,096 scenarios per batch on the size ladder, by
  4.03 times at 256 rows and by at least 2.03 times at every size up to 2,048
  rows; at 4,096 rows it is level (1.48 times);
- on cost per million scenarios on the 33-bus feeder and CIGRE LV, where it is
  the cheapest tool in this comparison.
<!-- cost.cheapest_grids_words: the grids of the cost table where pgml GPU has the smallest usd_per_million -->

pgml loses

- at a few scenarios, badly. At one scenario per batch it is behind
  power-grid-model on every grid by more than an order of magnitude (28.8 to 64.1
  times) and the slowest tool on three of the four; on CIGRE LV pandapower's
  three-phase solver is slower. It stays behind power-grid-model until 16,384
  scenarios per batch on the 33-bus feeder and the 294-row network, until 4,096 on
  CIGRE LV, and at every batch on the 1,176-row network. A batched engine pays its
  fixed cost once whether the batch holds one scenario or a hundred thousand.
<!-- batch.b1_loss_range_words: the range of pgm_over_pgml_b1 across the four grids, in words ("one to two orders of magnitude" if it spans 10 to 100) -->
- against power-grid-model from about 1,024 node-phase
  rows upward, where it is never ahead at any batch size measured, and from 2,048
  rows by 1.69 times or more at 4,096 scenarios per batch. power-grid-model is the tool to beat
  in this comparison, and on larger grids it is not beaten.
<!-- size.pgm_crossover_rows: the first rung of bench_fairness_tables.json size.rungs where pgml_gpu vs power_grid_model ratio_at_largest falls below 1 at batch 4,096 -->
- on harmonic studies of anything but the smallest grid. In the model OpenDSS
  itself solves, where every scenario carries its own admittance, OpenDSS's peak
  on the two larger grids in this campaign is 1.01 and 1.22 times that of pgml's
  best arm, eight CPU workers, which the 1.5 rule reads as level, and 7.4 and 73
  times that of pgml on the GPU (27 and 207 times on the workstation card of the
  shunt-basis subsection).
<!-- harm.<grid>.opendss_over_pgml: harm.<grid>.opendss_peak over harm.<grid>.pgml_peak from the harmonic subsection -->
- on standing memory whenever it is run over worker processes: eight
  interpreters cost about 3.5 to 3.9 GB before any scenario is
  solved.

One further thing a reader should take from the numbers rather than from the
headline.

pgml on the same eight cores the other tools get, over eight worker processes,
peaks at 18.6 per cent of the GPU peak on the 33-bus feeder, 26.0 on CIGRE LV,
34.0 on Kerber and 56.6 on the 1,176-row network. At that peak it is 2.52 and 1.10
times OpenDSS's peak on the feeder and on Kerber, 0.98 and 0.91 times on CIGRE LV
and the 1,176-row network, and 0.38 to 0.85 times power-grid-model's peak on every
grid. The GPU overtakes the pool at 1,024 to 4,096 scenarios per batch. The GPU
instance also costs 2.6 times the CPU instance, so a speed-up below 2.6 times is
not a saving.

### Why the large grids go the other way

That pgml factorises a dense matrix while the reference tools exploit the radial
structure is only part of the reason, and on the CPU it is not the reason at all.
Timing the phases of one batched solve says where the time goes. This diagnosis
was made on the workstation (RTX A2000, 2026-09-24), not on the cluster; the
shares below are that machine's.

Above 512 node-phase rows pgml's CPU path already factorises sparsely, and the
factorisation is a good one. It keeps a fill-reducing ordering, and on a radial
feeder that ordering comes within six per cent of the zero-fill optimum: the
factors hold four nonzeros per row at 1,176 rows and again at 4,096, so the
factorisation is linear in the grid. It is computed once per solve and reused by
every iteration and every scenario, which is what power-grid-model does too. At
4,096 scenarios per batch on the 1,176-row network, assembly and factorisation
together are two and a half per cent of the wall time. Neither a better ordering
nor a cheaper factorisation is where the gap lives.

What costs is the work repeated for every scenario in every iteration. In that
same solve the back-substitutions are 52 per cent of the wall time, the
device-current evaluation 16, the per-unit convergence test 15 and the residual
product 3, leaving 12 per cent of bookkeeping the phase timers do not attribute.
Nor is the back-substitution all solving: handing a batch of right-hand sides to
SuperLU means permuting a torch tensor into a Fortran-ordered numpy block and the
answer back again, and that copying is 30 to 40 per cent of the call.

On the GPU it is the simple story. The factorisation is dense, so one scenario's
back-substitution costs the square of the row count where a sparse one costs a
small multiple of it. A batch amortises the factorisation over every scenario in
it; it does not amortise the back-substitution, which every scenario pays in
full. That is why the GPU curve falls away with the grid at every batch size.

Four changes would narrow it, in the order of what they are worth. A sparse
back-substitution on the device would take the per-scenario cost from the square
of the grid to a small multiple of it, which is the whole of the large-grid gap,
and it needs a batched sparse triangular solve that torch does not have today.
Keeping the right-hand sides in one array layout would remove the copying around
SuperLU, about a fifth of a large CPU solve. Fusing the per-iteration passes,
which today walk a scenarios-by-rows tensor once each for the device currents,
the residual and the two convergence criteria, is worth about as much again.
Keeping the assembly and the factorisation across calls, which the harmonic path
already offers, is worth little inside one large batch and a great deal to a
caller that solves the same grid again and again at a small batch.

One more thing the timings show, which costs nothing here and a great deal
elsewhere. At the revision measured here, pgml assembles the admittance as a
dense matrix and the sparse backend converts it, so every factorisation scans
all of the row count squared entries to find the three per row that are nonzero.
Shared over 4,096 scenarios that is two microseconds each and invisible. It is
96 per cent of the cost of one factorisation at 1,176 rows and 99 per cent at
4,096, so wherever the matrix is per scenario rather than shared by the batch,
as it is in a harmonic study whose device shunt follows the operating point,
that conversion is the whole cost. That is what the structural route of the
harmonic subsection removes on the CPU sparse backend; it still holds for the
fundamental solve of this diagnosis and for every study on CUDA.

<sub>This subsection only: a workstation measurement, not the cluster campaign. Measured 2026-09-24, library 0.5.1, complex128, one NVIDIA
RTX A2000 12 GB and sixteen CPUs of one workstation on an idle node, phases timed by
wrapping the functions the solver calls, a CUDA configuration synchronising inside every
wrapper.</sub>

### What this means for choosing a tool

The grid size at which pgml stops keeping up with power-grid-model is not a
property of dense linear algebra alone, and it will not move by changing an
ordering. On the size ladder at 4,096 scenarios per batch a batched call is level
with power-grid-model up to about 1,024 node-phase rows, never more than 1.44 times
ahead, while it leads OpenDSS and pandapower by at least 1.5 times up to 2,048
rows. It wins against power-grid-model, on throughput at a large batch and on
price, only on the 33- and 132-row grids of the batch figure. From 2,048 rows up
power-grid-model is ahead at every batch size measured here, and the
reasons to reach for pgml there are that it differentiates through the solve and
that it holds a whole study in one process, not that it is faster.

## Reproduce

The scripts live in the [pgml paper repository](https://github.com/power-grid-ml/pgml-paper) (to be published soon) under `scripts/bench` and
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

`fair_batch_small` measures the IEEE 33 and CIGRE LV grids and `fair_batch_large`
the two Kerber networks, each with pgml's arms and the reference engines in one
job; `fair_harmonic` runs OpenDSS and pgml on one ladder in one job.

Without a cluster the same measurements run directly:

```bash
python scripts/bench/run_all.py --stages fair_batch_small fair_batch_large \
    fair_harmonic fair_size footprint footprint_harmonic fair_cost fair_tables
```

Then draw the figures in the pgml checkout:

```bash
pixi run -e cpu python run/readme/render.py
```

`assets/readme/provenance.json` lists the software versions and the SHA-256 of every result file the figures are drawn from.
