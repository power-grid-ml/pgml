---
description: Run the full load-flow validation sequence (Y-bus vs OpenDSS, results vs pandapower, differentiability, GPU) and summarize pass/fail.
---
Validate the current load-flow implementation end to end. Delegate to subagents;
keep your own context lean. Steps:

1. Ask `opendss-reference` to (re)run the Y-bus oracle test: assemble our Y(h=1)
   for the test feeder and compare to OpenDSS `Export Y`/`SystemY`.
2. Ask `pandapower-reference` to (re)run the result oracle test: our node voltages
   and branch flows vs `net.res_*` on the same feeder; and `pgm-reference` as a
   second results oracle.
3. Ask `test-runner` to run the differentiability gate (gradcheck, float64) on
   assembly+solver and the GPU device/dtype gate.
4. Ask `reviewer` for a differentiability/GPU/vectorization pass on any code
   touched since the last review.

Then summarize: which gates passed, the worst failures with likely causes, and
whether the milestone criteria (correct Y, correct results, differentiable, GPU)
are all met. Update the relevant module CONTEXT.md interface ledgers if signatures
changed.
