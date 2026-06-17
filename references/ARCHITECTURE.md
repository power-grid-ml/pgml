# pgml architecture (the big picture every agent should read once)

## What we are building
A single PyTorch library that (1) generates grids, (2) simulates harmonic power
quality in steady state (harmonic power flow, harmonics 1-50 plus interharmonics),
and (3) supports ML on the simulated data (GNN state estimation). End-to-end
differentiability is a core requirement: gradients flow from grid parameters
(incl. line geometry -> impedance) through Y-bus assembly and the solve to the
outputs, enabling gradient-based grid improvement and parameter recovery.

## Why all-PyTorch
Harmonic power flow decouples per harmonic into a LINEAR complex solve
Y(h) V(h) = I(h). A linear solve has a clean, cheap adjoint, so end-to-end
gradients flow without differentiating Newton iterations. PyTorch gives complex
tensors, complex autograd, batched `torch.linalg.solve`, GPU, and native
PyTorch-Geometric integration for the ML layer — one autograd tape end to end.
(JAX is a fallback only if PyTorch complex/sparse autograd proves insufficient,
bridged to PyG via DLPack. Do not start there.)

## The model
- Phase-domain, fully asymmetric. Each harmonic: build complex Y(h) per phase,
  solve for node voltages, derive currents/powers.
- Sources are harmonic current injections (Norton); loads/generators are a current
  source (spectrum) in parallel with a frequency-dependent shunt admittance.
- Batched over: harmonics, scenarios/steps, (later) time. Optional leading dims.

## Roadmap (we are at Phase 1)
0. Schemas frozen (done): grid / result / scenario.
1. Differentiable load flow: equations + Y-bus assembly + complex solve;
   validate Y against OpenDSS (and pandapower), validate results against
   pandapower; confirm gradcheck + GPU. (CURRENT)
2. Geometry -> impedance differentiable path (Carson, skin effect). DONE
   (`pgml.geometry`): Deri earth return + skin effect + Maxwell capacitance, bit-exact
   vs OpenDSS, differentiable/GPU/batched; closes the harmonic line-impedance gap.
3. Full harmonic range; validate harmonic results against OpenDSS. DONE — harmonic
   flow + OpenDSS-vs-pgml harmonic comparison on IEEE-33 + CIGRE LV (match via Carson).
4. Batching/scale (`pgml.scenarios`, increment 1 done); parquet persistence (deferred).
5. PyG state estimation + the inverse (parameter recovery) path.

## Validation philosophy
OpenDSS is the harmonic ground truth; pandapower/power-grid-model are load-flow
oracles. We compare the assembled Y-bus to OpenDSS's exported system Y, and node
voltages / branch flows to pandapower results, on small IEEE feeders.
