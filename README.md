# pgml

**Differentiable, GPU-ready, vectorized harmonic power-flow + ML for power grids.**

`pgml` is a single PyTorch library that (1) generates/loads power grids, (2) simulates
**harmonic power quality in steady state** (harmonic power flow, harmonics 1–50 plus
interharmonics), and (3) supports **machine learning on the simulated data** (graph-based
state estimation). The defining requirement is **end-to-end differentiability**:
gradients flow from grid parameters — down to line geometry — through Y-bus assembly and
the complex solve to the outputs, so the same code is a forward simulator, a
differentiable physics engine for ML, and an inverse / parameter-recovery tool.

Harmonic power flow decouples per harmonic into a *linear* complex solve
`Y(h)·V(h) = I(h)`, whose adjoint is cheap — so end-to-end gradients flow without
differentiating Newton iterations. PyTorch provides complex tensors, batched solves, GPU,
and native PyTorch-Geometric integration for the ML layer.

## Highlights

- **Phase-domain, fully asymmetric.** Per-phase, per-harmonic complex Y-bus; WYE / DELTA
  / grounded-neutral loads; two-winding transformers with real **vector groups** (a Dyn
  delta correctly traps zero-sequence / triplen harmonics).
- **Differentiable + GPU.** Every core op runs on CPU and CUDA unchanged, honors input
  device/dtype, and is vectorized/batched (no Python loops over nodes/branches/harmonics).
  `float64` `gradcheck` and a CPU/CUDA parity suite are gates.
- **Carson/Deri line model.** Conductor geometry → frequency-dependent `Z(h)`/`Yc(h)`,
  **bit-exact vs OpenDSS**, with R/X → geometry synthesis for sequence-defined feeders.
- **Validated against oracles.** Y-bus and voltages compared to pandapower /
  power-grid-model (load flow) and OpenDSS (harmonics) on IEEE-33 and CIGRE LV.
- **Reproducible batched scenarios.** QMC / cartesian sampling → operating points →
  batched solves, for ML training-data generation.

## Installation

Development uses [pixi](https://pixi.sh) (conda-based, pinned environments):

```bash
pixi run -e cpu pytest -q          # run the test suite (CPU)
pixi run -e cpu python examples/evaluate_ieee33.py
```

The package is also standard PEP 621 (`pyproject.toml`). Core install plus optional
extras (`convert`, `scenarios`, `ml`, `viz`, `opendss`, `docs`, `dev`, `all`):

```bash
pip install .                      # core differentiable engine
pip install ".[convert,viz]"       # + reference-library converters and plotting
```

## Quickstart

```python
import torch
from pgml.convert.pandapower import to_grid
from pgml.geometry import apply_default_harmonic_model
from pgml.solver import solve_harmonic_flow
import pandapower.networks as pn

grid, _ = to_grid(pn.create_cigre_network_lv())   # reference net -> pgml Grid
apply_default_harmonic_model(grid)                 # frequency-dependent line model
result = solve_harmonic_flow(grid, harmonic_orders=[1, 3, 5, 7], dtype=torch.complex128)
# result.v : complex node voltages [orders, nodes]; gradients flow back to grid params.
```

## Architecture

```
grid (schemas) ──▶ assembly ──▶ solver ──▶ result        ◀── evaluation (plots vs refs)
      │              ▲   │         ▲                       ◀── scenarios (batched inputs)
      │              │   └─ geometry (Carson Z(h)/Yc(h))   ◀── convert (pandapower/OpenDSS/pgm)
      └─ equations (residual laws, the source of physics) ─┘
```

- **`schemas/`** — frozen, framework-free contracts (`Grid`, `Node`, `Branch`,
  `Appliance`, `Result`, `Scenario`). Physical fields accept plain floats *or* tensors.
- **`equations/`** — residual-form (`0 = a − b`) SymPy registry + torch evaluators.
- **`assembly/`** — per-phase, per-harmonic, batched, differentiable Y-bus + injections.
- **`solver/`** — complex batched linear solve; nonlinear const-P/ZIP via the
  implicit-function theorem; harmonic flow.
- **`geometry/`** — differentiable Carson/Deri line constants + R/X → geometry synthesis.
- **`convert/`** — pandapower / power-grid-model / OpenDSS → `Grid`.
- **`scenarios/`** — reproducible config-driven batched sampling.
- **`evaluation/`** — comparison plots and reference oracles.
- **`config/`** — documented modeling defaults (single source of truth; `defaults.yaml`).

Each package has a `CONTEXT.md` interface ledger. Big-picture rationale lives in
`references/ARCHITECTURE.md`; an orientation guide in `HANDOFF.md`; open work in
`TODO.md`.

## Documentation

API reference and narrative docs are built with Sphinx (`docs/`):

```bash
pixi run -e docs docs            # build HTML into docs/_build/html
```

## License

See [LICENSE](LICENSE).
