# Installation and Quick Start

## Prerequisites

- [pixi](https://prefix.dev/) package manager
- Linux or Windows (x86-64); CUDA 12.4+ for the GPU environment

## Install

Clone the repository and install the CPU environment:

```bash
git clone <repo-url>
cd power-grid-ml
pixi install --environment cpu
```

For GPU support (requires CUDA 12.4+):

```bash
pixi install         # default environment uses GPU
```

## Running code

All commands are run via pixi to activate the correct environment:

```bash
pixi run --environment cpu python your_script.py
```

The `PYTHONPATH` is automatically set to `src`, so `import pgml` works
without a separate install step.

## Quick start

```python
import torch
from pgml.schemas import Grid, Node, Line, Source, Phase
from pgml.assembly import assemble_ybus
from pgml.solver import solve_harmonic

# Build a minimal two-bus grid
grid = Grid(
    nodes=[
        Node(id="bus1", u_rated_v=400.0, phases=(Phase.A, Phase.B, Phase.C)),
        Node(id="bus2", u_rated_v=400.0, phases=(Phase.A, Phase.B, Phase.C)),
    ],
    branches=[
        Line(
            id="line1",
            from_node="bus1",
            to_node="bus2",
            phases=(Phase.A, Phase.B, Phase.C),
            r_ohm_per_m=[[0.206, 0, 0], [0, 0.206, 0], [0, 0, 0.206]],
            l_h_per_m=[[6e-7, 0, 0], [0, 6e-7, 0], [0, 0, 6e-7]],
            length_m=100.0,
        )
    ],
    appliances=[
        Source(
            id="src1",
            node="bus1",
            phases=(Phase.A, Phase.B, Phase.C),
            u_ref_v=230.94,
            angle_rad=0.0,
            r_s_ohm=0.0,
            l_s_h=0.0,
        )
    ],
)

# Assemble Y-bus at fundamental (50 Hz)
frequencies = [50.0]
ybus = assemble_ybus(grid, frequencies, dtype=torch.complex128, device="cpu")

# Solve V = Y^{-1} I
# (build_injections is called internally by solve_harmonic_flow)
```

## Running tests

```bash
pixi run --environment cpu pytest -q                         # full suite
pixi run --environment cpu pytest -q tests/differentiability # gradcheck gate
pixi run --environment cpu pytest -q tests/gpu               # GPU gate
```

## Linting

```bash
pixi run --environment cpu ruff check src tests
pixi run --environment cpu ruff format src tests
```

## Building the documentation

```bash
pixi run --environment docs sphinx-build -b html docs docs/_build/html
```

Or with strict warnings-as-errors:

```bash
pixi run --environment docs sphinx-build -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser.
