# Examples

Runnable, documented studies, organised per package:

| package | examples | topic |
|---|---|---|
| **`pgml`** | [`pgml/`](pgml/) — see [`pgml/README.md`](pgml/README.md) | the differentiable harmonic power-flow core (scenario studies, OpenDSS/pandapower validation, benchmarks) |
| **`pgl`** | [`pgl/`](pgl/) — see [`pgl/README.md`](pgl/README.md) | harmonic state estimation (dataset generation, the training workflow, the estimator comparison) |
| `pgg`, … | *(added as the packages land)* | |

Run any example from the repository root with `pixi`:

```bash
pixi run -e cpu python run/examples/pgml/<script>.py [out_dir]   # or -e default on a GPU host
pixi run       python run/examples/pgl/<script>.py  [out_dir]
```

Each script's module docstring is the authoritative "what / how / outputs" reference; the
per-package README is the map.

## Output locations

Outputs go to the untracked **`data/`** root, per package — `data/pgml/evaluation_output/<name>/`
for the `pgml` examples, `data/pgl/<name>/` for the `pgl` examples — anchored to the
repository root via each script's own path, so a run writes there regardless of the
current working directory. The `pgl` examples honour `$PGML_EXPERIMENTS/<name>/` when that
variable is set (the cluster run dir).

## Shared config templates

`configs/` holds serializable run-config templates (`pgml_scenario.yaml`,
`pgl_experiment.yaml`) — inspect a config schema with `python -m <module> --json-schema`.
