# Examples

Runnable, documented studies of the differentiable harmonic power-flow core:
[`pgml/`](pgml/) — see [`pgml/README.md`](pgml/README.md) (scenario studies,
OpenDSS/pandapower validation, benchmarks). Each script's module docstring is the
authoritative "what / how / outputs" reference; the README is the map.

Run any example from the repository root with `pixi`:

```bash
pixi run -e cpu python run/examples/pgml/<script>.py [out_dir]   # or -e default on a GPU host
```

The examples of the other suite packages live in their own repositories under the same
layout (`run/examples/<pkg>/`).

## Output locations

Outputs go to the untracked **`data/`** root — `data/pgml/evaluation_output/<name>/` —
anchored to the repository root via each script's own path, so a run writes there
regardless of the current working directory.

## Shared config templates

`../configs/` holds the serializable run-config templates (`pgml_scenario.yaml`) — inspect a
config schema with `python -m pgml.scenarios.config --json-schema`.
