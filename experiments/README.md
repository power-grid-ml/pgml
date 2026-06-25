# experiments/ — experiment inputs & outputs (not version-controlled)

This is the default **experiments root**: where a run's reproducible-but-bulky artifacts
live — generated datasets, model checkpoints, MLflow tracking, and the resolved run
config (`experiment.yaml` / `scenario.yaml`) written next to them.

These belong with *your* experiment tracking, **not** in the library's git history, so the
directory is kept but its contents are ignored (see the local `.gitignore`). Point the
library somewhere else (e.g. fast cluster scratch) by setting the environment variable:

```bash
export PGML_EXPERIMENTS=/scratch/$USER/pgml-experiments
```

In code, resolve the root with `pgml.experiments_root()` (returns `$PGML_EXPERIMENTS` if
set, else `./experiments`). It does not create the directory; callers create what they
need under `experiments_root() / <run-name>`.

## Run *configuration* vs. run *outputs*
- **Config schemas** (code, shipped with the library): `pgl.config.ExperimentConfig`
  (training), `pgml.scenarios.config.ScenarioConfig` (data generation), and the future
  `pgg.config`. One config + its `seed` reproduces a run.
- **Config instances** (your YAML files): start from the templates in
  `examples/configs/`. Inspect a schema or print a fresh template with, e.g.
  `python -m pgl.config --json-schema` or `python -m pgl.config --example`.
- **Outputs**: written here (or under `$PGML_EXPERIMENTS`).
