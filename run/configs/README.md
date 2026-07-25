# run/configs — run-config templates

Starting points for the per-package, serializable run configs. Copy one into your
[experiments root](../../data/README.md) (or anywhere outside the repo), edit, and
point the tooling at it. Each config + its `seed` reproduces a run.

| Template | Schema | Used by |
|---|---|---|
| `pgml_scenario.yaml` | `pgml.scenarios.config.ScenarioConfig` | data generation (`run_scenarios` / `write_dataset`) |
| `pgl_experiment.yaml` | `pgl.config.ExperimentConfig` | training (`train_se_coherent.py --config …`) |

Inspect a schema or regenerate a fresh template directly from the code (the schema is the
source of truth, so these never drift):

```bash
python -m pgml.scenarios.config --json-schema   # or --example
python -m pgl.config --json-schema              # or --example
```

`pgg` will follow the same convention (`pgg.config`) once its generation layer lands.
