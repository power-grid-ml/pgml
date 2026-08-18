# run/configs — run-config templates

Starting points for the serializable run configs. Copy one into your experiments root
(`PGML_EXPERIMENTS`, default `./data`; or anywhere outside the repo), edit, and point the
tooling at it. Each config + its `seed` reproduces a run.

| Template | Schema | Used by |
|---|---|---|
| `pgml_scenario.yaml` | `pgml.scenarios.config.ScenarioConfig` | data generation (`run_scenarios` / `write_dataset`) |

Inspect a schema or regenerate a fresh template directly from the code (the schema is the
source of truth, so these never drift):

```bash
python -m pgml.scenarios.config --json-schema   # or --example
```

The dependent packages follow the same convention in their own repositories (`pgl.config`
for training, `pgl.data.multigrid` for the multi-grid corpus, `pgg.config` for generation).
