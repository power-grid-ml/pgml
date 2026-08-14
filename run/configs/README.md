# run/configs — run-config templates

Starting points for the per-package, serializable run configs. Copy one into your
[experiments root](../../data/README.md) (or anywhere outside the repo), edit, and
point the tooling at it. Each config + its `seed` reproduces a run.

| Template | Schema | Used by |
|---|---|---|
| `pgml_scenario.yaml` | `pgml.scenarios.config.ScenarioConfig` | data generation (`run_scenarios` / `write_dataset`) |
| `pgl_experiment.yaml` | `pgl.config.ExperimentConfig` | training (the config a driver dumps into its run directory) |
| `pgl_se_scenario.yaml` | `ScenarioConfig` + `CoherentSpectrumConfig` | `--scenario-config` of `train_se_full_workflow.py` / `run_se_ablations.py` |
| `pgl_multigrid_corpus.yaml` | `pgl.data.multigrid.MultiGridGenerationConfig` | `python -m pgl.data.multigrid --config …` (the multi-grid corpus) |
| `expected_ranges.yaml` | `pgl.data.validate.load_expected_ranges` | `--reference` of `python -m pgl.data.validate` / the drivers' `--expected-ranges` (the measured-population layer of the dataset validation) |

The state-estimation scenario RECIPE lives in code (`pgml.scenarios.presets`) and is what
the drivers generate from by default. `pgl_se_scenario.yaml` is a **deviation layer**: a
section it declares REPLACES the preset's shape wholesale, so it must stay a complete
recipe. Its `random:` section is the preset written out (a test asserts they stay equal);
its `coherent:` section shows how to deviate.

Inspect a schema or regenerate a fresh template directly from the code (the schema is the
source of truth, so these never drift):

```bash
python -m pgml.scenarios.config --json-schema   # or --example
python -m pgl.config --json-schema              # or --example
python -m pgl.data.multigrid --json-schema      # or --example
```

`pgg` will follow the same convention (`pgg.config`) once its generation layer lands.
