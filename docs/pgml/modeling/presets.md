# Reference model presets

pgml defaults describe its own modeling choices. A conformance comparison selects the
reference model explicitly, because agreement under different equivalent circuits does
not measure solver accuracy.

```python
from pgml import defaults
from pgml.convert.pandapower import to_grid
from pgml.solver import solve_power_flow

with defaults.use_preset("pandapower"):
    grid, id_map = to_grid(net)  # net is a pandapower network
    result = solve_power_flow(grid)
```

Keep conversion, direct geometry calculations, preparation and solving inside the same
context. The previous selection is restored on exit, including exceptions. Omitted model
arguments in the public geometry helpers resolve inside the call; explicit arguments
still win. The IFT backward retains a snapshot of the forward defaults even if the
gradient is requested after leaving the context or after the defaults source is reloaded.
A prepared power-flow system rejects different modeling defaults; prepare it again when
changing models.

Each YAML file in `pgml/data/presets/` lists only the supported reference overrides.
Unlisted settings retain the active pgml defaults; explicit fields already stored on
a component take precedence. A preset does not rewrite existing grids, emulate every
reference feature, or change an external solver's configuration.

| Preset | Overrides and scope |
|---|---|
| `pgml` | Internal defaults: split magnetizing shunt, guarded sub-linear Carson reactance, GMR internal inductance, ideal converter switches |
| `opendss` | Last/to-terminal magnetizing shunt; GMR/Bessel frequency-band rule; unguarded sub-linear Carson law; no skin correction on lumped R/X lines |
| `power-grid-model` | Split magnetizing shunt; unsupported harmonic settings retain pgml defaults |
| `pandapower` | Split magnetizing shunt for a comparison using pandapower's `trafo_model="pi"`; pandapower's default T equivalent remains a distinct model |

OpenDSS comparisons must also align Rg/Xg, base frequency, source impedance, load
connections, spectra and harmonic load-shunt assumptions. The preset does not invent
DER internal impedance or make a synthesized conductor radius into measured data.
For a geometry comparison, use the same geometry in both tools and the OpenDSS preset;
for a model sensitivity study, state each model explicitly.

The supported choices follow the reference [OpenDSS line model](https://opendss.epri.com/LineCode1.html),
[pandapower transformer models](https://pandapower.readthedocs.io/en/latest/elements/trafo.html),
and [power-grid-model components](https://power-grid-model.readthedocs.io/en/stable/user_manual/components.html).

## Version and persisted grids

Library version 0.5.0 uses schema version 0.2.0. The additions are optional
`LineGeometry.internal_inductance` and `EarthReturnModel.x0_nonnegative` fields.
Existing JSON loads with both fields unset. Unset choices resolve from current defaults,
so reproducing an older run requires its recorded defaults as well as its grid.
Use `PGML_DEFAULTS` or `defaults.reload(path)` for a complete historical defaults file.
The ideal converter-switch default preserves explicitly supplied switch impedance.

Schema0.2.0 adds optional `Generator.harmonic_impedance` and
`Storage.harmonic_impedance`. Older grids retain absent impedance. The explicit
OpenDSS import selects its admittance-frequency and source-reference conventions;
these fields are never inferred from signed P/Q.
