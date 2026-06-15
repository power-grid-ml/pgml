# Interface ledger: convert (external formats -> our schema)

One subpackage per source: `pandapower/`, `pgm/`, `opendss/`. Each exposes a pure
function producing a valid `grid_schema.Grid` (and, where relevant, the id map back
to the source so tests can align components).

Intended public API (record final forms here):
- [ ] `convert.pandapower.to_grid(net) -> (Grid, id_map)`
- [ ] `convert.pgm.to_grid(input_data) -> (Grid, id_map)`
- [ ] `convert.opendss.to_grid(dss_handle) -> (Grid, id_map)`
Conventions: convert engineering units -> SI; record source convention in
Provenance; map sequence/nameplate inputs via the schema's input-convention DTOs;
never invent fields (schema has extra="forbid").
