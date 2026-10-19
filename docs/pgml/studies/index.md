# Planning studies

{mod}`pgml.studies` answers planning questions about a grid rather than solving one operating
point. Each study takes a {class}`~pgml.schemas.grid_schema.Grid` and returns a result object
with a JSON-ready `to_dict()`.

- {doc}`continuation`: the P-V curve through its nose (maximum loadability) by continuation
  power flow, and the Q-V curve and reactive power margin of a bus.

```{toctree}
:hidden:

continuation
```
