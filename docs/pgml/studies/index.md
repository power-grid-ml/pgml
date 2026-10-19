# Planning studies

`pgml.studies` answers planning questions about a grid rather than solving one operating
point. Each study takes a {class}`~pgml.schemas.grid_schema.Grid`, runs batched solves, and
returns a frozen result object with host arrays, a JSON-ready `to_dict()` and a
`from_dict()`. The API reference is {mod}`pgml.studies`.

```{toctree}
:maxdepth: 1

hosting-capacity
```
