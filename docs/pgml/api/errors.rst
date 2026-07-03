pgml.errors
===========

Exception hierarchy for `pgml`.

A small, stable error model so callers can catch pgml-specific failures and a REST
adapter can map them to HTTP status codes without inspecting internals.

.. rubric:: Two branches under ``PgmlError``

- :class:`~pgml.errors.InputError` — the caller supplied something invalid (bad
  configuration, an unconvertible source network, an unsupported modeling choice).
  Maps to HTTP **422** (Unprocessable Entity).

- :class:`~pgml.errors.ComputationError` — the inputs were acceptable but the
  computation failed (e.g. the nonlinear power flow did not converge).
  Maps to HTTP **500**.

Each branch has leaf classes for specific failures: :class:`~pgml.errors.ConfigurationError`
/ :class:`~pgml.errors.ConversionError` / :class:`~pgml.errors.ModelingError` under
``InputError``, and :class:`~pgml.errors.ConvergenceError` (carries the solver
``iterations`` / ``residual`` / ``diagnostics``) under ``ComputationError``.

.. rubric:: Deprecated alias

``PgmError`` is kept as an alias of :class:`~pgml.errors.PgmlError` for existing catchers
(the ``pgm`` prefix reads as *power-grid-model*, the reference library, elsewhere in this
codebase — ``PgmlError`` is unambiguous). New code should catch or raise ``PgmlError``.

.. rubric:: Schema validation is NOT wrapped

Constructing a :class:`~pgml.schemas.grid_schema.Grid` (or any other schema model)
with invalid data raises ``pydantic.ValidationError`` directly.  Its structured,
field-level detail is exactly what a REST 422 body wants — re-wrapping it would lose
that.  Treat ``pydantic.ValidationError`` as the schema-input error alongside
:class:`~pgml.errors.InputError`.

.. rubric:: REST adapter pattern

::

    import pydantic
    from pgml import simulate, PgmlError

    try:
        result = simulate(grid, config)
    except pydantic.ValidationError as e:
        return JSONResponse(status_code=422, content=e.errors())
    except PgmlError as e:
        return JSONResponse(status_code=e.http_status, content={"error": str(e)})

.. automodule:: pgml.errors
   :members:
   :show-inheritance:
