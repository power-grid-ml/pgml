"""pgml exception hierarchy.

A small, stable error model so callers can catch pgml-specific failures and a REST
adapter can map them to HTTP status codes without knowing the internals.

Two branches under :class:`PgmError`:

- :class:`InputError` — the caller supplied something invalid (bad configuration,
  an unconvertible source network, an unsupported modeling choice). Maps to a 4xx;
  ``http_status`` is 422 (Unprocessable Entity).
- :class:`ComputationError` — the inputs were acceptable but the computation failed
  (e.g. the nonlinear power flow did not converge). Maps to a 5xx; ``http_status``
  is 500.

**Schema validation is deliberately NOT wrapped.** Constructing a :class:`~pgml.schemas`
model with invalid data raises pydantic's ``ValidationError``, whose structured,
field-level detail is exactly what a REST 422 body wants — re-wrapping it would lose
that. Treat ``pydantic.ValidationError`` as the schema-input error alongside
:class:`InputError`.

A REST adapter can therefore do::

    try:
        result = simulate(grid, config)
    except pydantic.ValidationError as e:
        return JSONResponse(status_code=422, content=e.errors())
    except PgmError as e:
        return JSONResponse(status_code=e.http_status, content={"error": str(e)})
"""

from __future__ import annotations

from typing import Optional


class PgmError(Exception):
    """Base class for all pgml-raised errors. ``http_status`` is a REST hint."""

    http_status: int = 500


# --------------------------------------------------------------------------- #
# Input errors (the caller's fault) -> 422
# --------------------------------------------------------------------------- #
class InputError(PgmError):
    """Invalid input: configuration, conversion, or an unsupported modeling choice."""

    http_status: int = 422


class ConfigurationError(InputError):
    """An invalid configuration value, or a malformed ``PGML_CONFIG`` override."""


class ConversionError(InputError):
    """A reference-library network could not be converted to a pgml :class:`Grid`."""


class ModelingError(InputError, NotImplementedError):
    """A requested model is not supported (e.g. an unmodelled transformer vector
    group / clock, an open/2-phase delta connection, or a zigzag winding).

    Also subclasses the builtin :class:`NotImplementedError` so that ``raise
    ModelingError(...)`` is backward-compatible with code (and tests) that catch
    ``NotImplementedError`` for an unsupported model, while still being catchable as
    :class:`InputError` / :class:`PgmError` and carrying ``http_status = 422``.
    """


# --------------------------------------------------------------------------- #
# Computation errors (the inputs were valid, the computation failed) -> 500
# --------------------------------------------------------------------------- #
class ComputationError(PgmError):
    """The inputs were acceptable but the computation failed."""

    http_status: int = 500


class ConvergenceError(ComputationError):
    """The nonlinear (const-P / ZIP) power flow did not converge.

    Carries the solver diagnostics so the caller can report or act on them: the
    number of ``iterations`` run, the final ``residual`` (infinity-norm of the
    nodal current mismatch), and a free-form ``diagnostics`` mapping (e.g. the worst
    nodes / phases). The same information is available non-fatally on
    ``PowerFlowResult`` when the strict facade is bypassed.
    """

    def __init__(
        self,
        message: str,
        *,
        iterations: Optional[int] = None,
        residual: Optional[float] = None,
        diagnostics: Optional[dict] = None,
    ) -> None:
        super().__init__(message)
        self.iterations = iterations
        self.residual = residual
        self.diagnostics = diagnostics or {}


__all__ = [
    "PgmError",
    "InputError",
    "ConfigurationError",
    "ConversionError",
    "ModelingError",
    "ComputationError",
    "ConvergenceError",
]
