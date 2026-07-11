"""pgml exception hierarchy.

A small, stable error model so callers can catch pgml-specific failures and a REST
adapter can map them to HTTP status codes without knowing the internals.

Two branches under :class:`PgmlError`:

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
    except PgmlError as e:
        return JSONResponse(status_code=e.http_status, content={"error": str(e)})
"""

from __future__ import annotations

from typing import Optional


class PgmlError(Exception):
    """Base class for all pgml-raised errors. ``http_status`` is a REST hint."""

    http_status: int = 500


#: Deprecated alias — in a codebase where ``pgm`` names the power-grid-model
#: reference library (:mod:`pgml.convert.pgm`), ``PgmError`` read as "a
#: power-grid-model error". Catchers of the old name keep working.
PgmError = PgmlError


# --------------------------------------------------------------------------- #
# Input errors (the caller's fault) -> 422
# --------------------------------------------------------------------------- #
class InputError(PgmlError, ValueError):
    """Invalid input: configuration, conversion, or an unsupported modeling choice.

    Also subclasses the builtin :class:`ValueError` so that migrating a runtime
    ``raise ValueError(...)`` to ``raise InputError(...)`` stays backward-compatible
    with callers (and tests) that catch ``ValueError``, while being catchable as
    :class:`PgmlError` and carrying ``http_status = 422``. (Schema validation is
    separate — it raises pydantic ``ValidationError``, not this.)
    """

    http_status: int = 422


class ConfigurationError(InputError):
    """An invalid configuration value, or a malformed ``PGML_DEFAULTS`` override."""


class ConversionError(InputError):
    """A reference-library network could not be converted to a pgml :class:`Grid`."""


class ConnectivityError(InputError):
    """Part of the grid has no galvanic path to any in-service :class:`Source`.

    A (node, phase) row without a path to a source has no defined voltage: the
    nodal system is singular there (or, with constant-power loads, the solver
    cannot converge). Raised by the pre-solve connectivity check
    (:func:`pgml.topology.connectivity_report`) BEFORE any factorization, so the
    failure names the disconnected nodes and how to fix them instead of surfacing
    as a numerical error.

    Typical causes and fixes (the message lists the concrete elements):

    - an OPEN :class:`Switch` on the only path — close it (``closed=True``) or
      accept the disconnection;
    - a :class:`Line` / :class:`Transformer` with ``in_service=False`` on the only
      path — set it in service;
    - no in-service :class:`Source` at all — add one (the slack);
    - deliberately disconnected nodes — remove them from the grid, or pass
      ``on_disconnected="zero"`` to the solver to solve the energized part and
      report 0 V on the disconnected rows.

    Attributes
    ----------
    unenergized_nodes:
        Ids of the nodes with at least one unenergized phase row.
    islands:
        The disconnected components, each a tuple of node ids.
    reconnectable:
        :class:`pgml.topology.ReconnectHint` entries (branch id, kind, terminal
        nodes, reason) for currently open / out-of-service branches that would
        reconnect an island.
    """

    def __init__(
        self,
        message: str,
        *,
        unenergized_nodes: tuple = (),
        islands: tuple = (),
        reconnectable: tuple = (),
    ) -> None:
        super().__init__(message)
        self.unenergized_nodes = unenergized_nodes
        self.islands = islands
        self.reconnectable = reconnectable


class ModelingError(InputError, NotImplementedError):
    """A requested model is not supported (e.g. an unmodelled transformer vector
    group / clock, an open/2-phase delta connection, or a zigzag winding).

    Also subclasses the builtin :class:`NotImplementedError` so that ``raise
    ModelingError(...)`` is backward-compatible with code (and tests) that catch
    ``NotImplementedError`` for an unsupported model, while still being catchable as
    :class:`InputError` / :class:`PgmlError` and carrying ``http_status = 422``.
    """


# --------------------------------------------------------------------------- #
# Computation errors (the inputs were valid, the computation failed) -> 500
# --------------------------------------------------------------------------- #
class ComputationError(PgmlError):
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
    "PgmlError",
    "PgmError",
    "InputError",
    "ConfigurationError",
    "ConnectivityError",
    "ConversionError",
    "ModelingError",
    "ComputationError",
    "ConvergenceError",
]
