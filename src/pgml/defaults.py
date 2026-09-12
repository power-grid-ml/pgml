"""Load and resolve pgml's modeling defaults (the packaged ``data/defaults.yaml``).

Single source of truth for default VALUES and default MODEL choices, so every modeling
decision is a deliberate, documented choice instead of a hidden implicit default. The
data file (ordered by component, each leaf carrying a value, units and a short
description) ships *inside* the package under ``pgml/data/defaults.yaml`` and is read via
:mod:`importlib.resources`, so it resolves identically from a source checkout and from an
installed wheel.

These are *internal* modeling defaults, not user run configuration. The user-facing,
serializable run schema for data generation is :mod:`pgml.scenarios.config`; downstream
packages that train models or generate grids on top of this engine follow the same
"config" naming convention for their own run schemas.

Public API
----------
- ``defaults()``                  -> the parsed defaults dict (cached).
- ``get(key, default=_RAISE)``    -> the ``value`` at a dotted ``key`` (e.g.
  ``"line.earth_return.resistivity_ohm_m"``).
- ``describe(key)`` / ``units(key)`` -> the documentation / units string at ``key``.
- ``resolve(key, explicit=None, converted=None)`` -> precedence resolution
  (explicit > defaults > converter).
- ``reload(path=None)``           -> reload the defaults (test / user-override hook).

The active file is the packaged ``data/defaults.yaml`` unless overridden by the
``PGML_DEFAULTS`` environment variable or an explicit ``reload(path)``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Optional

import yaml

from pgml.errors import ConfigurationError

#: The packaged defaults file, relative to the ``pgml`` package root.
_PACKAGE_DATA = "data/defaults.yaml"
#: Environment variable pointing at a replacement defaults file (optional override).
_ENV = "PGML_DEFAULTS"
#: Sentinel: ``get()`` with no ``default`` raises on a missing key.
_RAISE = object()
#: Cache key standing in for "the file packaged with pgml" (vs. a filesystem override).
_PACKAGED = "<packaged>"


def _source() -> str:
    """Active defaults source: ``PGML_DEFAULTS`` override path, else the packaged file."""
    env = os.environ.get(_ENV)
    return str(Path(env)) if env else _PACKAGED


@lru_cache(maxsize=None)
def _load(source: str) -> dict:
    if source == _PACKAGED:
        text = (files("pgml") / _PACKAGE_DATA).read_text(encoding="utf-8")
        origin = _PACKAGE_DATA
    else:
        text = Path(source).read_text(encoding="utf-8")
        origin = source
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"pgml defaults {origin!r} must be a mapping at the top level."
        )
    return data


def defaults() -> dict:
    """Return the parsed defaults dict (cached for the active source)."""
    return _load(_source())


def reload(path: Optional[str] = None) -> dict:
    """Clear the cache and reload the defaults (optionally from ``path``).

    A hook for tests and for users that ship a project-level override. With ``path`` the
    ``PGML_DEFAULTS`` env var is set so subsequent calls resolve against it.
    """
    if path is not None:
        os.environ[_ENV] = str(path)
    _load.cache_clear()
    return defaults()


def _node(key: str) -> dict:
    """Return the ``{value, units, description}`` leaf mapping at a dotted ``key``."""
    node: Any = defaults()
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"pgml defaults have no key {key!r} (missing {part!r}).")
        node = node[part]
    if not isinstance(node, dict) or "value" not in node:
        raise KeyError(
            f"pgml defaults key {key!r} is not a leaf with a 'value' "
            f"(got {type(node).__name__})."
        )
    return node


def get(key: str, default: Any = _RAISE) -> Any:
    """Return the ``value`` at dotted ``key``; ``default`` (if given) on a missing key."""
    try:
        return _node(key)["value"]
    except KeyError:
        if default is _RAISE:
            raise
        return default


def describe(key: str) -> str:
    """Return the ``description`` documented for ``key`` (empty string if absent)."""
    return str(_node(key).get("description", "")).strip()


def units(key: str) -> str:
    """Return the ``units`` string documented for ``key`` (empty string if absent)."""
    return str(_node(key).get("units", "")).strip()


def resolve(key: str, explicit: Any = None, converted: Any = None) -> Any:
    """Resolve a parameter by precedence: ``explicit`` > defaults(``key``) > ``converted``.

    ``explicit`` is a value the user set on the component/grid (wins when not ``None``).
    Falls back to the modeling default, then to a converter-inferred ``converted`` value
    (used only if the key is absent from the defaults). Raises ``KeyError`` if nothing
    resolves.
    """
    if explicit is not None:
        return explicit
    try:
        return get(key)
    except KeyError:
        if converted is not None:
            return converted
        raise


__all__ = [
    "defaults",
    "reload",
    "get",
    "describe",
    "units",
    "resolve",
]
