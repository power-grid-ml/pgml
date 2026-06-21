"""Load and resolve pgml's modeling defaults (``defaults.yaml``).

Single source of truth for default VALUES and default MODEL choices, so every modeling
decision is a deliberate, documented choice instead of a hidden implicit default. See
``defaults.yaml`` for the data + the resolution-precedence contract.

Public API
----------
- ``defaults()``                  -> the parsed config dict (cached).
- ``get(key, default=_RAISE)``    -> the ``value`` at a dotted ``key`` (e.g.
  ``"line.earth_return.resistivity_ohm_m"``).
- ``describe(key)`` / ``units(key)`` -> the documentation / units string at ``key``.
- ``resolve(key, explicit=None, converted=None)`` -> precedence resolution
  (explicit > config > converter).
- ``reload(path=None)``           -> reload the config (test / user-override hook).

The active config file is the packaged ``defaults.yaml`` unless overridden by the
``PGML_CONFIG`` environment variable or an explicit ``reload(path)``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from pgml.errors import ConfigurationError

_PACKAGED = Path(__file__).with_name("defaults.yaml")
_RAISE = object()  # sentinel: get() with no default raises on a missing key


def _config_path() -> Path:
    """Active config path: ``PGML_CONFIG`` env override, else the packaged defaults."""
    env = os.environ.get("PGML_CONFIG")
    return Path(env) if env else _PACKAGED


@lru_cache(maxsize=None)
def _load(path_str: str) -> dict:
    with open(path_str, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"pgml config {path_str!r} must be a mapping at the top level."
        )
    return data


def defaults() -> dict:
    """Return the parsed config dict (cached for the active config path)."""
    return _load(str(_config_path()))


def reload(path: Optional[str] = None) -> dict:
    """Clear the cache and reload the config (optionally from ``path``).

    A hook for tests and for users that ship a project-level override. With ``path`` the
    ``PGML_CONFIG`` env var is set so subsequent calls resolve against it.
    """
    if path is not None:
        os.environ["PGML_CONFIG"] = str(path)
    _load.cache_clear()
    return defaults()


def _node(key: str) -> dict:
    """Return the ``{value, units, description}`` leaf mapping at a dotted ``key``."""
    node: Any = defaults()
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"pgml config has no key {key!r} (missing {part!r}).")
        node = node[part]
    if not isinstance(node, dict) or "value" not in node:
        raise KeyError(
            f"pgml config key {key!r} is not a leaf with a 'value' (got {type(node).__name__})."
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
    """Resolve a parameter by precedence: ``explicit`` > config(``key``) > ``converted``.

    ``explicit`` is a value the user set on the component/grid (wins when not ``None``).
    Falls back to the config default, then to a converter-inferred ``converted`` value
    (used only if the key is absent from the config). Raises ``KeyError`` if nothing
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
