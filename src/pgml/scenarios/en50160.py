"""DIN EN 50160 compatibility levels: per-order harmonic voltage limits.

A small loader for the per-order maximum harmonic magnitudes (relative to the
fundamental) defined by DIN EN 50160. These serve as ready-made UPPER BOUNDS for
harmonic-spectrum sampling: a :class:`~pgml.scenarios.config.ParameterSpec` with
``field="h_mag"`` and ``harmonic_reference="en50160"`` samples a fraction (its
``distribution``, in ``[0, 1]``) of the per-order limit returned here.

The table is data, not code: it ships INSIDE the package at
``pgml/data/standards/en50160.yaml`` and is read via :mod:`importlib.resources`, so it
resolves identically from a source checkout and from an installed wheel. The active file
is the packaged table unless overridden by an explicit ``path`` argument or the
``PGML_EN50160`` environment variable (e.g. to ship a revised standard). The values are a
fixed standard, so a dataset stays reproducible as long as the file is unchanged.
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Optional

import yaml

from pgml.errors import InputError

#: Environment variable pointing at a replacement limits file (optional override).
_ENV = "PGML_EN50160"
#: The packaged limits file, relative to the ``pgml`` package root.
_PACKAGE_DATA = "data/standards/en50160.yaml"
#: Cache key standing in for "the file packaged with pgml" (vs. a filesystem override).
_PACKAGED = "<packaged>"


def _source(path: Optional[str]) -> str:
    """Active limits source: explicit ``path`` > ``PGML_EN50160`` > the packaged file."""
    if path is not None:
        return str(path)
    env = os.environ.get(_ENV)
    return str(env) if env else _PACKAGED


@lru_cache(maxsize=None)
def _load(source: str) -> dict:
    if source == _PACKAGED:
        text = (files("pgml") / _PACKAGE_DATA).read_text(encoding="utf-8")
        origin = _PACKAGE_DATA
    else:
        text = Path(source).read_text(encoding="utf-8")
        origin = source
    data = yaml.safe_load(text)
    table = data.get("max_harmonic_values") if isinstance(data, dict) else None
    if not isinstance(table, dict):
        raise InputError(
            f"{origin!r} must contain a 'max_harmonic_values' mapping of order -> limit."
        )
    return {int(k): float(v) for k, v in table.items()}


def en50160_limits(path: Optional[str] = None) -> dict:
    """Return ``{order: max_magnitude_pu}`` from the DIN EN 50160 table (cached).

    Loads and caches the per-order maximum harmonic voltage magnitudes (relative to
    the fundamental, in per-unit) defined by DIN EN 50160.  The returned dict covers
    orders 1 through 49 (orders 26+ were extended manually; see the YAML comment).

    Parameters
    ----------
    path : str, optional
        Explicit path to the limits YAML file.  When ``None`` the file is resolved
        in priority order: the ``PGML_EN50160`` environment variable, else the table
        packaged with pgml (``pgml/data/standards/en50160.yaml``).

    Returns
    -------
    dict
        ``{order: max_pu}`` for every tabulated order.  A copy is returned so callers
        may modify it without poisoning the cache.

    Examples
    --------
    ::

        from pgml.scenarios import en50160_limits
        limits = en50160_limits()
        # {1: 1.0, 2: 0.02, 3: 0.05, 5: 0.06, 7: 0.05, ...}
    """
    return dict(_load(_source(path)))


def en50160_limit(order: int, path: Optional[str] = None) -> float:
    """Return the DIN EN 50160 per-order harmonic voltage limit in per-unit.

    Parameters
    ----------
    order : int
        Harmonic order (must be present in the YAML table, typically 1–49).
    path : str, optional
        Optional explicit path to the limits YAML; see :func:`en50160_limits`.

    Returns
    -------
    float
        Maximum harmonic voltage magnitude relative to the fundamental [pu].

    Raises
    ------
    KeyError
        If ``order`` has no entry in the loaded table.

    Examples
    --------
    ::

        from pgml.scenarios import en50160_limit
        cap_h5 = en50160_limit(5)   # 0.06
        cap_h7 = en50160_limit(7)   # 0.05
    """
    table = _load(_source(path))
    if order not in table:
        raise KeyError(
            f"order {order} has no DIN EN 50160 limit in the table "
            f"(available 1..{max(table)})."
        )
    return table[order]


__all__ = ["en50160_limits", "en50160_limit"]
