"""DIN EN 50160 compatibility levels: per-order harmonic voltage limits.

A small loader for the per-order maximum harmonic magnitudes (relative to the
fundamental) defined by DIN EN 50160. These serve as ready-made UPPER BOUNDS for
harmonic-spectrum sampling: a :class:`~pgml.scenarios.config.ParameterSpec` with
``field="h_mag"`` and ``harmonic_reference="en50160"`` samples a fraction (its
``distribution``, in ``[0, 1]``) of the per-order limit returned here.

The table is data, not code: it lives in ``config/max_harmonic_values_din-en50160.yaml``
(repo root). The active file is resolved as: an explicit ``path`` argument, else the
``PGML_EN50160`` environment variable, else the first ``config/<filename>`` found by
walking up from this package. The values are a fixed standard, so a dataset stays
reproducible as long as the file is unchanged.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from pgml.errors import InputError

_ENV = "PGML_EN50160"
_FILENAME = "max_harmonic_values_din-en50160.yaml"


def _resolve_path() -> Path:
    """Active limits file: ``PGML_EN50160`` env, else a walk-up ``config/<file>``."""
    env = os.environ.get(_ENV)
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "config" / _FILENAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate {_FILENAME!r}; set the {_ENV} environment variable to its "
        "path or pass an explicit path to en50160_limits()."
    )


@lru_cache(maxsize=None)
def _load(path_str: str) -> dict:
    data = yaml.safe_load(Path(path_str).read_text(encoding="utf-8"))
    table = data.get("max_harmonic_values") if isinstance(data, dict) else None
    if not isinstance(table, dict):
        raise InputError(
            f"{path_str!r} must contain a 'max_harmonic_values' mapping of order -> limit."
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
        in priority order: ``PGML_EN50160`` environment variable, then the first
        ``config/max_harmonic_values_din-en50160.yaml`` found by walking up from
        the package root.

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
    return dict(_load(str(path) if path is not None else str(_resolve_path())))


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
    table = _load(str(path) if path is not None else str(_resolve_path()))
    if order not in table:
        raise KeyError(
            f"order {order} has no DIN EN 50160 limit in the table "
            f"(available 1..{max(table)})."
        )
    return table[order]


__all__ = ["en50160_limits", "en50160_limit"]
