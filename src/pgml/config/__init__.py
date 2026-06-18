"""pgml.config — explicit, documented modeling defaults (no hidden implicit values).

All default constants and default model selections live in ``defaults.yaml`` (ordered by
component, each with a value, units and a short description). Resolution precedence is
explicit (user) > config (this package) > converter (source library). See
:mod:`pgml.config.config` for the API and ``defaults.yaml`` for the data.
"""

from __future__ import annotations

from .config import defaults, describe, get, reload, resolve, units

__all__ = ["defaults", "get", "describe", "units", "resolve", "reload"]
