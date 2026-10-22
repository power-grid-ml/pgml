"""IEC 61000-2-2 / IEC 61000-2-12 compatibility levels for voltage distortion.

A loader for the compatibility levels of public supply networks: the per-order harmonic
voltage levels of low-voltage (IEC 61000-2-2) and medium-voltage (IEC 61000-2-12)
networks, and the low-voltage levels for symmetric voltage distortion from 2 kHz to
150 kHz (IEC 61000-2-2, clauses 4.11 and 4.12). Compatibility levels are reference levels
for EMC coordination, against which a computed voltage spectrum can be held; they are
not emission limits of a device.

The table ships inside the package at ``pgml/data/standards/iec61000_2_2.yaml``, whose
header states the clauses and what is not included (the interharmonic levels, given only
as a figure, and the very-short-term factor). An explicit ``path`` or the
``PGML_IEC61000_2_2`` environment variable replaces the packaged file.

Units: harmonic levels are fractions of the fundamental (rms). The 2-9 kHz levels are
fractions of the fundamental for a 200 Hz band; the 9-150 kHz levels are dB(uV) measured
with a quasi-peak detector in a 200 Hz bandwidth, converted here to volts
(``1e-6 * 10**(dB/20)``). For a steady line spectrum, as the frequency-domain solve
produces, the rms in a 200 Hz band is the comparable quantity (a detector calibrated in
rms reads a continuous sine wave at its rms value).
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Optional

import yaml

from pgml.errors import InputError

_ENV = "PGML_IEC61000_2_2"
_PACKAGE_DATA = "data/standards/iec61000_2_2.yaml"
_PACKAGED = "<packaged>"


def _source(path: Optional[str]) -> str:
    if path is not None:
        return str(path)
    env = os.environ.get(_ENV)
    return str(env) if env else _PACKAGED


def _read(source: str) -> str:
    if source == _PACKAGED:
        return (files("pgml") / _PACKAGE_DATA).read_text(encoding="utf-8")
    return Path(source).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _load(source: str) -> dict:
    data = yaml.safe_load(_read(source))
    if not isinstance(data, dict) or "harmonics" not in data:
        raise InputError(f"{source!r} must contain a 'harmonics' mapping.")
    return data


def _kind(h: int) -> str:
    if h % 2 == 0:
        return "even"
    return "odd_triplen" if h % 3 == 0 else "odd_non_triplen"


def iec61000_2_2_provenance(path: Optional[str] = None) -> dict:
    """``{"source", "override", "sha256"}`` of the table a run used."""
    import hashlib

    source = _source(path)
    return {
        "source": source,
        "override": source != _PACKAGED,
        "sha256": hashlib.sha256(_read(source).encode("utf-8")).hexdigest(),
    }


def iec61000_2_2_harmonic_levels(
    network: str = "lv", path: Optional[str] = None
) -> dict:
    """Long-term compatibility levels ``{order: fraction of the fundamental}``.

    ``network="lv"`` (IEC 61000-2-2, orders 2 to 40) or ``"mv"`` (IEC 61000-2-12,
    orders 2 to 50). Range rules ``a (b / h) + c`` of the tables are evaluated per order.
    """
    table = _load(_source(path))["harmonics"].get(network)
    if table is None:
        raise InputError(f"no harmonic compatibility levels for network {network!r}.")
    levels = {int(h): float(v) for h, v in table["explicit"].items()}
    for rule in table["ranges"]:
        for h in range(int(rule["h_min"]), int(rule["h_max"]) + 1):
            if _kind(h) == rule["kind"] and h not in levels:
                levels[h] = float(rule["a"]) * (float(rule["b"]) / h) + float(rule["c"])
    return {h: levels[h] for h in sorted(levels) if h <= int(table["max_order"])}


def iec61000_2_2_thd(
    network: str = "lv", *, very_short_term: bool = False, path: Optional[str] = None
) -> float:
    """Compatibility level of the total harmonic distortion (fraction)."""
    table = _load(_source(path))["harmonics"][network]
    return float(table["thd_very_short_term" if very_short_term else "thd_long_term"])


def iec61000_2_2_band_level_v(
    frequency_hz: float, u1_v: float, *, f1_hz: float = 50.0, path: Optional[str] = None
) -> Optional[float]:
    """Low-voltage compatibility level of a 200 Hz band centred at ``frequency_hz`` [V].

    Returns ``None`` outside 2 kHz to 150 kHz (2.4 kHz for 60 Hz networks), where the
    standard gives no band level. ``u1_v`` is the rms fundamental of the voltage between
    the conductors considered (phase to neutral for 9 kHz and above). Segments are
    half-open, ``[f_lo, f_hi)``, except the last, so at 50 kHz the lower level applies
    as the standard states.
    """
    segs = _load(_source(path))["supraharmonics"]["lv"]["segments"]
    f = float(frequency_hz)
    for k, seg in enumerate(segs):
        lo = float(
            seg.get("f_lo_hz_60", seg["f_lo_hz"]) if f1_hz == 60.0 else seg["f_lo_hz"]
        )
        hi = float(seg["f_hi_hz"])
        last = k == len(segs) - 1
        if not (lo <= f < hi or (last and f == hi)):
            continue
        a, b = float(seg["level_lo"]), float(seg["level_hi"])
        mode = seg["interpolation"]
        x = math.log(f / lo) / math.log(hi / lo) if hi > lo else 0.0
        if mode == "constant":
            level = a
        elif mode == "loglog":
            level = math.exp(math.log(a) + x * (math.log(b) - math.log(a)))
        elif mode == "dB_logf":
            level = a + x * (b - a)
        else:
            raise InputError(
                f"unknown interpolation {mode!r} in the compatibility table."
            )
        if seg["unit"] == "fraction_of_u1":
            return level * float(u1_v)
        if seg["unit"] == "dBuV":
            return 1e-6 * 10.0 ** (level / 20.0)
        raise InputError(f"unknown unit {seg['unit']!r} in the compatibility table.")
    return None


__all__ = [
    "iec61000_2_2_band_level_v",
    "iec61000_2_2_harmonic_levels",
    "iec61000_2_2_provenance",
    "iec61000_2_2_thd",
]
