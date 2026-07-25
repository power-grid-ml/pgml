"""IEC 61000-3-2 harmonic current emission limits: per-order device current fractions.

Loader and conversion helpers for the IEC 61000-3-2 appliance harmonic-current
emission limits (equipment drawing <= 16 A per phase). These are what a device is
permitted to INJECT into the supply -- the physically correct reference for a device
current fingerprint -- as opposed to the DIN EN 50160 voltage-compatibility levels
(:mod:`pgml.scenarios.en50160`), which bound the supply VOLTAGE distortion and are not
an appliance-emission model.

The standard defines four equipment classes with different unit conventions:

- Class A (balanced three-phase equipment and the general catch-all): absolute
  per-phase current limits in amperes.
- Class B (portable tools, arc-welding equipment): Class A limits scaled by 1.5.
- Class C (lighting equipment): percentage of the device fundamental input current;
  the 3rd-harmonic limit is ``30 * lambda`` percent (``lambda`` = circuit power factor).
- Class D (75-600 W equipment with a special wave shape, e.g. PCs/monitors/TVs):
  relative limit in milliamps per watt of active input power.

:func:`iec61000_3_2_fraction` converts a class/order limit into a FRACTION of the
device fundamental current, matching the ``harmonic_injection`` magnitude convention
(``magnitude_pu`` = harmonic current / fundamental current). The table is packaged data
(``pgml/data/standards/iec61000_3_2.yaml``), read via :mod:`importlib.resources` so it
resolves identically from a checkout and from an installed wheel; an explicit ``path``
or the ``PGML_IEC61000_3_2`` environment variable overrides it (e.g. a revised table).
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Optional

import yaml

from pgml.assembly._params import phase_voltage_magnitude
from pgml.errors import InputError
from pgml.schemas.grid_schema import Grid

#: Environment variable pointing at a replacement limits file (optional override).
_ENV = "PGML_IEC61000_3_2"
#: The packaged limits file, relative to the ``pgml`` package root.
_PACKAGE_DATA = "data/standards/iec61000_3_2.yaml"
#: Cache key standing in for "the file packaged with pgml" (vs. a filesystem override).
_PACKAGED = "<packaged>"

#: Valid concrete emission classes (``"auto"`` resolves per device before lookup).
_CLASSES = ("A", "B", "C", "D")

#: Active-power ceiling of the IEC 61000-3-2 Class D window; above it a device that
#: would otherwise be Class D falls back to the general Class A.
_CLASS_D_MAX_W = 600.0

# Auto-resolution of a device ``consumer_type`` to an IEC 61000-3-2 class. The standard
# classifies individual equipment (<= 16 A/phase); assigning a class to an aggregate LV
# load from its coarse ``consumer_type`` is a modeling approximation:
#   * lighting-dominated loads      -> Class C
#   * IT / consumer-electronics     -> Class D when nominal P <= 600 W, else Class A
#   * every other character (household, EV charging, PV/inverter, motor drives, ...)
#     and an unspecified type       -> Class A (the general/balanced catch-all)
# The closed :class:`~pgml.schemas.grid_schema.ConsumerType` taxonomy has no dedicated
# lighting member yet, so no ``consumer_type`` currently auto-resolves to Class C;
# select ``emission_class="C"`` explicitly for a lighting-dominated load.
_LIGHTING_TYPES: frozenset[str] = frozenset()
_ELECTRONICS_TYPES: frozenset[str] = frozenset({"office"})


def _source(path: Optional[str]) -> str:
    """Active limits source: explicit ``path`` > ``PGML_IEC61000_3_2`` > packaged file."""
    if path is not None:
        return str(path)
    env = os.environ.get(_ENV)
    return str(env) if env else _PACKAGED


@lru_cache(maxsize=None)
def _load(source: str) -> dict:
    """Parse and normalise the packaged (or overridden) IEC 61000-3-2 table (cached).

    Returns ``{"A"/"B"/"C"/"D": {"unit": str, "limits": {order: value},
    "power_factor_scaled_orders": (int, ...)}}`` with Class B expanded to explicit
    limits (1.5x Class A) so every class exposes a uniform per-order ``limits`` map.
    """
    if source == _PACKAGED:
        text = (files("pgml") / _PACKAGE_DATA).read_text(encoding="utf-8")
        origin = _PACKAGE_DATA
    else:
        text = Path(source).read_text(encoding="utf-8")
        origin = source
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise InputError(f"{origin!r} must be a mapping of class_a/b/c/d entries.")

    def _limits(entry: object, name: str) -> dict:
        if not isinstance(entry, dict) or not isinstance(entry.get("limits"), dict):
            raise InputError(f"{origin!r} {name} must contain a 'limits' mapping.")
        return {int(k): float(v) for k, v in entry["limits"].items()}

    a = _limits(data.get("class_a"), "class_a")
    c_entry = data.get("class_c") or {}
    d = _limits(data.get("class_d"), "class_d")
    b_factor = float((data.get("class_b") or {}).get("multiplier_of_class_a", 1.5))
    pf_orders = tuple(int(o) for o in (c_entry.get("power_factor_scaled_orders") or ()))
    return {
        "A": {"unit": "A", "limits": a},
        "B": {"unit": "A", "limits": {o: b_factor * v for o, v in a.items()}},
        "C": {
            "unit": "percent",
            "limits": _limits(c_entry, "class_c"),
            "power_factor_scaled_orders": pf_orders,
        },
        "D": {"unit": "mA_per_W", "limits": d},
    }


def _normalise_class(emission_class: str) -> str:
    """Validate + upper-case a concrete class letter (rejects ``"auto"``/unknown)."""
    cls = str(emission_class).upper()
    if cls not in _CLASSES:
        raise InputError(
            f"emission_class must be one of {_CLASSES} (got {emission_class!r}); "
            "resolve 'auto' to a concrete class before calling."
        )
    return cls


def iec61000_3_2_limits(
    emission_class: Optional[str] = None, path: Optional[str] = None
) -> dict:
    """Return the IEC 61000-3-2 emission-limit table (cached copy).

    Parameters
    ----------
    emission_class : str, optional
        A concrete class ``"A"``/``"B"``/``"C"``/``"D"`` (case-insensitive) to return
        just that class's entry (``{"unit": ..., "limits": {order: value}, ...}``), or
        ``None`` (default) to return the full ``{class: entry}`` mapping. Class B is
        expanded to explicit limits (1.5x Class A).
    path : str, optional
        Explicit path to the limits YAML. When ``None`` the file is resolved in
        priority order: the ``PGML_IEC61000_3_2`` environment variable, else the table
        packaged with pgml (``pgml/data/standards/iec61000_3_2.yaml``).

    Returns
    -------
    dict
        A copy (callers may mutate it without poisoning the cache).

    Examples
    --------
    ::

        from pgml.scenarios import iec61000_3_2_limits
        iec61000_3_2_limits("A")["limits"][3]   # 2.30  (amperes)
        iec61000_3_2_limits("D")["limits"][3]   # 3.4   (mA/W)
    """
    table = _load(_source(path))
    if emission_class is None:
        return {cls: _copy_entry(e) for cls, e in table.items()}
    return _copy_entry(table[_normalise_class(emission_class)])


def _copy_entry(entry: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in entry.items()}
    return out


def iec61000_3_2_fraction(
    order: int,
    *,
    emission_class: str,
    p_w: float,
    u_ln_v: float,
    power_factor: float = 1.0,
    path: Optional[str] = None,
) -> float:
    """IEC 61000-3-2 emission limit for one order as a fraction of the fundamental current.

    Converts the class/order limit into a per-unit magnitude relative to the device
    fundamental current ``I1 = p_w / (u_ln_v * power_factor)`` (per phase, treating the
    device as a single-phase equivalent), matching the ``harmonic_injection`` magnitude
    convention (``magnitude_pu`` = harmonic current / fundamental current):

    - Class A / B: ``limit_amperes / I1``.
    - Class C: the tabulated percentage / 100 (independent of ``I1``); the 3rd harmonic
      is scaled by ``power_factor`` as ``lambda`` (``30 * lambda`` percent).
    - Class D: ``limit_mA_per_W * p_w / 1000 / I1`` (the ``p_w`` cancels, so the fraction
      is ``limit_mA_per_W * u_ln_v * power_factor / 1000`` -- constant in power, as the
      relative limit intends).

    The returned fraction is clamped to ``<= 1.0`` (an emission above the fundamental is
    nonphysical for these device classes). Orders absent from the class table return
    ``0.0`` (e.g. even orders have no Class C/D limit).

    Parameters
    ----------
    order : int
        Harmonic order (>= 2).
    emission_class : str
        A concrete class ``"A"``/``"B"``/``"C"``/``"D"`` (case-insensitive); resolve
        ``"auto"`` via :func:`resolve_emission_class` first.
    p_w : float
        Device active power [W]. For a multi-phase device pass the PER-PHASE power
        (total / phase count) so ``I1`` is the per-phase fundamental current.
    u_ln_v : float
        Line-to-neutral voltage magnitude [V] at the device terminal.
    power_factor : float, optional
        Displacement/circuit power factor ``lambda`` (default 1.0). Enters ``I1`` and
        the Class C 3rd-harmonic scaling.
    path : str, optional
        Optional explicit path to the limits YAML; see :func:`iec61000_3_2_limits`.

    Returns
    -------
    float
        Emission-limit magnitude relative to the fundamental current [pu], in ``[0, 1]``.
    """
    entry = _load(_source(path))[_normalise_class(emission_class)]
    limits = entry["limits"]
    if order not in limits:
        return 0.0
    unit = entry["unit"]
    if unit == "percent":
        pct = limits[order]
        if order in entry.get("power_factor_scaled_orders", ()):
            pct = pct * power_factor  # h3: 30 * lambda %
        return min(pct / 100.0, 1.0)
    i1 = p_w / (u_ln_v * power_factor) if u_ln_v * power_factor != 0.0 else 0.0
    if i1 <= 0.0:
        # Zero / nonphysical fundamental current: the emission is unbounded relative to
        # it, so the fraction is fully clamped.
        return 1.0
    if unit == "mA_per_W":
        frac = limits[order] * p_w / 1000.0 / i1
    else:  # absolute amperes (Class A / B)
        frac = limits[order] / i1
    return min(frac, 1.0)


def resolve_emission_class(consumer_type: object, p_w: float) -> str:
    """Resolve ``emission_class="auto"`` to a concrete IEC 61000-3-2 class for a device.

    See the module-level mapping (``consumer_type`` character -> class). Lighting-like
    loads map to Class C, IT/consumer-electronics loads to Class D when nominal
    ``p_w <= 600 W`` else Class A, and everything else (including an unspecified
    ``consumer_type``) to the general Class A.

    Parameters
    ----------
    consumer_type : ConsumerType | str | None
        The device's :class:`~pgml.schemas.grid_schema.ConsumerType` (or its string
        value, or ``None`` when unspecified).
    p_w : float
        Device nominal active power [W] (total; only the Class D 600 W window uses it).

    Returns
    -------
    str
        A concrete class letter ``"A"``/``"B"``/``"C"``/``"D"``.
    """
    ct = getattr(consumer_type, "value", consumer_type)
    if ct in _LIGHTING_TYPES:
        return "C"
    if ct in _ELECTRONICS_TYPES:
        return "D" if float(p_w) <= _CLASS_D_MAX_W else "A"
    return "A"


def iec61000_3_2_device_caps(
    grid: Grid,
    ids: list[int],
    orders: list[int],
    *,
    emission_class: str = "auto",
    power_factor: float = 1.0,
    path: Optional[str] = None,
) -> dict[int, dict[int, float]]:
    """Per-device, per-order IEC 61000-3-2 emission fractions for a grid.

    Builds ``{device_id: {order: fraction}}`` for the given appliance ids, resolving
    each device's fundamental current from its nominal active power and its node's
    line-to-neutral voltage (via :func:`pgml.assembly._params.phase_voltage_magnitude`,
    the same convention the solver uses). When ``emission_class="auto"`` the class is
    resolved per device from its ``consumer_type`` (see :func:`resolve_emission_class`);
    otherwise the given concrete class applies to every device.

    The per-phase current uses the device's total nominal power divided by its phase
    count (single-phase equivalent), so Class A/B absolute (per-phase amperes) limits are
    referenced correctly. The caps are plain floats (off the autograd tape) -- a sampling
    bound, not a differentiable quantity.

    Parameters
    ----------
    grid : Grid
        The reference grid (for node voltages and device nameplates).
    ids : list[int]
        Appliance ids to build caps for (must be Load/Generator/Storage on the grid).
    orders : list[int]
        Harmonic orders to evaluate.
    emission_class : str, optional
        ``"auto"`` (default, per-device) or a concrete ``"A"``/``"B"``/``"C"``/``"D"``.
    power_factor : float, optional
        Circuit power factor used for the fundamental current and Class C h3 scaling
        (default 1.0 -- the caps derive from nominal P and voltage only).
    path : str, optional
        Optional explicit path to the limits YAML; see :func:`iec61000_3_2_limits`.

    Returns
    -------
    dict
        ``{device_id: {order: fraction}}`` with every fraction in ``[0, 1]``.
    """
    nodes_by_id = {n.id: n for n in grid.nodes}
    appliances_by_id = {a.id: a for a in grid.appliances}
    caps: dict[int, dict[int, float]] = {}
    for cid in ids:
        dev = appliances_by_id[cid]
        node = nodes_by_id[dev.node]
        n_phase = len(dev.phases)
        # Total nominal active power as a plain float (off-tape sampling bound).
        p_total = float(dev.p_nom_w)
        p_phase = p_total / n_phase if n_phase else p_total
        u_ln = phase_voltage_magnitude(float(node.u_rated_v), len(node.phases))
        cls = (
            resolve_emission_class(getattr(dev, "consumer_type", None), p_total)
            if str(emission_class).lower() == "auto"
            else emission_class
        )
        caps[cid] = {
            order: iec61000_3_2_fraction(
                order,
                emission_class=cls,
                p_w=p_phase,
                u_ln_v=u_ln,
                power_factor=power_factor,
                path=path,
            )
            for order in orders
        }
    return caps


__all__ = [
    "iec61000_3_2_limits",
    "iec61000_3_2_fraction",
    "resolve_emission_class",
    "iec61000_3_2_device_caps",
]
