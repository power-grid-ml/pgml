"""Grid-planning studies built on the harmonic power flow.

Analysis studies that answer planning questions about a grid rather than solve one
operating point: the frequency-dependent impedance seen from a bus, the voltage-stability
margin along a load or generation increase, and how much new generation or demand each bus
can take before a limit is reached. Each study takes a :class:`~pgml.Grid` and returns a
result object with a JSON-ready ``to_dict()``.
"""

from __future__ import annotations

from .continuation import PVCurve, QVCurve, pv_curve, qv_curve
from .impedance import ImpedanceScan, Resonance, impedance_scan

__all__: list[str] = [
    "ImpedanceScan",
    "PVCurve",
    "QVCurve",
    "Resonance",
    "impedance_scan",
    "pv_curve",
    "qv_curve",
]
