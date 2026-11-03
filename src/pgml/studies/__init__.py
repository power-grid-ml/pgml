"""Grid-planning studies built on the harmonic power flow.

Analysis studies that answer planning questions about a grid rather than solve one
operating point: the frequency-dependent impedance seen from a bus, the voltage-stability
margin along a load or generation increase, and how much new generation or demand each bus
can take before a limit is reached, and how inverter emission spreads through the grid. Each study takes a :class:`~pgml.Grid` and returns a
result object with a JSON-ready ``to_dict()``.
"""

from __future__ import annotations

from .continuation import PVCurve, QVCurve, pv_curve, qv_curve
from .emission import EmissionFlow, emission_flow, emission_frequencies
from .hosting import HostingCapacity, hosting_capacity
from .impedance import ImpedanceScan, Resonance, impedance_scan
from .limits import BranchRating, HostingLimits, branch_ratings

__all__: list[str] = [
    "BranchRating",
    "EmissionFlow",
    "HostingCapacity",
    "HostingLimits",
    "ImpedanceScan",
    "PVCurve",
    "QVCurve",
    "Resonance",
    "branch_ratings",
    "emission_flow",
    "emission_frequencies",
    "hosting_capacity",
    "impedance_scan",
    "pv_curve",
    "qv_curve",
]
