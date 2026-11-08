"""Grid-planning studies built on the harmonic power flow.

Analysis studies that answer planning questions about a grid rather than solve one
operating point: the frequency-dependent impedance seen from a bus, the voltage-stability
margin along a load or generation increase, how much new generation or demand each bus
can take before a limit is reached, and which upgrade relieves the most limit stress.
Each study takes a :class:`~pgml.Grid` and returns a result object with a JSON-ready
``to_dict()``.
"""

from __future__ import annotations

from .continuation import PVCurve, QVCurve, pv_curve, qv_curve
from .enhancement import EnhancementSensitivities, enhancement_sensitivities
from .hosting import HostingCapacity, hosting_capacity
from .impedance import ImpedanceScan, Resonance, impedance_scan
from .limits import BranchRating, HostingLimits, branch_ratings

__all__: list[str] = [
    "BranchRating",
    "EnhancementSensitivities",
    "HostingCapacity",
    "HostingLimits",
    "ImpedanceScan",
    "PVCurve",
    "QVCurve",
    "Resonance",
    "branch_ratings",
    "enhancement_sensitivities",
    "hosting_capacity",
    "impedance_scan",
    "pv_curve",
    "qv_curve",
]
