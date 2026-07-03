"""Session-wide test configuration.

pandapower 2.x imports the ``numpy.Inf`` / ``numpy.in1d`` aliases that NumPy 2 removed.
Restore them ONCE here — pytest imports the root ``conftest.py`` before collecting any
test module, so the shim is in place before the first ``import pandapower`` anywhere in
the suite. (Previously every pandapower-touching test file repeated this shim.)
"""

from __future__ import annotations

from pgml.convert.pandapower import ensure_numpy_compat

ensure_numpy_compat()
