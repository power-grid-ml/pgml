"""pgml — differentiable, GPU-ready harmonic power-flow + ML for power grids."""

from __future__ import annotations

import logging as _logging

#: Library version (single source of truth; the build reads this).
__version__ = "0.1.0"

__all__ = ["__version__"]

# Library logging convention: emit on the ``pgml`` logger and attach a NullHandler so
# importing pgml never prints "No handlers could be found". Applications opt in by
# configuring logging (e.g. ``logging.basicConfig(level=logging.INFO)``) to see the
# INFO modeling summary (calculation symmetry, neutral modeling, load connections).
_logging.getLogger("pgml").addHandler(_logging.NullHandler())
