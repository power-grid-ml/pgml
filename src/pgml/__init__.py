"""pgml — differentiable, GPU-ready harmonic power-flow + ML for power grids."""

from __future__ import annotations

import logging as _logging

# Library logging convention: emit on the ``pgml`` logger and attach a NullHandler so
# importing pgml never prints "No handlers could be found". Applications opt in by
# configuring logging (e.g. ``logging.basicConfig(level=logging.INFO)``) to see the
# INFO modeling summary (calculation symmetry, neutral modeling, load connections).
_logging.getLogger("pgml").addHandler(_logging.NullHandler())
