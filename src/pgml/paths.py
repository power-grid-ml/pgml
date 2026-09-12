"""Filesystem conventions for experiment inputs and outputs.

Packages in the ecosystem this engine is built for never write experiment data into the
library tree. A run's datasets, checkpoints, run configs and tracking go under the
EXPERIMENTS ROOT — a single directory the user owns and keeps out of version control.
This keeps reproducible-but-bulky experiment artifacts separate from the source, and
lets a run be relocated (e.g. to fast cluster scratch) by pointing one environment
variable elsewhere.

The root is the ``PGML_EXPERIMENTS`` environment variable when set, else ``./data``
relative to the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Environment variable selecting the experiments root (absolute or relative path).
_ENV = "PGML_EXPERIMENTS"
#: Default root when ``PGML_EXPERIMENTS`` is unset (relative to the working directory).
_DEFAULT = "data"


def experiments_root() -> Path:
    """Return the experiments root: ``PGML_EXPERIMENTS`` if set, else ``./data``.

    The directory is NOT created here — callers that write into it create only what they
    need. Use it as the base for a run's outputs, e.g. ``experiments_root() / run_name``.
    """
    return Path(os.environ.get(_ENV, _DEFAULT))


__all__ = ["experiments_root"]
