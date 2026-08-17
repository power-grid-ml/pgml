"""Code provenance for run artifacts.

Every persisted artifact (dataset ``meta.json``, multi-grid corpus manifest, training
checkpoint, cluster run directory) should record WHICH code produced it: the git commit,
whether the working tree was dirty, and the library/torch versions. Without that stamp,
two artifacts written from byte-identical configs and seeds can differ numerically —
because the code between them changed — with nothing in their metadata to show it.

The cluster mirror is rsynced WITHOUT ``.git``, so ``git rev-parse`` is not available
there. The sync step instead drops a :data:`PROVENANCE_FILENAME` file (the locally
resolved commit + dirty flag) at the repository root of the mirror; this module falls
back to reading it. A missing git AND missing file yields ``"unknown"`` rather than an
error — provenance must never abort a run, only describe it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

__all__ = ["PROVENANCE_FILENAME", "code_provenance", "git_state"]

#: File the cluster sync writes at the mirrored repository root (the mirror has no
#: ``.git``): ``{"git_sha": ..., "git_dirty": ...}`` resolved on the submitting machine.
PROVENANCE_FILENAME = ".git-provenance.json"

# How far above this file the repository root may sit. In a source checkout (editable
# install) provenance.py lives at <root>/src/pgml/provenance.py — two levels up.
_MAX_ASCENT = 6


def _repo_root() -> Optional[Path]:
    """The enclosing repository root: nearest ancestor with ``.git`` or the sync stamp."""
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents][:_MAX_ASCENT]:
        if (candidate / ".git").exists() or (candidate / PROVENANCE_FILENAME).is_file():
            return candidate
    return None


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state() -> dict[str, Any]:
    """The commit identity of the code, resolved from git or the sync stamp.

    Returns ``{"git_sha": str, "git_dirty": bool | None, "git_source": str}`` where
    ``git_source`` is ``"git"`` (live repository), ``"file"`` (the cluster sync stamp)
    or ``"none"`` (neither found; ``git_sha`` is ``"unknown"``, ``git_dirty`` ``None``).
    """
    root = _repo_root()
    if root is not None and (root / ".git").exists():
        sha = _git(root, "rev-parse", "HEAD")
        if sha:
            status = _git(root, "status", "--porcelain")
            return {
                "git_sha": sha,
                "git_dirty": bool(status) if status is not None else None,
                "git_source": "git",
            }
    if root is not None:
        stamp = root / PROVENANCE_FILENAME
        if stamp.is_file():
            try:
                data = json.loads(stamp.read_text())
            except (OSError, json.JSONDecodeError):
                data = {}
            if isinstance(data, dict) and data.get("git_sha"):
                return {
                    "git_sha": str(data["git_sha"]),
                    "git_dirty": data.get("git_dirty"),
                    "git_source": "file",
                }
    return {"git_sha": "unknown", "git_dirty": None, "git_source": "none"}


def code_provenance() -> dict[str, Any]:
    """Everything an artifact should record about the code that produced it.

    A plain, JSON-ready dict: the :func:`git_state` fields plus ``pgml_version`` and
    ``torch_version``. Cheap enough to call once per artifact write; never raises.
    """
    import torch

    import pgml

    return {
        **git_state(),
        "pgml_version": pgml.__version__,
        "torch_version": torch.__version__,
    }
