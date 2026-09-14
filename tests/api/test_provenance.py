"""pgml.provenance: the code-identity stamp every persisted artifact records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pgml
from pgml.provenance import PROVENANCE_FILENAME, code_provenance, git_state, _repo_root


def test_code_provenance_shape():
    prov = code_provenance()
    assert set(prov) == {
        "git_sha",
        "git_dirty",
        "git_source",
        "pgml_version",
        "torch_version",
    }
    assert prov["pgml_version"] == pgml.__version__
    assert isinstance(prov["git_sha"], str) and prov["git_sha"]
    # JSON-ready: no tensors, no paths, nothing exotic.
    json.dumps(prov)


def _is_checkout() -> bool:
    """Is the package under test inside a live git repository?"""
    root = _repo_root()
    return root is not None and Path(root / ".git").exists()


@pytest.mark.skipif(
    not _is_checkout(),
    reason="the package is a mirrored tree without .git; the sync-stamp path covers it",
)
def test_git_state_in_a_checkout_resolves_a_real_sha():
    # From a source checkout the live-git path must resolve; a mirrored tree (a cluster
    # sync, a wheel) reports from the sync stamp instead — test_sync_stamp_fallback.
    state = git_state()
    assert state["git_source"] == "git"
    assert len(state["git_sha"]) == 40
    assert state["git_dirty"] in (True, False)


def test_sync_stamp_fallback(tmp_path, monkeypatch):
    # A mirrored tree (no .git) with the sync stamp at its root must report from it.
    stamp = {"git_sha": "a" * 40, "git_dirty": False}
    (tmp_path / PROVENANCE_FILENAME).write_text(json.dumps(stamp))
    pkg = tmp_path / "src" / "pgml"
    pkg.mkdir(parents=True)

    import pgml.provenance as prov_mod

    monkeypatch.setattr(prov_mod, "__file__", str(pkg / "provenance.py"))
    state = prov_mod.git_state()
    assert state == {"git_sha": "a" * 40, "git_dirty": False, "git_source": "file"}
