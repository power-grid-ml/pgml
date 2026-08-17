"""pgml.provenance: the code-identity stamp every persisted artifact records."""

from __future__ import annotations

import json

import pgml
from pgml.provenance import PROVENANCE_FILENAME, code_provenance, git_state


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


def test_git_state_in_a_checkout_resolves_a_real_sha():
    # The test suite runs from a source checkout, so the live-git path must resolve.
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
