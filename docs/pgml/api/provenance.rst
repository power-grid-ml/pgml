pgml.provenance
================

Code provenance for persisted artifacts.

A dataset, a training checkpoint, or a cluster run directory can be produced from a
byte-identical config and seed and still differ numerically, because the CODE that ran
between the two changed. :mod:`pgml.provenance` is the single, reusable stamp that closes
that gap: it identifies the exact commit (or working-tree state) and library versions that
produced an artifact, so two artifacts can be told apart by their metadata alone.

Usage::

    from pgml.provenance import code_provenance

    provenance = code_provenance()
    # {"git_sha": "d6185fb...", "git_dirty": False, "git_source": "git",
    #  "pgml_version": "0.2.0", "torch_version": "2.4.0"}

Write this dict alongside anything a config + seed alone would otherwise be trusted to
reproduce.

What consumes it
-----------------

Write it alongside any artifact a reader might later need to tell apart from one written
under different code.

- :func:`~pgml.scenarios.generation_provenance` folds
  :func:`~pgml.provenance.code_provenance` into a dataset's ``meta.json``, alongside the
  active EN 50160 and IEC 61000-3-2 standards tables (see :doc:`scenarios`'s "Generation
  provenance" section).
- A downstream consumer stamps the same dict into its own manifests, model checkpoints and
  run directories, so each of those records which code produced it.

Resolving the commit without ``.git``
--------------------------------------

A live source checkout resolves the commit with ``git rev-parse HEAD`` and
``git status --porcelain`` directly (:func:`~pgml.provenance.git_state`'s ``"git"`` source).
The cluster mirror is rsynced WITHOUT ``.git`` (a bare working tree, deployed from CI), so
``git`` has nothing to ask there; the sync step instead drops a
:data:`~pgml.provenance.PROVENANCE_FILENAME` (``.git-provenance.json``) file at the mirror's
repository root — the commit and dirty flag resolved on the SUBMITTING machine, before the
sync — and :func:`~pgml.provenance.git_state` falls back to reading it (the ``"file"``
source). Neither being available yields ``{"git_sha": "unknown", "git_dirty": None,
"git_source": "none"}`` rather than an error: provenance must never abort a run, only
describe it as best it can.

A dirty working tree (uncommitted local changes) is recorded, not rejected — the run still
proceeds, but a reader comparing two artifacts with the same ``git_sha`` and different
``git_dirty`` knows one of them may not match what that commit alone would produce.

.. automodule:: pgml.provenance
   :members:
   :show-inheritance:
