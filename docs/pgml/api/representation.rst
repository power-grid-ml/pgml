pgml.representation
===================

Reading and migrating a grid's representation: the per-phase equivalent written by the
converters (``metadata.tags["phase_mode"] == "single_phase_equiv"``) versus three-phase
totals. A consumer that compares grids across representations reads voltages and powers
through :func:`~pgml.representation.system_totals`. An untagged grid is read as the
schema defines it (a one-phase node is a single-phase circuit with a line-to-neutral
rating); a grid written by an older converter (line-to-line one-phase nodes, three-phase
totals) is tagged ``"legacy_ll"`` with :func:`~pgml.representation.mark_legacy` and
migrated with :func:`~pgml.representation.to_per_phase_equivalent`. The rule itself is
described on the :doc:`conventions page <../modeling/conventions>`.

.. automodule:: pgml.representation
   :members:
   :show-inheritance:
