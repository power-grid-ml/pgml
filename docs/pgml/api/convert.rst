pgml.convert
============

Converters from external formats to the `pgml` schema.

Each sub-package provides a ``to_grid()`` function that converts a
source-library object to a :class:`~pgml.schemas.Grid` and an ``id_map``
that traces source element indices back to our schema ids.

.. automodule:: pgml.convert
   :members:
   :show-inheritance:

.. _convert-phase-mode:

Phase representation — ``PhaseMode``
--------------------------------------

Every ``to_grid`` function accepts a ``phase_mode`` keyword (default
``PhaseMode.SINGLE_PHASE_EQUIV``) that selects between two representations:

``PhaseMode.SINGLE_PHASE_EQUIV``
    The default. Every node and branch is ``phases=(Phase.A,)`` and every
    line carries 1x1 matrices. This is the positive-sequence single-phase
    equivalent that reproduces the historical converter output byte-for-byte.
    The assembly's const-Z formula treats ``u_rated_v`` as line-to-line,
    which is the convention all three converters follow.

``PhaseMode.THREE_PHASE``
    Genuine per-phase abc representation. Nodes and branches become
    ``(A, B, C)`` (or a source-native phase tuple for OpenDSS, which may
    include ``Phase.N``). Lines carry n×n matrices built from sequence data.
    Sources become balanced 3-phase Thevenin equivalents with phase angles
    offset by 0 / −120 / +120 degrees. Asymmetric load data is captured
    where the source library provides it.

Shared scaffold — ``pgml.convert._common``
------------------------------------------

The library-agnostic plumbing shared across all three converters lives in
:mod:`pgml.convert._common`. It is an internal module (underscore prefix);
only :class:`~pgml.convert.PhaseMode` is re-exported at the package level.
The scaffold provides:

- :class:`~pgml.convert._common.IdCounter` — a monotonic integer id
  allocator (one instance per conversion so node, branch and appliance ids
  never collide).
- :func:`~pgml.convert._common.phases_for` — the single place the
  node/branch phase tuple is decided for a given ``PhaseMode``.
- :func:`~pgml.convert._common.sequence_to_phase_matrices` — converts
  positive/zero-sequence quantities ``(r1, x1, c1, r0, x0, c0)`` to 3x3
  per-phase ``(R, L, C, G)`` matrices via the symmetric-component identity:

  .. math::

     Q_\text{self}   = (Q_0 + 2 Q_1) / 3 \\
     Q_\text{mutual} = (Q_0 -   Q_1) / 3

  applied independently to R, X, C and G (where
  :math:`L = X / (2\pi f_0)`).

- :func:`~pgml.convert._common.thevenin_from_z` and
  :func:`~pgml.convert._common.thevenin_from_sk` — Thevenin ``(R, L)``
  from an explicit series impedance or from short-circuit power and R/X
  ratio.
- :func:`~pgml.convert._common.build_node`,
  :func:`~pgml.convert._common.build_load`,
  :func:`~pgml.convert._common.build_source`,
  :func:`~pgml.convert._common.build_line_from_sequence`,
  :func:`~pgml.convert._common.build_line_from_matrices` — emit helpers
  that centralise the ``PhaseMode`` decision so all converters stamp
  identical schema objects for the same physical input.

Zero-sequence defaults (``THREE_PHASE`` line expansion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a positive-sequence line (``r1``, ``x1``, ``c1``) is expanded to an
abc phase-domain matrix and the source dataset provides no native
zero-sequence data, the zero-sequence quantities are synthesized from
config defaults::

    r0 = r1 * line.zero_sequence.r0_over_r1   (default 4.0)
    x0 = x1 * line.zero_sequence.x0_over_x1   (default 3.0)
    c0 = c1 * line.zero_sequence.c0_over_c1   (default 0.5)

An explicit native ``r0`` / ``x0`` / ``c0`` value from the source dataset
always overrides these defaults. OpenDSS provides native n×n matrices
directly and is not affected by this fallback.

Asymmetric load capture
~~~~~~~~~~~~~~~~~~~~~~~

Each library exposes different asymmetric load data:

- **pandapower** — ``net.asymmetric_load`` rows carry ``p_a_mw``,
  ``p_b_mw``, ``p_c_mw`` and the ``type`` column (``"wye"`` or
  ``"delta"``). Under ``THREE_PHASE``, each row is emitted as a
  :class:`~pgml.schemas.grid_schema.Load` with the corresponding
  :class:`~pgml.schemas.grid_schema.WindingConnection` and
  ``p_nom_per_phase_w`` / ``q_nom_per_phase_var``. Under
  ``SINGLE_PHASE_EQUIV`` the per-phase values are summed and logged at
  INFO level (no asymmetric representation is possible in 1-phase mode).

- **power-grid-model** — ``asym_load`` entries carry ``p_specified`` and
  ``q_specified`` as shape-``(3,)`` arrays (phases A, B, C). These are
  captured as per-phase loads with ``connection=WindingConnection.WYE``
  (power-grid-model has no load connection field; all pgm loads are
  modelled wye).

- **OpenDSS** — multi-phase and single-phase loads are read natively from
  the circuit. Under ``THREE_PHASE``, each load's connection is set from
  ``dss.Loads.IsDelta()`` (``True`` → ``DELTA``, ``False`` → ``WYE``).
  Single-phase loads are placed on their real bus-suffix phase (e.g. ``.1``
  → ``Phase.A``).

Transformer caveat (``THREE_PHASE``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under ``PhaseMode.THREE_PHASE``, transformers are stamped with a
per-phase diagonal admittance matrix. Vector-group phase coupling and the
zero-sequence path are **not yet modelled** (a warning is logged at
conversion time). Results are approximate for non-Dyn vector groups (e.g.
Yyn, YNyn). This limitation applies to the pandapower and power-grid-model
converters; OpenDSS does not yet convert Transformer elements.

pgml.convert.pandapower
------------------------

Convert a `pandapower <https://www.pandapower.org/>`_ network to a
:class:`~pgml.schemas.Grid`.

.. note::

   ``pandapower`` is mocked in the docs build (numpy 2.x incompatibility).
   The public API (``to_grid``) is documented from the source directly.

pandapower 2.14 — the newest release installable alongside this package — still reads
``np.Inf`` / ``np.in1d``, both removed in numpy 2.0. Call
:func:`~pgml.convert.pandapower.ensure_numpy_compat` once, before importing or running
pandapower, to restore the numpy-1.x aliases it needs::

    from pgml.convert.pandapower import ensure_numpy_compat, to_grid

    ensure_numpy_compat()    # numpy 2.x compat shim; idempotent
    import pandapower.networks as pn

    net = pn.case33bw()
    grid, id_map = to_grid(net)

Every builder in :mod:`pgml.grids` calls this shim internally, so callers of
:func:`~pgml.grids.ieee33_geometry_grid` and friends never need to call it directly.

.. automodule:: pgml.convert.pandapower
   :members:
   :show-inheritance:

pgml.convert.pgm
-----------------

Convert a `power-grid-model <https://power-grid-model.readthedocs.io/>`_
``input_data`` dict to a :class:`~pgml.schemas.Grid`.

.. automodule:: pgml.convert.pgm
   :members:
   :show-inheritance:

pgml.convert.opendss
---------------------

Convert an OpenDSS circuit (via `opendssdirect
<https://opendssdirect.readthedocs.io/>`_) to a :class:`~pgml.schemas.Grid`.

.. automodule:: pgml.convert.opendss
   :members:
   :show-inheritance:
