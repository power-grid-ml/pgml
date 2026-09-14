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

  A WYE load/generator/PVSystem/Storage additionally carries its OWN return-conductor
  choice: OpenDSS resolves each element's return conductor independently
  (``CktElement.NodeOrder()`` — grounded, or an explicit non-zero tie such as ``.4``),
  and the converter maps it to
  :attr:`~pgml.schemas.grid_schema.InjectionAppliance.return_path` per element rather
  than applying one shared rule to every WYE element on a bus. This is what lets a
  solidly-grounded load and a neutral-returning load coexist correctly on the SAME
  4-wire bus — see :doc:`/pgml/modeling/asymmetric` §4.

  ``Capacitor`` / ``Reactor`` elements convert to a
  :class:`~pgml.schemas.grid_schema.ShuntAppliance` — WYE (solidly grounded) by
  default, or DELTA (``connection=DELTA``, per-leg G/C from OpenDSS's own resolved
  per-leg ``Cuf`` / ``R`` / ``X``) when the DSS element is delta-connected.

Transformers (vector-group aware, both phase modes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All three converters read the winding connections and clock and set
:attr:`~pgml.schemas.grid_schema.Transformer.from_connection` /
:attr:`~pgml.schemas.grid_schema.Transformer.to_connection` /
``tap.shift_deg`` on the schema object, so assembly selects the vector-group
stamp automatically from the branch's own phase count — no ``THREE_PHASE``-specific
approximation remains:

- ``PhaseMode.SINGLE_PHASE_EQUIV`` folds the winding connections and clock into the
  classical scalar off-nominal-tap pi (magnitude + shift only, no topology).
- ``PhaseMode.THREE_PHASE`` builds the full phase-domain winding-incidence stamp
  (:mod:`pgml.assembly`'s ``Y = Nᵀ·Y_winding·N`` primitive — see
  :doc:`/pgml/modeling/transformer`): delta/zigzag phase coupling and the
  zero-sequence path (blocked by a delta or an ungrounded wye, transferred through a
  zigzag's limb incidence) are modelled for every winding pairing and every clock
  number consistent with the pairing's parity.

Both modes agree on the positive-sequence terminal admittance for every supported
connection pair. Each converter's own scope (which source fields are read, which
clocks/pairings are reachable from that source format) is documented in the
corresponding ``to_grid`` docstring below and in
:doc:`/pgml/modeling/conventions` §§2–3.

pgml.convert.pandapower
------------------------

Convert a `pandapower <https://www.pandapower.org/>`_ network to a
:class:`~pgml.schemas.Grid`.

.. note::

   ``pandapower`` is mocked in the docs build.
   The public API (``to_grid``) is documented from the source directly.

Usage::

    from pgml.convert.pandapower import to_grid

    import pandapower.networks as pn

    net = pn.case33bw()
    grid, id_map = to_grid(net)

Element coverage
~~~~~~~~~~~~~~~~

``bus``, ``line``, ``trafo`` (two-winding, vector-group and tap-changer aware),
bus-bus and bus-element ``switch``, ``load`` (including the four-column ZIP percentages),
``asymmetric_load``, ``sgen``, ``gen`` (see ``gen_mode`` below), ``storage`` and
``shunt`` convert.
``ext_grid`` converts with its zero-sequence short-circuit data (``x0x_max``,
``r0x0_max``).  Every remaining non-empty table — ``trafo3w``, ``impedance``,
``ward`` / ``xward``, ``dcline``, ``motor``, ``asymmetric_sgen`` —
raises a WARNING naming the kind and count; nothing is dropped silently.

A ``shunt`` row becomes a fixed WYE :class:`~pgml.schemas.grid_schema.ShuntAppliance`,
``G`` from ``p_mw`` and ``C`` from ``−q_mvar/(2πf₀)``, both referred to the row's own
``vn_kv``.  An INDUCTIVE shunt (``q_mvar > 0``) therefore becomes a negative capacitance:
exact at the fundamental, but its susceptance magnitude rises with frequency where a real
reactor's falls as ``1/h``, so harmonic results at such a bus are not faithful.  The
converter warns and names the count.

Open line and transformer terminals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pandapower bus-line (``et='l'``) or bus-transformer (``et='t'``) switch controls
one element terminal. The ``open_switch_model`` keyword selects its representation:

``"terminal"``
    The default and the full pandapower-compatible model. A singly-open element remains
    connected at its other end, while its open terminal is rewired to an auxiliary
    :class:`~pgml.schemas.grid_schema.Node`. This retains the connected terminal's line
    charging or transformer no-load current. ``id_map["open_terminal"]`` maps each open
    source switch index to its auxiliary node id.

``"drop_element"``
    The legacy reduced model. An open switch at either terminal omits the complete line
    or transformer, including the shunt at its connected end.

An element open at both terminals, or marked out of service, is omitted in both modes.
Closed bus-element switches leave their element unchanged. Bus-bus switches remain
:class:`~pgml.schemas.grid_schema.Switch` branches and are unaffected by this keyword.

For example, the legacy approximation is explicit::

    grid, id_map = to_grid(net, open_switch_model="drop_element")

Storage snapshot convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~

An in-service ``net.storage`` row becomes a
:class:`~pgml.schemas.grid_schema.Storage` at its original bus. pandapower's P/Q
sign is consumption-positive, so the converter negates ``p_mw`` and ``q_mvar``
after applying ``scaling`` to produce pgml's discharge-positive nameplate values.
Positive ``max_e_mwh`` maps to ``energy_capacity_wh``; ``soc_percent`` maps to a
fractional ``soc``, and ``min_e_mwh / max_e_mwh`` maps to ``soc_min``. These energy
fields record snapshot state and bounds. The converter does not invent missing
inverter ratings or dynamics.

.. _convert-pandapower-gen-mode:

Voltage-controlled generators — ``gen_mode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``net.gen`` is pandapower's **PV bus**: fixed active power, regulated voltage
magnitude ``vm_pu``, reactive power free between ``min_q_mvar`` and
``max_q_mvar``.  pgml models it exactly, through the residual row pair
``[P-balance; |V|² − V_set²]`` and a
:class:`~pgml.schemas.grid_schema.VoltageRegulation` block on the generator (see
:doc:`/pgml/modeling/der-pv-storage`).  The ``gen_mode`` keyword selects:

``GenMode.VOLTAGE_REGULATING``
    The default.  Each in-service row becomes a
    :class:`~pgml.schemas.grid_schema.Generator` carrying a
    :class:`~pgml.schemas.grid_schema.VoltageRegulation` block, so the solver holds
    ``vm_pu`` and solves the reactive power within the row's limits.  Rows on one bus are
    merged into a single terminal.  Against ``pp.runpp`` on the MATPOWER benchmarks the
    converged voltages agree to between 4.4e-16 and 8.9e-12 pu, and the generator reactive
    powers to 7.9e-9 Mvar.

``GenMode.DROP``
    ``net.gen`` is not read and is reported as a dropped element.  A transmission benchmark
    whose generators live in ``net.gen`` (MATPOWER ``case118``, ``case39``, …) then converts
    to loads plus a slack, and its operating point is **not** the source network's.

``GenMode.VOLT_VAR_APPROX``
    Each in-service row becomes a
    :class:`~pgml.schemas.grid_schema.Generator` whose
    :class:`~pgml.schemas.grid_schema.VoltVarControl` is a steep ``Q(|V|)`` droop
    centred on ``vm_pu`` and saturating at the row's reactive limits, with the
    active power mapped exactly as ``sgen`` is (generation-positive, ``scaling``
    applied). This **approximates** a PV bus: the bus settles off its setpoint by
    ``Q / (slope · Q_base)`` per unit, so the deviation falls as
    ``1 / gen_volt_var_slope_pu`` (default 500). A row on the ``ext_grid`` bus, or
    one flagged ``slack``, is skipped and logged — the ideal slack already fixes
    that bus's voltage. A row whose reactive limits coincide has no reactive
    freedom and converts as a plain PQ generator instead.

    Solve the result with ``solve_power_flow(..., method="newton")``: the
    current-injection fixed point does not contract on a stiff droop. The binding
    limitation is conditioning, not steady-state accuracy — outside the
    ``1/slope``-wide band the droop's ``dQ/d|V|`` is zero, so on a heavily loaded
    transmission network the solve can settle on the collapsed low-voltage branch.
    Reduce the steepness when that happens.  Use this mode to model a real
    droop-controlled DER, not to import a transmission benchmark.

.. automodule:: pgml.convert.pandapower
   :members:
   :show-inheritance:

pgml.convert.pgm
-----------------

Convert a `power-grid-model <https://power-grid-model.readthedocs.io/en/stable/>`_
``input_data`` dict to a :class:`~pgml.schemas.Grid`.

``node``, ``line``, ``transformer``, ``link``, ``sym_load``, ``asym_load``, ``sym_gen`` and
``source`` convert; ``source.z01_ratio`` is read into the Thévenin's zero-sequence split.
A ``link`` is power-grid-model's perfect connection, and it converts to an ideal closed
:class:`~pgml.schemas.grid_schema.Switch` that the solve collapses exactly — on a
source-link-line-load feeder the node voltages agree with power-grid-model to 3.5e-9 pu and
the link's own current to 6.0e-9 relative, both residuals being the reference's own
stand-in drop.  ``asym_gen``, ``shunt``, ``three_winding_transformer`` and
``transformer_tap_regulator`` are not read and are reported as dropped elements.
``voltage_regulator``, power-grid-model's own PV terminal, is not mapped yet, although the
pgml side of the mapping exists.

.. automodule:: pgml.convert.pgm
   :members:
   :show-inheritance:

pgml.convert.opendss
---------------------

Convert an OpenDSS circuit (via `OpenDSSDirect.py
<https://dss-extensions.org/OpenDSSDirect.py/>`_) to a :class:`~pgml.schemas.Grid`.

.. automodule:: pgml.convert.opendss
   :members:
   :show-inheritance:
