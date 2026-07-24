"""Pure conversion function: OpenDSS circuit (via opendssdirect) -> (Grid, id_map).

Conventions applied
-------------------
Unit conversion (OpenDSS engineering -> SI):

Lines
~~~~~
``Lines.RMatrix()`` returns the resistance matrix in Ohm per length-unit (per-length).
``Lines.XMatrix()`` returns the reactance matrix in Ohm per length-unit.
``Lines.CMatrix()`` returns the capacitance matrix in nF per length-unit.
``Lines.Length()``  returns the length in the line's length unit.
``Lines.Units()``   returns the unit code (0=none, 1=mi, 2=kft, 3=km, 4=m, 5=ft, 6=in, 7=cm).

When ``units == 0`` (none), length=1 and R/X are the TOTAL ohms for the branch;
otherwise R/X are per the stated length unit and must be multiplied by length to
get total ohms.

Conversion to our SI per-length schema::

    length_m        = length_in_unit * meters_per_unit
    R_total_ohm     = Rmatrix * length_in_unit       (Ohm)
    X_total_ohm     = Xmatrix * length_in_unit       (Ohm)
    C_total_F       = Cmatrix * length_in_unit * 1e-9  (F)
    r_per_m         = R_total_ohm / length_m         (Ohm/m)
    l_per_m         = X_total_ohm / (2*pi*f0) / length_m  (H/m)
    c_per_m         = C_total_F / length_m           (F/m)

Loads
~~~~~
``Loads.kW()``, ``Loads.kvar()``, ``Loads.kV()`` — total P/Q in kW/kVAR, kV L-N.
P in W = kW * 1e3; Q in VAR = kvar * 1e3.
Connection is read via ``dss.Loads.IsDelta()``:  ``True`` -> DELTA, ``False`` -> WYE.
Single-phase loads are placed on their real phase (bus suffix ``.k``); the
connection is WYE (L-N) as the default for a two-conductor single-phase load
following the OpenDSS convention (NeutralRules: grounded-wye path when phases=1).

``u_rated_v`` convention
~~~~~~~~~~~~~~~~~~~~~~~~
The pgml schema convention (and the pandapower/pgm converters) stores the
**line-to-line** rated voltage for all nodes, because the const-Z load shunt
formula is ``y = conj(S) / u_rated_v^2`` with ``u_rated_v`` being L-L.
OpenDSS ``Bus.kVBase()`` always returns ``BasekV_LL / sqrt(3)`` (line-to-neutral),
regardless of the phase count on the element that established the base voltage.
The converter therefore always applies ``u_rated_v = kVBase * sqrt(3) * 1000``.

For the canonical IEEE 33-bus (single-phase positive-sequence circuit built with
``phases=1`` and ``basekv=12.66 kV``): OpenDSS stores ``kVBase = 12.66/sqrt(3) =
7.31 kV``.  Applying ``* sqrt(3) * 1000`` recovers ``12660 V``, matching the
pandapower reference.  The old converter stored ``kVBase * 1000 = 7309 V``
(L-N), which was a factor-of-sqrt(3) error in the const-Z shunt for load-flow
studies.  The Y-bus oracle test (passive network, no loads) was unaffected; the
load-flow path (``solve_power_flow``) is corrected by this fix.

``phase_mode`` controls the node/branch representation:

- ``SINGLE_PHASE_EQUIV`` (default): every node/branch is ``phases=(Phase.A,)``;
  lines carry 1x1 matrices (the ``[0][0]`` element of the DSS matrix).  This is
  byte-identical to the historical converter output (except for the ``u_rated_v``
  fix above, which does not change the IEEE 33-bus numbers because that circuit uses
  single-phase elements with kVBase == kV_LL).
- ``THREE_PHASE``: nodes carry their real DSS phases (incl. ``Phase.N`` when the
  bus has a neutral conductor); lines carry the full n×n matrices from
  ``Lines.RMatrix()/XMatrix()/CMatrix()``; sources become balanced 3-phase
  Thevenins; loads capture their ``IsDelta()`` connection and real phase placement.

Vsource (external network)
~~~~~~~~~~~~~~~~~~~~~~~~~~
R1/X1 are read via ``dss.Text.Command('? Vsource.<name>.r1')`` etc. (in Ohms).
The Vsource is converted to a :class:`~pgml.schemas.grid_schema.Source` with
Thevenin impedance = (R1, L=X1/(2*pi*f0)), and ``u_ref_v = BasekV * pu * 1e3`` V.

``BasekV`` semantics (verified empirically; NOT fully spelled out by the general
OpenDSS documentation, which describes ``basekv`` as line-to-line only):

- For a Vsource with ``phases>=3``, ``BasekV`` genuinely is the line-to-line
  nominal -- OpenDSS internally divides by ``sqrt(3)`` to get the solved
  line-to-neutral EMF, matching the documented convention.
- For a ``phases=1`` Vsource, OpenDSS uses ``BasekV`` DIRECTLY, unscaled, as
  the magnitude of the single conductor-pair EMF -- there is no internal
  ``sqrt(3)`` anywhere for a 1-phase source (confirmed by comparing the
  solved ``Bus.Voltages()`` magnitude to ``BasekV*pu`` for both a genuine
  line-to-neutral ``basekv`` and the legacy positive-sequence-equivalent
  style, where ``basekv`` is set to the ORIGINAL 3-phase system's
  line-to-line nominal, e.g. the IEEE 33-bus fixtures used throughout this
  test suite). Because ``u_ref_v = BasekV * pu * 1e3`` and (for the node
  voltage base) ``u_rated_v = kVBase() * sqrt(3) * 1e3`` both simply mirror
  whatever OpenDSS itself does with ``BasekV`` for the given phase count,
  NEITHER formula needs a phase-count branch: they are correct for a
  ``phases>=3`` source (recovering the L-L nominal) and for a ``phases=1``
  source (recovering the L-N/single-conductor EMF) alike. See
  ``docs/pgml/modeling/references/opendss/index.md`` and
  ``tests/convert/test_opendss_vsource_basekv.py`` for the empirical proof.

Phase convention
~~~~~~~~~~~~~~~~
For a single-phase positive-sequence circuit (phases=1 nodes, .1 suffix),
every node gets ``phases=(Phase.A,)``. The Y-bus row for Phase.A corresponds
to OpenDSS YNodeOrder entries ending in ".1".

Node ordering
~~~~~~~~~~~~~
Nodes are registered in the order they first appear in ``dss.Circuit.YNodeOrder()``
(which is the same order as the Y matrix rows/columns). This guarantees that our
``node_phase_index`` row ordering matches the DSS Y matrix ordering after the
alignment step.

Only in-service elements are converted. The converter assumes the circuit has been
solved (or at least ``Calcvoltagebases`` has been called) before calling ``to_grid``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

from pgml.convert._common import (
    IdCounter,
    PhaseMode,
    build_generator,
    build_line_from_matrices,
    build_load,
    build_node,
    build_source,
    make_metadata,
    phases_for,
    thevenin_from_z,
    warn_dropped_elements,
)
from pgml.errors import ConversionError
from pgml.schemas.grid_schema import (
    ComplexTap,
    ConsumerType,
    Grid,
    LoadModel,
    Phase,
    Provenance,
    ShuntAppliance,
    SourceConvention,
    Storage,
    Transformer,
    WindingConnection,
    ZipCoefficients,
)

_logger = logging.getLogger("pgml")

# Length unit codes in opendssdirect -> meters per unit
_DSS_UNIT_TO_METERS: dict[int, float] = {
    0: 1.0,  # none (total values, length=1 by convention)
    1: 1609.344,  # miles
    2: 304.8,  # kft
    3: 1000.0,  # km
    4: 1.0,  # m
    5: 0.3048,  # ft
    6: 0.0254,  # in
    7: 0.01,  # cm
}

_PROVENANCE = Provenance(
    source_convention=SourceConvention.IMPEDANCE,
    notes=(
        "Converted from OpenDSS circuit (opendssdirect). "
        "Engineering units converted to SI. "
        "u_rated_v is line-to-line for every bus (= kVBase * sqrt(3) * 1000), "
        "following the pgml schema convention; OpenDSS kVBase() returns L-N. "
        "Load connection from IsDelta(); single-phase loads default to WYE (L-N)."
    ),
)

# sqrt(3): used to convert L-N kVBase -> L-L for every bus
_SQRT3 = math.sqrt(3.0)


def to_grid(
    dss: Any, *, phase_mode: PhaseMode = PhaseMode.SINGLE_PHASE_EQUIV
) -> tuple[Grid, dict[str, Any]]:
    """Convert the currently-loaded OpenDSS circuit to a :class:`~pgml.schemas.grid_schema.Grid`.

    Parameters
    ----------
    dss:
        The ``opendssdirect`` module (``import opendssdirect as dss;
        dss.Text.Command('Solve')``) with a circuit already loaded and solved
        (or ``Calcvoltagebases`` called).
    phase_mode:
        :class:`~pgml.convert._common.PhaseMode`. ``SINGLE_PHASE_EQUIV`` (default)
        keeps today's positive-sequence single-phase-equivalent: every node/branch
        is ``phases=(Phase.A,)`` and lines carry 1x1 matrices (the leading diagonal
        entry of the DSS matrix). ``THREE_PHASE`` emits the real DSS phases (incl.
        ``Phase.N`` for neutral conductors), full n×n line matrices, balanced
        3-phase Thevenin sources, and load ``connection`` from ``IsDelta()``.

    Returns
    -------
    tuple[Grid, dict]
        A ``(Grid, id_map)`` pair.  ``Grid`` is the materialised schema object
        (no ``type_ref``).  ``id_map`` maps DSS element names to our schema ids:

        - ``"bus"``       -> ``{dss_bus_name_lower: Node.id}``
        - ``"line"``      -> ``{dss_line_name_lower: Line.id}``
        - ``"trafo"``     -> ``{dss_trafo_name_lower: Transformer.id}``
        - ``"load"``      -> ``{dss_load_name_lower: Load.id}``
        - ``"vsource"``   -> ``{dss_vsrc_name_lower: Source.id}``
        - ``"capacitor"`` -> ``{dss_cap_name_lower: ShuntAppliance.id}``
        - ``"reactor"``   -> ``{dss_reactor_name_lower: ShuntAppliance.id}``
        - ``"generator"`` -> ``{dss_gen_name_lower: Generator.id}``
        - ``"pvsystem"``  -> ``{dss_pv_name_lower: Generator.id}``
        - ``"storage"``   -> ``{dss_storage_name_lower: Storage.id}``
        - ``"slack_v_complex"`` -> complex slack voltage phasor (V, line-to-line)
          from the first Vsource, for ideal-slack mode.

    Notes
    -----
    - DSS bus and element names are normalised to lowercase.
    - Node ids are assigned in YNodeOrder sequence (bus.phase pairs,
      alphabetical in DSS's internal order) so our compact node-phase index
      matches the DSS Y-matrix row ordering for alignment in oracle tests.
    - ``Line``, ``Transformer``, ``Vsource``, ``Load``, ``Capacitor``,
      ``Reactor``, ``Generator``, ``PVSystem`` and ``Storage`` element types
      are handled; every other non-empty DSS element class (``Isource``,
      controls, monitors/meters, ...) triggers a WARNING naming the kind and
      count -- nothing is dropped silently (see :func:`warn_dropped_elements`).
      Two-winding ``Transformer`` elements convert to
      :class:`~pgml.schemas.grid_schema.Transformer` (winding 1 = HV/from,
      winding 2 = LV/to; solidly grounded wye or delta windings only; see
      the module CONTEXT.md for the full field mapping and scope).
    - ``First()``/``Next()`` class iterators already skip DISABLED elements
      (verified empirically against opendssdirect 0.9.4); every element loop
      below therefore only ever sees in-service elements without an explicit
      ``CktElement.Enabled()`` check.
    - ``u_rated_v`` is line-to-line for every bus (kVBase * sqrt(3) * 1000),
      consistent with the pandapower/pgm converters and the assembly const-Z
      shunt formula. OpenDSS ``kVBase()`` returns L-N, recovered to L-L by the
      sqrt(3) factor.
    - A WYE ``Load``/``Generator``/``PVSystem``/``Storage`` reads its OWN resolved
      return conductor (``CktElement.NodeOrder()``) and carries it over as
      :attr:`~pgml.schemas.grid_schema.InjectionAppliance.return_path`
      (``"ground"``/``"neutral"``/``"auto"``) rather than applying one shared
      node-level rule, so two elements on the same four-wire bus can return
      differently, exactly as OpenDSS resolved them.
    - ``Capacitor``/``Reactor`` convert to a
      :class:`~pgml.schemas.grid_schema.ShuntAppliance`, WYE (solidly grounded) by
      default or DELTA when the DSS element is delta-connected (per-leg G/C from
      OpenDSS's own resolved per-leg ``Cuf``/``R``/``X``).
    """
    f0_hz: float = float(dss.Solution.Frequency())
    two_pi_f0 = 2.0 * math.pi * f0_hz

    _id = IdCounter()

    id_map: dict[str, Any] = {
        "bus": {},
        "line": {},
        "trafo": {},
        "load": {},
        "vsource": {},
        "capacitor": {},
        "reactor": {},
        "generator": {},
        "pvsystem": {},
        "storage": {},
        "slack_v_complex": None,
    }

    # ---------------------------------------------------------------------- #
    # 1. Nodes — register in YNodeOrder sequence                              #
    #    YNodeOrder entries are "BUSNAME.phase" (uppercase). We register      #
    #    one node per unique bus name (ignoring phases for now), then one     #
    #    (node_id, phase) per entry in YNodeOrder.                            #
    # ---------------------------------------------------------------------- #
    node_order = dss.Circuit.YNodeOrder()
    # Build: bus_name_lower -> Node.id (first occurrence wins)
    nodes: list = []
    bus_name_to_node_id: dict[str, int] = {}
    # Also track phases seen per bus
    bus_phases: dict[str, list[Phase]] = {}  # bus_name_lower -> ordered phases

    for entry in node_order:
        # entry: "BUSNAME.N" where N=1/2/3 (phase number)
        parts = entry.upper().split(".")
        bus_name_raw = parts[0].lower()
        phase_num = int(parts[1]) if len(parts) > 1 else 1

        phase = _phase_num_to_enum(phase_num)

        if bus_name_raw not in bus_phases:
            bus_phases[bus_name_raw] = []
        if phase not in bus_phases[bus_name_raw]:
            bus_phases[bus_name_raw].append(phase)

    # Now get rated voltage per bus from the circuit.
    # ``Bus.kVBase()`` always returns the line-to-neutral kV (BasekV / sqrt(3)),
    # regardless of the number of phases on the bus.  The pgml schema convention
    # (matching pandapower and pgm) stores the LINE-TO-LINE rated voltage so that
    # the const-Z load shunt formula ``y = conj(S) / u_rated_v^2`` is consistent
    # across all converters.  Therefore: ``u_rated_v = kVBase * sqrt(3) * 1000``.
    #
    # Rationale: OpenDSS sets ``kVBase = BasekV_LL / sqrt(3)`` internally for every
    # bus, so ``kVBase * sqrt(3) = BasekV_LL`` recovers the line-to-line nominal.
    # This applies equally to single-phase buses (e.g. IEEE 33-bus built with
    # ``basekv=12.66 kV, phases=1``): kVBase = 7.31 kV -> u_rated_v = 12660 V,
    # which matches the pandapower reference for the same circuit.
    for bus_name_lower, phases_list in bus_phases.items():
        node_id = _id.next()
        bus_name_to_node_id[bus_name_lower] = node_id

        dss.Circuit.SetActiveBus(bus_name_lower)
        kv_base_ln = dss.Bus.kVBase()  # L-N kV (always BasekV_LL / sqrt(3))
        # Recover line-to-line: u_rated_v = kVBase * sqrt(3) * 1000 V
        u_rated_v = kv_base_ln * _SQRT3 * 1_000.0

        native_phases = tuple(phases_list)
        nodes.append(
            build_node(
                id=node_id,
                name=bus_name_lower,
                u_rated_v=u_rated_v,
                mode=phase_mode,
                native_phases=native_phases,
            )
        )

    id_map["bus"] = dict(bus_name_to_node_id)

    # ---------------------------------------------------------------------- #
    # 2. Lines                                                                #
    # ---------------------------------------------------------------------- #
    branches: list = []

    ret = dss.Lines.First()
    while ret:
        line_name = dss.Lines.Name().lower()
        n_phases = dss.Lines.Phases()

        # Get bus connections (e.g. "bus0.1", "bus1.1")
        bus1_str = dss.Lines.Bus1().lower()  # "busname.phase"
        bus2_str = dss.Lines.Bus2().lower()

        from_bus_name, from_phases = _parse_bus_connection(bus1_str, n_phases)
        to_bus_name, to_phases = _parse_bus_connection(bus2_str, n_phases)

        # Skip lines whose buses are not in our node map
        if (
            from_bus_name not in bus_name_to_node_id
            or to_bus_name not in bus_name_to_node_id
        ):
            ret = dss.Lines.Next()
            continue

        line_id = _id.next()
        id_map["line"][line_name] = line_id

        # Get length and units
        length_in_unit = dss.Lines.Length()
        unit_code = dss.Lines.Units()
        meters_per_unit = _DSS_UNIT_TO_METERS.get(unit_code, 1.0)
        length_m = length_in_unit * meters_per_unit

        # Get R/X matrices (per length-unit)
        r_mat_flat = list(dss.Lines.RMatrix())  # Ohm/length-unit
        x_mat_flat = list(dss.Lines.XMatrix())  # Ohm/length-unit
        c_mat_flat = list(dss.Lines.CMatrix())  # nF/length-unit

        # Convert to total Ohm / H / F
        r_total = [v * length_in_unit for v in r_mat_flat]  # Ohm
        x_total = [v * length_in_unit for v in x_mat_flat]  # Ohm
        c_total_nf = [v * length_in_unit for v in c_mat_flat]  # nF

        # Convert to SI per-meter
        r_per_m_flat = [v / length_m for v in r_total]  # Ohm/m
        l_per_m_flat = [v / two_pi_f0 / length_m for v in x_total]  # H/m
        c_per_m_flat = [v * 1e-9 / length_m for v in c_total_nf]  # F/m

        if phase_mode is PhaseMode.SINGLE_PHASE_EQUIV:
            # Positive-sequence equivalent. A genuinely 1-phase DSS line (the
            # IEEE 33-bus oracle) keeps the exact [0][0] entry (byte-identical
            # to the historical converter); a coupled multi-phase line is
            # reduced to its POSITIVE-SEQUENCE impedance Z1 = Z_self - Z_mutual
            # (mean diagonal minus mean off-diagonal, applied independently to
            # R and L) rather than the self impedance alone -- using the self
            # entry ignores the mutual coupling entirely and overstates the
            # positive-sequence impedance. The same reduction applies to the
            # Maxwell C matrix; C's off-diagonals are negative (mutual
            # coupling reduces net charge), so C1 = C_self - C_mutual is
            # LARGER than C_self, which is the physically correct direction.
            r1 = _positive_sequence_scalar(r_per_m_flat, n_phases)
            l1 = _positive_sequence_scalar(l_per_m_flat, n_phases)
            c1 = _positive_sequence_scalar(c_per_m_flat, n_phases)
            r_mat = [[r1]]
            l_mat = [[l1]]
            c_mat = [[c1]]
            line_phases: tuple[Phase, ...] = phases_for(phase_mode)
            line_to_phases: Optional[tuple[Phase, ...]] = None
        else:
            # THREE_PHASE: emit the real n×n matrices from DSS. `to_phases` is
            # parsed independently from `bus2` -- a line whose two terminals
            # list their phase conductors in a DIFFERENT order (e.g.
            # `bus1=a.1.2.3 bus2=b.3.2.1`) is a genuine phase-transposing
            # connection; conductor k of the R/X/C matrix ties `from_phases[k]`
            # at `from_node` to `to_phases[k]` at `to_node` (the same
            # convention already used by Transformer windings; the assembly's
            # series-branch stamp (`_series_terminal_indices` in
            # `pgml.assembly.ybus`) already indexes the two terminals
            # independently, so no assembly change is needed for this case).
            r_mat = _flat_to_matrix(r_per_m_flat, n_phases)
            l_mat = _flat_to_matrix(l_per_m_flat, n_phases)
            c_mat = _flat_to_matrix(c_per_m_flat, n_phases)
            line_phases = phases_for(phase_mode, native=tuple(from_phases))
            line_to_phases = phases_for(phase_mode, native=tuple(to_phases))

        branches.append(
            build_line_from_matrices(
                id=line_id,
                name=line_name,
                from_node=bus_name_to_node_id[from_bus_name],
                to_node=bus_name_to_node_id[to_bus_name],
                phases=line_phases,
                to_phases=line_to_phases,
                length_m=length_m,
                r_matrix=r_mat,
                l_matrix=l_mat,
                c_matrix=c_mat,
                g_matrix=None,  # OpenDSS GMatrix is rare in distribution; omit
                provenance=_PROVENANCE,
            )
        )

        ret = dss.Lines.Next()

    # ---------------------------------------------------------------------- #
    # 3. Transformers (two-winding, vector-group aware)                       #
    # ---------------------------------------------------------------------- #
    # Winding 1 = HV/from, winding 2 = LV/to (standard OpenDSS convention: the
    # convention also used by this converter's own live-oracle transformer
    # builder, `pgml.evaluation.oracles.opendss_oracle
    # ._build_circuit_with_real_transformer`). The leakage is referred to the
    # TO/LV coil (pgml's storage convention, see docs/pgml/modeling/
    # transformer.md and conventions.md sec. 2): OpenDSS's per-winding `%R`
    # and inter-winding `XHL` are per-unit (base-invariant) quantities on the
    # STANDARD line-to-line base, so
    #     R_ll_ohm = (%R_wdg1 + %R_wdg2)/100 * Z_base_LV
    #     X_ll_ohm = %XHL/100 * Z_base_LV,   Z_base_LV = kV_lv^2*1000/kVA
    # recovers the total leakage on that standard base -- valid whenever both
    # windings share one kVA rating (checked below; OpenDSS's own convention
    # places `XHL` on winding 1's kVA base, which coincides with winding 2's
    # when the two are equal). pgml stores the leakage referred to the ACTUAL
    # TO-side COIL, which is the same as the standard L-L base for a wye/
    # zigzag LV winding but 3x LARGER for a delta LV winding (a delta coil is
    # rated at the L-L voltage with 1/3 the per-phase kVA, so its natural
    # impedance base is `3*Z_base_LV`; see
    # `tests/reference/test_transformer_clock_matrix.py`'s `y_LL = 3*y_coil`
    # pin and docs/pgml/modeling/references/opendss/index.md sec. "Transformer
    # leakage"), so `R_lv_ohm/X_lv_ohm = 3*R_ll_ohm/X_ll_ohm` whenever the LV
    # winding is DELTA.
    #
    # Vector group / clock: OpenDSS has no explicit clock parameter -- its
    # `LeadLag` toggle only distinguishes the 30-degree Dy/Yd shift (verified
    # against a live solve: `Lag` -> the LV bus lags the HV bus by ~30 deg
    # (Dyn1, `shift_deg=30`); `Lead` -> LV leads by ~30 deg (Dyn11,
    # `shift_deg=330`)). A matching Yy/Dd pairing has no inherent phase shift
    # from `LeadLag` (clock 0 baseline). On top of that baseline, a winding
    # whose bus connection cyclically rotates the phase-conductor order (e.g.
    # `bus=lv.2.3.1.0`) contributes a further +-4 clock steps (verified
    # against a live solve, see `_cyclic_rotation_steps` and the module
    # CONTEXT.md "Cyclic winding-bus rotation" section) -- this is how Dyn5,
    # YNd5, Yy4, etc. are converted even though OpenDSS has no explicit clock
    # field. The polarity-flip clocks {2, 6, 10} (e.g. Yy6/Dd6, the
    # 180-degree reversed-polarity group) need a genuinely reversed winding
    # construction that no bus wiring can express and are still not detected
    # here (see the module CONTEXT.md).
    #
    # Grounding: OpenDSS's shorthand bus notation (no explicit
    # (n_phases+1)-th conductor, or an explicit trailing `.0`) solidly grounds
    # a wye winding's neutral; ONLY that case is converted
    # (`WindingConnection.WYE_GROUNDED`). An explicit non-zero neutral node
    # (a genuinely floating or impedance-grounded neutral) is out of scope
    # (pgml's transformer assembly models solid grounding only) and raises.
    # ---------------------------------------------------------------------- #
    ret = dss.Transformers.First()
    while ret:
        trafo_name = dss.Transformers.Name().lower()
        n_wdg = dss.Transformers.NumWindings()
        if n_wdg != 2:
            raise ConversionError(
                f"OpenDSS transformer '{trafo_name}' has {n_wdg} windings; "
                "only two-winding transformers are supported."
            )

        dss.Circuit.SetActiveElement(f"Transformer.{trafo_name}")
        n_phases = dss.CktElement.NumPhases()
        bus_names_raw = [b.lower() for b in dss.CktElement.BusNames()]
        if len(bus_names_raw) != 2:
            raise ConversionError(
                f"OpenDSS transformer '{trafo_name}': expected 2 terminals, "
                f"found {len(bus_names_raw)}."
            )

        wdg_kv: list[float] = []
        wdg_kva: list[float] = []
        wdg_pct_r: list[float] = []
        wdg_tap: list[float] = []
        wdg_is_delta: list[bool] = []
        for w in (1, 2):
            dss.Transformers.Wdg(w)
            wdg_kv.append(float(dss.Transformers.kV()))
            wdg_kva.append(float(dss.Transformers.kVA()))
            wdg_pct_r.append(float(dss.Transformers.R()))
            wdg_tap.append(float(dss.Transformers.Tap()))
            wdg_is_delta.append(bool(dss.Transformers.IsDelta()))
        xhl_pct = float(dss.Transformers.Xhl())

        dss.Text.Command(f"? Transformer.{trafo_name}.%noloadloss")
        noloadloss_pct = float(dss.Text.Result().strip() or 0.0)
        dss.Text.Command(f"? Transformer.{trafo_name}.%imag")
        imag_pct = float(dss.Text.Result().strip() or 0.0)
        dss.Text.Command(f"? Transformer.{trafo_name}.xrconst")
        xrconst_str = dss.Text.Result().strip().lower()
        harmonic_xr_constant = xrconst_str in ("yes", "true", "1")

        kva_from, kva_to = wdg_kva
        if not math.isclose(kva_from, kva_to, rel_tol=1e-6):
            raise ConversionError(
                f"OpenDSS transformer '{trafo_name}': winding kVA ratings "
                f"differ ({kva_from} vs {kva_to} kVA); differing per-winding "
                "power bases are not supported (the %R/%XHL per-unit values "
                "are only base-invariant when both windings share one kVA "
                "rating)."
            )
        s_rated_va = kva_to * 1_000.0

        kv_from, kv_to = wdg_kv
        u_rated_from_v = kv_from * 1_000.0
        u_rated_to_v = kv_to * 1_000.0

        # Total leakage on the standard line-to-line base, then referred to
        # the ACTUAL TO-side coil (3x for a delta LV winding -- see the
        # comment block above this loop).
        z_base_lv_ohm = (kv_to**2 * 1_000.0) / kva_to
        r_pct_total = wdg_pct_r[0] + wdg_pct_r[1]
        r_ll_ohm = r_pct_total / 100.0 * z_base_lv_ohm
        x_ll_ohm = xhl_pct / 100.0 * z_base_lv_ohm
        _lv_coil_factor = 3.0 if wdg_is_delta[1] else 1.0
        r_lv_ohm = r_ll_ohm * _lv_coil_factor
        x_lv_ohm = x_ll_ohm * _lv_coil_factor
        l_lv_h = x_lv_ohm / two_pi_f0

        # Magnetizing shunt referred to the HV terminal (same derivation the
        # pandapower converter uses for pfe_kw/i0_percent -> G_m/B_m).
        pfe_w = noloadloss_pct / 100.0 * s_rated_va
        g_m = pfe_w / (u_rated_from_v**2) if pfe_w > 0.0 else 0.0
        l_m: Optional[float] = None
        if imag_pct > 0.0:
            i0_amp = imag_pct / 100.0 * s_rated_va / u_rated_from_v
            s_nl = u_rated_from_v * i0_amp
            q_nl_sq = s_nl**2 - pfe_w**2
            if q_nl_sq > 0.0:
                b_m = math.sqrt(q_nl_sq) / (u_rated_from_v**2)
                if b_m > 0.0:
                    l_m = 1.0 / (two_pi_f0 * b_m)

        from_bus_name, from_phase_list, from_grounded, from_rotation = (
            _parse_transformer_winding_bus(bus_names_raw[0], n_phases)
        )
        to_bus_name, to_phase_list, to_grounded, to_rotation = (
            _parse_transformer_winding_bus(bus_names_raw[1], n_phases)
        )
        if len(from_phase_list) != len(to_phase_list):
            raise ConversionError(
                f"OpenDSS transformer '{trafo_name}': the HV and LV windings "
                f"carry different phase counts ({len(from_phase_list)} vs "
                f"{len(to_phase_list)}); not supported."
            )

        if (
            from_bus_name not in bus_name_to_node_id
            or to_bus_name not in bus_name_to_node_id
        ):
            ret = dss.Transformers.Next()
            continue

        is_delta_from, is_delta_to = wdg_is_delta
        if is_delta_from:
            from_connection = WindingConnection.DELTA
        else:
            if not from_grounded:
                raise ConversionError(
                    f"OpenDSS transformer '{trafo_name}': the HV winding is an "
                    "ungrounded/impedance-grounded wye (explicit non-zero "
                    "neutral node); only solidly grounded wye and delta "
                    "windings are converted."
                )
            from_connection = WindingConnection.WYE_GROUNDED

        if is_delta_to:
            to_connection = WindingConnection.DELTA
        else:
            if not to_grounded:
                raise ConversionError(
                    f"OpenDSS transformer '{trafo_name}': the LV winding is an "
                    "ungrounded/impedance-grounded wye (explicit non-zero "
                    "neutral node); only solidly grounded wye and delta "
                    "windings are converted."
                )
            to_connection = WindingConnection.WYE_GROUNDED

        if is_delta_from != is_delta_to:
            # Dy / Yd pairing: OpenDSS's binary LeadLag toggle is the only
            # clock information available (verified empirically: `Lag` -> LV
            # lags HV by 30 deg; `Lead` -> LV leads HV by 30 deg).
            dss.Text.Command(f"? Transformer.{trafo_name}.leadlag")
            leadlag = dss.Text.Result().strip().lower()
            if leadlag in ("lag", "ansi", ""):
                base_shift_deg = 30.0
            elif leadlag in ("lead", "euro"):
                base_shift_deg = 330.0
            else:
                raise ConversionError(
                    f"OpenDSS transformer '{trafo_name}': unrecognized "
                    f"LeadLag value {leadlag!r}."
                )
        else:
            # Yy / Dd: `LeadLag` carries no clock information (clock 0
            # baseline; a further rotation is folded in below).
            base_shift_deg = 0.0

        # Fold any physical winding-bus rotation into the clock (see
        # `_cyclic_rotation_steps` and the "Cyclic winding-bus rotation"
        # section of the module CONTEXT.md for the sign convention, pinned
        # against a live OpenDSS solve): a cyclic rotation of the FROM/HV
        # winding's bus conductor order contributes +120 deg per step, a
        # TO/LV winding rotation -120 deg per step, REGARDLESS of the Dy/Yd
        # `LeadLag` baseline above. This reaches every clock of the pairing's
        # correct parity except the polarity-flip clocks {2, 6, 10}, which no
        # bus-connection rotation can express (`_cyclic_rotation_steps`
        # already rejects the non-cyclic phase-conductor order such a group
        # would require, so this converter never silently produces one).
        shift_deg = (
            base_shift_deg + 120.0 * from_rotation - 120.0 * to_rotation
        ) % 360.0

        tap_from, tap_to = wdg_tap
        ratio_magnitude = tap_from / tap_to

        trafo_id = _id.next()
        id_map["trafo"][trafo_name] = trafo_id

        tx_from_phases = phases_for(phase_mode, native=tuple(from_phase_list))
        tx_to_phases = phases_for(phase_mode, native=tuple(to_phase_list))

        branches.append(
            Transformer(
                id=trafo_id,
                name=trafo_name,
                from_node=bus_name_to_node_id[from_bus_name],
                to_node=bus_name_to_node_id[to_bus_name],
                from_phases=tx_from_phases,
                to_phases=tx_to_phases,
                s_rated_va=s_rated_va,
                u_rated_from_v=u_rated_from_v,
                u_rated_to_v=u_rated_to_v,
                from_connection=from_connection,
                to_connection=to_connection,
                series_resistance_ohm=r_lv_ohm,
                series_inductance_h=l_lv_h,
                magnetizing_conductance_s=g_m,
                magnetizing_inductance_h=l_m,
                tap=ComplexTap(ratio_magnitude=ratio_magnitude, shift_deg=shift_deg),
                harmonic_xr_constant=harmonic_xr_constant,
                provenance=_PROVENANCE,
            )
        )

        ret = dss.Transformers.Next()

    # ---------------------------------------------------------------------- #
    # 4. Vsources -> Source (Thevenin)                                        #
    # ---------------------------------------------------------------------- #
    appliances: list = []

    ret = dss.Vsources.First()
    while ret:
        vsrc_name = dss.Vsources.Name().lower()
        n_phases = dss.Vsources.Phases()
        basekv = dss.Vsources.BasekV()  # L-L kV
        pu = dss.Vsources.PU()
        angle_deg = dss.Vsources.AngleDeg()

        # Get bus name from circuit element
        dss.Circuit.SetActiveElement(f"Vsource.{vsrc_name}")
        bus_names_raw = dss.CktElement.BusNames()
        bus1_str = bus_names_raw[0].lower()  # first terminal (non-reference bus)
        src_bus_name, src_phases = _parse_bus_connection(bus1_str, n_phases)

        if src_bus_name not in bus_name_to_node_id:
            ret = dss.Vsources.Next()
            continue

        # Read R1/X1 via text command (Vsources API doesn't expose them directly)
        dss.Text.Command(f"? Vsource.{vsrc_name}.r1")
        r1_str = dss.Text.Result().strip()
        dss.Text.Command(f"? Vsource.{vsrc_name}.x1")
        x1_str = dss.Text.Result().strip()

        r1_ohm = float(r1_str) if r1_str else 0.0
        x1_ohm = float(x1_str) if x1_str else 0.0
        r_s, l_s = thevenin_from_z(r1_ohm, x1_ohm, two_pi_f0)

        # A non-negligible Thevenin impedance is silently UNUSED under the
        # default ideal slack: `solve_power_flow`/`solve_harmonic`'s
        # `slack="ideal"` pins the bus voltage exactly at `u_ref_v` regardless
        # of R1/X1 (only `slack="norton"` folds the source's own Norton shunt
        # into Y_eff and lets its finite impedance actually load the bus).
        # Warn once per Vsource so a converted grid is not silently solved as
        # an ideal source when the DSS circuit specified a real one; the
        # threshold (1e-4 Ohm) sits comfortably above the near-zero (<=1e-6
        # Ohm) placeholder impedances this test suite's own "ideal slack"
        # fixtures use, and well below a physically meaningful source
        # impedance (DSS's own un-set Vsource default is ~0.02+j0.08 Ohm).
        if r1_ohm > 1.0e-4 or x1_ohm > 1.0e-4:
            _logger.warning(
                "OpenDSS Vsource '%s' has a non-negligible source impedance "
                '(R1=%.6g Ohm, X1=%.6g Ohm). pgml\'s default slack="ideal" '
                "(solve_power_flow/solve_harmonic) pins this bus's voltage "
                "exactly at u_ref_v regardless of R1/X1 -- pass "
                'slack="norton" instead to reproduce the finite-impedance '
                "Thevenin source.",
                vsrc_name,
                r1_ohm,
                x1_ohm,
            )

        # BasekV is the L-L nominal for a >=3-phase Vsource, but for a
        # phases=1 Vsource OpenDSS uses it DIRECTLY as the single
        # conductor-pair EMF (no internal sqrt(3) either way) -- see the
        # module docstring's "Vsource" section for the empirical basis. This
        # single formula is correct for both cases: it simply reproduces
        # whatever magnitude OpenDSS itself solves for at that bus.
        u_ref_v = basekv * pu * 1_000.0  # V

        if id_map["slack_v_complex"] is None:
            id_map["slack_v_complex"] = u_ref_v * complex(
                math.cos(math.radians(angle_deg)),
                math.sin(math.radians(angle_deg)),
            )

        src_id = _id.next()
        id_map["vsource"][vsrc_name] = src_id

        native_src_phases = tuple(src_phases)
        appliances.append(
            build_source(
                id=src_id,
                name=vsrc_name,
                node=bus_name_to_node_id[src_bus_name],
                mode=phase_mode,
                u_ref_v=u_ref_v,
                u_angle_deg=angle_deg,
                r_ohm=r_s,
                l_h=l_s,
                native_phases=native_src_phases,
            )
        )

        ret = dss.Vsources.Next()

    # ---------------------------------------------------------------------- #
    # 5. Loads                                                                #
    # ---------------------------------------------------------------------- #
    ret = dss.Loads.First()
    while ret:
        load_name = dss.Loads.Name().lower()

        # Activate load element to get bus connection. `Loads.Phases()` is
        # the number of PHASE conductors (never includes the implicit/explicit
        # return conductor OpenDSS auto-appends to a WYE load's single
        # terminal); `_parse_appliance_bus_connection` resolves the return
        # conductor via `CktElement.NodeOrder()` (DSS's own bus-string
        # default-padding logic) rather than re-parsing the bus string.
        dss.Circuit.SetActiveElement(f"Load.{load_name}")
        n_phases = dss.CktElement.NumPhases()
        load_bus_name, load_phases, explicit_return = _parse_appliance_bus_connection(
            dss, n_phases
        )

        if load_bus_name not in bus_name_to_node_id:
            ret = dss.Loads.Next()
            continue

        p_w = dss.Loads.kW() * 1_000.0  # W total
        q_var = dss.Loads.kvar() * 1_000.0  # VAR total

        # Read load connection: IsDelta() returns True for delta, False for wye.
        # OpenDSS single-phase loads are always wye (L-N, two-conductor) per the
        # NeutralRules convention; delta requires at least 2 phases.
        is_delta = bool(dss.Loads.IsDelta())
        # DELTA needs >=2 phases (schema validator enforces this). A 1-phase load
        # flagged delta is a misconfiguration; treat it as WYE rather than let the
        # schema raise an error that does not point back to the converter.
        conn = (
            WindingConnection.DELTA
            if (is_delta and n_phases >= 2)
            else WindingConnection.WYE
        )
        if is_delta and n_phases < 2:
            _logger.info(
                "OpenDSS load %s is flagged delta but has a single phase; "
                "treating it as WYE (delta requires at least 2 phases).",
                load_name,
            )

        load_model, zip_coefficients = _resolve_load_model(dss, load_name, kind="Load")

        load_id = _id.next()
        id_map["load"][load_name] = load_id

        native_load_phases = tuple(load_phases)

        if phase_mode is PhaseMode.SINGLE_PHASE_EQUIV:
            # SINGLE_PHASE_EQUIV: collapse to phases=(A,), no connection stored.
            # Delta loads cannot be represented in 1-phase equivalent; log a note.
            if is_delta:
                _logger.info(
                    "OpenDSS delta load %s collapsed to single-phase equivalent "
                    "(SINGLE_PHASE_EQUIV cannot represent delta connection); "
                    "use THREE_PHASE to preserve the delta topology.",
                    load_name,
                )
            appliances.append(
                build_load(
                    id=load_id,
                    name=load_name,
                    node=bus_name_to_node_id[load_bus_name],
                    mode=phase_mode,
                    p_total_w=p_w,
                    q_total_var=q_var,
                    load_model=load_model,
                    zip_coefficients=zip_coefficients,
                )
            )
        else:
            # THREE_PHASE: emit real phases + connection.
            # Multi-phase balanced DSS loads (kW/kvar are the total, divided
            # equally by DSS internally) have no per-phase split; leave
            # p_per_phase_w=None so the assembly's symmetric/auto mode splits
            # the total equally.  Genuine per-phase imbalance in OpenDSS is
            # expressed via separate 1-phase Load objects (which naturally end
            # up on distinct phase rows through their bus suffix).
            load_obj = build_load(
                id=load_id,
                name=load_name,
                node=bus_name_to_node_id[load_bus_name],
                mode=phase_mode,
                p_total_w=p_w,
                q_total_var=q_var,
                connection=conn,
                native_phases=native_load_phases,
                load_model=load_model,
                zip_coefficients=zip_coefficients,
            )
            if not is_delta:
                rp = _resolve_wye_return_path(
                    bus_name=load_bus_name,
                    bus_phases=bus_phases,
                    explicit_return=explicit_return,
                )
                if rp != "auto":
                    load_obj.return_path = rp
            appliances.append(load_obj)

        ret = dss.Loads.Next()

    # ---------------------------------------------------------------------- #
    # 6. Capacitors -> ShuntAppliance (WYE, solidly grounded only)             #
    # ---------------------------------------------------------------------- #
    # OpenDSS's `Capacitor` is a 2-terminal element whose 2nd terminal defaults
    # to the SAME bus, every conductor tied to the universal ground reference
    # (node 0) -- exactly pgml's WYE `ShuntAppliance` (a per-phase-to-GROUND shunt
    # anchored at one node). A DELTA capacitor collapses to a SINGLE terminal in
    # OpenDSS (phase-to-phase legs, no ground return): it converts to a DELTA
    # `ShuntAppliance` whose per-leg capacitance is OpenDSS's own resolved `Cuf`
    # (per-leg for a delta bank, verified live). A non-grounded / genuine 2-bus
    # terminal-2 reference is still skipped with a warning (unrepresentable).
    ret = dss.Capacitors.First()
    while ret:
        cap_name = dss.Capacitors.Name().lower()
        dss.Circuit.SetActiveElement(f"Capacitor.{cap_name}")
        n_phases = dss.CktElement.NumPhases()
        bus_name = dss.CktElement.BusNames()[0].split(".")[0].lower()

        if bus_name not in bus_name_to_node_id:
            ret = dss.Capacitors.Next()
            continue

        is_delta = _dss_query(dss, "Capacitor", cap_name, "conn").lower() == "delta"
        if is_delta:
            phase_list = _shunt_phase_conductors(dss, n_phases)
            connection = WindingConnection.DELTA
            if phase_list is None or len(phase_list) < 2:
                _logger.warning(
                    "OpenDSS Capacitor '%s' is DELTA-connected but does not present "
                    "the expected >=2 phase-to-phase legs -- this element is NOT "
                    "converted.",
                    cap_name,
                )
                ret = dss.Capacitors.Next()
                continue
        else:
            phase_list = _grounded_shunt_phases(dss, n_phases)
            connection = WindingConnection.WYE
            if phase_list is None:
                _logger.warning(
                    "OpenDSS Capacitor '%s' is not a solidly-grounded WYE shunt "
                    "(a non-grounded/2-bus terminal-2 reference) -- pgml's "
                    "ShuntAppliance models a phase-to-ground or phase-to-phase "
                    "shunt only; this element is NOT converted.",
                    cap_name,
                )
                ret = dss.Capacitors.Next()
                continue

        dss.Text.Command(f"? Capacitor.{cap_name}.Cuf")
        cuf_per_step = _parse_dss_float_array(dss.Text.Result())
        states = list(dss.Capacitors.States())
        c_per_phase_f = sum(cuf for cuf, on in zip(cuf_per_step, states) if on) * 1.0e-6

        cap_id = _id.next()
        id_map["capacitor"][cap_name] = cap_id
        appliances.append(
            _build_shunt_appliance(
                id=cap_id,
                name=cap_name,
                node=bus_name_to_node_id[bus_name],
                mode=phase_mode,
                native_phases=tuple(phase_list),
                conductance_s=0.0,
                capacitance_f=c_per_phase_f,
                connection=connection,
            )
        )
        ret = dss.Capacitors.Next()

    # ---------------------------------------------------------------------- #
    # 7. Reactors -> ShuntAppliance (WYE, solidly grounded only)               #
    # ---------------------------------------------------------------------- #
    # A shunt Reactor is a genuine series R+X branch, but OpenDSS's OWN
    # default (2nd terminal grounded) makes a series impedance-to-GROUND
    # electrically identical to a shunt ADMITTANCE Y=1/(R+jX) to ground (there
    # is no "series vs shunt" distinction when one end is the fixed-zero
    # ground reference) -- so it converts to `ShuntAppliance` the same way a
    # Capacitor does, with G=Re(Y), C=Im(Y)/(2*pi*f0). This ALSO covers the
    # "grounding reactor" idiom used to tie a floating neutral conductor to
    # ground (`bus1=busname.k.0`, phases=1): OpenDSS resolves the single bus
    # string into `bus1=busname.k` / `bus2=busname.0` because no `bus2=` was
    # given, which `_grounded_shunt_phases` recognises the same way.
    #
    # CAVEAT (documented, not modeled): the C-based susceptance model
    # (`B(h)=2*pi*h*f0*C`) is exact ONLY at h=1 (fundamental) -- a genuinely
    # inductive reactor's true susceptance `-1/(h*2*pi*f0*L)` DECREASES with
    # frequency, while the fixed-C model INCREASES linearly with h (the wrong
    # trend). The schema has no inductive shunt-to-ground primitive (its
    # `ShuntReactor` branch type stores conductance + capacitance only, no
    # inductance), so this is the best available representation; load-flow
    # (fundamental-only) studies are exact, harmonic studies are not.
    ret = dss.Reactors.First()
    while ret:
        reactor_name = dss.Reactors.Name().lower()
        dss.Circuit.SetActiveElement(f"Reactor.{reactor_name}")
        n_phases = dss.CktElement.NumPhases()
        bus_name = dss.CktElement.BusNames()[0].split(".")[0].lower()

        if bus_name not in bus_name_to_node_id:
            ret = dss.Reactors.Next()
            continue

        is_delta = _dss_query(dss, "Reactor", reactor_name, "conn").lower() == "delta"
        if is_delta:
            phase_list = _shunt_phase_conductors(dss, n_phases)
            connection = WindingConnection.DELTA
            if phase_list is None or len(phase_list) < 2:
                _logger.warning(
                    "OpenDSS Reactor '%s' is DELTA-connected but does not present "
                    "the expected >=2 phase-to-phase legs -- this element is NOT "
                    "converted.",
                    reactor_name,
                )
                ret = dss.Reactors.Next()
                continue
        else:
            phase_list = _grounded_shunt_phases(dss, n_phases)
            connection = WindingConnection.WYE
            if phase_list is None:
                _logger.warning(
                    "OpenDSS Reactor '%s' is not a solidly-grounded WYE shunt "
                    "(a non-grounded/2-bus terminal-2 reference) -- pgml's "
                    "ShuntAppliance models a phase-to-ground or phase-to-phase "
                    "shunt only; this element is NOT converted.",
                    reactor_name,
                )
                ret = dss.Reactors.Next()
                continue

        r_ohm = float(dss.Reactors.R())
        x_ohm = float(dss.Reactors.X())
        r_mat = list(dss.Reactors.Rmatrix())
        x_mat = list(dss.Reactors.Xmatrix())
        if n_phases > 1 and (
            len(r_mat) == n_phases * n_phases or len(x_mat) == n_phases * n_phases
        ):
            _logger.warning(
                "OpenDSS Reactor '%s' specifies an explicit coupled "
                "Rmatrix/Xmatrix; only an uncoupled (scalar R/X, diagonal) "
                "shunt reactor is converted -- this element is NOT converted.",
                reactor_name,
            )
            ret = dss.Reactors.Next()
            continue

        z_ohm = complex(r_ohm, x_ohm)
        if z_ohm == 0.0:
            g_scalar, c_scalar = 0.0, 0.0
        else:
            y_s = 1.0 / z_ohm
            g_scalar = y_s.real
            c_scalar = y_s.imag / two_pi_f0

        reactor_id = _id.next()
        id_map["reactor"][reactor_name] = reactor_id
        appliances.append(
            _build_shunt_appliance(
                id=reactor_id,
                name=reactor_name,
                node=bus_name_to_node_id[bus_name],
                mode=phase_mode,
                native_phases=tuple(phase_list),
                conductance_s=g_scalar,
                capacitance_f=c_scalar,
                connection=connection,
            )
        )
        ret = dss.Reactors.Next()

    # ---------------------------------------------------------------------- #
    # 8. Generators -> Generator (generation-positive PQ injection)           #
    # ---------------------------------------------------------------------- #
    ret = dss.Generators.First()
    while ret:
        gen_name = dss.Generators.Name().lower()
        dss.Circuit.SetActiveElement(f"Generator.{gen_name}")
        n_phases = dss.CktElement.NumPhases()
        gen_bus_name, gen_phases, explicit_return = _parse_appliance_bus_connection(
            dss, n_phases
        )

        if gen_bus_name not in bus_name_to_node_id:
            ret = dss.Generators.Next()
            continue

        is_delta = bool(dss.Generators.IsDelta())
        conn = (
            WindingConnection.DELTA
            if (is_delta and n_phases >= 2)
            else WindingConnection.WYE
        )
        if is_delta and n_phases < 2:
            _logger.info(
                "OpenDSS generator %s is flagged delta but has a single "
                "phase; treating it as WYE (delta requires at least 2 "
                "phases).",
                gen_name,
            )

        p_w = float(dss.Generators.kW()) * 1_000.0
        q_var = float(dss.Generators.kvar()) * 1_000.0

        gen_id = _id.next()
        id_map["generator"][gen_name] = gen_id
        gen_obj = build_generator(
            id=gen_id,
            name=gen_name,
            node=bus_name_to_node_id[gen_bus_name],
            mode=phase_mode,
            p_total_w=p_w,
            q_total_var=q_var,
            connection=conn,
            native_phases=tuple(gen_phases),
        )
        if phase_mode is PhaseMode.THREE_PHASE and not is_delta:
            rp = _resolve_wye_return_path(
                bus_name=gen_bus_name,
                bus_phases=bus_phases,
                explicit_return=explicit_return,
            )
            if rp != "auto":
                gen_obj.return_path = rp
        appliances.append(gen_obj)
        ret = dss.Generators.Next()

    # ---------------------------------------------------------------------- #
    # 9. PVSystems -> Generator (consumer_type="pv")                          #
    # ---------------------------------------------------------------------- #
    # `PVsystems.kW()`/`.kvar()` report the PRESENT solved output (after the
    # Pmpp/irradiance/pf/kVA-limit derating OpenDSS itself applies), so no
    # extra derating arithmetic is needed here -- reading the present output
    # is the same "snapshot" convention used for a Storage element's present
    # kW/kvar below.
    ret = dss.PVsystems.First()
    while ret:
        pv_name = dss.PVsystems.Name().lower()
        dss.Circuit.SetActiveElement(f"PVSystem.{pv_name}")
        n_phases = dss.CktElement.NumPhases()
        pv_bus_name, pv_phases, explicit_return = _parse_appliance_bus_connection(
            dss, n_phases
        )

        if pv_bus_name not in bus_name_to_node_id:
            ret = dss.PVsystems.Next()
            continue

        is_delta = _dss_query(dss, "PVSystem", pv_name, "conn").lower() == "delta"
        conn = (
            WindingConnection.DELTA
            if (is_delta and n_phases >= 2)
            else WindingConnection.WYE
        )

        p_w = float(dss.PVsystems.kW()) * 1_000.0
        q_var = float(dss.PVsystems.kvar()) * 1_000.0

        pv_id = _id.next()
        id_map["pvsystem"][pv_name] = pv_id
        pv_obj = build_generator(
            id=pv_id,
            name=pv_name,
            node=bus_name_to_node_id[pv_bus_name],
            mode=phase_mode,
            p_total_w=p_w,
            q_total_var=q_var,
            connection=conn,
            native_phases=tuple(pv_phases),
            consumer_type=ConsumerType.PV,
        )
        if phase_mode is PhaseMode.THREE_PHASE and not is_delta:
            rp = _resolve_wye_return_path(
                bus_name=pv_bus_name,
                bus_phases=bus_phases,
                explicit_return=explicit_return,
            )
            if rp != "auto":
                pv_obj.return_path = rp
        appliances.append(pv_obj)
        ret = dss.PVsystems.Next()

    # ---------------------------------------------------------------------- #
    # 10. Storages -> Storage (signed, discharge-positive PQ injection)       #
    # ---------------------------------------------------------------------- #
    # `Storages` has no `build_generator`-style helper in `convert._common`;
    # the schema object is constructed directly here, mirroring the same
    # mode/connection/native_phases decisions `build_load`/`build_generator`
    # make. `kW`/`kvar` are read via text query (present output; DSS's own
    # sign convention already matches the schema's discharge-positive /
    # charge-negative one -- verified empirically: `state=DISCHARGING` reads a
    # positive `kW`, `state=CHARGING` reads a negative one).
    ret = dss.Storages.First()
    while ret:
        bat_name = dss.Storages.Name().lower()
        dss.Circuit.SetActiveElement(f"Storage.{bat_name}")
        n_phases = dss.CktElement.NumPhases()
        bat_bus_name, bat_phases, explicit_return = _parse_appliance_bus_connection(
            dss, n_phases
        )

        if bat_bus_name not in bus_name_to_node_id:
            ret = dss.Storages.Next()
            continue

        is_delta = _dss_query(dss, "Storage", bat_name, "conn").lower() == "delta"
        conn = (
            WindingConnection.DELTA
            if (is_delta and n_phases >= 2)
            else WindingConnection.WYE
        )

        p_w = float(_dss_query(dss, "Storage", bat_name, "kW") or 0.0) * 1_000.0
        q_var = float(_dss_query(dss, "Storage", bat_name, "kvar") or 0.0) * 1_000.0
        kwh_rated = float(_dss_query(dss, "Storage", bat_name, "kWhrated") or 0.0)
        kw_rated = float(_dss_query(dss, "Storage", bat_name, "kWrated") or 0.0)
        pct_stored = float(_dss_query(dss, "Storage", bat_name, "%stored") or 0.0)
        pct_reserve = float(_dss_query(dss, "Storage", bat_name, "%reserve") or 0.0)
        pct_eff_charge = float(
            _dss_query(dss, "Storage", bat_name, "%EffCharge") or 100.0
        )
        pct_eff_discharge = float(
            _dss_query(dss, "Storage", bat_name, "%EffDischarge") or 100.0
        )

        bat_id = _id.next()
        id_map["storage"][bat_name] = bat_id

        storage_kwargs: dict[str, Any] = {
            "id": bat_id,
            "name": bat_name,
            "node": bus_name_to_node_id[bat_bus_name],
            "p_nom_w": p_w,
            "q_nom_var": q_var,
            "soc_min": max(0.0, min(1.0, pct_reserve / 100.0)),
            "efficiency_charge": max(1.0e-6, min(1.0, pct_eff_charge / 100.0)),
            "efficiency_discharge": max(1.0e-6, min(1.0, pct_eff_discharge / 100.0)),
            "consumer_type": ConsumerType.BATTERY,
        }
        if kwh_rated > 0.0:
            storage_kwargs["energy_capacity_wh"] = kwh_rated * 1_000.0
            storage_kwargs["soc"] = max(0.0, min(1.0, pct_stored / 100.0))
        if kw_rated > 0.0:
            storage_kwargs["p_rated_w"] = kw_rated * 1_000.0

        if phase_mode is PhaseMode.SINGLE_PHASE_EQUIV:
            storage_kwargs["phases"] = (Phase.A,)
        else:
            storage_kwargs["phases"] = phases_for(phase_mode, native=tuple(bat_phases))
            storage_kwargs["connection"] = conn
            if not is_delta:
                rp = _resolve_wye_return_path(
                    bus_name=bat_bus_name,
                    bus_phases=bus_phases,
                    explicit_return=explicit_return,
                )
                if rp != "auto":
                    storage_kwargs["return_path"] = rp

        appliances.append(Storage(**storage_kwargs))
        ret = dss.Storages.Next()

    # ---------------------------------------------------------------------- #
    # 11. Elements the converter does NOT read: fail loud, never silently   #
    #     wrong. Every DSS circuit element (`Circuit.AllElementNames()`,     #
    #     `"ClassName.elementname"`) whose class is not one of the ones      #
    #     handled above triggers a WARNING naming the kind and count --      #
    #     e.g. `Isource`, `Monitor`, `EnergyMeter`, `RegControl`, `CapControl`,
    #     `InvControl`, `StorageController`, `Fault`, `Relay`, `Recloser`,   #
    #     `Fuse`, `Sensor`. A 3-winding `Transformer` is NOT counted here -- #
    #     it already raises `ConversionError` above (a hard scope boundary, #
    #     not a silent drop).                                               #
    # ---------------------------------------------------------------------- #
    _handled_dss_classes = {
        "vsource",
        "line",
        "load",
        "transformer",
        "capacitor",
        "reactor",
        "generator",
        "pvsystem",
        "storage",
    }
    _dropped_counts: dict[str, int] = {}
    for elt_name in dss.Circuit.AllElementNames():
        cls = elt_name.split(".", 1)[0].lower()
        if cls not in _handled_dss_classes:
            _dropped_counts[cls] = _dropped_counts.get(cls, 0) + 1
    warn_dropped_elements(_logger, "OpenDSS", _dropped_counts)

    description = f"Imported from OpenDSS circuit (f0={f0_hz} Hz). " + (
        "Single-phase positive-sequence equivalent."
        if phase_mode is PhaseMode.SINGLE_PHASE_EQUIV
        else "Three-phase (abc) with real DSS phases, n×n line matrices, and load connections."
    )
    grid = Grid(
        base_frequency_hz=f0_hz,
        nodes=nodes,
        branches=branches,
        appliances=appliances,
        metadata=make_metadata(name="opendss_import", description=description),
    )
    return grid, id_map


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _phase_num_to_enum(phase_num: int) -> Phase:
    """Map an OpenDSS bus-node index (1, 2, 3, 4) to our Phase enum (A, B, C, N).

    OpenDSS numbers bus conductors 1=A, 2=B, 3=C; index 4 on a four-wire bus
    (``bus1=Bus.1.2.3.4``) is the explicit neutral conductor. Index 0 is the
    grounded reference node — it never appears in ``YNodeOrder`` and must be
    dropped from connection strings by the caller, not mapped to a phase row.
    Any other index has no phase-domain equivalent and raises.
    """
    _MAP: dict[int, Phase] = {1: Phase.A, 2: Phase.B, 3: Phase.C, 4: Phase.N}
    try:
        return _MAP[phase_num]
    except KeyError:
        raise ConversionError(
            f"OpenDSS bus-node index {phase_num} has no phase-domain mapping "
            "(expected 1=A, 2=B, 3=C, 4=N; 0 is the grounded reference)."
        ) from None


def _parse_bus_connection(bus_str: str, n_phases: int) -> tuple[str, list[Phase]]:
    """Parse a DSS bus connection string ``'busname.1.2.3'`` into ``(bus_name, [phases])``.

    Node index ``0`` (a conductor tied to the grounded reference, e.g. the
    return conductor of ``"bus0.1.0"``) is dropped: the ground return is
    implicit in the grid model's grounded-wye convention and has no node-phase
    row.

    Parameters
    ----------
    bus_str:
        Lowercase DSS bus string, e.g. ``"bus0.1"`` or ``"bus0.1.2.3"``.
    n_phases:
        Number of phases for the element (used to default phases if not explicit).

    Returns
    -------
    (bus_name, [Phase, ...])
        ``bus_name``: lowercase bus name (without phase suffixes).
        ``phases``: ordered list of Phase enum values.
    """
    parts = bus_str.split(".")
    bus_name = parts[0]
    if len(parts) > 1:
        phase_nums = [int(p) for p in parts[1:] if p.isdigit()]
        phases = [_phase_num_to_enum(pn) for pn in phase_nums if pn != 0]
    else:
        # Default: phases 1..n_phases
        phases = [_phase_num_to_enum(i + 1) for i in range(n_phases)]
    return bus_name, phases


def _cyclic_rotation_steps(phase_nums: list[int], n_phases: int, bus_str: str) -> int:
    """Classify a winding's phase-conductor order as identity or a cyclic rotation.

    Returns ``r`` such that ``phase_nums[k] == ((k + r) % n_phases) + 1`` for
    every ``k`` -- e.g. ``[2, 3, 1]`` is ``r=1`` (the conductor order is
    (A, B, C) rotated one step: winding terminal 1 lands on the bus's phase
    B conductor, terminal 2 on C, terminal 3 on A).

    Only a genuine cyclic rotation of a 3-phase winding is realisable as a
    constant vector-group clock shift -- each step contributes exactly +-4
    clock steps (+-120 deg), verified against a live OpenDSS solve (see the
    module CONTEXT.md "Cyclic winding-bus rotation" section) and matching
    :mod:`pgml.assembly._transformer`'s own internal cyclic-permutation clock
    mechanism (``_cyclic_power``). Any OTHER permutation (e.g. swapping two
    conductors) reverses the phase-rotation sequence (A->C->B->A instead of
    A->B->C->A), which is a genuinely different physical winding and cannot
    be expressed by any bus-conductor rotation -- it raises rather than
    silently producing a wrong clock. Non-3-phase windings must use the
    identity order; the rotation concept is specific to the 3-phase A/B/C
    cyclic group.
    """
    if n_phases != 3:
        if phase_nums != list(range(1, n_phases + 1)):
            raise ConversionError(
                f"transformer winding bus {bus_str!r}: a non-identity phase "
                f"conductor order is only supported for 3-phase windings "
                f"(got {n_phases} phases)."
            )
        return 0
    for r in range(3):
        if all(phase_nums[k] == ((k + r) % 3) + 1 for k in range(3)):
            return r
    raise ConversionError(
        f"transformer winding bus {bus_str!r}: phase conductor order "
        f"{phase_nums} is neither the identity nor a cyclic rotation of "
        "(1, 2, 3) -- a non-cyclic permutation reverses the phase-rotation "
        "sequence and cannot be expressed as a constant vector-group clock "
        "shift; not supported."
    )


def _parse_transformer_winding_bus(
    bus_str: str, n_phases: int
) -> tuple[str, list[Phase], bool, int]:
    """Parse a transformer winding bus string into ``(bus_name, phases, grounded, rotation)``.

    ``phases`` is ALWAYS the canonical ``(Phase.A, Phase.B, ...)`` order for
    ``n_phases`` conductors -- never the raw DSS conductor-index order --
    because the phase-domain transformer stamp
    (:func:`pgml.assembly._transformer.block_incidence`) assumes winding
    position ``k`` IS bus phase ``k``. Any physical rotation of the bus
    connection (e.g. ``bus=lv.2.3.1.0``) is reported separately via
    ``rotation`` so the caller folds it into the vector-group clock instead
    of baking it into the row order (see :func:`_cyclic_rotation_steps` and
    the module CONTEXT.md).

    ``grounded`` applies only to a WYE winding (a DELTA winding has no neutral
    and the caller ignores it): ``True`` when the winding's neutral is
    solidly tied to the system ground reference node ``0`` -- either
    implicitly, via OpenDSS's shorthand-bus rule (a bus string with no
    explicit ``(n_phases+1)``-th conductor auto-grounds the missing neutral),
    or explicitly via a trailing ``.0``. An explicit NON-ZERO
    ``(n_phases+1)``-th conductor (e.g. ``.4``) creates a genuine
    floating/impedance-grounded neutral node (governed by the winding's
    ``Rneut``/``Xneut``), which this converter does not model (the caller
    raises :class:`~pgml.errors.ConversionError`).

    Parameters
    ----------
    bus_str:
        Lowercase DSS bus string for one transformer terminal, e.g.
        ``"hv.1.2.3"``, ``"lv.1.2.3.0"``, or the rotated ``"lv.2.3.1.0"``.
    n_phases:
        Number of phase conductors on the transformer (``CktElement.NumPhases()``).
    """
    parts = bus_str.split(".")
    bus_name = parts[0]
    node_nums = [int(p) for p in parts[1:] if p.isdigit()]
    if not node_nums:
        node_nums = list(range(1, n_phases + 1))
    phase_nums = node_nums[:n_phases]
    if len(phase_nums) != n_phases:
        raise ConversionError(
            f"transformer winding bus {bus_str!r}: expected {n_phases} phase "
            f"conductors, found {len(phase_nums)}."
        )
    rotation = _cyclic_rotation_steps(phase_nums, n_phases, bus_str)
    phases = [_phase_num_to_enum(i + 1) for i in range(n_phases)]
    grounded = len(node_nums) <= n_phases or node_nums[n_phases] == 0
    return bus_name, phases, grounded, rotation


def _flat_to_matrix(flat: list[float], n: int) -> list[list[float]]:
    """Reshape a flat list of n*n values (row-major) into an n x n list-of-lists."""
    return [[flat[i * n + j] for j in range(n)] for i in range(n)]


def _positive_sequence_scalar(mat_per_m_flat: list[float], n: int) -> float:
    """Reduce a flat n x n per-length matrix to its positive-sequence scalar.

    For a genuinely single-conductor line (``n == 1``) this is exactly the
    ``[0][0]`` entry (byte-identical to the historical converter). For a
    coupled multi-phase line, the positive-sequence quantity is
    ``Z1 = Z_self - Z_mutual`` (mean diagonal minus mean off-diagonal) --
    using the self entry alone (the historical bug) ignores the mutual
    coupling and overstates the positive-sequence impedance. The identical
    reduction applied to the Maxwell C matrix gives ``C1 = C_self - C_mutual``;
    C's off-diagonal entries are NEGATIVE (mutual coupling reduces net
    charge), so subtracting them INCREASES C1 above C_self, which is the
    physically correct direction (more shunt charging capacity in the
    positive-sequence than any single conductor sees on its own).
    """
    if n == 1:
        return mat_per_m_flat[0]
    mat = _flat_to_matrix(mat_per_m_flat, n)
    diag_mean = sum(mat[i][i] for i in range(n)) / n
    off_vals = [mat[i][j] for i in range(n) for j in range(n) if i != j]
    off_mean = sum(off_vals) / len(off_vals)
    return diag_mean - off_mean


def _parse_dss_float_array(text: str) -> list[float]:
    """Parse an OpenDSS ``? Class.name.Property`` array-valued query result.

    OpenDSS renders an array-valued property as e.g. ``"[ 1.2 3.4 5.6]"``
    (whitespace-separated, no reliable comma delimiter) -- this strips the
    brackets and splits on any whitespace/comma.
    """
    cleaned = text.strip().strip("[]").replace(",", " ")
    return [float(tok) for tok in cleaned.split()]


def _dss_query(dss: Any, class_name: str, elem_name: str, prop: str) -> str:
    """Read a single DSS element property via the ``? Class.name.Property`` text API.

    Used for properties not exposed by a dedicated ``opendssdirect`` accessor
    (e.g. ``PVSystem``/``Storage`` connection and rating properties).
    """
    dss.Text.Command(f"? {class_name}.{elem_name}.{prop}")
    return dss.Text.Result().strip()


def _parse_appliance_bus_connection(
    dss: Any, n_phases: int
) -> tuple[str, list[Phase], Optional[Phase]]:
    """Resolve a single-terminal PC element's (Load/Generator/Storage/PVSystem)
    bus connection using ``CktElement.NodeOrder()`` -- OpenDSS's OWN resolved
    conductor/return assignment (including its bus-string default-padding
    rules) -- rather than re-parsing the bus string.

    The active ``CktElement`` must already be the element in question.
    ``Phases=`` (``NumPhases()``) is the number of PHASE conductors; OpenDSS
    silently appends exactly ONE extra return conductor beyond that count for
    a (default) WYE connection -- defaulting to the grounded reference node 0
    when not given explicitly, or the caller's own explicit
    ``(n_phases+1)``-th bus-string suffix (typically ``4``, an explicit
    neutral tie) otherwise. A DELTA connection has NO such extra conductor
    (phase-to-phase only; ``NodeOrder()`` is exactly ``n_phases`` long).

    Returns ``(bus_name, phases, explicit_return)``: ``phases`` are the
    ``n_phases`` phase-conductor rows (canonical, positional order);
    ``explicit_return`` is ``None`` when the appliance's neutral point
    returns to solid GROUND (node 0, whether via an explicit ``.0`` or
    OpenDSS's own default), or the :class:`Phase` the return conductor is
    tied to (e.g. ``Phase.N`` for a ``.4`` suffix) when it is NOT simply
    ground. :func:`_resolve_wye_return_path` turns this into the appliance's
    :attr:`~pgml.schemas.grid_schema.InjectionAppliance.return_path` so pgml's
    per-node WYE/neutral routing faithfully reproduces OpenDSS's per-element
    return-conductor choice even when the node ALSO carries a ``Phase.N`` row
    from another element.
    """
    bus_name = dss.CktElement.BusNames()[0].split(".")[0].lower()
    node_order = [int(n) for n in dss.CktElement.NodeOrder()]
    phase_nums = node_order[:n_phases]
    phases = [_phase_num_to_enum(pn) for pn in phase_nums]
    explicit_return: Optional[Phase] = None
    if len(node_order) > n_phases:
        return_num = node_order[n_phases]
        if return_num != 0:
            explicit_return = _phase_num_to_enum(return_num)
    return bus_name, phases, explicit_return


def _resolve_wye_return_path(
    *,
    bus_name: str,
    bus_phases: dict[str, list[Phase]],
    explicit_return: Optional[Phase],
) -> str:
    """Return the WYE ``return_path`` (``InjectionAppliance.return_path``) faithful to
    OpenDSS's own per-element return-conductor choice.

    OpenDSS resolves each WYE PC element's return conductor independently: a
    ``.1.2.3.4`` suffix ties the return to the bus's 4th (``Phase.N``) conductor,
    while ``.1.2.3`` (or an explicit ``.0``) returns to TRUE GROUND. pgml's WYE
    incidence is a NODE property (``assembly._incidence.group_appliances``: return
    through the node's ``Phase.N`` row whenever the node carries one), so the
    per-appliance ``return_path`` reproduces OpenDSS's per-element decision even on a
    bus that carries a ``Phase.N`` row from another element:

    - an explicit ``Phase.N`` tie (``.4``) -> ``'neutral'`` (require the neutral);
    - solidly grounded (no explicit tie) on a node that ALSO carries ``Phase.N``
      -> ``'ground'`` (the return stays at true ground despite the shared neutral —
      previously an inexpressible, warned mismatch);
    - otherwise (a 3-wire node, or a non-neutral explicit conductor) -> ``'auto'``,
      which reduces to ground when the node has no ``Phase.N`` and matches pgml's
      historical node-level routing when it does.
    """
    if explicit_return == Phase.N:
        return "neutral"
    if explicit_return is None and Phase.N in bus_phases.get(bus_name, []):
        return "ground"
    return "auto"


def _grounded_shunt_phases(dss: Any, n_phases: int) -> Optional[list[Phase]]:
    """Return the phase list for a solidly-grounded WYE shunt (Capacitor/Reactor).

    Both elements are OpenDSS 2-terminal branches whose 2nd terminal defaults
    to the grounded reference; node/conductor index 0 is OpenDSS's single
    UNIVERSAL ground node (never bus-scoped), so checking every terminal-2
    conductor is 0 is sufficient regardless of which bus name was used to
    express it -- including the "grounding reactor" idiom
    (``bus1=busname.k.0``, phases=1), where OpenDSS folds the extra conductor
    into an auto-generated ``bus2=busname.0`` because only one bus string was
    given.

    Returns ``None`` (not convertible as a shunt-to-ground) when: the element
    is DELTA-connected (OpenDSS collapses it to a SINGLE terminal with no
    return path at all -- ``NumTerminals() == 1``, phase-to-phase legs only,
    which ``ShuntAppliance`` cannot represent), or terminal 2 is a genuine
    2-bus reference / not solidly grounded.
    """
    if dss.CktElement.NumTerminals() != 2:
        return None
    node_order = [int(n) for n in dss.CktElement.NodeOrder()]
    if len(node_order) != 2 * n_phases:
        return None
    from_conductors = node_order[:n_phases]
    return_conductors = node_order[n_phases:]
    if any(c != 0 for c in return_conductors):
        return None
    return [_phase_num_to_enum(c) for c in from_conductors]


def _shunt_phase_conductors(dss: Any, n_phases: int) -> Optional[list[Phase]]:
    """Phase list of a single-terminal (DELTA) shunt bank from ``NodeOrder()``.

    A DELTA-connected OpenDSS ``Capacitor``/``Reactor`` collapses to ONE terminal
    (phase-to-phase legs, no ground return): ``NodeOrder()`` is exactly the
    ``n_phases`` phase conductors. Returns ``None`` when that assumption does not
    hold (e.g. a stray ground reference among the phase conductors).
    """
    node_order = [int(n) for n in dss.CktElement.NodeOrder()]
    if len(node_order) < n_phases:
        return None
    conductors = node_order[:n_phases]
    if any(c == 0 for c in conductors):
        return None
    return [_phase_num_to_enum(c) for c in conductors]


def _build_shunt_appliance(
    *,
    id: int,
    name: str,
    node: int,
    mode: PhaseMode,
    native_phases: tuple[Phase, ...],
    conductance_s: float,
    capacitance_f: float,
    connection: WindingConnection = WindingConnection.WYE,
) -> ShuntAppliance:
    """Build a :class:`~pgml.schemas.grid_schema.ShuntAppliance`, mode-resolved.

    Under ``SINGLE_PHASE_EQUIV`` the shunt collapses to ``phases=(Phase.A,)``
    with the SAME per-phase ``conductance_s``/``capacitance_f`` scalar (a
    Capacitor/Reactor bank is uncoupled and uniform across phases, so its
    positive-sequence-equivalent value is simply that per-phase value, not a
    sum or a self-minus-mutual reduction). A DELTA bank has no single-phase
    representation, so it folds to the positive-sequence WYE equivalent (a
    balanced delta of per-leg admittance ``y`` presents ``3·y`` per phase).
    Under ``THREE_PHASE`` the (balanced) per-leg value is repeated across every
    phase the element carries; ``connection`` (WYE default, or DELTA for a
    phase-to-phase bank) sets the topology.
    """
    if mode is PhaseMode.SINGLE_PHASE_EQUIV:
        phases = phases_for(mode)
        if connection is WindingConnection.DELTA:
            # Positive-sequence equivalent of a balanced delta bank: Y_wye = 3·Y_leg.
            conductance_s = conductance_s * 3.0
            capacitance_f = capacitance_f * 3.0
        return ShuntAppliance(
            id=id,
            name=name,
            node=node,
            phases=phases,
            conductance_s=[conductance_s] * len(phases),
            capacitance_f=[capacitance_f] * len(phases),
        )
    phases = phases_for(mode, native=native_phases)
    n = len(phases)
    return ShuntAppliance(
        id=id,
        name=name,
        node=node,
        phases=phases,
        conductance_s=[conductance_s] * n,
        capacitance_f=[capacitance_f] * n,
        connection=connection,
    )


# DSS Loads.Model() code -> human-readable name (for the fallback warning).
_DSS_LOAD_MODEL_NAMES: dict[int, str] = {
    1: "constant P, Q",
    2: "constant impedance",
    3: "constant P, quadratic Q (motor-like)",
    4: "linear P, quadratic Q (motor-like)",
    5: "constant current magnitude",
    6: "constant P, fixed Q",
    7: "constant P, fixed Q impedance",
    8: "ZIPV (custom coefficients)",
}


def _resolve_load_model(
    dss: Any, load_name: str, *, kind: str
) -> tuple[Optional[LoadModel], Optional[ZipCoefficients]]:
    """Map ``Loads.Model()`` to a pgml ``(LoadModel, ZipCoefficients)`` pair.

    - ``1`` (constant P, Q)        -> ``CONST_POWER``.
    - ``2`` (constant impedance)   -> ``CONST_IMPEDANCE``.
    - ``5`` (constant current mag) -> ``CONST_CURRENT`` (both P and Q scale
      linearly with |V|, matching pgml's ``CONST_CURRENT`` ZIP triple exactly).
    - ``8`` (ZIPV)                 -> ``ZipCoefficients`` from the first six
      ``ZipV`` coefficients (``z_p, i_p, p_p, z_q, i_q, p_q``); the 7th ZIPV
      entry (the low-voltage cutoff, below which OpenDSS reverts to constant
      impedance) has no pgml equivalent (the ZIP law applies at every
      voltage) and is NOT modeled -- a warning names the load and cutoff.
    - ``3``/``4``/``6``/``7`` (asymmetric P-vs-Q voltage dependence, e.g.
      motor-like characteristics) have no faithful pgml ZIP equivalent
      (pgml's ``ZipCoefficients`` applies one law to P and Q independently,
      but these codes couple them in ways outside that model) -- falls back
      to ``CONST_POWER`` with a warning naming the model.

    Returns ``(None, None)`` for model 1 (the schema default already IS
    ``CONST_POWER``, keeping the historical byte-identical output when no
    caller ever needed to distinguish "unset" from "explicitly const-power").
    """
    model_code = int(dss.Loads.Model())
    if model_code == 1:
        return None, None
    if model_code == 2:
        return LoadModel.CONST_IMPEDANCE, None
    if model_code == 5:
        return LoadModel.CONST_CURRENT, None
    if model_code == 8:
        zipv = list(dss.Loads.ZipV())
        if len(zipv) < 6:
            _logger.warning(
                "OpenDSS %s '%s' uses Model=8 (ZIPV) but ZipV has only %d "
                "value(s) (need at least 6); falling back to CONST_POWER.",
                kind,
                load_name,
                len(zipv),
            )
            return LoadModel.CONST_POWER, None
        zip_coefficients = ZipCoefficients(
            z_p=zipv[0],
            i_p=zipv[1],
            p_p=zipv[2],
            z_q=zipv[3],
            i_q=zipv[4],
            p_q=zipv[5],
        )
        if len(zipv) >= 7 and zipv[6] > 0.0:
            _logger.warning(
                "OpenDSS %s '%s' uses Model=8 (ZIPV) with a low-voltage "
                "cutoff of %.4g pu (below which OpenDSS reverts to constant "
                "impedance); pgml's ZIP load model has no voltage cutoff (the "
                "ZIP law applies at every voltage) -- this is NOT modeled.",
                kind,
                load_name,
                zipv[6],
            )
        return None, zip_coefficients
    _logger.warning(
        "OpenDSS %s '%s' uses Model=%d (%s), which has no faithful pgml ZIP "
        "equivalent (the P/Q voltage dependence is coupled in a way "
        "`ZipCoefficients` cannot express); converting as CONST_POWER "
        "(Model=1) instead.",
        kind,
        load_name,
        model_code,
        _DSS_LOAD_MODEL_NAMES.get(model_code, "unknown"),
    )
    return LoadModel.CONST_POWER, None


__all__ = ["to_grid"]
