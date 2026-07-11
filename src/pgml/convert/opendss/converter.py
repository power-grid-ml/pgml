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
    build_line_from_matrices,
    build_load,
    build_node,
    build_source,
    make_metadata,
    phases_for,
    thevenin_from_z,
)
from pgml.errors import ConversionError
from pgml.schemas.grid_schema import (
    ComplexTap,
    Grid,
    Phase,
    Provenance,
    SourceConvention,
    Transformer,
    WindingConnection,
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

        - ``"bus"``     -> ``{dss_bus_name_lower: Node.id}``
        - ``"line"``    -> ``{dss_line_name_lower: Line.id}``
        - ``"trafo"``   -> ``{dss_trafo_name_lower: Transformer.id}``
        - ``"load"``    -> ``{dss_load_name_lower: Load.id}``
        - ``"vsource"`` -> ``{dss_vsrc_name_lower: Source.id}``
        - ``"slack_v_complex"`` -> complex slack voltage phasor (V, line-to-line)
          from the first Vsource, for ideal-slack mode.

    Notes
    -----
    - DSS bus and element names are normalised to lowercase.
    - Node ids are assigned in YNodeOrder sequence (bus.phase pairs,
      alphabetical in DSS's internal order) so our compact node-phase index
      matches the DSS Y-matrix row ordering for alignment in oracle tests.
    - ``Line``, ``Transformer``, ``Vsource``, and ``Load`` element types are
      handled; the structure extends to Capacitor/Reactor similarly.
      Two-winding ``Transformer`` elements convert to
      :class:`~pgml.schemas.grid_schema.Transformer` (winding 1 = HV/from,
      winding 2 = LV/to; solidly grounded wye or delta windings only; see
      the module CONTEXT.md for the full field mapping and scope).
    - ``u_rated_v`` is line-to-line for every bus (kVBase * sqrt(3) * 1000),
      consistent with the pandapower/pgm converters and the assembly const-Z
      shunt formula. OpenDSS ``kVBase()`` returns L-N, recovered to L-L by the
      sqrt(3) factor.
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
            # Positive-sequence equivalent: use the 1x1 [0][0] entry only.
            r_mat = [[r_per_m_flat[0]]]
            l_mat = [[l_per_m_flat[0]]]
            c_mat = [[c_per_m_flat[0]]]
            line_phases: tuple[Phase, ...] = phases_for(phase_mode)
        else:
            # THREE_PHASE: emit the real n×n matrices from DSS.
            r_mat = _flat_to_matrix(r_per_m_flat, n_phases)
            l_mat = _flat_to_matrix(l_per_m_flat, n_phases)
            c_mat = _flat_to_matrix(c_per_m_flat, n_phases)
            line_phases = phases_for(phase_mode, native=tuple(from_phases))

        branches.append(
            build_line_from_matrices(
                id=line_id,
                name=line_name,
                from_node=bus_name_to_node_id[from_bus_name],
                to_node=bus_name_to_node_id[to_bus_name],
                phases=line_phases,
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
    # transformer's kVA rating, so
    #     R_lv_ohm = (%R_wdg1 + %R_wdg2)/100 * Z_base_LV
    #     X_lv_ohm = %XHL/100 * Z_base_LV,   Z_base_LV = kV_lv^2*1000/kVA
    # recovers the total leakage referred to the LV coil -- valid whenever
    # both windings share one kVA rating (checked below; OpenDSS's own
    # convention places `XHL` on winding 1's kVA base, which coincides with
    # winding 2's when the two are equal).
    #
    # Vector group / clock: OpenDSS has no explicit clock parameter -- its
    # `LeadLag` toggle only distinguishes the 30-degree Dy/Yd shift (verified
    # against a live solve: `Lag` -> the LV bus lags the HV bus by ~30 deg
    # (Dyn1, `shift_deg=30`); `Lead` -> LV leads by ~30 deg (Dyn11,
    # `shift_deg=330`)). A matching Yy/Dd pairing has no inherent phase shift
    # (clock 0); OpenDSS cannot express a Yy6/Dd6 (180-degree) group through
    # this element, so it is not detected here (see the module CONTEXT.md).
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

        # Total leakage referred to the LV (to-side) coil.
        z_base_lv_ohm = (kv_to**2 * 1_000.0) / kva_to
        r_pct_total = wdg_pct_r[0] + wdg_pct_r[1]
        r_lv_ohm = r_pct_total / 100.0 * z_base_lv_ohm
        x_lv_ohm = xhl_pct / 100.0 * z_base_lv_ohm
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

        from_bus_name, from_phase_list, from_grounded = _parse_transformer_winding_bus(
            bus_names_raw[0], n_phases
        )
        to_bus_name, to_phase_list, to_grounded = _parse_transformer_winding_bus(
            bus_names_raw[1], n_phases
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
                shift_deg = 30.0
            elif leadlag in ("lead", "euro"):
                shift_deg = 330.0
            else:
                raise ConversionError(
                    f"OpenDSS transformer '{trafo_name}': unrecognized "
                    f"LeadLag value {leadlag!r}."
                )
        else:
            # Yy / Dd: no inherent phase shift is expressible through a plain
            # OpenDSS Transformer element (clock 0 only).
            shift_deg = 0.0

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

        # Activate load element to get bus connection
        dss.Circuit.SetActiveElement(f"Load.{load_name}")
        n_phases = dss.CktElement.NumPhases()
        bus_names_raw = dss.CktElement.BusNames()
        bus1_str = bus_names_raw[0].lower()
        load_bus_name, load_phases = _parse_bus_connection(bus1_str, n_phases)

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
            appliances.append(
                build_load(
                    id=load_id,
                    name=load_name,
                    node=bus_name_to_node_id[load_bus_name],
                    mode=phase_mode,
                    p_total_w=p_w,
                    q_total_var=q_var,
                    connection=conn,
                    native_phases=native_load_phases,
                )
            )

        ret = dss.Loads.Next()

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


def _parse_transformer_winding_bus(
    bus_str: str, n_phases: int
) -> tuple[str, list[Phase], bool]:
    """Parse a transformer winding bus string into ``(bus_name, phases, grounded)``.

    ``phases`` is the ordered list of the ``n_phases`` conducting-phase
    conductors (the leading node indices, or the default ``1..n_phases`` when
    no explicit list is given, mirroring :func:`_parse_bus_connection`).

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
        ``"hv.1.2.3"`` or ``"lv.1.2.3.0"``.
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
    phases = [_phase_num_to_enum(pn) for pn in phase_nums]
    grounded = len(node_nums) <= n_phases or node_nums[n_phases] == 0
    return bus_name, phases, grounded


def _flat_to_matrix(flat: list[float], n: int) -> list[list[float]]:
    """Reshape a flat list of n*n values (row-major) into an n x n list-of-lists."""
    return [[flat[i * n + j] for j in range(n)] for i in range(n)]


__all__ = ["to_grid"]
