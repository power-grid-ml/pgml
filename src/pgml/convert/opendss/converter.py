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

Conversion to our SI per-length schema:
  length_m        = length_in_unit * meters_per_unit
  R_total_ohm     = Rmatrix * length_in_unit       (Ohm)
  X_total_ohm     = Xmatrix * length_in_unit       (Ohm)
  C_total_F       = Cmatrix * length_in_unit * 1e-9  (F)
  r_per_m         = R_total_ohm / length_m         (Ohm/m)
  l_per_m         = X_total_ohm / (2*pi*f0) / length_m  (H/m)
  c_per_m         = C_total_F / length_m           (F/m)

Loads
~~~~~
``Loads.kW()``, ``Loads.kvar()``, ``Loads.kV()`` → total P/Q in kW/kVAR, kV L-N.
P in W = kW * 1e3; Q in VAR = kvar * 1e3.
Load kV in OpenDSS is L-N for single-phase loads. Our schema stores u_rated_v = V_LL.

Vsource (external network)
~~~~~~~~~~~~~~~~~~~~~~~~~~
R1/X1 are read via ``dss.Text.Command('? Vsource.<name>.r1')`` etc. (in Ohms).
The Vsource is converted to a :class:`~pgml.schemas.grid_schema.Source` with
Thevenin impedance = (R1, L=X1/(2*pi*f0)), and u_ref_v = BasekV * pu * 1e3 V
(BasekV is line-to-line kV; for single-phase positive-sequence convention we keep it).

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

import math
from typing import Any

from pgml.schemas.grid_schema import (
    ConstantParam,
    Grid,
    GridMetadata,
    Line,
    Load,
    Node,
    Phase,
    Provenance,
    ResistanceFrequencyModel,
    Source,
    SourceConvention,
)

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

_PHASE_A = (Phase.A,)
_TINY_R = 1.0e-6  # Ohm — near-ideal Thevenin for Vsource in Norton stamp
_TINY_L = 1.0e-12  # H   — near-ideal Thevenin for Vsource in Norton stamp
_PROVENANCE = Provenance(
    source_convention=SourceConvention.IMPEDANCE,
    notes=(
        "Converted from OpenDSS circuit (opendssdirect). "
        "Single-phase positive-sequence: phases=(A,), u_rated_v = BasekV*1000 (line-to-line). "
        "Engineering units converted to SI."
    ),
)


def to_grid(dss: Any) -> tuple[Grid, dict[str, Any]]:
    """Convert the currently-loaded OpenDSS circuit to a :class:`~pgml.schemas.grid_schema.Grid`.

    Parameters
    ----------
    dss:
        The ``opendssdirect`` module (``import opendssdirect as dss; dss.Text.Command('Solve')``)
        with a circuit already loaded and solved (or ``Calcvoltagebases`` called).

    Returns
    -------
    (Grid, id_map)
        ``Grid`` — materialised schema object (no ``type_ref``).
        ``id_map`` — ``dict`` mapping DSS element names to our schema ids:
            - ``"bus"``     : ``{dss_bus_name_lower: Node.id}``
            - ``"line"``    : ``{dss_line_name_lower: Line.id}``
            - ``"load"``    : ``{dss_load_name_lower: Load.id}``
            - ``"vsource"`` : ``{dss_vsrc_name_lower: Source.id}``

    Notes
    -----
    - DSS bus and element names are normalised to lowercase.
    - Node ids are assigned in YNodeOrder sequence (bus.phase pairs,
      alphabetical in DSS's internal order) so our compact node-phase index
      matches the DSS Y-matrix row ordering for alignment in oracle tests.
    - The converter handles single-phase circuits (phases=1 per element) and
      the positive-sequence single-phase equivalent convention from pandapower.
    - Only ``Line``, ``Vsource``, and ``Load`` element types are handled; the
      structure is designed to extend to Transformer etc.
    """
    f0_hz: float = float(dss.Solution.Frequency())
    two_pi_f0 = 2.0 * math.pi * f0_hz

    _id = _IdCounter()

    id_map: dict[str, Any] = {"bus": {}, "line": {}, "load": {}, "vsource": {}}

    # ---------------------------------------------------------------------- #
    # 1. Nodes — register in YNodeOrder sequence                              #
    #    YNodeOrder entries are "BUSNAME.phase" (uppercase). We register      #
    #    one node per unique bus name (ignoring phases for now), then one     #
    #    (node_id, phase) per entry in YNodeOrder.                            #
    # ---------------------------------------------------------------------- #
    node_order = dss.Circuit.YNodeOrder()
    # Build: bus_name_lower -> Node.id (first occurrence wins)
    nodes: list[Node] = []
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

    # Now get rated voltage per bus from the circuit
    # We query via dss.Bus API
    for bus_name_lower, phases_list in bus_phases.items():
        node_id = _id.next()
        bus_name_to_node_id[bus_name_lower] = node_id

        # Activate bus to read kVBase
        dss.Circuit.SetActiveBus(bus_name_lower)
        # kVBase is L-N or L-L depending on phase count? In OpenDSS:
        # For single-phase buses, kVBase is L-N. For 3-phase, it's L-L.
        # We keep u_rated_v as reported by OpenDSS (kVBase * 1000) —
        # since we're doing single-phase positive-sequence, kVBase = V_LL / sqrt(3)
        # when the bus is part of a 3-phase system. But we override this
        # with the circuit's BasekV below (in the Vsource section).
        kv_base = dss.Bus.kVBase()  # kV (L-N for single-phase, L-L for 3-phase)
        u_rated_v = kv_base * 1_000.0  # V

        nodes.append(
            Node(
                id=node_id,
                name=bus_name_lower,
                u_rated_v=u_rated_v,
                phases=tuple(phases_list),
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

        # Reshape to n_phases x n_phases matrices
        r_mat = _flat_to_matrix(r_per_m_flat, n_phases)
        l_mat = _flat_to_matrix(l_per_m_flat, n_phases)
        c_mat = _flat_to_matrix(c_per_m_flat, n_phases)

        branches.append(
            Line(
                id=line_id,
                name=line_name,
                from_node=bus_name_to_node_id[from_bus_name],
                to_node=bus_name_to_node_id[to_bus_name],
                from_phases=tuple(from_phases),
                to_phases=tuple(to_phases),
                length_m=length_m,
                series_resistance_ohm_per_m=r_mat,
                series_inductance_h_per_m=l_mat,
                shunt_capacitance_f_per_m=c_mat,
                shunt_conductance_s_per_m=None,  # OpenDSS GMatrix rare in distribution
                resistance_frequency=ResistanceFrequencyModel(
                    multiplier=ConstantParam(value=1.0)
                ),
                provenance=_PROVENANCE,
            )
        )

        ret = dss.Lines.Next()

    # ---------------------------------------------------------------------- #
    # 3. Vsources -> Source (Thevenin)                                        #
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
        bus1_str = bus_names_raw[0].lower()  # first terminal (the non-reference bus)
        src_bus_name, src_phases = _parse_bus_connection(bus1_str, n_phases)

        if src_bus_name not in bus_name_to_node_id:
            ret = dss.Vsources.Next()
            continue

        # Read R1/X1 via text command (Vsources API doesn't expose them directly)
        dss.Text.Command(f"? Vsource.{vsrc_name}.r1")
        r1_str = dss.Text.Result().strip()
        dss.Text.Command(f"? Vsource.{vsrc_name}.x1")
        x1_str = dss.Text.Result().strip()

        r1_ohm = float(r1_str) if r1_str else _TINY_R
        x1_ohm = float(x1_str) if x1_str else 0.0
        l1_h = x1_ohm / two_pi_f0 if x1_ohm > 0.0 else _TINY_L

        # Use tiny Z if the specified impedance is zero (to avoid singular Y)
        if r1_ohm == 0.0 and l1_h == 0.0:
            r1_ohm = _TINY_R
            l1_h = _TINY_L

        # u_ref_v: BasekV is L-L for single-phase positive-sequence equivalent
        u_ref_v = basekv * pu * 1_000.0  # V (L-L magnitude)

        # Build per-phase Thevenin: for n_phases phases, distribute 120-deg apart
        u_ref_tuple = tuple(u_ref_v for _ in range(n_phases))
        u_angle_deg_tuple = tuple(angle_deg - 120.0 * i for i in range(n_phases))

        r_mat = [
            [r1_ohm if i == j else 0.0 for j in range(n_phases)]
            for i in range(n_phases)
        ]
        l_mat = [
            [l1_h if i == j else 0.0 for j in range(n_phases)] for i in range(n_phases)
        ]

        src_id = _id.next()
        id_map["vsource"][vsrc_name] = src_id

        appliances.append(
            Source(
                id=src_id,
                name=vsrc_name,
                node=bus_name_to_node_id[src_bus_name],
                phases=tuple(src_phases),
                u_ref_v=u_ref_tuple,
                u_angle_deg=u_angle_deg_tuple,
                resistance_ohm=r_mat,
                inductance_h=l_mat,
            )
        )

        ret = dss.Vsources.Next()

    # ---------------------------------------------------------------------- #
    # 4. Loads                                                                #
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

        load_id = _id.next()
        id_map["load"][load_name] = load_id

        appliances.append(
            Load(
                id=load_id,
                name=load_name,
                node=bus_name_to_node_id[load_bus_name],
                phases=tuple(load_phases),
                p_nom_w=p_w,
                q_nom_var=q_var,
            )
        )

        ret = dss.Loads.Next()

    grid = Grid(
        base_frequency_hz=f0_hz,
        nodes=nodes,
        branches=branches,
        appliances=appliances,
        metadata=GridMetadata(
            name="opendss_import",
            description=(
                f"Imported from OpenDSS circuit (f0={f0_hz} Hz). "
                "Single-phase positive-sequence equivalent."
            ),
        ),
    )
    return grid, id_map


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _phase_num_to_enum(phase_num: int) -> Phase:
    """Map OpenDSS phase number (1, 2, 3) to our Phase enum (A, B, C).

    OpenDSS numbers phases 1=A, 2=B, 3=C (for positive-sequence circuits).
    Phase number 0 is the neutral.
    """
    _MAP = {1: Phase.A, 2: Phase.B, 3: Phase.C, 0: Phase.N}
    return _MAP.get(phase_num, Phase.A)


def _parse_bus_connection(bus_str: str, n_phases: int) -> tuple[str, list[Phase]]:
    """Parse a DSS bus connection string 'busname.1.2.3' into (bus_name, [phases]).

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
        phases = [_phase_num_to_enum(pn) for pn in phase_nums]
    else:
        # Default: phases 1..n_phases
        phases = [_phase_num_to_enum(i + 1) for i in range(n_phases)]
    return bus_name, phases


def _flat_to_matrix(flat: list[float], n: int) -> list[list[float]]:
    """Reshape a flat list of n*n values (row-major) into an n x n list-of-lists."""
    return [[flat[i * n + j] for j in range(n)] for i in range(n)]


class _IdCounter:
    """Monotonically increasing integer id generator."""

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


__all__ = ["to_grid"]
