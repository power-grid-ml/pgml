"""Pure conversion function: power-grid-model input_data -> (Grid, id_map).

Conventions applied
-------------------
power-grid-model (pgm) stores TOTAL (lumped) positive-sequence impedances, not
per-unit-length values.  The field mapping is:

  pgm line.r1 [Ohm]  -> series_resistance_ohm_per_m  = r1  (with length_m = 1.0)
  pgm line.x1 [Ohm]  -> series_inductance_h_per_m    = x1 / (2*pi*f0)
  pgm line.c1 [F]    -> shunt_capacitance_f_per_m     = c1  (with length_m = 1.0)
  pgm line.tan1      -> shunt_conductance_s_per_m     = tan1 * (2*pi*f0*c1)
                        (loss angle: G = tan(delta) * B_c)

Because pgm has no per-length or length field, we represent every line as a
virtual "1-metre" segment so that ``assembly`` computes
``Z_total = r_per_m * length_m = r1 * 1 = r1`` [Ohm].

Phase mode
----------
``phase_mode=PhaseMode.SINGLE_PHASE_EQUIV`` (default) mirrors pgm's symmetric (sym)
calculation: one positive-sequence equivalent per node, ``phases=(Phase.A,)``,
``u_rated_v = node.u_rated`` (line-to-line, V), 1x1 line matrices. Reproduces the
historical output exactly.

``phase_mode=PhaseMode.THREE_PHASE`` produces a genuine abc grid: nodes/branches
become ``(A, B, C)``; lines are expanded from sequence quantities via the
symmetric-component identity (zero-sequence from ``line`` ``r0/x0/c0`` fields when
present, else ``pgml.defaults``); sources become balanced 3-phase Thevenins
(angles ``u_ref_angle / -120 / +120``); ``asym_load`` entries are captured with
per-phase P/Q. pgm has NO load connection field, so every converted load is WYE
(power-grid-model models all loads wye, injecting at the node).

Source convention
-----------------
pgm ``source`` carries ``u_ref`` (fraction of rated voltage), ``u_ref_angle``
(radians), ``sk`` (short-circuit apparent power, VA) and ``rx_ratio`` (R/X). The
Thevenin impedance is derived in :func:`pgml.convert._common.thevenin_from_sk`. The
zero-sequence source impedance equals the positive-sequence impedance (no native
zero-sequence source data is read).

Voltage phasor stored in id_map
--------------------------------
``id_map["slack_v_complex"]`` holds the complex slack voltage phasor in SI volts
(line-to-line); only the FIRST in-service source (by array order) is recorded.

Load model
----------
``sym_load.type`` carries a ``LoadGenType`` value, recorded in
``id_map["load_types"]``; the schema ``Load`` is created with ``load_model``.

Supported pgm component types
------------------------------
``node``, ``line``, ``sym_load``, ``asym_load``, ``source``.  Unknown keys in
``input_data`` are silently ignored.

Only in-service elements are converted (``from_status``/``to_status`` for lines,
``status`` for loads and sources).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from pgml.convert._common import (
    IdCounter,
    PhaseMode,
    build_generator,
    build_line_from_sequence,
    build_load,
    build_node,
    build_source,
    make_metadata,
    thevenin_from_sk,
    warn_dropped_elements,
)
from pgml.schemas.grid_schema import (
    Grid,
    LoadModel,
    Provenance,
    SourceConvention,
    WindingConnection,
)

_logger = logging.getLogger("pgml")

_PROVENANCE = Provenance(
    source_convention=SourceConvention.SEQUENCE,
    notes=(
        "Converted from power-grid-model input_data (positive-sequence). "
        "Line: virtual length_m=1; per-m params equal pgm total ohms/farads. "
        "Source Z derived from sk and rx_ratio."
    ),
)


def to_grid(
    input_data: dict[str, Any],
    *,
    base_frequency_hz: float = 50.0,
    load_model: LoadModel = LoadModel.CONST_IMPEDANCE,
    phase_mode: PhaseMode = PhaseMode.SINGLE_PHASE_EQUIV,
) -> tuple[Grid, dict[str, Any]]:
    """Convert a power-grid-model ``input_data`` dict to a schema :class:`~pgml.schemas.grid_schema.Grid`.

    Parameters
    ----------
    input_data:
        Dict of numpy structured arrays, one key per pgm component type
        (``"node"``, ``"line"``, ``"sym_load"``, ``"asym_load"``, ``"source"``).
        Unknown keys are ignored.
    base_frequency_hz:
        System fundamental frequency in Hz.  pgm does not store f0 in the
        structured arrays; the caller must pass it explicitly (default 50 Hz).
    load_model:
        The ``LoadModel`` assigned to every converted load.  Use
        ``LoadModel.CONST_IMPEDANCE`` (the default) to match a const-Z linear
        reference solve.
    phase_mode:
        :class:`~pgml.convert._common.PhaseMode`. ``SINGLE_PHASE_EQUIV`` (default)
        reproduces the positive-sequence single-phase-equivalent output exactly;
        ``THREE_PHASE`` expands to a genuine abc grid (sequence->phase line
        matrices, balanced 3-phase sources, ``asym_load`` per-phase capture).

    Returns
    -------
    (Grid, id_map)
        ``Grid`` — materialised schema object.
        ``id_map`` — dict with the following keys:

        - ``"node"``         : ``{pgm_id: Node.id}``
        - ``"line"``         : ``{pgm_id: Line.id}``
        - ``"sym_load"``     : ``{pgm_id: Load.id}``
        - ``"asym_load"``    : ``{pgm_id: Load.id}`` (THREE_PHASE only)
        - ``"source"``       : ``{pgm_id: Source.id}``
        - ``"slack_v_complex"``: complex slack voltage phasor (V, line-to-line).
        - ``"load_types"``   : ``{pgm_id: LoadGenType value}`` original pgm type.
    """
    two_pi_f0 = 2.0 * math.pi * base_frequency_hz
    _id = IdCounter()

    id_map: dict[str, Any] = {
        "node": {},
        "line": {},
        "sym_load": {},
        "asym_load": {},
        "sym_gen": {},
        "source": {},
        "load_types": {},
        "slack_v_complex": None,
    }

    # ------------------------------------------------------------------ #
    # 1. Nodes                                                             #
    # ------------------------------------------------------------------ #
    nodes: list = []
    u_rated_by_pgm: dict[int, float] = {}

    for row in input_data.get("node", []):
        pgm_id = int(row["id"])
        u_rated_v = float(row["u_rated"])  # V, line-to-line
        node_id = _id.next()
        id_map["node"][pgm_id] = node_id
        u_rated_by_pgm[pgm_id] = u_rated_v
        nodes.append(
            build_node(
                id=node_id,
                name=f"node_{pgm_id}",
                u_rated_v=u_rated_v,
                mode=phase_mode,
            )
        )

    # ------------------------------------------------------------------ #
    # 2. Lines                                                             #
    # ------------------------------------------------------------------ #
    branches: list = []
    for row in input_data.get("line", []):
        if int(row["from_status"]) == 0 or int(row["to_status"]) == 0:
            continue  # out-of-service

        pgm_id = int(row["id"])
        from_pgm = int(row["from_node"])
        to_pgm = int(row["to_node"])
        if from_pgm not in id_map["node"] or to_pgm not in id_map["node"]:
            continue

        # pgm: r1/x1 in Ohm (total), c1 in F (total), tan1 dimensionless.
        # Virtual length = 1 m so per-m params equal the total values.
        r1 = float(row["r1"])  # Ohm total
        x1 = float(row["x1"])  # Ohm total
        c1 = float(row["c1"])  # F total
        tan1 = float(_field(row, "tan1", 0.0))

        # Positive-sequence shunt conductance from loss angle: G = tan(delta)*omega*C
        g1 = tan1 * two_pi_f0 * c1 if (c1 > 0.0 and tan1 != 0.0) else 0.0

        # Native zero-sequence fields (THREE_PHASE only; else config defaults).
        r0 = _opt_field(row, "r0")
        x0 = _opt_field(row, "x0")
        c0 = _opt_field(row, "c0")

        line_id = _id.next()
        id_map["line"][pgm_id] = line_id

        branches.append(
            build_line_from_sequence(
                id=line_id,
                name=f"line_{pgm_id}",
                from_node=id_map["node"][from_pgm],
                to_node=id_map["node"][to_pgm],
                mode=phase_mode,
                length_m=1.0,
                r1=r1,
                x1=x1,
                c1=c1,
                two_pi_f0=two_pi_f0,
                r0=r0,
                x0=x0,
                c0=c0,
                g1=g1,
                provenance=_PROVENANCE,
            )
        )

    # ------------------------------------------------------------------ #
    # 3. Sources                                                           #
    # ------------------------------------------------------------------ #
    appliances: list = []
    for row in input_data.get("source", []):
        if int(row["status"]) == 0:
            continue

        pgm_id = int(row["id"])
        pgm_node = int(row["node"])
        if pgm_node not in id_map["node"]:
            continue

        u_ref_pu = float(row["u_ref"])  # per-unit
        u_ref_angle_rad = float(row["u_ref_angle"])  # radians
        sk_va = float(row["sk"])  # short-circuit VA
        rx_ratio = float(row["rx_ratio"])  # R/X

        u_rated_v = u_rated_by_pgm.get(pgm_node, 12660.0)
        u_ref_v = u_ref_pu * u_rated_v  # magnitude (V, LL)
        u_ref_angle_deg = math.degrees(u_ref_angle_rad)

        r_s, l_s = thevenin_from_sk(u_rated_v, sk_va, rx_ratio, two_pi_f0)

        src_id = _id.next()
        id_map["source"][pgm_id] = src_id

        if id_map["slack_v_complex"] is None:
            id_map["slack_v_complex"] = u_ref_v * complex(
                math.cos(u_ref_angle_rad), math.sin(u_ref_angle_rad)
            )

        appliances.append(
            build_source(
                id=src_id,
                name=f"source_{pgm_id}",
                node=id_map["node"][pgm_node],
                mode=phase_mode,
                u_ref_v=u_ref_v,
                u_angle_deg=u_ref_angle_deg,
                r_ohm=r_s,
                l_h=l_s,
            )
        )

    # ------------------------------------------------------------------ #
    # 4. Symmetric loads (sym_load) — balanced total                       #
    # ------------------------------------------------------------------ #
    for row in input_data.get("sym_load", []):
        if int(row["status"]) == 0:
            continue

        pgm_id = int(row["id"])
        pgm_node = int(row["node"])
        if pgm_node not in id_map["node"]:
            continue

        p_w = float(row["p_specified"])  # W
        q_var = float(row["q_specified"])  # VAr
        pgm_type = int(row["type"])  # LoadGenType int value

        load_id = _id.next()
        id_map["sym_load"][pgm_id] = load_id
        id_map["load_types"][pgm_id] = pgm_type

        appliances.append(
            build_load(
                id=load_id,
                name=f"load_{pgm_id}",
                node=id_map["node"][pgm_node],
                mode=phase_mode,
                p_total_w=p_w,
                q_total_var=q_var,
                load_model=load_model,
            )
        )

    # ------------------------------------------------------------------ #
    # 5. Asymmetric loads (asym_load) — genuine per-phase split            #
    # ------------------------------------------------------------------ #
    # pgm p_specified/q_specified are shape (3,) per-phase. pgm has NO load   #
    # connection field, so every load is WYE. Under SINGLE_PHASE_EQUIV the    #
    # per-phase split is collapsed into a balanced 1-phase total (logged).    #
    # ------------------------------------------------------------------ #
    for row in input_data.get("asym_load", []):
        if int(row["status"]) == 0:
            continue

        pgm_id = int(row["id"])
        pgm_node = int(row["node"])
        if pgm_node not in id_map["node"]:
            continue

        p_phase = tuple(float(v) for v in row["p_specified"])  # (3,) W
        q_phase = tuple(float(v) for v in row["q_specified"])  # (3,) VAr
        p_total = sum(p_phase)
        q_total = sum(q_phase)
        pgm_type = int(row["type"])

        load_id = _id.next()
        id_map["asym_load"][pgm_id] = load_id
        id_map["load_types"][pgm_id] = pgm_type

        if phase_mode is PhaseMode.SINGLE_PHASE_EQUIV:
            _logger.info(
                "power-grid-model asym_load %s collapsed to a balanced "
                "single-phase total under SINGLE_PHASE_EQUIV "
                "(per-phase split discarded); use THREE_PHASE to keep it.",
                pgm_id,
            )
            appliances.append(
                build_load(
                    id=load_id,
                    name=f"asym_load_{pgm_id}",
                    node=id_map["node"][pgm_node],
                    mode=phase_mode,
                    p_total_w=p_total,
                    q_total_var=q_total,
                    load_model=load_model,
                )
            )
        else:
            appliances.append(
                build_load(
                    id=load_id,
                    name=f"asym_load_{pgm_id}",
                    node=id_map["node"][pgm_node],
                    mode=phase_mode,
                    p_total_w=p_total,
                    q_total_var=q_total,
                    connection=WindingConnection.WYE,
                    p_per_phase_w=p_phase,
                    q_per_phase_var=q_phase,
                    load_model=load_model,
                )
            )

    # ------------------------------------------------------------------ #
    # 6. Symmetric generators (sym_gen) -> Generator (PQ injection)        #
    # ------------------------------------------------------------------ #
    # pgm sym_gen is GENERATION-POSITIVE (p_specified > 0 injects), matching
    # the Generator nameplate convention; the assembly applies the sign.
    for row in input_data.get("sym_gen", []):
        if int(row["status"]) == 0:
            continue
        pgm_id = int(row["id"])
        pgm_node = int(row["node"])
        if pgm_node not in id_map["node"]:
            continue
        gen_id = _id.next()
        id_map["sym_gen"][pgm_id] = gen_id
        appliances.append(
            build_generator(
                id=gen_id,
                name=f"sym_gen_{pgm_id}",
                node=id_map["node"][pgm_node],
                mode=phase_mode,
                p_total_w=float(row["p_specified"]),
                q_total_var=float(row["q_specified"]),
            )
        )

    # Components the converter does NOT read: fail loud, never silently wrong.
    warn_dropped_elements(
        _logger,
        "power-grid-model",
        {
            kind: len(input_data.get(kind, []))
            for kind in (
                "transformer",
                "three_winding_transformer",
                "shunt",
                "asym_gen",
                "link",
                "transformer_tap_regulator",
            )
        },
    )

    description = (
        f"Imported from power-grid-model input_data (f0={base_frequency_hz} Hz). "
        + (
            "Single-phase positive-sequence equivalent. "
            "Line length_m=1 (virtual); per-m params equal pgm total ohms/farads."
            if phase_mode is PhaseMode.SINGLE_PHASE_EQUIV
            else "Three-phase (abc) expansion from sequence quantities."
        )
    )
    grid = Grid(
        base_frequency_hz=base_frequency_hz,
        nodes=nodes,
        branches=branches,
        appliances=appliances,
        metadata=make_metadata(name="pgm_import", description=description),
    )
    return grid, id_map


def _field(row: Any, name: str, default: float) -> float:
    """Read a field from a pgm structured-array row with a default (dict or void)."""
    if hasattr(row, "get"):
        return row.get(name, default)
    try:
        return row[name]
    except (KeyError, ValueError, IndexError):
        return default


def _opt_field(row: Any, name: str) -> float | None:
    """Read an optional pgm zero-sequence field; ``None`` if absent or NaN.

    pgm fills unspecified numeric fields with NaN; a NaN means "use the config
    zero-sequence default" rather than a literal value.
    """
    try:
        val = row[name]
    except (KeyError, ValueError, IndexError, TypeError):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


__all__ = ["to_grid"]
