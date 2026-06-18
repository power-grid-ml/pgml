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
virtual "1-metre" segment so that ``assembly`` computes:
    Z_total = r_per_m * length_m = r1 * 1 = r1 [Ohm]   ✓

Single-phase positive-sequence equivalent
-----------------------------------------
pgm symmetric (sym) calculation uses one positive-sequence equivalent per node.
We mirror this by assigning ``phases=(Phase.A,)`` to every node, identical to
the pandapower converter convention.  The rated voltage ``u_rated_v`` is the
line-to-line value stored in ``node.u_rated`` (pgm uses line-to-line, SI volts).

Source convention
-----------------
pgm ``source`` carries ``u_ref`` (fraction of rated voltage, dimensionless pu),
``u_ref_angle`` (radians), ``sk`` (short-circuit apparent power, VA) and
``rx_ratio`` (R/X of the source impedance).  We derive the Thevenin impedance::

    |Z_s|  = (u_rated)^2 / sk
    X_s    = |Z_s| / sqrt(1 + rx_ratio^2)
    R_s    = X_s * rx_ratio
    L_s    = X_s / (2*pi*f0)

Very large ``sk`` (ideal slack) produces near-zero Z, which is correct; the
oracle test uses ideal-slack mode so the Thevenin value is irrelevant.

Voltage phasor stored in id_map
--------------------------------
``id_map["slack_v_complex"]`` holds the complex slack voltage phasor in SI volts
(line-to-line) so the test can pass it directly to ``solve_harmonic``.  Multiple
sources are supported; only the FIRST source (by array order) is recorded under
``"slack_v_complex"``.

Load model
----------
pgm ``sym_load.type`` carries a ``LoadGenType`` value.  We record it in
``id_map["load_types"]`` for downstream use; we always create a schema ``Load``
with ``load_model=LoadModel.CONST_IMPEDANCE`` when the test requests it (the
caller sets the pgm ``type`` field before building the Grid).

Supported pgm component types
------------------------------
``node``, ``line``, ``sym_load``, ``source``.  Unknown keys in ``input_data``
are silently ignored so future components can be added to the dict without
breaking this converter.

Only in-service elements are converted (``from_status``/``to_status`` for
lines, ``status`` for loads and sources).
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
    LoadModel,
    Node,
    Phase,
    Provenance,
    ResistanceFrequencyModel,
    Source,
    SourceConvention,
)

_PHASE_A = (Phase.A,)
_PROVENANCE = Provenance(
    source_convention=SourceConvention.SEQUENCE,
    notes=(
        "Converted from power-grid-model input_data (positive-sequence symmetric). "
        "Single-phase-equivalent: phases=(A,), u_rated_v = node.u_rated (line-to-line, V). "
        "Line: virtual length_m=1; per-m params equal pgm total ohms/farads. "
        "Source Z derived from sk and rx_ratio."
    ),
)

# Fallback source impedance when sk is not finite / usably large.
# Any finite, non-zero value works because ideal-slack mode ignores Z_s.
_FALLBACK_R = 1.0e-6  # Ohm
_FALLBACK_L = 1.0e-12  # H


def to_grid(
    input_data: dict[str, Any],
    *,
    base_frequency_hz: float = 50.0,
    load_model: LoadModel = LoadModel.CONST_IMPEDANCE,
) -> tuple[Grid, dict[str, Any]]:
    """Convert a power-grid-model ``input_data`` dict to a schema :class:`~pgml.schemas.grid_schema.Grid`.

    Parameters
    ----------
    input_data:
        Dict of numpy structured arrays, one key per pgm component type
        (``"node"``, ``"line"``, ``"sym_load"``, ``"source"``).  Unknown keys
        are ignored.
    base_frequency_hz:
        System fundamental frequency in Hz.  pgm does not store f0 in the
        structured arrays; the caller must pass it explicitly (default 50 Hz).
    load_model:
        The ``LoadModel`` to assign to every converted ``sym_load``.  Use
        ``LoadModel.CONST_IMPEDANCE`` (the default) to match the M1 const-Z
        linear solver so that pgm and our solver solve the same linear system.

    Returns
    -------
    (Grid, id_map)
        ``Grid`` — materialised schema object.
        ``id_map`` — dict with the following keys:

        - ``"node"``         : ``{pgm_id: Node.id}``
        - ``"line"``         : ``{pgm_id: Line.id}``
        - ``"sym_load"``     : ``{pgm_id: Load.id}``
        - ``"source"``       : ``{pgm_id: Source.id}``
        - ``"slack_v_complex"``: complex slack voltage phasor (V, line-to-line)
          for ideal-slack mode (taken from the first in-service source).
        - ``"load_types"``   : ``{pgm_id: LoadGenType value}`` original pgm type
          for reference.
    """
    two_pi_f0 = 2.0 * math.pi * base_frequency_hz
    _id = _IdCounter()

    id_map: dict[str, Any] = {
        "node": {},
        "line": {},
        "sym_load": {},
        "source": {},
        "load_types": {},
        "slack_v_complex": None,
    }

    # ------------------------------------------------------------------ #
    # 1. Nodes                                                             #
    # ------------------------------------------------------------------ #
    nodes: list[Node] = []
    # Build a set of in-service pgm node ids for foreign-key checks.
    active_pgm_node_ids: set[int] = set()

    for row in input_data.get("node", []):
        pgm_id = int(row["id"])
        u_rated_v = float(row["u_rated"])  # V, line-to-line
        node_id = _id.next()
        id_map["node"][pgm_id] = node_id
        active_pgm_node_ids.add(pgm_id)
        nodes.append(
            Node(
                id=node_id,
                name=f"node_{pgm_id}",
                u_rated_v=u_rated_v,
                phases=_PHASE_A,
            )
        )

    # ------------------------------------------------------------------ #
    # 2. Lines                                                             #
    # ------------------------------------------------------------------ #
    branches: list = []
    for row in input_data.get("line", []):
        from_status = int(row["from_status"])
        to_status = int(row["to_status"])
        if from_status == 0 or to_status == 0:
            continue  # out-of-service

        pgm_id = int(row["id"])
        from_pgm = int(row["from_node"])
        to_pgm = int(row["to_node"])

        # Skip if either terminal node was not converted
        if from_pgm not in id_map["node"] or to_pgm not in id_map["node"]:
            continue

        # pgm: r1/x1 in Ohm (total), c1 in F (total), tan1 dimensionless
        r1 = float(row["r1"])  # Ohm total
        x1 = float(row["x1"])  # Ohm total
        c1 = float(row["c1"])  # F total
        tan1 = float(row.get("tan1", 0.0) if hasattr(row, "get") else row["tan1"])

        # Virtual length = 1 m so that per-m params equal total values
        length_m = 1.0

        l_per_m = x1 / two_pi_f0  # H/m (= H since length=1)
        c_per_m = c1  # F/m (= F since length=1)
        r_per_m = r1  # Ohm/m

        # Shunt conductance from loss angle: G = tan(delta) * omega * C
        g_per_m = tan1 * two_pi_f0 * c1 if (c1 > 0.0 and tan1 != 0.0) else None

        line_id = _id.next()
        id_map["line"][pgm_id] = line_id

        branches.append(
            Line(
                id=line_id,
                name=f"line_{pgm_id}",
                from_node=id_map["node"][from_pgm],
                to_node=id_map["node"][to_pgm],
                from_phases=_PHASE_A,
                to_phases=_PHASE_A,
                length_m=length_m,
                series_resistance_ohm_per_m=[[r_per_m]],
                series_inductance_h_per_m=[[l_per_m]],
                shunt_capacitance_f_per_m=[[c_per_m]],
                shunt_conductance_s_per_m=[[g_per_m]] if g_per_m is not None else None,
                resistance_frequency=ResistanceFrequencyModel(
                    multiplier=ConstantParam(value=1.0)
                ),
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

        # Rated voltage of the source node (LL, V)
        pgm_node_arr = input_data["node"]
        # Find the matching node row for the rated voltage
        node_matches = [r for r in pgm_node_arr if int(r["id"]) == pgm_node]
        u_rated_v = float(node_matches[0]["u_rated"]) if node_matches else 12660.0

        u_ref_v = u_ref_pu * u_rated_v  # magnitude (V, LL)
        u_ref_angle_deg = math.degrees(u_ref_angle_rad)

        # Derive Thevenin impedance from sk and rx_ratio
        r_s, l_s = _thevenin_from_sk(u_rated_v, sk_va, rx_ratio, two_pi_f0)

        src_id = _id.next()
        id_map["source"][pgm_id] = src_id

        # Record the first source's phasor for ideal-slack use
        if id_map["slack_v_complex"] is None:
            id_map["slack_v_complex"] = u_ref_v * complex(
                math.cos(u_ref_angle_rad), math.sin(u_ref_angle_rad)
            )

        appliances.append(
            Source(
                id=src_id,
                name=f"source_{pgm_id}",
                node=id_map["node"][pgm_node],
                phases=_PHASE_A,
                u_ref_v=(u_ref_v,),
                u_angle_deg=(u_ref_angle_deg,),
                resistance_ohm=[[r_s]],
                inductance_h=[[l_s]],
            )
        )

    # ------------------------------------------------------------------ #
    # 4. Loads (sym_load)                                                  #
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
            Load(
                id=load_id,
                name=f"load_{pgm_id}",
                node=id_map["node"][pgm_node],
                phases=_PHASE_A,
                p_nom_w=p_w,
                q_nom_var=q_var,
                load_model=load_model,
            )
        )

    grid = Grid(
        base_frequency_hz=base_frequency_hz,
        nodes=nodes,
        branches=branches,
        appliances=appliances,
        metadata=GridMetadata(
            name="pgm_import",
            description=(
                f"Imported from power-grid-model input_data (f0={base_frequency_hz} Hz). "
                "Single-phase positive-sequence equivalent. "
                "Line length_m=1 (virtual); per-m params equal pgm total ohms/farads."
            ),
        ),
    )
    return grid, id_map


def _thevenin_from_sk(
    u_rated_v: float,
    sk_va: float,
    rx_ratio: float,
    two_pi_f0: float,
) -> tuple[float, float]:
    """Derive (R_s [Ohm], L_s [H]) from short-circuit power and R/X ratio.

    |Z_s| = u_rated_v^2 / sk_va  (positive-sequence, 3-phase base)
    X_s   = |Z_s| / sqrt(1 + rx_ratio^2)
    R_s   = X_s * rx_ratio
    L_s   = X_s / two_pi_f0

    Falls back to tiny values if sk_va is unreasonably large (> 1e15 VA) or
    zero/negative to avoid numerical overflow.
    """
    if sk_va <= 0.0 or sk_va > 1.0e15:
        return _FALLBACK_R, _FALLBACK_L

    z_mag = (u_rated_v**2) / sk_va
    denom = math.sqrt(1.0 + rx_ratio**2)
    x_s = z_mag / denom
    r_s = x_s * rx_ratio
    l_s = x_s / two_pi_f0 if two_pi_f0 > 0.0 else _FALLBACK_L

    # Guard against degenerate near-zero values
    r_s = max(r_s, _FALLBACK_R)
    l_s = max(l_s, _FALLBACK_L)
    return r_s, l_s


class _IdCounter:
    """Monotonically increasing integer id generator."""

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


__all__ = ["to_grid"]
