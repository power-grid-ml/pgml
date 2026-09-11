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
``u_rated_v = node.u_rated`` (line-to-line, V), 1x1 line matrices.

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

Transformer convention
-----------------------
pgm's two-winding ``transformer`` uses the C++ core's own formulas (pinned against
the installed ``power-grid-model`` source, not just its docs):

- ``uk``/``pk`` (relative short-circuit voltage / copper loss) and ``i0``/``p0``
  (relative no-load current / core loss) are DEFINED on the TO-side (``u2``)
  nameplate voltage: ``z_LL = |uk|*u2**2/sn``, ``r_LL = pk*u2**2/sn**2``,
  ``y_shunt_LL = i0*sn/u2**2 ∠ (via p0/u2**2 real part)``. The ``u2`` used is the
  EFFECTIVE (tap-adjusted, when ``tap_side`` is the to-side) voltage, matching pgm's
  own ``transformer.hpp``.
- pgml stores ``series_resistance_ohm``/``series_inductance_h`` referred to the
  TO-side COIL (see ``docs/pgml/modeling/conventions.md`` sec. 2): a wye/zigzag TO
  winding stores the LL value unchanged; a DELTA TO winding needs the coil factor
  ``z_coil = 3*z_LL``.
- The magnetizing shunt is referred to the FROM/HV terminal by the square nameplate
  ratio ``(u2_eff/u1_eff)**2`` (a line-quantity referral, no delta/wye coil factor —
  mirrors the pandapower/OpenDSS converters). pgm's OWN internal model instead splits
  the (to-side) magnetizing admittance HALF onto its ``Y_tt`` and HALF (reflected
  through the tap) onto ``Y_ff`` — a different topology from pgml's HV-only shunt;
  the two agree only to the extent the magnetizing branch is small relative to the
  leakage (quantified in ``tests/reference/test_pgm_transformer.py``).
- Clock: ``tap.shift_deg = clock*30``, positive = TO LAGS FROM — pinned against a
  live ``PowerGridModel.calculate_power_flow`` solve; identical sign convention to
  pgml's own (pandapower/MATPOWER), no flip needed.
- Tap: ``tap_side`` (0 = from, 1 = to; any other/absent value behaves like "to",
  matching pgm's own default) selects which nameplate voltage
  ``(tap_pos-tap_nom)*tap_size`` volts are added to. ``u_rated_from_v``/
  ``u_rated_to_v`` stay the UNTAPPED nameplate ``u1``/``u2``; ``tap.ratio_magnitude``
  carries the resulting off-nominal fraction only — pinned against a live solve for
  both ``tap_side`` values.
- Grounding: ``r_grounding_*``/``x_grounding_*`` pass through as a
  ``GroundingImpedance``; the pgml core rejects a nonzero one (solid grounding
  only), raising ``ModelingError`` at assembly, not at conversion.
- NOT read/modelled: ``uk_min``/``uk_max``/``pk_min``/``pk_max`` (tap-dependent
  short-circuit parameters — ``uk``/``pk`` are treated as constant across the tap
  range), ``i0_zero_sequence``/``p0_zero_sequence`` (no explicit magnetizing
  zero-sequence override in the schema).

Supported pgm component types
------------------------------
``node``, ``line``, ``transformer``, ``sym_load``, ``asym_load``, ``source``.
Unknown keys in ``input_data`` are silently ignored.

Only in-service elements are converted (``from_status``/``to_status`` for lines and
transformers, ``status`` for loads and sources).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

from pgml.convert._common import (
    IdCounter,
    PhaseMode,
    ZeroSequenceDefaults,
    build_generator,
    build_line_from_sequence,
    build_load,
    build_node,
    build_source,
    make_metadata,
    phases_for,
    resolve_converted_line_models,
    thevenin_from_sk,
    warn_dropped_elements,
)
from pgml.errors import ConversionError
from pgml.schemas.grid_schema import (
    ComplexTap,
    Grid,
    GroundingImpedance,
    LoadModel,
    Provenance,
    SourceConvention,
    Transformer,
    WindingConnection,
)

_logger = logging.getLogger("pgml")

# pgm `WindingType` int values (not imported from `power_grid_model` -- the
# converter treats pgm `input_data` as a plain numpy-structured-array format, with
# no runtime dependency on the `power_grid_model` package itself).
_WINDING_MAP: dict[int, WindingConnection] = {
    0: WindingConnection.WYE,
    1: WindingConnection.WYE_GROUNDED,
    2: WindingConnection.DELTA,
    3: WindingConnection.ZIGZAG,
    4: WindingConnection.ZIGZAG_GROUNDED,
}
_INT8_NA = -128  # pgm's "not available" sentinel for int8 fields (no int NaN)

_PROVENANCE = Provenance(
    source_convention=SourceConvention.SEQUENCE,
    notes=(
        "Converted from power-grid-model input_data (positive-sequence). "
        "Line: virtual length_m=1; per-m params equal pgm total ohms/farads. "
        "Source Z derived from sk and rx_ratio. Transformer: uk/pk/i0/p0 referred "
        "to the to-side nameplate voltage (power-grid-model's own convention)."
    ),
)


def to_grid(
    input_data: dict[str, Any],
    *,
    base_frequency_hz: float = 50.0,
    load_model: LoadModel = LoadModel.CONST_IMPEDANCE,
    phase_mode: PhaseMode = PhaseMode.SINGLE_PHASE_EQUIV,
    harmonic_line_model: Optional[str] = None,
) -> tuple[Grid, dict[str, Any]]:
    """Convert a power-grid-model ``input_data`` dict to a schema :class:`~pgml.schemas.grid_schema.Grid`.

    Parameters
    ----------
    input_data:
        Dict of numpy structured arrays, one key per pgm component type
        (``"node"``, ``"line"``, ``"transformer"``, ``"sym_load"``,
        ``"asym_load"``, ``"source"``). Unknown keys are ignored.
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
    harmonic_line_model:
        Frequency-dependent line model written to every converted line
        (``"sequence_aware"``, ``"positive_sequence"``, ``"naive"``, or ``"none"`` to
        leave the lines unresolved). ``None`` (default) takes the modeling defaults
        ``line.harmonic_model.three_phase`` / ``.single_phase``; the applied model is
        logged once. power-grid-model is fundamental-only, so this is a pgml modeling
        decision, not a property of the source data.

    Returns
    -------
    (Grid, id_map)
        ``Grid`` — materialised schema object.
        ``id_map`` — dict with the following keys:

        - ``"node"``         : ``{pgm_id: Node.id}``
        - ``"line"``         : ``{pgm_id: Line.id}``
        - ``"transformer"``  : ``{pgm_id: Transformer.id}``
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
        "transformer": {},
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
    zero_sequence = ZeroSequenceDefaults()
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
        if phase_mode is PhaseMode.THREE_PHASE:
            zero_sequence.note(r0=r0, x0=x0, c0=c0)

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
    # 3. Transformers (two-winding, vector-group aware)                    #
    # ------------------------------------------------------------------ #
    # See the module docstring "Transformer convention" section for the full
    # derivation (pinned against the installed power-grid-model C++ source, not
    # just its docs). Summary:
    #   - uk/pk/i0/p0 are defined on the EFFECTIVE (tap-adjusted) to-side (u2)
    #     nameplate voltage; series R/L are referred to the to-side COIL (a
    #     delta TO winding needs the intrinsic sqrt(3)^2 = 3 coil factor).
    #   - The magnetizing shunt is referred to the FROM/HV terminal by the
    #     square nameplate ratio (a line-quantity referral, mirroring the
    #     pandapower/OpenDSS converters); pgm's own internal model splits it
    #     across both terminals instead, a documented, quantified residual.
    #   - clock -> tap.shift_deg = clock*30 (positive = TO lags FROM), the same
    #     sign pgml already uses; no flip.
    #   - tap_side (0=from, else=to, matching pgm's own default) selects which
    #     nameplate voltage the tap-changer volts are added to;
    #     tap.ratio_magnitude carries the resulting OFF-NOMINAL fraction only.
    #   - r_grounding_*/x_grounding_* pass through as GroundingImpedance; a
    #     nonzero value is rejected by the pgml core (solid grounding only),
    #     not here.
    # ------------------------------------------------------------------ #
    for row in input_data.get("transformer", []):
        if int(row["from_status"]) == 0 or int(row["to_status"]) == 0:
            continue  # out-of-service

        pgm_id = int(row["id"])
        from_pgm = int(row["from_node"])
        to_pgm = int(row["to_node"])
        if from_pgm not in id_map["node"] or to_pgm not in id_map["node"]:
            continue

        u1 = float(row["u1"])  # V, untapped from/HV nameplate
        u2 = float(row["u2"])  # V, untapped to/LV nameplate
        sn = float(row["sn"])  # VA
        uk = float(row["uk"])  # relative short-circuit voltage (p.u. of sn)
        pk = float(row["pk"])  # W, short-circuit (copper) loss
        i0 = float(row["i0"])  # relative no-load current (p.u. of sn)
        p0 = float(row["p0"])  # W, no-load (core) loss

        from_connection = _winding_connection(int(row["winding_from"]), pgm_id, "from")
        to_connection = _winding_connection(int(row["winding_to"]), pgm_id, "to")
        clock = int(row["clock"]) % 12

        # Off-nominal tap: effective (possibly tap-adjusted) nameplate voltages.
        tap_side = _opt_int_field(row, "tap_side")
        tap_pos = _opt_int_field(row, "tap_pos")
        tap_nom = _opt_int_field(row, "tap_nom")
        tap_min = _opt_int_field(row, "tap_min")
        tap_max = _opt_int_field(row, "tap_max")
        tap_size = _opt_field(row, "tap_size") or 0.0

        tap_nom_eff = tap_nom if tap_nom is not None else 0
        tap_pos_eff = tap_pos if tap_pos is not None else tap_nom_eff
        tap_min_eff = tap_min if tap_min is not None else tap_nom_eff
        tap_max_eff = tap_max if tap_max is not None else tap_nom_eff
        tap_direction = 1.0 if tap_max_eff > tap_min_eff else -1.0
        delta_u = tap_direction * (tap_pos_eff - tap_nom_eff) * tap_size

        u1_eff, u2_eff = u1, u2
        if tap_side == 0:  # tap on the from/HV side
            u1_eff = u1 + delta_u
        else:  # tap on the to/LV side (also pgm's default when absent)
            u2_eff = u2 + delta_u

        ratio_magnitude = (u1_eff / u2_eff) / (u1 / u2)

        # Series leakage: LL value at the effective to-side voltage. pgm's uk/pk
        # are connection-agnostic LINE quantities (no internal delta correction
        # anywhere in power-grid-model's own transformer.hpp). The schema stores
        # the leakage referred to the TO-side COIL (`docs/pgml/modeling/
        # transformer.md`, "Leakage referral"): identical to the LL value for a
        # wye / zigzag TO winding, `z_coil = 3*z_LL` for a DELTA TO winding.
        # Both assembly paths recover the same LL positive sequence from the
        # coil value (the 3-phase incidence through Mᵀ M, the single-phase
        # scalar pi through its own y_LL = 3·y_coil referral), so the stored
        # Grid is phase-mode-independent.
        z_ll_abs = abs(uk) * u2_eff * u2_eff / sn
        r_ll = pk * u2_eff * u2_eff / (sn * sn)
        uk_sign = 1.0 if uk >= 0.0 else -1.0
        x_ll_sq = z_ll_abs * z_ll_abs - r_ll * r_ll
        x_ll = uk_sign * math.sqrt(x_ll_sq) if x_ll_sq > 0.0 else 0.0
        coil_factor = 3.0 if to_connection == WindingConnection.DELTA else 1.0
        r_coil = coil_factor * r_ll
        x_coil = coil_factor * x_ll

        # Magnetizing shunt: to-side-referred (same u2_eff as the series leakage),
        # then referred to the FROM/HV terminal by the square nameplate ratio.
        g_m = 0.0
        l_m: Optional[float] = None
        if i0 > 0.0 or p0 > 0.0:
            y_shunt_abs = i0 * sn / (u2_eff * u2_eff)
            g_to = p0 / (u2_eff * u2_eff)
            b_to_sq = y_shunt_abs * y_shunt_abs - g_to * g_to
            b_to = math.sqrt(b_to_sq) if b_to_sq > 0.0 else 0.0
            ratio_sq = (u2_eff / u1_eff) ** 2
            g_m = g_to * ratio_sq
            b_m = b_to * ratio_sq
            if b_m > 0.0:
                l_m = 1.0 / (two_pi_f0 * b_m)

        from_grounding = _grounding_impedance(row, "from")
        to_grounding = _grounding_impedance(row, "to")

        trafo_id = _id.next()
        id_map["transformer"][pgm_id] = trafo_id
        tx_phases = phases_for(phase_mode)

        branches.append(
            Transformer(
                id=trafo_id,
                name=f"transformer_{pgm_id}",
                from_node=id_map["node"][from_pgm],
                to_node=id_map["node"][to_pgm],
                from_phases=tx_phases,
                to_phases=tx_phases,
                s_rated_va=sn,
                u_rated_from_v=u1,
                u_rated_to_v=u2,
                from_connection=from_connection,
                to_connection=to_connection,
                series_resistance_ohm=r_coil,
                series_inductance_h=x_coil / two_pi_f0,
                magnetizing_conductance_s=g_m,
                magnetizing_inductance_h=l_m,
                tap=ComplexTap(
                    ratio_magnitude=ratio_magnitude, shift_deg=float(clock * 30)
                ),
                from_grounding=from_grounding,
                to_grounding=to_grounding,
                provenance=_PROVENANCE,
            )
        )

    # ------------------------------------------------------------------ #
    # 4. Sources                                                           #
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
    # 5. Symmetric loads (sym_load) — balanced total                       #
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
    # 6. Asymmetric loads (asym_load) — genuine per-phase split            #
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
    # 7. Symmetric generators (sym_gen) -> Generator (PQ injection)        #
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
    zero_sequence.warn(_logger, tool="power-grid-model")
    resolve_converted_line_models(
        grid, _logger, tool="power-grid-model", requested=harmonic_line_model
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


def _opt_int_field(row: Any, name: str) -> int | None:
    """Read an optional pgm int8-sentinel field (tap/clock/winding); ``None`` if NA.

    Integer dtypes have no NaN, so pgm fills an unspecified int8 field with the
    dtype minimum (``-128``) as its "not available" sentinel.
    """
    try:
        val = row[name]
    except (KeyError, ValueError, IndexError, TypeError):
        return None
    try:
        i = int(val)
    except (TypeError, ValueError):
        return None
    return None if i == _INT8_NA else i


def _winding_connection(value: int, pgm_id: int, side: str) -> WindingConnection:
    """Map a pgm ``WindingType`` int to a :class:`WindingConnection`; raise if unknown."""
    try:
        return _WINDING_MAP[value]
    except KeyError as exc:
        raise ConversionError(
            f"transformer {pgm_id}: unknown pgm winding_{side} value {value!r} "
            f"(expected one of {sorted(_WINDING_MAP)})."
        ) from exc


def _grounding_impedance(row: Any, side: str) -> GroundingImpedance | None:
    """Read ``r_grounding_{side}``/``x_grounding_{side}``; ``None`` when solid (0/absent).

    A nonzero value is passed through so the pgml core raises ``ModelingError`` at
    assembly (solid grounding only), not here.
    """
    r = _opt_field(row, f"r_grounding_{side}")
    x = _opt_field(row, f"x_grounding_{side}")
    r = 0.0 if r is None else r
    x = 0.0 if x is None else x
    if r != 0.0 or x != 0.0:
        return GroundingImpedance(r_ohm=r, x_ohm=x)
    return None


__all__ = ["to_grid"]
