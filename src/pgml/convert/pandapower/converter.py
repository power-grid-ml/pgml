"""Pure conversion function: pandapower net -> (Grid, id_map).

Conventions applied
-------------------
Unit conversion (engineering -> SI):
  - vn_kv [kV]  -> u_rated_v [V]  : multiply by 1000
  - length_km   -> length_m        : multiply by 1000
  - r_ohm_per_km -> series_resistance_ohm_per_m  : divide by 1000
  - x_ohm_per_km -> series_inductance_h_per_m    : x/(2*pi*f0) / 1000
  - c_nf_per_km  -> shunt_capacitance_f_per_m    : multiply by 1e-9, divide by 1000 = 1e-12
  - g_us_per_km  -> shunt_conductance_s_per_m    : multiply by 1e-6, divide by 1000 = 1e-9
  - p_mw         -> p_nom_w [W]   : multiply by 1e6
  - q_mvar       -> q_nom_var [VAr]: multiply by 1e6

Phase mode
----------
``phase_mode=PhaseMode.SINGLE_PHASE_EQUIV`` (default) keeps today's positive-sequence
single-phase equivalent: every node/branch is ``phases=(Phase.A,)`` and lines carry
1x1 matrices. ``u_rated_v = vn_kv * 1000`` (line-to-line magnitude, retained as-is for
the 1-phase node because ``phase_voltage_magnitude`` returns ``u_rated_v`` unchanged
for nodes with fewer than 3 phases); this matches pandapower's const-Z reference
``y_const = conj(S_total)/(V_LL)^2``.

``phase_mode=PhaseMode.THREE_PHASE`` produces a genuine abc grid: nodes/branches become
``(A, B, C)``; lines are expanded from sequence quantities via the symmetric-component
identity (zero-sequence from ``net.line`` ``r0/x0/c0`` columns when present, else from
``pgml.defaults``); the slack becomes a balanced 3-phase Thevenin (angles
``0 / -120 / +120``); a non-empty ``net.asymmetric_load`` table is captured with its
WYE/DELTA connection and per-phase P/Q. The shared scaffold in
:mod:`pgml.convert._common` is the single place the phase decision lives.

ext_grid -> Source
------------------
The slack (ext_grid) is converted to a ``Source`` with a very small Thevenin
impedance (1e-6 Ohm, 1e-12 H) so the Norton stamp is near-zero.  In the oracle
test we use **ideal-slack mode** (``fixed_rows`` / ``v_fixed``) which makes the
Thevenin impedance irrelevant; the Source is still required by the schema so the
slack bus has an appliance. Its zero-sequence source impedance equals the
positive-sequence impedance (no short-circuit data is read here).

Slack voltage phasor stored in id_map
--------------------------------------
``id_map["slack_v_complex"]`` holds the complex slack phasor (in V, LL) as a
Python complex number so the test can pass it directly to ``solve_harmonic``.

``parallel`` (identical parallel systems)
------------------------------------------
A line/two-winding-transformer's ``parallel`` count (pandapower's number of
electrically identical systems in parallel) divides the series impedance and
multiplies the shunt admittance (line C/G, transformer magnetizing conductance/
susceptance) and the rated power (``s_rated_va``), mirroring
``pandapower.build_branch``'s own convention exactly (``_parallel_count``).
``parallel==1`` (pandapower's own default) leaves the series impedance and shunt
admittance unchanged. ``trafo3w`` is not converted at all (see below), so its own
``parallel`` column is moot.

Bus-line / bus-transformer switches (``et='l'``/``'t'``)
------------------------------------------------------------
An OPEN ``et='l'``/``'t'`` switch (``net.switch``) converts its line/transformer
as out-of-service (``_open_switch_targets``): the accepted approximation is that
the WHOLE element drops, not just the switched terminal, so the still-connected
terminal's shunt admittance is lost too (pandapower's own solver instead keeps
that terminal energized via an internal auxiliary bus -- a finer-grained model
this converter does not replicate). A closed switch, or no switch at all, changes
nothing. Bus-bus (``et='b'``) switches are unaffected (still a near-ideal
``Switch`` branch).

Voltage-dependent (ZIP) loads
-----------------------------
``net.load``'s ``const_z_p_percent``/``const_i_p_percent``/``const_z_q_percent``/
``const_i_q_percent`` columns map onto ``ZipCoefficients`` (``_zip_coefficients``),
honoured by pandapower's own ``runpp`` whenever ``voltage_depend_loads=True`` (the
default). All four at zero (pandapower's own default) converts with NO
``zip_coefficients``/``load_model`` set, leaving a plain constant-power load's
conversion unaffected.

Voltage-controlled generators (``net.gen``)
-------------------------------------------
``net.gen`` is pandapower's PV bus: fixed active power, regulated voltage
MAGNITUDE, free reactive power within ``min_q_mvar``/``max_q_mvar``. pgml has no
PV-bus appliance (a PV bus needs a mixed residual row pair ``[P-balance;
|V|^2 - V_set^2]``), so the table is DROPPED by default (``GenMode.DROP`` --
reported by ``warn_dropped_elements``). ``gen_mode=GenMode.VOLT_VAR_APPROX``
opts in to an APPROXIMATION: each row becomes a :class:`Generator` whose
Volt-VAr inverter control is a steep ``Q(|V|)`` droop centred on ``vm_pu`` and
saturating at the row's reactive limits, which holds the bus voltage near the
setpoint with Q as the free variable. See :class:`GenMode` and
``_gen_volt_var_control`` for the exact mapping, the fallbacks, and what the
approximation does and does not give.

Only in-service elements are converted.
"""

from __future__ import annotations

import logging
import math
import re
from enum import Enum
from typing import Any, Optional

from pgml.convert._common import (
    IdCounter,
    PhaseMode,
    build_generator,
    build_line_from_sequence,
    build_load,
    build_node,
    build_source,
    make_metadata,
    phases_for,
    warn_dropped_elements,
)
from pgml.errors import ConversionError
from pgml.schemas.grid_schema import (
    Characteristic,
    ComplexTap,
    Grid,
    Provenance,
    QReference,
    SourceConvention,
    Switch,
    Transformer,
    VoltVarControl,
    WindingConnection,
    ZipCoefficients,
)

_logger = logging.getLogger("pgml")

_TINY_R = 1.0e-6  # Ohm — near-ideal Thevenin for ext_grid in Norton stamp
_TINY_L = 1.0e-12  # H   — near-ideal Thevenin for ext_grid in Norton stamp
_SWITCH_R = 1.0e-4  # Ohm — near-ideal resistance for closed bus-bus switches

#: Reactive envelope [var] used for a ``net.gen`` row that carries neither a
#: reactive limit nor ``sn_mva`` nor active power — no size information at all.
#: Small enough to be inert (such a row regulates nothing); a WARNING names it.
_GEN_ENVELOPE_FLOOR_VAR = 1.0

#: Default Volt-VAr droop steepness for ``GenMode.VOLT_VAR_APPROX``, in units of
#: the generator's reactive base per per-unit terminal voltage (see
#: ``_gen_volt_var_control``). 500 pu/pu means the droop sweeps the full
#: reactive base over 1/500 = 0.002 pu (0.2 %) of terminal voltage — roughly
#: twenty times steeper than a grid-code Volt-VAr characteristic (a 4 % droop)
#: and tight enough that the residual regulation error stays around 1e-3 pu,
#: while the resulting reactive stiffness stays within one to two orders of
#: magnitude of the network's own admittance so Newton still converges.
DEFAULT_GEN_VOLT_VAR_SLOPE_PU = 500.0
_PROVENANCE = Provenance(
    source_convention=SourceConvention.SEQUENCE,
    notes=(
        "Converted from pandapower positive-sequence network. "
        "Engineering units converted to SI."
    ),
)


class GenMode(str, Enum):
    """How ``net.gen`` (pandapower's voltage-controlled generator) is converted.

    ``net.gen`` is a PV bus: fixed active power, regulated voltage MAGNITUDE
    ``vm_pu``, reactive power free between ``min_q_mvar`` and ``max_q_mvar``.
    pgml's appliance model has no PV bus -- that needs a mixed residual row pair
    (``[P-balance; |V|**2 - V_set**2]``) in the power-flow solver and an appliance
    to carry the setpoint.

    ``DROP`` (the default): the table is not read and
    :func:`~pgml.convert._common.warn_dropped_elements` reports it, so a
    transmission benchmark converts to loads plus a slack and its converted
    operating point is NOT the source network's.

    ``VOLT_VAR_APPROX`` opts in to an approximation built from the EXISTING device
    model: a steep Volt-VAr droop centred on ``vm_pu`` and bounded by the row's
    reactive limits. It APPROXIMATES a PV bus -- it does not implement one. See
    ``_gen_volt_var_control`` for the mapping, the fallbacks and the accuracy
    the approximation buys.

    It suits a DISTRIBUTION network with voltage-regulating generation, where it is
    accurate to well under 1e-3 pu and differentiable. It does NOT make a
    transmission benchmark importable: the droop carries no feedback outside a band
    of width ~1/slope, so a stiffly loaded case either fails to converge (loudly) or,
    on some networks, settles on the collapsed low-voltage branch while reporting
    convergence. Check the solved voltages against the source network before trusting
    such an import; a faithful PV bus needs the mixed residual row the solver does not
    have.
    """

    DROP = "drop"
    VOLT_VAR_APPROX = "volt_var_approx"


# IEC vector-group winding tokens (case-insensitive; matched on the lower-cased
# alphabetic prefix of the string). Longest tokens first so "yn"/"zn" are tried
# before their single-letter prefixes "y"/"z".
_WINDING_TOKENS: dict[str, WindingConnection] = {
    "yn": WindingConnection.WYE_GROUNDED,
    "zn": WindingConnection.ZIGZAG_GROUNDED,
    "y": WindingConnection.WYE,
    "d": WindingConnection.DELTA,
    "z": WindingConnection.ZIGZAG,
}
_WINDING_TOKENS_BY_LEN = sorted(_WINDING_TOKENS, key=len, reverse=True)

_VECTOR_GROUP_RE = re.compile(r"^([A-Za-z]+)(\d*)$")


def _parse_vector_group(
    vector_group: str,
) -> tuple[WindingConnection, WindingConnection, Optional[int]]:
    """Parse an IEC vector-group string into ``(from_connection, to_connection, clock)``.

    Handles ``Dyn5``, ``YNd5``, ``Yzn5``, ``Yy0``, ``YNyn0``, ``Dd0``, ``Dyn11``, ...
    pandapower conventionally uppercases the HV token and lowercases the LV token, but
    the split is resolved by exact (case-insensitive) token matching rather than case,
    so unconventional casing still parses: every HV-candidate token (longest first) is
    tried as a prefix of the lower-cased alphabetic part, and a split is accepted only
    when the remainder EXACTLY matches one of the same five tokens.

    The trailing clock digits are OPTIONAL: pandapower's own ``runpp_3ph`` zero-sequence
    transformer model requires the bare letter form (``'Dyn'``, ``'Yzn'``, no digit --
    it explicitly rejects a digit-suffixed string, "specified in net.trafo.shift_degree"),
    so a real pandapower network may carry ``vector_group='Dyn'`` with the clock ONLY in
    ``shift_degree``. Returns ``clock=None`` in that case (no cross-check is possible; the
    caller derives the clock from ``shift_degree`` alone).
    """
    m = _VECTOR_GROUP_RE.match(str(vector_group).strip())
    if not m:
        raise ConversionError(
            f"pandapower vector_group {vector_group!r}: expected letters "
            "optionally followed by a clock number (e.g. 'Dyn5' or 'Dyn')."
        )
    letters, clock_str = m.group(1).lower(), m.group(2)
    clock = int(clock_str) if clock_str else None
    for hv_token in _WINDING_TOKENS_BY_LEN:
        if letters.startswith(hv_token):
            lv_token = letters[len(hv_token) :]
            if lv_token in _WINDING_TOKENS:
                return _WINDING_TOKENS[hv_token], _WINDING_TOKENS[lv_token], clock
    raise ConversionError(
        f"pandapower vector_group {vector_group!r}: could not split {m.group(1)!r} "
        "into an HV/LV winding token pair."
    )


def _vector_group_string(net: Any, row: Any) -> Optional[str]:
    """Return the vector-group string for a ``net.trafo`` row, or ``None``.

    Precedence: an explicit, non-null ``row['vector_group']`` wins; else look up
    ``net.std_types['trafo'][std_type]['vector_group']`` via the row's ``std_type``.
    Returns ``None`` when neither source carries the string (a plain MATPOWER import
    or a benchmark net that only stamps ``shift_degree``).
    """
    vg = row.get("vector_group", None) if hasattr(row, "get") else None
    if vg is not None and not (isinstance(vg, float) and math.isnan(vg)):
        return str(vg)
    std_type = row.get("std_type", None) if hasattr(row, "get") else None
    if std_type is None or (isinstance(std_type, float) and math.isnan(std_type)):
        return None
    std_types = getattr(net, "std_types", None)
    if not std_types:
        return None
    entry = std_types.get("trafo", {}).get(std_type)
    if not entry:
        return None
    vg = entry.get("vector_group")
    return None if vg is None else str(vg)


def _resolve_transformer_connections(
    net: Any, row: Any, shift_deg: float
) -> tuple[WindingConnection, WindingConnection]:
    """Resolve a trafo row's ``(from_connection, to_connection)``.

    A vector-group string (row column or std_type catalog) is parsed and cross-checked
    against ``shift_degree``: pandapower's own balanced ``runpp`` uses only
    ``shift_degree`` (the string is otherwise-unread metadata), so a clock digit that
    disagrees with ``shift_degree`` means the source network is self-inconsistent and
    a silent choice would be wrong for someone reading the other field. A BARE
    vector-group (no clock digit, e.g. ``'Dyn'`` -- the form pandapower's own
    ``runpp_3ph`` zero-sequence model requires) carries no clock to cross-check, so its
    connections are combined with ``shift_degree``'s clock directly. Absent a
    vector-group string anywhere, the connection is derived from the shift parity (see
    the section-3 comment in :func:`to_grid` for the full rationale).
    """
    vg_str = _vector_group_string(net, row)
    if vg_str is not None:
        from_conn, to_conn, clock = _parse_vector_group(vg_str)
        shift_clock = int(round(shift_deg / 30.0)) % 12
        if clock is not None and clock % 12 != shift_clock:
            raise ConversionError(
                f"pandapower trafo: vector_group={vg_str!r} implies clock "
                f"{clock % 12}, but shift_degree={shift_deg} implies clock "
                f"{shift_clock} -- the source network is self-inconsistent "
                "(pandapower's own balanced runpp uses only shift_degree; "
                "vector_group is otherwise-unread metadata, so silently "
                "preferring one would produce a transformer that disagrees "
                "with the other for anyone relying on it). Fix the source "
                "data so the two agree."
            )
        return from_conn, to_conn

    off_clock = abs(shift_deg - round(shift_deg / 30.0) * 30.0)
    if off_clock > 1.0e-6:
        # A MATPOWER-style ideal phase shifter: not a physical vector group.
        # WYE_GROUNDED/WYE_GROUNDED keeps the zero-sequence path transparent; the
        # exact angle is passed through `tap.shift_deg` unconstrained (honoured
        # exactly by the single-phase-equivalent stamp; a genuine 3-phase stamp
        # would reject a non-multiple-of-30 shift, so this fallback is only
        # exact under SINGLE_PHASE_EQUIV).
        return WindingConnection.WYE_GROUNDED, WindingConnection.WYE_GROUNDED

    clock = int(round(shift_deg / 30.0)) % 12
    if clock % 2 == 0:
        # Even clock, no vector-group string: a zero-sequence-transparent
        # sequence-domain import (e.g. MATPOWER case118, shift_degree=0).
        return WindingConnection.WYE_GROUNDED, WindingConnection.WYE_GROUNDED
    # Odd clock, no vector-group string: the physical Dyn reality of most
    # MV/LV distribution transformers (e.g. CIGRE LV/MV, shift_degree=30).
    return WindingConnection.DELTA, WindingConnection.WYE_GROUNDED


def _scaling_factor(row: Any) -> float:
    """Read ``row['scaling']`` NaN-safely (missing/None/NaN -> 1.0, pandapower's own
    default). ``load``/``sgen``/``asymmetric_load`` all carry this per-element
    multiplier; pandapower's own ``runpp`` applies it to the nameplate P/Q before
    solving (``res_load.p_mw = load.p_mw * load.scaling``), so a converter that
    ignored it would silently disagree with the source network whenever a scenario
    sets it away from 1.0 (e.g. ``mv_oberrhein``'s default ``load.scaling=0.6``,
    ``sgen.scaling=0.0``)."""
    f = _opt_float(row, "scaling")
    return 1.0 if f is None else f


def _parallel_count(row: Any) -> float:
    """Read ``row['parallel']`` NaN-safely (missing/None/NaN -> 1.0, pandapower's own
    default number of identical parallel systems). pandapower's own build stage
    divides the series impedance by this count and multiplies the shunt admittance
    (line C/G, transformer magnetizing conductance/susceptance) and the rated power
    by it (see ``pandapower.build_branch``); the converter mirrors that exactly so
    a ``parallel>1`` line/transformer solves identically to ``parallel`` electrically
    identical copies wired in parallel."""
    f = _opt_float(row, "parallel")
    return 1.0 if f is None else f


def _open_switch_targets(net: Any) -> tuple[set[int], set[int]]:
    """Return ``(open_line_pp_indices, open_trafo_pp_indices)`` from ``net.switch``.

    A bus-line (``et='l'``) or bus-transformer (``et='t'``) switch is a per-terminal
    connectivity control: when OPEN it disconnects its element from that one bus.
    The converter's accepted approximation treats ANY open line/trafo switch as
    taking the WHOLE element out of service (equivalent to ``in_service=False`` on
    that line/trafo) -- this also drops the shunt admittance (line charging,
    transformer magnetizing) at the STILL-connected terminal, unlike pandapower's
    own solver, which keeps that terminal energized via an internal auxiliary bus
    (a finer-grained model this converter does not replicate). A closed switch (or
    no switch at all) changes nothing; bus-bus (``et='b'``) switches are handled
    separately (section 4, below).
    """
    open_lines: set[int] = set()
    open_trafos: set[int] = set()
    sw = getattr(net, "switch", None)
    if sw is None or not len(sw):
        return open_lines, open_trafos
    for _, row in sw.iterrows():
        if bool(row.get("closed", True)):
            continue
        et = str(row.get("et", ""))
        if et not in ("l", "t"):
            continue
        element = int(row["element"])
        (open_lines if et == "l" else open_trafos).add(element)
    return open_lines, open_trafos


def _zip_coefficients(row: Any) -> Optional[ZipCoefficients]:
    """Build :class:`~pgml.schemas.grid_schema.ZipCoefficients` from a ``net.load``
    row's ``const_z_p_percent``/``const_i_p_percent``/``const_z_q_percent``/
    ``const_i_q_percent`` columns (NaN-safe, default 0.0 -- pandapower's own default,
    a pure constant-power load). The four percentages independently fraction P and
    Q into constant-impedance/constant-current/constant-power shares (``p_* = 1 -
    z_* - i_*``), matching pandapower's own per-load ZIP model exactly (``runpp``'s
    ``voltage_depend_loads=True`` default; see ``pandapower.build_bus
    ._calc_pq_elements_and_add_on_ppc``). Returns ``None`` when all four percentages
    are zero so a plain constant-power load stays byte-identical (no
    ``zip_coefficients``/``load_model`` stamped)."""
    z_p_pct = _opt_float(row, "const_z_p_percent") or 0.0
    i_p_pct = _opt_float(row, "const_i_p_percent") or 0.0
    z_q_pct = _opt_float(row, "const_z_q_percent") or 0.0
    i_q_pct = _opt_float(row, "const_i_q_percent") or 0.0
    if z_p_pct == 0.0 and i_p_pct == 0.0 and z_q_pct == 0.0 and i_q_pct == 0.0:
        return None
    z_p, i_p = z_p_pct / 100.0, i_p_pct / 100.0
    z_q, i_q = z_q_pct / 100.0, i_q_pct / 100.0
    return ZipCoefficients(
        z_p=z_p,
        i_p=i_p,
        p_p=1.0 - z_p - i_p,
        z_q=z_q,
        i_q=i_q,
        p_q=1.0 - z_q - i_q,
    )


def _gen_reactive_envelope(row: Any, p_w: float) -> float:
    """Fallback reactive envelope [var] for a ``net.gen`` row with a missing limit.

    pandapower leaves ``min_q_mvar``/``max_q_mvar`` unset (NaN) on a generator
    created without limits, and its own ``runpp`` then treats the machine as
    effectively unbounded (``q_lim_default`` = 1e9 MVAr; the default
    ``enforce_q_lims=False`` ignores the limits altogether). An unbounded reactive
    range is not usable here: the Volt-VAr droop needs a FINITE reactive base both
    to size the capability circle and to scale the slope, and an arbitrarily large
    base would make the control arbitrarily stiff. The envelope is therefore sized
    from the machine itself, in this order:

    1. ``sn_mva`` (the apparent-power rating), as the reactive headroom left at the
       present active power, ``sqrt(sn**2 - p**2)`` -- the same capability circle
       the control itself enforces. If the active power already meets or exceeds the
       rating the rating itself is used (a nonzero, machine-sized value).
    2. ``|p_mw|`` when no rating is given: a reactive range equal to the active
       power, i.e. a machine able to run down to a 0.707 displacement power factor.
    3. ``_GEN_ENVELOPE_FLOOR_VAR`` when the row carries no size information at
       all (no limits, no rating, no active power). Such a row cannot regulate; the
       caller logs a WARNING naming it.
    """
    sn_mva = _opt_float(row, "sn_mva")
    if sn_mva is not None and sn_mva > 0.0:
        sn_va = sn_mva * 1.0e6
        headroom_sq = sn_va**2 - p_w**2
        return math.sqrt(headroom_sq) if headroom_sq > 0.0 else sn_va
    return abs(p_w) if p_w != 0.0 else _GEN_ENVELOPE_FLOOR_VAR


def _gen_reactive_bounds(row: Any, p_w: float) -> tuple[float, float, Optional[str]]:
    """Reactive bounds ``(q_min_var, q_max_var, fallback)`` of a ``net.gen`` row.

    ``min_q_mvar``/``max_q_mvar`` are read NaN-safely and converted MVAr -> var; a
    missing bound falls back to -/+ ``_gen_reactive_envelope``. Unlike ``p_mw``,
    the limits are NOT multiplied by ``scaling`` -- pandapower's own
    ``add_q_constraints`` reads them raw while ``p_mw`` is scaled.

    ``fallback`` is ``None`` when both bounds came from the row, else names the
    envelope the synthesised side was sized from -- ``"sn_mva"``, ``"p_mw"``, or
    ``"unsized"`` for a row that carries no size information at all -- so the caller
    can report how many rows were affected and how badly.
    """
    q_min = _opt_float(row, "min_q_mvar")
    q_max = _opt_float(row, "max_q_mvar")
    fallback: Optional[str] = None
    envelope = 0.0
    if q_min is None or q_max is None:
        sn_mva = _opt_float(row, "sn_mva")
        fallback = (
            "sn_mva"
            if sn_mva is not None and sn_mva > 0.0
            else ("p_mw" if p_w != 0.0 else "unsized")
        )
        envelope = _gen_reactive_envelope(row, p_w)
    q_min_var = -envelope if q_min is None else q_min * 1.0e6
    q_max_var = envelope if q_max is None else q_max * 1.0e6
    if q_max_var < q_min_var:
        raise ConversionError(
            f"pandapower gen: min_q_mvar={q_min_var / 1.0e6} exceeds "
            f"max_q_mvar={q_max_var / 1.0e6} -- the source row is inconsistent."
        )
    return q_min_var, q_max_var, fallback


def _gen_volt_var_control(
    row: Any,
    *,
    p_w: float,
    q_min_var: float,
    q_max_var: float,
    n_elem: int,
    slope_pu: float,
) -> tuple[Optional[VoltVarControl], float]:
    """Volt-VAr droop approximating a ``net.gen`` PV bus, plus the fixed ``q_nom_var``.

    ``q_min_var``/``q_max_var`` are the row's reactive bounds in var, already
    resolved (fallbacks applied) by ``_gen_reactive_bounds``.

    Returns ``(control, q_nom_var)``. ``control`` is ``None`` for a row whose
    reactive limits COINCIDE (``min_q_mvar == max_q_mvar``): such a machine has no
    reactive freedom, so it is not a PV bus at all but a plain PQ injection, and
    ``q_nom_var`` carries that fixed reactive power. Otherwise ``control`` is the
    droop and ``q_nom_var`` is 0.0 (a controlled appliance's nameplate reactive
    power is never read -- the control law supplies Q).

    The droop
    ---------
    ``Q(|V|) = clamp(-slope * q_base * (|V|/V0 - vm_pu), q_min, q_max)``, expressed
    as the two-point :class:`Characteristic` the schema stores (x = ``|V|`` in pu of
    the element's nominal voltage, y = ``Q / q_base``) with the endpoints held
    constant outside the range, which IS the clamp. ``q_base`` is the control's
    ``s_rated_va``; it is sized as ``hypot(P, max(|q_min|, |q_max|))`` so that the
    capability circle's remaining reactive headroom ``sqrt(S**2 - P**2)`` equals the
    widest reactive limit exactly -- the asymmetric ``[q_min, q_max]`` saturation
    then comes from the curve, and the (symmetric) circle never binds first. The
    smoothing half-width is 0, i.e. the exact piecewise curve and hard clamp, which
    is what a Q limit means physically.

    ``slope_pu`` is the droop STEEPNESS in units of ``q_base`` per per-unit terminal
    voltage: the droop sweeps one full ``q_base`` of reactive power over ``1 /
    slope_pu`` pu of voltage. Larger = closer to a PV bus, worse conditioned.

    Per-phase scaling
    -----------------
    The control is evaluated PER ELEMENT (per phase for a three-phase appliance,
    against that element's share of the active power), so the reactive limits and
    the rating are divided by ``n_elem``. The per-unit x axis is unaffected: the
    solver forms ``|V_terminal| / V0`` with ``V0`` the node's line-to-neutral (or
    line-to-line for a delta element) nominal, which equals pandapower's ``vm_pu``
    in both phase modes.

    What this does and does not give
    --------------------------------
    - It holds ``|V|`` APPROXIMATELY, not exactly: the regulated bus settles where
      the droop's reactive output balances the network, i.e. off the setpoint by
      ``Q_actual / (slope * q_base)`` in per unit. A steeper slope shrinks that
      offset in inverse proportion and stiffens the power-flow Jacobian in direct
      proportion -- the accuracy/conditioning trade is explicit, not hidden.
    - Reactive limits are enforced by the curve saturation (backed by the capability
      circle), so a generator that runs into its limit behaves like pandapower's
      PV->PQ switch only APPROXIMATELY: the switch happens smoothly along the last
      droop segment instead of discretely, and the post-switch bus voltage is the
      one the saturated Q produces.
    - The active power is a fixed injection, exactly as pandapower models it. There
      is no distributed-slack behaviour and no active-power limit enforcement.
    - The binding limit is CONDITIONING, not steady-state fidelity. Outside the
      ``1 / slope_pu``-wide band ``dQ/d|V|`` is exactly zero, so a Newton iterate
      that starts far from the setpoint sees no voltage-control feedback: on a
      heavily loaded transmission network the solve can fail, or settle on the
      collapsed low-voltage branch with every generator pinned at its maximum Q --
      a genuine second solution of the APPROXIMATED system that an exact
      ``|V| - V_set = 0`` residual row would exclude by construction. Reduce the
      steepness when that happens. Use ``method="newton"``; the current-injection
      fixed point does not contract on a stiff droop.
    """
    v_set_pu = _opt_float(row, "vm_pu")
    if v_set_pu is None:
        v_set_pu = 1.0
    if q_max_var == q_min_var:
        return None, q_max_var

    q_min_elem = q_min_var / n_elem
    q_max_elem = q_max_var / n_elem
    q_span_elem = max(abs(q_min_elem), abs(q_max_elem))
    # q_base = the capability circle whose reactive headroom at the present active
    # power is exactly the widest limit, so the (symmetric) circle bounds |Q| by
    # q_span while the curve applies the asymmetric [q_min, q_max] saturation.
    q_base = math.hypot(p_w / n_elem, q_span_elem)

    y_max = q_max_elem / q_base
    y_min = q_min_elem / q_base
    x_lo = v_set_pu - y_max / slope_pu
    x_hi = v_set_pu - y_min / slope_pu
    if not x_hi > x_lo:
        raise ConversionError(
            f"pandapower gen: gen_volt_var_slope_pu={slope_pu} collapses the droop's "
            "voltage band below floating-point resolution around "
            f"vm_pu={v_set_pu} -- the curve would not be strictly increasing. Use a "
            "shallower droop."
        )
    return (
        VoltVarControl(
            s_rated_va=q_base,
            q_reference=QReference.RATED,
            smoothing=0.0,
            characteristic=Characteristic(
                x_values=(x_lo, x_hi),
                y_values=(y_max, y_min),
            ),
        ),
        0.0,
    )


def _opt_float(row: Any, column: str) -> Optional[float]:
    """NaN-safe optional float read from a pandapower row (missing/None/NaN -> None)."""
    if not hasattr(row, "get"):
        return None
    val = row.get(column, None)
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _tap_ratio_magnitude(row: Any) -> float:
    """Off-nominal tap ratio from pandapower's ``tap_*`` columns (1.0 = no tap).

    ``delta = (tap_pos - tap_neutral) * tap_step_percent / 100``. Verified against a
    live pandapower ``runpp`` on a 2-bus net (see
    ``tests/convert/test_pandapower_vector_groups.py``): a tap on the HV side
    increases the effective HV turns and LOWERS the LV voltage
    (``ratio_magnitude = 1 + delta``, since pgml's tap multiplies the HV/LV ratio); a
    tap on the LV side increases the effective LV turns and RAISES the LV voltage
    (``ratio_magnitude = 1 / (1 + delta)``).

    Any of ``tap_pos``/``tap_neutral``/``tap_step_percent``/``tap_side`` missing (NaN
    or absent) means no tap-changer is configured -> returns 1.0. Only a plain
    ratio changer is modelled by the schema's real-valued ``ratio_magnitude``:
    an ideal / angle-shifting tap raises, whether declared through pandapower 3's
    ``tap_changer_type`` column (anything but ``"Ratio"``), the legacy
    ``tap_phase_shifter`` boolean, or a nonzero ``tap_step_degree``.
    """
    step_deg = _opt_float(row, "tap_step_degree")
    if step_deg is not None and abs(step_deg) > 1.0e-9:
        raise ConversionError(
            f"pandapower trafo: tap_step_degree={step_deg} (an ideal phase-shifter "
            "tap) is not supported; only a real-valued off-nominal tap ratio is "
            "modelled."
        )
    changer_type = row.get("tap_changer_type", None) if hasattr(row, "get") else None
    if isinstance(changer_type, float) and math.isnan(changer_type):
        changer_type = None
    if changer_type is not None and str(changer_type).strip().lower() not in (
        "",
        "none",
        "ratio",
    ):
        raise ConversionError(
            f"pandapower trafo: tap_changer_type={changer_type!r} is not supported; "
            "only a real-valued off-nominal ratio tap ('Ratio') is modelled."
        )
    phase_shifter = (
        row.get("tap_phase_shifter", False) if hasattr(row, "get") else False
    )
    if isinstance(phase_shifter, float) and math.isnan(phase_shifter):
        phase_shifter = False
    if bool(phase_shifter):
        raise ConversionError(
            "pandapower trafo: tap_phase_shifter=True (an ideal phase-shifter tap) "
            "is not supported; only a real-valued off-nominal tap ratio is modelled."
        )

    tap_pos = _opt_float(row, "tap_pos")
    tap_neutral = _opt_float(row, "tap_neutral")
    tap_step_pct = _opt_float(row, "tap_step_percent")
    tap_side = row.get("tap_side", None) if hasattr(row, "get") else None
    if isinstance(tap_side, float) and math.isnan(tap_side):
        tap_side = None
    if (
        tap_pos is None
        or tap_neutral is None
        or tap_step_pct is None
        or tap_side is None
    ):
        return 1.0

    delta = (tap_pos - tap_neutral) * tap_step_pct / 100.0
    side = str(tap_side).strip().lower()
    if side == "hv":
        return 1.0 + delta
    if side == "lv":
        return 1.0 / (1.0 + delta)
    raise ConversionError(
        f"pandapower trafo: tap_side={tap_side!r} not supported (expected 'hv' or 'lv')."
    )


def to_grid(
    net: Any,
    *,
    phase_mode: PhaseMode = PhaseMode.SINGLE_PHASE_EQUIV,
    gen_mode: GenMode = GenMode.DROP,
    gen_volt_var_slope_pu: float = DEFAULT_GEN_VOLT_VAR_SLOPE_PU,
) -> tuple[Grid, dict[str, Any]]:
    """Convert a pandapower network to a :class:`~pgml.schemas.grid_schema.Grid`.

    Parameters
    ----------
    net:
        A pandapower network object (the result of e.g.
        ``pandapower.networks.case33bw()``). The network must already have
        basic DataFrames (``bus``, ``line``, ``load``, ``ext_grid``).
        Unmaterialised std_type references in lines are accepted as long as
        explicit per-km parameters are present.
    phase_mode:
        :class:`~pgml.convert._common.PhaseMode`. ``SINGLE_PHASE_EQUIV`` (default)
        reproduces the positive-sequence single-phase-equivalent output exactly;
        ``THREE_PHASE`` expands to a genuine abc grid (sequence->phase line
        matrices, balanced 3-phase source, asymmetric-load capture). Transformers
        are vector-group aware in BOTH modes: ``SINGLE_PHASE_EQUIV`` folds the
        winding connections + clock into the classical scalar off-nominal-tap pi
        (magnitude + shift only, no topology); ``THREE_PHASE`` builds the full
        phase-domain winding-incidence stamp (delta/zigzag phase coupling and
        zero-sequence blocking), so the two modes agree on the positive-sequence
        terminal admittance for every supported connection pair. The only
        THREE_PHASE approximation is in the LINE model (sequence-expanded 3x3
        matrices introduce zero-sequence mutual coupling from config defaults
        when the dataset carries no native ``r0``/``x0``/``c0``), not the
        transformer.
    gen_mode:
        :class:`GenMode`. ``DROP`` (default) leaves ``net.gen`` unread and reports
        it as a dropped element. ``VOLT_VAR_APPROX`` converts each in-service row to a
        :class:`~pgml.schemas.grid_schema.Generator` whose Volt-VAr control
        approximates the PV bus (see :class:`GenMode` and
        ``_gen_volt_var_control``).
    gen_volt_var_slope_pu:
        Droop steepness of that approximation, in units of the generator's reactive
        base per per-unit terminal voltage (default
        :data:`DEFAULT_GEN_VOLT_VAR_SLOPE_PU`): the droop sweeps one full reactive
        base over ``1 / gen_volt_var_slope_pu`` pu of voltage. Steeper holds the
        setpoint more tightly and stiffens the power-flow Jacobian in direct
        proportion; too steep and the solve fails or settles on the collapsed
        low-voltage branch, which is the signal to reduce it. Solve the resulting
        grid with ``solve_power_flow(..., method="newton")`` -- the
        current-injection fixed point (the default, and what
        :func:`pgml.simulate` uses) does not contract on a stiff droop. Read only
        under ``gen_mode=GenMode.VOLT_VAR_APPROX``.

    Returns
    -------
    tuple[Grid, dict]
        A ``(Grid, id_map)`` pair.  ``Grid`` is the materialised schema object
        (no ``type_ref``).  ``id_map`` maps source element tables to our ids:
        ``"bus"`` -> ``{pp_bus_idx: Node.id}``,
        ``"line"`` -> ``{pp_line_idx: Line.id}``,
        ``"load"`` -> ``{pp_load_idx: Load.id}``,
        ``"asymmetric_load"`` -> ``{pp_asym_idx: Load.id}`` (THREE_PHASE only),
        ``"sgen"`` -> ``{pp_sgen_idx: Generator.id}``,
        ``"gen"`` -> ``{pp_gen_idx: Generator.id}`` (empty unless ``gen_mode`` is
        ``VOLT_VAR_APPROX``),
        ``"ext_grid"`` -> ``{pp_eg_idx: Source.id}``,
        ``"slack_v_complex"`` -> complex slack voltage phasor (V, LL) for
        ideal-slack mode.
    """
    if gen_volt_var_slope_pu <= 0.0:
        raise ConversionError(
            f"gen_volt_var_slope_pu={gen_volt_var_slope_pu} must be positive "
            "(it is a droop steepness in reactive base per per-unit voltage)."
        )
    f0_hz: float = float(getattr(net, "f_hz", 50.0))
    two_pi_f0 = 2.0 * math.pi * f0_hz

    _id = IdCounter()

    id_map: dict[str, Any] = {
        "bus": {},
        "line": {},
        "trafo": {},
        "switch": {},
        "load": {},
        "asymmetric_load": {},
        "sgen": {},
        "gen": {},
        "ext_grid": {},
        "slack_v_complex": None,
    }

    # ------------------------------------------------------------------ #
    # 1. Nodes (buses)                                                     #
    # ------------------------------------------------------------------ #
    nodes: list = []
    for pp_idx, row in net.bus.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        node_id = _id.next()
        id_map["bus"][pp_idx] = node_id
        nodes.append(
            build_node(
                id=node_id,
                name=str(row.get("name", f"bus_{pp_idx}") or f"bus_{pp_idx}"),
                u_rated_v=float(row["vn_kv"]) * 1_000.0,
                mode=phase_mode,
            )
        )

    # ------------------------------------------------------------------ #
    # 2. Lines                                                             #
    # ------------------------------------------------------------------ #
    # `open_line_switches`/`open_trafo_switches` (section 4's `_open_switch_targets`,
    # computed once up front so both element loops can see it): a line/trafo with
    # ANY open bus-element switch converts as out-of-service, same as
    # `in_service=False` -- see `_open_switch_targets`'s docstring for the accepted
    # approximation this implies (the still-connected terminal's shunt is dropped
    # with it).
    open_line_switches, open_trafo_switches = _open_switch_targets(net)

    branches: list = []
    for pp_idx, row in net.line.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        if pp_idx in open_line_switches:
            continue
        from_bus = int(row["from_bus"])
        to_bus = int(row["to_bus"])
        # Skip lines whose buses were not converted (e.g. out-of-service buses)
        if from_bus not in id_map["bus"] or to_bus not in id_map["bus"]:
            continue

        line_id = _id.next()
        id_map["line"][pp_idx] = line_id

        length_m = float(row["length_km"]) * 1_000.0

        # `parallel` identical systems: series impedance divides by the count,
        # shunt admittance (C/G) multiplies by it (pandapower.build_branch's own
        # convention -- see `_parallel_count`).
        parallel = _parallel_count(row)

        # Per-length positive-sequence SI parameters (1/m)
        r1 = float(row["r_ohm_per_km"]) / 1_000.0 / parallel  # Ohm/m
        x1 = float(row["x_ohm_per_km"]) / 1_000.0 / parallel  # Ohm/m (=2*pi*f0*L per m)
        c1 = (
            float(row.get("c_nf_per_km", 0.0) or 0.0) * 1.0e-9 / 1_000.0 * parallel
        )  # F/m
        g1 = (
            float(row.get("g_us_per_km", 0.0) or 0.0) * 1.0e-6 / 1_000.0 * parallel
        )  # S/m

        # Native zero-sequence columns (THREE_PHASE only; else config defaults);
        # the same parallel factor applies (pandapower's own pd2ppc_zero mirrors
        # the positive-sequence r_ohm_per_km/x_ohm_per_km/c_nf_per_km convention).
        r0 = _opt_per_km(row, "r0_ohm_per_km", 1_000.0)
        x0 = _opt_per_km(row, "x0_ohm_per_km", 1_000.0)
        c0_nf = _opt_per_km(row, "c0_nf_per_km", None)
        c0 = c0_nf * 1.0e-12 if c0_nf is not None else None
        if r0 is not None:
            r0 /= parallel
        if x0 is not None:
            x0 /= parallel
        if c0 is not None:
            c0 *= parallel

        branches.append(
            build_line_from_sequence(
                id=line_id,
                name=str(row.get("name", f"line_{pp_idx}") or f"line_{pp_idx}"),
                from_node=id_map["bus"][from_bus],
                to_node=id_map["bus"][to_bus],
                mode=phase_mode,
                length_m=length_m,
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
    # 3. Transformers (two-winding, vector-group + tap-changer aware)      #
    # ------------------------------------------------------------------ #
    # Winding connections come from `net.trafo['vector_group']` when the       #
    # column exists and is set for the row, else from                         #
    # `net.std_types['trafo'][std_type]['vector_group']`; parsed by            #
    # `_resolve_transformer_connections` (Dyn5, YNd5, Yzn5, Yy0, YNyn0, Dd0,   #
    # Dyn11, ...). The clock digit is OPTIONAL: `runpp_3ph`'s own zero-        #
    # sequence transformer model requires the bare letter form ('Dyn', 'Yzn', #
    # no digit -- it explicitly rejects a digit-suffixed string), so a real   #
    # network may carry `vector_group='Dyn'` with the clock only in           #
    # `shift_degree`; the bare form skips the cross-check below (there is     #
    # nothing to cross-check) and combines directly with `shift_degree`.      #
    # When neither source carries a vector-group string (plain                #
    # MATPOWER imports such as case118, or a benchmark net that only stamps   #
    # `shift_degree`) the connection is DERIVED from the shift parity: an     #
    # even clock (or shift_degree==0) is stamped WYE_GROUNDED/WYE_GROUNDED (a #
    # zero-sequence-transparent sequence-domain import — physically arbitrary #
    # but harmless for a positive-sequence-only study), an odd clock is       #
    # stamped DELTA/WYE_GROUNDED (the physical Dyn reality of most MV/LV      #
    # distribution transformers, e.g. CIGRE LV/MV). A shift that is not a     #
    # multiple of 30° (a MATPOWER ideal phase shifter) also falls back to     #
    # WYE_GROUNDED/WYE_GROUNDED with the exact angle passed through           #
    # unconstrained (honoured exactly by the single-phase-equivalent stamp;   #
    # `resolve_vector_group` rejects it under a genuine 3-phase stamp). A     #
    # vector-group string whose clock digit disagrees with `shift_degree` is  #
    # an inconsistent source network — this is a LOUD error, not a silent     #
    # pick, because pandapower's own balanced `runpp` uses only               #
    # `shift_degree` while the string is unread metadata; preferring either   #
    # source would silently disagree with the other for someone relying on    #
    # it (see `_resolve_transformer_connections`).                            #
    #                                                                          #
    # The nominal turns ratio and the vector-group phase shift come from the  #
    # rated voltages (`u_rated_from/to_v`) plus the winding connections, so    #
    # `tap.ratio_magnitude` carries the OFF-NOMINAL tap-changer deviation      #
    # (1.0 = no tap / at neutral) read from `tap_pos`/`tap_neutral`/           #
    # `tap_step_percent`/`tap_side` (`_tap_ratio_magnitude`; NaN-safe — a      #
    # trafo with no tap-changer columns set converts at 1.0). An ideal         #
    # phase-shifter tap (`tap_step_degree` nonzero or `tap_phase_shifter`      #
    # True) is not modelled and raises.                                       #
    #                                                                          #
    # Assembly builds the winding-incidence primitive Y = N^T Y_winding N: a  #
    # delta or zigzag winding blocks the zero sequence (traps triplen         #
    # harmonics) and supplies the intrinsic √3 (delta) or unit (zigzag)       #
    # magnitude + clock shift; `SINGLE_PHASE_EQUIV` collapses this to the     #
    # classical positive-sequence off-nominal-tap pi -- an EXACT reduction    #
    # of the same physical transformer, not a different model (see           #
    # `docs/pgml/modeling/transformer.md`). Leakage (`vk_percent`/            #
    # `vkr_percent`) is the TERMINAL (line-to-line-equivalent) impedance      #
    # referred to the LV side (`Z_LL = vk% * Z_base_LV`); the schema field    #
    # is always the TO-side COIL impedance, `3 * Z_LL` when the LV winding    #
    # is DELTA and `Z_LL` otherwise (`test_transformer_clock_matrix.py`'s     #
    # pinned "coil vs terminal" relation) -- applied REGARDLESS of            #
    # `phase_mode`: assembly itself undoes the factor for the terminal        #
    # admittance in BOTH modes (the p==1 scalar stamp via its own internal    #
    # `k_ll` factor in `_transformer_block_groups`, the p==3 stamp via the    #
    # winding-incidence transform), so the converter's job is only to supply  #
    # the coil-referred value, once, the same way for every phase mode.       #
    # Verified against a live pandapower runpp on a YNd5 transformer          #
    # (mv_oberrhein's '25 MVA 110/20 kV' std type): the assembled Y-bus       #
    # transformer entries match pandapower's own internal Ybus to machine     #
    # precision with the factor applied, and are wrong by 3x without it --    #
    # see `tests/reference/test_pandapower_grid_matrix.py`.                   #
    # ------------------------------------------------------------------ #
    if hasattr(net, "trafo") and len(net.trafo):
        for pp_idx, row in net.trafo.iterrows():
            if not bool(row.get("in_service", True)):
                continue
            if pp_idx in open_trafo_switches:
                continue
            hv_bus = int(row["hv_bus"])
            lv_bus = int(row["lv_bus"])
            if hv_bus not in id_map["bus"] or lv_bus not in id_map["bus"]:
                continue

            trafo_id = _id.next()
            id_map["trafo"][pp_idx] = trafo_id

            sn_va = float(row["sn_mva"]) * 1.0e6  # VA (single-unit rating)
            vn_hv_v = float(row["vn_hv_kv"]) * 1.0e3  # V
            vn_lv_v = float(row["vn_lv_kv"]) * 1.0e3  # V
            vk_pct = float(row["vk_percent"])
            vkr_pct = float(row["vkr_percent"])
            pfe_w = float(row.get("pfe_kw", 0.0) or 0.0) * 1.0e3  # W
            i0_pct = float(row.get("i0_percent", 0.0) or 0.0)
            shift_deg = float(row.get("shift_degree", 0.0) or 0.0)

            # `parallel` identical units: the leakage impedance is computed from
            # the SINGLE-unit `sn_mva` (pandapower's own convention, see
            # `_calc_r_x_from_dataframe`), then the RESULT divides by the count;
            # the magnetizing admittance (conductance/susceptance) and the rated
            # power multiply by it (`_calc_y_from_dataframe`) -- see
            # `_parallel_count`.
            parallel = _parallel_count(row)

            from_connection, to_connection = _resolve_transformer_connections(
                net, row, shift_deg
            )

            # Terminal (line-to-line-equivalent) leakage impedance referred to
            # the LV side: Z_LL = vk% * Z_base_LV, Z_base_LV = vn_lv_v^2/sn_va.
            z_base_lv = vn_lv_v**2 / sn_va
            z_ll = vk_pct / 100.0 * z_base_lv
            r_ll = vkr_pct / 100.0 * z_base_lv
            x_ll_sq = z_ll**2 - r_ll**2
            x_ll = math.sqrt(max(x_ll_sq, 0.0))
            l_ll = x_ll / two_pi_f0

            # TO-side coil referral (see the section comment above): the schema
            # field is ALWAYS coil-referred; assembly itself undoes the factor
            # for both phase modes (the p==1 scalar stamp via the `k_ll` factor
            # in `_transformer_block_groups`, the p==3 stamp via the
            # winding-incidence transform), so the converter applies it
            # unconditionally, regardless of `phase_mode`. The `parallel` divide
            # is applied last (order-independent relative to the coil factor).
            coil_factor = 3.0 if to_connection is WindingConnection.DELTA else 1.0
            r_sc = coil_factor * r_ll / parallel
            l_sc = coil_factor * l_ll / parallel

            # Magnetizing branch (referred to HV side; added to the HV diagonal).
            # `parallel` identical units contribute `parallel` times the single-
            # unit admittance: conductance multiplies directly; the equivalent
            # inductance (stored field) divides so its susceptance
            # (1/(2*pi*f0*L)) multiplies by `parallel` too.
            if pfe_w > 0.0:
                g_m = pfe_w / (vn_hv_v**2) * parallel
            else:
                g_m = 0.0

            l_m = None
            if i0_pct > 0.0:
                i0_amp = i0_pct / 100.0 * sn_va / vn_hv_v
                s_nl = vn_hv_v * i0_amp  # VA
                q_nl_sq = s_nl**2 - pfe_w**2
                if q_nl_sq > 0.0:
                    b_m = math.sqrt(q_nl_sq) / (vn_hv_v**2)
                    if b_m > 0.0:
                        l_m = 1.0 / (two_pi_f0 * b_m * parallel)

            tap_ratio = _tap_ratio_magnitude(row)

            tx_phases = phases_for(phase_mode)
            branches.append(
                Transformer(
                    id=trafo_id,
                    name=str(row.get("name", f"trafo_{pp_idx}") or f"trafo_{pp_idx}"),
                    from_node=id_map["bus"][hv_bus],  # from = HV side
                    to_node=id_map["bus"][lv_bus],  # to   = LV side
                    from_phases=tx_phases,
                    to_phases=tx_phases,
                    s_rated_va=sn_va * parallel,
                    u_rated_from_v=vn_hv_v,
                    u_rated_to_v=vn_lv_v,
                    from_connection=from_connection,
                    to_connection=to_connection,
                    series_resistance_ohm=r_sc,
                    series_inductance_h=l_sc,
                    magnetizing_conductance_s=g_m,
                    magnetizing_inductance_h=l_m,
                    # Nominal ratio comes from u_rated + connections; `tap` is the
                    # off-nominal tap-changer ratio plus the vector-group clock angle.
                    tap=ComplexTap(ratio_magnitude=tap_ratio, shift_deg=shift_deg),
                    provenance=_PROVENANCE,
                )
            )

    # ------------------------------------------------------------------ #
    # 4. Bus-bus switches (et='b', closed=True -> near-ideal Switch).      #
    #    Bus-line/bus-transformer switches (et='l'/'t') were already        #
    #    resolved up front (`_open_switch_targets`, used by sections 2/3):  #
    #    an OPEN one takes the whole line/trafo out of service.             #
    # ------------------------------------------------------------------ #
    if hasattr(net, "switch") and len(net.switch):
        for pp_idx, row in net.switch.iterrows():
            if str(row.get("et", "")) != "b":
                continue  # only bus-bus switches (l/t handled up front)
            if not bool(row.get("closed", True)):
                continue  # open switch: no branch
            bus_from = int(row["bus"])
            bus_to = int(row["element"])
            if bus_from not in id_map["bus"] or bus_to not in id_map["bus"]:
                continue

            sw_id = _id.next()
            id_map["switch"][pp_idx] = sw_id
            z_ohm = float(row.get("z_ohm", 0.0) or 0.0)
            r_sw = z_ohm if z_ohm > 0.0 else _SWITCH_R
            sw_phases = phases_for(phase_mode)
            branches.append(
                Switch(
                    id=sw_id,
                    name=str(row.get("name", f"switch_{pp_idx}") or f"switch_{pp_idx}"),
                    from_node=id_map["bus"][bus_from],
                    to_node=id_map["bus"][bus_to],
                    from_phases=sw_phases,
                    to_phases=sw_phases,
                    closed=True,
                    resistance_ohm=r_sw,
                    inductance_h=0.0,
                    provenance=_PROVENANCE,
                )
            )

    # ------------------------------------------------------------------ #
    # 5. ext_grid -> Source (Thevenin with near-zero Z; ideal-slack mode   #
    #    overrides this in the solver)                                     #
    # ------------------------------------------------------------------ #
    appliances: list = []
    for pp_idx, row in net.ext_grid.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        bus_pp = int(row["bus"])
        if bus_pp not in id_map["bus"]:
            continue

        src_id = _id.next()
        id_map["ext_grid"][pp_idx] = src_id

        vm_pu = float(row.get("vm_pu", 1.0))
        va_deg = float(row.get("va_degree", 0.0))
        u_rated_v = float(net.bus.at[bus_pp, "vn_kv"]) * 1_000.0
        u_ref_v = vm_pu * u_rated_v  # magnitude of the slack phasor (LL)

        # First slack wins: a network may have several ext_grids, but the ideal
        # slack solve takes a single fixed phasor. Guard so the last ext_grid does
        # not silently overwrite it (matches the pgm converter).
        if id_map["slack_v_complex"] is None:
            id_map["slack_v_complex"] = u_ref_v * complex(
                math.cos(math.radians(va_deg)), math.sin(math.radians(va_deg))
            )

        appliances.append(
            build_source(
                id=src_id,
                name=str(row.get("name", f"ext_grid_{pp_idx}") or f"ext_grid_{pp_idx}"),
                node=id_map["bus"][bus_pp],
                mode=phase_mode,
                u_ref_v=u_ref_v,
                u_angle_deg=va_deg,
                r_ohm=_TINY_R,
                l_h=_TINY_L,
            )
        )

    # ------------------------------------------------------------------ #
    # 6. Loads (balanced net.load)                                         #
    # ------------------------------------------------------------------ #
    # Voltage-dependent (ZIP) fractions: `const_z_p_percent`/`const_i_p_percent`/
    # `const_z_q_percent`/`const_i_q_percent` (`_zip_coefficients`) map onto
    # `ZipCoefficients`, honoured by pandapower's own `runpp` whenever
    # `voltage_depend_loads=True` (the default). A load with all four percentages
    # at zero (pandapower's own default -- a pure constant-power load) converts
    # with NO `zip_coefficients`/`load_model` set, leaving a plain constant-power
    # load's conversion unaffected.
    for pp_idx, row in net.load.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        bus_pp = int(row["bus"])
        if bus_pp not in id_map["bus"]:
            continue

        load_id = _id.next()
        id_map["load"][pp_idx] = load_id

        scaling = _scaling_factor(row)
        p_w = float(row["p_mw"]) * 1.0e6 * scaling
        q_var = float(row["q_mvar"]) * 1.0e6 * scaling

        # Balanced total: connection=None resolves to WYE from config; under
        # THREE_PHASE the symmetric/auto calc splits the total equally.
        appliances.append(
            build_load(
                id=load_id,
                name=str(row.get("name", f"load_{pp_idx}") or f"load_{pp_idx}"),
                node=id_map["bus"][bus_pp],
                mode=phase_mode,
                p_total_w=p_w,
                q_total_var=q_var,
                zip_coefficients=_zip_coefficients(row),
            )
        )

    # ------------------------------------------------------------------ #
    # 7. Asymmetric loads (net.asymmetric_load) — genuine per-phase split  #
    # ------------------------------------------------------------------ #
    # Under THREE_PHASE each asymmetric_load becomes an abc Load carrying    #
    # its WYE/DELTA connection and the per-phase P/Q. Under SINGLE_PHASE_    #
    # EQUIV the per-phase split cannot be represented, so the three phases   #
    # are summed into a balanced 1-phase total (logged at INFO).             #
    # ------------------------------------------------------------------ #
    asym = getattr(net, "asymmetric_load", None)
    if asym is not None and len(asym):
        for pp_idx, row in asym.iterrows():
            if not bool(row.get("in_service", True)):
                continue
            bus_pp = int(row["bus"])
            if bus_pp not in id_map["bus"]:
                continue

            scaling = _scaling_factor(row)
            p_a = float(row.get("p_a_mw", 0.0) or 0.0) * 1.0e6 * scaling
            p_b = float(row.get("p_b_mw", 0.0) or 0.0) * 1.0e6 * scaling
            p_c = float(row.get("p_c_mw", 0.0) or 0.0) * 1.0e6 * scaling
            q_a = float(row.get("q_a_mvar", 0.0) or 0.0) * 1.0e6 * scaling
            q_b = float(row.get("q_b_mvar", 0.0) or 0.0) * 1.0e6 * scaling
            q_c = float(row.get("q_c_mvar", 0.0) or 0.0) * 1.0e6 * scaling
            p_total = p_a + p_b + p_c
            q_total = q_a + q_b + q_c
            conn = (
                WindingConnection.DELTA
                if str(row.get("type", "wye")).lower() == "delta"
                else WindingConnection.WYE
            )

            load_id = _id.next()
            id_map["asymmetric_load"][pp_idx] = load_id
            name = str(row.get("name", f"asym_load_{pp_idx}") or f"asym_load_{pp_idx}")

            if phase_mode is PhaseMode.SINGLE_PHASE_EQUIV:
                _logger.info(
                    "pandapower asymmetric_load %s collapsed to a balanced "
                    "single-phase total under SINGLE_PHASE_EQUIV "
                    "(per-phase split discarded); use THREE_PHASE to keep it.",
                    pp_idx,
                )
                appliances.append(
                    build_load(
                        id=load_id,
                        name=name,
                        node=id_map["bus"][bus_pp],
                        mode=phase_mode,
                        p_total_w=p_total,
                        q_total_var=q_total,
                    )
                )
            else:
                appliances.append(
                    build_load(
                        id=load_id,
                        name=name,
                        node=id_map["bus"][bus_pp],
                        mode=phase_mode,
                        p_total_w=p_total,
                        q_total_var=q_total,
                        connection=conn,
                        p_per_phase_w=(p_a, p_b, p_c),
                        q_per_phase_var=(q_a, q_b, q_c),
                    )
                )

    # ------------------------------------------------------------------ #
    # 8. Static generators (net.sgen) -> Generator (PQ injection)          #
    # ------------------------------------------------------------------ #
    # pandapower sgen is GENERATION-POSITIVE (p_mw > 0 injects), matching the
    # Generator nameplate convention; the assembly applies the injection sign.
    sgen = getattr(net, "sgen", None)
    if sgen is not None and len(sgen):
        for pp_idx, row in sgen.iterrows():
            if not bool(row.get("in_service", True)):
                continue
            bus_pp = int(row["bus"])
            if bus_pp not in id_map["bus"]:
                continue
            gen_id = _id.next()
            id_map["sgen"][pp_idx] = gen_id
            scaling = _scaling_factor(row)
            appliances.append(
                build_generator(
                    id=gen_id,
                    name=str(row.get("name", f"sgen_{pp_idx}") or f"sgen_{pp_idx}"),
                    node=id_map["bus"][bus_pp],
                    mode=phase_mode,
                    p_total_w=float(row["p_mw"]) * 1.0e6 * scaling,
                    q_total_var=float(row.get("q_mvar", 0.0) or 0.0) * 1.0e6 * scaling,
                )
            )

    # ------------------------------------------------------------------ #
    # 9. Voltage-controlled generators (net.gen) -> Volt-VAr approximation #
    #    OPT-IN: only under `gen_mode=GenMode.VOLT_VAR_APPROX`; the default #
    #    `GenMode.DROP` leaves the table unread (warned below).             #
    # ------------------------------------------------------------------ #
    # pandapower `gen` is GENERATION-POSITIVE, like `sgen`, and carries the same
    # per-row `scaling` multiplier on `p_mw` (its reactive LIMITS are read raw --
    # pandapower's own `add_q_constraints` does not scale them). The regulated
    # voltage `vm_pu` centres the droop; `min_q_mvar`/`max_q_mvar` saturate it.
    #
    # SLACK-BUS RULE: a row on a bus that already carries an in-service `ext_grid`
    # is SKIPPED, as is a row flagged `slack=True`. The ideal slack fixes that
    # bus's voltage phasor outright, so a droop there would be a regulator fighting
    # an infinitely stiff reference -- and it would be redundant: pandapower's own
    # solve treats a generator at the reference bus the same way, absorbing its
    # `p_mw` into the slack dispatch rather than injecting it. Every skip is logged
    # at WARNING with the row index, because the row's active power really is
    # dropped from the converted grid.
    gen = getattr(net, "gen", None)
    if gen_mode is GenMode.VOLT_VAR_APPROX and gen is not None and len(gen):
        slack_buses = {
            int(r["bus"])
            for _, r in net.ext_grid.iterrows()
            if bool(r.get("in_service", True))
        }
        n_elem = len(phases_for(phase_mode))
        n_fallback_limits = 0
        n_unsized = 0
        for pp_idx, row in gen.iterrows():
            if not bool(row.get("in_service", True)):
                continue
            bus_pp = int(row["bus"])
            if bus_pp not in id_map["bus"]:
                continue
            is_slack_row = bool(row.get("slack", False))
            if bus_pp in slack_buses or is_slack_row:
                _logger.warning(
                    "pandapower gen %s at bus %s is %s and is NOT converted: the "
                    "slack fixes that bus's voltage, so its %.3f MW injection is "
                    "absorbed into the slack dispatch (pandapower's own solve "
                    "treats a generator at the reference bus the same way).",
                    pp_idx,
                    bus_pp,
                    "flagged slack=True" if is_slack_row else "the ext_grid bus",
                    float(row["p_mw"]),
                )
                continue

            scaling = _scaling_factor(row)
            p_w = float(row["p_mw"]) * 1.0e6 * scaling
            q_min_var, q_max_var, fallback = _gen_reactive_bounds(row, p_w)
            if fallback is not None:
                n_fallback_limits += 1
                n_unsized += fallback == "unsized"
            control, q_var = _gen_volt_var_control(
                row,
                p_w=p_w,
                q_min_var=q_min_var,
                q_max_var=q_max_var,
                n_elem=n_elem,
                slope_pu=gen_volt_var_slope_pu,
            )

            gen_id = _id.next()
            id_map["gen"][pp_idx] = gen_id
            appliances.append(
                build_generator(
                    id=gen_id,
                    name=str(row.get("name", f"gen_{pp_idx}") or f"gen_{pp_idx}"),
                    node=id_map["bus"][bus_pp],
                    mode=phase_mode,
                    p_total_w=p_w,
                    q_total_var=q_var,
                    control=control,
                )
            )

        if id_map["gen"]:
            _logger.info(
                "pandapower -> Grid: %d 'gen' row(s) converted as a Volt-VAr "
                "APPROXIMATION of a PV bus (droop steepness %.4g pu/pu centred on "
                "each row's vm_pu, saturating at its reactive limits). The bus "
                "voltage is held NEAR, not at, the setpoint.",
                len(id_map["gen"]),
                gen_volt_var_slope_pu,
            )
        if n_fallback_limits:
            _logger.warning(
                "pandapower -> Grid: %d converted 'gen' row(s) carry no "
                "min_q_mvar/max_q_mvar; the reactive range was synthesised from "
                "sn_mva (or, failing that, from |p_mw|) — see "
                "`_gen_reactive_envelope`. %d of them carry no size information at "
                "all and therefore regulate nothing.",
                n_fallback_limits,
                n_unsized,
            )

    # Elements the converter does NOT read: fail loud, never silently wrong.
    warn_dropped_elements(
        _logger,
        "pandapower",
        {
            kind: len(tbl)
            for kind in (
                *(("gen",) if gen_mode is GenMode.DROP else ()),
                "shunt",
                "trafo3w",
                "impedance",
                "ward",
                "xward",
                "dcline",
                "storage",
                "motor",
                "asymmetric_sgen",
            )
            if (tbl := getattr(net, kind, None)) is not None
        },
    )

    description = f"Imported from pandapower (f0={f0_hz} Hz). " + (
        "Single-phase positive-sequence equivalent."
        if phase_mode is PhaseMode.SINGLE_PHASE_EQUIV
        else "Three-phase (abc) expansion from sequence quantities."
    )
    grid = Grid(
        base_frequency_hz=f0_hz,
        nodes=nodes,
        branches=branches,
        appliances=appliances,
        metadata=make_metadata(
            name=str(getattr(net, "name", "") or "pandapower_import"),
            description=description,
        ),
    )
    return grid, id_map


def _opt_per_km(row: Any, column: str, divisor: float | None) -> float | None:
    """Read an optional per-km column from a pandapower line row, SI-scaled.

    Returns ``None`` when the column is absent or NaN (so the line falls back to
    config zero-sequence defaults). ``divisor`` converts per-km -> per-m when given
    (e.g. ``r0_ohm_per_km`` / 1000); pass ``None`` to leave the raw value (the
    caller scales it, e.g. nF -> F).
    """
    if not hasattr(row, "get"):
        return None
    val = row.get(column, None)
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f / divisor if divisor is not None else f


__all__ = ["to_grid", "GenMode", "DEFAULT_GEN_VOLT_VAR_SLOPE_PU"]
