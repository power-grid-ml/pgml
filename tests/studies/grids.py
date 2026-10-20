"""Small grids for the planning-study tests, each with a known closed form or rating."""

from __future__ import annotations

import math

from pgml.schemas.grid_schema import (
    ComplexTap,
    Grid,
    Line,
    LineType,
    Load,
    Node,
    Phase,
    ShuntAppliance,
    Source,
    Switch,
    Transformer,
    TransformerType,
    TypeLibrary,
    WindingConnection,
)

ABC = (Phase.A, Phase.B, Phase.C)
A = (Phase.A,)
F0 = 50.0


def _diag(x: float, n: int) -> list[list[float]]:
    return [[x if i == j else 0.0 for j in range(n)] for i in range(n)]


def two_bus(
    *,
    r_ohm: float = 0.2,
    x_ohm: float = 0.1,
    u_ln_v: float = 230.0,
    phases: tuple = A,
    i_max_a: float | None = None,
) -> Grid:
    """Ideal source bus 1 -- line (R + jX per conductor, no shunt) -- bus 2.

    Single-phase by default (``u_rated_v`` is then the line-to-neutral voltage); with
    ``phases=ABC`` a balanced three-phase line without mutual coupling, so the
    per-phase equivalent is the single-phase case at a third of the power. The line
    carries the catalog type ``"cable"`` whose ``max_i_a`` is ``i_max_a``.
    """
    n = len(phases)
    u_rated = u_ln_v if n == 1 else u_ln_v * math.sqrt(3.0)
    l_h = x_ohm / (2.0 * math.pi * F0)
    return Grid(
        base_frequency_hz=F0,
        nodes=[Node(id=i, u_rated_v=u_rated, phases=phases) for i in (1, 2)],
        branches=[
            Line(
                id=10,
                from_node=1,
                to_node=2,
                from_phases=phases,
                to_phases=phases,
                length_m=1.0,
                type_ref="cable",
                series_resistance_ohm_per_m=_diag(r_ohm, n),
                series_inductance_h_per_m=_diag(l_h, n),
                shunt_capacitance_f_per_m=_diag(0.0, n),
                harmonic_line_model="naive",
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=phases,
                u_ref_v=tuple(u_ln_v for _ in phases),
                u_angle_deg=tuple((0.0, -120.0, 120.0)[: len(phases)]),
                resistance_ohm=_diag(1.0e-3, n),
                inductance_h=_diag(1.0e-6, n),
            )
        ],
        types=TypeLibrary(
            lines={
                "cable": LineType(
                    id="cable",
                    n_phases=n,
                    series_resistance_ohm_per_m=_diag(r_ohm, n),
                    series_inductance_h_per_m=_diag(l_h, n),
                    shunt_capacitance_f_per_m=_diag(0.0, n),
                    max_i_a=i_max_a,
                )
            }
        ),
    )


def lv_feeder_with_switch(*, i_max_a: float = 250.0) -> Grid:
    """Three-phase 400 V feeder: source 1 -- ideal switch -- 2 -- cable -- 3 -- cable -- 4.

    The switch is closed with zero impedance, so the solve fuses nodes 1 and 2. The two
    cables share the catalog type ``"nayy"`` (ampacity ``i_max_a``); loads sit at 3 and 4.
    """
    r, l_h = 0.206e-3, 0.256e-6
    u_ln = 400.0 / math.sqrt(3.0)
    cable = {
        "type_ref": "nayy",
        "series_resistance_ohm_per_m": _diag(r, 3),
        "series_inductance_h_per_m": _diag(l_h, 3),
        "shunt_capacitance_f_per_m": _diag(0.0, 3),
        "harmonic_line_model": "naive",
    }
    return Grid(
        base_frequency_hz=F0,
        nodes=[Node(id=i, u_rated_v=400.0, phases=ABC) for i in (1, 2, 3, 4)],
        branches=[
            Switch(
                id=20,
                from_node=1,
                to_node=2,
                from_phases=ABC,
                to_phases=ABC,
                default_closed=True,
            ),
            Line(
                id=21,
                from_node=2,
                to_node=3,
                from_phases=ABC,
                to_phases=ABC,
                length_m=150.0,
                **cable,
            ),
            Line(
                id=22,
                from_node=3,
                to_node=4,
                from_phases=ABC,
                to_phases=ABC,
                length_m=200.0,
                **cable,
            ),
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ABC,
                u_ref_v=(u_ln, u_ln, u_ln),
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=_diag(1.0e-3, 3),
                inductance_h=_diag(1.0e-5, 3),
            ),
            Load(id=2, node=3, phases=ABC, p_nom_w=20.0e3, q_nom_var=5.0e3),
            Load(id=3, node=4, phases=ABC, p_nom_w=30.0e3, q_nom_var=8.0e3),
        ],
        types=TypeLibrary(
            lines={
                "nayy": LineType(
                    id="nayy",
                    n_phases=3,
                    series_resistance_ohm_per_m=_diag(r, 3),
                    series_inductance_h_per_m=_diag(l_h, 3),
                    shunt_capacitance_f_per_m=_diag(0.0, 3),
                    max_i_a=i_max_a,
                )
            }
        ),
    )


def mv_lv_transformer(*, s_rated_va: float = 100.0e3) -> Grid:
    """20 kV source -- Dyn5 transformer (``s_rated_va``) -- 400 V bus 2 -- short cable -- 3."""
    u_lv = 400.0
    z_base = u_lv**2 / s_rated_va
    r_t = 0.01 * z_base
    x_t = 0.04 * z_base
    return Grid(
        base_frequency_hz=F0,
        nodes=[
            Node(id=1, u_rated_v=20_000.0, phases=ABC),
            Node(id=2, u_rated_v=u_lv, phases=ABC),
            Node(id=3, u_rated_v=u_lv, phases=ABC),
        ],
        branches=[
            Transformer(
                id=30,
                from_node=1,
                to_node=2,
                from_phases=ABC,
                to_phases=ABC,
                s_rated_va=s_rated_va,
                u_rated_from_v=20_000.0,
                u_rated_to_v=u_lv,
                from_connection=WindingConnection.DELTA,
                to_connection=WindingConnection.WYE_GROUNDED,
                series_resistance_ohm=r_t,
                series_inductance_h=x_t / (2.0 * math.pi * F0),
                tap=ComplexTap(ratio_magnitude=1.0, shift_deg=150.0),
            ),
            Line(
                id=31,
                from_node=2,
                to_node=3,
                from_phases=ABC,
                to_phases=ABC,
                length_m=20.0,
                series_resistance_ohm_per_m=_diag(0.2e-3, 3),
                series_inductance_h_per_m=_diag(0.25e-6, 3),
                shunt_capacitance_f_per_m=_diag(0.0, 3),
                harmonic_line_model="naive",
            ),
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ABC,
                u_ref_v=tuple(20_000.0 / math.sqrt(3.0) for _ in range(3)),
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=_diag(0.01, 3),
                inductance_h=_diag(1.0e-4, 3),
            )
        ],
        types=TypeLibrary(
            transformers={
                "t100": TransformerType(
                    id="t100",
                    s_rated_va=s_rated_va,
                    u_rated_from_v=20_000.0,
                    u_rated_to_v=u_lv,
                    from_connection=WindingConnection.DELTA,
                    to_connection=WindingConnection.WYE_GROUNDED,
                    series_resistance_ohm=r_t,
                    series_inductance_h=x_t / (2.0 * math.pi * F0),
                    nominal_tap=ComplexTap(ratio_magnitude=1.0, shift_deg=150.0),
                )
            }
        ),
    )


def four_wire_two_bus(
    *,
    r_ohm_per_m: float = 0.2e-3,
    length_m: float = 200.0,
    i_max_a: float | None = None,
) -> Grid:
    """400 V four-wire line between an ideal source bus 1 and bus 2.

    Phase and neutral conductors are equal and uncoupled; the neutral of bus 1 is
    grounded through 0.01 ohm. A single-phase device at bus 2 then sees the loop
    impedance ``2 Z + 0.01 ohm`` between its phase-to-neutral terminals and the source.
    With ``i_max_a`` the line carries the catalog type ``"cable4"`` of that ampacity.
    """
    abcn = (Phase.A, Phase.B, Phase.C, Phase.N)
    u_ln = 230.0
    rated = {}
    if i_max_a is not None:
        rated = {
            "types": TypeLibrary(
                lines={
                    "cable4": LineType(
                        id="cable4",
                        n_phases=4,
                        series_resistance_ohm_per_m=_diag(r_ohm_per_m, 4),
                        series_inductance_h_per_m=_diag(0.25e-6, 4),
                        shunt_capacitance_f_per_m=_diag(0.0, 4),
                        max_i_a=i_max_a,
                    )
                }
            )
        }
    return Grid(
        base_frequency_hz=F0,
        nodes=[Node(id=i, u_rated_v=400.0, phases=abcn) for i in (1, 2)],
        branches=[
            Line(
                id=10,
                from_node=1,
                to_node=2,
                from_phases=abcn,
                to_phases=abcn,
                length_m=length_m,
                type_ref="cable4" if i_max_a is not None else None,
                series_resistance_ohm_per_m=_diag(r_ohm_per_m, 4),
                series_inductance_h_per_m=_diag(0.25e-6, 4),
                shunt_capacitance_f_per_m=_diag(0.0, 4),
                harmonic_line_model="naive",
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ABC,
                u_ref_v=(u_ln, u_ln, u_ln),
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=_diag(1.0e-3, 3),
                inductance_h=_diag(1.0e-6, 3),
            ),
            ShuntAppliance(
                id=5,
                node=1,
                phases=(Phase.N,),
                conductance_s=(100.0,),
                capacitance_f=(0.0,),
            ),
        ],
        **rated,
    )
