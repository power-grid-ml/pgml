"""Centre-tapped transformers and earthed star points against live OpenDSS.

Two circuit families, each solved by pgml and by OpenDSS (``opendssdirect``) at the
fundamental (60 Hz) and at harmonic order 3:

1. A 50 kVA single-phase unit with a centre-tapped 440/220 V secondary (OpenDSS
   ``Transformer phases=1 windings=3``, half-secondary ``kVs``, the standard
   ``%Rs=[0.6 1.2 1.2] XHL=XHT=2.04 XLT=1.36`` impedances) feeding unbalanced
   220 V loads on each half and a 440 V resistive element across X1-X2 over a
   three-wire line. Variants: a 7.967 kV phase-to-earth primary earthed through a
   ``10 + j2`` Ohm electrode with the centre tap on the neutral row earthed through
   0.3 / 5 / 25 Ohm; a 13.8 kV phase-to-phase primary with the centre tap bonded to
   the neutral, which is earthed only at the far end; and a solidly earthed primary
   and centre tap without neutral rows.
2. A 400 kVA Dyn1 20/0.4 kV unit whose LV star point is the neutral row, earthed
   through 0.3 / 5 / 25 Ohm, feeding unbalanced wye loads over a four-wire line.

OpenDSS models every earthing as its own ``Reactor`` to ground and the neutral as
bus node 4; pgml stamps the earthing inside the transformer (``GroundingImpedance``)
and the far-end electrode as a ``ShuntReactor``. The current a pgml transformer
reports at its star row therefore includes the earth current of its earthing
impedance; the OpenDSS side adds its earthing reactor's current to compare.

Loads are constant-power at the fundamental (``vminpu=0.1``) and pure harmonic
current sources at order 3 (``NeglectLoadY=Yes``, pgml ``load_shunt="none"``); lines
are lumped R/X with ``X ~ h`` and OpenDSS's earth-return correction off (``Rg=Xg=0``),
so the network, and in it the transformer and its earthing, is the model under test.
Measured agreement over all eleven cases (complex128, CPU, opendssdirect 0.9.4,
OpenDSS ``Tolerance=1e-12``), as the largest complex deviation over every row in pu
of the row's nominal voltage, and over every transformer terminal and earth current:

- fundamental voltages: 3.4e-10 pu in every case, independent of the variant (the
  fundamental iteration of the two tools, not the stamp);
- order-3 voltages: 6e-12 to 1.1e-11 pu (harmonic voltages of 0.1-2.5 V);
- currents: 1.4e-08 to 2.2e-08 A (terminal currents up to 48 A, earth currents
  0.02-0.6 A).

The explicit half-winding impedances and Kersting's default split give the same
result, because the standard OpenDSS centre-tap data is Kersting's split.
Bounds asserted: 1e-8 pu (h1), 1e-9 pu (h3), 1e-6 A.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

opendssdirect = pytest.importorskip("opendssdirect")
dss = opendssdirect

from pgml.assembly import branch_currents, node_phase_index  # noqa: E402
from pgml.evaluation.oracles.opendss_scenario_oracle import (  # noqa: E402
    _scratch_datapath,
)
from pgml.schemas.grid_schema import (  # noqa: E402
    CentreTap,
    ComplexTap,
    Grid,
    GroundingImpedance,
    HarmonicComponent,
    Line,
    Load,
    Node,
    Phase,
    ShuntReactor,
    Source,
    SpectrumPoint,
    StaticSpectrum,
    Transformer,
)
from pgml.solver import solve_harmonic_flow  # noqa: E402

pytestmark = [pytest.mark.opendss, pytest.mark.usefixtures("opendss_model_defaults")]

P = Phase
F0 = 60.0
W0 = 2.0 * math.pi * F0
_SUFFIX = {P.A: 1, P.B: 2, P.C: 3, P.N: 4}
_ATOL_V1_PU = 1e-8
_ATOL_V3_PU = 1e-9
_ATOL_I_A = 1e-6

# -- centre-tapped unit ------------------------------------------------------------
KVA_CT = 50.0
KV_X = 0.22  # one half winding
RH, RX, XHL, XHT, XLT = 0.6, 1.2, 2.04, 2.04, 1.36  # percent on 50 kVA
R_LL = 40.0  # 440 V resistive element X1-X2 [Ohm]
R_CUSTOMER = 25.0  # far-end neutral electrode [Ohm]
R3 = [[0.30, 0.05, 0.05], [0.05, 0.30, 0.05], [0.05, 0.05, 0.30]]  # Ohm/km
X3 = [[0.40, 0.15, 0.15], [0.15, 0.40, 0.15], [0.15, 0.15, 0.40]]
R4 = [[0.3 if i == j else 0.05 for j in range(4)] for i in range(4)]
X4 = [[0.4 if i == j else 0.15 for j in range(4)] for i in range(4)]
LEN_KM = 0.1

SPECTRUM = StaticSpectrum(
    spectrum=SpectrumPoint(
        components=[
            HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
            HarmonicComponent(order=3, magnitude_pu=0.2, phase_deg=30.0),
        ]
    )
)


def _mat(m) -> str:
    return "[" + " | ".join(" ".join(f"{v:g}" for v in r) for r in m) + "]"


def _bus(name: str, phases) -> str:
    return name + "".join(f".{_SUFFIX[p]}" for p in phases)


def _source(u_ll_v: float) -> Source:
    u = u_ll_v / math.sqrt(3.0)
    r = [[0.5 if i == j else 0.0 for j in range(3)] for i in range(3)]
    ell = [[1.5 / W0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    return Source(
        id=1,
        node=1,
        phases=(P.A, P.B, P.C),
        u_ref_v=[u] * 3,
        u_angle_deg=[0.0, -120.0, 120.0],
        resistance_ohm=r,
        inductance_h=ell,
    )


def _line(phases, r, x) -> Line:
    return Line(
        id=11,
        from_node=2,
        to_node=3,
        from_phases=phases,
        to_phases=phases,
        length_m=LEN_KM * 1e3,
        series_resistance_ohm_per_m=[[v / 1e3 for v in row] for row in r],
        series_inductance_h_per_m=[[v / 1e3 / W0 for v in row] for row in x],
        shunt_capacitance_f_per_m=[[0.0] * len(phases) for _ in phases],
        harmonic_line_model="naive",
    )


def _earth(rid: int, node: int, ohm: float) -> ShuntReactor:
    return ShuntReactor(
        id=rid,
        from_node=node,
        to_node=node,
        from_phases=(P.N,),
        to_phases=(P.N,),
        conductance_s=[[1.0 / ohm]],
        capacitance_f=[[0.0]],
    )


def _ct_impedances():
    """``Z_a`` (full secondary coil) and one half winding's ``Z`` from the DSS data."""
    zb_x = KV_X**2 * 1e3 / KVA_CT
    zb_full = (2 * KV_X) ** 2 * 1e3 / KVA_CT
    x_h = (XHL + XHT - XLT) / 2.0
    x_x = (XHL + XLT - XHT) / 2.0
    z_half = complex(RX, x_x) / 100.0 * zb_x
    z_a = complex(RH, x_h) / 100.0 * zb_full + 2.0 * z_half
    return z_a, z_half


def _centre_tap_case(variant: str, rg_ohm: float, explicit_half: bool):
    """DSS script, pgml grid, row map and earthing elements of one centre-tap case."""
    neutral = variant != "solid"
    lv = (P.A, P.B, P.N) if neutral else (P.A, P.B)
    if variant == "swer":
        prim_kv, prim_phases, prim_conn = 7.967, (P.A,), "wye_grounded"
        prim_bus = "src.1.5"
        prim_ground = GroundingImpedance(r_ohm=10.0, x_ohm=2.0)
    elif variant == "phase_to_phase":
        prim_kv, prim_phases, prim_conn = 13.8, (P.A, P.B), "delta"
        prim_bus, prim_ground = "src.1.2", None
    else:
        prim_kv, prim_phases, prim_conn = 7.967, (P.A,), "wye_grounded"
        prim_bus, prim_ground = "src.1.0", None
    ct_node = "4" if neutral else "0"
    lines = [
        "Clear",
        f"Set DefaultBaseFrequency={F0:g}",
        "New Circuit.ct basekv=13.8 pu=1.0 phases=3 bus1=src angle=0 "
        "R1=0.5 X1=1.5 R0=0.5 X0=1.5",
        "New Spectrum.clean NumHarm=1 harmonic=[1] %mag=[100] angle=[0]",
        "Edit Vsource.source spectrum=clean",
        "New Spectrum.s3 NumHarm=2 harmonic=[1 3] %mag=[100 20] angle=[0 30]",
        f"New Transformer.ct phases=1 windings=3 buses=[{prim_bus} lv.1.{ct_node} "
        f"lv.{ct_node}.2] conns=[{'delta' if prim_conn == 'delta' else 'wye'} wye wye] "
        f"kvs=[{prim_kv} {KV_X} {KV_X}] kvas=[{KVA_CT} {KVA_CT} {KVA_CT}] "
        f"%Rs=[{RH} {RX} {RX}] XHL={XHL} XHT={XHT} XLT={XLT} %noloadloss=0 %imag=0 "
        "ppm_antifloat=0",
    ]
    earth = []  # (dss reactor, pgml node id, resistance, in transformer)
    if variant == "swer":
        lines.append("New Reactor.ep phases=1 bus1=src.5 bus2=src.0 R=10 X=2")
    if variant == "swer":
        lines.append(f"New Reactor.es phases=1 bus1=lv.4 bus2=lv.0 R={rg_ohm:g} X=0")
        earth.append(("Reactor.es", 2, rg_ohm, True))
    if neutral:
        lines.append(
            f"New Reactor.ec phases=1 bus1=lv2.4 bus2=lv2.0 R={R_CUSTOMER:g} X=0"
        )
        earth.append(("Reactor.ec", 3, R_CUSTOMER, False))
    r_line = R3 if neutral else [row[:2] for row in R3[:2]]
    x_line = X3 if neutral else [row[:2] for row in X3[:2]]
    nc = len(lv)
    lines.append(
        f"New Line.ln phases={nc} bus1={_bus('lv', lv)} bus2={_bus('lv2', lv)} "
        f"length={LEN_KM} units=km rmatrix={_mat(r_line)} xmatrix={_mat(x_line)} "
        f"cmatrix={_mat([[0.0] * (i + 1) for i in range(nc)])} Rg=0 Xg=0"
    )
    lines.append(f"New Reactor.rll phases=1 bus1=lv2.1 bus2=lv2.2 R={R_LL} X=0")
    ret = ".4" if neutral else ".0"
    for name, ph, kw, kvar in (("l1", 1, 8.0, 2.0), ("l2", 2, 3.0, 1.0)):
        lines.append(
            f"New Load.{name} phases=1 bus1=lv2.{ph}{ret} kv={KV_X} kw={kw} "
            f"kvar={kvar} model=1 vminpu=0.1 vmaxpu=10 spectrum=s3"
        )

    z_a, z_half = _ct_impedances()
    centre_tap = (
        CentreTap(half_resistance_ohm=z_half.real, half_inductance_h=z_half.imag / W0)
        if explicit_half
        else CentreTap()
    )
    to_conn = "wye" if variant == "phase_to_phase" else "wye_grounded"
    to_ground = GroundingImpedance(r_ohm=rg_ohm) if variant == "swer" else None
    nodes = [
        Node(id=1, u_rated_v=13800.0, phases=(P.A, P.B, P.C)),
        Node(id=2, u_rated_v=440.0, phases=lv),
        Node(id=3, u_rated_v=440.0, phases=lv),
    ]
    branches = [
        Transformer(
            id=10,
            from_node=1,
            to_node=2,
            from_phases=prim_phases,
            to_phases=lv,
            s_rated_va=KVA_CT * 1e3,
            u_rated_from_v=prim_kv * 1e3,
            u_rated_to_v=2 * KV_X * 1e3,
            from_connection=prim_conn,
            to_connection=to_conn,
            series_resistance_ohm=z_a.real,
            series_inductance_h=z_a.imag / W0,
            from_grounding=prim_ground,
            to_grounding=to_ground,
            centre_tap=centre_tap,
        ),
        _line(lv, r_line, x_line),
        ShuntReactor(
            id=13,
            from_node=3,
            to_node=3,
            from_phases=(P.A, P.B),
            to_phases=(P.A, P.B),
            conductance_s=[[1 / R_LL, -1 / R_LL], [-1 / R_LL, 1 / R_LL]],
            capacitance_f=[[0.0, 0.0], [0.0, 0.0]],
        ),
    ]
    if neutral:
        branches.append(_earth(12, 3, R_CUSTOMER))
    appliances = [_source(13800.0)]
    for k, (ph, kw, kvar) in enumerate(((P.A, 8.0, 2.0), (P.B, 3.0, 1.0))):
        appliances.append(
            Load(
                id=20 + k,
                node=3,
                phases=(ph,),
                connection="wye",
                p_nom_w=kw * 1e3,
                q_nom_var=kvar * 1e3,
                spectrum=SPECTRUM,
            )
        )
    grid = Grid(
        base_frequency_hz=F0, nodes=nodes, branches=branches, appliances=appliances
    )
    rows = {(1, p): ("src", _SUFFIX[p], 13800 / math.sqrt(3)) for p in (P.A, P.B, P.C)}
    for nid, bus in ((2, "lv"), (3, "lv2")):
        for p in lv:
            rows[(nid, p)] = (bus, _SUFFIX[p], KV_X * 1e3)
    # OpenDSS conductors: primary 0-1, X1 winding 2-3, X2 winding 4-5.
    from_terms = [0, 1] if variant == "phase_to_phase" else [0]
    dss_terminals = {"from": from_terms, "to": [2, 5] + ([3, 4] if neutral else [])}
    return lines, grid, rows, earth, dss_terminals


def _dyn_case(rg_ohm: float):
    """A Dyn1 unit with its LV star on the neutral row, earthed through ``rg_ohm``."""
    abcn = (P.A, P.B, P.C, P.N)
    loads = ((P.A, 20.0, 5.0), (P.B, 5.0, 1.0), (P.C, 10.0, 4.0))
    lines = [
        "Clear",
        f"Set DefaultBaseFrequency={F0:g}",
        "New Circuit.dy basekv=20 pu=1.0 phases=3 bus1=src angle=0 "
        "R1=0.5 X1=1.5 R0=0.5 X0=1.5",
        "New Spectrum.clean NumHarm=1 harmonic=[1] %mag=[100] angle=[0]",
        "Edit Vsource.source spectrum=clean",
        "New Spectrum.s3 NumHarm=2 harmonic=[1 3] %mag=[100 20] angle=[0 30]",
        "New Transformer.t1 phases=3 windings=2 buses=[src.1.2.3 lv.1.2.3.4] "
        "conns=[delta wye] kvs=[20 0.4] kvas=[400 400] XHL=4 %Rs=[0.5 0.5] "
        "%noloadloss=0 %imag=0 ppm_antifloat=0",
        f"New Reactor.es phases=1 bus1=lv.4 bus2=lv.0 R={rg_ohm:g} X=0",
        f"New Reactor.ec phases=1 bus1=lv2.4 bus2=lv2.0 R={R_CUSTOMER:g} X=0",
        f"New Line.ln phases=4 bus1=lv.1.2.3.4 bus2=lv2.1.2.3.4 length={LEN_KM} "
        f"units=km rmatrix={_mat(R4)} xmatrix={_mat(X4)} "
        f"cmatrix={_mat([[0.0] * (i + 1) for i in range(4)])} Rg=0 Xg=0",
    ]
    for p, kw, kvar in loads:
        lines.append(
            f"New Load.l{_SUFFIX[p]} phases=1 bus1=lv2.{_SUFFIX[p]}.4 kv=0.23094 "
            f"kw={kw} kvar={kvar} model=1 vminpu=0.1 vmaxpu=10 spectrum=s3"
        )
    z_base = 0.4**2 * 1e6 / 400e3
    nodes = [
        Node(id=1, u_rated_v=20e3, phases=(P.A, P.B, P.C)),
        Node(id=2, u_rated_v=400.0, phases=abcn),
        Node(id=3, u_rated_v=400.0, phases=abcn),
    ]
    branches = [
        Transformer(
            id=10,
            from_node=1,
            to_node=2,
            from_phases=(P.A, P.B, P.C),
            to_phases=abcn,
            s_rated_va=400e3,
            u_rated_from_v=20e3,
            u_rated_to_v=400.0,
            from_connection="delta",
            to_connection="wye_grounded",
            series_resistance_ohm=0.01 * z_base,
            series_inductance_h=0.04 * z_base / W0,
            tap=ComplexTap(ratio_magnitude=1.0, shift_deg=30.0),
            to_grounding=GroundingImpedance(r_ohm=rg_ohm),
        ),
        _line(abcn, R4, X4),
        _earth(12, 3, R_CUSTOMER),
    ]
    appliances = [_source(20e3)]
    for k, (p, kw, kvar) in enumerate(loads):
        appliances.append(
            Load(
                id=20 + k,
                node=3,
                phases=(p,),
                connection="wye",
                p_nom_w=kw * 1e3,
                q_nom_var=kvar * 1e3,
                spectrum=SPECTRUM,
            )
        )
    grid = Grid(
        base_frequency_hz=F0, nodes=nodes, branches=branches, appliances=appliances
    )
    rows = {(1, p): ("src", _SUFFIX[p], 20e3 / math.sqrt(3)) for p in (P.A, P.B, P.C)}
    for nid, bus in ((2, "lv"), (3, "lv2")):
        for p in abcn:
            rows[(nid, p)] = (bus, _SUFFIX[p], 400 / math.sqrt(3))
    earth = [("Reactor.es", 2, rg_ohm, True), ("Reactor.ec", 3, R_CUSTOMER, False)]
    # OpenDSS terminal conductors: HV 0-2, HV star 3, LV 4-6, LV star 7.
    dss_terminals = {"from": [0, 1, 2], "to": [4, 5, 6, 7]}
    return lines, grid, rows, earth, dss_terminals


def _dss_solve(lines) -> None:
    for line in lines + [
        "Set NeglectLoadY=Yes",
        "Set Tolerance=1e-12",
        "Set MaxIterations=200",
        "Set Mode=Snap",
        "Solve",
    ]:
        dss.Text.Command(line)
    assert dss.Solution.Converged()


def _dss_state(element: str, earth) -> tuple[dict, np.ndarray, dict]:
    volts = {}
    for bus in ("src", "lv", "lv2"):
        dss.Circuit.SetActiveBus(bus)
        v = np.array(dss.Bus.Voltages())
        for k, node in enumerate(dss.Bus.Nodes()):
            volts[(bus, node)] = complex(v[2 * k], v[2 * k + 1])
    dss.Circuit.SetActiveElement(element)
    c = np.array(dss.CktElement.Currents())
    currents = c[0::2] + 1j * c[1::2]
    earth_i = {}
    for name, *_ in earth:
        dss.Circuit.SetActiveElement(name)
        c = np.array(dss.CktElement.Currents())
        earth_i[name] = complex(c[0], c[1])
    return volts, currents, earth_i


def _compare(case, element: str):
    lines, grid, rows, earth, terms = case
    _dss_solve(lines)
    ref = {1: _dss_state(element, earth)}
    with _scratch_datapath():
        for cmd in ("Set Mode=Harmonics", "Set Harmonic=3", "Solve"):
            dss.Text.Command(cmd)
        ref[3] = _dss_state(element, earth)
        dss.Text.Command("Set Mode=Snap")

    res = solve_harmonic_flow(
        grid, [1, 3], slack="norton", load_shunt="none", tol=1e-12
    )
    assert res.pf.converged
    v = res.v.numpy()
    index = node_phase_index(grid)
    currents = {
        b.branch_id: b for b in branch_currents(grid, res.v, [F0, 3 * F0], index)
    }[10]
    worst = {"v1": 0.0, "v3": 0.0, "i": 0.0}
    for k, h in enumerate((1, 3)):
        volts, i_dss, earth_dss = ref[h]
        dv = max(
            abs(v[k, index.row(n, p)] - volts[(bus, node)]) / base
            for (n, p), (bus, node, base) in rows.items()
        )
        worst["v1" if h == 1 else "v3"] = dv
        # pgml stamps the earthing inside the transformer: its star-row current
        # carries the earth current of that impedance.
        in_xfmr = sum(earth_dss[name] for name, _, _, inside in earth if inside)
        i_from_ref = np.array([i_dss[j] for j in terms["from"]])
        i_to_ref = np.array([i_dss[j] for j in terms["to"]])
        lv = [p for p in grid.branches[0].to_phases]
        if P.N in lv:
            if len(terms["to"]) > len(lv):  # centre tap: two DSS conductors on N
                i_to_ref = np.array(
                    [i_to_ref[0], i_to_ref[1], i_to_ref[2] + i_to_ref[3]]
                )
            i_to_ref[-1] += in_xfmr
        di = max(
            np.abs(currents.i_from.numpy()[k][: len(i_from_ref)] - i_from_ref).max(),
            np.abs(currents.i_to.numpy()[k] - i_to_ref).max(),
        )
        for name, node, ohm, _ in earth:
            i_earth = v[k, index.row(node, P.N)] / ohm
            di = max(di, abs(i_earth - earth_dss[name]))
        worst["i"] = max(worst["i"], float(di))
    return worst


@pytest.mark.parametrize("rg_ohm", [0.3, 5.0, 25.0])
@pytest.mark.parametrize("explicit_half", [True, False])
def test_swer_centre_tap_with_earthed_neutral_matches_opendss(rg_ohm, explicit_half):
    worst = _compare(_centre_tap_case("swer", rg_ohm, explicit_half), "Transformer.ct")
    assert worst["v1"] < _ATOL_V1_PU, worst
    assert worst["v3"] < _ATOL_V3_PU, worst
    assert worst["i"] < _ATOL_I_A, worst


@pytest.mark.parametrize("variant", ["phase_to_phase", "solid"])
def test_centre_tap_primary_and_neutral_variants_match_opendss(variant):
    worst = _compare(_centre_tap_case(variant, 0.0, True), "Transformer.ct")
    assert worst["v1"] < _ATOL_V1_PU, worst
    assert worst["v3"] < _ATOL_V3_PU, worst
    assert worst["i"] < _ATOL_I_A, worst


@pytest.mark.parametrize("rg_ohm", [0.3, 5.0, 25.0])
def test_dyn_star_point_on_neutral_row_earthed_matches_opendss(rg_ohm):
    worst = _compare(_dyn_case(rg_ohm), "Transformer.t1")
    assert worst["v1"] < _ATOL_V1_PU, worst
    assert worst["v3"] < _ATOL_V3_PU, worst
    assert worst["i"] < _ATOL_I_A, worst
