"""The harmonic device shunt against a live OpenDSS engine.

Two levels of comparison, both with the SAME device model on each side (pgml's
``load_shunt`` / ``HarmonicShuntModel`` against OpenDSS's ``%SeriesRL`` / ``puXharm`` /
``XRharm``, which the scenario exporter now writes onto every exported ``Load``):

1. the element admittance itself, read from the DSS ``Load``'s own ``YPrim`` in harmonics
   mode — the tightest available check, needing no power flow. It pins
   ``Load.pas``'s ``CalcYPrimMatrix`` for a 1-phase WYE, a 3-phase WYE and a 3-phase
   DELTA load, at ``%SeriesRL`` 0 / 50 / 100 and with the motor branch. Measured
   agreement 0 to 4.7e-16 relative, i.e. floating-point identical.
2. end-to-end harmonic bus voltages on two feeders, off and on a parallel resonance
   placed near order 7 by a capacitor bank. Measured agreement on this machine
   (complex128, CPU, opendssdirect 0.9.4), as the largest ``|d|V(h)||`` over every
   energised row and every injected order, in pu of nominal: 1.6e-12 on IEEE-33 with
   lumped R/X lines, 6.8e-12 with the resonant bank, 1.3e-09 on the same feeder with its
   synthesized Carson geometry, 4.5e-09 on the three-phase CIGRE LV benchmark (whose own
   fundamental floor is 3.3e-08 pu, set by the independent exporter). The bound asserted
   here is 1e-06 pu for every variant, which leaves at least three orders of magnitude of
   headroom.

Both feeders are compared with R-const / ``X ~ h`` lines and ``Rg=Xg=0``, so the only
model under test is the device shunt: pgml's default ``sequence_aware`` line model (skin
effect plus a Carson earth-return term) is a different line model from anything OpenDSS
can express with lumped R/X input.
"""

from __future__ import annotations

import math
import tempfile

import numpy as np
import pytest
import torch

from pgml.assembly import base_voltage_per_row, node_phase_index
from pgml.assembly._load_shunt import harmonic_shunt_element_admittance
from pgml.convert.pandapower import PhaseMode
from pgml.evaluation.oracles.opendss_scenario_oracle import (
    _extract_voltages,
    export_grid_to_opendss,
)
from pgml.geometry.synthesis import strip_grid_geometry
from pgml.grids import (
    _attach_spectrum_farthest,
    cigre_lv_full_grid,
    ieee33_geometry_grid,
)
from pgml.schemas.grid_schema import (
    HarmonicShuntModel,
    InjectionAppliance,
    Load,
    ShuntAppliance,
)
from pgml.solver import solve_harmonic_flow

pytest.importorskip("opendssdirect")
pytestmark = pytest.mark.opendss

CDT = torch.complex128
RDT = torch.float64

#: Six-pulse rectifier spectrum extended to order 25 (``1/h`` continuation of the
#: 6k+-1 family), so the comparison covers the whole order range of a harmonic study.
RECTIFIER_SPECTRUM = [
    (1, 1.00, 0.0),
    (3, 0.30, 0.0),
    (5, 0.20, 0.0),
    (7, 0.14, 0.0),
    (9, 0.08, 0.0),
    (11, 0.09, 0.0),
    (13, 0.07, 0.0),
    (17, 0.050, 0.0),
    (19, 0.040, 0.0),
    (23, 0.030, 0.0),
    (25, 0.025, 0.0),
]
ORDERS = [o for o, *_ in RECTIFIER_SPECTRUM]

#: variant -> (pgml ``load_shunt``, per-device override, DSS Load properties)
VARIANTS = {
    "none": ("none", None, ""),
    "series_rl_0": ("opendss", dict(series_rl_fraction=0.0), " %SeriesRL=0"),
    "series_rl_50": ("opendss", dict(series_rl_fraction=0.5), " %SeriesRL=50"),
    "series_rl_100": ("opendss", dict(series_rl_fraction=1.0), " %SeriesRL=100"),
    "motor": (
        "motor",
        dict(series_rl_fraction=0.5, motor_x_harm_pu=0.2, motor_xr_harm=6.0),
        " %SeriesRL=50 puXharm=0.2 XRharm=6",
    ),
}

#: Bank sizes that put a parallel resonance nearest order 7 at the feeder-end load node,
#: from a 45-point logarithmic search on the driving-point impedance of each grid.
RESONANT_BANK = {"ieee33_rx": (18, 370.26), "cigre_lv": (20, 69.26)}


# --------------------------------------------------------------------------- #
# 1. the element admittance against the DSS Load's own YPrim
# --------------------------------------------------------------------------- #
def _yprim_circuit(
    dss, nphases: int, conn: str, kv: float, kw: float, kvar: float, props: str
):
    """One load behind a line, so the Load element's YPrim can be read per order."""
    dss.Text.Command("Clear")
    dss.Text.Command("Set DefaultBaseFrequency=50")
    dss.Text.Command(f"Set DataPath={tempfile.mkdtemp(prefix='pgml_yprim_')}")
    dss.Text.Command(
        f"New Circuit.probe basekv={kv} phases={nphases} bus1=b1 pu=1.0 "
        "frequency=50 r1=0.05 x1=0.05 r0=0.05 x0=0.05"
    )
    busph = ".".join(str(k + 1) for k in range(nphases))
    dss.Text.Command(
        f"New Line.l1 phases={nphases} bus1=b1.{busph} bus2=b2.{busph} r1=0.3 x1=0.3 "
        "c1=0 r0=0.3 x0=0.3 c0=0 length=1 units=m"
    )
    dss.Text.Command("New Spectrum.clean NumHarm=1 harmonic=[1] %mag=[100] angle=[0]")
    dss.Text.Command("Edit Vsource.source spectrum=clean")
    dss.Text.Command(
        f"New Load.ld1 phases={nphases} bus1=b2.{busph} conn={conn} kv={kv} kW={kw} "
        f"kvar={kvar} model=1 vminpu=0.0001 vmaxpu=10000 spectrum=clean{props}"
    )
    dss.Text.Command(f"Set VoltageBases=[{kv}]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Set NeglectLoadY=No")
    dss.Text.Command("Set Tolerance=0.0000000001")
    dss.Text.Command("Set MaxIterations=100")


def _yprim_diagonal(dss) -> complex:
    dss.Circuit.SetActiveElement("Load.ld1")
    y = np.asarray(dss.CktElement.YPrim())
    n = int(round(math.sqrt(len(y) / 2)))
    return complex((y[0::2] + 1j * y[1::2]).reshape(n, n)[0, 0])


@pytest.mark.parametrize(
    "label,nphases,conn,kv,kw,kvar",
    [
        ("1ph wye 230 V", 1, "wye", 0.23, 2.0, 0.5),
        ("3ph wye 400 V", 3, "wye", 0.4, 30.0, 10.0),
        ("3ph delta 400 V", 3, "delta", 0.4, 30.0, 10.0),
    ],
)
@pytest.mark.parametrize(
    "variant", ["series_rl_0", "series_rl_50", "series_rl_100", "motor"]
)
def test_element_admittance_matches_opendss_yprim(
    label, nphases, conn, kv, kw, kvar, variant
):
    """pgml's element admittance IS OpenDSS's ``Load`` YPrim, order by order.

    A DELTA load's ``YPrim`` diagonal carries the leg admittance TWICE (the leg appears
    on both of the phases it connects), which is what the nodal incidence
    ``M^T diag(y) M`` reproduces; the comparison divides it out to test the element
    value itself.
    """
    import opendssdirect as dss

    _, override, props = VARIANTS[variant]
    _yprim_circuit(dss, nphases, conn, kv, kw, kvar, props)
    dss.Text.Command("Set Mode=Snap")
    dss.Text.Command("Solve")
    assert dss.Solution.Converged()

    is_delta = conn == "delta"
    v_rated = (
        kv * 1000.0 if (is_delta or nphases == 1) else kv * 1000.0 / math.sqrt(3.0)
    )
    p_elem = kw * 1000.0 / nphases
    q_elem = kvar * 1000.0 / nphases
    # OpenDSS's motor reactance is referred to the ELEMENT's kV and kVA base.
    kva_base = math.hypot(kw, kvar) * 1000.0 if is_delta else math.hypot(p_elem, q_elem)
    orders = [5, 7, 13, 25]
    y_pgml = harmonic_shunt_element_admittance(
        torch.tensor([[complex(p_elem, q_elem)]], dtype=CDT),
        torch.tensor([[v_rated]], dtype=RDT),
        torch.tensor([float(h) for h in orders], dtype=RDT),
        torch.tensor([[override["series_rl_fraction"]]], dtype=RDT),
        motor_x_pu=torch.tensor([[override.get("motor_x_harm_pu", 0.0)]], dtype=RDT),
        motor_xr=torch.tensor([[override.get("motor_xr_harm", 1.0)]], dtype=RDT),
        motor_s_base=torch.tensor([[kva_base]], dtype=RDT),
        cdtype=CDT,
    ).reshape(-1)

    dss.Text.Command("Set Mode=Harmonics")
    for k, h in enumerate(orders):
        dss.Text.Command(f"Set Harmonic={h}")
        dss.Text.Command("Solve")
        y_dss = _yprim_diagonal(dss) / (2.0 if is_delta else 1.0)
        y_ref = complex(y_pgml[k])
        err = abs(y_ref - y_dss) / abs(y_dss)
        # Measured 0 ... 4.7e-16 relative: the same expression to floating point.
        assert err < 1e-12, f"{label} {variant} h={h}: rel error {err:.2e}"


def test_neglect_load_y_is_the_pure_current_source():
    """``Set NeglectLoadY=Yes`` leaves ``EPSILON = 1e-12 S``, i.e. no shunt at all.

    This is the model ``load_shunt="none"`` implements, so the two are the same physics
    and not an approximation of each other.
    """
    import opendssdirect as dss

    _yprim_circuit(dss, 3, "wye", 0.4, 30.0, 10.0, "")
    dss.Text.Command("Set NeglectLoadY=Yes")
    dss.Text.Command("Set Mode=Snap")
    dss.Text.Command("Solve")
    dss.Text.Command("Set Mode=Harmonics")
    dss.Text.Command("Set Harmonic=5")
    dss.Text.Command("Solve")
    assert abs(_yprim_diagonal(dss)) <= 1e-11


# --------------------------------------------------------------------------- #
# 2. end-to-end harmonic voltages on two feeders
# --------------------------------------------------------------------------- #
def _add_capacitor(grid, node_id: int, kvar_total: float) -> None:
    """A WYE bank of ``kvar_total`` at ``node_id``, referred to the node's rated voltage."""
    node = next(nd for nd in grid.nodes if int(nd.id) == int(node_id))
    p = len(node.phases)
    u = float(node.u_rated_v)
    b_elem = kvar_total * 1e3 / (u * u)
    c_elem = b_elem / (2.0 * math.pi * float(grid.base_frequency_hz))
    grid.appliances.append(
        ShuntAppliance(
            id=max(a.id for a in grid.appliances) + 1,
            name=f"cap_{node_id}",
            node=int(node_id),
            phases=list(node.phases),
            conductance_s=[0.0] * p,
            capacitance_f=[c_elem] * p,
        )
    )


def _build_grid(name: str, *, bank: bool = False):
    """IEEE-33 with lumped R/X lines, or the three-phase CIGRE LV benchmark."""
    if name == "ieee33_rx":
        grid, _ = ieee33_geometry_grid(n_harmonic_loads=5, spectrum=RECTIFIER_SPECTRUM)
        strip_grid_geometry(grid)  # R const, X ~ h: OpenDSS's own lumped law
    elif name == "ieee33_geometry":
        grid, _ = ieee33_geometry_grid(n_harmonic_loads=5, spectrum=RECTIFIER_SPECTRUM)
    else:
        grid, _ = cigre_lv_full_grid(
            phase_mode=PhaseMode.THREE_PHASE, harmonic_line_model="naive"
        )
        _attach_spectrum_farthest(grid, 6, RECTIFIER_SPECTRUM)
    if bank:
        _add_capacitor(grid, *RESONANT_BANK[name])
    return grid


def _orders_for(grid, name: str) -> list[int]:
    """Every spectrum order, bounded below 1 kHz for the Carson-geometry grid.

    OpenDSS switches a geometry line's conductor spacing term from the published GMR to
    the physical radius at 1 kHz (``LineConstants.pas``), which pgml's Carson path does
    not do, so a geometry grid is only comparable below that frequency.
    """
    f0 = float(grid.base_frequency_hz)
    if name == "ieee33_geometry":
        return [h for h in ORDERS if h * f0 < 1000.0]
    return list(ORDERS)


def _set_override(grid, override) -> None:
    for a in grid.appliances:
        if isinstance(a, InjectionAppliance):
            a.harmonic_model = (
                None if override is None else HarmonicShuntModel(**override)
            )


def _dss_harmonic_voltages(grid, load_shunt: str, props: str, orders, geometry: bool):
    """Solve ``grid`` in a live OpenDSS engine, one voltage vector per order."""
    import opendssdirect as dss

    if geometry:
        index, busname = _build_geometry_circuit(dss, grid, load_shunt, props)
        rowmap = None
    else:
        circuit = export_grid_to_opendss(grid, mode="matched", load_shunt=load_shunt)
        index, rowmap = circuit.index, circuit.rowmap
        _attach_dss_spectra(dss, grid, circuit)
    dss.Text.Command("Set Mode=Snap")
    dss.Text.Command("Solve")
    assert dss.Solution.Converged(), "OpenDSS snapshot solve failed"
    out = {1: _read_rows(dss, index, rowmap, busname if geometry else None)}
    dss.Text.Command("Set Mode=Harmonics")
    for h in orders:
        if h == 1:
            continue
        dss.Text.Command(f"Set Harmonic={h}")
        dss.Text.Command("Solve")
        assert dss.Solution.Converged(), f"OpenDSS h={h} solve failed"
        out[h] = _read_rows(dss, index, rowmap, busname if geometry else None)
    return out


def _attach_dss_spectra(dss, grid, circuit) -> None:
    hs = ", ".join(str(o) for o, *_ in RECTIFIER_SPECTRUM)
    mags = ", ".join(f"{m * 100.0:.6g}" for _, m, _ in RECTIFIER_SPECTRUM)
    angs = ", ".join(f"{a:.6g}" for *_, a in RECTIFIER_SPECTRUM)
    dss.Text.Command(
        f"New Spectrum.rect NumHarm={len(RECTIFIER_SPECTRUM)} harmonic=[{hs}] "
        f"%mag=[{mags}] angle=[{angs}]"
    )
    for ld in grid.appliances:
        if isinstance(ld, Load) and ld.spectrum is not None:
            for name in circuit.loads[int(ld.id)].elements.values():
                dss.Text.Command(f"Edit Load.{name} spectrum=rect")


def _build_geometry_circuit(dss, grid, load_shunt: str, props: str):
    """The Carson-geometry IEEE-33 circuit (the scenario exporter refuses geometry lines)."""
    from pgml.evaluation.oracles.opendss_oracle import build_opendss_geometry_circuit

    f0 = float(grid.base_frequency_hz)
    dss.Text.Command("Clear")
    dss.Text.Command(f"Set DefaultBaseFrequency={f0:g}")
    dss.Text.Command(f"Set DataPath={tempfile.mkdtemp(prefix='pgml_geom_')}")
    busname = build_opendss_geometry_circuit(grid)
    hs = ", ".join(str(o) for o, *_ in RECTIFIER_SPECTRUM)
    mags = ", ".join(f"{m * 100.0:.6g}" for _, m, _ in RECTIFIER_SPECTRUM)
    angs = ", ".join(f"{a:.6g}" for *_, a in RECTIFIER_SPECTRUM)
    dss.Text.Command(
        f"New Spectrum.rect NumHarm={len(RECTIFIER_SPECTRUM)} harmonic=[{hs}] "
        f"%mag=[{mags}] angle=[{angs}]"
    )
    dss.Text.Command("New Spectrum.clean NumHarm=1 harmonic=[1] %mag=[100] angle=[0]")
    dss.Text.Command("Edit Vsource.source spectrum=clean")
    index = node_phase_index(grid)
    base_v = base_voltage_per_row(grid).numpy()
    for ld in grid.appliances:
        if not isinstance(ld, Load):
            continue
        node = int(ld.node)
        kv = base_v[index.rows(node)[0]] / 1e3
        spec = "rect" if ld.spectrum is not None else "clean"
        dss.Text.Command(
            f"New Load.ld{ld.id} phases=1 bus1={busname[node]}.1 conn=wye "
            f"kv={kv:.10g} kW={float(ld.p_nom_w) / 1e3:.10g} "
            f"kvar={float(ld.q_nom_var or 0.0) / 1e3:.10g} model=1 vminpu=0.0001 "
            f"vmaxpu=10000 spectrum={spec}{props}"
        )
    dss.Text.Command("Set NeglectLoadY=" + ("Yes" if load_shunt == "none" else "No"))
    dss.Text.Command("Set Tolerance=0.0000000001")
    dss.Text.Command("Set MaxIterations=100")
    return index, busname


def _read_rows(dss, index, rowmap, busname):
    if rowmap is not None:
        return _extract_voltages(dss, rowmap, index.size)
    from pgml.schemas.grid_schema import Phase

    ph_of = {1: Phase.A, 2: Phase.B, 3: Phase.C, 4: Phase.N}
    raw = np.asarray(dss.Circuit.AllBusVolts())
    volts = raw[0::2] + 1j * raw[1::2]
    out = np.zeros(index.size, dtype=complex)
    inv = {v: k for k, v in busname.items()}
    for nm, v in zip(dss.Circuit.AllNodeNames(), volts):
        bus, k = nm.lower().rsplit(".", 1)
        out[index.row(inv[bus], ph_of[int(k)])] = v
    return out


def _compare(name: str, variant: str, *, bank: bool = False, geometry: bool = False):
    """Max |Δ|V(h)|| over harmonic orders and energised rows, in pu of nominal."""
    load_shunt, override, props = VARIANTS[variant]
    grid = _build_grid(name, bank=bank)
    _set_override(grid, override)
    orders = _orders_for(grid, name)
    res = solve_harmonic_flow(
        grid, orders, slack="norton", dtype=CDT, load_shunt=load_shunt
    )
    assert res.pf.converged
    v_pgml = res.v.numpy()
    base_v = base_voltage_per_row(grid).numpy()
    v_dss = _dss_harmonic_voltages(grid, load_shunt, props, orders, geometry)
    live = np.abs(v_pgml[orders.index(1)]) > 0.1 * base_v
    worst = 0.0
    for k, h in enumerate(orders):
        if h == 1:
            continue
        d = np.abs(np.abs(v_pgml[k]) - np.abs(v_dss[h])) / base_v
        worst = max(worst, float(d[live].max()))
    return worst, v_pgml, base_v, live, orders


@pytest.mark.parametrize(
    "variant", ["none", "series_rl_0", "series_rl_50", "series_rl_100", "motor"]
)
def test_ieee33_harmonic_voltages_match_opendss(variant):
    """Every shunt variant agrees with OpenDSS on a 33-node feeder, orders 3 to 25.

    Measured 1.1e-12 (no shunt) to 1.6e-12 pu of nominal (CPU, complex128): the solve's
    own floor, not a model difference.
    """
    worst, *_ = _compare("ieee33_rx", variant)
    assert worst < 1e-6, f"{variant}: max |d|V_h|| = {worst:.2e} pu"


@pytest.mark.parametrize("variant", ["series_rl_0", "series_rl_50", "series_rl_100"])
def test_resonant_bank_damping_matches_opendss(variant):
    """On a parallel resonance near order 7 — where the shunt IS the damping — too.

    A 370 kvar bank at the feeder end puts the driving-point impedance peak at order
    6.9, where the undamped model overstates the peak by 40 to 75 %. Measured
    4.4e-12 to 6.8e-12 pu of nominal.
    """
    worst, *_ = _compare("ieee33_rx", variant, bank=True)
    assert worst < 1e-6, f"{variant}: max |d|V_h|| = {worst:.2e} pu"


def test_geometry_feeder_harmonic_voltages_match_opendss():
    """The default split on the Carson-geometry feeder, below OpenDSS's 1 kHz switch.

    Measured 1.3e-09 pu of nominal, the geometry path's own floor (the SI-vs-OpenDSS
    ``mu0`` constant difference carried through the solve).
    """
    worst, *_ = _compare("ieee33_geometry", "series_rl_50", geometry=True)
    assert worst < 1e-6, f"max |d|V_h|| = {worst:.2e} pu"


@pytest.mark.parametrize("variant", ["none", "series_rl_50", "motor"])
def test_cigre_lv_three_phase_harmonic_voltages_match_opendss(variant):
    """The three-phase CIGRE LV benchmark (132 rows, 3 Dyn transformers, 15 loads).

    Measured 2.6e-09 to 4.5e-09 pu of nominal, on a path whose own fundamental floor is
    3.3e-08 pu (the independent exporter's).
    """
    worst, *_ = _compare("cigre_lv", variant)
    assert worst < 1e-6, f"{variant}: max |d|V_h|| = {worst:.2e} pu"


def test_the_shunt_damps_the_resonance_peak():
    """The physics the shunt adds: the resonance peak falls, monotonically in the split.

    The parallel branch keeps the full load conductance at every order while the series
    branch's admittance falls as ``1/h``, so ``%SeriesRL=0`` damps most and
    ``%SeriesRL=100`` least. At the bank bus, order 7.
    """
    from pgml.schemas.grid_schema import Phase

    peaks = {}
    for variant in ("none", "series_rl_0", "series_rl_50", "series_rl_100"):
        load_shunt, override, _ = VARIANTS[variant]
        grid = _build_grid("ieee33_rx", bank=True)
        _set_override(grid, override)
        index = node_phase_index(grid)
        res = solve_harmonic_flow(
            grid, [1, 7], slack="norton", dtype=CDT, load_shunt=load_shunt
        )
        row = index.row(RESONANT_BANK["ieee33_rx"][0], Phase.A)
        peaks[variant] = abs(complex(res.v[1, row]))
    assert peaks["series_rl_0"] < peaks["series_rl_50"] < peaks["series_rl_100"]
    assert peaks["series_rl_100"] < peaks["none"]
    assert peaks["series_rl_0"] < 0.75 * peaks["none"]
