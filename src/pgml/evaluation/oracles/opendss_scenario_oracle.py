"""OpenDSS SCENARIO ORACLE — independent full-circuit export + batched scenario runs.

Unlike the LIVE PARITY oracles in :mod:`pgml.evaluation.oracles.opendss_oracle` (which
overwrite the transformer / source contributions of an OpenDSS ``SystemY`` with pgml's OWN
stamps to isolate one model component), this module exports a pgml :class:`~pgml.schemas
.grid_schema.Grid` as a GENUINE, independent OpenDSS circuit — a native ``Vsource``,
``Line`` (Rmatrix/Xmatrix/Cmatrix), ``Transformer`` (real windings/conn/tap/clock),
``Load``/``Generator`` (kW/kvar/conn/Model=), ``Capacitor``/``Reactor`` — so OpenDSS applies
its OWN physics end to end, with no pgml formula anywhere in the OpenDSS solve. It then
translates a realized :class:`~pgml.scenarios.SampledScenarios` batch (operating points +
harmonic injections) into per-scenario ``Edit`` commands and native DSS ``Spectrum``
objects, runs ``Solve`` (snap) + ``Solve mode=harmonics`` per requested order, and returns
the result as a :class:`~pgml.scenarios.ScenarioResult` aligned to
:func:`pgml.assembly.node_phase_index` rows — so it plugs into
:func:`pgml.scenarios.write_dataset` unchanged and can be diffed directly against
:func:`pgml.scenarios.run_scenarios`.

Two purposes
------------
1. **Numeric cross-validation**: :func:`compare_to_pgml` solves the SAME
   :class:`~pgml.scenarios.SampledScenarios` with both engines and reports the per-order
   voltage error (absolute + relative to the per-order RMS voltage), the pgml-vs-OpenDSS
   ground-truth agreement check.
2. **Independent evaluation set**: :func:`write_opendss_dataset` persists an OpenDSS-
   solved dataset in the exact same on-disk layout pgml's own generator writes, provenance-
   stamped (``engine="opendss"``), so a trained state estimator can be evaluated on data it
   never saw pgml solve.

Two assumption modes (``mode=``)
---------------------------------
- ``"matched"``: ``Set NeglectLoadY=Yes`` (pgml's harmonic model is a PURE per-device current
  source at every order — no load Norton shunt is implemented, see
  ``pgml.solver.harmonic_flow``'s ``include_load_shunt`` — so this is not merely a numerics
  nicety, it is REQUIRED for the two engines to solve the same physical model), ``Rg=0 Xg=0``
  on every line-like element (``Line``/``GenericBranch``/``Switch`` — ALL of them, including
  the SEQUENCE-form ``Switch``, which picks up the same nonzero earth-return default; pgml's
  non-geometry harmonic line models carry no Carson earth-return correction at all; OpenDSS's
  default ``Rg``/``Xg`` are calibrated for IMPERIAL units and would otherwise add a spurious,
  units-dependent zero-sequence term — see ``docs/pgml/modeling/conventions.md`` §8), a tight
  snap-solve ``Set Tolerance=1e-10``/``Set MaxIterations=100`` (OpenDSS's default ``1e-4``
  tolerance is loose enough to show up in a comparison, growing with system size/loading —
  measured up to ~2e-6 relative pre-fix on a 12-node feeder at heavy load), and — on EVERY
  exported ``Load`` (including a Generator/Storage represented as one, see below) —
  ``Vminpu=0.0001 Vmaxpu=10000`` (OpenDSS's default ``0.95``/``1.05`` band CLIPS the
  constant-power/current/ZIP law outside it, extrapolating toward constant impedance instead;
  pgml's ``LoadModel``/``ZipCoefficients`` laws have no such band — measured live on the CIGRE
  LV benchmark: a bus solved at 0.919 pu, an everyday voltage drop, made a default-banded
  Model=1 load deliver 6.8% less than its nameplate kW). ``DefaultBaseFrequency`` is always
  set from the grid's ``base_frequency_hz`` in BOTH modes (a basic modeling-correctness
  requirement, not a numerics-isolation switch). This mode isolates genuine NUMERIC agreement
  between the two harmonic engines — see the module's oracle test file for MEASURED figures
  (matched mode reaches ~1e-6 to 1e-9 relative on every tested case except one documented,
  irreducible model gap — see below).
- ``"default"``: leaves OpenDSS's own defaults (``NeglectLoadY=No`` — the load Norton shunt
  IS included in OpenDSS's harmonics but never in pgml's; earth-return ``Rg``/``Xg`` and
  ``Vminpu``/``Vmaxpu`` at their defaults). Expect a DOCUMENTED divergence from these sources,
  not a bug — this mode exists to characterize how far a "just point OpenDSS at the grid and
  solve" study would drift from pgml's own reduced model, not to be tight (measured: several
  hundred percent relative on triplen harmonics of a Dyn feeder, driven almost entirely by the
  earth-return term dominating the zero-sequence path — see ``opendss_scenario_oracle``'s test
  module docstring for the exact figures).

Exporter coverage
------------------
Converted: ``Source`` (balanced, diagonal — i.e. uncoupled — Thevenin only; the FIRST
in-service ``Source`` becomes the DSS ``Circuit``'s own slack, any further ones export as
additional ``Vsource`` elements with a warning), ``Line``/``GenericBranch`` (explicit
Rmatrix/Xmatrix/Cmatrix, phase-permuted terminals via independent ``from_phases``/
``to_phases`` bus suffixes), ``Transformer`` (native two-winding element; the 3-phase
vector-group clock is realised via ``LeadLag`` + a cyclic TO-side bus rotation, reusing the
SAME clock-realisation helpers already pinned against a live OpenDSS solve in
``opendss_oracle.py``; a 1-phase / positive-sequence-equivalent unit exports as a plain
``conn=wye`` ratio device — see the refusal list below for when its exact vector-group shift
cannot be reproduced), ``Switch`` (a near-ideal ``Line`` with ``Switch=yes`` FIRST on the
command — see ``_export_switch``'s docstring for why the parameter order matters — disabled
when open), ``Load``/``Generator``/``Storage`` (EVERY injection appliance, including a
Generator/Storage, exports as a native DSS ``Load`` — see ``_ApplianceExport``'s docstring for
why a genuine DSS ``Generator`` element cannot be used here; a WYE appliance is exported as
ONE single-phase ``Load`` PER PHASE, enabling per-phase P/Q scenario overrides a single
balanced multi-phase element cannot express; a DELTA appliance exports as one multi-phase
element and supports only a balanced total P/Q scenario override), ``ShuntAppliance``/
``ShuntReactor`` (WYE: a ``Capacitor`` for the C part + a diagonal-``Rmatrix`` ``Reactor``
for the G part; DELTA ``ShuntAppliance``: a ``conn=delta`` ``Capacitor`` (per-leg ``Cuf``) +
a ``conn=delta`` scalar-``R`` ``Reactor`` — an UNBALANCED delta bank is refused, OpenDSS's
Capacitor/Reactor banks being balanced per leg).

Refused (raises :class:`~pgml.errors.ConversionError`), with the reason:
- ``Line.conductor_geometry`` — out of scope for this exporter; the EXISTING geometry
  parity oracle (``opendss_oracle.build_opendss_geometry_circuit`` /
  ``opendss_geometry_systemy``) already gives bit-exact Carson-geometry parity for that case.
- An unresolved ``type_ref`` on a ``Line``/``Transformer`` — materialise against
  ``Grid.types`` before exporting (this module never reads the catalog).
- A ``Source`` with off-diagonal (phase-coupled) Thevenin impedance, or non-balanced
  per-phase magnitude/120°-spacing — an OpenDSS ``Vsource`` models a symmetric,
  uncoupled positive/zero-sequence source only.
- ``Transformer`` ``ZIGZAG``/``ZIGZAG_GROUNDED`` windings — no OpenDSS ``Transformer``
  connection exists for zigzag.
- A NONZERO exact phase shift on a 1-phase (positive-sequence equivalent) ``Transformer`` —
  OpenDSS's 1-phase ``Transformer`` has no delta/``LeadLag`` mechanism at all (verified live:
  ``conn=delta`` collapses to a degenerate near-zero-voltage result at ``phases=1``, and
  ``LeadLag`` has no effect between two wye windings), so only a zero-shift (plain ratio)
  1-phase unit is representable; pgml's OWN ``p==1`` stamp (``assembly.ybus
  ._transformer_block_groups``) applies the vector group's EXACT phase shift even at 1 phase
  (``resolve_vector_group(t, n_phases=1).shift_exact_deg``), so this is a genuine, not a
  lazy, refusal — a positive-sequence-equivalent grid converted with a real (e.g. Dyn) vector
  group, such as the CIGRE LV benchmark under ``PhaseMode.SINGLE_PHASE_EQUIV``, is out of
  scope for this reason.
- A 3-phase ``Transformer`` with a non-3-phase winding count other than 1 (e.g. 2 phases) —
  the clock-realising bus rotation is specific to the 3-phase A/B/C cyclic group (matches
  ``opendss_oracle``'s own scope); clock values ``{2, 6, 10}`` (the polarity-flip group) on a
  3-phase unit — no OpenDSS bus wiring reaches them (see
  ``opendss_oracle._dss_leadlag_and_rotation``).
- ``Transformer.from_grounding``/``to_grounding`` with non-zero impedance, or an explicit
  ``Transformer.zero_sequence`` override — these schema fields are NOT YET consumed by
  ``pgml.assembly`` (grep-verified: solid grounding / topology-derived Z0 only), so exporting
  them faithfully to OpenDSS would create a genuine, silent model mismatch rather than a
  representability gap; refused instead of silently diverging.
- A per-scenario per-phase P/Q override, or a per-phase nameplate, on a DELTA-connected
  appliance — a "phase" of a delta device is a LEG between two bus phases, not a
  independently-editable DSS sub-element; only the balanced total is supported.
- Any other branch/appliance schema type (there are none left uncovered today; a future
  schema addition would raise here rather than silently vanish).

Voltage-dependent load models agree at the tight matched-mode floor:
``pgml.solver.harmonic_flow`` anchors each device's spectrum to its MODEL-CONSISTENT
fundamental current (the ZIP-scaled / control-resolved ``S_eff`` at the converged
terminal voltage — the same power the nonlinear fundamental solve draws), matching
OpenDSS's per-model fundamental-current scaling for ``CONST_IMPEDANCE`` /
``CONST_CURRENT`` / ``ZIP`` loads alongside the const-power default.

``opendssdirect`` is imported lazily (inside functions), so this module is importable
without the package installed, matching the rest of ``pgml.evaluation.oracles``.
"""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from pgml.assembly import node_phase_index
from pgml.errors import ConversionError, InputError
from pgml.evaluation._util import to_float
from pgml.evaluation.oracles.opendss_oracle import (
    _dss_leadlag_and_rotation,
    _dss_rotated_phase_suffix,
)
from pgml.schemas.grid_schema import (
    Generator,
    GenericBranch,
    Grid,
    Line,
    Load,
    LoadModel,
    Phase,
    ShuntAppliance,
    ShuntReactor,
    Source,
    Storage,
    Switch,
    Transformer,
    WindingConnection,
)
from pgml.scenarios import SampledScenarios, ScenarioResult

_logger = logging.getLogger("pgml")

_SQRT3 = math.sqrt(3.0)
_PHASE_SUFFIX = {Phase.A: 1, Phase.B: 2, Phase.C: 3, Phase.N: 4}
_SUFFIX_PHASE = {v: k for k, v in _PHASE_SUFFIX.items()}
_DSS_LOAD_MODEL = {
    LoadModel.CONST_POWER: 1,
    LoadModel.CONST_IMPEDANCE: 2,
    LoadModel.CONST_CURRENT: 5,
    LoadModel.ZIP: 8,
}
#: One fundamental-only Spectrum (100% at h=1, nothing else), assigned EXPLICITLY to
#: every exported Load/Vsource so no element silently inherits OpenDSS's built-in
#: "defaultload"/"defaultgen"/"defaultvsource" spectra (which carry harmonic content a
#: pgml device without a spectrum -- injecting nothing at h>1 -- does not have). A
#: device's real, sampled injection (`_attach_spectra`) OVERRIDES this per element.
_FLAT_SPECTRUM_NAME = "pgml_flat1"


@contextlib.contextmanager
def _scratch_datapath():
    """Confine OpenDSS's scratch-file writes to a throwaway temp dir for one live solve.

    ``Solve mode=harmonics`` auto-writes a ``<circuit>_SavedVoltages.dbl`` scratch file to
    OpenDSS's OWN notion of its current directory -- which is set ONLY by
    ``opendssdirect``'s ``Basic.DataPath`` (a bare ``os.chdir`` does NOT change it,
    verified empirically: DSS keeps its own internal path pointer, refreshed only via
    ``DataPath``). ``Basic.DataPath`` itself chdirs the WHOLE process as a side effect, so
    this context manager restores it (via another ``DataPath`` call, the only mechanism
    that actually resets DSS's internal pointer too -- a bare ``os.chdir`` back would leave
    DSS still pointed at the (about-to-be-deleted) scratch dir for every LATER solve in the
    same process) before removing the scratch directory, leaving both the caller's `cwd`
    and DSS's own path state exactly as they were.
    """
    import opendssdirect as dss

    prev_cwd = os.getcwd()
    scratch = tempfile.mkdtemp(prefix="pgml_opendss_scenario_oracle_")
    dss.Basic.DataPath(scratch)
    try:
        yield
    finally:
        dss.Basic.DataPath(prev_cwd)
        shutil.rmtree(scratch, ignore_errors=True)


def _bus_conductor_str(phases: Sequence[Phase]) -> str:
    return ".".join(str(_PHASE_SUFFIX[p]) for p in phases)


def _scalar_at(x, b: int, t: Optional[int] = None) -> float:
    """Python float at scenario ``b`` (+ step ``t``) from a batched tensor or a plain number.

    Honors the ``SampledScenarios`` convention: a spec-varied field is ``Tensor[B]``
    (or ``[B, T]`` for a coherent harmonic injection); an order untouched by any spec
    but seeded from a device's stored spectrum stays a plain python float (broadcasts
    to every scenario/step).
    """
    if isinstance(x, (int, float)):
        return float(x)
    xt = x.detach() if hasattr(x, "detach") else x
    ndim = getattr(xt, "ndim", 0)
    if ndim == 0:
        return float(xt)
    if ndim == 1:
        return float(xt[b])
    return float(xt[b, t if t is not None else 0])


# ---------------------------------------------------------------------------
# Export result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _ApplianceExport:
    """How one Load/Generator/Storage appliance maps to native DSS elements.

    ``kind="split"``: one single-phase WYE DSS element PER connected phase (``elements``
    keyed by :class:`Phase`) — supports both a balanced total and an independent
    per-phase scenario override. ``kind="whole"``: one multi-phase DELTA DSS element
    (``elements={None: name}``) — balanced total only.

    ``dss_class`` is ALWAYS ``"Load"`` (see the module docstring's "Generator/Storage
    export as a negative-kW Load" note): a genuine DSS ``Generator`` element stamps its
    own linearized PQ shunt admittance into the harmonics-mode linear system REGARDLESS
    of ``NeglectLoadY``/``Model``/``Xdpp`` (verified empirically — no combination zeroes
    it), so a pgml ``Generator``/``Storage`` (a pure current injection, no internal
    admittance) is represented as a DSS ``Load`` with a NEGATED P/Q — the well-established
    OpenDSS negative-load generation idiom — which DOES become a true pure current source
    under ``NeglectLoadY=Yes`` (verified: post-solve ``YPrim`` ~1e-12, vs ~0.0375 S for a
    genuine ``Generator`` element on the same nameplate). ``sign`` (``+1.0`` for a real
    ``Load``, ``-1.0`` for a ``Generator``/``Storage``) is applied to every P/Q value
    written to this element, both at export and at every per-scenario edit.
    """

    kind: str
    elements: dict
    phases: tuple
    p_nom_pp: list
    q_nom_pp: list
    dss_class: str
    sign: float = 1.0


@dataclass(frozen=True)
class ExportedCircuit:
    """A live OpenDSS circuit built from a pgml :class:`~pgml.schemas.grid_schema.Grid`.

    Captures everything :func:`run_opendss_scenarios` needs to edit and re-solve the SAME
    circuit for every scenario without rebuilding it: the node-id -> DSS-bus-name map, the
    (stable, captured once) DSS row order aligned to :func:`~pgml.assembly.node_phase_index`
    rows, and the appliance/source -> DSS-element-name maps. ``spectra`` starts empty and is
    populated by :func:`run_opendss_scenarios` once it knows which orders are being solved
    (the export itself does not depend on a scenario batch).
    """

    grid: Grid
    mode: str
    busname: dict
    node_order: list
    rowmap: list
    index: object
    loads: dict
    generators: dict
    sources: dict
    spectra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Source (Circuit slack + any additional Vsource)
# ---------------------------------------------------------------------------
def _source_params(src: Source, f0: float) -> dict:
    n = len(src.phases)
    r_mat = src.resistance_ohm
    l_mat = src.inductance_h
    for i in range(n):
        for j in range(n):
            if i != j and (
                abs(to_float(r_mat[i][j])) > 1e-9 or abs(to_float(l_mat[i][j])) > 1e-9
            ):
                raise ConversionError(
                    f"Source {src.id}: off-diagonal (phase-coupled) Thevenin "
                    "impedance is not representable by an OpenDSS Vsource "
                    "(R1/X1/R0/X0 sequence parameters model a symmetric, "
                    "uncoupled source only)."
                )
    r1 = to_float(r_mat[0][0])
    x1 = 2.0 * math.pi * f0 * to_float(l_mat[0][0])
    u0 = to_float(src.u_ref_v[0])
    ang0 = to_float(src.u_angle_deg[0])
    if n >= 3:
        for k in range(1, n):
            uk = to_float(src.u_ref_v[k])
            if abs(uk - u0) > 1e-6 * max(abs(u0), 1.0):
                raise ConversionError(
                    f"Source {src.id}: unbalanced per-phase voltage magnitude "
                    "is not representable by an OpenDSS Vsource (models a "
                    "balanced source only)."
                )
        expected = {1: (ang0 - 120.0) % 360.0, 2: (ang0 + 120.0) % 360.0}
        for k in range(1, min(n, 3)):
            ak = to_float(src.u_angle_deg[k]) % 360.0
            if abs(((ak - expected[k] + 180.0) % 360.0) - 180.0) > 1e-3:
                raise ConversionError(
                    f"Source {src.id}: non-standard 120-degree phase spacing "
                    "is not representable by an OpenDSS Vsource (assumes a "
                    "balanced 3-phase source)."
                )
        basekv = u0 * _SQRT3 / 1000.0
    else:
        basekv = u0 / 1000.0
    return {
        "phases": n,
        "basekv": basekv,
        "pu": 1.0,
        "angle": ang0,
        "r1": r1,
        "x1": x1,
        "bus_suffix": _bus_conductor_str(src.phases),
    }


def _emit_vsource(
    dss, cmd_prefix: str, name: str, bus: str, p: dict, f0: float
) -> None:
    seq0 = f" r0={p['r1']:.10g} x0={p['x1']:.10g}" if p["phases"] >= 3 else ""
    dss.Text.Command(
        f"{cmd_prefix}.{name} basekv={p['basekv']:.10g} phases={p['phases']} "
        f"bus1={bus} pu={p['pu']:.10g} angle={p['angle']:.10g} frequency={f0:.10g} "
        f"r1={p['r1']:.10g} x1={p['x1']:.10g}{seq0} spectrum={_FLAT_SPECTRUM_NAME}"
    )


# ---------------------------------------------------------------------------
# Line / GenericBranch
# ---------------------------------------------------------------------------
def _matrix_command_body(n: int, r_of, x_of, c_of) -> str:
    r_rows, x_rows, c_rows = [], [], []
    for i in range(n):
        r_rows.append(" ".join(f"{r_of(i, j):.10g}" for j in range(i + 1)))
        x_rows.append(" ".join(f"{x_of(i, j):.10g}" for j in range(i + 1)))
        c_rows.append(" ".join(f"{c_of(i, j) * 1e9:.10g}" for j in range(i + 1)))
    return (
        f"Rmatrix=[{' | '.join(r_rows)}] Xmatrix=[{' | '.join(x_rows)}] "
        f"Cmatrix=[{' | '.join(c_rows)}]"
    )


def _export_line(dss, ln: Line, busname: dict, f0: float, mode: str) -> None:
    if ln.type_ref is not None and ln.series_resistance_ohm_per_m is None:
        raise ConversionError(
            f"Line {ln.id}: unresolved type_ref; materialise against Grid.types "
            "before exporting."
        )
    if ln.conductor_geometry is not None:
        raise ConversionError(
            f"Line {ln.id}: conductor_geometry lines are out of scope for this "
            "full-circuit scenario exporter -- use "
            "pgml.evaluation.oracles.opendss_oracle.build_opendss_geometry_circuit "
            "/ opendss_geometry_systemy (bit-exact Carson-geometry parity) for a "
            "geometry-based grid instead."
        )
    n = len(ln.from_phases)
    r, ell, c, g = (
        ln.series_resistance_ohm_per_m,
        ln.series_inductance_h_per_m,
        ln.shunt_capacitance_f_per_m,
        ln.shunt_conductance_s_per_m,
    )
    if g is not None and any(
        to_float(g[i][j]) != 0.0 for i in range(n) for j in range(n)
    ):
        _logger.warning(
            "Line %s: shunt conductance G has no OpenDSS Line equivalent "
            "(Rmatrix/Xmatrix/Cmatrix only) -- dropped.",
            ln.id,
        )
    length = to_float(ln.length_m)
    body = _matrix_command_body(
        n,
        lambda i, j: to_float(r[i][j]),
        lambda i, j: 2.0 * math.pi * f0 * to_float(ell[i][j]),
        lambda i, j: to_float(c[i][j]) if c is not None else 0.0,
    )
    bus1 = f"{busname[ln.from_node]}.{_bus_conductor_str(ln.from_phases)}"
    bus2 = f"{busname[ln.to_node]}.{_bus_conductor_str(ln.to_phases)}"
    earth = " Rg=0 Xg=0" if mode == "matched" else ""
    dss.Text.Command(
        f"New Line.l{ln.id} phases={n} bus1={bus1} bus2={bus2} {body} "
        f"length={length:.10g} units=m{earth}"
    )


def _export_generic_branch(
    dss, br: GenericBranch, busname: dict, f0: float, mode: str
) -> None:
    n = len(br.from_phases)
    r, ell = br.series_resistance_ohm, br.series_inductance_h
    c_from, c_to = br.shunt_capacitance_from_f, br.shunt_capacitance_to_f
    if c_from is not None and c_to is not None:
        if any(
            abs(to_float(c_from[i][j]) - to_float(c_to[i][j])) > 1e-15
            for i in range(n)
            for j in range(n)
        ):
            _logger.warning(
                "GenericBranch %s: differing from/to shunt capacitance is "
                "approximated as the symmetric average (OpenDSS's Line Cmatrix "
                "splits identically to both ends).",
                br.id,
            )

        def c_of(i, j):
            return 0.5 * (to_float(c_from[i][j]) + to_float(c_to[i][j]))
    elif c_from is not None:

        def c_of(i, j):
            return to_float(c_from[i][j])
    elif c_to is not None:

        def c_of(i, j):
            return to_float(c_to[i][j])
    else:

        def c_of(i, j):
            return 0.0

    body = _matrix_command_body(
        n,
        lambda i, j: to_float(r[i][j]),
        lambda i, j: 2.0 * math.pi * f0 * to_float(ell[i][j]),
        c_of,
    )
    bus1 = f"{busname[br.from_node]}.{_bus_conductor_str(br.from_phases)}"
    bus2 = f"{busname[br.to_node]}.{_bus_conductor_str(br.to_phases)}"
    earth = " Rg=0 Xg=0" if mode == "matched" else ""
    dss.Text.Command(
        f"New Line.gb{br.id} phases={n} bus1={bus1} bus2={bus2} {body} "
        f"length=1 units=none{earth}"
    )


def _export_switch(dss, sw: Switch, busname: dict, f0: float, mode: str) -> None:
    """A near-ideal ``Line`` (``Switch=yes``, disabled when open).

    Two OpenDSS-specific gotchas, both verified live on a minimal switch-only repro
    (a plain 3-phase feeder is exact; adding one switch alone desyncs a matched-mode
    comparison by up to ~0.4% at non-triplen harmonics):

    1. **Parameter ORDER matters for ``Switch=yes``.** Setting ``Switch=yes`` AFTER
       ``r1``/``x1``/``r0``/``x0`` on the SAME ``New Line`` command silently RESETS the
       line's impedance to an internal built-in default (observed: R~0.001 Ohm, X~0 at
       fundamental) regardless of the values just given -- confirmed by reading back
       the element's own ``YPrim`` (``0.001+0.001j`` Ohm instead of the specified
       ``0.0001+0j``). Putting ``Switch=yes`` FIRST on the command avoids the reset.
    2. Like every other line-like element in "matched" mode, the SEQUENCE form
       (``r1/x1/r0/x0``, unlike ``_export_line``'s matrix form) still picks up OpenDSS's
       nonzero default earth-return ``Rg``/``Xg`` correction -- confirmed by reading it
       back at harmonics even after fix #1 (a genuinely zero-reactance resistor should
       show NO frequency dependence at all; without ``Rg=0 Xg=0`` it does) -- so the
       SAME ``Rg=0 Xg=0`` suffix as every matrix-form ``Line`` is required here too.
    """
    n = len(sw.from_phases)
    bus1 = f"{busname[sw.from_node]}.{_bus_conductor_str(sw.from_phases)}"
    bus2 = f"{busname[sw.to_node]}.{_bus_conductor_str(sw.to_phases)}"
    r = to_float(sw.resistance_ohm)
    x = 2.0 * math.pi * f0 * to_float(sw.inductance_h)
    r_eff = r if r > 0.0 else 1.0e-6
    enabled = "yes" if sw.closed else "no"
    earth = " Rg=0 Xg=0" if mode == "matched" else ""
    dss.Text.Command(
        f"New Line.sw{sw.id} Switch=yes phases={n} bus1={bus1} bus2={bus2} "
        f"r1={r_eff:.10g} x1={x:.10g} c1=0 r0={r_eff:.10g} x0={x:.10g} c0=0 "
        f"length=1 units=none enabled={enabled}{earth}"
    )


# ---------------------------------------------------------------------------
# Transformer (native two-winding element, real vector-group clock)
# ---------------------------------------------------------------------------
def _transformer_pct_r_xhl(
    t: Transformer, w0: float, u_to_kv: float, kva_ref: float
) -> tuple:
    """``(pct_r_per_winding, xhl)`` from the LV-coil-referred series R/L (shared by the
    1- and 3-phase export paths -- the leakage-referral algebra does not depend on
    phase count, only on whether the LV winding is DELTA)."""
    is_delta_to = t.to_connection == WindingConnection.DELTA
    r_t = to_float(t.series_resistance_ohm)
    x_t = w0 * to_float(t.series_inductance_h)
    lv_coil_factor = 3.0 if is_delta_to else 1.0
    r_ll = r_t / lv_coil_factor
    x_ll = x_t / lv_coil_factor
    z_base_lv = (u_to_kv**2 * 1.0e6) / (kva_ref * 1.0e3)
    vkr_total = (r_ll / z_base_lv) * 100.0
    vk_total = (abs(complex(r_ll, x_ll)) / z_base_lv) * 100.0
    xhl = math.sqrt(max(vk_total**2 - vkr_total**2, 0.0))
    return vkr_total / 2.0, xhl


def _transformer_magnetizing_pct(
    t: Transformer, w0: float, u_from_kv: float, kva_ref: float
) -> Optional[tuple]:
    """``(pct_noloadloss, pct_imag)`` or ``None`` when the unit has no magnetizing branch."""
    g_m = to_float(t.magnetizing_conductance_s)
    u_hv_v = u_from_kv * 1000.0
    pfe_w = g_m * u_hv_v**2
    b_m = (
        1.0 / (w0 * to_float(t.magnetizing_inductance_h))
        if t.magnetizing_inductance_h is not None
        else 0.0
    )
    q_nl = b_m * u_hv_v**2
    s_nl = math.hypot(pfe_w, q_nl)
    s_rated = kva_ref * 1000.0
    if s_rated > 0.0 and (pfe_w > 0.0 or s_nl > 0.0):
        return pfe_w / s_rated * 100.0, s_nl / s_rated * 100.0
    return None


def _export_transformer(dss, t: Transformer, busname: dict, f0: float) -> None:
    if t.from_connection is None or t.u_rated_from_v is None:
        raise ConversionError(
            f"Transformer {t.id}: unresolved type_ref/ratings; materialise "
            "against Grid.types before exporting."
        )
    if t.from_connection in (
        WindingConnection.ZIGZAG,
        WindingConnection.ZIGZAG_GROUNDED,
    ) or t.to_connection in (
        WindingConnection.ZIGZAG,
        WindingConnection.ZIGZAG_GROUNDED,
    ):
        raise ConversionError(
            f"Transformer {t.id}: OpenDSS's Transformer element has no zigzag "
            "winding connection; not representable."
        )
    p = len(t.from_phases)
    if p not in (1, 3):
        raise ConversionError(
            f"Transformer {t.id}: only 1-phase (positive-sequence equivalent) "
            f"or 3-phase transformers are supported (the clock-realising bus "
            f"rotation is specific to the 3-phase A/B/C cyclic group; got "
            f"{p} phases)."
        )
    for grounding, side in ((t.from_grounding, "HV"), (t.to_grounding, "LV")):
        if grounding is not None and (
            to_float(grounding.r_ohm) != 0.0 or to_float(grounding.x_ohm) != 0.0
        ):
            raise ConversionError(
                f"Transformer {t.id}: an impedance-grounded {side} neutral is "
                "not exported -- pgml.assembly does not yet consume "
                "from_grounding/to_grounding (solid grounding only), so "
                "faithfully reproducing it on the OpenDSS side would silently "
                "diverge from what pgml actually solves."
            )
    if t.zero_sequence is not None:
        raise ConversionError(
            f"Transformer {t.id}: an explicit zero_sequence override is not "
            "exported -- pgml.assembly does not yet consume it (the "
            "zero-sequence path is always topology-derived), so exporting a "
            "matching OpenDSS override is not possible without diverging from "
            "what pgml actually solves."
        )

    w0 = 2.0 * math.pi * f0
    u_from_kv = to_float(t.u_rated_from_v) / 1000.0
    u_to_kv = to_float(t.u_rated_to_v) / 1000.0
    kva_ref = to_float(t.s_rated_va) / 1000.0
    pct_r, xhl = _transformer_pct_r_xhl(t, w0, u_to_kv, kva_ref)
    xrconst = "Yes" if t.harmonic_xr_constant else "No"

    if p == 1:
        # Single-phase / positive-sequence equivalent: pgml's OWN stamp
        # (`assembly.ybus._transformer_block_groups`, p==1 branch) folds the
        # vector group into an EXACT complex ratio `n*e^{j*shift_exact_deg}` on
        # a plain scalar off-nominal-tap pi -- there is no 3-phase winding
        # topology at all, so `from_connection`/`to_connection` (DELTA/WYE/
        # WYE_GROUNDED) affect only the leakage-coil referral above, never a
        # DSS `conn=` keyword. Empirically verified (no bus-conductor/LeadLag
        # lever reaches this at phases=1): a DSS Transformer with `conn=delta`
        # collapses to a near-zero-voltage DEGENERATE result at `phases=1` (a
        # delta winding needs >=2 conductors to form a loop) and `LeadLag`
        # has NO effect when both windings are wye (verified on a live solve)
        # -- so a NONZERO exact phase shift genuinely cannot be represented by
        # a native 1-phase DSS Transformer and is refused rather than
        # silently dropped; a zero (or negligible) shift exports as a plain
        # `conn=wye` ratio device (matches the historical IEEE-33-style unit).
        from pgml.assembly._transformer import resolve_vector_group

        vg = resolve_vector_group(t, n_phases=1)
        shift = float(vg.shift_exact_deg) % 360.0
        if min(shift, 360.0 - shift) > 1e-6:
            raise ConversionError(
                f"Transformer {t.id}: a NONZERO phase shift "
                f"({vg.shift_exact_deg:.6g} deg) on a single-phase "
                "(positive-sequence equivalent) transformer is not "
                "representable by an OpenDSS Transformer element -- it has no "
                "delta/LeadLag mechanism at phases=1 (verified empirically: "
                "conn=delta collapses to a degenerate near-zero-voltage "
                "result, and LeadLag has no effect between two wye windings) "
                "-- only a zero-shift (plain ratio) single-phase unit can be "
                "exported."
            )
        bus_from = f"{busname[t.from_node]}.1"
        bus_to = f"{busname[t.to_node]}.1"
        dss.Text.Command(f"New Transformer.T{t.id} phases=1 windings=2")
        dss.Text.Command(
            f"~ wdg=1 bus={bus_from} conn=wye kV={u_from_kv:.8g} "
            f"kVA={kva_ref:.6g} %R={pct_r:.10g}"
        )
        dss.Text.Command(
            f"~ wdg=2 bus={bus_to} conn=wye kV={u_to_kv:.8g} "
            f"kVA={kva_ref:.6g} %R={pct_r:.10g}"
        )
        dss.Text.Command(f"~ XHL={xhl:.10g} XRConst={xrconst}")
    else:
        is_delta_from = t.from_connection == WindingConnection.DELTA
        is_delta_to = t.to_connection == WindingConnection.DELTA
        shift = to_float(t.tap.shift_deg)
        clock = int(round(shift / 30.0)) % 12
        shifting_pairing = is_delta_from != is_delta_to
        lead_lag, r_to = _dss_leadlag_and_rotation(clock, shifting_pairing)

        from_conn = "delta" if is_delta_from else "wye"
        to_conn = "delta" if is_delta_to else "wye"
        bus_from = f"{busname[t.from_node]}.{_dss_rotated_phase_suffix(p, 0)}"
        bus_to = f"{busname[t.to_node]}.{_dss_rotated_phase_suffix(p, r_to)}"

        dss.Text.Command(f"New Transformer.T{t.id} windings=2")
        dss.Text.Command(
            f"~ wdg=1 bus={bus_from} conn={from_conn} kV={u_from_kv:.8g} "
            f"kVA={kva_ref:.6g} %R={pct_r:.10g}"
        )
        dss.Text.Command(
            f"~ wdg=2 bus={bus_to} conn={to_conn} kV={u_to_kv:.8g} "
            f"kVA={kva_ref:.6g} %R={pct_r:.10g} Rneut=0 Xneut=0"
        )
        dss.Text.Command(f"~ XHL={xhl:.10g} XRConst={xrconst} LeadLag={lead_lag}")

    mag = _transformer_magnetizing_pct(t, w0, u_from_kv, kva_ref)
    if mag is not None:
        dss.Text.Command(f"~ %noloadloss={mag[0]:.10g} %imag={mag[1]:.10g}")


# ---------------------------------------------------------------------------
# Load / Generator / Storage (Storage -> DSS Generator, snapshot-signed)
# ---------------------------------------------------------------------------
def _dss_load_model(a) -> tuple:
    lm = a.load_model
    if lm in (
        LoadModel.CONST_POWER,
        LoadModel.CONST_IMPEDANCE,
        LoadModel.CONST_CURRENT,
    ):
        return _DSS_LOAD_MODEL[lm], None
    if lm == LoadModel.ZIP:
        z = a.zip_coefficients
        return 8, [z.z_p, z.i_p, z.p_p, z.z_q, z.i_q, z.p_q]
    raise ConversionError(  # pragma: no cover - defensive, LoadModel is a closed enum
        f"appliance {a.id}: load_model {lm!r} has no OpenDSS Model= mapping."
    )


def _emit_pq_element(dss, name, bus, n, kv, kw, kvar, conn, model, zipv) -> None:
    """Emit a native DSS ``Load`` element (see ``_ApplianceExport``'s docstring for why
    EVERY injection appliance -- including a pgml ``Generator``/``Storage`` -- exports as
    a ``Load``, with a negated P/Q for the generation-type ones).

    ``Vminpu``/``Vmaxpu`` are set to an effectively unbounded range: OpenDSS's default
    (``0.95``/``1.05``) CLIPS every load model's constant-power/current/ZIP law outside
    that per-unit voltage band (extrapolating toward constant impedance instead) --
    pgml's ``LoadModel``/``ZipCoefficients`` laws have NO such band, they apply exactly
    at any voltage. On a real feeder under load this is not a corner case: verified live
    on the CIGRE LV benchmark, a downstream bus solved at 0.919 pu (a realistic, everyday
    voltage drop) made DSS's default-banded Model=1 load deliver 6.8% LESS than its
    nameplate kW -- silently double-counting a voltage-support behaviour pgml's model
    does not have.
    """
    dss.Text.Command(
        f"New Load.{name} phases={n} bus1={bus} kV={kv:.10g} kW={kw:.10g} "
        f"kvar={kvar:.10g} conn={conn} model={model} spectrum={_FLAT_SPECTRUM_NAME} "
        f"Vminpu=0.0001 Vmaxpu=10000"
    )
    if zipv is not None:
        zstr = " ".join(f"{v:.10g}" for v in zipv)
        dss.Text.Command(f"Edit Load.{name} ZIPV=[{zstr}]")


def _export_injection_appliance(
    dss, a, node, busname: dict, *, name_prefix: str, sign: float = 1.0
) -> _ApplianceExport:
    """Export a Load/Generator/Storage appliance as native DSS ``Load`` element(s).

    ``sign`` is ``+1.0`` for a real ``Load`` (consumption-positive, matches DSS's own
    ``Load`` convention directly) and ``-1.0`` for a ``Generator``/``Storage``
    (generation-positive in pgml; negated so the exported ``Load``'s ``kW``/``kvar`` -- and
    every later per-scenario edit -- carry the correct sign for DSS's consumption-positive
    ``Load`` convention while representing net injection).
    """
    from pgml.assembly._params import phase_voltage_magnitude

    phases = a.phases
    n = len(phases)
    is_delta = a.connection == WindingConnection.DELTA
    p_total = to_float(a.p_nom_w)
    q_total = to_float(a.q_nom_var)
    p_pp = (
        [to_float(x) for x in a.p_nom_per_phase_w]
        if a.p_nom_per_phase_w is not None
        else [p_total / n] * n
    )
    q_pp = (
        [to_float(x) for x in a.q_nom_per_phase_var]
        if a.q_nom_per_phase_var is not None
        else [q_total / n] * n
    )
    model, zipv = _dss_load_model(a)
    node_id = int(a.node)

    if is_delta:
        if a.p_nom_per_phase_w is not None or a.q_nom_per_phase_var is not None:
            raise ConversionError(
                f"appliance {a.id}: an explicit per-phase P/Q nameplate on a "
                "DELTA-connected appliance has no unambiguous single-element "
                "OpenDSS representation (a 'phase' of a delta device is a leg "
                "between two bus phases); only a balanced total is supported "
                "for a DELTA appliance in this exporter."
            )
        kv = (
            phase_voltage_magnitude(
                to_float(node.u_rated_v), len(node.phases), line_to_line=True
            )
            / 1000.0
        )
        name = f"{name_prefix}{a.id}"
        bus = f"{busname[node_id]}.{_bus_conductor_str(phases)}"
        _emit_pq_element(
            dss,
            name,
            bus,
            n,
            kv,
            sign * p_total / 1000.0,
            sign * q_total / 1000.0,
            "delta",
            model,
            zipv,
        )
        return _ApplianceExport(
            kind="whole",
            elements={None: name},
            phases=phases,
            p_nom_pp=p_pp,
            q_nom_pp=q_pp,
            dss_class="Load",
            sign=sign,
        )

    kv = (
        phase_voltage_magnitude(
            to_float(node.u_rated_v), len(node.phases), line_to_line=False
        )
        / 1000.0
    )
    # WYE return conductor: mirror the assembly rule (_incidence) — an appliance
    # returns through the node's explicit neutral when return_path="neutral", or
    # "auto" on a Phase.N-carrying node; "ground" (or a node without Phase.N)
    # keeps DSS's implicit ground (a 1-conductor load pads terminal 2 to node 0).
    return_path = getattr(a, "return_path", "auto")
    node_has_n = Phase.N in node.phases
    if return_path == "neutral" and not node_has_n:
        raise ConversionError(
            f"appliance {a.id}: return_path='neutral' but node {node_id} carries "
            "no Phase.N conductor."
        )
    neutral_return = node_has_n and return_path in ("auto", "neutral")
    n_suffix = f".{_PHASE_SUFFIX[Phase.N]}" if neutral_return else ""
    elements = {}
    for k, ph in enumerate(phases):
        name = f"{name_prefix}{a.id}_{ph.value}"
        bus = f"{busname[node_id]}.{_PHASE_SUFFIX[ph]}{n_suffix}"
        _emit_pq_element(
            dss,
            name,
            bus,
            1,
            kv,
            sign * p_pp[k] / 1000.0,
            sign * q_pp[k] / 1000.0,
            "wye",
            model,
            zipv,
        )
        elements[ph] = name
    return _ApplianceExport(
        kind="split",
        elements=elements,
        phases=phases,
        p_nom_pp=p_pp,
        q_nom_pp=q_pp,
        dss_class="Load",
        sign=sign,
    )


# ---------------------------------------------------------------------------
# ShuntAppliance / ShuntReactor -> Capacitor (+ Reactor for the G part)
# ---------------------------------------------------------------------------
def _export_shunt(
    dss,
    name: str,
    bus: str,
    phases,
    conductance_s,
    capacitance_f,
    node,
    f0: float,
    *,
    connection: WindingConnection = WindingConnection.WYE,
) -> None:
    from pgml.assembly._params import phase_voltage_magnitude

    n = len(phases)
    g = [to_float(x) for x in conductance_s]
    c = [to_float(x) for x in capacitance_f]
    bus_str = f"{bus}.{_bus_conductor_str(phases)}"

    if connection is WindingConnection.DELTA:
        # OpenDSS's Capacitor/Reactor banks are BALANCED per leg (a single Cuf / R
        # applies to every phase-to-phase leg), and the leg value is used directly
        # (verified live: Cuf == per-leg C, R == 1/(per-leg G)); an unbalanced delta
        # bank has no single-element representation.
        if any(abs(x - c[0]) > 1e-15 * max(1.0, abs(c[0])) for x in c) or any(
            abs(x - g[0]) > 1e-15 * max(1.0, abs(g[0])) for x in g
        ):
            raise ConversionError(
                f"shunt {name}: an UNBALANCED DELTA ShuntAppliance (unequal per-leg "
                "G/C) has no single balanced-bank OpenDSS representation; only a "
                "balanced delta bank is exported."
            )
        kv_ll = (
            phase_voltage_magnitude(
                to_float(node.u_rated_v), len(node.phases), line_to_line=True
            )
            / 1000.0
        )
        if abs(c[0]) > 1e-15:
            dss.Text.Command(
                f"New Capacitor.{name}c phases={n} bus1={bus_str} kV={kv_ll:.10g} "
                f"Cuf=[{c[0] * 1.0e6:.10g}] conn=delta"
            )
        if abs(g[0]) > 1e-15:
            dss.Text.Command(
                f"New Reactor.{name}r phases={n} bus1={bus_str} "
                f"R={1.0 / g[0]:.10g} X=0 conn=delta"
            )
        return

    kv_ln = (
        phase_voltage_magnitude(
            to_float(node.u_rated_v), len(node.phases), line_to_line=False
        )
        / 1000.0
    )
    if any(abs(x) > 1e-15 for x in c):
        cuf_str = " ".join(f"{x * 1.0e6:.10g}" for x in c)
        dss.Text.Command(
            f"New Capacitor.{name}c phases={n} bus1={bus_str} kV={kv_ln:.10g} "
            f"Cuf=[{cuf_str}] conn=wye"
        )
    if any(abs(x) > 1e-15 for x in g):
        rows = []
        for i in range(n):
            rows.append(
                " ".join(
                    (f"{1.0 / g[i]:.10g}" if i == j else "0") for j in range(i + 1)
                )
            )
        xrows = [" ".join("0" for _ in range(i + 1)) for i in range(n)]
        dss.Text.Command(
            f"New Reactor.{name}r phases={n} bus1={bus_str} "
            f"Rmatrix=[{' | '.join(rows)}] Xmatrix=[{' | '.join(xrows)}]"
        )


# ---------------------------------------------------------------------------
# Row alignment
# ---------------------------------------------------------------------------
def _build_rowmap(node_order: list, busname: dict, index) -> list:
    inv = {v: k for k, v in busname.items()}
    rowmap = []
    for entry in node_order:
        parts = entry.split(".")
        bus = parts[0].lower()
        node_id = inv[bus]
        suffix = int(parts[1]) if len(parts) > 1 else 1
        phase = _SUFFIX_PHASE.get(suffix, Phase.A)
        rowmap.append(index.row(node_id, phase))
    return rowmap


def _extract_voltages(dss, rowmap: list, n: int) -> np.ndarray:
    flat = np.array(dss.Circuit.AllBusVolts(), dtype=np.float64)
    v_dss = flat[0::2] + 1j * flat[1::2]
    out = np.zeros(n, dtype=complex)
    for di, row in enumerate(rowmap):
        out[row] = v_dss[di]
    return out


# ---------------------------------------------------------------------------
# Full-circuit export (public)
# ---------------------------------------------------------------------------
def export_grid_to_opendss(
    grid: Grid, *, mode: str = "matched", circuit_name: str = "pgml_scenario_oracle"
) -> ExportedCircuit:
    """Export ``grid`` as a genuine, independent OpenDSS circuit (own opendssdirect engine).

    Builds native ``Vsource``/``Line``/``Transformer``/``Load``/``Generator``/``Capacitor``/
    ``Reactor`` elements (see the module docstring for exact coverage and refusals), solves
    an initial nominal snapshot to validate the circuit and capture the stable DSS row order,
    and returns an :class:`ExportedCircuit` ready for :func:`run_opendss_scenarios` to edit
    and re-solve per scenario.

    Parameters
    ----------
    grid:
        A materialised (no unresolved ``type_ref``) :class:`~pgml.schemas.grid_schema.Grid`.
    mode:
        ``"matched"`` (default) sets ``NeglectLoadY=Yes`` and ``Rg=Xg=0`` on every line —
        the model pgml's own harmonic solver implements. ``"default"`` leaves OpenDSS's own
        defaults (a load Norton shunt at harmonics, imperial-calibrated earth return) — see
        the module docstring for what to expect from each.
    circuit_name:
        The DSS ``Circuit`` name.

    Raises
    ------
    pgml.errors.ConversionError
        For any grid feature this exporter does not (yet, or by design) represent — see the
        module docstring's refusal list.
    """
    if mode not in ("matched", "default"):
        raise InputError(f"mode must be 'matched' or 'default', got {mode!r}.")
    import opendssdirect as dss

    f0 = float(grid.base_frequency_hz)
    index = node_phase_index(grid)
    busname = {int(n.id): f"n{int(n.id)}" for n in grid.nodes}
    node_by_id = {int(n.id): n for n in grid.nodes}

    dss.Text.Command("Clear")
    dss.Text.Command(f"Set DefaultBaseFrequency={f0:.10g}")
    # Defined BEFORE anything references it; every Load/Vsource is assigned this
    # explicitly (see `_FLAT_SPECTRUM_NAME`'s docstring) so nothing silently inherits
    # OpenDSS's own non-trivial "defaultload"/"defaultgen"/"defaultvsource" spectra.
    dss.Text.Command(
        f"New Spectrum.{_FLAT_SPECTRUM_NAME} NumHarm=1 harmonic=[1] %mag=[100] angle=[0]"
    )

    sources = [a for a in grid.appliances if isinstance(a, Source) and a.in_service]
    if not sources:
        raise ConversionError(
            "grid has no in-service Source; the OpenDSS scenario oracle needs one "
            "to anchor the DSS Circuit's slack Vsource."
        )
    src0 = sources[0]
    p0 = _source_params(src0, f0)
    seq0 = f" r0={p0['r1']:.10g} x0={p0['x1']:.10g}" if p0["phases"] >= 3 else ""
    dss.Text.Command(
        f"New Circuit.{circuit_name} basekv={p0['basekv']:.10g} phases={p0['phases']} "
        f"bus1={busname[int(src0.node)]}.{p0['bus_suffix']} pu={p0['pu']:.10g} "
        f"angle={p0['angle']:.10g} frequency={f0:.10g} r1={p0['r1']:.10g} "
        f"x1={p0['x1']:.10g}{seq0} spectrum={_FLAT_SPECTRUM_NAME}"
    )
    sources_out = {int(src0.id): {"name": "source", "pu_base": p0["pu"]}}
    kv_bases = {round(p0["basekv"], 6)}
    for extra in sources[1:]:
        pe = _source_params(extra, f0)
        name = f"src{extra.id}"
        _emit_vsource(
            dss,
            "New Vsource",
            name,
            f"{busname[int(extra.node)]}.{pe['bus_suffix']}",
            pe,
            f0,
        )
        sources_out[int(extra.id)] = {"name": name, "pu_base": pe["pu"]}
        kv_bases.add(round(pe["basekv"], 6))
        _logger.warning(
            "grid has multiple in-service Sources; Source %s exports as an "
            "additional Vsource element (only the first Source becomes the "
            "DSS Circuit's own slack).",
            extra.id,
        )

    for br in grid.branches:
        if not br.in_service:
            continue
        if isinstance(br, Line):
            _export_line(dss, br, busname, f0, mode)
        elif isinstance(br, Transformer):
            _export_transformer(dss, br, busname, f0)
            kv_bases.add(round(to_float(br.u_rated_from_v) / 1000.0, 6))
            kv_bases.add(round(to_float(br.u_rated_to_v) / 1000.0, 6))
        elif isinstance(br, Switch):
            _export_switch(dss, br, busname, f0, mode)
        elif isinstance(br, ShuntReactor):
            _export_shunt(
                dss,
                f"shr{br.id}",
                busname[br.from_node],
                br.from_phases,
                br.conductance_s,
                br.capacitance_f,
                node_by_id[br.from_node],
                f0,
            )
        elif isinstance(br, GenericBranch):
            _export_generic_branch(dss, br, busname, f0, mode)
        else:
            raise ConversionError(
                f"branch {br.id}: unsupported branch type {type(br).__name__} "
                "for the OpenDSS scenario exporter."
            )

    loads: dict = {}
    generators: dict = {}
    for a in grid.appliances:
        if not a.in_service or isinstance(a, Source):
            continue
        node = node_by_id[int(a.node)]
        if isinstance(a, Load):
            loads[int(a.id)] = _export_injection_appliance(
                dss, a, node, busname, name_prefix="lo", sign=1.0
            )
        elif isinstance(a, Generator):
            generators[int(a.id)] = _export_injection_appliance(
                dss, a, node, busname, name_prefix="ge", sign=-1.0
            )
        elif isinstance(a, Storage):
            generators[int(a.id)] = _export_injection_appliance(
                dss, a, node, busname, name_prefix="st", sign=-1.0
            )
        elif isinstance(a, ShuntAppliance):
            _export_shunt(
                dss,
                f"sha{a.id}",
                busname[int(a.node)],
                a.phases,
                a.conductance_s,
                a.capacitance_f,
                node,
                f0,
                connection=a.connection,
            )
        else:
            raise ConversionError(
                f"appliance {a.id}: unsupported appliance type "
                f"{type(a).__name__} for the OpenDSS scenario exporter."
            )

    kv_str = ", ".join(f"{k:.6g}" for k in sorted(kv_bases))
    dss.Text.Command(f"Set VoltageBases=[{kv_str}]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Set NeglectLoadY=" + ("Yes" if mode == "matched" else "No"))
    # OpenDSS's default snap-solve convergence tolerance (1e-4 relative on the mismatch)
    # is loose enough to be visible in a pgml-vs-OpenDSS comparison at fundamental (it
    # scales with system size/loading -- measured up to ~2e-6 relative on a 12-node
    # feeder at heavy load, a genuine numeric floor, not a modeling difference). Tighten
    # it in BOTH modes (a solver-precision setting, not a "matched"-only physics switch)
    # and raise the iteration cap so the tighter tolerance is actually reached.
    dss.Text.Command("Set Tolerance=0.0000000001")
    dss.Text.Command("Set MaxIterations=100")
    dss.Text.Command("Set Mode=Snap")
    with _scratch_datapath():
        dss.Text.Command("Solve")
    if not dss.Solution.Converged():
        raise ConversionError(
            "the exported OpenDSS circuit did not converge on its initial "
            "nominal-operating-point solve; check the exported topology/ratings."
        )

    node_order = list(dss.Circuit.YNodeOrder())
    rowmap = _build_rowmap(node_order, busname, index)

    return ExportedCircuit(
        grid=grid,
        mode=mode,
        busname=busname,
        node_order=node_order,
        rowmap=rowmap,
        index=index,
        loads=loads,
        generators=generators,
        sources=sources_out,
    )


# ---------------------------------------------------------------------------
# Scenario translation + batch run (public)
# ---------------------------------------------------------------------------
def _apply_pq(
    dss, exp: _ApplianceExport, entry: dict, b: int, t: Optional[int] = None
) -> None:
    """Edit ``exp``'s DSS element(s) to the scenario's operating point (``exp.sign``
    negates a Generator/Storage's generation-positive P/Q into DSS's consumption-positive
    ``Load`` convention -- see ``_ApplianceExport``'s docstring).

    ``t`` selects the STEP for a node-coherent batch. An operating-point entry is
    ``[B]`` (drawn once per scenario, constant across steps -- the plain/no-profile
    case) or ``[B, T]`` (a :class:`~pgml.scenarios.LoadProfileConfig`-lifted, per-step
    fundamental); :func:`_scalar_at` slices either shape correctly from the SAME ``t``
    argument (a ``[B]`` entry ignores ``t``, so the constant-across-steps case is
    unaffected).
    """
    n = len(exp.phases)
    sign = exp.sign
    p_pp_key = entry.get("p_per_phase_w")
    q_pp_key = entry.get("q_per_phase_var")
    if p_pp_key is not None or q_pp_key is not None:
        if exp.kind != "split":
            raise ConversionError(
                f"{exp.dss_class} exported as a single DELTA element received "
                "a per-phase operating-point override, which has no per-leg "
                "edit point on that element."
            )
        for k, ph in enumerate(exp.phases):
            name = exp.elements[ph]
            p_val = (
                _scalar_at(p_pp_key[k], b, t)
                if p_pp_key is not None
                else exp.p_nom_pp[k]
            )
            q_val = (
                _scalar_at(q_pp_key[k], b, t)
                if q_pp_key is not None
                else exp.q_nom_pp[k]
            )
            kw = sign * p_val / 1000.0
            kvar = sign * q_val / 1000.0
            dss.Text.Command(
                f"Edit {exp.dss_class}.{name} kW={kw:.10g} kvar={kvar:.10g}"
            )
        return
    p_w = _scalar_at(entry.get("p_w", sum(exp.p_nom_pp)), b, t)
    q_var = _scalar_at(entry.get("q_var", sum(exp.q_nom_pp)), b, t)
    if exp.kind == "whole":
        name = exp.elements[None]
        dss.Text.Command(
            f"Edit {exp.dss_class}.{name} kW={sign * p_w / 1000.0:.10g} "
            f"kvar={sign * q_var / 1000.0:.10g}"
        )
    else:
        for ph in exp.phases:
            name = exp.elements[ph]
            dss.Text.Command(
                f"Edit {exp.dss_class}.{name} kW={sign * p_w / n / 1000.0:.10g} "
                f"kvar={sign * q_var / n / 1000.0:.10g}"
            )


def _apply_operating_point(
    dss,
    circuit: ExportedCircuit,
    sampled: SampledScenarios,
    b: int,
    t: Optional[int] = None,
) -> None:
    """Edit every targeted element to scenario ``b`` (+ step ``t``)'s operating point.

    Called once per STEP (not once per scenario) so a ``[B, T]`` per-step operating
    point (:class:`~pgml.scenarios.LoadProfileConfig`) is applied correctly; a plain
    ``[B]`` entry is unaffected (``_scalar_at`` ignores ``t`` for a 1-D tensor), so this
    is a strict superset of the old once-per-scenario behaviour, not a change to it.
    """
    for aid, entry in sampled.operating_point.items():
        if aid in circuit.sources:
            src = circuit.sources[aid]
            scale = _scalar_at(entry.get("u_ref_scale", 1.0), b, t)
            pu = src["pu_base"] * scale
            dss.Text.Command(f"Edit Vsource.{src['name']} pu={pu:.10g}")
            continue
        exp = circuit.loads.get(aid) or circuit.generators.get(aid)
        if exp is None:
            _logger.warning(
                "scenario operating point targets appliance %d, which the "
                "scenario oracle did not export -- ignored.",
                aid,
            )
            continue
        _apply_pq(dss, exp, entry, b, t)


def _attach_spectra(
    dss, circuit: ExportedCircuit, sampled: SampledScenarios, orders: list
) -> None:
    """Create one native DSS Spectrum PER DEVICE that carries a harmonic injection.

    The harmonic order LIST is fixed for the whole run (the union of every requested order
    plus order 1, the fundamental reference); only ``%mag``/``angle`` are re-edited per
    scenario/step by :func:`_apply_harmonic_spectra`.
    """
    dev_orders = sorted(set(int(h) for h in orders) | {1})
    for aid in sampled.harmonic_injection:
        exp = circuit.loads.get(aid) or circuit.generators.get(aid)
        if exp is None:
            continue
        name = f"spec{aid}"
        h_str = " ".join(str(h) for h in dev_orders)
        mag0 = " ".join("100" if h == 1 else "0" for h in dev_orders)
        ang0 = " ".join("0" for h in dev_orders)
        dss.Text.Command(
            f"New Spectrum.{name} NumHarm={len(dev_orders)} harmonic=[{h_str}] "
            f"%mag=[{mag0}] angle=[{ang0}]"
        )
        names = (
            [exp.elements[None]] if exp.kind == "whole" else list(exp.elements.values())
        )
        for el_name in names:
            dss.Text.Command(f"Edit {exp.dss_class}.{el_name} spectrum={name}")
        circuit.spectra[aid] = {"name": name, "orders": dev_orders}


def _apply_harmonic_spectra(
    dss, circuit: ExportedCircuit, sampled: SampledScenarios, b: int, t: Optional[int]
) -> None:
    for aid, spec in circuit.spectra.items():
        per_order = sampled.harmonic_injection.get(aid, {})
        mags, angs = [], []
        for h in spec["orders"]:
            mag, ang = per_order.get(h, (1.0 if h == 1 else 0.0, 0.0))
            mags.append(_scalar_at(mag, b, t) * 100.0)
            angs.append(_scalar_at(ang, b, t))
        mag_str = " ".join(f"{m:.10g}" for m in mags)
        ang_str = " ".join(f"{a:.10g}" for a in angs)
        dss.Text.Command(
            f"Edit Spectrum.{spec['name']} %mag=[{mag_str}] angle=[{ang_str}]"
        )


def run_opendss_scenarios(
    grid: Grid,
    sampled: SampledScenarios,
    *,
    harmonic_orders: Sequence[int],
    mode: str = "matched",
    dtype: torch.dtype = torch.complex128,
) -> ScenarioResult:
    """Run a realized :class:`~pgml.scenarios.SampledScenarios` batch through a live OpenDSS
    engine -> a :class:`~pgml.scenarios.ScenarioResult` aligned to
    :func:`pgml.assembly.node_phase_index` rows (same layout ``run_scenarios`` returns).

    Exports the circuit ONCE (:func:`export_grid_to_opendss`), attaches one native DSS
    ``Spectrum`` per device with a harmonic injection, then per scenario (and per STEP for a
    node-coherent batch, detected via ``sampled.samples["time_s"]``): edits every targeted
    ``Load``/``Generator``/``Vsource`` to the scenario's operating point (``Edit ... kW=...
    kvar=...`` / per-phase / ``pu=...``), edits every device's ``Spectrum`` ``%mag``/``angle``
    to the realized harmonic injection, ``Solve`` (snap — the nonlinear fundamental; order 1
    of the result), then ``Set Mode=Harmonics`` + ``Set Harmonic=<h>`` + ``Solve`` per
    remaining requested order, extracting ``Circuit.AllBusVolts()`` after each solve.

    Parameters
    ----------
    grid:
        The grid ``sampled`` was drawn against.
    sampled:
        A realized batch from :func:`pgml.scenarios.sample` /
        :func:`pgml.scenarios.sample_coherent_spectra` (accepted as-is; never re-sampled).
    harmonic_orders:
        Orders to solve; order 1 is always included even if omitted.
    mode:
        ``"matched"`` or ``"default"`` — see :func:`export_grid_to_opendss`.
    dtype:
        Complex dtype of the returned ``ScenarioResult.v`` (this oracle itself is a plain
        double-precision numpy computation; ``dtype`` only controls the final cast, matching
        ``run_scenarios``'s own signature for direct comparison).

    Returns
    -------
    pgml.scenarios.ScenarioResult
        ``v`` is ``[B, H, N]`` for a snapshot batch or ``[B, T, H, N]`` for a node-coherent
        one; ``converged`` is ``True`` iff EVERY scenario/step/order converged.
    """
    import opendssdirect as dss

    orders = sorted(set(int(h) for h in harmonic_orders) | {1})
    f0 = float(grid.base_frequency_hz)
    index = node_phase_index(grid)
    n = index.size
    b = int(sampled.n_samples)
    is_coherent = "time_s" in sampled.samples
    t_steps = int(sampled.samples["time_s"].shape[-1]) if is_coherent else 1

    circuit = export_grid_to_opendss(grid, mode=mode)
    _attach_spectra(dss, circuit, sampled, orders)

    v_out = np.zeros((b, t_steps, len(orders), n), dtype=complex)
    converged = True
    failed: list = []
    with _scratch_datapath():
        for bi in range(b):
            for ti in range(t_steps):
                _apply_operating_point(
                    dss, circuit, sampled, bi, ti if is_coherent else None
                )
                _apply_harmonic_spectra(
                    dss, circuit, sampled, bi, ti if is_coherent else None
                )
                dss.Text.Command("Set Mode=Snap")
                dss.Text.Command("Solve")
                if not dss.Solution.Converged():
                    converged = False
                    failed.append(bi)
                    _logger.error(
                        "OpenDSS scenario oracle: scenario %d step %d did not "
                        "converge at the fundamental (order 1).",
                        bi,
                        ti,
                    )
                v_out[bi, ti, 0] = _extract_voltages(dss, circuit.rowmap, n)
                if len(orders) > 1:
                    dss.Text.Command("Set Mode=Harmonics")
                    for k, h in enumerate(orders[1:], start=1):
                        dss.Text.Command(f"Set Harmonic={h}")
                        dss.Text.Command("Solve")
                        if not dss.Solution.Converged():
                            converged = False
                            failed.append(bi)
                            _logger.error(
                                "OpenDSS scenario oracle: scenario %d step %d "
                                "order %d did not converge.",
                                bi,
                                ti,
                                h,
                            )
                        v_out[bi, ti, k] = _extract_voltages(dss, circuit.rowmap, n)

    v_tensor = torch.tensor(v_out, dtype=dtype)
    if not is_coherent:
        v_tensor = v_tensor[:, 0]  # [B, H, N]
    freqs = torch.tensor([h * f0 for h in orders], dtype=torch.float64)
    return ScenarioResult(
        v=v_tensor,
        index=index,
        sampled=sampled,
        frequencies_hz=freqs,
        converged=converged,
        failed_states=tuple(sorted(set(failed))),
    )


# ---------------------------------------------------------------------------
# Dataset persistence with provenance
# ---------------------------------------------------------------------------
def write_opendss_dataset(
    grid: Grid,
    sampled: SampledScenarios,
    path,
    *,
    harmonic_orders: Sequence[int],
    mode: str = "matched",
    layout: str = "wide",
    dtype: torch.dtype = torch.complex128,
) -> Path:
    """Solve ``sampled`` with the live OpenDSS oracle and persist it as a pgml scenario
    dataset (:func:`pgml.scenarios.write_dataset`), then stamp the ``meta.json`` sidecar with
    engine provenance so it is never mistaken for a pgml-generated dataset.

    ``meta.json`` gains: ``engine="opendss"``, ``oracle_mode`` (``"matched"``/``"default"``),
    ``opendssdirect_version``, ``opendss_engine_version`` (``Basic.Version()``'s full string —
    the DSS C-API library + underlying OpenDSS SVN revision). The dataset is otherwise
    byte-for-byte the same layout :func:`pgml.scenarios.read_dataset` and every downstream
    data consumer already expects, so it plugs into training/evaluation unchanged.
    """
    from pgml.scenarios import write_dataset

    result = run_opendss_scenarios(
        grid, sampled, harmonic_orders=harmonic_orders, mode=mode, dtype=dtype
    )
    out = write_dataset(result, path, layout=layout)
    _stamp_provenance(out, mode=mode)
    return out


def _stamp_provenance(path, *, mode: str) -> None:
    import opendssdirect as dss

    meta_path = Path(path) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["engine"] = "opendss"
    meta["oracle_mode"] = mode
    meta["opendssdirect_version"] = getattr(dss, "__version__", None)
    meta["opendss_engine_version"] = dss.Basic.Version()
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


# ---------------------------------------------------------------------------
# Numeric cross-validation report
# ---------------------------------------------------------------------------
def _per_order_error_report(
    v_ref: np.ndarray, v_pgml: np.ndarray, orders: list
) -> dict:
    """Per-order abs + relative-to-order-RMS error stats (mean/p95/max) over the batch."""
    h_axis = -2
    err = np.abs(v_ref - v_pgml)
    per_order = {}
    for k, h in enumerate(orders):
        e = np.take(err, k, axis=h_axis).reshape(-1)
        ref_h = np.take(np.abs(v_ref), k, axis=h_axis).reshape(-1)
        rms_h = float(np.sqrt(np.mean(ref_h**2))) if ref_h.size else 0.0
        rel = e / rms_h if rms_h > 0.0 else np.full_like(e, np.nan)
        per_order[int(h)] = {
            "abs_mean": float(np.mean(e)),
            "abs_p95": float(np.percentile(e, 95)),
            "abs_max": float(np.max(e)),
            "rel_mean": float(np.nanmean(rel)),
            "rel_p95": float(np.nanpercentile(rel, 95)),
            "rel_max": float(np.nanmax(rel)),
            "ref_rms_v": rms_h,
        }
    return {"orders": [int(h) for h in orders], "per_order": per_order}


def _write_report(report: dict, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "opendss_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (out_dir / "opendss_comparison.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "order",
                "abs_mean",
                "abs_p95",
                "abs_max",
                "rel_mean",
                "rel_p95",
                "rel_max",
                "ref_rms_v",
            ]
        )
        for h, s in report["per_order"].items():
            w.writerow(
                [
                    h,
                    s["abs_mean"],
                    s["abs_p95"],
                    s["abs_max"],
                    s["rel_mean"],
                    s["rel_p95"],
                    s["rel_max"],
                    s["ref_rms_v"],
                ]
            )


def compare_to_pgml(
    grid: Grid,
    sampled: SampledScenarios,
    *,
    harmonic_orders: Sequence[int],
    mode: str = "matched",
    out_dir=None,
    slack: str = "norton",
    symmetry: Optional[str] = None,
    dtype: torch.dtype = torch.complex128,
) -> dict:
    """Solve the SAME ``sampled`` batch with both engines and report the per-order agreement.

    Runs :func:`run_opendss_scenarios` (the ground truth) and
    :func:`pgml.scenarios.run_scenarios` (``calculation="harmonic"``) on the identical
    :class:`~pgml.scenarios.SampledScenarios`, then reports per order (over the whole batch):
    absolute ``|V_opendss - V_pgml|`` and that error RELATIVE TO the order's RMS voltage
    (``sqrt(mean(|V_opendss|^2))`` over the batch), each as mean/p95/max — the numeric
    cross-validation deliverable. Optionally writes ``opendss_comparison.json``/``.csv`` to
    ``out_dir``.

    ``slack`` defaults to ``"norton"``, NOT pgml's own library default (``"ideal"``): an
    OpenDSS ``Vsource`` always behaves as a finite-impedance Thevenin source (its ``R1``/
    ``X1`` sag the bus voltage under load), whereas pgml's ``slack="ideal"`` pins the bus
    voltage exactly at ``u_ref_v`` and silently IGNORES the ``Source``'s own impedance (see
    ``src/pgml/convert/opendss/CONTEXT.md``'s "Vsource impedance under the default ideal
    slack"). For a grid whose ``Source`` carries a non-negligible impedance, comparing
    against pgml's default ``"ideal"`` solve would report a genuine several-hundred-pu-ppm
    "error" that is actually just two DIFFERENT slack models, not a numeric discrepancy —
    measured empirically at ~1.3e-3 relative on ``pgml.grids.synthetic_feeder``'s ~0.16+j5e-3
    Ohm source before this fix, ~3e-10 after switching to ``slack="norton"``.

    Returns
    -------
    dict
        ``{"orders": [...], "per_order": {order: {abs_mean, abs_p95, abs_max, rel_mean,
        rel_p95, rel_max, ref_rms_v}}, "opendss_converged": bool, "pgml_converged": bool}``.
    """
    from pgml.scenarios import run_scenarios

    orders = sorted(set(int(h) for h in harmonic_orders) | {1})
    oracle = run_opendss_scenarios(
        grid, sampled, harmonic_orders=orders, mode=mode, dtype=dtype
    )
    pgml_result = run_scenarios(
        grid,
        sampled,
        calculation="harmonic",
        harmonic_orders=orders,
        slack=slack,
        symmetry=symmetry,
        dtype=dtype,
    )
    v_ref = oracle.v.detach().cpu().numpy()
    v_pgml = pgml_result.v.detach().cpu().numpy()
    if v_ref.shape != v_pgml.shape:
        raise InputError(
            f"OpenDSS oracle result shape {v_ref.shape} != pgml result shape "
            f"{v_pgml.shape} -- both must solve the SAME SampledScenarios/orders."
        )
    report = _per_order_error_report(v_ref, v_pgml, orders)
    report["opendss_converged"] = bool(oracle.converged)
    report["pgml_converged"] = bool(pgml_result.converged)
    if out_dir is not None:
        _write_report(report, out_dir)
    return report


__all__ = [
    "ExportedCircuit",
    "export_grid_to_opendss",
    "run_opendss_scenarios",
    "write_opendss_dataset",
    "compare_to_pgml",
]
