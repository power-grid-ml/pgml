"""Export a :class:`pgml.schemas.Grid` as a live OpenDSS circuit with a report.

The circuit itself is written by
:func:`pgml.evaluation.oracles.export_grid_to_opendss`, the full-circuit exporter
the harmonic oracle tests are pinned against (native ``Vsource``, ``Line`` matrices,
``Transformer`` windings with their clock, ``Load`` with the matched harmonic device
shunt, ``Capacitor``/``Reactor``). This module adds what a conformance check needs
around it:

* in ``mode="matched"`` every lumped line receives the ``Rg``/``Xg``/``rho`` that
  reproduce its pgml earth-return law, so OpenDSS applies the same zero-sequence
  frequency correction instead of none (the oracle writes ``Rg=Xg=0``, which is
  pgml's ``naive`` and ``positive_sequence`` line model but not the
  ``sequence_aware`` one);
* an inverter control law or a voltage regulation on an exported appliance is
  refused unless ``allow_approximation=True``, because the oracle exports the
  nameplate P/Q; the reduction is then recorded;
* a :class:`~pgml.convert.ConversionReport` names every reduction and every pgml
  model feature OpenDSS cannot represent exactly (the skin-effect multiplier, the
  guarded reactance law), with the pgml setting that closes each one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from pgml.convert._export import (
    UnsupportedGridError,
    detached,
    record_reduction,
    scalar,
)
from pgml.convert._model_differences import OPENDSS, finalize_report
from pgml.convert._report import ConversionReport, ModelMatch
from pgml.errors import ConversionError
from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    Line,
    Load,
    Source,
    Storage,
    Switch,
    Transformer,
)

#: series resistance standing in for an ideal source (OpenDSS refuses Z1 = 0)
IDEAL_SOURCE_R_OHM = 1e-6


@dataclass
class OpenDSSExport:
    """A live OpenDSS circuit built from a ``Grid`` plus its conversion report.

    ``circuit`` is the :class:`~pgml.evaluation.oracles.ExportedCircuit` (bus names,
    row map aligned to :func:`pgml.assembly.node_phase_index`, element maps) that
    :func:`pgml.evaluation.oracles.run_opendss_scenarios` consumes. The circuit is
    the one currently loaded in ``opendssdirect``.
    """

    circuit: Any
    bus_of_node: dict[int, str]
    line_of_branch: dict[int, str]
    element_of_appliance: dict[int, Any]
    mode: str
    reductions: list[str] = field(default_factory=list)
    report: ConversionReport = field(
        default_factory=lambda: ConversionReport(tool=OPENDSS, direction="export")
    )


def _earth_parameters(line: Line, f0: float) -> tuple[float, float, float, bool]:
    """``(Rg, Xg, rho)`` per metre reproducing one line's earth law, and its clamp.

    OpenDSS corrects every entry by ``Rg*(h-1)`` and ``-0.5*KXg*ln h`` with
    ``KXg = Xg/ln(658.5*sqrt(rho/f0))``, the zero-sequence law of the
    ``sequence_aware`` model with ``Rg = resistance_coeff*f0`` and
    ``KXg = reactance_coeff*f0``. A ``linear`` reactance law is ``Xg = 0``; the
    other lumped models carry no earth term at all.
    """
    from pgml import defaults as _d

    rho = float(_d.get("line.earth_return.resistivity_ohm_m"))
    if getattr(line, "harmonic_line_model", None) != "sequence_aware":
        return 0.0, 0.0, rho, False
    er = getattr(line, "earth_return", None)

    def pick(name: str, key: str) -> Any:
        value = getattr(er, name, None)
        return _d.get(key) if value is None else value

    rc = float(
        pick(
            "resistance_coeff_ohm_per_m_per_hz",
            "line.earth_return.resistance_coeff_ohm_per_m_per_hz",
        )
    )
    kx = float(
        pick(
            "reactance_coeff_ohm_per_m_per_hz",
            "line.earth_return.reactance_coeff_ohm_per_m_per_hz",
        )
    )
    law = pick("x0_frequency", "line.earth_return.x0_frequency")
    clamp = bool(pick("x0_nonnegative", "line.earth_return.x0_nonnegative"))
    rg = rc * f0
    xg = (
        kx * f0 * math.log(658.5 * math.sqrt(rho / f0))
        if law == "carson_sublinear"
        else 0.0
    )
    return rg, xg, rho, clamp and law == "carson_sublinear"


def from_grid(
    grid: Grid,
    *,
    mode: str = "matched",
    load_shunt: Optional[str] = None,
    allow_approximation: bool = False,
    circuit_name: str = "pgml_export",
) -> OpenDSSExport:
    """Build the OpenDSS circuit of ``grid`` in the live ``opendssdirect`` engine.

    Parameters
    ----------
    grid:
        A materialised grid (no unresolved ``type_ref``).
    mode:
        ``"matched"`` (default) configures OpenDSS to the model pgml solves: the
        harmonic device shunt named by ``load_shunt``, wide load voltage bands, a
        tight solve tolerance, and each line's own earth-return parameters.
        ``"default"`` leaves OpenDSS's defaults; the report then lists what differs.
    load_shunt:
        Harmonic device shunt for ``"matched"`` mode, as in
        :func:`pgml.solver.solve_harmonic_flow` (``None`` = the modeling default).
    allow_approximation:
        Permit exporting an appliance with an inverter control law or a voltage
        regulation as its nameplate P/Q, recording the reduction. The default
        raises :class:`~pgml.convert._export.UnsupportedGridError`.
    circuit_name:
        Name of the DSS ``Circuit``.

    A grid with generation devices needs ``load_shunt="none"`` in matched mode
    unless ``appliance.harmonic_shunt.generation_model`` is ``load_style``, because
    OpenDSS derives a harmonic shunt from every Load, generation included, and the
    circuit exporter refuses to match a device model pgml does not carry.

    Returns
    -------
    OpenDSSExport
        The live circuit, element maps and the conversion report.

    Raises
    ------
    UnsupportedGridError
        For an appliance whose control needs the approximation, or any grid feature
        the underlying circuit exporter refuses (conductor geometry, zigzag
        windings, an unbalanced source; see
        :mod:`pgml.evaluation.oracles.opendss_scenario_oracle`).
    """
    from pgml.evaluation.oracles.opendss_scenario_oracle import export_grid_to_opendss

    if mode not in ("matched", "default"):
        raise UnsupportedGridError(f"mode must be 'matched' or 'default', got {mode!r}")
    out = OpenDSSExport(
        circuit=None,
        bus_of_node={},
        line_of_branch={},
        element_of_appliance={},
        mode=mode,
    )
    out.report.options.update(
        mode=mode, load_shunt=load_shunt, allow_approximation=allow_approximation
    )

    # -- reductions the circuit exporter applies silently ------------------ #
    for ap in grid.appliances:
        if not ap.in_service or not isinstance(ap, (Load, Generator, Storage)):
            continue
        if getattr(ap, "harmonic_impedance", None) is not None:
            continue  # native DER export refuses controls itself
        pending = []
        if getattr(ap, "control", None) is not None:
            pending.append(
                (
                    "approx.inverter_control",
                    "inverter control replaced by nameplate P/Q",
                )
            )
        if getattr(ap, "voltage_regulation", None) is not None:
            pending.append(
                (
                    "approx.voltage_regulation",
                    "voltage regulation replaced by nameplate P/Q",
                )
            )
        if pending and not allow_approximation:
            raise UnsupportedGridError(
                f"appliance {ap.id}: {'; '.join(t for _, t in pending)}; pass "
                "allow_approximation=True to export the nameplate and record it"
            )
        for key, text in pending:
            record_reduction(
                out,
                key,
                f"appliance {ap.id}: {text}",
                element_type=type(ap).__name__,
                element_id=ap.id,
            )

    for br in grid.branches:
        if not isinstance(br, Switch) or not br.in_service:
            continue
        if (
            scalar(br.shunt_capacitance_f) == 0.0
            and scalar(br.shunt_conductance_s) == 0.0
        ):
            continue
        if not allow_approximation:
            raise UnsupportedGridError(
                f"switch {br.id}: the OpenDSS switch element carries no shunt terms; "
                "pass allow_approximation=True to drop and record them"
            )
        record_reduction(
            out,
            "approx.switch.dropped_terms",
            f"switch {br.id}: dropped shunt conductance/capacitance",
            element_type="Switch",
            element_id=br.id,
        )

    # OpenDSS refuses a Vsource with Z1 = 0; an ideal pgml Source gets a resistance
    # far below any branch impedance and is recorded.
    exported_grid = grid
    ideal = [
        ap.id
        for ap in grid.appliances
        if isinstance(ap, Source)
        and ap.in_service
        and not np.any(np.asarray(detached(ap.resistance_ohm), dtype=float) != 0.0)
        and not np.any(np.asarray(detached(ap.inductance_h), dtype=float) != 0.0)
    ]
    if ideal:
        exported_grid = grid.model_copy(deep=True)
        for ap in exported_grid.appliances:
            if ap.id in ideal:
                n = len(ap.phases)
                ap.resistance_ohm = [
                    [IDEAL_SOURCE_R_OHM if i == j else 0.0 for j in range(n)]
                    for i in range(n)
                ]
            record_reduction(
                out,
                "approx.source.ideal_as_finite",
                f"source {ap.id}: ideal voltage boundary represented by "
                f"R1 = {IDEAL_SOURCE_R_OHM:g} Ohm",
                element_type="Source",
                element_id=ap.id,
                values={"r_ohm": IDEAL_SOURCE_R_OHM},
            ) if ap.id in ideal else None

    try:
        circuit = export_grid_to_opendss(
            exported_grid, mode=mode, load_shunt=load_shunt, circuit_name=circuit_name
        )
    except ConversionError as exc:
        raise UnsupportedGridError(str(exc)) from exc
    out.circuit = circuit
    out.bus_of_node = dict(circuit.busname)
    out.line_of_branch = {
        br.id: f"l{br.id}"
        for br in grid.branches
        if isinstance(br, Line) and br.in_service
    }
    out.element_of_appliance = {
        **circuit.loads,
        **circuit.generators,
        **circuit.sources,
    }

    import opendssdirect as dss

    # -- off-nominal tap: the ratio scales the from-side winding voltage ----- #
    for br in grid.branches:
        if not isinstance(br, Transformer) or not br.in_service:
            continue
        ratio = float(
            np.asarray(detached(br.tap.ratio_magnitude), dtype=float).reshape(-1)[0]
        )
        if ratio != 1.0:
            dss.Text.Command(f"Edit Transformer.t{br.id} wdg=1 tap={ratio:.10g}")

    # -- earth-return law of every lumped line -------------------------------- #
    f0 = float(grid.base_frequency_hz)
    if mode == "matched":
        clamped: list[int] = []
        for br in grid.branches:
            if not isinstance(br, Line) or not br.in_service:
                continue
            rg, xg, rho, clamp = _earth_parameters(br, f0)
            if clamp:
                clamped.append(br.id)
            if rg == 0.0 and xg == 0.0:
                continue
            dss.Text.Command(
                f"Edit Line.l{br.id} Rg={rg:.10g} Xg={xg:.10g} rho={rho:.10g}"
            )
        if clamped:
            out.report.model_difference(
                "line.earth_return_clamp",
                "These lines clamp the sub-linear zero-sequence reactance at zero "
                "(x0_nonnegative); OpenDSS applies the unguarded law and lets X0 "
                "turn negative. Identical until the clamp binds.",
                element_type="Line",
                ids=clamped,
                affects=("harmonic",),
                source_model="unguarded Carson reactance law",
                pgml_model="x0_nonnegative=True",
                match=ModelMatch(
                    preset="opendss",
                    settings={"line.earth_return.x0_nonnegative": False},
                ),
            )
    # the circuit exporter solved before these edits; solve the edited circuit
    dss.Text.Command("Solve")
    if not dss.Solution.Converged():
        raise UnsupportedGridError(
            "the exported OpenDSS circuit did not converge after applying the tap "
            "and earth-return parameters"
        )
    finalize_report(out.report, grid)
    # In matched mode the exported loads carry pgml's resolved device shunt and the
    # lines pgml's earth parameters, so those catalogue entries are closed here.
    if mode == "matched":
        closed = {"model.load.harmonic_shunt", "model.line.earth_return_law"}
        from dataclasses import replace

        out.report.entries = [
            replace(e, matched=True) if e.key in closed else e
            for e in out.report.entries
        ]
    return out


__all__ = ["OpenDSSExport", "UnsupportedGridError", "from_grid"]
