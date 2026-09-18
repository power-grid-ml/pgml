"""Default-model differences between pgml and each reference tool.

Even when every element of a network converts, pgml and the other tool may solve
different equations for it by default. This module derives those differences from
what a grid actually contains and adds one ``model.*`` entry per difference to a
:class:`~pgml.convert._report.ConversionReport`, each with the pgml setting, call
argument or reference preset that reproduces the other tool's model, or with no
match when pgml cannot reproduce it.

The same catalogue serves both directions: after an import it tells the caller why
a pgml solve of the converted grid can differ from the tool's own solve, and after
an export why the tool's solve of the exported case can differ from pgml's.

Values of pgml's side are read from the active modeling defaults, so an entry is
marked ``matched`` when the conversion runs inside the matching
:func:`pgml.defaults.use_preset` context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pgml.convert._report import ConversionReport, ModelMatch
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

PANDAPOWER = "pandapower"
OPENDSS = "opendss"
PGM = "power-grid-model"
TOOLS = (PANDAPOWER, OPENDSS, PGM)

#: tools that solve harmonic orders
_HARMONIC_TOOLS = (OPENDSS,)


def _nonzero(value: Any) -> bool:
    if value is None:
        return False
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    try:
        return bool(np.any(np.asarray(value, dtype=float) != 0.0))
    except (TypeError, ValueError):
        return True


@dataclass
class GridFeatures:
    """What a grid contains, as far as a model difference depends on it."""

    lines: list = field(default_factory=list)
    line_models: dict = field(default_factory=dict)
    multiphase_lines: list = field(default_factory=list)
    geometry_lines: list = field(default_factory=list)
    skin_lines: list = field(default_factory=list)
    transformers: list = field(default_factory=list)
    magnetizing_transformers: list = field(default_factory=list)
    tapped_transformers: list = field(default_factory=list)
    switches: list = field(default_factory=list)
    impedance_sources: list = field(default_factory=list)
    zero_sequence_sources: list = field(default_factory=list)
    limited_generators: list = field(default_factory=list)
    loads: list = field(default_factory=list)
    injections: list = field(default_factory=list)
    spectrum_appliances: list = field(default_factory=list)


def grid_features(grid: Grid) -> GridFeatures:
    """Collect the element groups the model-difference catalogue keys on."""
    f = GridFeatures()
    for br in grid.branches:
        if not br.in_service:
            continue
        if isinstance(br, Line):
            f.lines.append(br.id)
            model = br.harmonic_line_model or "unresolved"
            if br.conductor_geometry is not None:
                model = "geometry"
                f.geometry_lines.append(br.id)
            f.line_models.setdefault(model, []).append(br.id)
            skin = br.harmonic_skin_effect
            if skin is None:
                skin = bool(_default("line.harmonic_model.skin_effect"))
            if skin and model in ("sequence_aware", "positive_sequence"):
                f.skin_lines.append(br.id)
            if len([p for p in br.from_phases if p.value != "n"]) >= 3:
                f.multiphase_lines.append(br.id)
        elif isinstance(br, Transformer):
            f.transformers.append(br.id)
            if _nonzero(br.magnetizing_conductance_s) or (
                br.magnetizing_inductance_h is not None
            ):
                f.magnetizing_transformers.append(br.id)
            tap = br.tap
            if tap is not None and _nonzero(
                np.asarray(_host(tap.ratio_magnitude), dtype=float) - 1.0
            ):
                f.tapped_transformers.append(br.id)
        elif isinstance(br, Switch):
            f.switches.append(br.id)
    for ap in grid.appliances:
        if not ap.in_service:
            continue
        if isinstance(ap, Source):
            if _nonzero(ap.resistance_ohm) or _nonzero(ap.inductance_h):
                f.impedance_sources.append(ap.id)
                r = np.asarray(_host(ap.resistance_ohm), dtype=float)
                ell = np.asarray(_host(ap.inductance_h), dtype=float)
                for m in (r, ell):
                    if m.ndim == 2 and m.shape[0] > 1:
                        off = m - np.diag(np.diag(m))
                        if np.any(off != 0.0):
                            f.zero_sequence_sources.append(ap.id)
                            break
            continue
        if isinstance(ap, (Load, Generator, Storage)):
            f.injections.append(ap.id)
            if isinstance(ap, Load):
                f.loads.append(ap.id)
            if getattr(ap, "spectrum", None) is not None or getattr(
                ap, "spectrum_per_phase", None
            ):
                f.spectrum_appliances.append(ap.id)
            reg = getattr(ap, "voltage_regulation", None)
            if reg is not None and (
                reg.q_min_var is not None or reg.q_max_var is not None
            ):
                f.limited_generators.append(ap.id)
    return f


def _host(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    return detach().cpu().numpy() if callable(detach) else value


def _default(name: str) -> Any:
    from pgml import defaults

    return defaults.get(name)


def add_model_differences(
    report: ConversionReport, grid: Grid, *, tool: str | None = None
) -> ConversionReport:
    """Append every default-model difference that applies to ``grid``.

    ``tool`` defaults to ``report.tool``. Only differences whose elements are present
    in the grid are reported, with the pgml ids of those elements.
    """
    tool = tool or report.tool
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool!r}; expected one of {TOOLS}")
    f = grid_features(grid)
    _transformer_differences(report, f, tool)
    _source_differences(report, f, tool)
    _generator_differences(report, f, tool)
    _line_differences(
        report, f, tool, [br for br in grid.branches if isinstance(br, Line)]
    )
    _harmonic_device_differences(report, f, tool)
    return report


# --------------------------------------------------------------------------- #
# Transformers
# --------------------------------------------------------------------------- #
def _transformer_differences(report, f: GridFeatures, tool: str) -> None:
    if not f.magnetizing_transformers:
        return
    placement = _default("transformer.magnetizing_placement")
    ids = f.magnetizing_transformers
    if tool == OPENDSS:
        report.model_difference(
            "transformer.magnetizing_placement",
            "OpenDSS connects the whole magnetizing branch to the terminal of the "
            f"last winding; pgml places it as '{placement}'. The magnetizing "
            "current then sees a different share of the leakage impedance.",
            element_type="Transformer",
            ids=ids,
            source_model="whole branch at the last (to) winding terminal",
            pgml_model=f"transformer.magnetizing_placement = {placement}",
            match=ModelMatch(
                preset="opendss",
                settings={"transformer.magnetizing_placement": "to_terminal"},
            ),
        )
    elif tool == PGM:
        tapped = sorted(set(ids) & set(f.tapped_transformers))
        if tapped:
            report.model_difference(
                "transformer.magnetizing_tap_reflection",
                "power-grid-model reflects the from-side half of the magnetizing "
                "branch through the off-nominal tap ratio; pgml's 'split' refers "
                "each half through the rated ratio only. The two differ by the tap "
                "deviation squared on that half, visible when the from terminal is "
                "not held by an ideal slack (measured 5e-7 pu at tap 1.025, 0.5 % "
                "magnetizing current, behind a source impedance).",
                element_type="Transformer",
                ids=tapped,
                affects=("fundamental", "unbalanced"),
                source_model="from-side half through the tap",
                pgml_model="from-side half through the rated ratio",
                match=None,
            )
        report.model_difference(
            "transformer.magnetizing_placement",
            "power-grid-model splits the magnetizing branch in halves onto both "
            f"terminals of its pi equivalent; pgml places it as '{placement}'.",
            element_type="Transformer",
            ids=ids,
            source_model="pi equivalent, half the branch on each terminal",
            pgml_model=f"transformer.magnetizing_placement = {placement}",
            match=ModelMatch(
                preset="power-grid-model",
                settings={"transformer.magnetizing_placement": "split"},
            ),
        )
    else:
        report.model_difference(
            "transformer.magnetizing_placement",
            "pandapower's default transformer is a T equivalent (trafo_model='t') "
            "with the magnetizing branch at the star point between two half "
            f"leakage impedances; pgml places the branch as '{placement}' and has "
            "no T equivalent. pandapower's trafo_model='pi' equals pgml's 'split'.",
            element_type="Transformer",
            ids=ids,
            source_model="T equivalent (runpp default trafo_model='t')",
            pgml_model=f"transformer.magnetizing_placement = {placement}",
            match=ModelMatch(
                preset="pandapower",
                settings={"transformer.magnetizing_placement": "split"},
                reference="runpp(net, trafo_model='pi')",
            ),
        )


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def _source_differences(report, f: GridFeatures, tool: str) -> None:
    if not f.impedance_sources:
        return
    ids = f.impedance_sources
    if tool == PANDAPOWER:
        report.model_difference(
            "source.impedance",
            "pandapower's balanced power flow holds the ext_grid bus as an ideal "
            "slack and uses its short-circuit data for fault calculations only. "
            "The converted Source keeps that impedance; pgml ignores it under "
            "slack='ideal' (the default) and applies it under slack='norton'.",
            element_type="Source",
            ids=ids,
            source_model="ideal slack at the ext_grid bus",
            pgml_model="ideal slack by default, Thevenin source under slack='norton'",
            match=ModelMatch(arguments={"solve_power_flow.slack": "ideal"}),
            matched=True,
        )
        if f.zero_sequence_sources or f.multiphase_lines:
            report.model_difference(
                "source.zero_sequence",
                "pandapower's unbalanced power flow (runpp_3ph) pins the positive "
                "sequence as an ideal slack, puts the short-circuit impedance in "
                "the negative- and zero-sequence networks and scales the "
                "zero-sequence value by the IEC voltage factor c = 1.1. pgml "
                "applies one Thevenin matrix with c = 1 to all sequences under "
                "slack='norton' and none under slack='ideal'.",
                element_type="Source",
                ids=ids,
                affects=("unbalanced", "harmonic"),
                source_model="sequence networks with c = 1.1 on the zero sequence",
                pgml_model="per-phase Thevenin matrix, c = 1",
                match=None,
            )
    else:
        name = "OpenDSS" if tool == OPENDSS else "power-grid-model"
        report.model_difference(
            "source.impedance",
            f"{name} always solves with the source behind its short-circuit "
            "impedance. pgml ignores that impedance under slack='ideal' (the "
            "default) and applies it under slack='norton'.",
            element_type="Source",
            ids=ids,
            source_model="voltage source behind its Thevenin impedance",
            pgml_model="ideal slack by default",
            match=ModelMatch(
                arguments={
                    "solve_power_flow.slack": "norton",
                    "solve_harmonic_flow.slack": "norton",
                }
            ),
        )


# --------------------------------------------------------------------------- #
# Voltage-regulating generators
# --------------------------------------------------------------------------- #
def _generator_differences(report, f: GridFeatures, tool: str) -> None:
    if not f.limited_generators:
        return
    enforce = bool(_default("appliance.generator.enforce_q_limits"))
    ids = f.limited_generators
    if tool != PANDAPOWER:
        # OpenDSS holds a model=3 generator inside Maxkvar/Minkvar, and
        # power-grid-model pins a voltage_regulator at q_min/q_max, as pgml does.
        return
    report.model_difference(
        "generator.q_limit_enforcement",
        "pandapower does not enforce the reactive limits of a voltage-regulating "
        "generator by default (runpp enforce_q_lims=False); "
        f"pgml's default is enforce_q_limits={enforce}, which releases the "
        "voltage setpoint of a generator that reaches a limit.",
        element_type="Generator",
        ids=ids,
        affects=("fundamental", "unbalanced"),
        source_model="runpp default enforce_q_lims=False: limits are ignored",
        pgml_model=f"appliance.generator.enforce_q_limits = {enforce}",
        match=ModelMatch(
            settings={"appliance.generator.enforce_q_limits": False},
            arguments={"solve_power_flow.enforce_q_limits": False},
            reference="runpp(net, enforce_q_lims=True) keeps pgml's default instead",
        ),
    )


# --------------------------------------------------------------------------- #
# Lines
# --------------------------------------------------------------------------- #
def _line_differences(report, f: GridFeatures, tool: str, grid_lines=()) -> None:
    if not f.lines:
        return
    lumped = {m: ids for m, ids in f.line_models.items() if m != "geometry"}
    if tool not in _HARMONIC_TOOLS:
        return
    # a line with an explicit reactance law states it; only lines on the
    # modeling default can differ from OpenDSS through that default
    explicit = {
        br.id
        for br in grid_lines
        if getattr(br, "earth_return", None) is not None
        and br.earth_return.x0_frequency is not None
    }
    seq = [i for i in lumped.get("sequence_aware", []) if i not in explicit]
    if seq:
        law = _default("line.earth_return.x0_frequency")
        guard = bool(_default("line.earth_return.x0_nonnegative"))
        rc = float(_default("line.earth_return.resistance_coeff_ohm_per_m_per_hz"))
        kx = float(_default("line.earth_return.reactance_coeff_ohm_per_m_per_hz"))
        report.model_difference(
            "line.earth_return_law",
            "OpenDSS corrects every lumped line for the Carson earth return as "
            "R += Rg*(h-1) and X = h*(X - 0.5*KXg*ln h) on every matrix entry, "
            "which is the zero-sequence law R0 += 3*Rg*(h-1), "
            "X0 = h*(X0 - 1.5*KXg*ln h) without a lower bound. pgml's "
            f"sequence-aware lines use x0_frequency='{law}' with "
            f"x0_nonnegative={guard}.",
            element_type="Line",
            ids=seq,
            affects=("harmonic",),
            source_model="Carson sub-linear X0, unguarded",
            pgml_model=f"x0_frequency={law}, x0_nonnegative={guard}",
            match=ModelMatch(
                preset="opendss",
                settings={
                    "line.earth_return.x0_frequency": "carson_sublinear",
                    "line.earth_return.x0_nonnegative": False,
                },
            ),
            values={
                "pgml_resistance_coeff_ohm_per_m_per_hz": rc,
                "pgml_reactance_coeff_ohm_per_m_per_hz": kx,
            },
        )
    if f.skin_lines:
        report.model_difference(
            "line.skin_effect",
            "OpenDSS keeps the conductor resistance of a lumped R/X line constant "
            "over frequency. These lines carry harmonic_skin_effect=True, so pgml "
            "raises their conductor resistance with a skin-effect multiplier. The "
            "flag is written at conversion from line.harmonic_model.skin_effect.",
            element_type="Line",
            ids=f.skin_lines,
            affects=("harmonic",),
            source_model="constant conductor resistance",
            pgml_model="skin-effect multiplier on the conductor resistance",
            match=ModelMatch(
                preset="opendss",
                settings={"line.harmonic_model.skin_effect": False},
                reference="convert inside the preset, or clear harmonic_skin_effect",
            ),
            matched=False,
        )
    pos = [
        i for i in lumped.get("positive_sequence", []) if i in set(f.multiphase_lines)
    ]
    if pos:
        report.model_difference(
            "line.positive_sequence_model",
            "These multi-phase lines use the positive-sequence harmonic model, "
            "which carries no earth-return term at all, whereas OpenDSS applies "
            "its Rg/Xg correction to every lumped line.",
            element_type="Line",
            ids=pos,
            affects=("harmonic",),
            source_model="Carson earth-return correction on every entry",
            pgml_model="harmonic_line_model = positive_sequence",
            match=ModelMatch(
                arguments={"to_grid.harmonic_line_model": "sequence_aware"},
                reference="Rg=0 Xg=0 on the OpenDSS lines",
            ),
        )


# --------------------------------------------------------------------------- #
# Harmonic device model
# --------------------------------------------------------------------------- #
def _harmonic_device_differences(report, f: GridFeatures, tool: str) -> None:
    if tool not in _HARMONIC_TOOLS:
        if f.injections or f.lines:
            name = "pandapower" if tool == PANDAPOWER else "power-grid-model"
            report.model_difference(
                "harmonics.unsupported",
                f"{name} solves the fundamental only. Harmonic line models, device "
                "shunts and spectra of the pgml grid have no counterpart, so only "
                "fundamental results can be compared.",
                element_type="Grid",
                count=1,
                affects=("harmonic",),
                source_model="no harmonic power flow",
                pgml_model="harmonic power flow with frequency-dependent elements",
                match=None,
            )
        return
    if not f.injections:
        return
    model = _default("appliance.harmonic_shunt.model")
    fraction = float(_default("appliance.harmonic_shunt.series_rl_fraction"))
    generation = _default("appliance.harmonic_shunt.generation_model")
    report.model_difference(
        "load.harmonic_shunt",
        "OpenDSS represents each load at harmonic orders as an admittance in "
        "parallel with its current spectrum (NeglectLoadY=No, %SeriesRL=50 by "
        "default, per load). pgml resolves the same choice from "
        f"appliance.harmonic_shunt.model='{model}' with "
        f"series_rl_fraction={fraction:g} unless a device carries its own "
        "harmonic_model.",
        element_type="Load",
        ids=f.loads,
        affects=("harmonic",),
        source_model="%SeriesRL series R-L / parallel R-L split per load",
        pgml_model=f"appliance.harmonic_shunt.model = {model}, "
        f"series_rl_fraction = {fraction:g}, generation_model = {generation}",
        match=ModelMatch(
            settings={
                "appliance.harmonic_shunt.model": "opendss",
                "appliance.harmonic_shunt.series_rl_fraction": 0.5,
            },
            arguments={"solve_harmonic_flow.load_shunt": "opendss"},
        ),
    )


def finalize_report(report: ConversionReport, grid: Grid, logger=None):
    """Complete a conversion report with the model differences and log it."""
    add_model_differences(report, grid)
    report.log(logger)
    return report


__all__ = [
    "GridFeatures",
    "add_model_differences",
    "finalize_report",
    "grid_features",
]
