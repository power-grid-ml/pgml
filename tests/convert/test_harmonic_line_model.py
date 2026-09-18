"""Converters resolve the configured harmonic line model (and say which one).

A source library has no frequency-dependent line model: pandapower and
power-grid-model are fundamental-only, and OpenDSS exports only its fundamental
matrices. A converted grid therefore has to be given the model that reproduces the
harmonic behaviour, or a harmonic solve silently falls back to constant ``R`` with
``X`` proportional to ``h`` (the naive model the modeling defaults deliberately do not
choose). These tests pin that the converters apply the documented default, log it, let
the caller override it, and do not change the FUNDAMENTAL in doing so.
"""

from __future__ import annotations

import logging
import math

import pytest
import torch

from pgml.assembly import assemble_ybus
from pgml.convert._common import PhaseMode
from pgml.defaults import get as cfg
from pgml.schemas.grid_schema import Line

CDT = torch.complex128


def _lines(grid):
    return [b for b in grid.branches if isinstance(b, Line)]


def _models(grid) -> set:
    return {ln.harmonic_line_model for ln in _lines(grid)}


@pytest.fixture(scope="module")
def pp_net():
    pp = pytest.importorskip("pandapower")
    pn = pytest.importorskip("pandapower.networks")
    net = pn.case33bw()
    pp.runpp(net, numba=False)
    return net


# ---------------------------------------------------------------------------
# pandapower
# ---------------------------------------------------------------------------
def test_pandapower_single_phase_equiv_gets_positive_sequence(pp_net):
    """1-phase equivalent lines get ``line.harmonic_model.single_phase``."""
    from pgml.convert.pandapower import to_grid

    grid, _ = to_grid(pp_net)
    assert cfg("line.harmonic_model.single_phase") == "positive_sequence"
    assert _models(grid) == {"positive_sequence"}
    # The skin flag stays unset so that it resolves with the other model options at
    # assembly time (a later preset can still switch it).
    assert all(ln.harmonic_skin_effect is None for ln in _lines(grid))


def test_pandapower_three_phase_gets_sequence_aware(pp_net):
    """3-phase lines get ``line.harmonic_model.three_phase`` (the 4-wire model)."""
    from pgml.convert.pandapower import to_grid

    grid, _ = to_grid(pp_net, phase_mode=PhaseMode.THREE_PHASE)
    assert cfg("line.harmonic_model.three_phase") == "sequence_aware"
    assert _models(grid) == {"sequence_aware"}


@pytest.mark.parametrize(
    "requested,expected",
    [
        ("naive", {"naive"}),
        ("positive_sequence", {"positive_sequence"}),
        ("none", {None}),
    ],
)
def test_pandapower_keyword_overrides_the_default(pp_net, requested, expected):
    """``harmonic_line_model=`` wins over the configured default for every line."""
    from pgml.convert.pandapower import to_grid

    grid, _ = to_grid(pp_net, harmonic_line_model=requested)
    assert _models(grid) == expected


def test_pandapower_unknown_model_raises(pp_net):
    """An unrecognised model name raises instead of silently doing something else."""
    from pgml.convert.pandapower import to_grid
    from pgml.errors import InputError

    with pytest.raises(InputError, match="Unknown harmonic line model"):
        to_grid(pp_net, harmonic_line_model="sequence-aware")


def test_pandapower_logs_the_applied_default(pp_net, caplog):
    """One INFO line per grid names the model and where it came from."""
    from pgml.convert.pandapower import to_grid

    with caplog.at_level(logging.INFO, logger="pgml"):
        to_grid(pp_net)
    msgs = [r.getMessage() for r in caplog.records]
    applied = [m for m in msgs if "harmonic line model applied" in m]
    assert len(applied) == 1, msgs
    assert "line.harmonic_model.*" in applied[0]
    assert "positive_sequence" in applied[0]


def test_pandapower_warns_once_about_invented_zero_sequence(pp_net, caplog):
    """The invented R0/X0/C0 ratios are warned once per grid, with their values."""
    from pgml.convert.pandapower import to_grid

    with caplog.at_level(logging.INFO, logger="pgml"):
        to_grid(pp_net, phase_mode=PhaseMode.THREE_PHASE)
    warns = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "zero-sequence data" in r.getMessage()
    ]
    assert len(warns) == 1, warns
    assert "R0/R1=4" in warns[0] and "X0/X1=3" in warns[0] and "C0/C1=0.5" in warns[0]
    assert f"{len(_lines(to_grid(pp_net)[0]))} three-phase lines" in warns[0]


def test_single_phase_equivalent_does_not_warn_about_zero_sequence(pp_net, caplog):
    """The 1-phase equivalent never uses zero-sequence data, so it must not warn."""
    from pgml.convert.pandapower import to_grid

    with caplog.at_level(logging.INFO, logger="pgml"):
        to_grid(pp_net)
    assert not [r for r in caplog.records if "zero-sequence data" in r.getMessage()]


def test_model_choice_leaves_the_fundamental_untouched(pp_net):
    """Every model reproduces the f0 Y-bus of the raw stored parameters exactly.

    The models differ only in how Z scales with frequency: at ``f0`` the skin
    multiplier is 1 by construction and the sequence decomposition is its own inverse,
    so a converted grid's load-flow answer cannot change with the harmonic model. The
    tolerance is pure floating point (the sequence round trip is not bit-identical).
    """
    from pgml.convert.pandapower import to_grid

    f0 = float(pp_net.f_hz)
    ref = None
    for mode in (PhaseMode.SINGLE_PHASE_EQUIV, PhaseMode.THREE_PHASE):
        for model in ("none", None, "naive", "positive_sequence"):
            if model == "sequence_aware" and mode is PhaseMode.SINGLE_PHASE_EQUIV:
                continue
            grid, _ = to_grid(pp_net, phase_mode=mode, harmonic_line_model=model)
            y = assemble_ybus(grid, [f0], dtype=CDT).Y
            if model == "none":
                ref = y
                continue
            rel = float((y - ref).abs().max() / ref.abs().max())
            assert rel < 1e-14, f"{mode}/{model}: f0 Y-bus moved by {rel:.2e}"


def test_harmonic_model_changes_the_harmonic_ybus(pp_net):
    """The models are genuinely different above the fundamental (not a no-op).

    Compared on one line's series admittance (the Y-bus maximum is the source Norton,
    which no line model touches). The skin effect raises ``R1`` at ``h=13`` by about
    10 %, and at that order the reactance dominates ``|Z|``, so the series admittance
    moves by a few parts per thousand.
    """
    from pgml.convert.pandapower import to_grid

    f0 = float(pp_net.f_hz)
    g_naive, _ = to_grid(pp_net, harmonic_line_model="naive")
    g_def, _ = to_grid(pp_net)  # positive_sequence + skin
    yb_naive = assemble_ybus(g_naive, [13.0 * f0], dtype=CDT)
    yb_def = assemble_ybus(g_def, [13.0 * f0], dtype=CDT)
    ln = _lines(g_naive)[0]
    row = yb_naive.index.row(ln.from_node, ln.from_phases[0])
    col = yb_naive.index.row(ln.to_node, ln.to_phases[0])
    y_naive = complex(yb_naive.Y[0, row, col])
    y_def = complex(yb_def.Y[0, row, col])
    rel = abs(y_def - y_naive) / abs(y_naive)
    assert rel > 1e-3, f"h=13 series admittance is model-independent ({rel:.2e})"


# ---------------------------------------------------------------------------
# power-grid-model
# ---------------------------------------------------------------------------
def test_pgm_converter_applies_the_default():
    """The power-grid-model converter resolves the model and warns about R0/X0/C0."""
    pgm = pytest.importorskip("power_grid_model")
    from pgml.convert.pgm import to_grid

    node = pgm.initialize_array("input", "node", 2)
    node["id"] = [1, 2]
    node["u_rated"] = [400.0, 400.0]
    line = pgm.initialize_array("input", "line", 1)
    line["id"] = [3]
    line["from_node"] = [1]
    line["to_node"] = [2]
    line["from_status"] = [1]
    line["to_status"] = [1]
    line["r1"] = [0.02]
    line["x1"] = [0.01]
    line["c1"] = [0.0]
    line["tan1"] = [0.0]
    source = pgm.initialize_array("input", "source", 1)
    source["id"] = [4]
    source["node"] = [1]
    source["status"] = [1]
    source["u_ref"] = [1.0]
    source["sk"] = [1.0e8]
    data = {"node": node, "line": line, "source": source}

    grid, _ = to_grid(data, phase_mode=PhaseMode.THREE_PHASE)
    assert _models(grid) == {"sequence_aware"}
    grid, _ = to_grid(data, harmonic_line_model="naive")
    assert _models(grid) == {"naive"}


# ---------------------------------------------------------------------------
# OpenDSS
# ---------------------------------------------------------------------------
def test_opendss_converter_applies_the_default():
    """The OpenDSS converter resolves the model for its matrix-defined lines."""
    dss = pytest.importorskip("opendssdirect")
    from pgml.convert.opendss import to_grid

    dss.Text.Command("Clear")
    dss.Text.Command("Set DefaultBaseFrequency=50")
    dss.Text.Command(
        "New Circuit.t basekv=0.4 phases=3 bus1=a frequency=50 r1=1e-3 x1=1e-3"
    )
    dss.Text.Command(
        "New Line.l1 phases=3 bus1=a.1.2.3 bus2=b.1.2.3 r1=0.2 x1=0.08 "
        "r0=0.8 x0=0.24 c1=0 c0=0 length=0.1 units=km"
    )
    dss.Text.Command("New Load.ld1 bus1=b.1.2.3 phases=3 kv=0.4 kw=10 kvar=3")
    dss.Text.Command("Set voltagebases=[0.4]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")

    grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    assert _models(grid) == {"sequence_aware"}
    grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE, harmonic_line_model="none")
    assert _models(grid) == {None}

    # The fundamental is unchanged by the model choice (see the pandapower test).
    y_a = assemble_ybus(grid, [50.0], dtype=CDT).Y
    grid_b, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    y_b = assemble_ybus(grid_b, [50.0], dtype=CDT).Y
    rel = float((y_b - y_a).abs().max() / y_a.abs().max())
    assert rel < 1e-14, f"f0 Y-bus moved by {rel:.2e}"


# ---------------------------------------------------------------------------
# hand-built grids stay the caller's responsibility, but loudly
# ---------------------------------------------------------------------------
def _unresolved_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "no harmonic line model" in r.getMessage()
    ]


def test_unresolved_lines_warn_at_assembly(caplog):
    """A grid built without a converter logs a WARNING naming the entry point."""
    from pgml.grids import synthetic_feeder

    grid = synthetic_feeder(4)
    with caplog.at_level(logging.INFO, logger="pgml"):
        assemble_ybus(grid, [2.0 * math.pi], dtype=CDT)
    warns = _unresolved_warnings(caplog)
    assert len(warns) == 1, warns
    assert "apply_default_harmonic_model" in warns[0]


def test_unresolved_lines_warn_once_per_harmonic_solve(caplog):
    """The harmonic study names the count and the first line ids, exactly once.

    A grid assembled without a converter keeps the naive model at every order while a
    converted one resolves to the sequence-aware model, so the solve that is actually
    affected has to say so — and a study over several orders may say it only once.
    """
    from pgml.grids import synthetic_feeder
    from pgml.solver import solve_harmonic_flow

    grid = synthetic_feeder(4)
    ids = [int(b.id) for b in grid.branches if isinstance(b, Line)]
    with caplog.at_level(logging.INFO, logger="pgml"):
        solve_harmonic_flow(grid, [1, 5, 7], dtype=CDT)
    warns = _unresolved_warnings(caplog)
    assert len(warns) == 1, warns
    assert f"{len(ids)} of {len(ids)} lines" in warns[0]
    assert str(ids[0]) in warns[0]
    assert "apply_default_harmonic_model" in warns[0]


def test_a_resolved_grid_is_silent(caplog):
    """Applying the documented default removes the warning."""
    from pgml.geometry import apply_default_harmonic_model
    from pgml.grids import synthetic_feeder
    from pgml.solver import solve_harmonic_flow

    grid = apply_default_harmonic_model(synthetic_feeder(4))
    with caplog.at_level(logging.INFO, logger="pgml"):
        solve_harmonic_flow(grid, [1, 5], dtype=CDT)
    assert _unresolved_warnings(caplog) == []
