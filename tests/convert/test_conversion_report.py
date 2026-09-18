"""The conversion report data model and the model-difference catalogue."""

from __future__ import annotations

import json
import logging
import math

import pytest

from pgml import defaults
from pgml.convert import (
    ConversionReport,
    ModelMatch,
    ReportCategory,
    add_model_differences,
)
from pgml.schemas.grid_schema import (
    ComplexTap,
    Grid,
    Line,
    Load,
    Node,
    Phase,
    Source,
    Transformer,
    WindingConnection,
)

OMEGA = 2.0 * math.pi * 50.0


def _line_grid(*, transformer: bool = False) -> Grid:
    nodes = [
        Node(id=1, u_rated_v=400.0, phases=(Phase.A,)),
        Node(id=2, u_rated_v=400.0, phases=(Phase.A,)),
    ]
    branches = [
        Line(
            id=10,
            from_node=1,
            to_node=2,
            from_phases=(Phase.A,),
            to_phases=(Phase.A,),
            length_m=100.0,
            series_resistance_ohm_per_m=[[3e-4]],
            series_inductance_h_per_m=[[2e-4 / OMEGA]],
            shunt_capacitance_f_per_m=[[0.0]],
            harmonic_line_model="positive_sequence",
            harmonic_skin_effect=False,
        )
    ]
    if transformer:
        nodes.append(Node(id=3, u_rated_v=20e3, phases=(Phase.A,)))
        branches.append(
            Transformer(
                id=11,
                from_node=3,
                to_node=1,
                from_phases=(Phase.A,),
                to_phases=(Phase.A,),
                s_rated_va=400e3,
                u_rated_from_v=20e3,
                u_rated_to_v=400.0,
                from_connection=WindingConnection.WYE_GROUNDED,
                to_connection=WindingConnection.WYE_GROUNDED,
                series_resistance_ohm=0.004,
                series_inductance_h=0.016 / OMEGA,
                magnetizing_conductance_s=1e-6,
                magnetizing_inductance_h=5e3 / OMEGA,
                tap=ComplexTap(ratio_magnitude=1.0),
            )
        )
    source_node = 3 if transformer else 1
    return Grid(
        base_frequency_hz=50.0,
        nodes=nodes,
        branches=branches,
        appliances=[
            Source(
                id=20,
                node=source_node,
                phases=(Phase.A,),
                u_ref_v=(nodes[-1].u_rated_v if transformer else 400.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.0]],
                inductance_h=[[0.0]],
            ),
            Load(id=21, node=2, phases=(Phase.A,), p_nom_w=1e3, q_nom_var=2e2),
        ],
    )


def test_report_round_trips_through_json():
    report = ConversionReport(tool="opendss", direction="import", options={"a": 1})
    report.dropped("invcontrol", "not converted", ids=["ctl1", "ctl2"])
    report.approximated(
        "pvsystem.snapshot",
        "imported at the solved operating point",
        element_type="pvsystem",
        ids=["pv1"],
        values={"kw": 12.5},
    )
    report.model_difference(
        "transformer.magnetizing_placement",
        "different placement",
        element_type="Transformer",
        ids=[3],
        match=ModelMatch(
            preset="opendss",
            settings={"transformer.magnetizing_placement": "to_terminal"},
        ),
    )
    data = json.loads(report.to_json())
    assert data["counts"] == {"dropped": 1, "approximated": 1, "model_difference": 1}
    restored = ConversionReport.from_dict(data)
    assert restored.to_dict() == report.to_dict()
    assert restored.keys() == [
        "dropped.invcontrol",
        "approx.pvsystem.snapshot",
        "model.transformer.magnetizing_placement",
    ]
    assert "dropped.invcontrol" in report
    assert report.get("dropped.invcontrol")[0].count == 2


def test_summary_names_every_entry_and_the_match():
    report = ConversionReport(tool="pandapower", direction="import")
    report.dropped("ward", "2 ward element(s) are not converted", ids=[0, 1])
    report.model_difference("x", "differs", match=None)
    text = report.summary()
    assert "[dropped.ward] 2 x ward" in text
    assert "Match: none." in text
    assert text.splitlines()[0].startswith("pandapower -> Grid: 1 dropped")


def test_long_id_lists_are_truncated_but_counted():
    report = ConversionReport(tool="pandapower", direction="import")
    entry = report.dropped("motor", "many", ids=range(500))
    assert entry.count == 500
    assert len(entry.ids) == 50


def test_model_difference_is_matched_inside_the_preset():
    grid = _line_grid(transformer=True)
    report = add_model_differences(ConversionReport("opendss", "import"), grid)
    (entry,) = report.get("model.transformer.magnetizing_placement")
    assert entry.ids == (11,)
    assert not entry.matched
    assert entry.match.preset == "opendss"
    with defaults.use_preset("opendss"):
        inside = add_model_differences(ConversionReport("opendss", "import"), grid)
    assert inside.get("model.transformer.magnetizing_placement")[0].matched
    assert "transformer.magnetizing_placement" in report.match_settings()


def test_differences_follow_the_grid_content():
    report = add_model_differences(
        ConversionReport("opendss", "import"), _line_grid(transformer=False)
    )
    assert "model.transformer.magnetizing_placement" not in report
    # a single-phase positive-sequence line without skin effect has no earth term
    assert "model.line.earth_return_law" not in report


def test_fundamental_only_tools_report_harmonics_without_raising_the_level(caplog):
    report = ConversionReport(
        "power-grid-model", "import", comparable=("fundamental", "unbalanced")
    )
    add_model_differences(report, _line_grid())
    assert report.keys() == ["model.harmonics.unsupported"]
    assert report.is_exact("fundamental")
    assert not report.is_exact("harmonic")
    with caplog.at_level(logging.INFO, logger="pgml"):
        report.log()
    assert all(r.levelno == logging.INFO for r in caplog.records)


def test_open_entries_are_logged_at_warning(caplog):
    report = ConversionReport("pandapower", "import")
    report.dropped("ward", "not converted", ids=[0])
    report.dropped("motor", "already announced", ids=[0], announced=True)
    with caplog.at_level(logging.INFO, logger="pgml"):
        report.log()
    messages = [r.getMessage() for r in caplog.records]
    assert any("[dropped.ward]" in m for m in messages)
    assert not any("[dropped.motor]" in m for m in messages)
    assert caplog.records[-1].levelno == logging.WARNING
    assert "2 dropped" in messages[-1]


def test_unknown_result_class_is_rejected():
    report = ConversionReport("pandapower", "import")
    with pytest.raises(ValueError):
        report.add("k", ReportCategory.DROPPED, "m", affects=("transient",))
