"""Increment 0: calculation-symmetry / connection resolution + modeling logging.

Pins the power-grid-model-style resolution rule (`references/asymmetric_modeling.md`
§1) and the INFO modeling summary (neutral modeled iff a node carries Phase.N).
"""

from __future__ import annotations

import logging

import pytest

from pgml.assembly._symmetry import (
    log_modeling_summary,
    resolve_asymmetric,
    resolve_connection,
)
from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    Load,
    Node,
    Phase,
    WindingConnection,
)

ABC = (Phase.A, Phase.B, Phase.C)
ABCN = (Phase.A, Phase.B, Phase.C, Phase.N)


def _grid(nodes, appliances):
    return Grid(nodes=nodes, appliances=appliances)


def _balanced_grid(node_phases=ABC):
    return _grid(
        [Node(id=1, u_rated_v=400.0, phases=node_phases)],
        [Load(id=1, node=1, phases=ABC, p_nom_w=3000.0)],
    )


def _per_phase_grid():
    return _grid(
        [Node(id=1, u_rated_v=400.0, phases=ABC)],
        [Load(id=1, node=1, phases=ABC, p_nom_w=3000.0,
              p_nom_per_phase_w=(1500.0, 1000.0, 500.0))],
    )


# --- resolve_asymmetric: forced modes ---------------------------------------
def test_forced_symmetric_and_asymmetric():
    g = _per_phase_grid()  # has per-phase data; forcing must override the data
    assert resolve_asymmetric(g, mode="symmetric") is False
    assert resolve_asymmetric(_balanced_grid(), mode="asymmetric") is True


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="calculation symmetry must be one of"):
        resolve_asymmetric(_balanced_grid(), mode="balanced")


# --- resolve_asymmetric: auto -----------------------------------------------
def test_auto_symmetric_when_no_per_phase_data():
    assert resolve_asymmetric(_balanced_grid(), mode="auto") is False


def test_auto_asymmetric_from_appliance_per_phase():
    assert resolve_asymmetric(_per_phase_grid(), mode="auto") is True


def test_auto_asymmetric_from_operating_point():
    op = {1: {"p_per_phase_w": (1500.0, 1000.0, 500.0)}}
    assert resolve_asymmetric(_balanced_grid(), op, mode="auto") is True
    # total-only operating point does NOT trigger asymmetric
    assert resolve_asymmetric(_balanced_grid(), {1: {"p_w": 3000.0}}, mode="auto") is False


def test_default_mode_is_config_auto():
    # mode=None reads config 'calculation.symmetry' (default 'auto').
    assert resolve_asymmetric(_balanced_grid()) is False
    assert resolve_asymmetric(_per_phase_grid()) is True


# --- resolve_connection ------------------------------------------------------
def test_connection_explicit_wins():
    a = Load(id=1, node=1, phases=(Phase.A, Phase.B), p_nom_w=2000.0,
             connection=WindingConnection.DELTA)
    assert resolve_connection(a) == WindingConnection.DELTA


def test_connection_config_default_multi_and_single_phase():
    multi = Load(id=1, node=1, phases=ABC, p_nom_w=3000.0)
    single = Load(id=2, node=1, phases=(Phase.A,), p_nom_w=1000.0)
    assert resolve_connection(multi) == WindingConnection.WYE
    assert resolve_connection(single) == WindingConnection.WYE


def test_connection_resolves_for_generator():
    g = Generator(id=1, node=1, phases=ABC, p_nom_w=3000.0)
    assert resolve_connection(g) == WindingConnection.WYE


# --- log_modeling_summary ----------------------------------------------------
def test_logs_neutral_modeled_when_phase_n_present(caplog):
    g = _grid(
        [Node(id=1, u_rated_v=400.0, phases=ABCN)],
        [Load(id=1, node=1, phases=ABC, p_nom_w=3000.0)],
    )
    with caplog.at_level(logging.INFO, logger="pgml"):
        log_modeling_summary(g, asymmetric=True)
    text = caplog.text
    assert "NEUTRAL modeled" in text
    assert "node(s): [1]" in text
    assert "ASYMMETRIC" in text


def test_logs_ground_return_when_no_neutral(caplog):
    with caplog.at_level(logging.INFO, logger="pgml"):
        log_modeling_summary(_balanced_grid(), asymmetric=False)
    assert "no Phase.N present" in caplog.text
    assert "wyex1" in caplog.text  # one WYE-resolved load
    assert "SYMMETRIC" in caplog.text


def test_resolve_asymmetric_emits_info(caplog):
    with caplog.at_level(logging.INFO, logger="pgml"):
        resolve_asymmetric(_balanced_grid(), mode="auto")
    assert "calculation symmetry: SYMMETRIC" in caplog.text
