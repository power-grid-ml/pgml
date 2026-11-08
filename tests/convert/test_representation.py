"""``pgml.representation``: reading and migrating the per-phase equivalent."""

from __future__ import annotations

import math

import pytest

pp = pytest.importorskip("pandapower")

from pgml.convert import PhaseMode  # noqa: E402
from pgml.convert._export import system_view  # noqa: E402
from pgml.convert.pandapower import to_grid  # noqa: E402
from pgml.representation import (  # noqa: E402
    PHASE_MODE_TAG,
    PHASE_MODES,
    phase_mode_of,
    positive_sequence_resistance_ohm_per_m,
    system_totals,
    to_per_phase_equivalent,
)
from pgml.schemas.grid_schema import Line, Load, Source, Transformer  # noqa: E402


def _net():
    net = pp.create_empty_network(f_hz=50.0, sn_mva=1.0)
    b0 = pp.create_bus(net, vn_kv=20.0)
    b1 = pp.create_bus(net, vn_kv=0.4)
    b2 = pp.create_bus(net, vn_kv=0.4)
    pp.create_ext_grid(net, b0, vm_pu=1.02)
    pp.create_transformer_from_parameters(
        net,
        b0,
        b1,
        sn_mva=0.63,
        vn_hv_kv=20.0,
        vn_lv_kv=0.4,
        vkr_percent=1.0,
        vk_percent=4.0,
        pfe_kw=0.5,
        i0_percent=0.2,
        shift_degree=150.0,
        vector_group="Dyn5",
    )
    pp.create_line_from_parameters(
        net,
        b1,
        b2,
        length_km=0.3,
        r_ohm_per_km=0.2,
        x_ohm_per_km=0.08,
        c_nf_per_km=100.0,
        max_i_ka=0.27,
    )
    pp.create_load(net, b2, p_mw=0.2, q_mvar=0.06)
    pp.create_sgen(net, b2, p_mw=0.05, q_mvar=0.01)
    return net


def test_phase_mode_of_reads_the_tag():
    grid, _ = to_grid(_net())
    assert phase_mode_of(grid) == "single_phase_equiv"
    grid.metadata.tags.pop(PHASE_MODE_TAG)
    assert phase_mode_of(grid) is None
    grid.metadata.tags[PHASE_MODE_TAG] = "nonsense"
    with pytest.raises(ValueError):
        phase_mode_of(grid)
    assert set(m.value for m in PhaseMode) <= set(PHASE_MODES)


@pytest.mark.parametrize("mode", list(PhaseMode), ids=lambda m: m.value)
def test_system_totals_agree_across_representations(mode):
    """Every representation reads back the source network's system values."""
    net = _net()
    grid, _ = to_grid(net, phase_mode=mode)
    st = system_totals(grid)
    assert not st.legacy
    by_v = sorted(st.node_voltage_ll_v(n) for n in grid.nodes)
    assert by_v == pytest.approx(sorted([400.0, 400.0, 20000.0]), rel=1e-12)
    load = next(a for a in grid.appliances if isinstance(a, Load))
    assert st.total(load.p_nom_w) == pytest.approx(0.2e6, rel=1e-12)
    trafo = next(b for b in grid.branches if isinstance(b, Transformer))
    assert st.total(trafo.s_rated_va) == pytest.approx(0.63e6, rel=1e-12)
    assert st.transformer_voltage_ll_v(trafo.u_rated_from_v) == pytest.approx(
        20_000.0, rel=1e-12
    )
    assert st.transformer_voltage_ll_v(trafo.u_rated_to_v) == pytest.approx(
        400.0, rel=1e-12
    )
    assert st.total(None) == 0.0


def test_legacy_grid_is_read_as_stored():
    grid, _ = to_grid(_net())
    legacy = system_view(grid)
    legacy.metadata.tags.pop(PHASE_MODE_TAG)
    st = system_totals(legacy)
    assert st.legacy and st.power_factor == 1.0
    assert st.node_voltage_ll_v(legacy.nodes[0]) == pytest.approx(20_000.0, rel=1e-12)


def test_migration_restores_the_per_phase_equivalent():
    per_phase, _ = to_grid(_net())
    legacy = system_view(per_phase)
    legacy.metadata.tags.pop(PHASE_MODE_TAG)
    migrated = to_per_phase_equivalent(legacy)
    assert phase_mode_of(migrated) == "single_phase_equiv"
    assert PHASE_MODE_TAG not in legacy.metadata.tags  # input untouched
    for a, b in zip(migrated.nodes, per_phase.nodes):
        assert a.u_rated_v == pytest.approx(b.u_rated_v, rel=1e-14)
    for a, b in zip(migrated.appliances, per_phase.appliances):
        if isinstance(a, Load):
            assert a.p_nom_w == pytest.approx(b.p_nom_w, rel=1e-14)
        if isinstance(a, Source):
            assert a.u_ref_v == pytest.approx(b.u_ref_v, rel=1e-14)
    for a, b in zip(migrated.branches, per_phase.branches):
        if isinstance(a, Transformer):
            assert a.s_rated_va == pytest.approx(b.s_rated_va, rel=1e-14)
            assert a.u_rated_from_v == pytest.approx(b.u_rated_from_v, rel=1e-14)
    assert to_per_phase_equivalent(migrated) is migrated


def test_migration_refuses_other_representations():
    three, _ = to_grid(_net(), phase_mode=PhaseMode.THREE_PHASE)
    with pytest.raises(ValueError):
        to_per_phase_equivalent(three)
    three.metadata.tags.pop(PHASE_MODE_TAG)
    with pytest.raises(ValueError):
        to_per_phase_equivalent(three)  # untagged but three-phase nodes


def test_positive_sequence_resistance_is_mode_independent():
    net = _net()
    values = []
    for mode in PhaseMode:
        grid, _ = to_grid(net, phase_mode=mode)
        line = next(b for b in grid.branches if isinstance(b, Line))
        values.append(positive_sequence_resistance_ohm_per_m(line))
    assert values[0] == pytest.approx(0.2 / 1000.0, rel=1e-12)
    assert values[1] == pytest.approx(values[0], rel=1e-12)
    assert math.isfinite(values[1])
