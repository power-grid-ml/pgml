"""``pgml.representation``: reading and migrating the per-phase equivalent."""

from __future__ import annotations

import math

import pytest

pp = pytest.importorskip("pandapower")

from pgml.convert import PhaseMode  # noqa: E402
from pgml.convert._export import system_view  # noqa: E402
from pgml.convert.pandapower import to_grid  # noqa: E402
from pgml.errors import InputError, PgmlError  # noqa: E402
from pgml.representation import (  # noqa: E402
    PHASE_MODE_TAG,
    PHASE_MODES,
    mark_legacy,
    phase_mode_of,
    positive_sequence_resistance_ohm_per_m,
    system_totals,
    to_per_phase_equivalent,
)
from pgml.scenarios import iec61000_3_2_device_caps  # noqa: E402
from pgml.schemas.grid_schema import (  # noqa: E402
    ConsumerType,
    Line,
    Load,
    Source,
    Transformer,
)


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
    grid.metadata.tags[PHASE_MODE_TAG] = "legacy_ll"
    assert phase_mode_of(grid) == "legacy_ll"
    grid.metadata.tags[PHASE_MODE_TAG] = "nonsense"
    with pytest.raises(InputError, match="nonsense") as info:
        phase_mode_of(grid)
    # The pgml input error, still a ValueError for callers that catch the builtin.
    assert isinstance(info.value, ValueError)
    assert isinstance(info.value, PgmlError)
    with pytest.raises(InputError):
        system_totals(grid)
    assert set(m.value for m in PhaseMode) <= set(PHASE_MODES)


def _legacy_ll(net=None):
    """The grid an older converter wrote: one-phase nodes, line-to-line, totals, no tag."""
    grid, _ = to_grid(_net() if net is None else net)
    legacy = system_view(grid)
    legacy.metadata.tags.pop(PHASE_MODE_TAG)
    return legacy


def _representations(net):
    """The source network in every representation that reads back its system values."""
    out = {m.value: to_grid(net, phase_mode=m)[0] for m in PhaseMode}
    out["legacy_ll"] = mark_legacy(_legacy_ll(net))
    return out


@pytest.mark.parametrize(
    "mode", ["single_phase_equiv", "three_phase", "legacy_ll"], ids=str
)
def test_system_totals_agree_across_representations(mode):
    """Every representation reads back the source network's system values."""
    grid = _representations(_net())[mode]
    st = system_totals(grid)
    assert st.legacy == (mode == "legacy_ll")
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


def test_untagged_grid_is_read_by_the_schema_rule():
    """No tag: a one-phase node is a single-phase circuit with a line-to-neutral rating."""
    per_phase, _ = to_grid(_net())
    untagged = per_phase.model_copy(deep=True)
    untagged.metadata.tags.pop(PHASE_MODE_TAG)
    st = system_totals(untagged)
    assert st.phase_mode is None and not st.legacy and st.power_factor == 1.0
    node = untagged.nodes[0]
    assert st.node_voltage_ll_v(node) == pytest.approx(
        float(node.u_rated_v) * math.sqrt(3.0), rel=1e-12
    )
    load = next(a for a in untagged.appliances if isinstance(a, Load))
    assert st.total(load.p_nom_w) == pytest.approx(float(load.p_nom_w), rel=1e-12)


def test_legacy_ll_grid_is_read_as_stored():
    legacy = mark_legacy(_legacy_ll())
    st = system_totals(legacy)
    assert st.legacy and st.power_factor == 1.0
    assert st.node_voltage_ll_v(legacy.nodes[0]) == pytest.approx(20_000.0, rel=1e-12)


def test_mark_legacy_tags_a_copy_and_changes_no_value():
    untagged = _legacy_ll()
    tagged = mark_legacy(untagged)
    assert phase_mode_of(tagged) == "legacy_ll"
    assert PHASE_MODE_TAG not in untagged.metadata.tags  # input untouched
    assert [n.u_rated_v for n in tagged.nodes] == [n.u_rated_v for n in untagged.nodes]
    assert mark_legacy(tagged) is tagged
    per_phase, _ = to_grid(_net())
    with pytest.raises(InputError):
        mark_legacy(per_phase)  # tagged single_phase_equiv
    three, _ = to_grid(_net(), phase_mode=PhaseMode.THREE_PHASE)
    three.metadata.tags.pop(PHASE_MODE_TAG)
    with pytest.raises(InputError):
        mark_legacy(three)  # untagged but three-phase nodes


def test_legacy_ll_grid_exports_as_written():
    legacy = mark_legacy(_legacy_ll())
    assert system_view(legacy) is legacy


def _assert_matches_per_phase(migrated, per_phase):
    assert phase_mode_of(migrated) == "single_phase_equiv"
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


def test_migration_restores_the_per_phase_equivalent():
    per_phase, _ = to_grid(_net())
    legacy = mark_legacy(_legacy_ll())
    migrated = to_per_phase_equivalent(legacy)
    assert phase_mode_of(legacy) == "legacy_ll"  # input untouched
    _assert_matches_per_phase(migrated, per_phase)
    assert to_per_phase_equivalent(migrated) is migrated


def test_migration_of_an_untagged_grid_needs_assume_legacy():
    per_phase, _ = to_grid(_net())
    untagged = _legacy_ll()
    with pytest.raises(InputError, match="assume_legacy"):
        to_per_phase_equivalent(untagged)
    migrated = to_per_phase_equivalent(untagged, assume_legacy=True)
    assert PHASE_MODE_TAG not in untagged.metadata.tags  # input untouched
    _assert_matches_per_phase(migrated, per_phase)


def test_migration_refuses_other_representations():
    three, _ = to_grid(_net(), phase_mode=PhaseMode.THREE_PHASE)
    with pytest.raises(InputError):
        to_per_phase_equivalent(three)
    three.metadata.tags.pop(PHASE_MODE_TAG)
    with pytest.raises(InputError):
        to_per_phase_equivalent(three, assume_legacy=True)  # three-phase nodes


def _office_net():
    """A 0.4 kV feeder with a 900 W office load: Class A by its total nameplate."""
    net = _net()
    pp.create_load(net, 2, p_mw=900e-6, q_mvar=0.0)
    return net


def test_iec_device_caps_agree_across_representations():
    """Caps and the Class D window read the system total in every representation."""
    reps = _representations(_office_net())
    caps = {}
    for mode, grid in reps.items():
        st = system_totals(grid)
        k, office = next(
            (k, a)
            for k, a in enumerate(grid.appliances)
            if isinstance(a, Load) and st.total(a.p_nom_w) == pytest.approx(900.0)
        )
        grid.appliances[k] = office.model_copy(
            update={"consumer_type": ConsumerType.OFFICE}
        )
        caps[mode] = iec61000_3_2_device_caps(grid, [office.id], [3, 5, 7])[office.id]
    # Class A: limit [A] over the per-phase fundamental P / (3 * 400 V / sqrt(3)).
    i1 = 900.0 / (3.0 * 400.0 / math.sqrt(3.0))
    assert caps["single_phase_equiv"][7] == pytest.approx(0.77 / i1, rel=1e-12)
    for mode in ("three_phase", "legacy_ll"):
        for order in (3, 5, 7):
            assert caps[mode][order] == pytest.approx(
                caps["single_phase_equiv"][order], rel=1e-12
            )


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


def test_line_to_line_voltage_of_two_conductor_nodes():
    """A node with two phase conductors that sets u_rated_ln_v stores the phase-to-phase
    voltage in u_rated_v; without u_rated_ln_v a node of fewer than three rows follows
    the schema rule (line-to-neutral)."""
    from pgml.representation import SystemTotals
    from pgml.schemas.grid_schema import Node, Phase

    split = Node(id=1, u_rated_v=440.0, u_rated_ln_v=220.0, phases=(Phase.A, Phase.B))
    two_of_three = Node(
        id=2,
        u_rated_v=400.0,
        u_rated_ln_v=400.0 / math.sqrt(3.0),
        phases=(Phase.A, Phase.C),
    )
    one = Node(id=3, u_rated_v=230.0, u_rated_ln_v=230.0, phases=(Phase.A, Phase.N))
    plain = Node(id=4, u_rated_v=230.0, phases=(Phase.A, Phase.B))
    split_n = Node(
        id=5, u_rated_v=440.0, u_rated_ln_v=220.0, phases=(Phase.A, Phase.B, Phase.N)
    )
    for mode in ("as_built", "three_phase", "single_phase_equiv", None):
        st = SystemTotals(mode, 3.0 if mode == "single_phase_equiv" else 1.0)
        assert st.node_voltage_ll_v(split) == pytest.approx(440.0, rel=1e-12)
        assert st.node_voltage_ll_v(two_of_three) == pytest.approx(400.0, rel=1e-12)
        assert st.node_voltage_ll_v(split_n) == pytest.approx(440.0, rel=1e-12)
        assert st.node_voltage_ll_v(one) == pytest.approx(230.0 * math.sqrt(3.0))
        assert st.node_voltage_ll_v(plain) == pytest.approx(230.0 * math.sqrt(3.0))


def test_iec_device_caps_use_the_split_phase_neutral_voltage():
    """A one-phase device on X1-N of a 440/220 V split-phase node draws its fundamental
    at 220 V (Node.u_rated_ln_v), not 440 V / sqrt(3)."""
    from pgml.schemas.grid_schema import Grid, Node, Phase

    def grid(u_ln):
        node = Node(
            id=1,
            u_rated_v=440.0,
            u_rated_ln_v=u_ln,
            phases=(Phase.A, Phase.B, Phase.N),
        )
        load = Load(
            id=10,
            node=1,
            phases=(Phase.A, Phase.N),
            p_nom_w=900.0,
            q_nom_var=0.0,
            consumer_type=ConsumerType.OFFICE,
        )
        return Grid(
            nodes=[node],
            appliances=[load],
            metadata={"tags": {PHASE_MODE_TAG: "as_built"}},
        )

    split = iec61000_3_2_device_caps(grid(220.0), [10], [7])[10][7]
    rule = iec61000_3_2_device_caps(grid(None), [10], [7])[10][7]
    # Class A limit [A] over the fundamental P / U_LN: the cap scales with U_LN.
    assert split == pytest.approx(0.77 / (900.0 / 220.0), rel=1e-12)
    assert rule == pytest.approx(0.77 / (900.0 / (440.0 / math.sqrt(3.0))), rel=1e-12)
