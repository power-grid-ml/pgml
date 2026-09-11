"""Unit tests for the shared converter scaffold ``pgml.convert._common``.

Every public helper is exercised directly: id allocation, the phase decision,
the sequence->phase symmetric-component identity, zero-sequence defaulting from
config, the Thevenin math, and the schema emit helpers' phase/connection wiring.
"""

from __future__ import annotations

import math

import pytest

from pgml import defaults as config
from pgml.convert._common import (
    IdCounter,
    PhaseMode,
    build_line_from_matrices,
    build_line_from_sequence,
    build_load,
    build_node,
    build_source,
    phases_for,
    sequence_to_phase_matrices,
    single_phase_matrix,
    source_zero_sequence_ratios,
    thevenin_from_sk,
    thevenin_from_z,
    zero_sequence_ratios,
)
from pgml import defaults
from pgml.schemas.grid_schema import Phase, WindingConnection

ABC = (Phase.A, Phase.B, Phase.C)
TWO_PI_F0 = 2.0 * math.pi * 50.0


# ----------------------------------------------------------------------- #
# IdCounter
# ----------------------------------------------------------------------- #
def test_id_counter_is_monotonic_from_one():
    c = IdCounter()
    assert [c.next() for _ in range(4)] == [1, 2, 3, 4]


def test_id_counter_instances_are_independent():
    a, b = IdCounter(), IdCounter()
    assert a.next() == 1
    assert a.next() == 2
    assert b.next() == 1  # second counter unaffected


# ----------------------------------------------------------------------- #
# phases_for
# ----------------------------------------------------------------------- #
def test_phases_single_phase_equiv():
    assert phases_for(PhaseMode.SINGLE_PHASE_EQUIV) == (Phase.A,)
    # native is ignored for the single-phase equivalent
    assert phases_for(PhaseMode.SINGLE_PHASE_EQUIV, native=ABC) == (Phase.A,)


def test_phases_three_phase_default_abc():
    assert phases_for(PhaseMode.THREE_PHASE) == ABC


def test_phases_three_phase_native_passthrough():
    native = (Phase.A, Phase.B, Phase.C, Phase.N)
    assert phases_for(PhaseMode.THREE_PHASE, native=native) == native


# ----------------------------------------------------------------------- #
# zero_sequence_ratios / sequence_to_phase_matrices
# ----------------------------------------------------------------------- #
def test_zero_sequence_ratios_match_config():
    r, x, c = zero_sequence_ratios()
    assert r == config.get("line.zero_sequence.r0_over_r1")
    assert x == config.get("line.zero_sequence.x0_over_x1")
    assert c == config.get("line.zero_sequence.c0_over_c1")


def test_sequence_to_phase_identity_self_and_mutual():
    # Explicit sequence values -> hand-computed self/mutual.
    r1, x1, c1 = 0.1, 0.2, 1e-9
    r0, x0, c0 = 0.4, 0.6, 0.5e-9
    R, L, C, G = sequence_to_phase_matrices(
        r1, x1, c1, r0=r0, x0=x0, c0=c0, two_pi_f0=TWO_PI_F0
    )

    def self_mut(q0, q1):
        return (q0 + 2 * q1) / 3.0, (q0 - q1) / 3.0

    r_self, r_mut = self_mut(r0, r1)
    assert R[0][0] == pytest.approx(r_self)
    assert R[0][1] == pytest.approx(r_mut)
    assert R[1][0] == pytest.approx(r_mut)  # symmetric
    # off-diagonal == mutual everywhere
    for i in range(3):
        for j in range(3):
            assert R[i][j] == pytest.approx(r_self if i == j else r_mut)

    # L derived from the per-phase reactance matrix / two_pi_f0
    x_self, x_mut = self_mut(x0, x1)
    assert L[0][0] == pytest.approx(x_self / TWO_PI_F0)
    assert L[0][1] == pytest.approx(x_mut / TWO_PI_F0)

    c_self, c_mut = self_mut(c0, c1)
    assert C[0][0] == pytest.approx(c_self)
    assert C[0][1] == pytest.approx(c_mut)
    # no conductance supplied -> all zeros
    assert all(G[i][j] == 0.0 for i in range(3) for j in range(3))


def test_balanced_sequence_gives_symmetric_circulant():
    # When Z0 == Z1 the mutual term vanishes -> pure diagonal (decoupled phases).
    r1 = x1 = c1 = 0.3
    R, L, C, _ = sequence_to_phase_matrices(
        r1, x1, c1, r0=r1, x0=x1, c0=c1, two_pi_f0=TWO_PI_F0
    )
    for i in range(3):
        for j in range(3):
            if i != j:
                assert R[i][j] == pytest.approx(0.0)
                assert C[i][j] == pytest.approx(0.0)
            else:
                assert R[i][i] == pytest.approx(r1)


def test_zero_sequence_defaulting_from_config():
    # r0/x0/c0 None -> defaulted to r1*ratio etc.
    rr, xr, cr = zero_sequence_ratios()
    r1, x1, c1 = 0.1, 0.2, 2e-9
    R, L, C, _ = sequence_to_phase_matrices(r1, x1, c1, two_pi_f0=TWO_PI_F0)

    r0, x0, c0 = r1 * rr, x1 * xr, c1 * cr

    def self_mut(q0, q1):
        return (q0 + 2 * q1) / 3.0, (q0 - q1) / 3.0

    assert R[0][0] == pytest.approx(self_mut(r0, r1)[0])
    assert R[0][1] == pytest.approx(self_mut(r0, r1)[1])
    assert C[0][0] == pytest.approx(self_mut(c0, c1)[0])
    # Inductance defaults via x0 = x1 * ratio; L = X / (2*pi*f0).
    l1, l0 = x1 / TWO_PI_F0, x0 / TWO_PI_F0
    assert L[0][0] == pytest.approx(self_mut(l0, l1)[0])
    assert L[0][1] == pytest.approx(self_mut(l0, l1)[1])


# ----------------------------------------------------------------------- #
# single_phase_matrix
# ----------------------------------------------------------------------- #
def test_single_phase_matrix_exact():
    assert single_phase_matrix(0.123) == [[0.123]]


# ----------------------------------------------------------------------- #
# thevenin helpers
# ----------------------------------------------------------------------- #
def test_thevenin_from_z_basic():
    r, ind = thevenin_from_z(2.0, TWO_PI_F0 * 1e-3, TWO_PI_F0)
    assert r == pytest.approx(2.0)
    assert ind == pytest.approx(1e-3)


def test_thevenin_from_z_floors_zero():
    r, ind = thevenin_from_z(0.0, 0.0, TWO_PI_F0)
    assert r > 0.0
    assert ind > 0.0


def test_thevenin_from_sk_math():
    u, sk, rx = 10000.0, 5e8, 0.1
    r, ind = thevenin_from_sk(u, sk, rx, TWO_PI_F0)
    z_mag = u**2 / sk
    x_s = z_mag / math.sqrt(1 + rx**2)
    assert r == pytest.approx(x_s * rx)
    assert ind == pytest.approx(x_s / TWO_PI_F0)


def test_thevenin_from_sk_fallback_on_huge_sk():
    r, ind = thevenin_from_sk(10000.0, 1e16, 0.1, TWO_PI_F0)
    assert (r, ind) == (1.0e-6, 1.0e-12)


# ----------------------------------------------------------------------- #
# build_node
# ----------------------------------------------------------------------- #
def test_build_node_single_phase():
    n = build_node(id=1, u_rated_v=400.0, mode=PhaseMode.SINGLE_PHASE_EQUIV)
    assert n.phases == (Phase.A,)
    assert n.u_rated_v == 400.0


def test_build_node_three_phase():
    n = build_node(id=1, u_rated_v=400.0, mode=PhaseMode.THREE_PHASE)
    assert n.phases == ABC


def test_build_node_native_phases():
    native = (Phase.A, Phase.B, Phase.C, Phase.N)
    n = build_node(
        id=1, u_rated_v=400.0, mode=PhaseMode.THREE_PHASE, native_phases=native
    )
    assert n.phases == native


# ----------------------------------------------------------------------- #
# build_load
# ----------------------------------------------------------------------- #
def test_build_load_single_phase_total_only():
    ld = build_load(
        id=1,
        node=2,
        mode=PhaseMode.SINGLE_PHASE_EQUIV,
        p_total_w=1000.0,
        q_total_var=200.0,
    )
    assert ld.phases == (Phase.A,)
    assert ld.p_nom_w == 1000.0
    assert ld.q_nom_var == 200.0
    assert ld.p_nom_per_phase_w is None
    assert ld.connection is None  # default not set -> resolves from config


def test_build_load_single_phase_ignores_per_phase_and_connection():
    # Under single-phase equiv the per-phase split / connection are not threaded.
    ld = build_load(
        id=1,
        node=2,
        mode=PhaseMode.SINGLE_PHASE_EQUIV,
        p_total_w=900.0,
        q_total_var=0.0,
        connection=WindingConnection.DELTA,
        p_per_phase_w=(300.0, 300.0, 300.0),
    )
    assert ld.phases == (Phase.A,)
    assert ld.p_nom_per_phase_w is None
    assert ld.connection is None


def test_build_load_three_phase_per_phase_and_connection():
    ld = build_load(
        id=1,
        node=2,
        mode=PhaseMode.THREE_PHASE,
        p_total_w=600.0,
        q_total_var=30.0,
        connection=WindingConnection.WYE,
        p_per_phase_w=(100.0, 200.0, 300.0),
        q_per_phase_var=(10.0, 10.0, 10.0),
    )
    assert ld.phases == ABC
    assert ld.connection == WindingConnection.WYE
    assert tuple(ld.p_nom_per_phase_w) == (100.0, 200.0, 300.0)
    assert tuple(ld.q_nom_per_phase_var) == (10.0, 10.0, 10.0)


def test_build_load_three_phase_balanced_no_per_phase():
    ld = build_load(
        id=1,
        node=2,
        mode=PhaseMode.THREE_PHASE,
        p_total_w=600.0,
        q_total_var=0.0,
    )
    assert ld.phases == ABC
    assert ld.connection is None  # resolves from config -> WYE
    assert ld.p_nom_per_phase_w is None


def test_build_load_native_single_phase_under_three_phase():
    # A genuine 1-phase asymmetric load keeps its single native phase.
    ld = build_load(
        id=1,
        node=2,
        mode=PhaseMode.THREE_PHASE,
        p_total_w=100.0,
        q_total_var=0.0,
        connection=WindingConnection.WYE,
        native_phases=(Phase.A,),
        p_per_phase_w=(100.0,),
        q_per_phase_var=(0.0,),
    )
    assert ld.phases == (Phase.A,)


# ----------------------------------------------------------------------- #
# build_source
# ----------------------------------------------------------------------- #
def test_build_source_single_phase():
    s = build_source(
        id=1,
        node=2,
        mode=PhaseMode.SINGLE_PHASE_EQUIV,
        u_ref_v=231.0,
        u_angle_deg=0.0,
        r_ohm=1e-6,
        l_h=1e-12,
    )
    assert s.phases == (Phase.A,)
    assert s.u_ref_v == (231.0,)
    assert s.u_angle_deg == (0.0,)
    assert s.resistance_ohm == [[1e-6]]


def test_build_source_three_phase_balanced_angles():
    # ``u_ref_v`` is the LINE-TO-LINE magnitude; under THREE_PHASE it is converted to
    # the per-phase line-to-neutral phase-to-ground EMF (÷sqrt(3)) for the wye source.
    s = build_source(
        id=1,
        node=2,
        mode=PhaseMode.THREE_PHASE,
        u_ref_v=400.0,
        u_angle_deg=0.0,
        r_ohm=0.01,
        l_h=1e-5,
        two_pi_f0=TWO_PI_F0,
    )
    u_ln = 400.0 / math.sqrt(3.0)
    assert s.phases == ABC
    assert s.u_ref_v == pytest.approx((u_ln, u_ln, u_ln))
    assert s.u_angle_deg == (0.0, -120.0, -240.0)
    # The default source zero-sequence ratios are 1.0 (Z0 = Z1), so the mutual term
    # vanishes and the Thevenin stays the plain diagonal stamp, unrounded.
    assert s.resistance_ohm == [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]
    assert s.inductance_h == [[1e-5, 0, 0], [0, 1e-5, 0], [0, 0, 1e-5]]


def test_build_source_three_phase_zero_sequence_split():
    """Explicit R0/X0 become the symmetric-component self/mutual terms."""
    r1, x1 = 0.01, 0.04
    r0, x0 = 0.05, 0.30
    s = build_source(
        id=1,
        node=2,
        mode=PhaseMode.THREE_PHASE,
        u_ref_v=400.0,
        u_angle_deg=0.0,
        r_ohm=r1,
        l_h=x1 / TWO_PI_F0,
        r0_ohm=r0,
        x0_ohm=x0,
        two_pi_f0=TWO_PI_F0,
    )
    r_self, r_mut = (r0 + 2.0 * r1) / 3.0, (r0 - r1) / 3.0
    x_self, x_mut = (x0 + 2.0 * x1) / 3.0, (x0 - x1) / 3.0
    for i in range(3):
        for j in range(3):
            assert s.resistance_ohm[i][j] == pytest.approx(r_self if i == j else r_mut)
            assert s.inductance_h[i][j] * TWO_PI_F0 == pytest.approx(
                x_self if i == j else x_mut
            )


def test_build_source_three_phase_defaults_warn(caplog):
    """Falling back to the configured zero-sequence ratio names the element."""
    with caplog.at_level("WARNING"):
        build_source(
            id=7,
            node=2,
            mode=PhaseMode.THREE_PHASE,
            u_ref_v=400.0,
            u_angle_deg=0.0,
            r_ohm=0.01,
            l_h=1e-5,
            two_pi_f0=TWO_PI_F0,
            element="test source",
        )
    assert any(
        "test source" in r.message and "zero-sequence" in r.message
        for r in caplog.records
    )


def test_build_source_single_phase_ignores_zero_sequence():
    """A positive-sequence equivalent has no zero sequence: R0/X0 are not read."""
    s = build_source(
        id=1,
        node=2,
        mode=PhaseMode.SINGLE_PHASE_EQUIV,
        u_ref_v=231.0,
        u_angle_deg=0.0,
        r_ohm=0.01,
        l_h=1e-5,
        r0_ohm=5.0,
        x0_ohm=9.0,
        two_pi_f0=TWO_PI_F0,
    )
    assert s.resistance_ohm == [[0.01]]
    assert s.inductance_h == [[1e-5]]


def test_source_zero_sequence_ratios_are_documented_defaults():
    r0_over_r1, x0_over_x1 = source_zero_sequence_ratios()
    assert r0_over_r1 > 0.0 and x0_over_x1 > 0.0
    assert "zero-sequence" in defaults.describe("source.zero_sequence.r0_over_r1")


# ----------------------------------------------------------------------- #
# build_line_from_matrices / build_line_from_sequence
# ----------------------------------------------------------------------- #
def test_build_line_from_matrices_phases_and_shape():
    ln = build_line_from_matrices(
        id=1,
        from_node=1,
        to_node=2,
        phases=ABC,
        length_m=10.0,
        r_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        l_matrix=[[1e-6, 0, 0], [0, 1e-6, 0], [0, 0, 1e-6]],
        c_matrix=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    )
    assert ln.from_phases == ABC
    assert ln.to_phases == ABC
    assert ln.length_m == 10.0
    assert ln.shunt_conductance_s_per_m is None


def test_build_line_from_sequence_single_phase_exact():
    r1, x1, c1 = 0.5, 0.3, 2e-9
    ln = build_line_from_sequence(
        id=1,
        from_node=1,
        to_node=2,
        mode=PhaseMode.SINGLE_PHASE_EQUIV,
        length_m=100.0,
        r1=r1,
        x1=x1,
        c1=c1,
        two_pi_f0=TWO_PI_F0,
    )
    assert ln.from_phases == (Phase.A,)
    assert ln.series_resistance_ohm_per_m == [[r1]]
    assert ln.series_inductance_h_per_m == [[x1 / TWO_PI_F0]]
    assert ln.shunt_capacitance_f_per_m == [[c1]]
    assert ln.shunt_conductance_s_per_m is None  # g1==0 -> None


def test_build_line_from_sequence_single_phase_conductance():
    ln = build_line_from_sequence(
        id=1,
        from_node=1,
        to_node=2,
        mode=PhaseMode.SINGLE_PHASE_EQUIV,
        length_m=1.0,
        r1=0.1,
        x1=0.1,
        c1=1e-9,
        two_pi_f0=TWO_PI_F0,
        g1=3e-7,
    )
    assert ln.shunt_conductance_s_per_m == [[3e-7]]


def test_build_line_from_sequence_three_phase_matrix():
    r1, x1, c1 = 0.1, 0.2, 1e-9
    ln = build_line_from_sequence(
        id=1,
        from_node=1,
        to_node=2,
        mode=PhaseMode.THREE_PHASE,
        length_m=1.0,
        r1=r1,
        x1=x1,
        c1=c1,
        two_pi_f0=TWO_PI_F0,
        r0=0.4,
        x0=0.6,
        c0=0.5e-9,
    )
    assert ln.from_phases == ABC
    R = ln.series_resistance_ohm_per_m
    assert len(R) == 3 and all(len(r) == 3 for r in R)
    r_self = (0.4 + 2 * r1) / 3.0
    r_mut = (0.4 - r1) / 3.0
    assert R[0][0] == pytest.approx(r_self)
    assert R[0][1] == pytest.approx(r_mut)
