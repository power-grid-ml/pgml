"""Increment 0: Load/Generator `connection` field + validators.

`connection` is now Optional (None => resolve from config). DELTA is line-to-line
(needs >= 2 phases); ZIGZAG is transformer-only and rejected on appliances. See
`docs/pgml/modeling/asymmetric.md`.
"""

from __future__ import annotations

import pytest

from pgml.schemas.grid_schema import Generator, Load, Phase, WindingConnection

ABC = (Phase.A, Phase.B, Phase.C)


def test_connection_defaults_to_none_not_wye():
    """Default is None so the config default applies at assembly (not a baked WYE)."""
    assert Load(id=1, node=1, phases=ABC, p_nom_w=3000.0).connection is None
    assert Generator(id=1, node=1, phases=ABC, p_nom_w=3000.0).connection is None


def test_explicit_wye_and_delta_three_phase_ok():
    assert (
        Load(
            id=1, node=1, phases=ABC, p_nom_w=3000.0, connection=WindingConnection.WYE
        ).connection
        == WindingConnection.WYE
    )
    assert (
        Load(
            id=2,
            node=1,
            phases=(Phase.A, Phase.B),
            p_nom_w=2000.0,
            connection=WindingConnection.DELTA,
        ).connection
        == WindingConnection.DELTA
    )


def test_delta_single_phase_rejected():
    with pytest.raises(ValueError, match="DELTA requires at least 2 phases"):
        Load(
            id=1,
            node=1,
            phases=(Phase.A,),
            p_nom_w=1000.0,
            connection=WindingConnection.DELTA,
        )


@pytest.mark.parametrize(
    "conn", [WindingConnection.ZIGZAG, WindingConnection.ZIGZAG_GROUNDED]
)
def test_zigzag_rejected_on_appliances(conn):
    with pytest.raises(ValueError, match="zigzag"):
        Load(id=1, node=1, phases=ABC, p_nom_w=3000.0, connection=conn)
    with pytest.raises(ValueError, match="zigzag"):
        Generator(id=1, node=1, phases=ABC, p_nom_w=3000.0, connection=conn)
