"""Schema construction validators: mis-sized matrices fail LOUD, at build time.

A wrong-dimension per-phase matrix must never survive to assembly (where it
would surface as an opaque shape error, or worse, a silent broadcast).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pgml.schemas import SCHEMA_VERSION
from pgml.schemas.grid_schema import ComplexTap, GenericBranch, Phase, ShuntReactor

ABC = (Phase.A, Phase.B, Phase.C)


def test_schema_version_is_current():
    assert SCHEMA_VERSION == "0.0.2"


def test_shunt_reactor_rejects_mis_sized_matrix():
    with pytest.raises(ValidationError, match="3x3"):
        ShuntReactor(
            id=1,
            from_node=1,
            to_node=1,
            from_phases=ABC,
            to_phases=ABC,
            conductance_s=[[1.0e-4, 0.0], [0.0, 1.0e-4]],  # 2x2 on a 3-phase shunt
            capacitance_f=[
                [1.0e-7 if i == j else 0.0 for j in range(3)] for i in range(3)
            ],
        )


def test_generic_branch_rejects_mis_sized_matrix():
    ok = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    with pytest.raises(ValidationError, match="3x3"):
        GenericBranch(
            id=1,
            from_node=1,
            to_node=2,
            from_phases=ABC,
            to_phases=ABC,
            series_resistance_ohm=ok,
            series_inductance_h=[[1.0e-4]],  # 1x1 on a 3-phase branch
        )


def test_generic_branch_rejects_phase_length_mismatch():
    ok = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    with pytest.raises(ValidationError, match="equal length"):
        GenericBranch(
            id=1,
            from_node=1,
            to_node=2,
            from_phases=ABC,
            to_phases=(Phase.A,),
            series_resistance_ohm=ok,
            series_inductance_h=ok,
        )


def test_complex_tap_shift_deg_is_plain_float():
    """The clock selector is not tensor-capable: whatever goes in, the STORED
    value is a plain python float (pydantic coerces at construction), so no
    autograd graph can reach the discrete clock selection downstream."""
    torch = pytest.importorskip("torch")
    tap = ComplexTap(ratio_magnitude=1.0, shift_deg=30)
    assert isinstance(tap.shift_deg, float)
    coerced = ComplexTap(ratio_magnitude=1.0, shift_deg=torch.tensor(30.0))
    assert isinstance(coerced.shift_deg, float) and coerced.shift_deg == 30.0
