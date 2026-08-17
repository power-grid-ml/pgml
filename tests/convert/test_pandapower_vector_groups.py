"""Unit tests for the pandapower converter's vector-group parsing and tap-changer
math (``pgml.convert.pandapower.converter``).

Covers, in isolation from any pandapower network object:

- ``_parse_vector_group`` — every IEC vector-group string form named in the
  converter's contract (``Dyn5``, ``YNd5``, ``Yzn5``, ``Yy0``, ``YNyn0``, ``Dd0``,
  ``Dyn11``), plus malformed-string rejection.
- ``_vector_group_string`` — the ``net.trafo['vector_group']`` column vs
  ``net.std_types['trafo'][std_type]`` catalog lookup precedence.
- ``_resolve_transformer_connections`` — the vector-group-vs-``shift_degree``
  consistency check, and the shift-parity fallback when no vector-group string
  exists anywhere.
- ``_tap_ratio_magnitude`` — the tap-changer delta formula, NaN-safe missing-field
  handling, and the ``tap_step_degree``/``tap_phase_shifter`` rejection.
"""

from __future__ import annotations

import pytest

from pgml.convert.pandapower.converter import (
    _parse_vector_group,
    _resolve_transformer_connections,
    _tap_ratio_magnitude,
    _vector_group_string,
)
from pgml.errors import ConversionError
from pgml.schemas.grid_schema import WindingConnection

W = WindingConnection


# ----------------------------------------------------------------------- #
# _parse_vector_group
# ----------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "vg_str,expected_from,expected_to,expected_clock",
    [
        ("Dyn5", W.DELTA, W.WYE_GROUNDED, 5),
        ("YNd5", W.WYE_GROUNDED, W.DELTA, 5),
        ("Yzn5", W.WYE, W.ZIGZAG_GROUNDED, 5),
        ("Yy0", W.WYE, W.WYE, 0),
        ("YNyn0", W.WYE_GROUNDED, W.WYE_GROUNDED, 0),
        ("Dd0", W.DELTA, W.DELTA, 0),
        ("Dyn11", W.DELTA, W.WYE_GROUNDED, 11),
        ("Yy6", W.WYE, W.WYE, 6),
        ("Dz5", W.DELTA, W.ZIGZAG, 5),
        ("Zy1", W.ZIGZAG, W.WYE, 1),
    ],
)
def test_parse_vector_group_forms(vg_str, expected_from, expected_to, expected_clock):
    from_conn, to_conn, clock = _parse_vector_group(vg_str)
    assert from_conn == expected_from
    assert to_conn == expected_to
    assert clock == expected_clock


def test_parse_vector_group_case_robustness():
    """Matching is by exact token, case-insensitive -- not literal HV/LV case."""
    from_conn, to_conn, clock = _parse_vector_group("dyn5")
    assert (from_conn, to_conn, clock) == (W.DELTA, W.WYE_GROUNDED, 5)
    from_conn, to_conn, clock = _parse_vector_group("DYN5")
    assert (from_conn, to_conn, clock) == (W.DELTA, W.WYE_GROUNDED, 5)
    from_conn, to_conn, clock = _parse_vector_group("YND5")
    assert (from_conn, to_conn, clock) == (W.WYE_GROUNDED, W.DELTA, 5)


def test_parse_vector_group_strips_whitespace():
    from_conn, to_conn, clock = _parse_vector_group("  Dyn5  ")
    assert (from_conn, to_conn, clock) == (W.DELTA, W.WYE_GROUNDED, 5)


@pytest.mark.parametrize("bad", ["5Dyn", "", "Dxn5", "Dynn5", "D5yn"])
def test_parse_vector_group_rejects_malformed(bad):
    with pytest.raises(ConversionError):
        _parse_vector_group(bad)


def test_parse_vector_group_bare_form_no_clock():
    """A clock-less vector_group ('Dyn', 'Yzn') is the form pandapower's own
    runpp_3ph zero-sequence transformer model requires (it explicitly rejects a
    digit-suffixed string); parses to clock=None."""
    assert _parse_vector_group("Dyn") == (W.DELTA, W.WYE_GROUNDED, None)
    assert _parse_vector_group("Yzn") == (W.WYE, W.ZIGZAG_GROUNDED, None)
    assert _parse_vector_group("YNyn") == (W.WYE_GROUNDED, W.WYE_GROUNDED, None)


# ----------------------------------------------------------------------- #
# _vector_group_string precedence
# ----------------------------------------------------------------------- #
class _NetStub:
    """Minimal ``net``-like stand-in exposing only ``std_types``."""

    def __init__(self, std_types=None):
        self.std_types = std_types or {}


def test_vector_group_string_row_column_wins():
    net = _NetStub({"trafo": {"some_type": {"vector_group": "Yy0"}}})
    row = {"vector_group": "Dyn5", "std_type": "some_type"}
    assert _vector_group_string(net, row) == "Dyn5"


def test_vector_group_string_falls_back_to_std_type_catalog():
    net = _NetStub({"trafo": {"0.4 MVA 10/0.4 kV": {"vector_group": "Dyn5"}}})
    row = {"vector_group": None, "std_type": "0.4 MVA 10/0.4 kV"}
    assert _vector_group_string(net, row) == "Dyn5"


def test_vector_group_string_nan_row_value_falls_back_to_std_type():
    net = _NetStub({"trafo": {"t1": {"vector_group": "YNd5"}}})
    row = {"vector_group": float("nan"), "std_type": "t1"}
    assert _vector_group_string(net, row) == "YNd5"


def test_vector_group_string_none_when_neither_present():
    net = _NetStub({})
    row = {"vector_group": None, "std_type": None}
    assert _vector_group_string(net, row) is None

    row2 = {"vector_group": None, "std_type": "unknown_type"}
    assert _vector_group_string(net, row2) is None


def test_vector_group_string_missing_columns_entirely():
    """A row/net with no vector_group or std_type columns at all -> None."""
    net = _NetStub(None)
    row = {}
    assert _vector_group_string(net, row) is None


# ----------------------------------------------------------------------- #
# _resolve_transformer_connections
# ----------------------------------------------------------------------- #
def test_resolve_connections_from_matching_vector_group():
    net = _NetStub({})
    row = {"vector_group": "Dyn5", "std_type": None}
    from_conn, to_conn = _resolve_transformer_connections(net, row, shift_deg=150.0)
    assert (from_conn, to_conn) == (W.DELTA, W.WYE_GROUNDED)


def test_resolve_connections_yzn5_from_std_type():
    net = _NetStub({"trafo": {"0.25 MVA 20/0.4 kV": {"vector_group": "Yzn5"}}})
    row = {"vector_group": None, "std_type": "0.25 MVA 20/0.4 kV"}
    from_conn, to_conn = _resolve_transformer_connections(net, row, shift_deg=150.0)
    assert (from_conn, to_conn) == (W.WYE, W.ZIGZAG_GROUNDED)


def test_resolve_connections_mismatch_raises():
    """vector_group clock disagrees with shift_degree -> loud ConversionError,
    never a silent preference of one source over the other."""
    net = _NetStub({})
    row = {"vector_group": "Dyn5", "std_type": None}  # clock 5 -> 150 deg
    with pytest.raises(ConversionError, match="self-inconsistent"):
        _resolve_transformer_connections(net, row, shift_deg=30.0)  # clock 1


def test_resolve_connections_fallback_even_clock_wye_grounded():
    """No vector-group string anywhere; shift_degree=0 (even clock) -> WYN/WYN."""
    net = _NetStub({})
    row = {"vector_group": None, "std_type": None}
    from_conn, to_conn = _resolve_transformer_connections(net, row, shift_deg=0.0)
    assert (from_conn, to_conn) == (W.WYE_GROUNDED, W.WYE_GROUNDED)


def test_resolve_connections_fallback_odd_clock_dyn():
    """No vector-group string anywhere; shift_degree=30 (odd clock) -> Dyn."""
    net = _NetStub({})
    row = {"vector_group": None, "std_type": None}
    from_conn, to_conn = _resolve_transformer_connections(net, row, shift_deg=30.0)
    assert (from_conn, to_conn) == (W.DELTA, W.WYE_GROUNDED)


def test_resolve_connections_fallback_non_clock_shift_phase_shifter():
    """A non-multiple-of-30 shift (MATPOWER ideal phase shifter) -> WYN/WYN,
    passed through exactly (verified by the caller via tap.shift_deg)."""
    net = _NetStub({})
    row = {"vector_group": None, "std_type": None}
    from_conn, to_conn = _resolve_transformer_connections(net, row, shift_deg=12.5)
    assert (from_conn, to_conn) == (W.WYE_GROUNDED, W.WYE_GROUNDED)


def test_resolve_connections_bare_vector_group_combines_with_shift():
    """A bare vector_group (no clock) skips the cross-check and combines with
    shift_degree directly (runpp_3ph's own required form)."""
    net = _NetStub({})
    row = {"vector_group": "Dyn", "std_type": None}
    from_conn, to_conn = _resolve_transformer_connections(net, row, shift_deg=150.0)
    assert (from_conn, to_conn) == (W.DELTA, W.WYE_GROUNDED)


def test_resolve_connections_case118_style_zero_shift():
    """MATPOWER-imported nets (e.g. case118): std_type is None, no vector_group
    column, shift_degree=0.0 for the un-phase-shifted units."""
    net = _NetStub({})
    row = {"vector_group": None, "std_type": None}
    from_conn, to_conn = _resolve_transformer_connections(net, row, shift_deg=0.0)
    assert (from_conn, to_conn) == (W.WYE_GROUNDED, W.WYE_GROUNDED)


# ----------------------------------------------------------------------- #
# _tap_ratio_magnitude
# ----------------------------------------------------------------------- #
def test_tap_ratio_no_tap_changer_columns_missing():
    assert _tap_ratio_magnitude({}) == pytest.approx(1.0)


def test_tap_ratio_no_tap_changer_nan_fields():
    row = {
        "tap_pos": float("nan"),
        "tap_neutral": 0.0,
        "tap_step_percent": 2.5,
        "tap_side": "hv",
    }
    assert _tap_ratio_magnitude(row) == pytest.approx(1.0)


def test_tap_ratio_hv_side_positive_pos_raises_ratio():
    """tap_pos=+2 on the HV side: ratio_magnitude = 1 + delta (LOWERS the LV
    voltage; verified against a live pandapower runpp)."""
    row = {
        "tap_pos": 2.0,
        "tap_neutral": 0.0,
        "tap_step_percent": 2.5,
        "tap_side": "hv",
    }
    assert _tap_ratio_magnitude(row) == pytest.approx(1.05)


def test_tap_ratio_hv_side_negative_pos_lowers_ratio():
    row = {
        "tap_pos": -2.0,
        "tap_neutral": 0.0,
        "tap_step_percent": 2.5,
        "tap_side": "hv",
    }
    assert _tap_ratio_magnitude(row) == pytest.approx(0.95)


def test_tap_ratio_lv_side_positive_pos_lowers_ratio():
    """tap_pos=+2 on the LV side: ratio_magnitude = 1 / (1 + delta) (RAISES the
    LV voltage; the opposite sense of the HV-side tap)."""
    row = {
        "tap_pos": 2.0,
        "tap_neutral": 0.0,
        "tap_step_percent": 2.5,
        "tap_side": "lv",
    }
    assert _tap_ratio_magnitude(row) == pytest.approx(1.0 / 1.05)


def test_tap_ratio_case118_style_min1_step_1p5_percent():
    """case118's HV-side taps: tap_pos=-1, tap_neutral=0, tap_step_percent in
    {1.5, 4.0, 6.5}."""
    for step_pct, expected in [
        (1.5, 1.0 - 0.015),
        (4.0, 1.0 - 0.04),
        (6.5, 1.0 - 0.065),
    ]:
        row = {
            "tap_pos": -1.0,
            "tap_neutral": 0.0,
            "tap_step_percent": step_pct,
            "tap_side": "hv",
        }
        assert _tap_ratio_magnitude(row) == pytest.approx(expected)


def test_tap_ratio_neutral_offset():
    """delta uses (tap_pos - tap_neutral), not tap_pos alone."""
    row = {
        "tap_pos": 3.0,
        "tap_neutral": 1.0,
        "tap_step_percent": 2.5,
        "tap_side": "hv",
    }
    # delta = (3-1)*2.5/100 = 0.05
    assert _tap_ratio_magnitude(row) == pytest.approx(1.05)


def test_tap_ratio_step_degree_rejected():
    row = {
        "tap_pos": 1.0,
        "tap_neutral": 0.0,
        "tap_step_percent": 1.5,
        "tap_side": "hv",
        "tap_step_degree": 10.0,
    }
    with pytest.raises(ConversionError, match="phase-shifter"):
        _tap_ratio_magnitude(row)


def test_tap_ratio_step_degree_zero_is_fine():
    row = {
        "tap_pos": 1.0,
        "tap_neutral": 0.0,
        "tap_step_percent": 1.5,
        "tap_side": "hv",
        "tap_step_degree": 0.0,
    }
    assert _tap_ratio_magnitude(row) == pytest.approx(1.015)


def test_tap_ratio_phase_shifter_flag_rejected():
    row = {
        "tap_pos": 1.0,
        "tap_neutral": 0.0,
        "tap_step_percent": 1.5,
        "tap_side": "hv",
        "tap_phase_shifter": True,
    }
    with pytest.raises(ConversionError, match="phase-shifter"):
        _tap_ratio_magnitude(row)


def test_tap_ratio_unknown_side_raises():
    row = {
        "tap_pos": 1.0,
        "tap_neutral": 0.0,
        "tap_step_percent": 1.5,
        "tap_side": "mv",
    }
    with pytest.raises(ConversionError, match="tap_side"):
        _tap_ratio_magnitude(row)


def test_tap_ratio_missing_tap_side_returns_neutral():
    row = {
        "tap_pos": 1.0,
        "tap_neutral": 0.0,
        "tap_step_percent": 1.5,
        "tap_side": None,
    }
    assert _tap_ratio_magnitude(row) == pytest.approx(1.0)
