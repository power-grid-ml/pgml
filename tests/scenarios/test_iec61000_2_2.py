"""IEC 61000-2-2 / -2-12 compatibility levels: table values and the band levels."""

from __future__ import annotations

import math

import pytest

from pgml.scenarios import (
    iec61000_2_2_band_level_v,
    iec61000_2_2_harmonic_levels,
    iec61000_2_2_provenance,
    iec61000_2_2_thd,
)


def test_low_voltage_harmonic_table():
    lv = iec61000_2_2_harmonic_levels("lv")
    assert min(lv) == 2 and max(lv) == 40
    # Explicit entries of Table 1.
    for h, pct in {
        2: 2,
        3: 5,
        4: 1,
        5: 6,
        6: 0.5,
        7: 5,
        8: 0.5,
        9: 1.5,
        11: 3.5,
        13: 3,
        15: 0.4,
        21: 0.3,
    }.items():
        assert lv[h] == pytest.approx(pct / 100)
    # Range rules: 2.27 (17/h) - 0.27 for odd non-triplen 17..37, 0.2 % for triplen
    # above 21, 0.25 (10/h) + 0.25 for even 10..40.
    assert lv[17] == pytest.approx(0.02)
    assert lv[37] == pytest.approx((2.27 * 17 / 37 - 0.27) / 100)
    assert lv[27] == lv[39] == pytest.approx(0.002)
    assert lv[10] == pytest.approx(0.005) and lv[40] == pytest.approx(0.003125)
    assert (
        iec61000_2_2_thd("lv") == 0.08
        and iec61000_2_2_thd("lv", very_short_term=True) == 0.11
    )


def test_medium_voltage_extends_to_order_50():
    mv = iec61000_2_2_harmonic_levels("mv")
    lv = iec61000_2_2_harmonic_levels("lv")
    assert max(mv) == 50 and 49 in mv and 45 in mv
    assert all(mv[h] == pytest.approx(lv[h]) for h in lv)
    assert mv[49] == pytest.approx((2.27 * 17 / 49 - 0.27) / 100)


def test_supraharmonic_band_levels():
    u1 = 230.0
    assert iec61000_2_2_band_level_v(1900.0, u1) is None
    assert iec61000_2_2_band_level_v(2100.0, u1) == pytest.approx(0.014 * u1)
    assert iec61000_2_2_band_level_v(3000.0, u1) == pytest.approx(0.014 * u1)
    # 3-9 kHz: log level linear in log f, 1.4 % -> 0.65 %; geometric midpoint is the
    # geometric mean of the end levels.
    mid = math.sqrt(3000.0 * 9000.0)
    assert iec61000_2_2_band_level_v(mid, u1) == pytest.approx(
        math.sqrt(0.014 * 0.0065) * u1
    )

    def dbuv(db):
        return 1e-6 * 10 ** (db / 20)

    assert iec61000_2_2_band_level_v(9000.0, u1) == pytest.approx(dbuv(129.5))
    assert iec61000_2_2_band_level_v(30000.0, u1) == pytest.approx(dbuv(122.0))
    assert iec61000_2_2_band_level_v(math.sqrt(9e3 * 30e3), u1) == pytest.approx(
        dbuv(125.75)
    )
    # At 50 kHz the lower level applies.
    assert iec61000_2_2_band_level_v(50000.0, u1) == pytest.approx(dbuv(113.0))
    assert iec61000_2_2_band_level_v(150000.0, u1) == pytest.approx(dbuv(89.0))
    assert iec61000_2_2_band_level_v(150001.0, u1) is None
    assert iec61000_2_2_provenance()["override"] is False
