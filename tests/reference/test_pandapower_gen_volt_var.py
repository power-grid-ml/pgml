"""Oracle test: ``net.gen`` as a Volt-VAr droop vs a live pandapower ``runpp``.

``GenMode.VOLT_VAR_APPROX`` converts pandapower's PV bus (``net.gen``: fixed P,
regulated ``vm_pu``, free Q within ``min_q_mvar``/``max_q_mvar``) into a
:class:`~pgml.schemas.grid_schema.Generator` carrying a steep ``Q(|V|)`` droop.
This file measures how close that gets to pandapower on MATPOWER transmission
benchmarks, and pins the two properties the approximation is claimed to have.

Oracle setup
------------
- ``pp.runpp(net, enforce_q_lims=True)`` is the apples-to-apples reference: the
  converted droop ENFORCES the reactive limits (via the curve saturation backed by
  the capability circle), while pandapower's DEFAULT ``enforce_q_lims=False``
  ignores them entirely and lets a PV bus deliver whatever Q holds the setpoint.
- ``net.shunt`` is taken out of service in BOTH tools wherever a benchmark carries
  one. The converter does not read ``net.shunt`` at all (an open coverage gap), so
  leaving the shunts in would fold that gap into every number here and hide what
  the ``gen`` mapping itself is worth. ``case9`` carries no shunt, so its
  comparison is against the untouched benchmark.
- ``method="newton"``: the current-injection fixed point does not contract on a
  stiff droop.

What the numbers mean
---------------------
The regulated bus settles where the droop's reactive output balances the network,
i.e. off its setpoint by ``Q_actual / (slope * q_base)`` per unit — so the per-bus
deviation falls as ``1 / gen_volt_var_slope_pu``. The tolerances below are the
measured values with roughly a factor-two margin; they are ACCURACY OF AN
APPROXIMATION, not solver tolerances, and they are three to four orders of
magnitude looser than the const-P oracle tolerances in
``test_pandapower_grid_matrix.py`` for that reason.

The ceiling is conditioning, not steady-state fidelity: outside the
``1/slope``-wide band the droop's ``dQ/d|V|`` is exactly zero, so a Newton iterate
that starts far from the setpoint sees no voltage-control feedback and can settle
on the collapsed low-voltage branch. ``case118`` reaches that ceiling at a droop
steepness of about 5, which is why its test asserts a percent-level tolerance
while ``case9`` asserts a 1e-4 one. The full record, including the steepness
sweep across five benchmarks, is ``src/pgml/convert/pandapower/CONTEXT.md``.
"""

from __future__ import annotations

import math

import pandapower as pp
import pandapower.networks as pn
import pytest
import torch

from pgml.convert.pandapower import GenMode, to_grid
from pgml.schemas.grid_schema import Phase
from pgml.solver import solve_power_flow

CDT = torch.complex128


def _angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def _reference(factory, *, disable_shunts: bool = True):
    """A converged pandapower benchmark, optionally with its shunts switched off."""
    net = factory()
    if disable_shunts and len(net.shunt):
        net.shunt["in_service"] = False
    pp.runpp(net, numba=False, enforce_q_lims=True)
    assert net.converged, "pandapower did not converge -- check the test setup"
    return net


def _deviations(net, *, slope: float, max_iter: int = 80, **to_grid_kwargs):
    """Per-bus ``(|V| [pu], angle [deg])`` deviations from ``net.res_bus``."""
    grid, id_map = to_grid(
        net,
        gen_mode=GenMode.VOLT_VAR_APPROX,
        gen_volt_var_slope_pu=slope,
        **to_grid_kwargs,
    )
    result = solve_power_flow(
        grid,
        slack="ideal",
        method="newton",
        tol=1e-8,
        max_iter=max_iter,
        dtype=CDT,
        criticality="never",
    )
    assert result.converged, (
        f"solve_power_flow did not converge at slope={slope} "
        f"(residual={float(result.residual):.3e}, iterations={result.iterations}) -- "
        "the droop's basin of attraction shrinks as it steepens; reduce the slope"
    )
    v_dev, a_dev = [], []
    flat = result.v.reshape(-1)
    for pp_idx, node_id in id_map["bus"].items():
        v = flat[result.index.row(node_id, Phase.A)].item()
        u_rated_v = float(net.bus.at[pp_idx, "vn_kv"]) * 1_000.0
        v_dev.append(abs(abs(v) / u_rated_v - float(net.res_bus.at[pp_idx, "vm_pu"])))
        a_dev.append(
            abs(
                _angle_diff_deg(
                    math.degrees(math.atan2(v.imag, v.real)),
                    float(net.res_bus.at[pp_idx, "va_degree"]),
                )
            )
        )
    return v_dev, a_dev


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


# ---------------------------------------------------------------------------
# 1. case9 -- 2 PV buses, no shunt, no transformer: the clean measurement
# ---------------------------------------------------------------------------
def test_case9_matches_pandapower_at_the_default_slope():
    """At the default steepness the whole 9-bus operating point lands within
    2e-4 pu / 5e-3 deg of pandapower's own PV-bus solve (measured 8.4e-5 pu /
    1.6e-3 deg)."""
    net = _reference(pn.case9, disable_shunts=False)
    v_dev, a_dev = _deviations(net, slope=500.0)
    assert max(v_dev) < 2.0e-4, f"|V| deviation {max(v_dev):.3e} pu"
    assert max(a_dev) < 5.0e-3, f"angle deviation {max(a_dev):.3e} deg"


# ---------------------------------------------------------------------------
# 2. The accuracy/steepness law
# ---------------------------------------------------------------------------
def test_deviation_falls_inversely_with_droop_steepness():
    """The regulation error is ``Q / (slope * q_base)`` pu, so a tenfold steeper
    droop buys about a tenfold smaller deviation. Measured on case57 (6 PV buses,
    17 transformers, shunts disabled in both tools): 4.3e-2 -> 6.8e-3 -> 7.6e-4 pu
    for slopes 5 -> 50 -> 500."""
    net = _reference(pn.case57)
    worst = {}
    for slope in (5.0, 50.0, 500.0):
        v_dev, _ = _deviations(net, slope=slope)
        worst[slope] = max(v_dev)
    assert worst[500.0] < worst[50.0] < worst[5.0]
    # A tenfold steepening must buy at least a fivefold improvement (the law is
    # 1/slope; the margin covers the generators that sit on a reactive limit,
    # where the deviation is set by pandapower's discrete PV->PQ switch instead).
    assert worst[50.0] < worst[5.0] / 5.0
    assert worst[500.0] < worst[50.0] / 5.0
    assert worst[500.0] < 2.0e-3


# ---------------------------------------------------------------------------
# 3. case118 -- the motivating transmission benchmark
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_case118_gen_conversion_recovers_the_operating_point():
    """``case118`` carries all 53 of its generators in ``net.gen``. Dropping them
    (the converter's default) leaves a network with no voltage support that does
    not even converge; the Volt-VAr approximation recovers a plausible operating
    point — but only at a shallow droop, which caps the fidelity at the percent
    level. Both halves are asserted, because the second is the honest limit of
    this approximation and the first is the reason it exists at all."""
    net = _reference(pn.case118)

    v_dev, a_dev = _deviations(net, slope=5.0)
    assert max(v_dev) < 8.0e-2, f"|V| deviation {max(v_dev):.3e} pu"
    assert _median(v_dev) < 4.0e-2, f"median |V| deviation {_median(v_dev):.3e} pu"
    assert max(a_dev) < 2.5, f"angle deviation {max(a_dev):.3e} deg"

    # The default (gen dropped) is not a usable operating point for this grid.
    dropped, _ = to_grid(net)
    result = solve_power_flow(
        dropped,
        slack="ideal",
        method="newton",
        tol=1e-8,
        max_iter=25,
        dtype=CDT,
        criticality="never",
    )
    assert not result.converged, (
        "case118 without its net.gen rows unexpectedly converged -- the drop "
        "warning and the opt-in mode's rationale need revisiting"
    )
