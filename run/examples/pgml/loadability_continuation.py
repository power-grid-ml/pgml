"""Loadability continuation: how much load until collapse, and which bus/load is the limit.

WHAT THIS DEMONSTRATES
----------------------
:func:`pgml.solver.loadability_limit` answers the question a failed power flow raises —
*"how much more load can this grid take, and which injection at which node makes it
non-convergent?"* It is a CONTINUATION power flow: it ramps every load by a multiplier
``λ`` (``λ=1`` = the nameplate load) from a feasible base, Newton-corrects at each step,
and bisects onto the breaking ``λ*`` — the nose of the P-V curve. At the nose the
power-flow Jacobian is singular, and its singular vectors localize the collapse:

- the RIGHT singular vector is the voltage-collapse mode -> the **critical bus(es)** (where
  the voltage gives way);
- the LEFT singular vector, projected on each load's current, is the margin sensitivity ->
  the **limiting load(s)** (which apparent-power injection most reduces the margin).

On the FULL CIGRE LV benchmark this script:

1. runs :func:`loadability_limit` and prints the margin + critical bus + limiting load;
2. traces the P-V curve at the critical bus (ramping the load and solving with the Newton
   method, ``solve_power_flow(method="newton")``, which converges near the nose);
3. plots the voltage profile at the nose with the critical bus(es) highlighted, and a bar
   chart of the loads ranked by how much they limit the margin.

RUN
---
::

    pixi run -e cpu python run/examples/pgml/loadability_continuation.py [out_dir]

Outputs (default ``evaluation_output/loadability/``): ``pv_nose.svg`` (P-V curve),
``voltage_profile_nose.svg`` (critical bus highlighted), ``limiting_loads.svg`` (bar
chart), and a printed summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from pgml.assembly._params import phase_voltage_magnitude
from pgml.convert.pandapower import PhaseMode
from pgml.evaluation.oracles.grids import cigre_lv_full_grid
from pgml.evaluation.topology import distance_from_slack
from pgml.schemas.grid_schema import Generator, Load, Phase
from pgml.solver import loadability_limit, solve_power_flow

CDT = torch.complex128


# Example outputs are anchored at run/examples/ (not the cwd), so a run writes
# under run/examples/evaluation_output/ rather than the repository root.
_OUT = Path(__file__).resolve().parent.parent / "evaluation_output"


def scale_loads(grid, lam: float):
    """A deep copy of ``grid`` with every load/generator's P/Q scaled by ``lam``."""
    g = grid.model_copy(deep=True)
    for a in g.appliances:
        if not isinstance(a, (Load, Generator)):
            continue
        if getattr(a, "p_nom_w", None) is not None:
            a.p_nom_w = float(a.p_nom_w) * lam
        if getattr(a, "q_nom_var", None) is not None:
            a.q_nom_var = float(a.q_nom_var) * lam
        if getattr(a, "p_nom_per_phase_w", None) is not None:
            a.p_nom_per_phase_w = [float(x) * lam for x in a.p_nom_per_phase_w]
        if getattr(a, "q_nom_per_phase_var", None) is not None:
            a.q_nom_per_phase_var = [float(x) * lam for x in a.q_nom_per_phase_var]
    return g


def _node_pu(result, grid, node_id: int) -> float:
    """``|V| / V_LN`` at a node's phase A from a solved result."""
    node = next(n for n in grid.nodes if int(n.id) == node_id)
    base = phase_voltage_magnitude(float(node.u_rated_v), len(node.phases))
    row = result.index.row(node_id, Phase.A)
    return abs(result.v.reshape(-1)[row].item()) / base


def main(out_dir: str = str(_OUT / "loadability")) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)

    # 1. The continuation diagnostic: margin + critical bus + limiting load.
    res = loadability_limit(
        grid, slack="ideal", lambda_max=8.0, lambda_step=0.5, dtype=CDT
    )
    crit = res.critical_nodes[0]
    lim = res.limiting_loads[0]
    print(
        f"loadability: breaking λ* = {res.breaking_lambda:.2f}  "
        f"(feasible={res.feasible}, margin = {res.margin:.2f}× nameplate); "
        f"nose voltage {res.nose_voltage_min_pu:.3f} pu"
    )
    print(
        f"  critical bus (voltage-collapse mode): node {crit['node_id']} "
        f"(participation {crit['participation']:.2f})"
    )
    print(
        f"  limiting load (most reduces the margin): appliance {lim['appliance_id']} "
        f"at node {lim['node_id']} (S={lim['s_nominal_va'] / 1e3:.1f} kVA, "
        f"responsibility {lim['responsibility']:.2f})"
    )

    # 2. Trace the P-V curve at the critical bus by ramping the load and solving.
    crit_id = int(crit["node_id"])
    dist = distance_from_slack(grid)
    ref_id = min(  # a slack-adjacent load bus, for contrast (stays near 1.0 pu)
        (int(a.node) for a in grid.appliances if isinstance(a, Load)),
        key=lambda nid: dist.get(nid, float("inf")),
    )
    lams, v_crit, v_ref, last = [], [], [], None
    lam = 0.5
    while lam <= res.breaking_lambda + 0.3:
        r = solve_power_flow(
            scale_loads(grid, lam),
            slack="ideal",
            method="newton",
            tol=1e-8,
            max_iter=60,
            dtype=CDT,
        )
        if not r.converged:
            break
        lams.append(lam)
        v_crit.append(_node_pu(r, grid, crit_id))
        v_ref.append(_node_pu(r, grid, ref_id))
        last = r
        lam += 0.2

    _plot_pv(lams, v_crit, v_ref, crit_id, ref_id, res, out / "pv_nose.svg")
    if last is not None:
        _plot_profile(last, grid, res, dist, out / "voltage_profile_nose.svg")
    _plot_limiting(res, out / "limiting_loads.svg")
    print(
        f"figures -> {out.resolve()}/{{pv_nose,voltage_profile_nose,limiting_loads}}.svg"
    )


def _plot_pv(lams, v_crit, v_ref, crit_id, ref_id, res, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(
        lams, v_crit, "-o", ms=3, color="C3", label=f"critical bus (node {crit_id})"
    )
    ax.plot(
        lams, v_ref, "-o", ms=3, color="C0", label=f"slack-adjacent bus (node {ref_id})"
    )
    ax.axvline(
        res.breaking_lambda,
        ls="--",
        color="C7",
        label=f"nose λ* = {res.breaking_lambda:.2f} (margin {res.margin:+.2f})",
    )
    ax.axvline(1.0, ls=":", color="k", lw=1, label="nameplate load (λ=1)")
    ax.set(
        xlabel="load multiplier λ  (×nameplate)",
        ylabel=r"$|V|/V_{LN}$  [pu]",
        title="P-V nose: voltage vs load, traced by the continuation\n"
        "(the critical bus collapses first; λ* is the loadability limit)",
    )
    ax.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def _plot_profile(result, grid, res, dist, path: Path) -> None:
    import matplotlib.pyplot as plt

    crit_ids = {int(c["node_id"]) for c in res.critical_nodes[:3]}
    xs, ys, crit_x, crit_y = [], [], [], []
    for node in grid.nodes:
        nid = int(node.id)
        try:
            row = result.index.row(nid, Phase.A)
        except (KeyError, ValueError):
            continue
        base = phase_voltage_magnitude(float(node.u_rated_v), len(node.phases))
        pu = abs(result.v.reshape(-1)[row].item()) / base
        d = dist.get(nid, 0.0)
        if nid in crit_ids:
            crit_x.append(d)
            crit_y.append(pu)
        else:
            xs.append(d)
            ys.append(pu)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(xs, ys, s=18, color="C0", label="bus voltage")
    ax.scatter(
        crit_x,
        crit_y,
        s=90,
        color="C3",
        marker="*",
        zorder=5,
        label="critical bus (voltage-collapse mode)",
    )
    lim = res.limiting_loads[0]
    ax.annotate(
        f"limiting load: appliance {lim['appliance_id']} @ node {lim['node_id']}",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    ax.set(
        xlabel="distance from slack [km]",
        ylabel=r"$|V|/V_{LN}$  [pu]",
        title=f"Voltage profile near the nose (λ ≈ {res.breaking_lambda:.2f}) — "
        "critical bus highlighted",
    )
    ax.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def _plot_limiting(res, path: Path) -> None:
    import matplotlib.pyplot as plt

    top = res.limiting_loads[:8]
    labels = [f"a{d['appliance_id']}\n@n{d['node_id']}" for d in top]
    vals = [d["responsibility"] for d in top]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(range(len(top)), vals, color="C3")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set(
        ylabel="margin responsibility  [normalized]",
        title="Loads ranked by how much their apparent power limits the loadability margin",
    )
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(_OUT / "loadability"))
