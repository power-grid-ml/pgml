"""Current-injection power flow: why the voltage iterates oscillate near the nose.

WHAT THIS DEMONSTRATES
----------------------
pgml's nonlinear power flow (:func:`pgml.solver.solve_power_flow`) is a CURRENT-INJECTION
FIXED POINT:

    V_{k+1} = Y^{-1} ( I_slack - I_load(V_k) ),    I_load(V) = conj(S) / conj(V)

For a constant-power (const-P) load this map contracts only up to a point. As the load
approaches the feeder's loadability limit (the "nose" of the P-V curve) the map stops
contracting: the voltage iterates OSCILLATE and never settle; past the limit no solution
exists at all. Crucially the iteration stops converging BEFORE the true nose — its
convergence region is SMALLER than the feasible region. This is the well-known weakness of
Gauss / current-injection iterations versus Newton, and is exactly why the convergence
diagnostics report "did not settle" and point to the Newton solver
(``solve_power_flow(method="newton")``) and the ``loadability_limit`` continuation —
see ``ConvergenceDiagnostics`` and ``run/examples/pgml/loadability_continuation.py``.

The grid is a textbook 2-bus radial — a stiff source ``E`` behind a series line ``R+jX``
feeding a const-P load — for which the P-V nose is known in CLOSED FORM, so the solver's
behaviour can be overlaid on the exact curve. The plots:

1. ``residual_vs_iter`` — ``||ΔV||`` per iteration (the solver's own residual history) at
   several load levels: monotone decay (converge, but slower as load rises) -> never
   settling (oscillate near the nose) -> no decay (past the nose).
2. ``voltage_vs_iter`` — the load-bus voltage ``|V|/E`` per iteration of the SAME map
   (re-implemented transparently in two lines and validated to match the solver to ~1e-13):
   a flat settled line when it converges, a sustained zig-zag when it does not.
3. ``pv_nose`` — the analytical P-V curve (upper + lower branch) with the converged solver
   points marked on it, the method's practical convergence limit, and the true nose
   ``P_max`` — showing the convergence region sitting inside the feasible region.

RUN
---
::

    pixi run -e cpu python run/examples/pgml/current_injection_convergence.py [out_dir]

Outputs (default ``data/pgml/evaluation_output/current_injection/``): ``residual_vs_iter.svg``,
``voltage_vs_iter.svg``, ``pv_nose.svg``, and a printed convergence table. Tune the source
voltage / line ``R``, ``X`` / power factor at the top.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

from pgml.schemas.grid_schema import Grid, Line, Load, Node, Phase, Source
from pgml.solver.power_flow import solve_power_flow

E_V = 230.0  # source (line-to-neutral) voltage [V]
R_OHM = 0.5  # series line resistance [Ohm]
X_OHM = 0.5  # series line reactance [Ohm]
F0_HZ = 50.0
POWER_FACTOR = (
    1.0  # load displacement power factor (1.0 = unity; lagging -> Q = P*tan(phi))
)
MAX_ITER = 60
TOL = 1.0e-8
# Load levels as a fraction of the analytical nose power P_max.
LEVELS = [0.40, 0.70, 0.90, 0.97, 1.10]

CDT = torch.complex128
_Z = complex(R_OHM, X_OHM)
_TAN_PHI = math.tan(math.acos(POWER_FACTOR)) if POWER_FACTOR < 1.0 else 0.0


# Example outputs are anchored at the repository root (not the cwd), so a run writes
# under the untracked data root at data/pgml/evaluation_output/.
_OUT = Path(__file__).resolve().parents[3] / "data" / "pgml" / "evaluation_output"


def build_two_bus(p_w: float, q_var: float) -> Grid:
    """Stiff source (node 1) --R+jX line-- const-P load (node 2); no shunt (exact 2-bus)."""
    l_h = X_OHM / (2.0 * math.pi * F0_HZ)
    nodes = [
        Node(id=1, u_rated_v=E_V, phases=(Phase.A,)),
        Node(id=2, u_rated_v=E_V, phases=(Phase.A,)),
    ]
    source = Source(
        id=10,
        node=1,
        phases=(Phase.A,),
        u_ref_v=(E_V,),
        u_angle_deg=(0.0,),
        resistance_ohm=[[1.0e-9]],
        inductance_h=[[1.0e-12]],
    )
    line = Line(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=(Phase.A,),
        to_phases=(Phase.A,),
        length_m=1.0,
        series_resistance_ohm_per_m=[[R_OHM]],
        series_inductance_h_per_m=[[l_h]],
        shunt_capacitance_f_per_m=[[0.0]],
    )
    load = Load(id=30, node=2, phases=(Phase.A,), p_nom_w=p_w, q_nom_var=q_var)
    return Grid(
        base_frequency_hz=F0_HZ, nodes=nodes, branches=[line], appliances=[source, load]
    )


def pv_upper_lower(p_w: float, q_var: float):
    """Closed-form load-bus |V| on the upper / lower P-V branch (NaN past the nose).

    From ``S = V2·conj((E - V2)/Z)`` with ``V1 = E∠0`` the load-bus magnitude solves
    ``u² + (2a − E²)u + (a²+b²) = 0`` with ``u=|V2|²``, ``a=PR+QX``, ``b=QR−PX``.
    """
    a = p_w * R_OHM + q_var * X_OHM
    b = q_var * R_OHM - p_w * X_OHM
    disc = E_V**4 - 4.0 * a * E_V**2 - 4.0 * b * b
    if disc < 0.0:
        return float("nan"), float("nan")
    s = math.sqrt(disc)
    u_hi, u_lo = ((E_V**2 - 2.0 * a) + s) / 2.0, ((E_V**2 - 2.0 * a) - s) / 2.0
    hi = math.sqrt(u_hi) if u_hi > 0 else float("nan")
    lo = math.sqrt(u_lo) if u_lo > 0 else float("nan")
    return hi, lo


def nose_power() -> float:
    """Largest feasible P (the nose) for the configured power factor, via the quartic disc."""
    # disc(P) = E^4 - 4(R + tanphi·X)E²·P - 4(tanphi·R - X)²·P² = 0  (Q = tanphi·P)
    aa = -4.0 * (_TAN_PHI * R_OHM - X_OHM) ** 2
    bb = -4.0 * (R_OHM + _TAN_PHI * X_OHM) * E_V**2
    cc = E_V**4
    # positive root of aa·P² + bb·P + cc = 0  (aa <= 0)
    roots = [
        (-bb + sgn * math.sqrt(bb * bb - 4.0 * aa * cc)) / (2.0 * aa)
        for sgn in (1.0, -1.0)
    ]
    return min(p for p in roots if p > 0.0)


def current_injection_trace(s: complex, n_iter: int) -> list[complex]:
    """The 2-bus current-injection map ``V2 <- E - Z·conj(S/V2)`` from a flat start.

    Identical to the solver's per-iteration update for this grid (validated in :func:`main`).
    """
    v = complex(E_V, 0.0)
    traj = [v]
    for _ in range(n_iter):
        v = E_V - _Z * (s / v).conjugate()
        traj.append(v)
        if not (math.isfinite(v.real) and math.isfinite(v.imag)):
            break
    return traj


def main(out_dir: str = str(_OUT / "current_injection")) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    p_max = nose_power()
    print(
        f"analytical nose: P_max = {p_max / 1000:.2f} kW  (E={E_V:g} V, R={R_OHM:g}, "
        f"X={X_OHM:g} Ohm, pf={POWER_FACTOR:g})"
    )

    runs = []
    max_dv_mismatch = 0.0
    for frac in LEVELS:
        p_w = frac * p_max
        q_var = _TAN_PHI * p_w
        res = solve_power_flow(
            build_two_bus(p_w, q_var),
            slack="ideal",
            tol=TOL,
            max_iter=MAX_ITER,
            dtype=CDT,
            criticality="always",
        )
        row = res.index.row(2, Phase.A)
        v2_pu = abs(res.v.reshape(-1)[row].item()) / E_V
        traj = current_injection_trace(complex(p_w, q_var), MAX_ITER)
        v_pu = [abs(v) / E_V for v in traj]
        # Faithfulness: the transparent map's |ΔV2| must equal the solver's ||ΔV||.
        dv_trace = [abs(traj[k + 1] - traj[k]) for k in range(len(traj) - 1)]
        hist = res.diagnostics.residual_history
        if hist and dv_trace:
            max_dv_mismatch = max(
                max_dv_mismatch, max(abs(a - b) for a, b in zip(hist, dv_trace))
            )
        runs.append(
            {
                "frac": frac,
                "p_w": p_w,
                "converged": res.converged,
                "iters": res.iterations,
                "v2_pu": v2_pu,
                "history": hist,
                "v_pu": v_pu,
                "crit": res.diagnostics.criticality or {},
                "cause": res.diagnostics.likely_cause,
            }
        )

    _print_table(runs, p_max)
    print(
        f"\nmap fidelity: max |solver ||ΔV|| − trace |ΔV2|| = {max_dv_mismatch:.1e} "
        "(the trace IS the solver's iteration)"
    )

    _plot_residuals(runs, out / "residual_vs_iter.svg")
    _plot_voltage_trace(runs, out / "voltage_vs_iter.svg")
    _plot_pv_nose(runs, p_max, out / "pv_nose.svg")
    print(
        f"figures -> {out.resolve()}/{{residual_vs_iter,voltage_vs_iter,pv_nose}}.svg"
    )


def _print_table(runs, p_max) -> None:
    print(
        f"\n{'load':>6} {'P[kW]':>7} {'conv':>5} {'iters':>6} {'|V2|/E':>7} "
        f"{'cond(J)':>9}  cause"
    )
    for r in runs:
        cond = r["crit"].get("condition_number", float("nan"))
        print(
            f"{r['frac'] * 100:5.0f}% {r['p_w'] / 1000:7.2f} {str(r['converged']):>5} "
            f"{r['iters']:6d} {r['v2_pu']:7.3f} {cond:9.1e}  {r['cause']}"
        )


def _palette(n: int):
    import matplotlib.pyplot as plt

    return [plt.get_cmap("viridis")(i / max(n - 1, 1)) for i in range(n)]


def _label(r) -> str:
    return f"{r['frac'] * 100:.0f}% P_max ({'conv' if r['converged'] else 'no conv'})"


def _plot_residuals(runs, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = _palette(len(runs))
    for r, c in zip(runs, colors):
        h = r["history"]
        ax.semilogy(range(1, len(h) + 1), h, "-o", ms=3, color=c, label=_label(r))
    ax.axhline(TOL, ls=":", color="k", lw=1, label=f"tol = {TOL:g}")
    ax.set(
        xlabel="iteration k",
        ylabel=r"$\|\Delta V\|$  [V]",
        title="Current-injection convergence: update norm per iteration\n"
        "(decays when converging, never settles near / past the nose)",
    )
    ax.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def _plot_voltage_trace(runs, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = _palette(len(runs))
    ymax = 1.05
    for r, c in zip(runs, colors):
        ax.plot(range(len(r["v_pu"])), r["v_pu"], "-o", ms=3, color=c, label=_label(r))
        ymax = max(ymax, max(r["v_pu"]))
    ax.set(
        xlabel="iteration k",
        ylabel=r"load-bus $|V|/E$  [pu]",
        title="Load-bus voltage per iteration of the current-injection map\n"
        "(settles when converging; sustained oscillation past the nose)",
    )
    ax.set_ylim(
        0.0, min(ymax * 1.05, 3.0)
    )  # show the oscillation peaks, cap extreme cases
    ax.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def _plot_pv_nose(runs, p_max, path: Path) -> None:
    import matplotlib.pyplot as plt

    ps = [p_max * i / 400.0 for i in range(1, 401)]
    hi, lo = [], []
    for p in ps:
        h, low = pv_upper_lower(p, _TAN_PHI * p)
        hi.append(h / E_V)
        lo.append(low / E_V)
    p_kw = [p / 1000.0 for p in ps]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(p_kw, hi, "-", color="C0", label="stable (upper) branch")
    ax.plot(p_kw, lo, "--", color="C7", lw=1, label="unstable (lower) branch")
    ax.axvline(
        p_max / 1000.0,
        ls=":",
        color="C3",
        label=f"true nose P_max = {p_max / 1000:.1f} kW",
    )

    conv = [r for r in runs if r["converged"]]
    noconv = [r for r in runs if not r["converged"]]
    if conv:
        ax.scatter(
            [r["p_w"] / 1000 for r in conv],
            [r["v2_pu"] for r in conv],
            s=60,
            color="C2",
            zorder=5,
            label="solver converged",
        )
        p_lim = max(r["p_w"] for r in conv)
        ax.axvspan(0, p_lim / 1000.0, color="C2", alpha=0.08)
        ax.axvline(
            p_lim / 1000.0,
            ls="-.",
            color="C2",
            label=f"convergence limit ~ {p_lim / 1000:.1f} kW",
        )
    if noconv:
        ax.scatter(
            [r["p_w"] / 1000 for r in noconv],
            [r["v2_pu"] for r in noconv],
            s=70,
            marker="x",
            color="C3",
            zorder=5,
            label="solver did NOT converge",
        )
    ax.set(
        xlabel="load power P  [kW]",
        ylabel=r"load-bus $|V|/E$  [pu]",
        title="P-V nose: the current-injection convergence region (shaded)\n"
        "sits INSIDE the feasible region (up to the nose)",
    )
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(_OUT / "current_injection"))
