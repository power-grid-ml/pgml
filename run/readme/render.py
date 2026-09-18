"""Render README evidence from recorded JSON; no solver or benchmark is run."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "readme"
plt.rcParams.update(
    {
        "font.size": 11,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
        "svg.hashsalt": "pgml-readme",
    }
)


def save(fig, stem):
    svg = ASSETS / f"{stem}.svg"
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    fig.savefig(Path("/tmp") / f"pgml-readme-{stem}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


#: One colour and marker per tool in every performance figure. The reference
#: tools keep the colours of the conformance figures (pandapower blue,
#: power-grid-model green, OpenDSS rust); pgml is near-black on the CPU and
#: purple on the GPU, drawn solid, the reference tools dashed.
TOOLS = {
    "pgml_cpu": ("pgml · CPU", "#111827", "o-"),
    "pgml_gpu": ("pgml · GPU", "#7c3aed", "D-"),
    "pandapower": ("pandapower · CPU", "#2b6cb0", "^--"),
    "power_grid_model": ("power-grid-model · CPU", "#2f855a", "v--"),
    "opendss": ("OpenDSS · CPU", "#c05621", "s--"),
}
GUARD_TOL_PU = 1e-6


def _pgml_point(row, where):
    """Throughput of a pgml record, or an error for an invalid published point."""
    if "throughput" not in row:
        return None
    error = row.get("max_dV_pu", np.nan)
    ok = row.get("verdict") == "ok" and row.get("converged") is True
    if not ok or not np.isfinite(error) or error > GUARD_TOL_PU:
        raise ValueError(f"Invalid pgml point: {where}/{row}")
    return row["throughput"]


def _reference_point(row, where, *, every_scenario):
    """Throughput of a reference-tool record that matched pgml, else an error."""
    if "throughput" not in row:
        return None
    error = row.get("max_dV_vs_pgml_cpu_c128_pu", np.nan)
    ok = row.get("status") == "ok" and np.isfinite(error) and error <= GUARD_TOL_PU
    if every_scenario:
        ok &= row.get("comparison_scenarios") == row["batch"]
    if where.startswith("pandapower"):
        ok &= row.get("diagnostics", {}).get("numba_active") is True
    if not ok:
        raise ValueError(f"Invalid comparison point: {where}/{row}")
    return row["throughput"]


def _tool_points(series, baselines, where, *, every_scenario):
    """``{tool: {batch: throughput}}``; pgml CPU is the faster CPU backend."""
    out = {tool: {} for tool in TOOLS}
    for key, rows in series.items():
        tool = "pgml_gpu" if key.startswith("gpu_") else "pgml_cpu"
        for row in rows:
            value = _pgml_point(row, f"{where}/{key}")
            if value is not None:
                batch = row["batch"]
                out[tool][batch] = max(out[tool].get(batch, 0.0), value)
    for key, rows in baselines.items():
        for row in rows:
            value = _reference_point(
                row, f"{key}/{where}", every_scenario=every_scenario
            )
            if value is not None:
                out[key][row["batch"]] = value
    return out


def _draw(ax, curves):
    """Draw ``{tool: [(x, y), ...]}`` in the shared tool style."""
    for tool, (label, color, style) in TOOLS.items():
        points = sorted(curves.get(tool, []))
        if points:
            ax.plot(
                *zip(*points),
                style,
                color=color,
                label=label,
                linewidth=1.8,
                markersize=4,
            )
    ax.set(xscale="log", yscale="log")
    ax.grid(alpha=0.2, which="both")


def _legend_below(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))


def throughput():
    data = json.loads((ASSETS / "batch_throughput.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharey=True)
    for ax, (name, title) in zip(
        axes, [("ieee33", "IEEE 33-bus feeder"), ("kerber", "Kerber · 294 buses")]
    ):
        grid = data["grids"][name]
        points = _tool_points(
            grid["series"], grid["baselines"], name, every_scenario=True
        )
        batches = [row["batch"] for row in grid["series"]["gpu_dense_c128"]]
        for tool, by_batch in points.items():
            if sorted(by_batch) != batches:
                raise ValueError(f"Incomplete curve: {name}/{tool}")
        _draw(ax, {tool: list(by_batch.items()) for tool, by_batch in points.items()})
        ax.set(xlabel="Scenarios per batch", title=title)
    axes[0].set_ylabel("Solved scenarios / second")
    _legend_below(fig, axes[0])
    save(fig, "batch_throughput")


def size_scaling():
    data = json.loads((ASSETS / "size_scaling.json").read_text())
    batches = data["batches"]
    fig, axes = plt.subplots(1, len(batches), figsize=(10.8, 4.1), sharey=True)
    curves = {batch: {tool: [] for tool in TOOLS} for batch in batches}
    for rung in data["rungs"]:
        points = _tool_points(
            rung["series"],
            rung.get("baselines", {}),
            rung["name"],
            every_scenario=True,
        )
        for tool, by_batch in points.items():
            for batch, value in by_batch.items():
                curves[batch][tool].append((rung["rows"], value))
    for ax, batch in zip(axes, batches):
        _draw(ax, curves[batch])
        noun = "scenario" if batch == 1 else "scenarios"
        ax.set(xlabel="Grid size (buses)", title=f"{batch} {noun} per batch")
    axes[0].set_ylabel("Solved scenarios / second")
    _legend_below(fig, axes[0])
    save(fig, "size_scaling")


def resistance():
    data = json.loads((ASSETS / "app_digital_twin.json").read_text())
    n = len(data["case"]["fitted_lines"])
    x = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(10.8, 3.8), layout="constrained")
    ax.step(
        np.r_[x - 0.5, n + 0.5],
        np.r_[data["theta_true"][:n], data["theta_true"][n - 1]],
        where="post",
        color="#111827",
        linewidth=2,
        label="True installed resistance",
    )
    for key, label, offset, color in [
        ("V", "Fundamental voltage", -0.22, "#64748b"),
        ("V+I", "+ current", 0, "#2563eb"),
        ("V+I+harm", "+ harmonic voltages", 0.22, "#d97706"),
    ]:
        d = data["configs"][key]
        ax.errorbar(
            x + offset,
            d["theta_mean"][:n],
            yerr=d["theta_std"][:n],
            fmt="o",
            capsize=3,
            linewidth=1.3,
            markersize=5,
            color=color,
            label=label,
        )
    ax.axhline(1, color="#059669", linestyle=":", label="Initial catalogue value")
    ax.set(
        xticks=x,
        xlabel="Trunk line, counted from the feeder head",
        ylabel="Resistance / catalogue resistance",
        ylim=(0.38, 1.58),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=9)
    save(fig, "resistance_recovery")


if __name__ == "__main__":
    throughput()
    size_scaling()
    resistance()
