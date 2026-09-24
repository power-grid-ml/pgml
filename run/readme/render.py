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
#: power-grid-model green, OpenDSS rust); pgml is purple on the GPU, magenta on
#: the CPU under the same parallelisation the reference tools get, and near-black
#: for a single batched CPU call. pgml is drawn solid, the reference tools dashed.
TOOLS = {
    "pgml_gpu": ("pgml · GPU", "#7c3aed", "D-"),
    "pgml_cpu_pool": ("pgml · CPU, 8 workers", "#97266d", "o-"),
    "pgml_cpu_single": ("pgml · CPU, one call", "#111827", "o:"),
    "pandapower": ("pandapower · CPU", "#2b6cb0", "^--"),
    "power_grid_model": ("power-grid-model · CPU", "#2f855a", "v--"),
    "opendss": ("OpenDSS · CPU", "#c05621", "s--"),
}

#: The README carries one figure, and it compares tools rather than
#: configurations of one tool: pgml on a GPU and pgml on the CPU allocation every
#: reference tool is given, against those tools. The single batched CPU call is a
#: second configuration of the same engine and belongs on the performance page.
README_TOOLS = (
    "pgml_gpu",
    "pgml_cpu_pool",
    "pandapower",
    "power_grid_model",
    "opendss",
)

GUARD_TOL_PU = 1e-6

#: pgml series prefix -> tool, longest prefix first
SERIES_TOOL = (
    ("cpu_pool_", "pgml_cpu_pool"),
    ("gpu_", "pgml_gpu"),
    ("cpu_", "pgml_cpu_single"),
)


def _tool_of(series_key):
    for prefix, tool in SERIES_TOOL:
        if series_key.startswith(prefix):
            return tool
    raise ValueError(f"Unknown pgml series: {series_key}")


def _pgml_point(row, where):
    """Throughput of a pgml record, or an error for an invalid published point."""
    if "throughput" not in row:
        return None
    error = row.get("max_dV_pu", np.nan)
    ok = row.get("verdict") == "ok" and row.get("converged") is True
    if not ok or not np.isfinite(error) or error > GUARD_TOL_PU:
        raise ValueError(f"Invalid pgml point: {where}/{row}")
    return row["throughput"]


#: Fewest scenarios of a batch that must have been re-solved and compared for a
#: reference point to be drawn. A smaller batch is checked in full.
VALIDATED_FLOOR = 32


def _reference_point(row, where, *, validated):
    """Throughput of a reference-tool record that matched pgml, else an error.

    ``validated`` is how many scenarios of a batch the published data records as
    re-solved and compared, which is the deepest check any record actually got.
    Every record must have had that same check, and it must reach the floor. The
    tolerance is the one the record itself carries, because the harmonic
    comparison is judged at a looser figure than the fundamental one and says so.
    """
    if validated < VALIDATED_FLOOR:
        raise ValueError(
            f"{where}: the data validates {validated} scenarios per batch, "
            f"below the floor of {VALIDATED_FLOOR}"
        )
    if "throughput" not in row:
        return None
    error = row.get("max_dV_vs_pgml_cpu_c128_pu", np.nan)
    tol = row.get("comparison_tolerance_pu", GUARD_TOL_PU)
    ok = row.get("status") == "ok" and np.isfinite(error) and error <= tol
    ok &= row.get("comparison_scenarios", 0) >= min(row["batch"], validated)
    if where.startswith("pandapower"):
        ok &= row.get("diagnostics", {}).get("numba_active") is True
    if not ok:
        raise ValueError(f"Invalid comparison point: {where}/{row}")
    return row["throughput"]


def _tool_points(series, baselines, where, *, validated):
    """``{tool: {batch: throughput}}``; each tool keeps its best configuration."""
    out = {tool: {} for tool in TOOLS}
    for key, rows in series.items():
        tool = _tool_of(key)
        for row in rows:
            value = _pgml_point(row, f"{where}/{key}")
            if value is not None:
                batch = row["batch"]
                out[tool][batch] = max(out[tool].get(batch, 0.0), value)
    for key, rows in baselines.items():
        if key not in out:
            continue
        for row in rows:
            value = _reference_point(row, f"{key}/{where}", validated=validated)
            if value is not None:
                out[key][row["batch"]] = value
    return out


def _draw(ax, curves, tools=None):
    """Draw ``{tool: [(x, y), ...]}`` in the shared tool style."""
    for tool in tools or TOOLS:
        label, color, style = TOOLS[tool]
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


def _legend_below(fig, ax, ncol=5):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=10
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))


def _grid_titles(data, names):
    return [
        (name, data["grids"][name]["label"]) for name in names if name in data["grids"]
    ]


def _sweep_curves(data, name):
    grid = data["grids"][name]
    return _tool_points(
        grid["series"],
        grid.get("baselines", {}),
        name,
        validated=data.get("validated_scenarios_per_batch", VALIDATED_FLOOR),
    )


def _throughput_figure(source, stem, *, names, tools, unit, titles=None):
    data = json.loads((ASSETS / source).read_text())
    pairs = titles or _grid_titles(data, names)
    fig, axes = plt.subplots(
        1, len(pairs), figsize=(5.4 * len(pairs), 4.1), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, (name, title) in zip(axes, pairs):
        points = _sweep_curves(data, name)
        rows = data["grids"][name]["rows"]
        _draw(
            ax,
            {tool: list(by_batch.items()) for tool, by_batch in points.items()},
            tools=tools,
        )
        ax.set(xlabel=f"{unit} per batch", title=f"{title} · {rows} rows")
    axes[0].set_ylabel(f"Solved {unit} / second")
    _legend_below(fig, axes[0], ncol=min(len(tools), 5))
    save(fig, stem)


def throughput():
    """The README's single performance figure."""
    _throughput_figure(
        "batch_throughput.json",
        "batch_throughput",
        names=("ieee33", "kerber"),
        tools=README_TOOLS,
        unit="Scenarios",
    )


def throughput_all():
    """The performance page's version, with the single-call CPU configuration."""
    _throughput_figure(
        "batch_throughput.json",
        "batch_throughput_all",
        names=("ieee33", "cigre_lv3", "kerber", "kerber_x4"),
        tools=tuple(TOOLS),
        unit="Scenarios",
    )


def harmonic_throughput():
    """Whole harmonic studies instead of a single power flow."""
    _throughput_figure(
        "harmonic_throughput.json",
        "harmonic_throughput",
        names=("ieee33", "kerber", "kerber_x4"),
        tools=("pgml_gpu", "pgml_cpu_pool", "pgml_cpu_single", "opendss"),
        unit="Studies",
    )


def size_scaling():
    data = json.loads((ASSETS / "size_scaling.json").read_text())
    batches = data["batches"]
    validated = data.get("validated_scenarios_per_batch", VALIDATED_FLOOR)
    fig, axes = plt.subplots(
        1, len(batches), figsize=(3.4 * len(batches), 4.1), sharey=True
    )
    axes = np.atleast_1d(axes)
    curves = {batch: {tool: [] for tool in TOOLS} for batch in batches}
    for rung in data["rungs"]:
        points = _tool_points(
            rung["series"], rung.get("baselines", {}), rung["name"], validated=validated
        )
        for tool, by_batch in points.items():
            for batch, value in by_batch.items():
                if batch in curves:
                    curves[batch][tool].append((rung["rows"], value))
    for ax, batch in zip(axes, batches):
        _draw(ax, curves[batch])
        noun = "scenario" if batch == 1 else "scenarios"
        ax.set(xlabel="Grid size (buses)", title=f"{batch} {noun} per batch")
    axes[0].set_ylabel("Solved scenarios / second")
    _legend_below(fig, axes[0], ncol=3)
    save(fig, "size_scaling")


#: memory figure labels, in the order the panels show them
MEMORY_TOOLS = (
    "pgml_gpu",
    "pgml_cpu_single",
    "pgml_cpu_pool",
    "pandapower",
    "power_grid_model",
    "opendss",
)


def memory():
    """Host memory for every tool and device memory for pgml, per batch size."""
    data = json.loads((ASSETS / "footprint.json").read_text())
    grids = data["grids"]
    fig, axes = plt.subplots(
        2, len(grids), figsize=(4.2 * len(grids), 7.2), sharex="col"
    )
    axes = axes.reshape(2, -1)
    for column, grid in enumerate(grids):
        host, device = axes[0][column], axes[1][column]
        for tool in MEMORY_TOOLS:
            label, color, style = TOOLS[tool]
            points = sorted(
                (r["batch"], r["tree_peak_pss_mib"])
                for r in grid["points"]
                if r["engine"] == tool and r.get("status") == "ok"
            )
            if points:
                host.plot(*zip(*points), style, color=color, label=label, markersize=4)
        points = sorted(
            (r["batch"], r["device_peak_allocated_mib"])
            for r in grid["points"]
            if r["engine"] == "pgml_gpu"
            and r.get("status") == "ok"
            and r.get("device_peak_allocated_mib")
        )
        if points:
            device.plot(
                *zip(*points),
                "D-",
                color=TOOLS["pgml_gpu"][1],
                label="pgml · device allocator",
                markersize=4,
            )
        total = grid.get("device_total_gib")
        if total:
            device.axhline(
                total * 1024,
                color="#c53030",
                linestyle=":",
                linewidth=1.4,
                label=f"card capacity ({total:.0f} GiB)",
            )
        for ax in (host, device):
            ax.set(xscale="log", yscale="log")
            ax.grid(alpha=0.2, which="both")
        host.set_title(f"{grid['label']} · {grid['rows']} rows")
        device.set_xlabel("Scenarios per batch")
    axes[0][0].set_ylabel("Host memory (MiB)")
    axes[1][0].set_ylabel("Device memory (MiB)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    extra_handles, extra_labels = axes[1][0].get_legend_handles_labels()
    fig.legend(
        handles + extra_handles,
        labels + extra_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "memory_footprint")


#: cost figure series -> tool, so the money figure uses the same colours
COST_SERIES_TOOL = {
    "pandapower": "pandapower",
    "power_grid_model": "power_grid_model",
    "opendss": "opendss",
}


def _cost_tool(series_key):
    if series_key in COST_SERIES_TOOL:
        return COST_SERIES_TOOL[series_key]
    base = series_key.rsplit("_", 1)[0]
    return _tool_of(base + "_")


def cost():
    """Cost per million solved scenarios against batch size, on one grid."""
    data = json.loads((ASSETS / "cost.json").read_text())
    block = data["cost_vs_batch"]
    best = {}
    for series_key, points in block["series"].items():
        try:
            tool = _cost_tool(series_key)
        except ValueError:
            continue
        for point in points:
            batch, value = point["batch"], point["usd_per_million"]
            if np.isfinite(value):
                current = best.setdefault(tool, {})
                current[batch] = min(current.get(batch, np.inf), value)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _draw(ax, {tool: sorted(points.items()) for tool, points in best.items()})
    ax.set(
        xlabel="Scenarios per batch",
        ylabel="USD per million solved scenarios",
        title=f"{block['grid']} · {block['rows']} rows",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "cost_per_million")


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


#: every figure, and whether a missing source file is an error. The README's own
#: figure and the resistance panel are required; the performance page's extra
#: panels are drawn when their result is published.
FIGURES = (
    (throughput, True),
    (throughput_all, False),
    (harmonic_throughput, False),
    (size_scaling, False),
    (memory, False),
    (cost, False),
    (resistance, True),
)


if __name__ == "__main__":
    for figure, required in FIGURES:
        try:
            figure()
        except FileNotFoundError as exc:
            if required:
                raise
            print(f"[skip] {figure.__name__}: {exc}")
