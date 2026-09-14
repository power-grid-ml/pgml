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


def throughput():
    data = json.loads((ASSETS / "batch_throughput.json").read_text())
    curves = [
        ("series", "cpu_dense_c128", "pgml · CPU dense", "#2563eb", "o-"),
        ("series", "cpu_sparse_c128", "pgml · CPU sparse", "#0891b2", "s-"),
        ("series", "gpu_dense_c128", "pgml · GPU", "#d97706", "o-"),
        ("baselines", "pandapower", "pandapower · CPU", "#7c3aed", "^--"),
        ("baselines", "power_grid_model", "power-grid-model · CPU", "#15803d", "v--"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharey=True)
    for ax, (name, title) in zip(
        axes, [("ieee33", "IEEE 33-bus feeder"), ("kerber", "Kerber · 294 buses")]
    ):
        grid = data["grids"][name]
        for section, key, label, color, style in curves:
            records = grid[section][key]
            for row in records:
                valid = np.isfinite(row.get("throughput", np.nan))
                if section == "series":
                    valid &= row.get("verdict") == "ok" and row.get("converged") is True
                    error = row.get("max_dV_pu", np.nan)
                else:
                    valid &= row.get("status") == "ok"
                    valid &= row.get("comparison_scenarios") == row["batch"]
                    error = row.get("max_dV_vs_pgml_cpu_c128_pu", np.nan)
                if not valid or not np.isfinite(error) or error > 1e-6:
                    raise ValueError(f"Invalid comparison point: {name}/{key}/{row}")
            ax.plot(
                [r["batch"] for r in records],
                [r["throughput"] for r in records],
                style,
                color=color,
                label=label,
                linewidth=1.8,
                markersize=4,
            )
        ax.set(xscale="log", yscale="log", xlabel="Scenarios per batch", title=title)
        ax.grid(alpha=0.2, which="both")
    axes[0].set_ylabel("Solved scenarios / second")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(
        "Fundamental power flow · double precision · one GPU / eight allocated CPUs",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.96))
    save(fig, "batch_throughput")


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
    resistance()
