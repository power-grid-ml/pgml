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
    fig.savefig(ASSETS / f"{stem}.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(Path("/tmp") / f"pgml-readme-{stem}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def throughput():
    data = json.loads((ASSETS / "batch_throughput.json").read_text())
    grid = data["grids"]["ieee33"]
    fig, ax = plt.subplots(figsize=(10.8, 3.5), layout="constrained")
    for key, label, color in [
        ("cpu_dense_c128", "pgml · CPU", "#2563eb"),
        ("gpu_dense_c128", "pgml · GPU", "#d97706"),
    ]:
        records = grid["series"][key]
        valid = [
            r
            for r in records
            if r.get("verdict") == "ok"
            and r.get("converged") is True
            and np.isfinite(r.get("throughput", np.nan))
            and np.isfinite(r.get("max_dV_pu", np.nan))
            and r["max_dV_pu"] <= 1e-6
        ]
        if not valid:
            raise ValueError(f"No valid throughput points for {key}")
        ax.plot(
            [r["batch"] for r in valid],
            [r["throughput"] for r in valid],
            "o-",
            color=color,
            label=label,
            linewidth=2,
            markersize=5,
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Scenarios per batch",
        ylabel="Solved scenarios / second",
        title="Batching one IEEE 33-bus feeder · double precision",
    )
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False)
    date = data["environment"]["date"][:10]
    provenance = json.loads((ASSETS / "provenance.json").read_text())
    suffix = (
        ""
        if provenance.get("performance_current_review", False)
        else " · refresh pending"
    )
    fig.supxlabel(
        f"Recorded {date} · Core i7-12700 / RTX A2000 12 GB{suffix}", fontsize=10
    )
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
