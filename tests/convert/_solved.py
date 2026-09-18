"""Solve one model in pgml and in a reference tool, and compare node voltages.

Shared by the conversion fidelity tests. Every helper returns complex per-unit
positive-sequence node voltages keyed by pgml node id, so an import test and an
export test compare the same quantity.
"""

from __future__ import annotations

import cmath
import math

import numpy as np

from pgml.schemas.grid_schema import Phase


def pgml_node_voltages_pu(grid, **solve_kwargs) -> dict[int, complex]:
    """Phase-A (or positive-sequence) voltage of every node in per unit."""
    from pgml.solver import solve_power_flow

    result = solve_power_flow(grid, **solve_kwargs)
    assert bool(result.converged), "pgml power flow did not converge"
    v = result.v.detach().cpu().numpy().reshape(-1)
    out = {}
    for node in grid.nodes:
        row = result.index.row(node.id, Phase.A)
        base = float(node.u_rated_v)
        if len(node.phases) >= 3:
            base /= math.sqrt(3.0)
        out[node.id] = complex(v[row]) / base
    return out


def pandapower_node_voltages_pu(exported, **runpp_kwargs) -> dict[int, complex]:
    """Solve an exported pandapower net; voltages keyed by pgml node id."""
    import pandapower as pp

    runpp_kwargs.setdefault("numba", False)
    runpp_kwargs.setdefault("tolerance_mva", 1e-12)
    runpp_kwargs.setdefault("trafo_model", "pi")
    pp.runpp(exported.net, **runpp_kwargs)
    res = exported.net.res_bus
    return {
        node: cmath.rect(
            float(res.at[bus, "vm_pu"]), math.radians(float(res.at[bus, "va_degree"]))
        )
        for node, bus in exported.bus_of_node.items()
    }


def pgm_node_voltages_pu(exported) -> dict[int, complex]:
    """Solve an exported power-grid-model case; voltages keyed by pgml node id."""
    from power_grid_model import CalculationMethod, PowerGridModel

    model = PowerGridModel(exported.input_data, system_frequency=exported.frequency_hz)
    out = model.calculate_power_flow(
        calculation_method=CalculationMethod.newton_raphson,
        error_tolerance=1e-12,
        max_iterations=50,
    )
    by_id = {
        int(i): cmath.rect(float(u), float(a))
        for i, u, a in zip(
            out["node"]["id"], out["node"]["u_pu"], out["node"]["u_angle"]
        )
    }
    return {node: by_id[pid] for node, pid in exported.pgm_of_node.items()}


def max_voltage_error(a: dict[int, complex], b: dict[int, complex]) -> float:
    """Largest complex per-unit voltage difference over the shared nodes."""
    shared = sorted(set(a) & set(b))
    assert shared, "no shared nodes to compare"
    return float(np.max([abs(a[n] - b[n]) for n in shared]))
