"""Per-node harmonic "error"-source sweep (the transient-injection study).

Sweeps a per-node harmonic source (the Thévenin/Norton source of
``references/error_injection.md``) over a set of nodes — one node per scenario — and
returns the batched node voltages, so you can map how a harmonic injected at each node
spreads through the grid. Unlike :func:`~pgml.scenarios.spectrum_sweep` (a device Norton
current tied to a load's fundamental), this injects at ANY node of a user-set strength
``source_power_va`` and leaves the fundamental exact.

The source modifies ``Y(h)`` (voltage kind adds a shunt at the node), so the sweep can't
share a single batched ``Y``; it LOOPS ``solve_harmonic_flow`` once per node (cheap for
benchmark feeders) and stacks the results into ``[B, H, N]``.
"""

from __future__ import annotations

import torch

from pgml.schemas.grid_schema import Grid
from pgml.solver import NodeHarmonicSource, solve_harmonic_flow

from .config import NodeInjectionSweepConfig
from .run import ScenarioResult
from .sampler import SampledScenarios


def run_node_injection_sweep(
    grid: Grid,
    config: NodeInjectionSweepConfig,
    *,
    slack: str = "norton",
    dtype: torch.dtype = torch.complex128,
    device=None,
) -> ScenarioResult:
    """Sweep the per-node harmonic source over ``config.node_ids`` (default: all nodes).

    Returns a :class:`ScenarioResult` with ``v`` ``[B, H, N]`` (B = #swept nodes, the
    fundamental is order 1 of H = ``[1, *config.orders]``) and the swept node id per
    scenario in ``sampled.samples["<name>_node_id"]``.
    """
    node_ids = (
        list(config.node_ids)
        if config.node_ids is not None
        else [int(n.id) for n in grid.nodes]
    )
    node_by_id = {int(n.id): n for n in grid.nodes}
    spectrum = {
        o: (m, p)
        for o, m, p in zip(config.orders, config.magnitudes_pu, config.phases_deg)
    }
    orders = [1, *config.orders]

    vs = []
    index = freqs = None
    for nid in node_ids:
        phases = (
            tuple(config.phases) if config.phases else tuple(node_by_id[nid].phases)
        )
        src = NodeHarmonicSource(
            node_id=nid,
            phases=phases,
            spectrum=spectrum,
            source_power_va=config.source_power_va,
            kind=config.kind,
        )
        res = solve_harmonic_flow(
            grid, orders, node_sources=[src], slack=slack, dtype=dtype, device=device
        )
        vs.append(res.v)  # [H, N]
        index, freqs = res.index, res.frequencies_hz

    v = torch.stack(vs, dim=0)  # [B, H, N]
    sampled = SampledScenarios(
        operating_point={},
        samples={f"{config.name}_node_id": torch.tensor(node_ids, dtype=torch.long)},
        n_samples=len(node_ids),
        config=config,
    )
    return ScenarioResult(v=v, index=index, sampled=sampled, frequencies_hz=freqs)


__all__ = ["run_node_injection_sweep"]
