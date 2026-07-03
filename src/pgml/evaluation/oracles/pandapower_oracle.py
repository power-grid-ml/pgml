"""pandapower oracle adapters: Ybus and voltage profile as evaluation data containers.

Adapts pandapower's internal ``Ybus`` (pu -> SI) and ``res_bus`` voltage results
into :class:`~pgml.evaluation.data.LabeledMatrix` and
:class:`~pgml.evaluation.data.VoltageProfile` respectively.  pandapower is imported
lazily inside functions; only ``numpy`` is imported at module level.

Importing this module does NOT require pandapower — the dependency is checked only
when the functions are called.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pgml.schemas.grid_schema import Phase

from pgml.evaluation.data import LabeledMatrix, VoltageProfile, row_labels
from pgml.evaluation.topology import distance_from_slack


def _numpy_shim() -> None:
    """numpy 2.x compatibility shim required by pandapower 2.14 (Inf/in1d)."""
    np.Inf = np.inf  # type: ignore[attr-defined]
    np.in1d = np.isin  # type: ignore[attr-defined]


def pandapower_ybus(
    net, grid, id_map: dict, index, *, label: str = "pandapower"
) -> LabeledMatrix:
    """pandapower internal Ybus (pu -> SI siemens), aligned to our node·phase rows.

    This is the PURE NETWORK admittance (lines + explicit shunts, no const-Z load /
    source Norton shunts) — compare it to our ``assemble_network_ybus``.
    """
    _numpy_shim()
    y_pu = net._ppc["internal"]["Ybus"].toarray()
    base_mva = float(net._ppc["baseMVA"])
    # MATPOWER mixed per-unit: each bus carries its own voltage base (ppc bus
    # column 9, BASE_KV, line-to-line), so Y_SI[i, j] = Y_pu[i, j] * S_base /
    # (V_base_i * V_base_j). On a single voltage level this reduces to the
    # familiar S_base / V_base^2; across a transformer boundary the two buses
    # have different bases and a scalar base would produce wrong admittances.
    base_kv = np.asarray(net._ppc["bus"][:, 9], dtype=float)
    y_si = y_pu * (base_mva / np.outer(base_kv, base_kv))
    bus_lookup = net._pd2ppc_lookups["bus"]

    n = index.size
    out = np.zeros((n, n), dtype=complex)
    for pp_i, node_i in id_map["bus"].items():
        ri = index.row(node_i, Phase.A)
        ppi = int(bus_lookup[pp_i])
        for pp_j, node_j in id_map["bus"].items():
            rj = index.row(node_j, Phase.A)
            ppj = int(bus_lookup[pp_j])
            out[ri, rj] = y_si[ppi, ppj]
    return LabeledMatrix(matrix=out, label=label, row_labels=row_labels(index))


def pandapower_voltage_profile(
    net,
    grid,
    id_map: dict,
    *,
    label: str = "pandapower",
    slack: Optional[int] = None,
) -> VoltageProfile:
    """Voltage profile from a solved pandapower net (``res_bus.vm_pu`` is already pu)."""
    dist = distance_from_slack(grid, slack)
    ds, pus, nids = [], [], []
    for pp_bus, node_id in id_map["bus"].items():
        ds.append(dist[int(node_id)])
        pus.append(float(net.res_bus.at[pp_bus, "vm_pu"]))
        nids.append(int(node_id))
    order = np.argsort(ds)
    return VoltageProfile(
        distances_km=np.asarray(ds)[order],
        v_pu=np.asarray(pus)[order],
        label=label,
        node_ids=np.asarray(nids)[order],
    )


__all__ = [
    "pandapower_ybus",
    "pandapower_voltage_profile",
]
