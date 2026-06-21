"""Connection-aware terminal incidence for WYE / DELTA / neutral load modeling.

A Load/Generator has internal ELEMENTS (per-phase impedance / current branches).
A constant real incidence ``M`` ``[n_elem, n_used]`` maps the node-phase voltages
the appliance touches (``V_used``, the ``n_used`` "used rows") to the element
("terminal") voltages::

    V_term = M @ V_used          (element voltage = the L-N / L-L difference)

The nodal admittance contribution of per-element admittances ``y_elem`` is then::

    Y_block = M^T @ diag(y_elem) @ M          (shape [n_used, n_used])

and the nodal current from per-element currents ``i_elem`` is::

    I_used = M^T @ i_elem.

Connection cases (``n = len(appliance.phases)``):

- WYE, node has NO ``Phase.N`` (return to ground): ``n_elem == n``, the used rows
  are the ``n`` phase rows, ``M = I_n``. This reduces EXACTLY to the diagonal
  const-Z / device stamp — THE regression guarantee.
- WYE, node HAS ``Phase.N`` (4-wire): ``n_elem == n``, used rows are the ``n``
  phase rows PLUS the node's ``Phase.N`` row, ``M = [I_n | -1]`` (``n x (n+1)``;
  element ``k`` is ``V_phase_k - V_N``). Current scatters into the phase rows AND
  the ``N`` row (the ``N`` row gets ``-sum_k i_k``, Kirchhoff).
- DELTA, ``n == 3``: ``n_elem == 3``, used rows are the 3 phase rows,
  ``M = [[1,-1,0],[0,1,-1],[-1,0,1]]`` (circulant difference; element ``k`` is
  between phase ``k`` and phase ``(k+1) % 3``). Same matrix as pandapower's
  ``v_del_xfmn`` (reference doc section 2). Per-phase value ``k`` maps to delta
  branch ``k`` (documented convention).
- DELTA, ``n != 3``: raises ``NotImplementedError`` (the schema already forbids
  DELTA on 1 phase; open / 2-phase delta loads are not modeled yet).

The matrices are real and constant (built once per group via ``torch.as_tensor``,
then cast to complex for the ``Y`` block); ``index_add_`` / out-of-place scatter
carry the differentiable per-element power/voltage. No autograd flows through
``M`` itself (it is a topology constant), only through ``y_elem`` / ``i_elem``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pgml.errors import ModelingError
from pgml.schemas.grid_schema import Phase, WindingConnection

from .index import NodePhaseIndex
from ._symmetry import resolve_connection


@dataclass(frozen=True)
class IncidenceGroup:
    """A group of appliances sharing one terminal incidence ``M``.

    Attributes
    ----------
    connection:
        The effective :class:`WindingConnection` (WYE or DELTA; WYE_GROUNDED is
        folded to WYE).
    n_phases:
        ``len(appliance.phases)`` shared by every appliance in the group.
    has_neutral_return:
        ``True`` iff WYE and the node carries ``Phase.N`` (4-wire return).
    appliances:
        The python schema objects in this group (deterministic order).
    n_elem:
        Number of internal elements (== ``n_phases`` for WYE/DELTA-3).
    n_used:
        Number of distinct node rows the incidence touches.
    """

    connection: WindingConnection
    n_phases: int
    has_neutral_return: bool
    appliances: list
    n_elem: int
    n_used: int


def _group_key(appliance, node_phases) -> tuple:
    """Hashable grouping key ``(connection, n_phases, has_neutral_return)``."""
    conn = resolve_connection(appliance)
    if conn in (WindingConnection.WYE, WindingConnection.WYE_GROUNDED):
        conn = WindingConnection.WYE
    n = len(appliance.phases)
    has_neutral = conn == WindingConnection.WYE and Phase.N in node_phases
    return (conn, n, has_neutral)


def group_appliances(appliances, node_map) -> list[IncidenceGroup]:
    """Group Load/Generator objects by ``(connection, n_phases, has_neutral_return)``.

    ``node_map`` maps node id -> node (for the ``Phase.N`` membership test, which is
    a property of the appliance's NODE, not its own phases). Raises for an
    unsupported DELTA arity.
    """
    by_key: dict[tuple, list] = {}
    for a in appliances:
        node = node_map[a.node]
        key = _group_key(a, node.phases)
        by_key.setdefault(key, []).append(a)

    groups: list[IncidenceGroup] = []
    for (conn, n, has_neutral), group in by_key.items():
        if conn == WindingConnection.DELTA and n != 3:
            raise ModelingError(
                "open/2-phase delta load not supported yet; use WYE or a 3-phase "
                f"DELTA (got a DELTA appliance with {n} phase(s))."
            )
        n_elem = n
        n_used = n + 1 if has_neutral else n
        groups.append(
            IncidenceGroup(
                connection=conn,
                n_phases=n,
                has_neutral_return=has_neutral,
                appliances=group,
                n_elem=n_elem,
                n_used=n_used,
            )
        )
    return groups


def build_incidence(grp: IncidenceGroup, rdt: torch.dtype, device) -> Tensor:
    """Real incidence ``M`` ``[n_elem, n_used]`` for ``grp`` (constant topology).

    WYE-ground -> ``I_n``; WYE-neutral -> ``[I_n | -1]``; DELTA-3 -> the circulant
    difference matrix. Built with ``torch.as_tensor`` (no autograd through ``M``).
    """
    n = grp.n_phases
    eye = torch.eye(n, dtype=rdt, device=device)
    if grp.connection == WindingConnection.DELTA:
        # element k = phase_k - phase_{(k+1)%3}; rows are elements, cols phases.
        m = torch.zeros((3, 3), dtype=rdt, device=device)
        rows = torch.arange(3, device=device)
        m[rows, rows] = 1.0
        m[rows, (rows + 1) % 3] = -1.0
        return m
    # WYE
    if grp.has_neutral_return:
        neg_one = -torch.ones((n, 1), dtype=rdt, device=device)
        return torch.cat([eye, neg_one], dim=-1)  # [n, n+1]
    return eye  # [n, n]


def used_rows(grp: IncidenceGroup, index: NodePhaseIndex, device) -> Tensor:
    """Global node-rows each appliance's incidence touches: ``[K, n_used]`` int64.

    For WYE-neutral the last column is the node's ``Phase.N`` row; otherwise the
    columns are exactly the appliance's phase rows (in ``phases`` order).
    """
    idx_rows: list[list[int]] = []
    for a in grp.appliances:
        phase_rows = [index.row(a.node, ph) for ph in a.phases]
        if grp.has_neutral_return:
            phase_rows = phase_rows + [index.row(a.node, Phase.N)]
        idx_rows.append(phase_rows)
    return torch.as_tensor(idx_rows, dtype=torch.int64, device=device)


__all__ = [
    "IncidenceGroup",
    "group_appliances",
    "build_incidence",
    "used_rows",
]
