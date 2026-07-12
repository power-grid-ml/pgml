"""Multi-grid batching: solve an ensemble of independent grids in ONE call.

A disjoint union of grids is itself a valid :class:`~pgml.schemas.grid_schema.Grid`:
no branch connects the members, so the assembled admittance is exactly the
block-diagonal ``Y = diag(Y_1, …, Y_G)`` — no special solver support is needed, and
the sparse factorization backend handles the union in ~O(Σ nnz) where a dense LU
would pay O((Σ N)³). Every member keeps its own :class:`Source`, so the pre-solve
connectivity check passes per member, ideal-slack rows pin per member, and the
whole pgml pipeline (``solve_power_flow`` / ``solve_harmonic_flow`` / batched
operating points / ``branch_states`` / ``prepare_power_flow``) applies unchanged.

:func:`merge_grids` builds that union with per-member id remapping and returns a
:class:`MergedGrid` that translates between member-local and merged identifiers:

>>> merged = merge_grids([g1, g2, g3])
>>> op = merged.operating_point([op1, op2, op3])       # member-local appliance ids
>>> res = solve_power_flow(merged.grid, operating_point=op)
>>> v1, v2, v3 = merged.split(res.v)                    # per-member row slices

The merged grid SHARES the members' parameter objects (and any tensor leaves they
hold), so gradients computed through a merged solve flow back to the ORIGINAL
grids' leaf tensors — merging is transparent to the differentiable path.

Semantics to be aware of (documented, deliberate):

- All members must share ``base_frequency_hz`` and be materialised consistently
  (identical ``types`` entries may repeat across members; conflicting definitions
  under one name raise).
- Convergence is evaluated on the UNION state vector: the fixed point iterates all
  members together and the absolute ``tol`` applies to the concatenated update
  norm, so one hard member keeps iterating an already-settled easy member (cheap —
  the extra iterations are back-substitutions). Per-scenario ``converged_mask``
  semantics are unchanged.
- The calculation symmetry resolves once for the union: one asymmetric member
  makes the whole batch solve asymmetric (correct for every member, marginally
  more work for the symmetric ones).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from torch import Tensor

from pgml.errors import InputError
from pgml.schemas.grid_schema import Grid, GridMetadata, TypeLibrary


@dataclass(frozen=True)
class GridMember:
    """One member of a :class:`MergedGrid`: its row slice and id translations.

    Attributes
    ----------
    index:
        Position of the member in the ``merge_grids`` input sequence.
    row_start, n_rows:
        The member's node-phase rows occupy ``[row_start, row_start + n_rows)`` of
        the merged state vector (members are contiguous, in input order).
    node_ids, branch_ids, appliance_ids:
        Member-local id -> merged id (the first member keeps its ids; later
        members are offset per kind to stay disjoint).
    """

    index: int
    row_start: int
    n_rows: int
    node_ids: dict[int, int]
    branch_ids: dict[int, int]
    appliance_ids: dict[int, int]


@dataclass(frozen=True)
class MergedGrid:
    """A disjoint-union grid plus the member bookkeeping to address into it."""

    grid: Grid
    members: tuple[GridMember, ...]

    @property
    def n_rows(self) -> int:
        """Total node-phase rows of the merged state vector."""
        last = self.members[-1]
        return last.row_start + last.n_rows

    def split(self, v: Tensor) -> tuple[Tensor, ...]:
        """Slice a merged state ``[..., N_total]`` into per-member views.

        Works on any solved quantity whose LAST axis is the merged node-phase row
        axis (``PowerFlowResult.v`` ``[*batch, N]``, ``HarmonicFlowResult.v``
        ``[*batch, H, N]``, residuals, …). The returned tensors are views
        (``narrow``), so gradients flow through unchanged.
        """
        if v.shape[-1] != self.n_rows:
            raise InputError(
                f"split expects the merged row axis last (N={self.n_rows}); "
                f"got a tensor with last dim {v.shape[-1]}."
            )
        return tuple(v.narrow(-1, m.row_start, m.n_rows) for m in self.members)

    def operating_point(self, per_member: Sequence[Optional[dict]]) -> Optional[dict]:
        """Merge per-member operating points (member-local appliance ids) into one.

        ``per_member`` aligns with :attr:`members`; ``None`` entries contribute
        nothing. Values pass through untouched (floats, tensors, batched tensors —
        the usual ``operating_point`` conventions apply on the merged solve).
        """
        return self._remap("appliance_ids", per_member, "operating_point")

    def branch_states(self, per_member: Sequence[Optional[dict]]) -> Optional[dict]:
        """Merge per-member ``branch_states`` (member-local branch ids) into one."""
        return self._remap("branch_ids", per_member, "branch_states")

    def _remap(
        self, id_attr: str, per_member: Sequence[Optional[dict]], what: str
    ) -> Optional[dict]:
        if len(per_member) != len(self.members):
            raise InputError(
                f"{what} expects one entry per member "
                f"({len(self.members)}), got {len(per_member)}."
            )
        out: dict = {}
        for member, entry in zip(self.members, per_member):
            if not entry:
                continue
            ids = getattr(member, id_attr)
            for local_id, value in entry.items():
                merged = ids.get(int(local_id))
                if merged is None:
                    raise InputError(
                        f"{what}: member {member.index} has no "
                        f"{id_attr[:-4]} with id {local_id}."
                    )
                out[merged] = value
        return out or None


def _merged_types(grids: Sequence[Grid]) -> TypeLibrary:
    """Union of the members' type catalogs; a name may repeat only identically."""
    lines: dict = {}
    transformers: dict = {}
    for gi, g in enumerate(grids):
        for target, source in (
            (lines, g.types.lines),
            (transformers, g.types.transformers),
        ):
            for name, entry in source.items():
                if name in target and target[name] != entry:
                    raise InputError(
                        f"merge_grids: type {name!r} of member {gi} conflicts with "
                        "an earlier member's definition; materialise the grids "
                        "(expand type_ref) or rename the type before merging."
                    )
                target[name] = entry
    return TypeLibrary(lines=lines, transformers=transformers)


def merge_grids(grids: Sequence[Grid], *, name: Optional[str] = None) -> MergedGrid:
    """Disjoint-union ``grids`` into one solvable :class:`Grid` with id bookkeeping.

    Node / branch / appliance ids are remapped per kind so they stay unique: the
    FIRST member keeps its ids, later members are shifted by a per-kind offset
    (the :class:`GridMember` maps record every translation). Element objects are
    rebuilt only where an id or node reference changes; all physical parameters —
    including tensor leaves — are SHARED with the originals, so a merged solve is
    differentiable w.r.t. the member grids' own parameters.

    Requirements: at least one member; identical ``base_frequency_hz``; type
    catalogs may overlap only with identical entries. Each member should carry its
    own in-service :class:`Source` — the solvers' connectivity check will
    otherwise name the unenergized member nodes.
    """
    if not grids:
        raise InputError("merge_grids needs at least one grid.")
    f0 = float(grids[0].base_frequency_hz)
    for gi, g in enumerate(grids[1:], 1):
        if float(g.base_frequency_hz) != f0:
            raise InputError(
                f"merge_grids: member {gi} has base_frequency_hz="
                f"{float(g.base_frequency_hz)} but member 0 has {f0}; an ensemble "
                "solves at ONE fundamental frequency."
            )

    nodes: list = []
    branches: list = []
    appliances: list = []
    members: list[GridMember] = []
    next_node = next_branch = next_appliance = 0
    row_start = 0
    for gi, g in enumerate(grids):
        node_ids: dict[int, int] = {}
        branch_ids: dict[int, int] = {}
        appliance_ids: dict[int, int] = {}

        node_off = 0 if gi == 0 else next_node - min(int(n.id) for n in g.nodes)
        for nd in g.nodes:
            new_id = int(nd.id) + node_off
            node_ids[int(nd.id)] = new_id
            nodes.append(
                nd if new_id == nd.id else nd.model_copy(update={"id": new_id})
            )
        next_node = max((int(n.id) for n in nodes), default=-1) + 1

        branch_off = (
            0
            if gi == 0
            else (next_branch - min((int(b.id) for b in g.branches), default=0))
        )
        for b in g.branches:
            new_id = int(b.id) + branch_off
            branch_ids[int(b.id)] = new_id
            update = {
                "from_node": node_ids[int(b.from_node)],
                "to_node": node_ids[int(b.to_node)],
            }
            if new_id != b.id:
                update["id"] = new_id
            branches.append(b.model_copy(update=update))
        next_branch = max((int(b.id) for b in branches), default=-1) + 1

        appliance_off = (
            0
            if gi == 0
            else (next_appliance - min((int(a.id) for a in g.appliances), default=0))
        )
        for a in g.appliances:
            new_id = int(a.id) + appliance_off
            appliance_ids[int(a.id)] = new_id
            update = {"node": node_ids[int(a.node)]}
            if new_id != a.id:
                update["id"] = new_id
            appliances.append(a.model_copy(update=update))
        next_appliance = max((int(a.id) for a in appliances), default=-1) + 1

        n_rows = sum(len(nd.phases) for nd in g.nodes)
        members.append(
            GridMember(
                index=gi,
                row_start=row_start,
                n_rows=n_rows,
                node_ids=node_ids,
                branch_ids=branch_ids,
                appliance_ids=appliance_ids,
            )
        )
        row_start += n_rows

    merged = Grid(
        base_frequency_hz=f0,
        nodes=nodes,
        branches=branches,
        appliances=appliances,
        types=_merged_types(grids),
        metadata=GridMetadata(
            name=name or f"merged[{len(grids)}]",
            description=f"Disjoint union of {len(grids)} member grids "
            "(pgml.multigrid.merge_grids).",
        ),
    )
    return MergedGrid(grid=merged, members=tuple(members))


__all__ = ["GridMember", "MergedGrid", "merge_grids"]
