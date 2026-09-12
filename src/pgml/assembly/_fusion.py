"""Exact bus fusion for zero-impedance branches (ideal closed switches, jumpers).

A branch whose series impedance is EXACTLY zero — a closed switch with no impedance
data, a bus coupler or jumper modelled as a zero-impedance line, a zero-length line —
has no primitive admittance: the nodal formulation inverts the series impedance, and
``Z = 0`` has no inverse. The branch is nevertheless a perfectly ordinary network
element: it constrains the voltages of its two terminals to be equal.

This module expresses that constraint exactly, by COLLAPSING the terminal node-phase
rows of such a branch into ONE row of the solved system::

    V = P v_red          (prolongation: fused rows share one unknown)
    P^T Y P v_red = P^T I  (restriction: the fused rows' current balances are summed)

``P`` is the 0/1 matrix of the fusion partition. It is never materialised: a
:class:`FusionMap` carries a REDUCED :class:`~pgml.assembly.index.NodePhaseIndex` whose
``(node, phase) -> row`` map is many-to-one, so every stamp, injection and incidence
path that indexes through an index scatters straight into the reduced system — which IS
``P^T Y P`` — and the prolongation back to the full row layout is one gather.

Consequences, all exact rather than approximate:

- the fused branch itself is NOT stamped (it has no admittance to stamp), and its
  parameters are not parameters of the solve, so gradients w.r.t. them are structurally
  zero;
- the reduced system is better conditioned than any small-impedance stand-in, which
  buys its finite voltage drop with a row scale of ``1/R``;
- results are reported on the ORIGINAL node ids: the fused nodes share one voltage, and
  the current through a fused branch follows from Kirchhoff's current law at the fused
  node (:func:`fused_branch_currents`).

Which branches fuse: a :class:`~pgml.schemas.grid_schema.Switch` (closed),
:class:`~pgml.schemas.grid_schema.Line` or
:class:`~pgml.schemas.grid_schema.GenericBranch` in service whose EFFECTIVE series
resistance and inductance (schema value or ``param_overrides`` substitution) are both
exactly zero and whose shunt admittance is zero, and which is NOT listed in
``branch_states`` (a swept branch has to stay a stamped branch — see
:func:`fusion_map`). A :class:`~pgml.schemas.grid_schema.Transformer` is never fused:
its ratio and vector group relate the two terminals by more than equality.

This module is structural bookkeeping (python ints, int64 index tensors and small
constant real matrices built under ``no_grad``), so plain python loops over nodes and
branches are fine here — the differentiable tensor path stays vectorized.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from pgml import defaults
from pgml.errors import InputError, ModelingError
from pgml.schemas.grid_schema import (
    GenericBranch,
    Grid,
    Line,
    Phase,
    Source,
    Switch,
    Transformer,
)

from .index import NodePhaseIndex, node_phase_index

_log = logging.getLogger(__name__)

#: Branch kinds whose zero-impedance form is an ideal conductor between its terminals.
FUSABLE_COMPONENTS = ("line", "switch", "generic_branch")

_PHASE_CODE = {Phase.A: 0, Phase.B: 1, Phase.C: 2, Phase.N: 3}


# ---------------------------------------------------------------------------
# zero-impedance detection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ZeroImpedanceBranch:
    """One branch with no primitive admittance, and whether it can be fused.

    Attributes
    ----------
    branch_id:
        Id of the offending branch.
    component:
        Its schema ``component`` discriminator (``"line"``, ``"switch"``, …).
    detail:
        Why it has no stamp, in words (for the error / log message).
    fusable:
        ``True`` when collapsing its terminals into one row reproduces the element
        exactly (an ideal conductor). ``False`` for a transformer (a ratio and a vector
        group are more than equality) and for a zero-series branch that still carries a
        shunt admittance (fusing would drop that shunt).
    """

    branch_id: int
    component: str
    detail: str
    fusable: bool


class _NoFusion:
    """Type of :data:`NO_FUSION`."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "NO_FUSION"


#: A fusion resolution that produced no map: this grid carries nothing to collapse, and
#: that answer is already known. Pass it as ``fusion=`` to an assembly entry point
#: (:func:`~pgml.assembly.assemble_ybus`,
#: :func:`~pgml.assembly.assemble_network_ybus`) instead of ``None``, which means "not
#: resolved yet" and makes the assembler walk the branch list again to rediscover the
#: same empty answer. A solve resolves the zero-impedance structure once and hands it
#: down in this form, so the walk is paid once per solve rather than once per assembly.
NO_FUSION = _NoFusion()


def _is_zero(*values) -> bool:
    """Are all given parameter values exactly zero (``None`` counts as zero)?

    Reads tensor-valued (autograd-carrying) parameters under ``no_grad`` — this is a
    structural decision about the network, never a quantity on the tape.
    """
    for v in values:
        if v is None:
            continue
        if isinstance(v, (int, float)):
            if float(v) != 0.0:
                return False
            continue
        if isinstance(v, (list, tuple)):
            if not _is_zero(*v):
                return False
            continue
        with torch.no_grad():
            t = torch.as_tensor(v)
            if t.numel() and float(t.abs().max()) != 0.0:
                return False
    return True


def _effective(
    param_overrides: Optional[dict], kind: str, bid: int, field_: str, value
):
    """The parameter a stamp would actually use: a ``param_overrides`` entry wins."""
    if param_overrides:
        key = (kind, int(bid), field_)
        if key in param_overrides:
            return param_overrides[key]
    return value


def zero_impedance_branches(
    grid: Grid, *, param_overrides: Optional[dict] = None
) -> list[ZeroImpedanceBranch]:
    """Every in-service branch whose EFFECTIVE series impedance is exactly zero.

    The effective value is the one the stamp would use: a ``param_overrides`` entry for
    the branch's resistance / inductance overrides the schema value, so a gradcheck that
    substitutes a finite impedance for a zero-impedance branch keeps it stamped (and a
    substitution of zero makes an ordinary branch fuse).

    Returns one :class:`ZeroImpedanceBranch` per offending branch in ``grid.branches``
    order, each flagged ``fusable`` or not. Pure bookkeeping; values are read under
    ``no_grad``.
    """
    out: list[ZeroImpedanceBranch] = []
    for b in grid.branches:
        if not getattr(b, "in_service", True):
            continue
        bid = int(b.id)
        if isinstance(b, Line):
            if b.conductor_geometry is not None:
                continue  # the geometry path always yields a finite impedance
            r = _effective(
                param_overrides,
                "line",
                bid,
                "series_resistance_ohm_per_m",
                b.series_resistance_ohm_per_m,
            )
            ll = _effective(
                param_overrides,
                "line",
                bid,
                "series_inductance_h_per_m",
                b.series_inductance_h_per_m,
            )
            zero_length = _is_zero(b.length_m)
            if not (zero_length or _is_zero(r, ll)):
                continue
            g = _effective(
                param_overrides,
                "line",
                bid,
                "shunt_conductance_s_per_m",
                b.shunt_conductance_s_per_m,
            )
            c = _effective(
                param_overrides,
                "line",
                bid,
                "shunt_capacitance_f_per_m",
                b.shunt_capacitance_f_per_m,
            )
            shunt_zero = zero_length or _is_zero(g, c)
            detail = (
                "length_m is zero"
                if zero_length
                else "series R and L per metre are both zero"
            )
            if not shunt_zero:
                detail += " while its shunt G/C is not"
            out.append(ZeroImpedanceBranch(bid, "line", detail, shunt_zero))
        elif isinstance(b, Switch):
            if not b.closed:
                continue
            r = _effective(
                param_overrides, "switch", bid, "resistance_ohm", b.resistance_ohm
            )
            ll = _effective(
                param_overrides, "switch", bid, "inductance_h", b.inductance_h
            )
            if not _is_zero(r, ll):
                continue
            shunt_zero = _is_zero(b.shunt_conductance_s, b.shunt_capacitance_f)
            detail = "closed with zero resistance and inductance"
            if not shunt_zero:
                detail += " while its shunt G/C is not"
            out.append(ZeroImpedanceBranch(bid, "switch", detail, shunt_zero))
        elif isinstance(b, GenericBranch):
            r = _effective(
                param_overrides,
                "generic_branch",
                bid,
                "series_resistance_ohm",
                b.series_resistance_ohm,
            )
            ll = _effective(
                param_overrides,
                "generic_branch",
                bid,
                "series_inductance_h",
                b.series_inductance_h,
            )
            if not _is_zero(r, ll):
                continue
            shunt_zero = _is_zero(b.shunt_capacitance_from_f, b.shunt_capacitance_to_f)
            detail = "series R and L are both zero"
            if not shunt_zero:
                detail += " while its shunt C is not"
            out.append(ZeroImpedanceBranch(bid, "generic_branch", detail, shunt_zero))
        elif isinstance(b, Transformer):
            r = _effective(
                param_overrides,
                "transformer",
                bid,
                "series_resistance_ohm",
                b.series_resistance_ohm,
            )
            ll = _effective(
                param_overrides,
                "transformer",
                bid,
                "series_inductance_h",
                b.series_inductance_h,
            )
            if _is_zero(r, ll):
                out.append(
                    ZeroImpedanceBranch(
                        bid,
                        "transformer",
                        "series (leakage) R and L are both zero",
                        False,
                    )
                )
    return out


def describe_unfusable(
    bad: Sequence[ZeroImpedanceBranch], *, fused_available: bool = True
) -> str:
    """The message naming zero-impedance branches that cannot be collapsed."""
    r_ideal = float(defaults.get("branch.near_ideal_series_resistance_ohm"))
    shown = "; ".join(f"{z.component} {z.branch_id} ({z.detail})" for z in bad[:8])
    more = "" if len(bad) <= 8 else f" (+{len(bad) - 8} more)"
    ways = (
        f"Give it a small finite series impedance (the documented near-ideal value is "
        f"{r_ideal:g} Ohm), or merge the two nodes it joins into one"
    )
    if fused_available:
        ways += (
            ", or — where the element really is an ideal conductor — model it as a "
            "closed Switch / zero-impedance Line, whose terminals the solve collapses "
            "into one row exactly"
        )
    return (
        f"{len(bad)} branch(es) have zero series impedance and no primitive admittance, "
        f"and cannot be collapsed: {shown}{more}. {ways}."
    )


# ---------------------------------------------------------------------------
# the fusion map
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FusionMap:
    """The exact node-phase fusion of a grid's zero-impedance branches.

    ``index`` is the REDUCED row layout the solve runs on (``M <= N`` rows); every other
    member relates it to the full layout ``full_index`` (``N`` rows), which is what
    results are reported on.

    Attributes
    ----------
    size:
        Number of reduced rows ``M``.
    index:
        Reduced :class:`~pgml.assembly.index.NodePhaseIndex`: its
        ``(node, phase) -> row`` map is MANY-TO-ONE (fused rows share a row), so
        stamping through it accumulates ``P^T Y P`` directly. ``node_ids`` /
        ``phase_codes`` / :meth:`~pgml.assembly.index.NodePhaseIndex.node_id_of` report
        each reduced row's REPRESENTATIVE (node, phase).
    full_index:
        The unreduced layout (``N`` rows) the grid's own
        :func:`~pgml.assembly.index.node_phase_index` produces.
    row_to_reduced:
        int64 ``[N]`` — reduced row of every full row (the diagonal of ``P``).
    representative:
        int64 ``[M]`` — the first full row of each reduced row.
    fused_branch_ids:
        Ids of the branches collapsed, in ``grid.branches`` order.
    groups:
        One entry per reduced row that carries MORE than one full row: the full rows
        it holds, ascending. The diagnostics' "which nodes were fused".
    indeterminate_branch_ids:
        Fused branches whose own current is not determined by the solution (a loop of
        fused branches in parallel, or a fused group holding two slack terminals).
        Their recovered current is the minimum-norm choice.
    """

    size: int
    index: NodePhaseIndex
    full_index: NodePhaseIndex
    row_to_reduced: Tensor
    representative: Tensor
    fused_branch_ids: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    indeterminate_branch_ids: tuple[int, ...] = ()
    # Current-recovery structure (see `fused_branch_currents`), padded per group.
    _group_rows: Tensor = field(
        default_factory=lambda: torch.zeros(0, 0, dtype=torch.int64), repr=False
    )
    _group_coeff: Tensor = field(
        default_factory=lambda: torch.zeros(0, 0, 0), repr=False
    )
    _edge_slots: Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.int64), repr=False
    )
    _edge_branches: tuple[tuple[int, int], ...] = field(default=(), repr=False)

    # -- layout transforms ---------------------------------------------------
    def prolong(self, v_red: Tensor) -> Tensor:
        """Reduced state ``[..., M]`` -> full row layout ``[..., N]`` (``V = P v``).

        A gather: every full row reads the value of its reduced row, so fused rows share
        one voltage exactly. Differentiable (its adjoint is :meth:`restrict`), device and
        dtype follow ``v_red``.
        """
        if v_red.shape[-1] != self.size:
            raise InputError(
                f"prolong expects the reduced row axis last (M={self.size}); got "
                f"{v_red.shape[-1]}."
            )
        return v_red.index_select(-1, self.row_to_reduced.to(v_red.device))

    def restrict(self, x_full: Tensor) -> Tensor:
        """Full-layout quantity ``[..., N]`` -> reduced ``[..., M]`` by SUMMING (``P^T x``).

        The adjoint of :meth:`prolong`, and the right reduction for an extensive
        quantity (a current injection, a residual): the fused rows' current balances add
        up. Out-of-place ``index_add`` so gradients flow.
        """
        if x_full.shape[-1] != self.full_index.size:
            raise InputError(
                f"restrict expects the full row axis last (N={self.full_index.size}); "
                f"got {x_full.shape[-1]}."
            )
        out = torch.zeros(
            (*x_full.shape[:-1], self.size), dtype=x_full.dtype, device=x_full.device
        )
        return out.index_add(-1, self.row_to_reduced.to(x_full.device), x_full)

    def sample(self, x_full: Tensor) -> Tensor:
        """Full-layout quantity ``[..., N]`` -> reduced ``[..., M]`` by SAMPLING.

        The right reduction for an INTENSIVE quantity whose fused entries agree (a
        solved voltage, a per-row voltage base): read the representative row. Exact for
        a voltage, because fusion makes the group's voltages equal.
        """
        if x_full.shape[-1] != self.full_index.size:
            raise InputError(
                f"sample expects the full row axis last (N={self.full_index.size}); "
                f"got {x_full.shape[-1]}."
            )
        return x_full.index_select(-1, self.representative.to(x_full.device))

    def reduce_rows(self, rows: Tensor) -> Tensor:
        """Map full row indices to reduced ones, keeping the first of each duplicate.

        Used for row SETS that must stay a set after fusion: the ideal-slack
        ``fixed_rows``, and a block-diagonal row partition. Returns an int64 tensor on
        ``rows``' device, in the input order with later duplicates dropped.
        """
        mapped = self.row_to_reduced.to(rows.device).index_select(0, rows)
        seen: set[int] = set()
        keep: list[int] = []
        for i, r in enumerate(mapped.tolist()):
            if r in seen:
                continue
            seen.add(r)
            keep.append(i)
        return mapped.index_select(
            0, torch.as_tensor(keep, dtype=torch.int64, device=rows.device)
        )

    def duplicate_rows(self, rows: Tensor) -> list[tuple[int, ...]]:
        """Full rows of ``rows`` that fusion maps onto the SAME reduced row.

        A non-empty answer means the caller's row set collapses (two slack terminals or
        two regulated terminals joined by an ideal switch), which the solve must refuse
        rather than silently pin one of them.
        """
        by_reduced: dict[int, list[int]] = {}
        mapped = self.row_to_reduced.to(rows.device).index_select(0, rows)
        for full, red in zip(rows.tolist(), mapped.tolist()):
            by_reduced.setdefault(int(red), []).append(int(full))
        return [tuple(v) for v in by_reduced.values() if len(v) > 1]

    # -- reporting -----------------------------------------------------------
    def node_groups(self) -> tuple[tuple[tuple[int, str], ...], ...]:
        """Per fused group, its ``(node_id, phase)`` members — the log / report form."""
        idx = self.full_index
        return tuple(
            tuple((idx.node_id_of(r), idx.phase_of(r).value) for r in grp)
            for grp in self.groups
        )

    def describe(self) -> str:
        """One-line summary: how many rows, branches and groups were collapsed."""
        n = self.full_index.size
        shown = [
            "{" + ", ".join(f"{nid}{ph}" for nid, ph in grp) + "}"
            for grp in self.node_groups()[:6]
        ]
        more = "" if len(self.groups) <= 6 else f", … (+{len(self.groups) - 6} more)"
        return (
            f"{len(self.fused_branch_ids)} zero-impedance branch(es) fused: "
            f"{n} node-phase rows collapse to {self.size} in "
            f"{len(self.groups)} group(s) {', '.join(shown)}{more}"
        )

    def to(self, device) -> "FusionMap":
        """A copy whose index tensors live on ``device`` (the constants are tiny)."""
        return FusionMap(
            size=self.size,
            index=self.index,
            full_index=self.full_index,
            row_to_reduced=self.row_to_reduced.to(device),
            representative=self.representative.to(device),
            fused_branch_ids=self.fused_branch_ids,
            groups=self.groups,
            indeterminate_branch_ids=self.indeterminate_branch_ids,
            _group_rows=self._group_rows.to(device),
            _group_coeff=self._group_coeff.to(device),
            _edge_slots=self._edge_slots.to(device),
            _edge_branches=self._edge_branches,
        )


class _UnionFind:
    """Path-compressing union-find over hashable keys (structural, torch-free)."""

    def __init__(self) -> None:
        self._parent: dict = {}

    def find(self, x):
        parent = self._parent
        root = parent.setdefault(x, x)
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _fused_terminal_pairs(branch) -> list[tuple[tuple[int, Phase], tuple[int, Phase]]]:
    """The ``(from_row_key, to_row_key)`` pairs an ideal branch short-circuits.

    A zero series impedance MATRIX means each from-terminal conductor is shorted to the
    to-terminal conductor at the SAME position (the primitive block's identity
    structure), so the pairing is positional — a phase-rotating jumper
    (``from_phases=(A,B,C)``, ``to_phases=(B,C,A)``) pairs A with B and so on.
    """
    if len(branch.from_phases) != len(branch.to_phases):
        raise ModelingError(
            f"branch {int(branch.id)} has {len(branch.from_phases)} from-phase(s) and "
            f"{len(branch.to_phases)} to-phase(s); a zero-impedance branch shorts its "
            "terminals conductor by conductor, which needs equal phase counts."
        )
    return [
        ((int(branch.from_node), pf), (int(branch.to_node), pt))
        for pf, pt in zip(branch.from_phases, branch.to_phases)
    ]


def _reduced_index(grid: Grid, full: NodePhaseIndex, row_to_reduced: Tensor, size: int):
    """Build the many-to-one reduced :class:`NodePhaseIndex`."""
    mapped = row_to_reduced.tolist()
    row_of: dict[tuple[int, Phase], int] = {}
    rows_of_node: dict[int, list[int]] = {}
    r = 0
    for node in grid.nodes:
        node_rows: list[int] = []
        for ph in node.phases:
            red = mapped[r]
            row_of[(int(node.id), ph)] = red
            node_rows.append(red)
            r += 1
        rows_of_node[int(node.id)] = node_rows
    rep = _representatives(row_to_reduced, size)
    node_ids = full.node_ids.index_select(0, rep)
    phase_codes = full.phase_codes.index_select(0, rep)
    return NodePhaseIndex(
        size=size,
        node_ids=node_ids,
        phase_codes=phase_codes,
        _row_of=row_of,
        _rows_of_node=rows_of_node,
        _phase_order=(Phase.A, Phase.B, Phase.C, Phase.N),
    )


def _representatives(row_to_reduced: Tensor, size: int) -> Tensor:
    """First full row of each reduced row, int64 ``[M]``."""
    rep = [-1] * size
    for full, red in enumerate(row_to_reduced.tolist()):
        if rep[red] < 0:
            rep[red] = full
    return torch.as_tensor(rep, dtype=torch.int64)


def _slack_row_keys(grid: Grid) -> set[tuple[int, Phase]]:
    """``(node, phase)`` keys carrying an in-service Source terminal.

    The current a source injects into its own row is not part of the nodal injection an
    ideal-slack solve knows (it is the slack's answer, not its input), so those rows'
    current balances are left OUT of the fused-branch current recovery. Under a Norton
    slack the row's injection IS known and its equation would be redundant, so dropping
    it is exact either way.
    """
    keys: set[tuple[int, Phase]] = set()
    for a in grid.appliances:
        if isinstance(a, Source) and getattr(a, "in_service", True):
            for ph in a.phases:
                keys.add((int(a.node), ph))
    return keys


def _recovery_structure(
    grid: Grid,
    full: NodePhaseIndex,
    row_to_reduced: Tensor,
    fused: list,
    slack_keys: set[tuple[int, Phase]],
) -> tuple[Tensor, Tensor, Tensor, tuple[tuple[int, int], ...], tuple[int, ...]]:
    """Constant maps that turn the KCL defect into each fused branch's current.

    Within one fused group the unknowns are the currents ``i`` of its fused edges and
    the equations are the per-row current balances ``A i = d`` — ``A`` the group's
    oriented incidence (``+1`` at the edge's from-row, ``-1`` at its to-row) and ``d``
    the defect the rest of the network leaves at that row (see
    :func:`fused_branch_currents`). The rows of a slack terminal are dropped (their
    injection is not an input), leaving ``A_known``; the map ``pinv(A_known)`` is a
    structural CONSTANT, so the recovery is one small matmul on the differentiable
    defect.

    A tree of fused edges gives a square (or consistent over-determined) system and a
    unique current. A LOOP of fused edges, or a group holding two slack terminals,
    leaves a circulating current undetermined — the physics of ideal conductors, not a
    numerical defect; the pseudo-inverse then returns the minimum-norm split and the
    branch is listed in ``indeterminate_branch_ids``.

    Returns ``(group_rows [G, kmax], group_coeff [G, mmax, kmax],
    edge_slots [E], edge_branches, indeterminate_ids)``.
    """
    # Group the fused EDGES (one per branch phase) by their reduced row.
    edges_by_group: dict[int, list[tuple[int, int, int, int]]] = {}
    mapped = row_to_reduced.tolist()
    for b in fused:
        for pos, (key_from, key_to) in enumerate(_fused_terminal_pairs(b)):
            rf = full.row(*key_from)
            rt = full.row(*key_to)
            edges_by_group.setdefault(mapped[rf], []).append((int(b.id), pos, rf, rt))

    slack_rows = {full.row(*k) for k in slack_keys if full.has(*k)}
    rows_by_group: dict[int, list[int]] = {}
    for full_row, red in enumerate(mapped):
        rows_by_group.setdefault(red, []).append(full_row)

    group_ids = sorted(edges_by_group)
    if not group_ids:
        return (
            torch.zeros(0, 0, dtype=torch.int64),
            torch.zeros(0, 0, 0, dtype=torch.float64),
            torch.zeros(0, dtype=torch.int64),
            (),
            (),
        )
    kmax = max(len(rows_by_group[g]) for g in group_ids)
    mmax = max(len(edges_by_group[g]) for g in group_ids)
    g_rows = torch.zeros((len(group_ids), kmax), dtype=torch.int64)
    g_coeff = torch.zeros((len(group_ids), mmax, kmax), dtype=torch.float64)
    edge_slots: list[int] = []
    edge_branches: list[tuple[int, int]] = []
    indeterminate: list[int] = []

    for gi, g in enumerate(group_ids):
        rows = rows_by_group[g]
        local = {r: i for i, r in enumerate(rows)}
        edges = edges_by_group[g]
        g_rows[gi, : len(rows)] = torch.as_tensor(rows, dtype=torch.int64)
        inc = torch.zeros((len(rows), len(edges)), dtype=torch.float64)
        for e, (_bid, _pos, rf, rt) in enumerate(edges):
            inc[local[rf], e] += 1.0
            inc[local[rt], e] -= 1.0
        known = [i for i, r in enumerate(rows) if r not in slack_rows]
        a_known = inc.index_select(0, torch.as_tensor(known, dtype=torch.int64))
        pinv = torch.linalg.pinv(a_known)  # [m, len(known)]
        rank = int(torch.linalg.matrix_rank(a_known))
        if rank < len(edges):
            indeterminate.extend(bid for bid, _p, _f, _t in edges)
        for col, row_local in enumerate(known):
            g_coeff[gi, : len(edges), row_local] = pinv[:, col]
        for e, (bid, pos, _rf, _rt) in enumerate(edges):
            edge_slots.append(gi * mmax + e)
            edge_branches.append((bid, pos))

    return (
        g_rows,
        g_coeff,
        torch.as_tensor(edge_slots, dtype=torch.int64),
        tuple(edge_branches),
        tuple(dict.fromkeys(indeterminate)),
    )


def fusion_map(
    grid: Grid,
    *,
    param_overrides: Optional[dict] = None,
    branch_states: Optional[dict] = None,
    zero: Optional[Sequence[ZeroImpedanceBranch]] = None,
) -> Optional[FusionMap]:
    """Build the exact bus fusion of ``grid``'s zero-impedance branches.

    Collapses the terminal node-phase rows of every FUSABLE zero-impedance branch
    (:func:`zero_impedance_branches`) into one row per connected group, and returns the
    :class:`FusionMap` that relates the reduced layout to the grid's full one.
    ``None`` when nothing fuses — the usual case, and the signal to every caller to take
    the unreduced path unchanged.

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.schemas.grid_schema.Grid`.
    param_overrides:
        The solve's parameter substitutions. Fusion reads the EFFECTIVE impedance, so a
        branch whose zero impedance is overridden by a finite leaf stays stamped (and
        keeps its gradient), while a substitution of zero fuses.
    branch_states:
        The solve's switch-state mask. A branch listed there is NEVER fused: its state
        scales a stamped primitive block, which a fused branch does not have. A listed
        branch with zero impedance therefore has no representation at all and raises
        :class:`~pgml.errors.ModelingError` — the two mechanisms are mutually exclusive
        by construction, and the fix (a finite impedance for a swept switch) is named.
    zero:
        The result of :func:`zero_impedance_branches` for this grid and these
        ``param_overrides``, when the caller already holds it. A solve walks the branch
        list once and hands the list to every consumer of it (the map, the modeling
        gate) instead of rediscovering the same empty answer per call.

    Raises
    ------
    ~pgml.errors.ModelingError
        When a zero-impedance branch can be neither stamped nor fused: a transformer, a
        zero-series branch that still carries a shunt admittance, or a branch under a
        ``branch_states`` sweep. Also when two fused nodes have different rated
        voltages (an ideal conductor between two voltage levels is a data error).
    """
    if zero is None:
        zero = zero_impedance_branches(grid, param_overrides=param_overrides)
    if not zero:
        return None
    swept = {int(b) for b in (branch_states or {})}
    by_id = {int(b.id): b for b in grid.branches}

    conflicting = [z for z in zero if z.branch_id in swept]
    if conflicting:
        shown = ", ".join(f"{z.component} {z.branch_id}" for z in conflicting[:8])
        r_ideal = float(defaults.get("branch.near_ideal_series_resistance_ohm"))
        raise ModelingError(
            f"{len(conflicting)} branch(es) with zero series impedance are listed in "
            f"branch_states: {shown}. A swept branch is reached by SCALING its stamped "
            "primitive admittance, and a zero-impedance branch has none, while a FUSED "
            "branch has no state to scale. Give every swept switch a finite series "
            f"impedance (the documented near-ideal value is {r_ideal:g} Ohm), or drop it "
            "from branch_states to have its terminals collapsed exactly."
        )
    unfusable = [z for z in zero if not z.fusable]
    if unfusable:
        raise ModelingError(describe_unfusable(unfusable))

    fused = [by_id[z.branch_id] for z in zero]
    full = node_phase_index(grid)

    uf = _UnionFind()
    for node in grid.nodes:
        for ph in node.phases:
            uf.find((int(node.id), ph))
    node_by_id = {int(nd.id): nd for nd in grid.nodes}
    for b in fused:
        for a, c in _fused_terminal_pairs(b):
            for key in (a, c):
                if not full.has(*key):
                    raise ModelingError(
                        f"branch {int(b.id)} connects node {key[0]} phase "
                        f"{key[1].value}, which that node does not carry; a fused "
                        "branch must join existing node-phase rows."
                    )
            u_a = float(_scalar(node_by_id[a[0]].u_rated_v))
            u_c = float(_scalar(node_by_id[c[0]].u_rated_v))
            if u_a != u_c:
                raise ModelingError(
                    f"branch {int(b.id)} has zero series impedance but joins node "
                    f"{a[0]} ({u_a:g} V) and node {c[0]} ({u_c:g} V). An ideal "
                    "conductor cannot connect two voltage levels — give the branch its "
                    "real impedance, or correct the rated voltages."
                )
            uf.union(a, c)

    # Reduced row numbering: full rows in order, each new group root taking the next
    # reduced index, so the reduced layout stays monotone in the full one (contiguous
    # blocks of a merged ensemble stay contiguous).
    reduced_of_root: dict[object, int] = {}
    mapping: list[int] = []
    r = 0
    for node in grid.nodes:
        for ph in node.phases:
            root = uf.find((int(node.id), ph))
            if root not in reduced_of_root:
                reduced_of_root[root] = len(reduced_of_root)
            mapping.append(reduced_of_root[root])
            r += 1
    size = len(reduced_of_root)
    row_to_reduced = torch.as_tensor(mapping, dtype=torch.int64)
    rep = _representatives(row_to_reduced, size)

    groups: dict[int, list[int]] = {}
    for full_row, red in enumerate(mapping):
        groups.setdefault(red, []).append(full_row)
    multi = tuple(tuple(v) for v in groups.values() if len(v) > 1)

    g_rows, g_coeff, edge_slots, edge_branches, indeterminate = _recovery_structure(
        grid, full, row_to_reduced, fused, _slack_row_keys(grid)
    )
    return FusionMap(
        size=size,
        index=_reduced_index(grid, full, row_to_reduced, size),
        full_index=full,
        row_to_reduced=row_to_reduced,
        representative=rep,
        fused_branch_ids=tuple(int(b.id) for b in fused),
        groups=multi,
        indeterminate_branch_ids=indeterminate,
        _group_rows=g_rows,
        _group_coeff=g_coeff,
        _edge_slots=edge_slots,
        _edge_branches=edge_branches,
    )


def _scalar(v) -> float:
    """A python float from a float or a (possibly tracked) tensor parameter."""
    if hasattr(v, "detach"):
        with torch.no_grad():
            return float(torch.as_tensor(v).reshape(-1)[0])
    return float(v)


def resolve_fusion(
    grid: Grid,
    fusion: Union[FusionMap, bool, None, _NoFusion],
    *,
    param_overrides: Optional[dict] = None,
    branch_states: Optional[dict] = None,
    zero: Optional[Sequence[ZeroImpedanceBranch]] = None,
) -> Optional[FusionMap]:
    """Resolve a ``fusion`` argument to a :class:`FusionMap` or ``None``.

    - ``None`` (default everywhere): resolve the documented policy
      ``branch.zero_impedance`` — ``"fuse"`` builds the map, ``"error"`` refuses a
      zero-impedance branch by name (the behaviour of a grid that carries none is
      identical either way, and the overwhelming majority carry none).
    - ``False``: never fuse; a zero-impedance branch raises.
    - a :class:`FusionMap`: use it as given (the prepared / shared map of a solve).
    - :data:`NO_FUSION`: a resolution that produced no map, handed on as such.

    ``zero`` passes an already-collected :func:`zero_impedance_branches` list through to
    :func:`fusion_map`.
    """
    if isinstance(fusion, FusionMap):
        return fusion
    if fusion is NO_FUSION:
        return None
    policy = "error" if fusion is False else str(defaults.get("branch.zero_impedance"))
    if policy not in ("fuse", "error"):
        from pgml.errors import ConfigurationError

        raise ConfigurationError(
            f"branch.zero_impedance must be 'fuse' or 'error', got {policy!r}."
        )
    if policy == "error":
        bad = (
            zero_impedance_branches(grid, param_overrides=param_overrides)
            if zero is None
            else zero
        )
        if bad:
            raise ModelingError(describe_unfusable(bad))
        return None
    return fusion_map(
        grid, param_overrides=param_overrides, branch_states=branch_states, zero=zero
    )


def log_fusion_summary(fusion: Optional[FusionMap]) -> None:
    """INFO-log which node-phase rows were fused (once per solve entry point)."""
    if fusion is None or not fusion.groups:
        return
    _log.info("%s; results are reported on the original node ids.", fusion.describe())
    if fusion.indeterminate_branch_ids:
        _log.warning(
            "fused branch(es) %s form a loop of ideal conductors (or share a group "
            "with two slack terminals), so the current through each of them is not "
            "determined by the node voltages; the reported current is the "
            "minimum-norm split that satisfies Kirchhoff's law at every fused node.",
            list(fusion.indeterminate_branch_ids[:8]),
        )


# ---------------------------------------------------------------------------
# branch-current recovery
# ---------------------------------------------------------------------------
def fused_branch_currents(
    fusion: FusionMap, defect: Tensor
) -> dict[int, dict[int, Tensor]]:
    """Per fused branch and phase position, the current into its FROM terminal.

    ``defect`` is the full-layout ``[..., N]`` current Kirchhoff's law leaves for the
    fused branches to carry::

        defect = i_inj - Y_network_without_fused_branches @ V

    with ``i_inj`` the nodal injection the solve used. At every fused row the fused
    branches' terminal currents must sum to that defect, which the constant
    (structural) maps of :func:`_recovery_structure` invert in one matmul per group.
    Linear in ``defect``, so gradients flow from the recovered current back to the
    voltages, the injections and the network parameters; batched over every leading
    dim; device and dtype follow ``defect``.

    Returns ``{branch_id: {phase_position: current [...]}}`` (complex, the shape of
    ``defect`` without its row axis).
    """
    if not fusion.fused_branch_ids:
        return {}
    coeff = fusion._group_coeff.to(device=defect.device, dtype=defect.dtype)
    rows = fusion._group_rows.to(defect.device)
    g, kmax = rows.shape
    d_grp = defect.index_select(-1, rows.reshape(-1)).reshape(
        *defect.shape[:-1], g, kmax
    )
    i_edges = torch.einsum("gmk,...gk->...gm", coeff, d_grp)  # [..., G, mmax]
    flat = i_edges.reshape(*i_edges.shape[:-2], -1)
    picked = flat.index_select(-1, fusion._edge_slots.to(defect.device))
    out: dict[int, dict[int, Tensor]] = {}
    for e, (bid, pos) in enumerate(fusion._edge_branches):
        out.setdefault(bid, {})[pos] = picked[..., e]
    return out


__all__ = [
    "FUSABLE_COMPONENTS",
    "NO_FUSION",
    "FusionMap",
    "ZeroImpedanceBranch",
    "describe_unfusable",
    "fused_branch_currents",
    "fusion_map",
    "log_fusion_summary",
    "resolve_fusion",
    "zero_impedance_branches",
]
