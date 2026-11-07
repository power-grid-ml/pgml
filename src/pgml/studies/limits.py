"""Operating limits of a planning study and the branch ratings they are checked against.

:class:`HostingLimits` is the set of limits a hosting-capacity search keeps (voltage band,
thermal loading of lines and transformers, optional unbalance and harmonic distortion).
:func:`branch_ratings` resolves the thermal rating of every branch from the grid itself:
a line's ampacity comes from its catalog type, a transformer's apparent-power rating from
the component or its catalog type. Both are plain data, independent of any solve.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple, Optional

import torch
from torch import Tensor

from pgml.assembly import base_voltage_per_row, node_phase_index
from pgml.errors import InputError
from pgml.geometry.sequence import fortescue_matrix
from pgml.schemas.grid_schema import Grid, Line, Phase, Transformer

from ._common import live_phases

if TYPE_CHECKING:
    from pgml.simulation import SolvedState


class BranchRating(NamedTuple):
    """Thermal rating of one branch; either field is ``None`` when the grid declares none.

    Attributes
    ----------
    i_max_a:
        Ampacity in A (per conductor), set for a :class:`~pgml.schemas.grid_schema.Line`
        whose catalog type declares ``max_i_a``.
    s_rated_va:
        Rated apparent power in VA (three-phase total), set for a
        :class:`~pgml.schemas.grid_schema.Transformer`.
    """

    i_max_a: Optional[float]
    s_rated_va: Optional[float]


def _as_float(value: Any) -> float:
    """A float from a schema field that may hold a float or a tensor (float/tensor duality)."""
    if hasattr(value, "detach"):
        return float(value.detach().reshape(-1)[0])
    return float(value)


def branch_ratings(grid: Grid) -> dict[int, BranchRating]:
    """Thermal rating of every branch of ``grid``, keyed by branch id (pure, no solve).

    The resolution rule:

    - a :class:`~pgml.schemas.grid_schema.Line` takes ``max_i_a`` of the catalog type its
      ``type_ref`` names (``grid.types.lines``); a line with explicit parameters and no
      ``type_ref`` reports no ampacity;
    - a :class:`~pgml.schemas.grid_schema.Transformer` takes its own ``s_rated_va`` or,
      when that is unset, the one of its catalog type (``grid.types.transformers``);
    - every other branch kind (switch, generic branch, shunt reactor) reports neither.

    Every branch is listed, in service or not, so a caller can tell "no rating" from
    "unknown branch".

    Parameters
    ----------
    grid:
        The grid whose branches to rate. Not modified.

    Returns
    -------
    dict[int, BranchRating]
        ``{branch_id: BranchRating(i_max_a, s_rated_va)}`` in ``grid.branches`` order;
        values are plain floats in A and VA, or ``None``.
    """
    ratings: dict[int, BranchRating] = {}
    for branch in grid.branches:
        i_max_a: Optional[float] = None
        s_rated_va: Optional[float] = None
        if isinstance(branch, Line):
            if branch.type_ref is not None:
                line_type = grid.types.lines.get(branch.type_ref)
                if line_type is not None and line_type.max_i_a is not None:
                    i_max_a = float(line_type.max_i_a)
        elif isinstance(branch, Transformer):
            rated = branch.s_rated_va
            if rated is None and branch.type_ref is not None:
                transformer_type = grid.types.transformers.get(branch.type_ref)
                if transformer_type is not None:
                    rated = transformer_type.s_rated_va
            if rated is not None:
                s_rated_va = _as_float(rated)
        ratings[int(branch.id)] = BranchRating(i_max_a=i_max_a, s_rated_va=s_rated_va)
    return ratings


@dataclass(frozen=True)
class HostingLimits:
    """The limits a hosting-capacity search keeps; ``None`` disables a limit.

    Attributes
    ----------
    v_min_pu, v_max_pu:
        Voltage band in per unit of each node's nominal line-to-neutral voltage, applied
        to the fundamental phase-to-neutral voltage magnitude of every phase conductor
        (phase-to-ground on a node without a neutral conductor). The defaults 0.90 /
        1.10 are the EN 50160 supply band. A stricter rise limit is a different upper
        bound: VDE-AR-N 4105, for example, allows new generation in a low-voltage grid
        to raise the voltage by at most 3 %, which ``v_max_pu=1.03`` expresses for a
        grid operated at 1.0 pu.
    line_loading:
        Largest allowed conductor current of a line as a fraction of its ampacity
        (:func:`branch_ratings`), RMS over the solved orders, checked at both terminals
        on every PHASE conductor. A neutral conductor is not held to the phase ampacity:
        a low-voltage cable's neutral often has a reduced section, so its limit is
        :attr:`neutral_loading`.
    transformer_loading:
        Largest allowed transformer loading as a fraction of its rated power: the
        per-phase current times the terminal node's nominal line-to-neutral voltage,
        against ``s_rated_va`` divided by the number of phase conductors, at both
        terminals (the current-based loading pandapower reports).
    vuf_max:
        Largest voltage unbalance factor ``|V2| / |V1|`` (fraction, e.g. 0.02) at nodes
        that carry phases a, b and c.
    thd_max:
        Largest voltage total harmonic distortion (fraction, e.g. 0.08) at every phase
        conductor, over the solved orders. Needs a harmonic calculation.
    neutral_loading:
        Largest allowed neutral-conductor current of a line as a fraction of the line's
        (phase) ampacity, RMS over the solved orders, at both terminals; ``None``
        (default) does not check the neutral. A reduced neutral of half the phase
        section, for example, is ``neutral_loading=0.5`` with ``line_loading=1.0``;
        ``1.0`` holds the neutral to the phase ampacity. Unbalance and triplen harmonics
        load the neutral of a four-wire line even when every phase is within its rating.
    """

    v_min_pu: Optional[float] = 0.90
    v_max_pu: Optional[float] = 1.10
    line_loading: Optional[float] = 1.0
    transformer_loading: Optional[float] = 1.0
    vuf_max: Optional[float] = None
    thd_max: Optional[float] = None
    neutral_loading: Optional[float] = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is None:
                continue
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise InputError(
                    f"HostingLimits.{name} must be a positive finite number or None, "
                    f"got {value!r}."
                )
        if (
            self.v_min_pu is not None
            and self.v_max_pu is not None
            and self.v_min_pu >= self.v_max_pu
        ):
            raise InputError(
                f"HostingLimits: v_min_pu ({self.v_min_pu}) must be below v_max_pu "
                f"({self.v_max_pu})."
            )

    def to_dict(self) -> dict[str, Optional[float]]:
        """The limits as a JSON-ready dict (field name -> float or ``None``)."""
        return {k: (None if v is None else float(v)) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HostingLimits":
        """Rebuild from :meth:`to_dict` output; absent keys take the defaults."""
        return cls(**dict(data))


class _LimitElements:
    """The elements every limit of :class:`HostingLimits` is checked on, and their values.

    Built once per grid (pure tables, no solve); :meth:`evaluate` turns a solved state
    into the limit quantities of every element. The hosting-capacity search and the
    enhancement sensitivities both judge a grid through this one class, so they read
    the same voltages and loadings.

    Elements, each in grid order:

    - ``live_desc``: ``(node_id, phase)`` of every phase conductor (rows of ``v_pu``,
      ``thd``);
    - ``abc_nodes``: nodes carrying phases a, b and c (rows of ``vuf``);
    - ``lines``: ``(branch_id, i_max_a, live_from, live_to)`` of every in-service line
      with an ampacity (:func:`branch_ratings`); ``neutral_lines`` the subset with a
      neutral conductor at either terminal;
    - ``transformers``: ``(branch_id, terminals)`` of every in-service rated
      transformer, ``terminals`` the ``(phase mask, u_ln * n_live / s_rated)`` pair of
      the from and the to terminal.
    """

    def __init__(self, grid: Grid) -> None:
        index = node_phase_index(grid)
        self.n_rows = index.size
        base = base_voltage_per_row(grid, dtype=torch.float64)

        live, neutral = [], []
        for node in grid.nodes:
            n_row = index.row(node.id, Phase.N) if Phase.N in node.phases else -1
            for ph in live_phases(node.phases):
                live.append(index.row(node.id, ph))
                neutral.append(n_row)
        self.live_rows = torch.tensor(live, dtype=torch.long)
        self.neutral_rows = torch.tensor(neutral, dtype=torch.long)
        self.live_base = base[self.live_rows]
        self.live_desc = [(index.node_id_of(r), index.phase_of(r).value) for r in live]

        abc_phases = (Phase.A, Phase.B, Phase.C)
        abc_nodes = [n for n in grid.nodes if all(p in n.phases for p in abc_phases)]
        self.abc_rows = torch.tensor(
            [[index.row(n.id, p) for p in abc_phases] for n in abc_nodes],
            dtype=torch.long,
        ).reshape(-1, 3)
        self.abc_nodes = [n.id for n in abc_nodes]

        ratings = branch_ratings(grid)
        self.lines = []
        for b in grid.branches:
            if not (isinstance(b, Line) and b.in_service and ratings[b.id].i_max_a):
                continue
            live_from = torch.tensor([p != Phase.N for p in b.from_phases])
            live_to = torch.tensor([p != Phase.N for p in b.to_phases])
            self.lines.append((b.id, ratings[b.id].i_max_a, live_from, live_to))
        self.neutral_lines = [
            entry for entry in self.lines if not (entry[2].all() and entry[3].all())
        ]
        self.transformers = []
        for b in grid.branches:
            if not (isinstance(b, Transformer) and b.in_service):
                continue
            s_rated = ratings[b.id].s_rated_va
            if not s_rated:
                continue
            terminals = []
            for node_id, phases in (
                (b.from_node, b.from_phases),
                (b.to_node, b.to_phases),
            ):
                mask = torch.tensor([p != Phase.N for p in phases])
                n_live = int(mask.sum())
                u_ln = float(base[index.row(node_id, live_phases(phases)[0])])
                # loading = |I| * U_ln * n_live / S_rated per phase conductor
                terminals.append((mask, u_ln * n_live / s_rated))
            self.transformers.append((b.id, terminals))

    def evaluate(
        self,
        v: Tensor,
        state: "SolvedState",
        limits: HostingLimits,
        *,
        h1: int,
    ) -> dict[str, Tensor]:
        """Limit quantities of every element of the limits ``limits`` enables.

        Parameters
        ----------
        v:
            Solved node-phase voltages ``[B, H, N]`` (complex, any device), the
            fundamental at order position ``h1``.
        state:
            The solved state of ``v``; its ``branch_currents()`` supplies the line and
            transformer currents (read only when a loading limit is enabled).
        limits:
            Which quantities to evaluate: a quantity whose limit is ``None`` is left out.

        Returns
        -------
        dict[str, Tensor]
            float64 tensors on ``v``'s device, on the autograd tape of ``v``:
            ``"v_pu"`` ``[B, R]`` fundamental phase-to-neutral magnitude per unit of
            the nominal line-to-neutral voltage (``live_desc`` order; present when
            either voltage bound is set); ``"vuf"`` ``[B, K]`` ``|V2| / |V1|``
            (``abc_nodes``); ``"thd"`` ``[B, R]`` voltage THD over the solved orders;
            ``"line_loading"`` ``[B, L]`` largest phase-conductor RMS current over
            both terminals per ampacity (``lines``); ``"neutral_loading"`` ``[B, Ln]``
            the same for the neutral conductor (``neutral_lines``);
            ``"transformer_loading"`` ``[B, T]`` largest per-phase current times the
            terminal's nominal line-to-neutral voltage per rated power per phase
            (``transformers``), the larger of ``"transformer_terminal_loading"``
            ``[B, T, 2]`` (the from and the to terminal).
        """
        b = v.shape[0]
        dev = v.device
        out: dict[str, Tensor] = {}
        # Phase-to-neutral voltage of every phase conductor (phase-to-ground without N).
        v_live = v[..., self.live_rows.to(dev)]
        has_n = (self.neutral_rows >= 0).to(dev)
        v_n = v[..., self.neutral_rows.clamp_min(0).to(dev)]
        v_pn = v_live - torch.where(has_n, v_n, torch.zeros_like(v_n))  # [B, H, R]
        mag = v_pn.abs().to(torch.float64)
        if limits.v_max_pu is not None or limits.v_min_pu is not None:
            out["v_pu"] = mag[:, h1, :] / self.live_base.to(dev)
        if limits.vuf_max is not None and self.abc_nodes:
            v1 = v[:, h1, :][:, self.abc_rows.to(dev)]  # [B, K, 3]
            a_inv = torch.linalg.inv(fortescue_matrix(v1.dtype, v1.device))
            seq = torch.einsum("ij,bkj->bki", a_inv, v1)  # [B, K, (0, +, -)]
            out["vuf"] = (seq[..., 2].abs() / seq[..., 1].abs()).to(torch.float64)
        if limits.thd_max is not None:
            sq = mag.pow(2)
            out["thd"] = (sq.sum(dim=1) - sq[:, h1, :]).clamp_min(0.0).sqrt() / mag[
                :, h1, :
            ]

        need_lines = limits.line_loading is not None and self.lines
        need_neutral = limits.neutral_loading is not None and self.neutral_lines
        need_trafos = limits.transformer_loading is not None and self.transformers
        if not (need_lines or need_neutral or need_trafos):
            return out
        currents = {bc.branch_id: bc for bc in state.branch_currents()}

        def _rms(i: Tensor) -> Tensor:  # [B, H, P] -> [B, P]
            return i.abs().pow(2).sum(dim=-2).sqrt()

        def _line_loading(entries, live: bool) -> Tensor:
            """Largest conductor loading [B, L]: phase (``live``) or neutral ones."""
            cols = []
            for bid, i_max, m_from, m_to in entries:
                bc = currents[bid]
                sel_from = (m_from if live else ~m_from).to(dev)
                sel_to = (m_to if live else ~m_to).to(dev)
                i_cond = torch.cat(
                    [_rms(bc.i_from)[..., sel_from], _rms(bc.i_to)[..., sel_to]],
                    dim=-1,
                )
                cols.append(i_cond.amax(dim=-1) / i_max)
            return torch.stack(cols, dim=-1).reshape(b, -1).to(torch.float64)

        if need_lines:
            out["line_loading"] = _line_loading(self.lines, live=True)
        if need_neutral:
            out["neutral_loading"] = _line_loading(self.neutral_lines, live=False)
        if need_trafos:
            cols = []
            for bid, ((m_from, k_from), (m_to, k_to)) in self.transformers:
                bc = currents[bid]
                i_from = _rms(bc.i_from)[..., m_from.to(dev)] * k_from
                i_to = _rms(bc.i_to)[..., m_to.to(dev)] * k_to
                cols.append(
                    torch.stack([i_from.amax(dim=-1), i_to.amax(dim=-1)], dim=-1)
                )
            terminal = torch.stack(cols, dim=-2).reshape(b, -1, 2).to(torch.float64)
            out["transformer_terminal_loading"] = terminal
            out["transformer_loading"] = terminal.amax(dim=-1)
        return out


__all__ = ["BranchRating", "HostingLimits", "branch_ratings"]
