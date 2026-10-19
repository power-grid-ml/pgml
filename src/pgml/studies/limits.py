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
from typing import Any, Mapping, NamedTuple, Optional

from pgml.errors import InputError
from pgml.schemas.grid_schema import Grid, Line, Transformer


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
        and on every conductor including a neutral.
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
    """

    v_min_pu: Optional[float] = 0.90
    v_max_pu: Optional[float] = 1.10
    line_loading: Optional[float] = 1.0
    transformer_loading: Optional[float] = 1.0
    vuf_max: Optional[float] = None
    thd_max: Optional[float] = None

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


__all__ = ["BranchRating", "HostingLimits", "branch_ratings"]
