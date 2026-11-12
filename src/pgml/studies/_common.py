"""Argument handling, operating points and serialization checks shared by the studies."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import torch
from torch import Tensor

from pgml.errors import InputError
from pgml.schemas.grid_schema import Generator, Grid, Load, Phase, Source

#: A study's base operating point: ``None`` / ``"worst_case"``, ``"nameplate"``, or a
#: mapping ``{appliance_id: {"p_w", "q_var", "p_per_phase_w", "q_per_phase_var"}}``.
BaseOperatingPoint = Union[None, str, Mapping[int, Mapping[str, Any]]]

_COMPLEX_DTYPES = {"complex128": torch.complex128, "complex64": torch.complex64}


def resolve_complex_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """The complex working dtype named by ``dtype``.

    Accepts ``torch.complex64`` / ``torch.complex128`` or their names (``"complex64"``,
    ``"complex128"``, also with a ``"torch."`` prefix); anything else raises
    :class:`~pgml.errors.InputError`.
    """
    if isinstance(dtype, torch.dtype):
        if dtype in _COMPLEX_DTYPES.values():
            return dtype
    elif isinstance(dtype, str):
        resolved = _COMPLEX_DTYPES.get(dtype.removeprefix("torch."))
        if resolved is not None:
            return resolved
    raise InputError(
        f"dtype must be torch.complex64 or torch.complex128 (or their names "
        f"{sorted(_COMPLEX_DTYPES)}), got {dtype!r}."
    )


def check_study(data: Mapping[str, Any], expected: str) -> None:
    """Refuse a result dict whose ``"study"`` key does not name ``expected``."""
    found = data.get("study") if isinstance(data, Mapping) else None
    if found != expected:
        raise InputError(
            f'expected a {expected!r} result dict ("study": {expected!r}), '
            f'found "study": {found!r}.'
        )


def live_phases(phases: Sequence[Phase]) -> tuple[Phase, ...]:
    """The phase conductors of a terminal (its neutral excluded)."""
    return tuple(p for p in phases if p != Phase.N)


def resolve_candidates(grid: Grid, candidates: Optional[Sequence[int]]) -> list[int]:
    """Candidate bus ids: as given (validated), else every bus without a source."""
    node_ids = [n.id for n in grid.nodes]
    if candidates is None:
        source_nodes = {
            a.node for a in grid.appliances if isinstance(a, Source) and a.in_service
        }
        return [
            n.id
            for n in grid.nodes
            if n.id not in source_nodes and live_phases(n.phases)
        ]
    chosen = [int(c) for c in candidates]
    unknown = sorted(set(chosen) - set(node_ids))
    if unknown:
        raise InputError(f"candidates name unknown nodes {unknown}")
    if len(set(chosen)) != len(chosen):
        raise InputError("candidates must not repeat a node")
    if not chosen:
        raise InputError("candidates must name at least one node")
    return chosen


def base_operating_values(
    grid: Grid,
    direction: str,
    base_operating_point: BaseOperatingPoint,
    real: torch.dtype,
    device: Optional[torch.device],
) -> tuple[str, dict]:
    """The base operating point as ``batch_from_values`` mappings, plus its label.

    ``None`` / ``"worst_case"``: for generation every in-service load draws nothing and
    the existing generation runs at nameplate; for load every in-service generator
    injects nothing and the loads draw nameplate. Storage keeps its nameplate setpoint in
    both. ``"nameplate"``: every device at nameplate. A mapping: ``{appliance_id:
    {"p_w", "q_var", "p_per_phase_w", "q_per_phase_var"}}`` overrides on nameplate.

    Returns ``(label, values)``: ``label`` is ``"worst_case"``, ``"nameplate"`` or
    ``"custom"``; ``values`` maps each of the four keys to ``{appliance_id: tensor}``
    (per-phase keys: a list of tensors), real dtype ``real`` on ``device``.
    """
    values: dict = {"p_w": {}, "q_var": {}, "p_per_phase_w": {}, "q_per_phase_var": {}}

    def _t(x) -> Tensor:
        return torch.as_tensor(x, dtype=real, device=device)

    if base_operating_point is None or base_operating_point == "worst_case":
        zeroed = Load if direction == "generation" else Generator
        for a in grid.appliances:
            if a.in_service and isinstance(a, zeroed):
                values["p_w"][a.id] = _t(0.0)
                values["q_var"][a.id] = _t(0.0)
        return "worst_case", values
    if base_operating_point == "nameplate":
        return "nameplate", values
    if isinstance(base_operating_point, str):
        raise InputError(
            "base_operating_point must be None, 'worst_case', 'nameplate' or a mapping "
            f"of appliance ids, got {base_operating_point!r}"
        )
    for cid, entry in base_operating_point.items():
        unknown = set(entry) - set(values)
        if unknown:
            raise InputError(
                f"base_operating_point[{cid}] has unknown keys {sorted(unknown)}; use "
                f"{sorted(values)}"
            )
        for key, value in entry.items():
            if key in ("p_per_phase_w", "q_per_phase_var"):
                values[key][int(cid)] = [_t(x) for x in value]
            else:
                values[key][int(cid)] = _t(value)
    return "custom", values
