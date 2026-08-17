"""Per-target structured perturbation sweep (inject one error per node).

Use case 1 from the batching roadmap: "inject a specific error ONCE at each node and
measure how it spreads." This is an ENUMERATION over which target is perturbed — a
batch of ``B = #targets`` scenarios where scenario ``j`` perturbs exactly target ``j``
(all other targets nominal) — not a cartesian product of levels.

``perturbation_sweep(grid, selector, perturbation)`` builds the diagonal batch as a
:class:`~pgml.scenarios.sampler.SampledScenarios` (an ``operating_point`` override per
selected target) and records the ground truth as
:class:`~pgml.schemas.scenario_schema.ParameterPerturbation` rows. Feed it to
:func:`~pgml.scenarios.run.run_scenarios` like any other sampled batch.

Scope: this perturbs an operating-point quantity (P / Q injection) at a load/generator.
Perturbing a network PARAMETER (line/transformer impedance) — the inverse-problem use
case — needs a branch-aware selector and matrix-valued ground truth (the schema's
scalar ``nominal_value``/``perturbed_value`` do not fit a per-phase matrix), so it is
deferred to the parameter-recovery phase.
"""

from __future__ import annotations

import torch

from pgml.errors import InputError
from pgml.schemas.grid_schema import Grid
from pgml.schemas.scenario_schema import ParameterPerturbation

from .config import Perturbation, Selector
from .sampler import SampledScenarios, _nominal

_F64 = torch.float64

# (field key, operating_point key, parameter_path, unit, _Nominal attribute)
_FIELDS = {
    "p": ("p_w", "p_nom_w", "W", "p_total"),
    "q": ("q_var", "q_nom_var", "var", "q_total"),
}


def _perturb(base: float, mode: str, value: float) -> float:
    """Apply the perturbation to a nominal scalar."""
    if mode == "scale":
        return base * value
    if mode == "delta":
        return base + value
    return value  # "set"


def perturbation_sweep(
    grid: Grid, selector: Selector, perturbation: Perturbation
) -> SampledScenarios:
    """Sweep one operating-point error across each selected target.

    Parameters
    ----------
    grid:
        The reference grid (targets keep their nominal operating point except where
        perturbed).
    selector:
        Selects the target loads/generators; the batch has ``B = #targets`` scenarios.
    perturbation:
        The error to inject (field / mode / value), applied to exactly one target per
        scenario.

    Returns
    -------
    SampledScenarios
        ``operating_point`` holds a ``[B]`` column per selected target (nominal except
        at its own scenario index); ``perturbations`` holds the per-scenario
        :class:`ParameterPerturbation` ground truth; ``samples`` records
        ``"<name>_target_id"`` ``[B]`` (the perturbed component id per scenario) and
        ``"<name>_perturbed_<field>"`` ``[B]`` (the applied value).
    """
    ids = selector.resolve(grid)
    if not ids:
        raise InputError(
            "perturbation_sweep selector matched no in-service components."
        )
    b = len(ids)
    nominal = _nominal(grid)
    kind = selector.component
    fkeys = ["p", "q"] if perturbation.field == "pq" else [perturbation.field]

    operating_point: dict = {}
    perturbations: list = []
    diag: dict[str, list] = {fk: [] for fk in fkeys}

    for j, tid in enumerate(ids):
        rec = nominal[tid]
        for fk in fkeys:
            op_key, ppath, unit, attr = _FIELDS[fk]
            base = float(getattr(rec, attr))
            applied = _perturb(base, perturbation.mode, perturbation.value)
            col = torch.full((b,), base, dtype=_F64)
            col[j] = applied
            operating_point.setdefault(tid, {})[op_key] = col
            diag[fk].append(applied)
            perturbations.append(
                ParameterPerturbation(
                    scenario_id=j,
                    component_kind=kind,
                    component_id=tid,
                    parameter_path=ppath,
                    nominal_value=base,
                    perturbed_value=applied,
                    unit_short=unit,
                )
            )

    samples: dict = {
        f"{perturbation.name}_target_id": torch.tensor(ids, dtype=torch.long)
    }
    for fk in fkeys:
        samples[f"{perturbation.name}_perturbed_{fk}"] = torch.tensor(
            diag[fk], dtype=_F64
        )

    return SampledScenarios(
        operating_point=operating_point,
        samples=samples,
        n_samples=b,
        config=perturbation,
        perturbations=perturbations,
    )


__all__ = ["perturbation_sweep"]
