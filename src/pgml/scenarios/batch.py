"""Build a scenario batch from explicit values (no sampler, no distributions).

The batch object the solver consumes is a set of per-component DELTAS on one grid:
what a dict names is varied, what it leaves out keeps the grid's nominal value. A
sampler is one way to fill that dict; handing over tensors you computed elsewhere —
measured load curves, an optimiser's iterate, a sequence a downstream generator drew —
is the other, and :func:`batch_from_values` is that entry point.

The result is an ordinary :class:`~pgml.scenarios.SampledScenarios`, so it runs through
:func:`~pgml.scenarios.run_scenarios`, persists through
:func:`~pgml.scenarios.write_dataset`, and reaches the solver through the same path a
sampled batch takes. Values pass through untouched, so gradients flow from the tensors
you supply into the solve, and their device / dtype are the ones the solver will use.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor

from pgml.errors import InputError
from pgml.schemas.grid_schema import Grid, InjectionAppliance, Source

from .sampler import SampledScenarios


def broadcast_operating_point(operating_point: dict, b: int, t: int) -> dict:
    """Broadcast a mixed ``[B]`` / ``[B, T]`` operating point to a uniform ``[B, T]``.

    A sequence batch may mix per-scenario and per-step entries: a device whose power was
    drawn once per scenario carries ``[B]``, one driven by a time series carries
    ``[B, T]``. This expands every ``[B]`` entry to ``[B, T]`` (constant over the
    sequence) and promotes a ``[B]`` source ``u_ref_scale`` to ``[B, 1]``, so every
    component shares one leading batch shape. Absent entries (nominal) and scalars
    broadcast in the solver and are left untouched. The input is not mutated.

    Parameters
    ----------
    operating_point:
        ``{appliance_id: {key: Tensor | float | list[Tensor]}}`` as
        :class:`~pgml.scenarios.SampledScenarios` carries it.
    b, t:
        Scenario count and step count of the target batch shape.

    Returns
    -------
    dict
        A new operating point with the same keys, uniformly ``[B, T]``-shaped.
    """

    def _lift(x):
        if isinstance(x, Tensor) and x.ndim == 1 and x.shape[0] == b:
            return x.unsqueeze(-1).expand(b, t).contiguous()
        return x

    out: dict = {}
    for cid, entry in operating_point.items():
        lifted: dict = {}
        for key, value in entry.items():
            if key == "u_ref_scale":
                tensor = torch.as_tensor(value)
                lifted[key] = tensor.unsqueeze(-1) if tensor.ndim == 1 else tensor
            elif key in ("p_per_phase_w", "q_per_phase_var"):
                lifted[key] = [_lift(x) for x in value]
            else:
                lifted[key] = _lift(value)
        out[cid] = lifted
    return out


def _write(
    operating_point: dict,
    values: Optional[Mapping[int, object]],
    key: str,
    ids: set,
    kind: str,
) -> None:
    """Write one field of an explicit value mapping into the operating point."""
    for cid, value in (values or {}).items():
        cid = int(cid)
        if cid not in ids:
            raise InputError(
                f"batch_from_values: {key!r} names component id {cid}, which is not an "
                f"in-service {kind} of the grid."
            )
        operating_point.setdefault(cid, {})[key] = value


def batch_from_values(
    grid: Grid,
    *,
    n_samples: int,
    n_steps: int = 1,
    p_w: Optional[Mapping[int, Tensor]] = None,
    q_var: Optional[Mapping[int, Tensor]] = None,
    p_per_phase_w: Optional[Mapping[int, Sequence[Tensor]]] = None,
    q_per_phase_var: Optional[Mapping[int, Sequence[Tensor]]] = None,
    u_ref_scale: Optional[Mapping[int, Tensor]] = None,
    harmonic_injection: Optional[Mapping[int, Mapping[int, tuple]]] = None,
    node_sources: Sequence = (),
    samples: Optional[Mapping[str, Tensor]] = None,
    shared_samples: Optional[Mapping[str, Tensor]] = None,
    config: object = None,
) -> SampledScenarios:
    """Assemble a scenario batch from explicit per-component values.

    Every mapping is keyed by component id and holds tensors whose leading axes
    broadcast against the batch shape — ``[B]`` (or a scalar) for a snapshot batch,
    ``[B, T]`` / ``[B]`` / a scalar for a sequence batch. A component that no mapping
    names, and a field a mapping leaves out, KEEP the grid's nominal value; there is no
    fill value to get wrong.

    Parameters
    ----------
    grid:
        The grid the batch varies. Component ids are resolved against its in-service
        appliances, so a typo raises here instead of being silently ignored by the solver.
    n_samples, n_steps:
        Batch shape ``B`` and step count ``T`` (``T = 1``, the default, is a snapshot
        batch). Declared rather than inferred, so the result rank is known before the
        solve: ``[B, N]`` / ``[B, H, N]`` for ``T = 1``, ``[B, T, H, N]`` for ``T > 1``.
    p_w, q_var:
        Active / reactive power per injection appliance, in W and var (absolute values,
        NOT a scale factor on the nameplate).
    p_per_phase_w, q_per_phase_var:
        Per-phase power as one tensor per connected phase, in the appliance's phase
        order. Supplying either promotes the solve to asymmetric.
    u_ref_scale:
        Multiplier on a :class:`~pgml.schemas.grid_schema.Source`'s ``u_ref_v`` — the
        batched fundamental boundary condition of the ideal-slack solve.
    harmonic_injection:
        ``{appliance_id: {order: (magnitude_pu, phase_deg)}}``, magnitude in per unit of
        the device's own fundamental current. Order 1 is the reference and is not
        injected. Passed to ``solve_harmonic_flow`` unchanged.
    node_sources:
        :class:`~pgml.solver.NodeHarmonicSource` entries (an upstream background or a
        node-level disturbance source). Build one from a
        :class:`~pgml.scenarios.BackgroundHarmonicConfig` with
        :func:`~pgml.scenarios.build_background_sources`.
    samples:
        Per-scenario records to carry along and persist, each ``[B, ...]`` (the input
        record a dataset is trained on).
    shared_samples:
        Records with no leading scenario axis (a ``time_s`` step vector, a device-id
        column). See :class:`~pgml.scenarios.SampledScenarios`.
    config:
        Any pydantic model describing how these values were produced. Serialized into
        the dataset sidecar by :func:`~pgml.scenarios.write_dataset`; a batch written
        without one cannot record its own provenance, so supply it when persisting.

    Returns
    -------
    SampledScenarios
        Validated against ``grid`` and the declared batch shape.

    Examples
    --------
    Two scenarios that halve and double one load's active power::

        batch = batch_from_values(
            grid, n_samples=2, p_w={load_id: torch.tensor([5.0e3, 2.0e4])}
        )
        result = run_scenarios(grid, batch)
    """
    injection_ids = {
        a.id
        for a in grid.appliances
        if isinstance(a, InjectionAppliance) and a.in_service
    }
    source_ids = {
        a.id for a in grid.appliances if isinstance(a, Source) and a.in_service
    }

    operating_point: dict = {}
    _write(operating_point, p_w, "p_w", injection_ids, "injection appliance")
    _write(operating_point, q_var, "q_var", injection_ids, "injection appliance")
    _write(
        operating_point,
        p_per_phase_w,
        "p_per_phase_w",
        injection_ids,
        "injection appliance",
    )
    _write(
        operating_point,
        q_per_phase_var,
        "q_per_phase_var",
        injection_ids,
        "injection appliance",
    )
    _write(operating_point, u_ref_scale, "u_ref_scale", source_ids, "source")

    injection: dict = {}
    for cid, per_order in (harmonic_injection or {}).items():
        cid = int(cid)
        if cid not in injection_ids:
            raise InputError(
                f"batch_from_values: harmonic_injection names component id {cid}, "
                "which is not an in-service injection appliance of the grid."
            )
        for order, pair in per_order.items():
            if len(pair) != 2:
                raise InputError(
                    f"batch_from_values: harmonic_injection[{cid}][{order}] must be a "
                    "(magnitude_pu, phase_deg) pair."
                )
        injection[cid] = {int(o): tuple(pair) for o, pair in per_order.items()}

    batch = SampledScenarios(
        operating_point=operating_point,
        samples=dict(samples or {}),
        n_samples=int(n_samples),
        config=config,
        harmonic_injection=injection,
        node_sources=list(node_sources),
        n_steps=int(n_steps),
        shared_samples=dict(shared_samples or {}),
    )
    batch.validate(grid)
    return batch


__all__ = ["batch_from_values", "broadcast_operating_point"]
