"""Harmonic power flow: nonlinear fundamental + linear per-harmonic solves.

Public API
----------
- ``solve_harmonic_flow(grid, harmonic_orders, *, slack, operating_point,
  harmonic_injection, node_sources, load_shunt, tol, tol_update_pu,
  s_base_va, max_iter, dtype, precision, device, symmetry, param_overrides)
  -> HarmonicFlowResult``
- ``assemble_harmonic_system(grid, harmonic_orders, v1, *, operating_point,
  harmonic_injection, node_sources, symmetry, dtype, device, param_overrides)
  -> (Y, I, index)`` —
  the assembled per-harmonic LINEAR system ``Y(h) V(h) = I(h)`` for orders
  ``h > 1`` (the building block of :func:`solve_harmonic_flow`'s harmonic slices),
  so ``r(V) = Y(h)·V − I(h)`` is the physics-consistency residual.
- ``NodeHarmonicSource`` — per-node harmonic "error" source (Thevenin / Norton),
  injected only at orders ``h > 1`` (see ``docs/pgml/modeling/error-injection.md``).

Model (matches OpenDSS ``Solve mode=harmonics`` — see
``docs/pgml/modeling/references/opendss/harmonics.md``):

1. Solve the nonlinear fundamental power flow (:func:`solve_power_flow`). Order 1 of
   the result is this solution.
2. From the converged fundamental voltage, each Load/Generator's per-ELEMENT
   FUNDAMENTAL current is ``I1_elem = sign * conj(S_eff) / conj(V_term)`` (load
   convention, ``sign`` +1 load / -1 gen), where ``S_eff`` is the power the device
   ACTUALLY draws at the converged voltage: the control-resolved (P, Q) for an
   inverter-controlled device, the ZIP-scaled ``S0*(z*r^2 + i*r + p)`` at
   ``r = |V_term|/V0`` for a voltage-dependent ``load_model``, and the base
   operating point for the const-power default — identical to the nonlinear
   fundamental solve's ``device_current_injections``. The ELEMENT (terminal)
   voltage ``V_term = M @ V_used`` uses the SAME connection incidence ``M`` the
   load flow uses (``pgml.assembly._incidence``): WYE-ground ``M = I``,
   WYE-neutral ``[I|-1]`` (terminal = ``V_phase - V_N``), DELTA-3 circulant
   (terminal = L-L difference).
3. For each harmonic order ``h > 1`` (batched over all orders): the network is
   LINEAR. ``Y(h) = assemble_network_ybus(grid, [h*f0]) + source Norton shunt``.
   Each device injects (load-terminal convention) a per-element harmonic current
   ``|I_h^e| = (mag_h^e/mag_1^e) * |I1_elem|`` and
   ``arg(I_h^e) = ang_h^e + h*(arg(I1_elem) - ang_1^e)`` from its per-ELEMENT
   spectrum coefficients (device ``spectrum`` broadcast on all elements,
   ``spectrum_per_phase`` per phase/branch, or the ``harmonic_injection`` override).
   The NODAL injection is ``-(M^T @ i_h_elem)`` (the device draws it, same sense as
   the fundamental). The source is a Norton shunt held at zero harmonic voltage (no
   ideal slack here), so ``V(h) = solve_harmonic(Y(h), I(h))`` in Norton mode.

Differentiability + GPU: the fundamental carries IFT gradients (via
:func:`solve_power_flow`); the harmonics are linear and differentiable. Gradients
flow to network params, load P/Q, and the harmonic injections. Batched over leading
scenario dims. No ``.item()/.detach()/.numpy()`` on the tape; honors device/dtype.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import torch
from torch import Tensor

from pgml.assembly import (
    FusionMap,
    NodePhaseIndex,
    assemble_network_ybus,
    node_phase_index,
)
from pgml.assembly._fusion import resolve_fusion
from pgml.assembly._incidence import build_incidence, group_appliances, used_rows
from pgml.assembly._load_shunt import (
    generation_shunt_is_neglected,
    harmonic_shunt_element_admittance,
    resolve_harmonic_shunt,
    resolve_shunt_model_name,
)
from pgml.assembly._params import phase_voltage_magnitude, resolve_operating_power
from pgml.assembly._scatter import scatter_blocks_into
from pgml.assembly._stamps import _cdtype, _rdtype
from pgml.assembly._symmetry import resolve_asymmetric
from pgml.assembly.ybus import _override, _stamp_sources
from pgml.errors import InputError
from pgml.assembly._control import resolve_injection_power
from pgml.schemas.grid_schema import (
    Grid,
    InjectionAppliance,
    Load,
    LoadModel,
    Phase,
    StaticSpectrum,
    WindingConnection,
)

from .equilibration import resolve_equilibration
from .harmonic import lu_factor_system, solve_factored, solve_harmonic
from .power_flow import (
    PowerFlowResult,
    _expand_zeroed_result,
    _reduce_block_rows,
    check_branch_impedances,
    check_connectivity,
    solve_power_flow,
)

_log = logging.getLogger("pgml")


def _integer_orders(harmonic_orders: Sequence) -> list[int]:
    """Validate + normalize harmonic orders to a non-empty list of integers.

    The harmonic machinery (spectra keyed by integer order, the per-order
    assembly) is defined for INTEGER multiples of the fundamental only. A
    non-integer order (an interharmonic, e.g. ``2.4``) would otherwise silently
    truncate to the wrong frequency, so it is rejected. Integral floats
    (``3.0``) are accepted and normalized.
    """
    orders: list[int] = []
    for h in harmonic_orders:
        hf = float(h)
        if hf != int(hf):
            raise InputError(
                f"harmonic order {h!r} is not an integer multiple of the "
                "fundamental. Interharmonics are not supported: spectra and the "
                "per-order harmonic assembly are defined for integer orders only."
            )
        orders.append(int(hf))
    if not orders:
        raise InputError("harmonic_orders must be non-empty.")
    return orders


@dataclass(frozen=True)
class HarmonicFlowResult:
    """Per-order harmonic power-flow solution.

    Attributes
    ----------
    v:
        Complex node voltages ``[*batch, H, N]`` — one slice per requested order, in
        the order of ``harmonic_orders``. Order 1 (if requested) is the nonlinear
        :func:`solve_power_flow` solution; other orders are the linear per-harmonic
        solves. DIFFERENTIABLE.
    frequencies_hz:
        Real tensor ``[H]`` = ``order * f0`` for each requested order.
    index:
        The compact :class:`NodePhaseIndex` describing the row layout of ``v``.
    pf:
        The fundamental :class:`PowerFlowResult` (convergence info + order-1 V). The
        harmonic orders are direct linear solves, so all convergence telemetry —
        :attr:`converged`, :attr:`converged_mask`, :attr:`failed_states` — comes from
        the fundamental and is re-exposed here for convenience.
    fusion:
        The :class:`~pgml.assembly.FusionMap` the solve collapsed zero-impedance
        branches with, else ``None``. ``v`` and ``index`` are the grid's FULL row
        layout at every order.
    """

    v: Tensor
    frequencies_hz: Tensor
    index: NodePhaseIndex
    pf: PowerFlowResult
    fusion: Optional[FusionMap] = None

    @property
    def converged(self) -> bool:
        """``True`` iff every scenario's fundamental solve converged."""
        return self.pf.converged

    @property
    def converged_mask(self):
        """Per-scenario convergence flags (``None`` if unbatched); see PowerFlowResult."""
        return self.pf.converged_mask

    @property
    def failed_states(self) -> tuple[int, ...]:
        """Flat indices of scenarios whose fundamental solve did not converge."""
        return self.pf.failed_states


@dataclass(frozen=True)
class NodeHarmonicSource:
    """A per-node harmonic disturbance ("error") source, injected at orders ``h > 1``.

    Models a Thevenin (``kind="voltage"``) or Norton (``kind="current"``) harmonic
    source at ANY node — independent of whether a load/generator sits there. The
    physics + math are pinned in ``docs/pgml/modeling/error-injection.md`` (authoritative).
    Because pgml solves each harmonic as its own linear system, the source is added
    ONLY at ``h > 1`` and the fundamental power flow is preserved EXACTLY (no damping
    reactor needed).

    Attributes
    ----------
    node_id:
        Id of the node to inject at.
    phases:
        Phase set to inject on (phase-to-ground). ``None`` (default) injects on ALL
        of the node's phases.
    spectrum:
        ``{order: (magnitude_pu, phase_deg)}``. Magnitudes are RELATIVE to the
        fundamental (order 1 = reference). For ``kind="voltage"`` this is a VOLTAGE
        spectrum. Order 1 may be omitted (``mag_1`` defaults to 1.0, ``ang_1`` to
        0.0). Each ``magnitude_pu`` / ``phase_deg`` may be a python float or a 0-d /
        ``[*batch]`` tensor (gradients flow; leading SCENARIO batch dims broadcast).
    source_power_va:
        Source STRENGTH ``S_sc`` (the short-circuit power, MVAsc in OpenDSS terms).
        Larger => stiffer => more of the spectrum appears at the node. A python float
        or a 0-d / ``[*batch]`` tensor (differentiable, scenario-batchable).
    kind:
        ``"voltage"`` (Thevenin: a finite-strength source — stamps the shunt ``Y_s``
        on the diagonal AND the Norton current ``I_N``) or ``"current"`` (Norton: an
        ideal current injection ``I_N`` independent of the network).
    """

    node_id: int
    phases: Optional[tuple[Phase, ...]] = None
    spectrum: dict[int, tuple] = field(default_factory=dict)
    source_power_va: float = 0.0
    kind: Literal["voltage", "current"] = "voltage"


def solve_harmonic_flow(
    grid: Grid,
    harmonic_orders,
    *,
    slack: str = "ideal",
    method: str = "current_injection",
    operating_point: Optional[dict] = None,
    harmonic_injection: Optional[dict] = None,
    node_sources: Optional[Sequence[NodeHarmonicSource]] = None,
    load_shunt: Optional[str] = None,
    tol: Optional[float] = None,
    tol_update_pu: Optional[float] = None,
    s_base_va: Optional[float] = None,
    max_iter: int = 100,
    dtype: torch.dtype = torch.complex128,
    precision: str = "full",
    device: Optional[torch.device] = None,
    symmetry: Optional[str] = None,
    on_disconnected: str = "raise",
    branch_states: Optional[dict] = None,
    branch_states_method: str = "assemble",
    param_overrides: Optional[dict] = None,
    enforce_q_limits: Optional[bool] = None,
    linear_solver: str = "auto",
    block_rows: Optional[Sequence[Tensor]] = None,
    criticality: str = "auto",
    equilibrate: Optional[str] = None,
) -> HarmonicFlowResult:
    """Solve the harmonic power flow (nonlinear fundamental + linear harmonics).

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.schemas.grid_schema.Grid`.
    harmonic_orders:
        Iterable of integer orders to solve (e.g. ``[1, 5, 7]``; 1 = fundamental).
    slack, method, operating_point, tol, tol_update_pu, s_base_va, enforce_q_limits, max_iter, dtype, device:
        Passed to the fundamental :func:`solve_power_flow`. Use ``method="newton"``
        for a stiff inverter control loop (Volt-VAr / Volt-Watt), where the
        current-injection fixed point can oscillate. The convergence tolerances are
        PER UNIT (power mismatch / voltage update) and apply to the nonlinear
        fundamental only: every harmonic order is a direct linear solve with no
        iteration and therefore no convergence criterion of its own.
        ``enforce_q_limits`` bounds a voltage-regulating generator's reactive power
        (``None`` -> the documented default); regulation is a fundamental-frequency
        concept, so it has no effect on the harmonic orders, where such a machine is
        the same Norton current source as any other generator.
    precision:
        Working precision of the linear algebra, as in :func:`solve_power_flow`:
        ``"full"`` (default) or ``"mixed"`` (complex64 factorization refined against
        complex128 residuals). It applies to the fundamental solve AND to every
        per-order harmonic solve, which is a direct solve and therefore gets the
        classic iterative refinement of :func:`pgml.solver.lu_factor_system`.
    param_overrides:
        Optional differentiability hook ``{(component_kind, element_id, field): tensor}``
        substituting network (R/L/C/Z) or device (P/Q) parameters, exactly as in
        :func:`solve_power_flow` and :func:`pgml.assembly.assemble_ybus`. It reaches the
        fundamental solve, the per-order admittance ``Y(h)``, the source stamp, and the
        device powers behind each harmonic injection, so a gradient w.r.t. an overridden
        parameter flows into every order.
    harmonic_injection:
        Optional SCENARIO override of per-device spectra, tensor-friendly so
        scenarios can vary harmonic injections differentiably. Format:
        ``{appliance_id: {order:int -> (magnitude_pu, phase_deg)}}``. Each
        ``magnitude_pu`` / ``phase_deg`` value follows an UNAMBIGUOUS convention:

        - a python ``list``/``tuple`` is PER-ELEMENT — it MUST have length ``n_elem``
          (aligned to the device's elements: WYE phase ``k`` / DELTA branch ``k``);
          each entry may itself be a python float or a 0-d / ``[*batch]`` tensor. A
          list/tuple of any other length is an error.
        - a SCALAR or bare tensor (python float, 0-d tensor, or a ``[*batch]`` tensor
          carrying ONLY leading SCENARIO batch dims, NO element axis) is BROADCAST
          identically to every element (the backward-compatible path).

        The override wins over the device's stored ``spectrum`` / ``spectrum_per_phase``.
    node_sources:
        Optional sequence of :class:`NodeHarmonicSource` — per-node harmonic "error"
        sources injected ONLY at orders ``h > 1`` (the fundamental is preserved
        exactly). Each is a Thevenin voltage source (stamps a resistive shunt ``Y_s``
        on the node-phase diagonal + a Norton current ``I_N``) or a Norton current
        source (``I_N`` only), per ``docs/pgml/modeling/error-injection.md``. Multiple
        simultaneous sources are allowed. ``None`` (default) injects no node
        harmonic sources. ``source_power_va`` and the spectrum may carry
        leading SCENARIO batch dims (differentiable; a batched voltage source's
        ``Y_s`` makes ``Y(h)`` ``[*batch, H, N, N]``).
    load_shunt:
        Which harmonic Norton shunt each Load/Generator/Storage carries in PARALLEL
        with its harmonic current source: ``"opendss"`` (the operating-point
        admittance ``conj(S)/V_rated^2`` split into a series and a parallel R-L
        branch, OpenDSS's own default), ``"motor"`` (the same with a fixed
        blocked-rotor series reactance) or ``"none"`` (a pure current source at every
        order, OpenDSS ``Set NeglectLoadY=Yes``). ``None`` (default) resolves the
        documented modeling default ``appliance.harmonic_shunt.model``; an unknown
        name raises. A device's own ``harmonic_model``
        (:class:`~pgml.schemas.grid_schema.HarmonicShuntModel`) overrides the choice
        per device, except under ``"none"``, which carries no shunt anywhere. The
        shunt is the dominant DAMPING term at a feeder parallel resonance and is
        derived from the CONVERGED fundamental operating point, so gradients flow
        from the harmonic voltages through it to P, Q and the network parameters.
    symmetry:
        Calculation-symmetry mode ``None`` / ``"auto"`` / ``"symmetric"`` /
        ``"asymmetric"`` (``None`` -> config). Resolved ONCE here and threaded into
        the fundamental :func:`solve_power_flow` (single log) and the harmonic
        injection power resolution.
    linear_solver, block_rows:
        The inner factorization backend, as in :func:`solve_power_flow`, applied to the
        nonlinear FUNDAMENTAL solve AND to every per-order harmonic solve (each order is
        one direct solve of ``Y(h) V(h) = I(h)``, which has the same sparsity and the same
        row partition as the fundamental system). ``"auto"`` (default) takes the sparse
        SuperLU factorization on large CPU systems and the batched dense torch LU
        otherwise; ``"dense"`` / ``"sparse"`` force the choice, and ``"block"`` with
        ``block_rows`` factors a block-diagonal ensemble member by member.
        ``"matrix_free"`` is a Newton-only option of the fundamental solve and leaves the
        harmonic orders on the automatic choice.
    criticality:
        When the fundamental solve runs its Jacobian criticality analysis, as in
        :func:`solve_power_flow` (``"auto"`` / ``"always"`` / ``"never"``).
    equilibrate:
        Diagonal equilibration of every factored system, as in
        :func:`solve_power_flow`: the fundamental admittance, the Newton Jacobian, the
        gradient path's adjoint, and each harmonic order's ``Y(h)``. ``None`` (default)
        resolves ``solver.equilibration.mode``. The harmonic orders are where it matters
        most: ``Y(h)``'s condition number grows with the order (measured on IEEE-33 at
        order 13: 8.0e8 as assembled, 1.5e3 equilibrated), because the series reactances
        scale with ``h`` while the diagonal collects shunt terms that do not.
    on_disconnected:
        Pre-solve connectivity handling, as in :func:`solve_power_flow`:
        ``"raise"`` (default) raises :class:`~pgml.errors.ConnectivityError` when a
        (node, phase) row has no path to an in-service source; ``"zero"`` solves the
        energized sub-grid and reports 0 V on the disconnected rows at every order
        (full-grid row layout preserved); ``"ignore"`` skips the check. The check runs
        HERE, once, for the whole harmonic study (the inner fundamental solve is then
        told not to repeat it); no policy silently disables it.
    branch_states:
        Optional topology / switch-state batching ``{branch_id: state}``, as in
        :func:`solve_power_flow`: the state (float / 0-d / ``[*batch]`` tensor,
        0 = open) OVERRIDES the branch's static flags and scales its stamp at the
        fundamental AND every harmonic order, so one batched call solves every
        switch configuration end to end. ``on_disconnected="zero"`` is unsupported
        with states (the fundamental solve enforces this).
    branch_states_method:
        How the FUNDAMENTAL solve reaches each switch state (``"assemble"`` /
        ``"woodbury"``, as in :func:`solve_power_flow`). The harmonic orders always
        assemble their own per-state ``Y(h)``: a Woodbury update is built from one
        frequency's stamps, so it does not carry to another order.

    Returns
    -------
    HarmonicFlowResult
        ``v`` complex ``[*batch, H, N]`` per requested order, frequencies, index, pf.
    """
    harmonic_shunt = resolve_shunt_model_name(load_shunt)
    orders = _integer_orders(harmonic_orders)
    fusion = resolve_fusion(
        grid, None, param_overrides=param_overrides, branch_states=branch_states
    )
    check_branch_impedances(grid, fusion=fusion, param_overrides=param_overrides)
    if on_disconnected not in ("raise", "zero", "ignore"):
        raise InputError(
            f"Unsupported on_disconnected {on_disconnected!r} "
            "(use 'raise'/'zero'/'ignore')."
        )
    eq_mode = resolve_equilibration(equilibrate)
    # The harmonic orders are direct solves of a system with the same sparsity and row
    # partition as the fundamental one, so they take the same backend. "matrix_free" is a
    # Newton option of the fundamental solve only and leaves them on the automatic choice.
    harmonic_backend = (
        linear_solver if linear_solver in ("dense", "sparse", "block") else "auto"
    )
    # The connectivity policy is executed exactly ONCE, here, for the whole study; the
    # inner fundamental solve is then told to skip the repeat. ``inner_on_disconnected``
    # makes that explicit instead of leaving a silent "ignore" in the call below.
    inner_on_disconnected = "ignore"
    if on_disconnected == "raise":
        if branch_states is None:
            check_connectivity(grid)
        else:
            # A per-scenario topology has no single connectivity verdict: the check is
            # vectorized over the state batch inside the fundamental solve, so the policy
            # is forwarded rather than run here.
            inner_on_disconnected = "raise"

    elif on_disconnected == "zero":
        if branch_states is not None:
            raise InputError(
                'on_disconnected="zero" is unsupported with branch_states: a '
                "per-scenario topology has no single energized sub-grid. Use "
                '"raise" or "ignore".'
            )
        from pgml.topology import energized_subgrid

        sub, dropped = energized_subgrid(grid)
        if dropped:
            _log.warning(
                "solve_harmonic_flow: %d disconnected node(s) %s solved as 0 V "
                '(on_disconnected="zero"); the energized sub-grid carries the '
                "solution.",
                len(dropped),
                list(dropped[:10]),
            )
            sub_res = solve_harmonic_flow(
                sub,
                orders,
                slack=slack,
                method=method,
                operating_point=operating_point,
                harmonic_injection=harmonic_injection,
                node_sources=node_sources,
                load_shunt=harmonic_shunt,
                tol=tol,
                tol_update_pu=tol_update_pu,
                s_base_va=s_base_va,
                max_iter=max_iter,
                dtype=dtype,
                precision=precision,
                device=device,
                symmetry=symmetry,
                on_disconnected="ignore",
                param_overrides=param_overrides,
                enforce_q_limits=enforce_q_limits,
                branch_states_method=branch_states_method,
                linear_solver=linear_solver,
                criticality=criticality,
                equilibrate=eq_mode,
            )
            return _expand_zeroed_harmonic_result(grid, sub_res, fusion)

    rdt = _rdtype(dtype)
    f0 = float(grid.base_frequency_hz)
    index = node_phase_index(grid)

    # Resolve calculation symmetry ONCE; thread the canonical string into the
    # fundamental PF (which emits the single modeling-summary log, and the fusion one).
    asymmetric = resolve_asymmetric(grid, operating_point, mode=symmetry)
    sym_resolved = "asymmetric" if asymmetric else "symmetric"

    # 1. Fundamental nonlinear power flow (order 1). Connectivity was already
    # handled above, so the inner solve skips the (redundant) check.
    pf = solve_power_flow(
        grid,
        slack=slack,
        method=method,
        operating_point=operating_point,
        tol=tol,
        tol_update_pu=tol_update_pu,
        s_base_va=s_base_va,
        max_iter=max_iter,
        dtype=dtype,
        precision=precision,
        device=device,
        symmetry=sym_resolved,
        on_disconnected=inner_on_disconnected,
        branch_states=branch_states,
        branch_states_method=branch_states_method,
        param_overrides=param_overrides,
        enforce_q_limits=enforce_q_limits,
        linear_solver=linear_solver,
        block_rows=block_rows,
        criticality=criticality,
        equilibrate=eq_mode,
    )
    v1 = pf.v  # [*batch, N] complex
    if device is None:
        device = v1.device

    # 2./3. Harmonic orders (> 1), batched. A per-scenario operating point makes v1
    # ``[B, N]`` while a node-coherent harmonic injection carries a DEEPER ``[B, T]`` batch.
    # ``assemble_harmonic_system`` keeps v1 in step with the (same-batch) operating point
    # when it forms each device's fundamental current, then broadcasts THAT current across
    # the injection's extra step axis. Only the ORDER-1 slice returned to the caller needs
    # its batch rank lifted to the injection's, so it stacks against the ``[B, T, N]``
    # harmonic slices (a no-op for the snapshot / nominal cases).
    harm = [h for h in orders if h != 1]
    v_by_order: dict[int, Tensor] = {1: _align_v1_batch_rank(v1, harmonic_injection)}
    if harm:
        yh, ih, _ = assemble_harmonic_system(
            grid,
            harm,
            v1,
            operating_point=operating_point,
            harmonic_injection=harmonic_injection,
            node_sources=node_sources,
            load_shunt=harmonic_shunt,
            symmetry=sym_resolved,
            dtype=dtype,
            device=device,
            branch_states=branch_states,
            param_overrides=param_overrides,
            fusion=fusion,
        )
        # Norton mode -> [*batch, Hh, N]. When Y(h) is scenario-independent (the usual
        # case — the batch varies injections, not the network), factor each order ONCE
        # and back-substitute the whole batch instead of re-factoring per scenario. A
        # batched voltage node_source promotes Y(h) to [*batch, Hh, N, N]; that path keeps
        # the per-element solve.
        if yh.ndim == 3:
            vh = solve_factored(
                lu_factor_system(
                    yh,
                    backend=harmonic_backend,
                    # Each order is factored on the FUSED rows, so a block-diagonal
                    # ensemble's partition is mapped onto them as well.
                    block_rows=_reduce_block_rows(block_rows, fusion),
                    precision=precision,
                    equilibrate=eq_mode,
                ),
                ih,
            )
        else:
            vh = solve_harmonic(yh, ih, precision=precision, equilibrate=eq_mode)
        for k, h in enumerate(harm):
            # Each order is solved on the fused rows; report it on the grid's own.
            v_by_order[h] = (
                fusion.prolong(vh[..., k, :]) if fusion is not None else vh[..., k, :]
            )

    n = index.size
    cols = [v_by_order[h] for h in orders]
    bshape = torch.broadcast_shapes(*[c.shape[:-1] for c in cols])
    cols = [c.broadcast_to(*bshape, n) for c in cols]
    v = torch.stack(cols, dim=-2)  # [*batch, H, N]
    frequencies_hz = torch.as_tensor([h * f0 for h in orders], dtype=rdt, device=device)
    return HarmonicFlowResult(
        v=v, frequencies_hz=frequencies_hz, index=index, pf=pf, fusion=fusion
    )


def _expand_zeroed_harmonic_result(
    grid: Grid, res: HarmonicFlowResult, fusion=None
) -> HarmonicFlowResult:
    """Scatter a sub-grid harmonic solution back to the full grid (0 V dead rows).

    The ``on_disconnected="zero"`` reassembly at every order: rows absent from the
    energized sub-grid report 0 V in ``v`` and in the embedded fundamental
    :class:`PowerFlowResult`. Out-of-place ``index_copy`` (gradients preserved).
    ``fusion`` is the FULL grid's map (see :func:`_expand_zeroed_result`).
    """
    full_index = node_phase_index(grid)
    sub_index = res.index
    rows = torch.as_tensor(
        [
            full_index.row(sub_index.node_id_of(r), sub_index.phase_of(r))
            for r in range(sub_index.size)
        ],
        dtype=torch.int64,
        device=res.v.device,
    )
    v_full = torch.zeros(
        (*res.v.shape[:-1], full_index.size), dtype=res.v.dtype, device=res.v.device
    ).index_copy(-1, rows, res.v)
    return HarmonicFlowResult(
        v=v_full,
        frequencies_hz=res.frequencies_hz,
        index=full_index,
        pf=_expand_zeroed_result(grid, res.pf, fusion),
        fusion=fusion,
    )


def assemble_harmonic_system(
    grid: Grid,
    harmonic_orders,
    v1: Tensor,
    *,
    operating_point: Optional[dict] = None,
    harmonic_injection: Optional[dict] = None,
    node_sources: Optional[Sequence[NodeHarmonicSource]] = None,
    load_shunt: Optional[str] = None,
    symmetry: Optional[str] = None,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    branch_states: Optional[dict] = None,
    param_overrides: Optional[dict] = None,
    fusion: Optional[object] = None,
) -> tuple[Tensor, Tensor, NodePhaseIndex]:
    """Assemble the per-harmonic LINEAR system ``Y(h) V(h) = I(h)`` for orders ``h > 1``.

    Returns EXACTLY the ``(Y, I)`` that :func:`solve_harmonic_flow` builds for the
    requested harmonic orders, so ``solve_harmonic(Y, I)`` reproduces the harmonic
    slices of :func:`solve_harmonic_flow`. The harmonic network is LINEAR, so
    ``V(h) = solve_harmonic(Y, I)`` and ``r(V) = Y(h)·V − I(h)`` is the
    physics-consistency residual (``≈ 0`` at the true ``V``). This is the hook a
    downstream package uses to form that residual without re-deriving the assembly.

    ``Y(h)`` is the passive network admittance at ``h·f0``
    (:func:`pgml.assembly.assemble_network_ybus`) plus the source Norton shunt plus each
    device's harmonic shunt (``load_shunt``) — the source is held at zero harmonic
    voltage (no ideal slack at harmonics) unless a ``node_sources`` voltage source
    stamps a shunt. ``I(h)`` is the sum of each
    device's harmonic current injection (:func:`_harmonic_injections`) plus any
    ``node_sources`` Norton/Thevenin current. The fundamental voltage ``v1`` enters
    ``I(h)`` through each device's fundamental terminal current
    ``I1_elem = sign·conj(S0_elem)/conj(V_term)`` (and, for voltage ``node_sources``,
    through ``E_h``), so gradients flow to ``v1``, to grid parameters (via ``Y(h)``
    and ``I1_elem``), and to the harmonic injection / node-source spectra.

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.schemas.grid_schema.Grid`.
    harmonic_orders:
        Iterable of integer orders ``h > 1`` to assemble (order 1 is the fundamental
        and is solved nonlinearly by :func:`solve_power_flow`; passing 1 here raises).
    v1:
        Converged fundamental node voltage ``[*batch, N]`` complex (typically
        ``solve_power_flow(grid, ...).v``), in the grid's FULL node-phase layout —
        which is the layout a result reports, fused or not.
    operating_point:
        Optional scenario P/Q override, forwarded to the harmonic-injection power
        resolution (same meaning as in :func:`solve_harmonic_flow`).
    harmonic_injection:
        Optional per-device spectrum override (same format/convention as in
        :func:`solve_harmonic_flow`).
    node_sources:
        Optional per-node Thevenin/Norton harmonic disturbance sources (see
        :class:`NodeHarmonicSource`), applied at the requested orders.
    load_shunt:
        Which device Norton shunt enters ``Y(h)`` (same meaning and default as in
        :func:`solve_harmonic_flow`). The shunt is derived from the operating point at
        the given ``v1``, so a per-scenario operating point or a batched ``v1``
        promotes ``Y`` to ``[*batch, Hh, N, N]``.
    symmetry:
        Calculation-symmetry mode ``None`` / ``"auto"`` / ``"symmetric"`` /
        ``"asymmetric"`` (``None`` -> config), governing per-phase vs balanced load
        modeling in the harmonic injection. Resolve it ONCE upstream and pass the
        canonical string when reproducing :func:`solve_harmonic_flow` exactly.
    dtype, device:
        Complex dtype and device for the assembled system (``device=None`` ->
        ``v1.device``). Honoured throughout; gradients flow on the live tape.
    branch_states:
        Optional topology / switch-state mask ``{branch_id: state}`` (see
        :func:`pgml.assembly.assemble_ybus`) — MUST match the states the
        fundamental ``v1`` was solved with. A batched state promotes ``Y`` to
        ``[*batch, Hh, N, N]``.
    param_overrides:
        Optional parameter substitution ``{(component_kind, element_id, field): tensor}``
        (see :func:`pgml.assembly.assemble_ybus`), reaching ``Y(h)``, the source stamp and
        the device powers behind the harmonic injection — the same hook
        :func:`solve_power_flow` takes, so an override used for the fundamental can be
        reused here unchanged and gradients flow to it at every order.
    fusion:
        Exact bus fusion of the zero-impedance branches (see
        :func:`pgml.assembly.assemble_ybus`). A zero-impedance branch is an ideal
        conductor at EVERY frequency, so one map serves the fundamental and every
        harmonic order. ``v1`` stays in the full layout; the returned ``Y`` / ``I`` and
        ``index`` are the reduced one.

    Returns
    -------
    Y:
        Complex ``[Hh, N, N]`` (one slice per requested order) — or ``[*batch, Hh,
        N, N]`` if a batched device shunt, a BATCHED voltage ``node_source`` or batched
        ``branch_states`` promotes it.
    I:
        Complex ``[*batch, Hh, N]`` harmonic nodal current injection.
    index:
        The compact :class:`NodePhaseIndex` describing the row layout of ``Y`` / ``I``
        (the REDUCED layout when ``fusion`` applies; ``fusion.prolong`` maps a solved
        ``V(h)`` back to the grid's full rows).
    """
    orders = _integer_orders(harmonic_orders)
    if any(h == 1 for h in orders):
        raise InputError(
            "assemble_harmonic_system assembles the LINEAR harmonic orders h > 1; "
            "order 1 is the nonlinear fundamental solved by solve_power_flow."
        )

    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)
    f0 = float(grid.base_frequency_hz)
    fused = resolve_fusion(
        grid, fusion, param_overrides=param_overrides, branch_states=branch_states
    )
    index = fused.index if fused is not None else node_phase_index(grid)
    if device is None:
        device = v1.device
    if fused is not None and v1.shape[-1] == fused.full_index.size:
        # The fundamental comes in the full layout; a fused group's rows share one
        # voltage, so reading the representative row is exact.
        v1 = fused.sample(v1)
    asymmetric = resolve_asymmetric(grid, operating_point, mode=symmetry)

    freqs = [h * f0 for h in orders]
    fvec = torch.as_tensor(freqs, dtype=rdt, device=device)
    yh = assemble_network_ybus(
        grid,
        freqs,
        dtype=dtype,
        device=device,
        branch_states=branch_states,
        param_overrides=param_overrides,
        fusion=fused,
    ).Y
    if yh.ndim == 2:  # single harmonic returned [N, N] -> [1, N, N]
        yh = yh.unsqueeze(0)
    yh = _stamp_sources(
        grid, fvec, yh, index, cdt, rdt, device, param_overrides
    )  # [Hh, N, N]
    harmonic_shunt = resolve_shunt_model_name(load_shunt)
    if harmonic_shunt != "none":
        yh = _stamp_harmonic_load_shunt(
            grid,
            yh,
            v1,
            index,
            orders,
            operating_point,
            harmonic_shunt,
            param_overrides,
            asymmetric,
            cdt,
            rdt,
            device,
        )
    ih = _harmonic_injections(
        grid,
        v1,
        index,
        orders,
        operating_point,
        harmonic_injection,
        cdt,
        rdt,
        device,
        asymmetric,
        param_overrides,
    )  # [*batch, Hh, N]
    if node_sources:
        yh, ih = _apply_node_sources(
            node_sources, grid, v1, index, orders, yh, ih, cdt, rdt, device
        )
    return _align_y_batch_rank(yh, v1, harmonic_injection), ih, index


def _align_y_batch_rank(
    yh: Tensor, v1: Tensor, harmonic_injection: Optional[dict]
) -> Tensor:
    """Right-pad a BATCHED ``Y(h)``'s scenario dims to the injection's batch rank.

    A scenario-dependent ``Y(h)`` (a device shunt or a voltage node source driven by a
    per-scenario operating point) carries the FUNDAMENTAL's batch ``[*vbatch, Hh, N,
    N]``, while a node-coherent harmonic injection carries a deeper ``[B, T]`` batch in
    ``I(h)``. Inserting the missing singleton step axes just before the order axis lets
    one matrix per scenario serve every step of that scenario. A no-op for an unbatched
    ``Y`` (which broadcasts anyway) and for the snapshot / nominal cases.
    """
    if yh.ndim <= 3:
        return yh
    extra = _injection_batch_rank(harmonic_injection) - (v1.ndim - 1)
    for _ in range(max(0, extra)):
        yh = yh.unsqueeze(-4)
    return yh


def harmonic_injections(
    grid: Grid,
    v1: Tensor,
    harmonic_orders,
    *,
    operating_point: Optional[dict] = None,
    harmonic_injection: Optional[dict] = None,
    node_sources: Optional[Sequence[NodeHarmonicSource]] = None,
    symmetry: Optional[str] = None,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    param_overrides: Optional[dict] = None,
    index: Optional[NodePhaseIndex] = None,
) -> Tensor:
    """The harmonic nodal current injection ``I(h)`` ``[*batch, Hh, N]`` alone.

    The RHS half of :func:`assemble_harmonic_system`, without assembling ``Y(h)``: each
    injecting device's connection-aware harmonic current from its converged fundamental
    terminal current and its spectrum, plus the Norton current of every
    CURRENT-kind :class:`NodeHarmonicSource`. Differentiable in ``v1``, the device
    powers and the spectra.

    ``index`` defaults to the grid's full node-phase layout, which is also the layout
    ``v1`` is expected in; pass a reduced (fused) index to obtain the injection summed
    onto the fused rows. A VOLTAGE-kind node source is refused here: it is a Thevenin
    branch to ground, i.e. an admittance as well as a current, so it belongs to the
    system assembly (:func:`assemble_harmonic_system`) and not to an injection vector.
    """
    orders = _integer_orders(harmonic_orders)
    if any(h == 1 for h in orders):
        raise InputError(
            "harmonic_injections covers the orders h > 1; the fundamental's nodal "
            "injection is the device current of the nonlinear solve "
            "(pgml.assembly.device_current_injections)."
        )
    voltage_kinds = [
        src for src in (node_sources or []) if getattr(src, "kind", None) == "voltage"
    ]
    if voltage_kinds:
        raise InputError(
            "harmonic_injections cannot represent a voltage-kind NodeHarmonicSource "
            f"(node {voltage_kinds[0].node_id}): a Thevenin background source adds a "
            "shunt admittance as well as a current, so it is part of the harmonic "
            "system (assemble_harmonic_system), not of the injection vector."
        )
    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)
    if device is None:
        device = v1.device
    idx = index if index is not None else node_phase_index(grid)
    asymmetric = resolve_asymmetric(grid, operating_point, mode=symmetry)
    ih = _harmonic_injections(
        grid,
        v1,
        idx,
        orders,
        operating_point,
        harmonic_injection,
        cdt,
        rdt,
        device,
        asymmetric,
        param_overrides,
    )
    if node_sources:
        zero_y = torch.zeros(
            (len(orders), idx.size, idx.size), dtype=cdt, device=device
        )
        _, ih = _apply_node_sources(
            node_sources, grid, v1, idx, orders, zero_y, ih, cdt, rdt, device
        )
    return ih


def assemble_harmonic_ybus(
    grid: Grid,
    harmonic_orders,
    *,
    v1: Optional[Tensor] = None,
    operating_point: Optional[dict] = None,
    load_shunt: Optional[str] = None,
    symmetry: Optional[str] = None,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    branch_states: Optional[dict] = None,
    param_overrides: Optional[dict] = None,
    fusion: Optional[object] = None,
) -> tuple[Tensor, NodePhaseIndex]:
    """The harmonic system MATRIX ``Y(h)`` for orders ``h > 1`` — no injection RHS assembled.

    Returns exactly the ``Y(h)`` of :func:`assemble_harmonic_system` (the passive network
    admittance :func:`pgml.assembly.assemble_network_ybus` plus the source Norton shunt plus
    each device's harmonic shunt, so the matrix is non-singular at the harmonics), WITHOUT
    the data-derived current ``I(h)``. This is
    the operator a physics-informed decoder learns to invert: it predicts the nodal injection
    ``I_pred(h)`` and reconstructs ``V(h) = solve_harmonic(Y(h), I_pred(h))`` — a self-consistency
    that uses ONLY the (differentiable) grid description, never a ground-truth injection. ``Y`` is
    grid-constant (factor once, reuse across a batch) and differentiable w.r.t. the network
    parameters, so the same call powers a learned grid-parameter calibration.

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.schemas.grid_schema.Grid`.
    harmonic_orders:
        Iterable of integer orders ``h > 1`` (passing order 1 raises — the fundamental is the
        passive :func:`pgml.assembly.assemble_network_ybus` at ``f0`` with an ideal slack, not a
        Norton-shunted harmonic system).
    v1, operating_point, load_shunt, symmetry:
        The device-shunt inputs (same meaning as in :func:`assemble_harmonic_system`). With
        the default ``v1=None`` there is no fundamental solution to read the operating point
        from, so every device is evaluated at its RATED terminal voltage — exact for a
        constant-power device, an approximation for a ZIP or inverter-controlled one, which is
        named in a WARNING. Pass the converged ``v1`` (or use
        :func:`assemble_harmonic_system`) to reproduce :func:`solve_harmonic_flow`'s matrix
        exactly. ``load_shunt="none"`` returns the pure network + source matrix.
    dtype, device:
        Complex dtype and device for the assembled matrix (``device=None`` -> CPU). Honoured
        throughout; gradients flow w.r.t. the network parameters on the live tape.
    branch_states:
        Optional topology / switch-state mask ``{branch_id: state}`` (see
        :func:`pgml.assembly.assemble_ybus`); a batched state promotes ``Y`` to
        ``[*batch, Hh, N, N]``.
    param_overrides:
        Optional parameter substitution (see :func:`pgml.assembly.assemble_ybus`),
        applied to the network admittance and the source stamp.
    fusion:
        Exact bus fusion of the zero-impedance branches (see
        :func:`pgml.assembly.assemble_ybus`); the returned ``index`` is then the reduced
        row layout of ``Y``.

    Returns
    -------
    Y:
        Complex ``[Hh, N, N]`` (one slice per requested order; a batched device shunt or
        batched ``branch_states`` prepend their scenario dims).
    index:
        The compact :class:`NodePhaseIndex` describing the row layout of ``Y``.
    """
    orders = _integer_orders(harmonic_orders)
    if any(h == 1 for h in orders):
        raise InputError(
            "assemble_harmonic_ybus assembles the LINEAR harmonic orders h > 1; order 1 is "
            "the fundamental (assemble_network_ybus at f0 with an ideal slack)."
        )
    harmonic_shunt = resolve_shunt_model_name(load_shunt)
    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)
    f0 = float(grid.base_frequency_hz)
    fused = resolve_fusion(
        grid, fusion, param_overrides=param_overrides, branch_states=branch_states
    )
    index = fused.index if fused is not None else node_phase_index(grid)
    if device is None:
        device = torch.device("cpu")
    if fused is not None and v1 is not None and v1.shape[-1] == fused.full_index.size:
        # A caller's fundamental comes in the grid's full layout, which the device shunt
        # reads through the REDUCED index; a fused group's rows share one voltage, so
        # reading the representative row is exact.
        v1 = fused.sample(v1)
    freqs = [h * f0 for h in orders]
    fvec = torch.as_tensor(freqs, dtype=rdt, device=device)
    yh = assemble_network_ybus(
        grid,
        freqs,
        dtype=dtype,
        device=device,
        branch_states=branch_states,
        param_overrides=param_overrides,
        fusion=fused,
    ).Y
    if yh.ndim == 2:  # single harmonic returned [N, N] -> [1, N, N]
        yh = yh.unsqueeze(0)
    yh = _stamp_sources(
        grid, fvec, yh, index, cdt, rdt, device, param_overrides
    )  # [Hh, N, N]
    if harmonic_shunt != "none":
        yh = _stamp_harmonic_load_shunt(
            grid,
            yh,
            v1,
            index,
            orders,
            operating_point,
            harmonic_shunt,
            param_overrides,
            resolve_asymmetric(grid, operating_point, mode=symmetry),
            cdt,
            rdt,
            device,
        )
    return yh, index


# ---------------------------------------------------------------------------
# harmonic current injection from spectra (connection-aware, per-element)
# ---------------------------------------------------------------------------
def _as_rt(x, rdt, device) -> Tensor:
    """Coerce a float or tensor to a real tensor, preserving grad for tensors."""
    if isinstance(x, Tensor):
        return x.to(dtype=rdt, device=device)
    return torch.as_tensor(x, dtype=rdt, device=device)


def _static_spectrum_dict(spec_obj) -> Optional[dict]:
    """``{order: (magnitude_pu, phase_deg)}`` (floats) from a ``StaticSpectrum``, or None."""
    if isinstance(spec_obj, StaticSpectrum):
        return {
            c.order: (c.magnitude_pu, c.phase_deg) for c in spec_obj.spectrum.components
        }
    return None


def _element_coeff(value, n_elem: int, rdt, device) -> Tensor:
    """Per-element coefficient tensor with trailing dim ``n_elem``.

    UNAMBIGUOUS convention (no trailing-dim sniffing, which would silently misread a
    length-``n_elem`` SCENARIO batch as per-element):

    - a python ``list``/``tuple`` is PER-ELEMENT: it MUST have length ``n_elem`` (each
      entry may itself be a python float or a 0-d / ``[*batch]`` tensor, stacked on a
      new trailing element axis, preserving per-entry grad). Any other length is an
      error.
    - anything else (python float, 0-d tensor, or a ``[*batch]`` tensor carrying ONLY
      leading SCENARIO batch dims) is SCALAR/BATCH: it is BROADCAST identically to all
      ``n_elem`` elements (append + ``expand`` the element axis — NOT by matching a
      trailing dim to ``n_elem``).

    Returns a real tensor whose trailing dim is ``n_elem`` (with any leading batch dims
    preserved).
    """
    if isinstance(value, (list, tuple)):
        if len(value) != n_elem:
            raise InputError(
                f"per-element harmonic coefficient (list/tuple) has length "
                f"{len(value)} != n_elem {n_elem}. A list/tuple is ALWAYS per-element; "
                f"use a scalar / bare tensor to broadcast across elements."
            )
        return torch.stack([_as_rt(x, rdt, device) for x in value], dim=-1)
    # scalar / leading-batch-only tensor -> broadcast across the element axis.
    t = _as_rt(value, rdt, device)
    return t.unsqueeze(-1).expand(*t.shape, n_elem)


def _device_element_spectra(
    appliance, harmonic_injection, n_elem, rdt, device
) -> Optional[dict]:
    """Per-ELEMENT per-order ``{order: (mag_elem, phase_deg_elem)}`` for a device.

    Each value is a real tensor whose trailing dim is ``n_elem`` (broadcastable with
    leading SCENARIO batch dims). Three sources, override winning:

    - ``harmonic_injection[id]`` (runtime override): ``{order: (mag, phase_deg)}``
      where a python list/tuple of length ``n_elem`` applies PER ELEMENT and a scalar
      or bare tensor BROADCASTS to all elements (see :func:`_element_coeff`).
    - ``spectrum_per_phase`` (schema): element ``k`` uses the spectrum stored at
      ``phases[k]`` (for DELTA-3 the delta branch ``k`` keys off ``phases[k]``); an
      element whose phase has no entry injects NO harmonics (all orders 0).
    - ``spectrum`` (schema, ``StaticSpectrum``): the SAME spectrum on every element.

    Returns ``None`` if the device injects no harmonics at all.
    """
    phases = appliance.phases  # element k <-> phases[k] (WYE phase / DELTA branch)

    override = (
        harmonic_injection.get(appliance.id) if harmonic_injection is not None else None
    )
    if override is not None:
        return {
            int(o): (
                _element_coeff(mag, n_elem, rdt, device),
                _element_coeff(ph, n_elem, rdt, device),
            )
            for o, (mag, ph) in override.items()
        }

    spp = getattr(appliance, "spectrum_per_phase", None)
    if spp is not None:
        # Gather the set of orders any element carries; element k uses phases[k].
        per_elem_dicts = [
            _static_spectrum_dict(spp.get(phases[k])) for k in range(n_elem)
        ]
        all_orders: set[int] = set()
        for d in per_elem_dicts:
            if d is not None:
                all_orders.update(d.keys())
        if not all_orders:
            return None
        out: dict[int, tuple] = {}
        for o in all_orders:
            mags = [
                (d[o][0] if (d is not None and o in d) else 0.0) for d in per_elem_dicts
            ]
            phs = [
                (d[o][1] if (d is not None and o in d) else 0.0) for d in per_elem_dicts
            ]
            out[o] = (
                torch.stack([_as_rt(m, rdt, device) for m in mags], dim=-1),
                torch.stack([_as_rt(p, rdt, device) for p in phs], dim=-1),
            )
        return out

    sd = _static_spectrum_dict(getattr(appliance, "spectrum", None))
    if sd is None:
        return None
    return {
        o: (
            _as_rt(mag, rdt, device).reshape(()).expand(n_elem),
            _as_rt(ph, rdt, device).reshape(()).expand(n_elem),
        )
        for o, (mag, ph) in sd.items()
    }


def _injection_batch_rank(harmonic_injection: Optional[dict]) -> int:
    """The deepest leading batch rank of a ``harmonic_injection`` override (0 if none).

    Each ``(magnitude, phase)`` coefficient may be a python float / 0-d tensor (scalar,
    rank 0), a ``[B]`` tensor (snapshot, rank 1), or a ``[B, T]`` tensor (node-coherent
    sequence, rank 2). Per-element list/tuple coefficients recurse element-wise.
    """
    if not harmonic_injection:
        return 0

    def rank(x) -> int:
        if isinstance(x, (list, tuple)):
            return max((rank(e) for e in x), default=0)
        return x.ndim if isinstance(x, Tensor) else 0

    r = 0
    for order_map in harmonic_injection.values():
        for coeff in order_map.values():
            for part in coeff:
                r = max(r, rank(part))
    return r


def _pad_batch_before_elem(t: Tensor, target_ndim: int) -> Tensor:
    """Insert singleton axes just before ``t``'s trailing (element) axis up to ``target_ndim``.

    ``t`` is ``[*batch, n_elem]``; padding to ``[*batch, 1, ..., 1, n_elem]`` lets a
    per-scenario tensor broadcast against a deeper-batched one that shares the ``n_elem``
    trailing axis. A no-op when ``t`` already has ``>= target_ndim`` dims.
    """
    for _ in range(max(0, target_ndim - t.ndim)):
        t = t.unsqueeze(-2)
    return t


def _align_v1_batch_rank(v1: Tensor, harmonic_injection: Optional[dict]) -> Tensor:
    """Right-pad v1's batch with singleton axes to reach the injection's batch rank.

    v1 is ``[*vbatch, N]``; the injection may carry a deeper batch (a node-coherent
    ``[B, T]`` injection over a ``[B]`` fundamental). Inserting the missing singleton
    axes just before the N axis (``[B, N]`` -> ``[B, 1, N]``) lets the fundamental
    broadcast across the extra (step) dims. A no-op (returns v1 unchanged) when v1's
    batch rank already meets the injection's — the snapshot and nominal cases.
    """
    extra = _injection_batch_rank(harmonic_injection) - (v1.ndim - 1)
    for _ in range(max(0, extra)):
        v1 = v1.unsqueeze(-2)
    return v1


def _element_terminal_voltage(m_c: Tensor, rows: Tensor, v1: Tensor) -> Tensor:
    """Per-ELEMENT terminal voltage ``V_term = M @ V_used`` ``[*vbatch, n_elem]``.

    ``m_c`` is the group's complex incidence ``[n_elem, n_used]``, ``rows`` the
    device's global ``used_rows`` ``[n_used]``: WYE-ground reads the phase rows,
    WYE-neutral the phase-minus-neutral difference, DELTA-3 the line-to-line
    difference.
    """
    v_used = v1.index_select(-1, rows).to(m_c.dtype)  # [*vbatch, n_used]
    return torch.einsum("eu,...u->...e", m_c, v_used)


def _effective_element_power(
    a,
    grp,
    node_map,
    vt: Tensor,
    operating_point,
    param_overrides,
    asymmetric: bool,
    cdt,
    rdt,
    device,
) -> Tensor:
    """Per-ELEMENT complex power the device draws at the fundamental ``[*batch, n_elem]``.

    The load-convention signed power (``+`` for a :class:`~pgml.schemas.grid_schema
    .Load`, ``-`` for a Generator/Storage) the device ACTUALLY draws at the converged
    fundamental terminal voltage ``vt``: the control-resolved ``(P, Q)`` for an
    inverter-controlled device, the ZIP-scaled ``S0*(z*r^2 + i*r + p)`` at
    ``r = |V_term|/V0`` for a voltage-dependent ``load_model``, and the base operating
    point for the const-power default — exactly as the nonlinear fundamental solve
    resolves it (:func:`pgml.assembly.device_current_injections`). Both the harmonic
    current injection and the harmonic shunt are built from this one quantity, so they
    cannot describe different operating points. Differentiable in P, Q and (through the
    voltage-dependent laws) in ``vt``.
    """
    sign = 1.0 if isinstance(a, Load) else -1.0
    p_list, q_list = resolve_operating_power(a, operating_point, asymmetric=asymmetric)
    p_t = torch.stack([_as_rt(x, rdt, device) for x in p_list], dim=-1)  # [*b,n_elem]
    q_t = torch.stack([_as_rt(x, rdt, device) for x in q_list], dim=-1)
    # Same parameter-substitution hook the fundamental solve uses, so a differentiable
    # P/Q override drives the harmonic path too (the override replaces the whole
    # per-element vector, as in the assembly).
    kind = "load" if isinstance(a, Load) else "generator"
    p_t = _override(param_overrides, (kind, a.id, "p_nom_per_phase_w"), p_t)
    q_t = _override(param_overrides, (kind, a.id, "q_nom_per_phase_var"), q_t)

    lm = getattr(a, "load_model", None)
    if getattr(a, "control", None) is None and (
        lm is None or lm is LoadModel.CONST_POWER
    ):
        return torch.complex(sign * p_t, sign * q_t).to(cdt)  # [*b, n_elem]

    is_delta = grp.connection == WindingConnection.DELTA
    v0 = phase_voltage_magnitude(
        node_map[a.node].u_rated_v,
        len(node_map[a.node].phases),
        line_to_line=is_delta,
    )
    if getattr(a, "control", None) is not None:
        v_pu = (torch.abs(vt) / v0).unsqueeze(-2)  # [*vbatch, 1, n_elem]
        p_eff, q_eff = resolve_injection_power(
            a.control, p_t, v_pu, rdt=rdt, device=device
        )
        return torch.complex(sign * p_eff.squeeze(-2), sign * q_eff.squeeze(-2)).to(cdt)

    from pgml.assembly.ybus import _zip_coeffs

    zip_p, zip_q = _zip_coeffs(a, rdt, device)  # [3] constants
    r = torch.abs(vt) / v0  # [*vbatch, n_elem]
    scale_p = zip_p[0] * r * r + zip_p[1] * r + zip_p[2]
    scale_q = zip_q[0] * r * r + zip_q[1] * r + zip_q[2]
    return torch.complex(sign * p_t * scale_p, sign * q_t * scale_q).to(cdt)


def _harmonic_injections(
    grid,
    v1,
    index,
    harm_orders,
    operating_point,
    harmonic_injection,
    cdt,
    rdt,
    device,
    asymmetric=True,
    param_overrides=None,
) -> Tensor:
    """Per-device, connection-aware harmonic nodal current injection ``[*batch, Hh, N]``.

    Mirrors :func:`pgml.assembly.ybus.device_current_injections`: each injecting
    Load/Generator has a terminal incidence ``M`` ``[n_elem, n_used]``
    (``V_term = M @ V_used``); WYE-ground ``M = I``, WYE-neutral ``[I|-1]``, DELTA-3
    circulant. The per-ELEMENT fundamental current is
    ``I1_elem = sign*conj(S_eff)/conj(V_term)`` (TERMINAL voltage, not the phase
    row; ``S_eff`` = the model-consistent power at the converged voltage —
    control-resolved, ZIP-scaled, or the base operating point, exactly as the
    fundamental solve draws it), then per element and order

        ``|I_h^e| = (mag_h^e/mag_1^e)*|I1_elem|``
        ``arg(I_h^e) = ang_h^e + h*(arg(I1_elem) - ang_1^e)``

    with PER-ELEMENT spectrum coefficients (device ``spectrum`` broadcast on all
    elements, ``spectrum_per_phase`` per phase/branch, or the ``harmonic_injection``
    override). In the override, a coefficient given as a python ``list``/``tuple`` of
    length ``n_elem`` is PER-ELEMENT, while a scalar or bare tensor is BROADCAST to all
    elements (see :func:`_element_coeff`). The NODAL injection (drawn) is
    ``I_used = -(M^T @ i_h_elem)`` scattered into the device's ``used_rows``.
    WYE-to-ground reduces EXACTLY to the historical per-phase form (bit-exact
    regression). Fully broadcasting so per-element / per-order coefficients may carry
    leading SCENARIO batch dims (differentiable override).
    """
    n = index.size
    node_map = {nd.id: nd for nd in grid.nodes}

    loads = [
        a for a in grid.appliances if isinstance(a, InjectionAppliance) and a.in_service
    ]

    # Per device that injects: (group, rows[n_used], i1_mag[*b,n_elem],
    # i1_ang[*b,n_elem], {order:(mag_elem,ang1_elem?)} via element spectra).
    # We collect the per-element fundamental current and the element spectra, grouped
    # so the incidence M (a topology constant) is shared.
    entries = []  # (m_c, rows, i1_mag, i1_ang, elem_spectra, n_elem)
    for grp in group_appliances(loads, node_map):
        n_elem = grp.n_elem
        m = build_incidence(grp, rdt, device)  # [n_elem, n_used] real
        m_c = m.to(cdt)
        rows_grp = used_rows(grp, index, device)  # [K, n_used] int64
        for ki, a in enumerate(grp.appliances):
            elem_spectra = _device_element_spectra(
                a, harmonic_injection, n_elem, rdt, device
            )
            if elem_spectra is None:
                continue  # device injects no harmonics.
            rows = rows_grp[ki]  # [n_used]
            vt = _element_terminal_voltage(m_c, rows, v1)  # [*vbatch, n_elem]
            # The harmonic current scales from the FUNDAMENTAL current the device
            # actually draws, so the injected spectrum is anchored to the current the
            # device carries at order 1.
            s0 = _effective_element_power(
                a,
                grp,
                node_map,
                vt,
                operating_point,
                param_overrides,
                asymmetric,
                cdt,
                rdt,
                device,
            )  # [*batch, n_elem]
            # Guard the conj(vt) divide for a dead/disconnected terminal (vt == 0)
            # or a gradcheck perturbation toward zero: mask the DENOMINATOR before
            # dividing (so conj(s0)/0 never enters the graph), then mask the RESULT
            # to 0 afterwards. Same two-step pattern as the mag1 == 0 ratio guard.
            vtc = torch.conj(vt)
            safe_vtc = torch.where(vtc.abs() < 1e-300, torch.ones_like(vtc), vtc)
            i1 = torch.where(
                vtc.abs() < 1e-300, torch.zeros_like(s0), torch.conj(s0) / safe_vtc
            )  # [*batch, n_elem]

            mag1_e, ang1_e = elem_spectra.get(
                1,
                (
                    torch.ones(n_elem, dtype=rdt, device=device),
                    torch.zeros(n_elem, dtype=rdt, device=device),
                ),
            )
            ang1_e = ang1_e * (math.pi / 180.0)
            entries.append(
                (
                    m_c,
                    rows,
                    torch.abs(i1),
                    torch.angle(i1),
                    mag1_e,
                    ang1_e,
                    elem_spectra,
                )
            )

    hh = len(harm_orders)
    if not entries:
        return torch.zeros((hh, n), dtype=cdt, device=device)

    # ONE vectorized evaluation per device: the spectra coefficients are stacked
    # over the order axis ([Hh, n_elem], python dict lookups only), so the ratio /
    # phase / polar math and the incidence contraction run for every order at once
    # and the nodal scatter is a single index_add per device — instead of tape ops
    # per (device × order), which dominated multi-order assemblies.
    h_vec = torch.as_tensor([float(h) for h in harm_orders], dtype=rdt, device=device)
    zero_e = None
    parts = []  # (rows [n_used], i_used [*batch, Hh, n_used])
    for m_c, rows, i1_mag, i1_ang, mag1_e, ang1_e, elem_spectra in entries:
        n_elem = mag1_e.shape[-1]
        if zero_e is None or zero_e.shape[-1] != n_elem:
            zero_e = torch.zeros(n_elem, dtype=rdt, device=device)
        pairs = [elem_spectra.get(h, (zero_e, zero_e)) for h in harm_orders]
        cshape = torch.broadcast_shapes(*[t.shape for p in pairs for t in p])
        mag_h = torch.stack([p[0].broadcast_to(cshape) for p in pairs], dim=-2)
        ph_h = torch.stack([p[1].broadcast_to(cshape) for p in pairs], dim=-2)
        # mag_h / ph_h: [*cbatch, Hh, n_elem]; i1 terms gain the order axis at -2.
        # Safe ratio mag_h/mag1: an element with NO spectrum has mag1 == 0 (and
        # mag_h == 0) -> it injects nothing; avoid the 0/0 NaN with a guarded
        # divide (the where keeps the gradient finite on the live elements).
        mag1 = mag1_e.unsqueeze(-2)
        safe1 = torch.where(mag1 == 0, torch.ones_like(mag1), mag1)
        ratio = torch.where(
            mag1 == 0, torch.zeros_like(mag_h), mag_h / safe1
        )  # [*cbatch, Hh, n_elem]
        # The device's fundamental current ``i1`` (``[*vbatch, n_elem]``) follows the
        # operating point / v1 batch; the spectrum ratio may carry a DEEPER batch (a
        # node-coherent ``[B, T]`` injection over a ``[B]`` fundamental). Insert the
        # missing singleton step axes just before the element axis so the per-scenario
        # fundamental current broadcasts across the extra step dims (a no-op when the
        # batches already match — the snapshot / nominal cases).
        i1_mag_b = _pad_batch_before_elem(i1_mag, mag_h.ndim - 1)
        i1_ang_b = _pad_batch_before_elem(i1_ang, mag_h.ndim - 1)
        mag = ratio * i1_mag_b.unsqueeze(-2)
        phase = ph_h * (math.pi / 180.0) + h_vec[:, None] * (
            i1_ang_b.unsqueeze(-2) - ang1_e.unsqueeze(-2)
        )
        i_h_elem = torch.polar(mag, phase)  # [*batch, Hh, n_elem]
        # Nodal current at the used rows: I_used = -(M^T @ i_elem) (drawn).
        i_used = torch.einsum("eu,...e->...u", m_c, i_h_elem)  # [*batch, Hh, n_used]
        parts.append((rows, i_used))

    bshape = torch.broadcast_shapes(*[p.shape[:-2] for _, p in parts])
    out = torch.zeros((*bshape, hh, n), dtype=cdt, device=device)
    for rows, i_used in parts:
        # `broadcast_to` returns a VIEW; a non-contiguous complex tensor can fail
        # the CUDA index_add backend, so materialise it (defensive, matches
        # ybus._scatter_injection). `.contiguous()` is autograd-safe.
        i_used = i_used.broadcast_to(*bshape, hh, rows.shape[0]).contiguous()
        # Out-of-place index_add (GPU-safe for COMPLEX, unlike scatter_add).
        out = out.index_add(-1, rows, -i_used)
    return out  # [*batch, Hh, N]


# ---------------------------------------------------------------------------
# harmonic device shunt (OpenDSS load model) — assembly/_load_shunt.py
# ---------------------------------------------------------------------------
def _stamp_harmonic_load_shunt(
    grid,
    yh,
    v1,
    index,
    harm_orders,
    operating_point,
    harmonic_shunt,
    param_overrides,
    asymmetric,
    cdt,
    rdt,
    device,
):
    """Add every device's harmonic Norton shunt to ``Y(h)`` ``[*batch, Hh, N, N]``.

    Connection-aware exactly like the harmonic current injection: the per-ELEMENT
    admittance ``y_elem(h)`` of :func:`pgml.assembly._load_shunt
    .harmonic_shunt_element_admittance` is mapped to the nodal block
    ``M^T diag(y_elem(h)) M`` with the SAME terminal incidence ``M``
    (:mod:`pgml.assembly._incidence`), so a WYE-ground device stamps a diagonal, a
    WYE-neutral device returns through the neutral row, and a DELTA device puts each
    leg admittance on both of its phase diagonals (reproducing OpenDSS's delta
    ``YPrim``, whose diagonal reads ``2*Y_leg``).

    ``harmonic_shunt`` is the validated run-level model name; each device's own
    ``harmonic_model`` may override it (:func:`pgml.assembly._load_shunt
    .resolve_harmonic_shunt`). The element power is the one the device draws at the
    fundamental (:func:`_effective_element_power`), so ``Y(h)`` depends on the
    fundamental solution ``v1`` for a voltage-dependent device and gradients flow from
    the harmonic voltages through the shunt to P, Q and ``v1``. A per-scenario
    operating point therefore promotes ``Y(h)`` to ``[*batch, Hh, N, N]``.

    ``v1 = None`` (no fundamental solution available, as in
    :func:`assemble_harmonic_ybus`) evaluates every device at its RATED terminal
    voltage, which is exact for a constant-power device and an approximation for a
    ZIP / controlled one — such a device is named in a WARNING.
    """
    loads = [
        a for a in grid.appliances if isinstance(a, InjectionAppliance) and a.in_service
    ]
    if not loads:
        return yh
    node_map = {nd.id: nd for nd in grid.nodes}
    h_vec = torch.as_tensor([float(h) for h in harm_orders], dtype=rdt, device=device)
    rated_only = []
    generation_free = [a.id for a in loads if generation_shunt_is_neglected(a)]
    for grp in group_appliances(loads, node_map):
        specs = [resolve_harmonic_shunt(a, harmonic_shunt) for a in grp.appliances]
        keep = [ki for ki, sp in enumerate(specs) if sp.kind != "none"]
        if not keep:
            continue
        is_delta = grp.connection == WindingConnection.DELTA
        m_c = build_incidence(grp, rdt, device).to(cdt)  # [n_elem, n_used]
        rows_grp = used_rows(grp, index, device)  # [K, n_used]
        s_list, v_list = [], []
        for ki in keep:
            a = grp.appliances[ki]
            node = node_map[a.node]
            v0 = phase_voltage_magnitude(
                node.u_rated_v, len(node.phases), line_to_line=is_delta
            )
            v0_t = _as_rt(v0, rdt, device)
            if v1 is None:
                # Rated-voltage terminal: exact for a constant-power device.
                vt = torch.polar(
                    v0_t.expand(grp.n_elem),
                    torch.zeros(grp.n_elem, dtype=rdt, device=device),
                ).to(cdt)
                if getattr(a, "control", None) is not None or (
                    getattr(a, "load_model", None) not in (None, LoadModel.CONST_POWER)
                ):
                    rated_only.append(a.id)
            else:
                vt = _element_terminal_voltage(m_c, rows_grp[ki], v1)
            s_list.append(
                _effective_element_power(
                    a,
                    grp,
                    node_map,
                    vt,
                    operating_point,
                    param_overrides,
                    asymmetric,
                    cdt,
                    rdt,
                    device,
                )
            )
            v_list.append(v0_t)
        # Stack the kept devices at -2: [*batch, K, n_elem]; a device without a batched
        # override broadcasts its nominal power into the common scenario batch.
        lead = torch.broadcast_shapes(*[t.shape[:-1] for t in s_list])
        s_elem = torch.stack([t.broadcast_to(*lead, t.shape[-1]) for t in s_list], -2)
        v_rated = torch.stack(v_list, -1).unsqueeze(-1)  # [K, 1]
        series_rl = torch.as_tensor(
            [specs[ki].series_rl_fraction for ki in keep], dtype=rdt, device=device
        ).unsqueeze(-1)
        motor_x_pu = torch.as_tensor(
            [specs[ki].motor_x_harm_pu for ki in keep], dtype=rdt, device=device
        ).unsqueeze(-1)
        motor_xr = torch.as_tensor(
            [specs[ki].motor_xr_harm for ki in keep], dtype=rdt, device=device
        ).unsqueeze(-1)
        # Apparent-power base of the motor reactance: the element's own power for a WYE
        # device (one single-phase element per phase), the device total for a DELTA one.
        motor_s_base = (
            torch.abs(s_elem.sum(-1, keepdim=True)) if is_delta else torch.abs(s_elem)
        )
        y_elem = harmonic_shunt_element_admittance(
            s_elem,
            v_rated,
            h_vec,
            series_rl,
            motor_x_pu=motor_x_pu,
            motor_xr=motor_xr,
            motor_s_base=motor_s_base,
            cdtype=cdt,
        )  # [*batch, Hh, K, n_elem]
        block = torch.einsum("ei,...ke,ej->...kij", m_c, y_elem, m_c)
        rows = rows_grp[torch.as_tensor(keep, dtype=torch.int64, device=device)]
        yh = scatter_blocks_into(yh, block, rows, rows)
    if rated_only:
        _log.warning(
            "harmonic load shunt: %d voltage-dependent device(s) %s evaluated at their "
            "RATED terminal voltage (no fundamental solution given); pass the "
            "fundamental voltage for the exact operating point.",
            len(rated_only),
            rated_only[:10],
        )
    if generation_free:
        _log.warning(
            "harmonic device shunt: %d generation device(s) %s carry NO shunt and stay "
            "pure harmonic current sources (appliance.harmonic_shunt.generation_model = "
            "'none'): the load expression conj(S)/V_rated^2 has a negative conductance "
            "for an injecting device. Set that default to 'load_style' to apply it "
            "anyway (OpenDSS's negative-kW Load idiom), or give the device the motor "
            "model.",
            len(generation_free),
            generation_free[:10],
        )
    return yh


def _harmonic_shunt_currents(
    grid: Grid,
    v1: Tensor,
    vh: Tensor,
    harmonic_orders,
    *,
    operating_point: Optional[dict] = None,
    load_shunt: Optional[str] = None,
    symmetry: Optional[str] = None,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    param_overrides: Optional[dict] = None,
    index: Optional[NodePhaseIndex] = None,
) -> Tensor:
    """The current the devices' harmonic Norton shunts DRAW, ``[*batch, Hh, N]``.

    ``Y_shunt(h) V(h)`` for the shunt model ``load_shunt`` selects (``None`` = the
    documented default; ``"none"`` returns zeros): the part of each device's harmonic
    nodal current that :func:`assemble_harmonic_system` carries on the LEFT-hand side,
    inside ``Y(h)``. The DEVICE-side nodal current at order ``h`` is therefore
    :func:`harmonic_injections` MINUS this term, which is what a nodal current balance
    outside the solve needs — the recovery of the current through a fused
    (zero-impedance) branch, for instance.

    ``v1`` is the converged fundamental ``[*batch, N]`` (the shunt is derived from the
    operating point there), ``vh`` the solved harmonic voltages ``[*batch, Hh, N]`` for
    the SAME orders, both in the row layout of ``index`` (default: the grid's full
    node-phase layout). Differentiable in ``v1``, ``vh`` and the device powers; batched;
    device and dtype follow the inputs.
    """
    orders = _integer_orders(harmonic_orders)
    if any(h == 1 for h in orders):
        raise InputError(
            "_harmonic_shunt_currents covers the orders h > 1; the fundamental's "
            "device current is the nonlinear solve's own "
            "(pgml.assembly.device_current_injections)."
        )
    harmonic_shunt = resolve_shunt_model_name(load_shunt)
    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)
    if device is None:
        device = vh.device
    idx = index if index is not None else node_phase_index(grid)
    if harmonic_shunt == "none":
        return torch.zeros(vh.shape, dtype=cdt, device=device)
    y_shunt = _stamp_harmonic_load_shunt(
        grid,
        torch.zeros((len(orders), idx.size, idx.size), dtype=cdt, device=device),
        v1,
        idx,
        orders,
        operating_point,
        harmonic_shunt,
        param_overrides,
        resolve_asymmetric(grid, operating_point, mode=symmetry),
        cdt,
        rdt,
        device,
    )  # [*batch, Hh, N, N]
    return torch.einsum("...hij,...hj->...hi", y_shunt, vh.to(cdt))


# ---------------------------------------------------------------------------
# per-node harmonic "error" source (Thevenin / Norton) — docs/pgml/modeling/error-injection.md
# ---------------------------------------------------------------------------
def _apply_node_sources(
    node_sources,
    grid,
    v1,
    index,
    harm_orders,
    yh,
    ih,
    cdt,
    rdt,
    device,
):
    """Stamp per-node harmonic disturbance sources into ``Y(h)`` and ``I(h)``.

    For each :class:`NodeHarmonicSource` and each requested order ``h > 1``, at the
    source node-phase rows (``index.rows_for_terminal``) — per
    ``docs/pgml/modeling/error-injection.md``:

    - ``V_base`` = node line-to-neutral base
      (:func:`pgml.assembly._params.phase_voltage_magnitude`).
    - ``Y_s = source_power_va / V_base**2`` (REAL, frequency-flat / resistive).
    - ``E_h = (mag_h/mag_1)*|V1| * exp(j*(rad(ang_h) + h*(angle(V1) - rad(ang_1))))``
      using the SAME phase convention as :func:`_harmonic_injections`. ``V1`` is the
      converged fundamental voltage at the row.
    - ``I_N = E_h * Y_s``.
    - ``kind="voltage"``: add ``Y_s`` to the diagonal ``Y(h)[..., row, row]`` AND
      ``I_N`` to ``I(h)[..., row]``.
    - ``kind="current"``: add ``I_N`` to ``I(h)[..., row]`` only.

    All adds are out-of-place (``index_add`` on a flattened diagonal of a fresh zero
    tensor for ``Y``; ``index_add`` on the current). ``source_power_va`` and the
    spectrum may carry leading SCENARIO batch dims; a batched voltage-source ``Y_s``
    promotes ``Y(h)`` to ``[*batch, Hh, N, N]`` (broadcast, then added).
    Differentiable w.r.t. ``source_power_va``, the spectrum, and (via ``V1``) grid
    params; GPU-safe. ``yh`` is ``[Hh, N, N]`` (or already batched); ``ih`` is
    ``[*batch, Hh, N]``. Returns the updated ``(yh, ih)``.
    """
    n = index.size
    node_map = {nd.id: nd for nd in grid.nodes}

    # Accumulate per-order current and (voltage-source) diagonal Y_s contributions.
    i_cols: list[Tensor] = []  # one [*batch, N] per order (current contribution)
    y_diag_cols: list[Tensor] = []  # one [*batch, N] per order (diagonal Y_s add)
    has_voltage = False
    for h in harm_orders:
        i_acc = torch.zeros((n,), dtype=cdt, device=device)
        y_acc = torch.zeros((n,), dtype=cdt, device=device)
        for src in node_sources:
            if src.kind not in ("voltage", "current"):
                raise InputError(
                    f"NodeHarmonicSource.kind must be 'voltage' or 'current', "
                    f"got {src.kind!r}."
                )
            node = node_map[src.node_id]
            phases = src.phases if src.phases is not None else node.phases
            rows = index.rows_for_terminal(src.node_id, phases, device=device)  # [P]

            v_base = phase_voltage_magnitude(node.u_rated_v, len(node.phases))
            # Y_s = S_sc / V_base^2 (REAL, frequency-flat). Keeps grad to S_sc.
            s_sc = _as_rt(src.source_power_va, rdt, device)  # 0-d or [*batch]
            y_s = (s_sc / (v_base * v_base)).to(cdt)  # [*batch]

            # Order-1 reference (mag_1, ang_1): default mag_1=1.0, ang_1=0.0.
            mag1, ang1_deg = src.spectrum.get(1, (1.0, 0.0))
            mag1 = _as_rt(mag1, rdt, device)
            ang1 = _as_rt(ang1_deg, rdt, device) * (math.pi / 180.0)

            mag_h_raw, ang_h_deg = src.spectrum.get(h, (0.0, 0.0))
            mag_h = _as_rt(mag_h_raw, rdt, device)
            ang_h = _as_rt(ang_h_deg, rdt, device) * (math.pi / 180.0)

            # Guarded ratio mag_h / mag_1 (mag_1 == 0 -> contributes nothing).
            safe1 = torch.where(mag1 == 0, torch.ones_like(mag1), mag1)
            ratio = torch.where(mag1 == 0, torch.zeros_like(mag_h), mag_h / safe1)

            # Fundamental voltage at the source rows: V1[..., rows] -> [*vbatch, P].
            v1_rows = v1.index_select(-1, rows).to(cdt)
            absv1 = torch.abs(v1_rows)  # [*vbatch, P]
            argv1 = torch.angle(v1_rows)  # [*vbatch, P]

            # E_h = ratio*|V1| * exp(j*(ang_h + h*(arg(V1) - ang_1))).
            mag_e = ratio.unsqueeze(-1) * absv1  # broadcast -> [*batch, P]
            phase_e = ang_h.unsqueeze(-1) + float(h) * (argv1 - ang1.unsqueeze(-1))
            e_h = torch.polar(mag_e, phase_e).to(cdt)  # [*batch, P]

            i_n = e_h * y_s.unsqueeze(-1)  # [*batch, P]
            i_n = i_n.broadcast_to(*i_n.shape[:-1], rows.shape[0]).contiguous()
            i_acc = _index_add_into(i_acc, rows, i_n, n, cdt, device)

            if src.kind == "voltage":
                has_voltage = True
                y_row = (
                    y_s.unsqueeze(-1)
                    .broadcast_to(*y_s.shape, rows.shape[0])
                    .contiguous()
                )
                y_acc = _index_add_into(y_acc, rows, y_row, n, cdt, device)
        i_cols.append(i_acc)
        y_diag_cols.append(y_acc)

    # Stack the per-order current contribution -> [*batch, Hh, N] and add to ih.
    ibshape = torch.broadcast_shapes(*[c.shape[:-1] for c in i_cols])
    i_stack = torch.stack(
        [c.broadcast_to(*ibshape, n) for c in i_cols], dim=-2
    )  # [*batch, Hh, N]
    ih = ih + i_stack

    # Voltage sources add Y_s to the diagonal. Build the diagonal add [*batch, Hh, N]
    # and add it onto Y's diagonal out-of-place (broadcasting Y if it gains a batch).
    if has_voltage:
        ybshape = torch.broadcast_shapes(*[c.shape[:-1] for c in y_diag_cols])
        y_diag = torch.stack(
            [c.broadcast_to(*ybshape, n) for c in y_diag_cols], dim=-2
        )  # [*batch, Hh, N]
        yh = _add_to_diagonal(yh, y_diag, cdt, device)

    return yh, ih


def _index_add_into(acc, rows, values, n, cdt, device):
    """Out-of-place ``acc.index_add(-1, rows, values)`` with leading-batch broadcast.

    ``acc`` may be a bare ``[N]`` (the running zero) — promote it to the broadcast
    batch of ``values`` (a FRESH zero tensor, never an in-place mutation) before the
    complex-safe out-of-place ``index_add``.
    """
    target_batch = values.shape[:-1]
    base = torch.zeros((*target_batch, n), dtype=cdt, device=device)
    base = base + acc  # broadcast the previous accumulation into the batch (fresh).
    return base.index_add(-1, rows, values)


def _add_to_diagonal(yh, y_diag, cdt, device):
    """Add ``y_diag`` ``[*batch, Hh, N]`` onto the diagonal of ``yh`` out-of-place.

    ``yh`` is ``[Hh, N, N]`` or ``[*batch, Hh, N, N]``; ``y_diag`` carries the per-row
    diagonal additions (possibly with a leading SCENARIO batch). The result broadcasts
    to ``[*batch, Hh, N, N]``. Builds a diagonal-only complex matrix via an
    ``index_add`` on the flattened ``(N*N)`` last two dims of a FRESH zero tensor (no
    in-place op on the tracked ``yh``; GPU-safe for complex).
    """
    n = yh.shape[-1]
    # Common batch shape over the leading dims of yh[...,N,N] and y_diag[...,N].
    bshape = torch.broadcast_shapes(yh.shape[:-2], y_diag.shape[:-1])
    yh_b = yh.broadcast_to(*bshape, n, n)
    y_diag_b = y_diag.broadcast_to(*bshape, n).contiguous()

    # Build a diagonal matrix [*bshape, N, N] from y_diag_b via a flattened index_add.
    diag_flat = torch.zeros((*bshape, n * n), dtype=cdt, device=device)
    diag_rows = torch.arange(n, device=device) * (n + 1)  # diagonal positions in N*N
    diag_flat = diag_flat.index_add(-1, diag_rows, y_diag_b)
    diag_mat = diag_flat.reshape(*bshape, n, n)
    return yh_b + diag_mat


__all__ = [
    "solve_harmonic_flow",
    "assemble_harmonic_system",
    "assemble_harmonic_ybus",
    "harmonic_injections",
    "HarmonicFlowResult",
    "NodeHarmonicSource",
]
