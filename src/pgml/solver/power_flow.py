"""Nonlinear fundamental power flow (const-P / ZIP) via current injection.

Public API
----------
- ``solve_power_flow(grid, *, slack, method, tol, max_iter, dtype, device,
  operating_point, param_overrides) -> PowerFlowResult``

Forward
-------
A current-injection FIXED POINT at the fundamental frequency ``f0 =
grid.base_frequency_hz``. With the passive-network admittance ``Y_net =
assemble_network_ybus`` (plus the source Norton shunt when ``slack="norton"``),
iterate

    V_{k+1} = solve_harmonic(Y_eff, I_slack - I_device(V_k), slack...)

until ``||V_{k+1} - V_k|| < tol`` (or ``max_iter``), where ``I_device`` is the
ZIP voltage-dependent device current from
:func:`pgml.assembly.device_current_injections`. The iteration runs under
``torch.no_grad()`` (CORRECT for implicit diff: the gradient is supplied
analytically, NOT by unrolling).

Backward (IMPLICIT FUNCTION THEOREM, REAL coordinates)
------------------------------------------------------
The power flow is non-holomorphic in ``V`` (``I_device`` uses ``|V|`` and
``conj(V)``), so the residual / Jacobian / adjoint are formed in REAL (re/im
split) coordinates. The converged ``V*`` satisfies the full-row real residual

    R(x, theta) = 0,   x = [Re(V); Im(V)]  (size 2N per system),

where free rows carry the complex power-balance residual ``Re/Im(Y_eff V +
I_device(V) - I_slack)`` and ideal-slack rows carry ``Re/Im(V - V_fixed)``
(pinned). For an output cotangent ``grad_V`` on ``V*`` the IFT gives:

    solve  J^T lambda = grad_x          (ONE real linear solve per system),
    grad_theta = -(dR/dtheta)^T lambda  (a vjp of a single residual eval at V*).

Implemented as a :class:`torch.autograd.Function`: ``forward`` returns the
no_grad ``V*``; ``backward`` builds the real ``[2N, 2N]`` Jacobian at ``V*``,
solves the adjoint, and runs ``torch.autograd.grad`` on a single residual
evaluation to produce gradients for every parameter leaf (network params, device
P/Q, slack voltage).

Differentiability + GPU (CLAUDE.md): the differentiable path is the IFT backward
(no unrolling). ``no_grad`` in the forward iteration is expected. No
``.item()/.detach()/.numpy()`` on the autograd tape, no in-place on tracked
tensors, no python control flow on tensor VALUES (the convergence test is a scalar
norm under ``no_grad``). Honors input device/dtype; runs unchanged on CPU/CUDA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from pgml.assembly import (
    NodePhaseIndex,
    assemble_network_ybus,
    build_injections,
    device_current_injections,
    node_phase_index,
)
from pgml.assembly._stamps import _cdtype, _rdtype
from pgml.assembly.ybus import _stamp_sources
from pgml.schemas.grid_schema import Grid, Source

from .harmonic import solve_harmonic


@dataclass(frozen=True)
class PowerFlowResult:
    """Converged fundamental power-flow solution.

    Attributes
    ----------
    v:
        Complex node voltages ``[*batch, N]`` (DIFFERENTIABLE through the IFT path).
    index:
        The compact :class:`NodePhaseIndex` describing ``v``'s row layout.
    iterations:
        Number of fixed-point iterations performed (python int).
    residual:
        Real scalar tensor: the final ``||V_{k+1} - V_k||`` (max over batch).
    converged:
        ``True`` if the residual fell below ``tol`` within ``max_iter``.
    """

    v: Tensor
    index: NodePhaseIndex
    iterations: int
    residual: Tensor
    converged: bool


# ---------------------------------------------------------------------------
# leaf discovery (tensor-duality + overrides + slack voltage)
# ---------------------------------------------------------------------------
def _collect_leaves(obj, out: list, seen: set) -> None:
    """Recursively collect distinct grad-requiring tensors reachable from ``obj``."""
    if isinstance(obj, Tensor):
        if obj.requires_grad and id(obj) not in seen:
            seen.add(id(obj))
            out.append(obj)
        return
    if isinstance(obj, (list, tuple)):
        for el in obj:
            _collect_leaves(el, out, seen)
        return
    if isinstance(obj, dict):
        for el in obj.values():
            _collect_leaves(el, out, seen)
        return
    fields = getattr(type(obj), "model_fields", None)
    if fields is not None:
        for name in fields:
            _collect_leaves(getattr(obj, name), out, seen)


def _grid_param_leaves(
    grid: Grid, param_overrides: Optional[dict], v_fixed: Optional[Tensor]
) -> list[Tensor]:
    """All distinct autograd leaves the residual depends on (deterministic order)."""
    out: list[Tensor] = []
    seen: set = set()
    for node in grid.nodes:
        _collect_leaves(node, out, seen)
    for b in grid.branches:
        _collect_leaves(b, out, seen)
    for a in grid.appliances:
        _collect_leaves(a, out, seen)
    if param_overrides is not None:
        for val in param_overrides.values():
            _collect_leaves(val, out, seen)
    if v_fixed is not None and isinstance(v_fixed, Tensor) and v_fixed.requires_grad:
        if id(v_fixed) not in seen:
            seen.add(id(v_fixed))
            out.append(v_fixed)
    return out


# ---------------------------------------------------------------------------
# slack handling
# ---------------------------------------------------------------------------
def _slack_rows_and_vref(
    grid: Grid, index: NodePhaseIndex, rdt: torch.dtype, cdt: torch.dtype, device
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Ideal-slack fixed rows + reference voltages ``u_ref∠u_angle`` at them.

    Returns ``(fixed_rows[int64, S], v_fixed[complex, S])`` from in-service
    sources, or ``(None, None)`` if there is no source. ``v_fixed`` stays
    differentiable when ``u_ref``/``u_angle`` are tensors (tensor duality).
    """
    sources = [a for a in grid.appliances if isinstance(a, Source) and a.in_service]
    if not sources:
        return None, None
    rows: list[int] = []
    vref: list[Tensor] = []
    for s in sources:
        u_ref = torch.as_tensor(s.u_ref_v, dtype=rdt, device=device)
        ang = torch.as_tensor(s.u_angle_deg, dtype=rdt, device=device) * (
            math.pi / 180.0
        )
        vth = torch.polar(u_ref, ang).to(cdt)  # [P]
        for j, ph in enumerate(s.phases):
            rows.append(index.row(s.node, ph))
            vref.append(vth[j])
    fixed_rows = torch.as_tensor(rows, dtype=torch.int64, device=device)
    v_fixed = torch.stack(vref, 0)  # [S]
    return fixed_rows, v_fixed


# ---------------------------------------------------------------------------
# system builders (used by both the forward fixed point and the IFT backward)
# ---------------------------------------------------------------------------
def _y_eff_and_islack(grid, f0, index, dtype, device, slack, param_overrides):
    """Effective admittance ``Y_eff`` ``[1,N,N]`` and slack current ``[1,N]`` or ``[N]``.

    ``slack="norton"``: ``Y_eff = Y_net + Y_srcNorton``, ``I_slack`` = source
    Norton current. ``slack="ideal"``: ``Y_eff = Y_net``, ``I_slack`` = 0 (slack
    rows pinned by the Schur solve in :func:`solve_harmonic`).
    """
    yb = assemble_network_ybus(
        grid, [f0], dtype=dtype, device=device, param_overrides=param_overrides
    )
    y = yb.Y  # [1, N, N]
    if slack == "norton":
        cdt = _cdtype(dtype)
        rdt = _rdtype(dtype)
        f = torch.as_tensor([f0], dtype=rdt, device=y.device)
        y = _stamp_sources(grid, f, y, index, cdt, rdt, y.device, param_overrides)
        i_slack = build_injections(
            grid,
            [f0],
            index,
            dtype=dtype,
            device=device,
            param_overrides=param_overrides,
        )  # [1, N]
    else:
        i_slack = torch.zeros(y.shape[-1], dtype=y.dtype, device=y.device)
    return y, i_slack


def solve_power_flow(
    grid: Grid,
    *,
    slack: str = "ideal",
    method: str = "current_injection",
    tol: float = 1e-8,
    max_iter: int = 100,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    operating_point: Optional[dict] = None,
    param_overrides: Optional[dict] = None,
) -> PowerFlowResult:
    """Solve the const-P / ZIP fundamental power flow (differentiable, batched).

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.schemas.grid_schema.Grid` (type_ref expanded).
    slack:
        ``"ideal"`` (default) fixes source-node V = ``u_ref∠u_angle`` exactly via
        the Schur path in :func:`solve_harmonic` (matches pandapower / pgm).
        ``"norton"`` folds the source as a Norton shunt (matches OpenDSS Vsource).
    method:
        ``"current_injection"`` (implemented). ``"newton"`` is a later add with the
        same interface.
    tol:
        Fixed-point convergence tolerance on ``||ΔV||`` (max over batch).
    max_iter:
        Maximum fixed-point iterations.
    dtype:
        Complex dtype (``complex128`` for gradcheck; ``complex64`` ok).
    device:
        Target device; defaults to the device of the first parameter leaf, else CPU.
    operating_point:
        Optional override of nameplate P/Q (see ``resolve_operating_power``).
    param_overrides:
        Optional differentiability hook (network R/L/C/Z, device P/Q) for gradcheck.

    Returns
    -------
    PowerFlowResult
        ``v`` complex ``[*batch, N]`` (DIFFERENTIABLE via the IFT), the index, the
        iteration count, the final update-norm residual, and convergence flag.
    """
    if method not in ("current_injection",):
        raise ValueError(f"Unsupported method {method!r} (only 'current_injection').")
    if slack not in ("ideal", "norton"):
        raise ValueError(f"Unsupported slack {slack!r} (use 'ideal' or 'norton').")

    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)

    index = node_phase_index(grid)
    n = index.size
    f0 = float(grid.base_frequency_hz)

    leaves = _grid_param_leaves(grid, param_overrides, None)
    if device is None:
        device = leaves[0].device if leaves else torch.device("cpu")

    # Slack rows are constant indices; the reference VOLTAGE is recomputed fresh
    # from the (possibly tensor) u_ref/u_angle on every residual eval so the
    # graph is not reused across gradcheck's multiple backward passes.
    def v_fixed_fn():
        if slack != "ideal":
            return None
        _, vf = _slack_rows_and_vref(grid, index, rdt, cdt, device)
        return vf

    fixed_rows = (
        _slack_rows_and_vref(grid, index, rdt, cdt, device)[0]
        if slack == "ideal"
        else None
    )
    v_fixed = v_fixed_fn()
    # v_fixed may be / contain a differentiable leaf (u_ref/u_angle as tensors).
    leaves = _grid_param_leaves(grid, param_overrides, v_fixed)

    # ----- closures over the CURRENT leaf values ----------------------------
    def build_system():
        return _y_eff_and_islack(grid, f0, index, dtype, device, slack, param_overrides)

    def residual_complex(v_cmplx: Tensor, y_eff: Tensor, i_slack: Tensor) -> Tensor:
        """F_c(V) = Y_eff @ V + I_device(V) - I_slack  (all rows, complex)."""
        i_dev = device_current_injections(
            grid,
            v_cmplx,
            index,
            [f0],
            dtype=dtype,
            device=device,
            operating_point=operating_point,
            param_overrides=param_overrides,
        ).squeeze(-2)  # [*b, N]
        yv = torch.matmul(y_eff, v_cmplx.unsqueeze(-1)).squeeze(-1)  # [*b, N]
        return yv + i_dev - i_slack

    real_res = _make_real_residual(
        build_system, residual_complex, fixed_rows, v_fixed_fn, n, cdt
    )

    # ----- forward: fixed point under no_grad -------------------------------
    with torch.no_grad():
        y_eff0, i_slack0 = build_system()
        lead = torch.broadcast_shapes(i_slack0.shape[:-1], y_eff0.shape[:-2])
        # Flat warm start: the source reference magnitude at every row (or 1∠0).
        if v_fixed is not None:
            v0_scalar = v_fixed.reshape(-1)[0]
        else:
            _, vr = _slack_rows_and_vref(grid, index, rdt, cdt, device)
            v0_scalar = (
                vr.reshape(-1)[0]
                if vr is not None
                else torch.ones((), dtype=cdt, device=device)
            )
        v = v0_scalar.to(cdt).reshape(()).expand(*lead, n).clone()

        residual_norm = torch.zeros((), dtype=rdt, device=device)
        iterations = 0
        converged = False
        for _ in range(max_iter):
            i_dev = device_current_injections(
                grid,
                v,
                index,
                [f0],
                dtype=dtype,
                device=device,
                operating_point=operating_point,
                param_overrides=param_overrides,
            ).squeeze(-2)  # [*b, N]
            rhs = i_slack0 - i_dev
            v_new = solve_harmonic(y_eff0, rhs, fixed_rows=fixed_rows, v_fixed=v_fixed)
            # solve_harmonic returns [*b, H=1, N]; drop the singleton H axis.
            if v_new.ndim >= 2 and v_new.shape[-2] == 1 and v_new.shape[-1] == n:
                v_new = v_new.squeeze(-2)
            delta = torch.linalg.vector_norm(v_new - v, dim=-1)
            residual_norm = delta.max()
            v = v_new
            iterations += 1
            if bool(residual_norm < tol):
                converged = True
                break

    v_star = v  # detached (built under no_grad)

    if leaves:
        v_out = _IFTPowerFlow.apply(v_star, real_res, n, rdt, cdt, *leaves)
    else:
        v_out = v_star

    return PowerFlowResult(
        v=v_out,
        index=index,
        iterations=iterations,
        residual=residual_norm.reshape(()),
        converged=bool(converged),
    )


# ---------------------------------------------------------------------------
# real-coordinate residual + IFT autograd.Function
# ---------------------------------------------------------------------------
def _make_real_residual(
    build_system,
    residual_complex,
    fixed_rows: Optional[Tensor],
    v_fixed_fn,
    n: int,
    cdt: torch.dtype,
):
    """Return a closure ``R(x) -> [*b, 2N]`` real residual with slack pinning.

    ``x = [Re(V); Im(V)]``. Free rows hold ``Re/Im(F_c)``; ideal-slack rows hold
    ``Re/Im(V - V_fixed)``. The complex system tensors come from ``build_system``
    and the slack reference from ``v_fixed_fn`` (recomputed each call) so they
    stay differentiable w.r.t. the parameter leaves and do not reuse a freed graph.
    """

    def _pin_and_split(fc: Tensor, v: Tensor, x: Tensor) -> Tensor:
        v_fixed = v_fixed_fn() if v_fixed_fn is not None else None
        if fixed_rows is not None and v_fixed is not None:
            vf = v_fixed.to(dtype=cdt, device=x.device)  # [S]
            v_at_fixed = v.index_select(-1, fixed_rows)  # [*b, S]
            pin = v_at_fixed - vf  # [*b, S]
            lead = torch.broadcast_shapes(fc.shape[:-1], pin.shape[:-1])
            s = fixed_rows.shape[0]
            fc_b = fc.broadcast_to(*lead, n)
            idx = fixed_rows.expand(*lead, s)
            fc = fc_b.scatter(-1, idx, pin.broadcast_to(*lead, s))
        return torch.cat([fc.real, fc.imag], dim=-1)  # [*b, 2N]

    def real_residual(x: Tensor) -> Tensor:
        """Full real residual; rebuilds the differentiable system from leaves."""
        v_re = x[..., :n]
        v_im = x[..., n:]
        v = torch.complex(v_re, v_im).to(cdt)  # [*b, N]
        y_eff, i_slack = build_system()
        fc = residual_complex(v, y_eff, i_slack)  # [*b, N] complex (all rows)
        return _pin_and_split(fc, v, x)

    def state_residual(
        x: Tensor, y_re: Tensor, y_im: Tensor, islack_re: Tensor, islack_im: Tensor
    ) -> Tensor:
        """Real residual at FIXED (real-split) system tensors — for the state Jacobian.

        ``jacrev`` rejects complex inputs, so the (constant) system tensors are
        passed as real/imag pairs and recombined here. Only ``x`` is differentiated.
        """
        v_re = x[..., :n]
        v_im = x[..., n:]
        v = torch.complex(v_re, v_im).to(cdt)
        y_eff = torch.complex(y_re, y_im).to(cdt)
        i_slack = torch.complex(islack_re, islack_im).to(cdt)
        fc = residual_complex(v, y_eff, i_slack)
        return _pin_and_split(fc, v, x)

    real_residual.state_residual = state_residual
    real_residual.build_system = build_system
    return real_residual


class _IFTPowerFlow(torch.autograd.Function):
    """Attach the IFT gradient to a detached converged ``V*``.

    ``forward`` returns ``V*`` unchanged. ``backward`` builds the real
    ``[2N, 2N]`` Jacobian of the real residual at ``V*``, solves the adjoint
    ``J^T λ = grad_x`` (one solve per batch system), then forms the parameter
    gradients ``-(dR/dθ)^T λ`` via a single vjp of the residual at ``V*``.
    """

    @staticmethod
    def forward(ctx, v_star, real_res, n, rdt, cdt, *leaves):
        ctx.real_res = real_res
        ctx.n = n
        ctx.rdt = rdt
        ctx.cdt = cdt
        ctx.num_leaves = len(leaves)
        ctx.save_for_backward(v_star, *leaves)
        return v_star

    @staticmethod
    def backward(ctx, grad_v):
        saved = ctx.saved_tensors
        v_star = saved[0]
        leaves = saved[1 : 1 + ctx.num_leaves]
        n = ctx.n
        rdt = ctx.rdt
        real_res = ctx.real_res
        state_residual = real_res.state_residual
        build_system = real_res.build_system
        twon = 2 * n

        # x* = [Re(V*); Im(V*)] -> [*b, 2N]
        x_star = torch.cat([v_star.real, v_star.imag], dim=-1).to(rdt)
        grad_x = torch.cat([grad_v.real, grad_v.imag], dim=-1).to(rdt)  # [*b, 2N]
        lead = x_star.shape[:-1]
        b = int(torch.tensor(lead).prod().item()) if lead else 1

        # Detached, per-batch system tensors for the STATE Jacobian (J = dR/dx).
        with torch.no_grad():
            y_eff, i_slack = build_system()  # [1,N,N] (or [*b,N,N]); [N] or [*b,N]
        y_eff = y_eff.detach()
        i_slack = i_slack.detach()
        x_flat = x_star.reshape(b, twon)
        gx_flat = grad_x.reshape(b, twon)
        y_flat = y_eff.reshape(-1, n, n)
        y_flat = y_flat.expand(b, n, n) if y_flat.shape[0] == 1 else y_flat
        islack_flat = i_slack.reshape(-1, n)
        islack_flat = (
            islack_flat.expand(b, n) if islack_flat.shape[0] == 1 else islack_flat
        )

        # Real state Jacobian J = dR/dx at x*, per batch system. The batched
        # residual R[b] depends only on x[b], so the full jacobian is block
        # diagonal; differentiate the batched map and take the per-batch diagonal
        # blocks. (Avoids vmap, which does not compose with the assembly's
        # in-place index_add_ scatter.)
        def batched_state_res(xb):
            return state_residual(
                xb,
                y_flat.real,
                y_flat.imag,
                islack_flat.real,
                islack_flat.imag,
            )  # [B, 2N]

        jac_full = torch.autograd.functional.jacobian(
            batched_state_res, x_flat, create_graph=False, vectorize=True
        )  # [B, 2N, B, 2N]
        idx_b = torch.arange(b, device=x_flat.device)
        j_batched = jac_full[idx_b, :, idx_b, :]  # [B, 2N, 2N]
        # Adjoint: J^T λ = grad_x  ->  λ = J^{-T} grad_x  (batched solve).
        lam_flat = torch.linalg.solve(
            j_batched.transpose(-1, -2), gx_flat.unsqueeze(-1)
        ).squeeze(-1)  # [B, 2N]
        lam = lam_flat.reshape(*lead, twon) if lead else lam_flat.reshape(twon)

        # grad_theta = -(dR/dθ)^T λ via a single residual vjp at x* (θ tracking).
        x_const = x_star.detach()
        with torch.enable_grad():
            r_theta = real_res(x_const)  # [*b', 2N]; depends on leaves
            # r_theta may carry a broadcast singleton batch dim; align λ to it.
            grad_out = (-lam).reshape(r_theta.shape).to(r_theta.dtype)
            grads = torch.autograd.grad(
                r_theta,
                leaves,
                grad_outputs=grad_out,
                retain_graph=False,
                allow_unused=True,
            )

        grad_leaves = tuple(
            g if g is not None else torch.zeros_like(leaf)
            for g, leaf in zip(grads, leaves)
        )
        # forward inputs: (v_star, real_res, n, rdt, cdt, *leaves)
        return (None, None, None, None, None, *grad_leaves)


__all__ = ["solve_power_flow", "PowerFlowResult"]
