"""Sherman-Morrison-Woodbury update-solve on top of a cached factorization.

A switch-state sweep solves the SAME network under ``S`` per-branch admittance
scalings. Because a branch enters the Y-bus only through its primitive stamp on
its own terminal rows (``Y[rows, rows] += block``, see
:func:`pgml.assembly.branch_stamp_blocks`), scaling a ``P``-phase branch by ``s``
changes ``Y`` by ``(s − 1)`` times a rank-``≤ 2P`` term. Re-assembling and
re-factoring per state costs ``O(S·n³)``; with the Woodbury identity

.. math::
    (A + U C V^H)^{-1} b = A^{-1}b - A^{-1}U\\,(C^{-1} + V^H A^{-1} U)^{-1}
                            V^H A^{-1} b

the base factorization is built ONCE and each state costs an ``O(n²k + k³)``
correction (``k`` = the total rank of that state's switched branches), which wins
whenever ``k ≪ n``.

The implementation uses the algebraically equivalent arrangement

.. math::
    (A + U C V^H)^{-1} = A^{-1} - A^{-1}U\\,(I + C V^H A^{-1} U)^{-1} C V^H A^{-1}

which never inverts ``C``. That matters here: every state is reached exactly,
including the ``s → 0`` limit of an OPEN switch (``C = −\\text{block}`` is finite,
but ``C^{-1}`` need not exist — a series stamp ``[[y, −y], [−y, y]]`` is singular),
while a branch sitting AT its base state gives ``C = 0``, whose correction is
identically zero and whose result is bit-for-bit the base solve. The capacitance
matrix ``I + C V^H A^{-1}U`` is invertible exactly when the updated system is, so
the only failure mode is a state that islands part of the grid — which the solvers
reject up front with a connectivity check.

WHICH BASE (accuracy): the correction reads the base solution's entries, which
carry the usual ``~eps`` rounding, and amplifies them by
``‖(I + CZ)^{-1} CZ‖`` with ``Z = V^H A^{-1} U``. Adding admittance to the base
keeps that factor at ``O(1)``; REMOVING a near-ideal branch pushes it to
``|y · z_thevenin|``, because the quantity the downdate multiplies is that branch's
voltage drop — the difference of two nearly equal node voltages, which floating
point resolves poorly. Measured on a 20 kV feeder, opening a switch from a base
that HOLDS it costs ~1 digit per decade of switch conductance (a 1e-4 Ω contact
already spends 4, and the power flow then stops converging), while the same sweep
based on the network WITHOUT the switch stays at ``~1e-12`` relative. So build the
base from the LOWEST admittance each branch takes over the sweep wherever the grid
stays connected — what :func:`pgml.solver.power_flow._woodbury_base_states` does —
and heed the warning :func:`_amplification` raises otherwise (the nonlinear
solvers widen their convergence floors by the same factor, so a solve that keeps
fewer digits is not reported as non-converged).

Public API
----------
- ``low_rank_update(fac, u, c, *, v=None) -> LowRankUpdate`` — precompute
  ``A^{-1}U``, the capacitance matrix and its factorization for a
  :class:`~pgml.solver.harmonic.FactoredSystem`.
- ``solve_factored_updated(system, i_inj, *, u=None, c=None, v=None,
  v_fixed=None) -> Tensor`` — solve the UPDATED system for a new right-hand side.
- ``branch_state_terms(grid, index, branch_states, frequency_hz, ...) -> (u, c)``
  — build ``U`` / ``C`` for a batch of switch states from the assembly's own
  primitive stamps.
- ``LowRankOperator`` — the updated matrix as a matrix-free operator (applies
  ``A·x + U C V^H x`` without materialising the per-state ``[B, N, N]`` matrix).

Differentiability + GPU (CLAUDE.md): pure torch throughout, so gradients flow to
the factored matrix, to ``U`` / ``C`` / ``V``, to the right-hand side and to
``v_fixed``; device and dtype follow the inputs; every backend of
:func:`~pgml.solver.harmonic.lu_factor_system` (dense, sparse, block) is supported
through its own back-substitution.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor

from pgml.assembly import NodePhaseIndex, branch_stamp_blocks
from pgml.assembly._stamps import _cdtype, _rdtype
from pgml.errors import InputError
from pgml.schemas.grid_schema import Grid

from .harmonic import (
    FactoredSystem,
    back_substitute,
    ideal_slack_rhs,
    scatter_slack_solution,
)

_log = logging.getLogger("pgml")

# The correction ``W (I + C V^H W)^-1 C V^H v0`` multiplies quantities read off the
# base solution, which carry the usual ~eps relative rounding. ``(I + CZ)^-1 CZ``
# (Z = V^H A^-1 U) is the factor by which that rounding reaches the result, so it
# is the number of digits the update spends. It stays O(1) when the update ADDS
# admittance and grows like |y·z_thevenin| when it REMOVES a stiff one (a
# near-ideal closed switch), where the branch voltage drop the downdate multiplies
# is itself lost to cancellation. Warn above a million (six of float64's ~16
# digits) — the regime where a fixed point can stop converging.
_AMPLIFICATION_WARN = 1.0e6


def _as_terms(u: Tensor, c: Tensor, v: Optional[Tensor], n: int, ref: Tensor):
    """Validate / normalise ``(U, C, V)`` against a system of ``n`` rows."""
    if u.ndim != 2 or u.shape[0] != n:
        raise InputError(
            f"The low-rank factor u must be [N, k] with N={n} (the system's FULL row "
            f"count); got {tuple(u.shape)}."
        )
    k = u.shape[1]
    if c.ndim < 2 or c.shape[-2:] != (k, k):
        raise InputError(
            f"The low-rank core c must be [*batch, k, k] with k={k} (u's columns); "
            f"got {tuple(c.shape)}."
        )
    v = u if v is None else v
    if v.shape != u.shape:
        raise InputError(
            f"The low-rank factor v must have u's shape {tuple(u.shape)}; "
            f"got {tuple(v.shape)}."
        )
    dtype, device = ref.dtype, ref.device
    return (
        u.to(dtype=dtype, device=device),
        c.to(dtype=dtype, device=device),
        v.to(dtype=dtype, device=device),
        k,
    )


def _vh_x(v: Tensor, x: Tensor) -> Tensor:
    """``V^H x`` for ``V`` ``[m, k]`` and ``x`` ``[*batch, m]`` -> ``[*batch, k]``."""
    return torch.matmul(x.unsqueeze(-2), v.conj()).squeeze(-2)


@dataclass(frozen=True)
class LowRankOperator:
    """``A' = A + U C V^H`` as a matrix-free linear operator.

    Carries the per-state admittance of a switch-state sweep WITHOUT materialising
    the ``[B, N, N]`` tensor the per-state assembly would build: the base ``A`` is
    shared by every state and each state's deviation is the rank-``k`` term. Used
    wherever the solvers need ``Y·V`` (residuals, diagnostics) rather than a
    factorization.

    Attributes
    ----------
    base:
        Complex ``[*fb, N, N]`` base admittance (the state-independent matrix that
        was factored).
    u, v:
        Complex ``[N, k]`` low-rank factors in the FULL row space (``v`` defaults
        to ``u`` for the symmetric branch stamps).
    c:
        Complex ``[*ub, k, k]`` core; its leading dims are the state batch.
    """

    base: Tensor
    u: Tensor
    c: Tensor
    v: Tensor

    @property
    def shape(self) -> torch.Size:
        """Shape of the matrix this operator stands for, ``[*batch, N, N]``."""
        lead = torch.broadcast_shapes(self.base.shape[:-2], self.c.shape[:-2])
        n = self.base.shape[-1]
        return torch.Size((*lead, n, n))

    @property
    def dtype(self) -> torch.dtype:
        return self.base.dtype

    @property
    def device(self) -> torch.device:
        return self.base.device

    def correction(self, x: Tensor) -> Tensor:
        """The low-rank part ``U C V^H x`` for ``x`` ``[*batch, N]``."""
        cvx = torch.matmul(self.c, _vh_x(self.v, x).unsqueeze(-1))  # [*b, k, 1]
        return torch.matmul(self.u, cvx).squeeze(-1)  # [*b, N]


@dataclass(frozen=True)
class LowRankUpdate:
    """A factorization plus a low-rank modification of the matrix it factored.

    Built by :func:`low_rank_update` and consumed by
    :func:`solve_factored_updated`, which back-substitutes new right-hand sides
    against the UPDATED system ``A + U C V^H`` at ``O(n²k)`` per right-hand side
    (the base back-substitution) plus ``O(k²)`` for the correction — the ``k × k``
    capacitance matrix is factored ONCE here and reused by every solve, mirroring
    the factor-once-solve-many design of :class:`FactoredSystem` itself.

    Attributes
    ----------
    fac:
        The base :class:`~pgml.solver.harmonic.FactoredSystem`.
    u, v, c:
        The low-rank terms as passed in: ``u`` / ``v`` complex ``[N, k]`` in the
        FULL row space, ``c`` complex ``[*ub, k, k]``.
    w:
        ``A^{-1} U`` restricted to the factored space, complex ``[*fb, m, k]``
        (``m = N`` in Norton mode, the free-row count under ideal slack).
    k_lu, k_piv:
        LU factors of the capacitance matrix ``I + C V^H A^{-1} U``,
        ``[*kb, k, k]`` / ``[*kb, k]``.
    """

    fac: FactoredSystem
    u: Tensor
    v: Tensor
    c: Tensor
    w: Tensor
    k_lu: Tensor
    k_piv: Tensor
    u_free: Tensor  # u restricted to the factored rows [m, k]
    v_free: Tensor
    v_slack: Optional[Tensor]  # v on the fixed rows [S, k] (ideal slack only)
    amplification: float = 1.0  # by how much the update amplifies the base's rounding

    @property
    def backend(self) -> str:
        """Factorization backend of the base system (``"dense"``/``"sparse"``/``"block"``)."""
        return self.fac.backend

    @property
    def precision(self) -> str:
        """Working precision of the base factorization (``"full"`` / ``"mixed"``)."""
        return self.fac.precision

    @property
    def mode(self) -> str:
        """Slack mode of the base system (``"norton"`` / ``"ideal"``)."""
        return self.fac.mode

    @property
    def n(self) -> int:
        """Row count of the FULL system."""
        return self.fac.n

    @property
    def rank(self) -> int:
        """``k`` — the number of columns of the update."""
        return self.u.shape[1]

    def _corrected(self, v0: Tensor) -> Tensor:
        """Apply the Woodbury correction to a base solution ``v0`` ``[*batch, m]``."""
        ct = torch.matmul(self.c, _vh_x(self.v_free, v0).unsqueeze(-1))  # [*b, k, 1]
        x = torch.linalg.lu_solve(self.k_lu, self.k_piv, ct)  # [*b, k, 1]
        return v0 - torch.matmul(self.w, x).squeeze(-1)


def _amplification(k_lu: Tensor, k_piv: Tensor, cz: Tensor) -> float:
    """How much the correction amplifies the base solution's rounding; warn if extreme.

    The ∞-norm of ``(I + CZ)^{-1} CZ``, maximised over the batched updates, is the factor
    by which the base solution's rounding reaches the corrected result. It is returned so
    the nonlinear solvers can widen their convergence floors by it — a solve path that
    keeps fewer digits cannot meet a tolerance tighter than that — and a large factor is
    warned about. No autograd, no effect on the solved value.
    """
    with torch.no_grad():
        amp = torch.linalg.lu_solve(k_lu, k_piv, cz).abs().sum(-1).max()
        factor = float(amp)
    if factor > _AMPLIFICATION_WARN:
        _log.warning(
            "low_rank_update: the update amplifies the base solution's rounding by "
            "~%.1e, so the corrected solve keeps only ~%.0f of float64's ~16 digits "
            "(a solver tolerance below that will not be reached). This is the "
            "signature of REMOVING a near-ideal branch — an (almost) zero-impedance "
            "switch whose admittance dwarfs the network's. Give the switch a finite "
            "resistance, base the update on the network WITHOUT it, or assemble per "
            "state.",
            factor,
            max(0.0, 16.0 - math.log10(factor)),
        )
    return max(1.0, factor)


def low_rank_update(
    fac: FactoredSystem,
    u: Tensor,
    c: Tensor,
    *,
    v: Optional[Tensor] = None,
) -> LowRankUpdate:
    """Prepare the Woodbury solve of ``(A + U C V^H) x = b`` from ``A``'s factors.

    Precomputes the two quantities every subsequent right-hand side reuses:
    ``W = A^{-1}U`` (``k`` back-substitutions against the base factorization) and
    the LU of the capacitance matrix ``I + C V^H W``. With ``S`` states batched
    into ``c``'s leading dims, all ``S`` capacitance matrices are factored in one
    batched call.

    Parameters
    ----------
    fac:
        Base factorization from :func:`~pgml.solver.harmonic.lu_factor_system`
        (any backend). Under IDEAL slack it factors the free-row block ``Y_ff``;
        the update is reduced to that space here, and the modification of the
        slack coupling ``Y_fs`` is applied to the right-hand side at solve time —
        so ``u`` / ``v`` are always given in the FULL ``N``-row space.
    u:
        Complex ``[N, k]`` left factor (rows of the FULL system).
    c:
        Complex ``[*batch, k, k]`` core. Its leading dims are independent updates
        (e.g. one per switch state) and must broadcast against the
        factorization's own leading dims.
    v:
        Complex ``[N, k]`` right factor; defaults to ``u`` (the symmetric case
        every branch stamp produces, ``Y[rows, rows] += block``).

    Returns
    -------
    LowRankUpdate
        The cached update. Differentiable w.r.t. the factored matrix, ``u``,
        ``c`` and ``v``; device/dtype follow the factorization.
    """
    ref = fac._fb_tensor
    u, c, v, k = _as_terms(u, c, v, fac.n, ref)
    fb = tuple(ref.shape[:-2])

    if fac.mode == "ideal":
        u_free = u.index_select(0, fac.free_rows)
        v_free = v.index_select(0, fac.free_rows)
        v_slack = v.index_select(0, fac.fixed_rows)
    else:
        u_free, v_free, v_slack = u, v, None
    m = u_free.shape[0]

    # W = A^-1 U: the k columns are just k right-hand sides of the SAME
    # factorization (the leading column axis is a scenario dim for
    # ``back_substitute``, so no factor is tiled).
    cols = u_free.mT.reshape(k, *(1,) * len(fb), m)  # [k, *1, m]
    w = back_substitute(fac, cols).movedim(0, -1)  # [*fb, m, k]

    z = torch.matmul(v_free.conj().mT, w)  # V^H A^-1 U  [*fb, k, k]
    eye = torch.eye(k, dtype=w.dtype, device=w.device)
    cz = torch.matmul(c, z)
    k_lu, k_piv = torch.linalg.lu_factor(eye + cz)
    return LowRankUpdate(
        fac=fac,
        u=u,
        v=v,
        c=c,
        w=w,
        k_lu=k_lu,
        k_piv=k_piv,
        u_free=u_free,
        v_free=v_free,
        v_slack=v_slack,
        amplification=_amplification(k_lu, k_piv, cz),
    )


def solve_factored_updated(
    system,
    i_inj: Tensor,
    *,
    u: Optional[Tensor] = None,
    c: Optional[Tensor] = None,
    v: Optional[Tensor] = None,
    v_fixed: Optional[Tensor] = None,
) -> Tensor:
    """Solve ``(Y + U C V^H) V = I`` against a cached base factorization.

    Same contract as :func:`~pgml.solver.harmonic.solve_factored` — batched
    right-hand sides ``[*batch, N]``, both slack modes, the full ``[*batch, N]``
    voltage returned — but the system solved is the LOW-RANK MODIFICATION of the
    factored one. The result equals a fresh factorization of ``Y + U C V^H`` to
    round-off.

    Parameters
    ----------
    system:
        A prepared :class:`LowRankUpdate` (the ``k × k`` capacitance matrix is then
        factored once for all calls), or a
        :class:`~pgml.solver.harmonic.FactoredSystem` together with ``u`` / ``c``
        (``v`` optional) — the one-shot form, which prepares the update internally.
    i_inj:
        Complex right-hand side ``[*batch, N]`` (or ``[N]``), broadcast against the
        factorization's and the update's leading dims exactly as
        :func:`~pgml.solver.harmonic.solve_factored` does.
    u, c, v:
        The low-rank terms; give them only with a bare ``FactoredSystem``.
    v_fixed:
        Ideal-slack reference, complex broadcastable to ``[*batch, S]``. Required
        when the base factorization is in ideal-slack mode. The update's effect on
        the slack coupling (``ΔY_fs = U_f C V_s^H``) is applied to the right-hand
        side, so a switched branch incident to a slack node is handled exactly.

    Returns
    -------
    Tensor
        Complex ``[*batch, N]`` node voltages, where ``batch`` broadcasts the
        right-hand side, the factorization and the update's state dims.
    """
    if isinstance(system, LowRankUpdate):
        if u is not None or c is not None or v is not None:
            raise InputError(
                "solve_factored_updated: pass u/c/v with a FactoredSystem, or a "
                "prepared LowRankUpdate — not both."
            )
        upd = system
    elif isinstance(system, FactoredSystem):
        if u is None or c is None:
            raise InputError(
                "solve_factored_updated: a FactoredSystem needs the update terms "
                "u [N, k] and c [*batch, k, k] (v defaults to u)."
            )
        upd = low_rank_update(system, u, c, v=v)
    else:
        raise InputError(
            "solve_factored_updated: `system` must be a LowRankUpdate or a "
            f"FactoredSystem; got {type(system).__name__}."
        )

    fac = upd.fac
    if fac.mode == "norton":
        return upd._corrected(back_substitute(fac, i_inj))

    rhs, vf_b = ideal_slack_rhs(fac, i_inj, v_fixed)  # I_free - Y_fs v_fixed
    # The update also modifies the slack coupling: subtract (U_f C V_s^H) v_fixed.
    cvs = torch.matmul(upd.c, _vh_x(upd.v_slack, vf_b).unsqueeze(-1))  # [*b, k, 1]
    rhs = rhs - torch.matmul(upd.u_free, cvs).squeeze(-1)  # [*b, F]
    v_free = upd._corrected(back_substitute(fac, rhs))
    return scatter_slack_solution(fac, v_free, vf_b)


def branch_state_terms(
    grid: Grid,
    index: NodePhaseIndex,
    branch_states: dict,
    frequency_hz: float,
    *,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    param_overrides: Optional[dict] = None,
    base_states: Union[float, dict] = 1.0,
) -> tuple[Tensor, Tensor]:
    """``U`` and ``C`` such that ``Y(states) = Y(base_states) + U C U^H``.

    Each listed branch contributes its primitive stamp
    (:func:`pgml.assembly.branch_stamp_blocks` — the SAME block the assembly
    scatters, no stamp physics re-derived) on its own ``M`` terminal rows:
    ``U`` selects those rows (``[N, k]`` with ``k = Σ M_b``) and ``C`` is
    block-diagonal with ``(s_b − base_b)·block_b`` per branch. Because ``U`` is a
    selection matrix, ``U C U^H`` scatters exactly the difference between the
    state's stamp and the base's.

    The BASE the caller assembles and factors must use the same ``base_states``.
    Prefer a base that OMITS the switched branches (``base_states=0``) wherever the
    grid stays connected without them: the update then ADDS admittance, which is
    numerically benign, whereas removing a near-ideal switch multiplies a branch
    voltage drop that floating point resolves poorly (see
    :func:`_amplification`).

    This builder is the seam a second consumer reuses: any change that scales,
    adds or removes a branch of an already-factored grid — for example solving a
    MUTATED CHILD grid from its parent's factorization — is expressible as the
    same ``(U, C)`` pair over the child's differing branches (state 0 removes a
    parent branch; a branch absent from the parent enters as a state-1 column over
    a base state of 0).

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.schemas.grid_schema.Grid`.
    index:
        The compact :class:`~pgml.assembly.NodePhaseIndex` of ``grid``.
    branch_states:
        ``{branch_id: state}`` as :func:`pgml.solver.solve_power_flow` takes it —
        a python float, a 0-d tensor, or a ``[*batch]`` state batch. Batched states
        give ``C`` those leading dims (one update per scenario).
    frequency_hz:
        The single frequency the stamps are evaluated at (the fundamental for a
        power-flow sweep); loop orders externally for a harmonic sweep.
    dtype, device, param_overrides:
        As in :func:`pgml.assembly.assemble_ybus`.
    base_states:
        The state each branch carries in the BASE admittance: one float for all,
        or ``{branch_id: state}`` (missing ids default to 1.0).

    Returns
    -------
    (u, c):
        ``u`` complex ``[N, k]`` (real 0/1 selection, in the working complex dtype)
        and ``c`` complex ``[*batch, k, k]``. Differentiable w.r.t. the states and
        the branch parameters.
    """
    if not branch_states:
        raise InputError("branch_state_terms needs a non-empty branch_states map.")
    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)
    ids = list(branch_states.keys())
    stamps = branch_stamp_blocks(
        grid,
        [frequency_hz],
        ids,
        index,
        dtype=dtype,
        device=device,
        param_overrides=param_overrides,
    )
    dev = stamps[0].block.device
    n = index.size
    sizes = [int(s.rows.numel()) for s in stamps]
    k = sum(sizes)
    if k >= n:
        _log.warning(
            "branch_state_terms: the switched branches span k=%d rows of an N=%d-row "
            "system, so the update is no longer LOW rank — its O(k³) correction costs "
            "more than re-assembling. Switch fewer branches per sweep, or solve each "
            'state from its own assembly (branch_states_method="assemble").',
            k,
            n,
        )

    rows_cat = torch.cat([s.rows for s in stamps]).to(dev)  # [k]
    u = torch.nn.functional.one_hot(rows_cat, n).to(cdt).mT  # [N, k]

    c = torch.zeros((k, k), dtype=cdt, device=dev)
    offset = 0
    for stamp, size in zip(stamps, sizes):
        state = branch_states[stamp.branch_id]
        st = (
            state.to(dtype=rdt, device=dev)
            if isinstance(state, Tensor)
            else torch.as_tensor(float(state), dtype=rdt, device=dev)
        )
        base = (
            float(base_states.get(stamp.branch_id, 1.0))
            if isinstance(base_states, dict)
            else float(base_states)
        )
        delta = (st - base).to(cdt)  # [*batch]
        piece = delta[..., None, None] * stamp.block.select(0, 0)  # [*batch, M, M]
        pad = (offset, k - offset - size)
        c = c + torch.nn.functional.pad(piece, (*pad, *pad))
        offset += size
    return u, c


__all__ = [
    "LowRankOperator",
    "LowRankUpdate",
    "branch_state_terms",
    "low_rank_update",
    "solve_factored_updated",
]
