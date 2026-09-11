"""Voltage-regulating generator terminals (PV buses) in the nonlinear power flow.

A :class:`~pgml.schemas.grid_schema.Generator` carrying a
:class:`~pgml.schemas.grid_schema.VoltageRegulation` block is a PV terminal: its
active power is given, its terminal voltage MAGNITUDE is held at the setpoint, and
its reactive power is whatever that takes (bounded by the block's reactive limits).

The residual substitution
-------------------------
The solver's real residual is ``R(x) = [Re(F_c); Im(F_c)]`` with the complex nodal
current mismatch ``F_c = Y_eff V + I_device(V) - I_slack`` (``solver/power_flow.py``).
At a regulating terminal's row ``r`` the two real equations are replaced by the
POWER-form pair, scaled by the row's nominal voltage ``v0`` so both stay in amperes
like every other free row::

    g_r = conj(V_r) * F_c,r / v0        (Re = active balance, Im = reactive balance)

    row 2r   (real part):  Re(g_r)                       — active power balance
    row 2r+N (imag part):  (|V_reg|**2 - V_set**2) / (2 V_set)    — the setpoint

The generator's reactive power enters ``F_c,r`` ONLY through the imaginary part of
``g_r``: its reactive current is ``-conj(j Q_e)/conj(V_r)``, and ``conj(V_r)`` times
that is ``+j Q_e`` — purely imaginary. Replacing the imaginary row therefore frees Q
exactly, and the active-balance row is untouched by it. This is the rectangular-
coordinate form of the textbook PV bus (drop the reactive mismatch, add
``|V| = V_set``), with Q eliminated analytically instead of carried as an extra
unknown: the state stays ``[Re V; Im V]``, so the ``[2N, 2N]`` IFT Jacobian and the
adjoint are unchanged, and the reactive injection is recovered from the converged
solution as ``Q_e = Q_e,pinned - Im(conj(V_r) F_c,r)``.

The replacement row is quadratic in ``V`` (no ``sqrt``, no ``abs``), hence smooth and
gradcheck-clean, and its Jacobian entries are O(1) like the ideal-slack pinning rows.

Regulated quantity
------------------
``RegulatedQuantity.POSITIVE_SEQUENCE`` (the default, balanced regulation) holds the
positive-sequence magnitude ``|V1|`` of the terminal. A three-phase unit then has ONE
free reactive power split equally over its phases, so the imaginary rows of its
phases 2 and 3 carry the equal-split conditions ``Im(g_2) - Im(g_1) = 0`` and
``Im(g_3) - Im(g_1) = 0`` (both reactive-power-free) and phase 1 carries the setpoint
row. ``RegulatedQuantity.PER_PHASE`` holds each phase magnitude at the setpoint with
its own free reactive power (independent single-phase regulators); the reactive
CAPABILITY stays a machine total.

Reactive limits: PV-to-PQ switching
-----------------------------------
Limits are enforced the standard way, by switching the terminal's bus type between
rounds of the solve: a unit whose required reactive power leaves ``[q_min, q_max]``
is re-solved as a plain PQ injection pinned at the violated limit, and is released
back to regulation once its terminal voltage crosses the setpoint from the other
side. A small hysteresis band (``pgml.defaults``
``appliance.generator.q_limit_hysteresis_*``) keeps solver noise from cycling the
decision. The switching DECISION is off-tape (a comparison of converged values); the
residual at the resolved active set is on-tape, so the IFT adjoint is exact for the
solved configuration: at a regulating terminal the gradient flows through
``v_set_pu``, at a pinned one through the binding limit. The alternative, a smooth
complementarity / saturation formulation, would blur the limit itself (the solution
would satisfy ``Q = q_max`` only to the smoothing width, which is the quantity a
cross-tool comparison checks) while buying only differentiability of a
measure-zero switching boundary.

Every tensor here follows the engine's device/dtype rules (real dtype ``rdt``,
complex ``cdt``, batched over leading scenario dims) and the setpoint / limits keep
the float-tensor duality, so ``dV/dv_set`` and ``dV/dq_limit`` flow through the IFT.
"""

from __future__ import annotations

import cmath
import logging
import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import torch
from torch import Tensor

from pgml import defaults
from pgml.assembly._params import phase_voltage_magnitude
from pgml.assembly._symmetry import resolve_connection
from pgml.errors import InputError, ModelingError
from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    Phase,
    RegulatedQuantity,
    Source,
    WindingConnection,
)

_log = logging.getLogger("pgml")

#: Phase index in the symmetric-component rotation (A -> 1, B -> a, C -> a**2).
_SEQ_INDEX = {Phase.A: 0, Phase.B: 1, Phase.C: 2}


@dataclass(frozen=True)
class PVGroup:
    """Regulating generators sharing an element count and a regulated quantity.

    ``rows[k, e]`` is the global node-phase row of element ``e`` of generator ``k``
    (WYE return to ground, so element and row coincide). ``state`` is the active set:
    ``0`` regulating, ``+1`` pinned at ``q_max``, ``-1`` pinned at ``q_min``; it is a
    decision variable, never on the autograd tape. ``v_set_pu``, ``q_min`` and
    ``q_max`` may carry leading scenario-batch dims and stay differentiable.
    """

    gen_ids: tuple[int, ...]
    rows: Tensor  # [K, P] int64
    flat_rows: Tensor  # [K * P] int64
    v0: Tensor  # [K] real line-to-neutral base voltage
    v_set_pu: Tensor  # [*batch, K] real
    q_min: Tensor  # [*batch, K] real (-inf = unbounded)
    q_max: Tensor  # [*batch, K] real (+inf = unbounded)
    per_phase: bool
    seq: Optional[Tensor]  # [P] complex positive-sequence weights (None if P == 1)
    state: Tensor  # [*batch, K] int8

    @property
    def n_elem(self) -> int:
        return int(self.rows.shape[1])


@dataclass(frozen=True)
class PVTerminals:
    """All regulating terminals of one solve (grouped), plus the limit policy."""

    groups: tuple[PVGroup, ...]
    enforce_q_limits: bool
    hysteresis_pu: float
    hysteresis_rel: float
    max_rounds: int

    @property
    def rows(self) -> Tensor:
        """Every regulated node-phase row, concatenated (int64 ``[R]``)."""
        return torch.cat([g.flat_rows for g in self.groups])

    @property
    def n_terminals(self) -> int:
        return sum(len(g.gen_ids) for g in self.groups)

    # -- residual -------------------------------------------------------------
    def transform(self, fc: Tensor, v: Tensor) -> Tensor:
        """Substitute the PV rows into the complex nodal residual ``fc``.

        ``fc`` / ``v`` are complex ``[*batch, N]`` (broadcastable against each other
        and against the group batch). Returns a new ``[*lead, N]`` residual whose
        regulated rows carry ``Re(g) + j*(setpoint row)`` and whose pinned rows carry
        the power-form mismatch ``g`` (an invertible row scaling of the original
        current mismatch, so a pinned terminal solves the ordinary PQ equations).
        """
        for g in self.groups:
            fc = _transform_group(g, fc, v)
        return fc

    # -- operating point ------------------------------------------------------
    def pinned_operating_point(self, operating_point: Optional[dict]) -> dict:
        """``operating_point`` with each regulating generator's ``q_var`` pinned.

        A regulating terminal's reactive power is free, so the value written here is
        irrelevant to its rows (0 is used); a terminal pinned at a reactive limit is
        a plain PQ injection and the limit VALUE enters the residual through this
        entry — differentiably, which is how ``dV/dq_limit`` flows.
        """
        op: dict = dict(operating_point) if operating_point else {}
        for g in self.groups:
            q_pin = _pinned_q(g)  # [*batch, K]
            for ki, gid in enumerate(g.gen_ids):
                entry = dict(op.get(gid, {}))
                entry.pop("q_per_phase_var", None)
                entry["q_var"] = q_pin[..., ki]
                op[gid] = entry
        return op

    # -- solution readout / switching -----------------------------------------
    def required_q(self, fc: Tensor, v: Tensor) -> dict[int, Tensor]:
        """Solved reactive injection per generator id (total over its phases, var)."""
        out: dict[int, Tensor] = {}
        for g in self.groups:
            q = _required_q(g, fc, v)  # [*batch, K]
            for ki, gid in enumerate(g.gen_ids):
                out[gid] = q[..., ki]
        return out

    def regulating_mask(self) -> dict[int, Tensor]:
        """Per generator id: ``True`` where the terminal holds its voltage setpoint."""
        out: dict[int, Tensor] = {}
        for g in self.groups:
            reg = g.state == 0
            for ki, gid in enumerate(g.gen_ids):
                out[gid] = reg[..., ki]
        return out

    def switch(self, fc: Tensor, v: Tensor) -> tuple["PVTerminals", bool]:
        """One PV-to-PQ switching round at the converged residual (off-tape).

        Returns the updated terminals and whether any terminal's bus type changed.
        """
        changed = False
        groups = []
        for g in self.groups:
            new_state = _switch_group(g, fc, v, self.hysteresis_pu, self.hysteresis_rel)
            if bool((new_state != g.state.broadcast_to(new_state.shape)).any()):
                changed = True
            groups.append(replace(g, state=new_state))
        return replace(self, groups=tuple(groups)), changed

    def setpoint_volts(self) -> tuple[Tensor, Tensor]:
        """``(rows [R] int64, v_set [*batch, R])`` in VOLTS over every regulated row.

        A balanced warm start places each regulated phase row at the setpoint
        magnitude (for positive-sequence regulation the balanced phase magnitude IS
        ``|V1|``), which puts the Newton seed on the regulation constraint.
        """
        rows = []
        volts = []
        for g in self.groups:
            k, p = g.rows.shape
            rows.append(g.flat_rows)
            vset = g.v_set_pu * g.v0.to(
                dtype=g.v_set_pu.dtype, device=g.v_set_pu.device
            )
            volts.append(
                vset.unsqueeze(-1)
                .expand(*vset.shape, p)
                .reshape(*vset.shape[:-1], k * p)
            )
        if len(volts) > 1:
            lead = torch.broadcast_shapes(*[t.shape[:-1] for t in volts])
            volts = [t.broadcast_to(*lead, t.shape[-1]) for t in volts]
        return torch.cat(rows), torch.cat(volts, dim=-1)

    # -- batch plumbing -------------------------------------------------------
    def slice(self, i: int) -> "PVTerminals":
        """The single-scenario terminals at batch index ``i`` (grad-preserving)."""
        return replace(self, groups=tuple(_slice_group(g, i) for g in self.groups))

    def flatten_batch(self, batch_shape: Sequence[int]) -> "PVTerminals":
        """Leading scenario dims collapsed to one, matching a flattened residual.

        The IFT backward evaluates the residual over a single flattened scenario
        axis; a setpoint / limit batch that matches ``batch_shape`` is reshaped the
        same way (broadcast singletons are left alone), mirroring
        :func:`pgml.assembly.flatten_plan_batch`.
        """
        bshape = tuple(int(s) for s in batch_shape)
        if len(bshape) < 2:
            return self
        flat = math.prod(bshape)

        def _f(t: Tensor) -> Tensor:
            lead = tuple(t.shape[:-1])
            if lead == bshape:
                return t.reshape(flat, t.shape[-1])
            return t

        return replace(
            self,
            groups=tuple(
                replace(
                    g,
                    v_set_pu=_f(g.v_set_pu),
                    q_min=_f(g.q_min),
                    q_max=_f(g.q_max),
                    state=_f(g.state),
                )
                for g in self.groups
            ),
        )

    def describe_state(self) -> str:
        """One-line summary of the active set (for logging)."""
        n_reg = 0
        n_hi = 0
        n_lo = 0
        for g in self.groups:
            n_reg += int((g.state == 0).sum())
            n_hi += int((g.state > 0).sum())
            n_lo += int((g.state < 0).sum())
        return f"{n_reg} regulating, {n_hi} at q_max, {n_lo} at q_min"


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------
def collect_pv_terminals(
    grid: Grid,
    index,
    rdt: torch.dtype,
    cdt: torch.dtype,
    device,
    operating_point: Optional[dict] = None,
    *,
    enforce_q_limits: Optional[bool] = None,
) -> Optional[PVTerminals]:
    """Build the :class:`PVTerminals` of ``grid`` (``None`` if it has no PV terminal).

    Reads every in-service :class:`~pgml.schemas.grid_schema.Generator` with a
    ``voltage_regulation`` block. A per-scenario setpoint override is taken from
    ``operating_point[gen_id]["v_set_pu"]`` (a float or ``[*batch]`` tensor, on the
    same per-unit base as the schema field). ``enforce_q_limits=None`` resolves from
    ``pgml.defaults`` (``appliance.generator.enforce_q_limits``).

    Raises :class:`~pgml.errors.ModelingError` for a regulating terminal the
    residual substitution cannot express: a DELTA connection, a WYE connection
    returning through the node's neutral row, a positive-sequence setpoint on a
    2-phase terminal, or a terminal on an ideal-slack source node.
    """
    regs = [
        a
        for a in grid.appliances
        if isinstance(a, Generator)
        and a.in_service
        and getattr(a, "voltage_regulation", None) is not None
    ]
    _check_unconsumed_overrides(grid, operating_point, regs)
    if not regs:
        return None

    node_map = {nd.id: nd for nd in grid.nodes}
    slack_nodes = {
        a.node for a in grid.appliances if isinstance(a, Source) and a.in_service
    }
    by_key: dict[tuple, list] = {}
    for a in regs:
        _validate_terminal(a, node_map[a.node], slack_nodes)
        by_key.setdefault((len(a.phases), a.voltage_regulation.regulated), []).append(a)

    groups: list[PVGroup] = []
    for (n_ph, regulated), gens in by_key.items():
        rows = torch.as_tensor(
            [[index.row(a.node, ph) for ph in a.phases] for a in gens],
            dtype=torch.int64,
            device=device,
        )  # [K, P]
        v0 = torch.stack(
            [
                _as_real(
                    phase_voltage_magnitude(
                        node_map[a.node].u_rated_v, len(node_map[a.node].phases)
                    ),
                    rdt,
                    device,
                )
                for a in gens
            ]
        )  # [K]
        v_set = _stack_batched(
            [_setpoint(a, operating_point, rdt, device) for a in gens]
        )
        q_min = _stack_batched(
            [
                _limit(a.voltage_regulation.q_min_var, -math.inf, rdt, device)
                for a in gens
            ]
        )
        q_max = _stack_batched(
            [
                _limit(a.voltage_regulation.q_max_var, math.inf, rdt, device)
                for a in gens
            ]
        )
        per_phase = regulated == RegulatedQuantity.PER_PHASE
        seq = None
        if not per_phase and n_ph > 1:
            a0 = cmath.exp(2j * math.pi / 3.0)
            seq = torch.as_tensor(
                [a0 ** _SEQ_INDEX[ph] for ph in gens[0].phases],
                dtype=cdt,
                device=device,
            )  # [P]
        state = torch.zeros(len(gens), dtype=torch.int8, device=device)
        groups.append(
            PVGroup(
                gen_ids=tuple(int(a.id) for a in gens),
                rows=rows,
                flat_rows=rows.reshape(-1),
                v0=v0,
                v_set_pu=v_set,
                q_min=q_min,
                q_max=q_max,
                per_phase=per_phase,
                seq=seq,
                state=state,
            )
        )

    enforce = (
        bool(defaults.get("appliance.generator.enforce_q_limits"))
        if enforce_q_limits is None
        else bool(enforce_q_limits)
    )
    return PVTerminals(
        groups=tuple(groups),
        enforce_q_limits=enforce,
        hysteresis_pu=float(defaults.get("appliance.generator.q_limit_hysteresis_pu")),
        hysteresis_rel=float(
            defaults.get("appliance.generator.q_limit_hysteresis_rel")
        ),
        max_rounds=int(defaults.get("appliance.generator.q_limit_switch_rounds_max")),
    )


def _validate_terminal(gen, node, slack_nodes: set) -> None:
    """Refuse a regulating terminal whose residual row the substitution cannot form."""
    conn = resolve_connection(gen)
    if conn == WindingConnection.DELTA:
        raise ModelingError(
            f"generator {gen.id}: voltage_regulation is modelled for a WYE terminal "
            "only. A DELTA element's reactive current is shared between two node rows, "
            "so its reactive degrees of freedom are not separable into one row pair "
            "(only the circulating total is observable). Connect the regulating unit "
            "in WYE."
        )
    return_path = getattr(gen, "return_path", "auto")
    has_neutral = Phase.N in node.phases and return_path != "ground"
    if has_neutral:
        raise ModelingError(
            f"generator {gen.id}: voltage_regulation on node {node.id} would return "
            "through the node's neutral row, whose reactive balance the setpoint row "
            "does not free. Set return_path='ground' (a grounded-wye regulating unit) "
            "or regulate on a node without a neutral conductor."
        )
    if len(gen.phases) == 2 and (
        gen.voltage_regulation.regulated == RegulatedQuantity.POSITIVE_SEQUENCE
    ):
        raise ModelingError(
            f"generator {gen.id}: a positive-sequence setpoint needs 1 or 3 phases "
            f"(got {len(gen.phases)}). Use regulated='per_phase'."
        )
    if gen.node in slack_nodes:
        raise ModelingError(
            f"generator {gen.id}: voltage_regulation on node {gen.node} conflicts with "
            "the in-service Source there — the slack already fixes that node's voltage "
            "phasor. Move the regulating unit, or model the source as the only "
            "reference (pandapower's own solve absorbs a generator at the reference "
            "bus into the slack dispatch)."
        )


def _check_unconsumed_overrides(grid: Grid, operating_point, regs) -> None:
    """Fail loud on operating-point keys a regulating terminal makes meaningless."""
    if not operating_point:
        return
    reg_ids = {int(a.id) for a in regs}
    for cid, entry in operating_point.items():
        if not isinstance(entry, dict):
            continue
        if "v_set_pu" in entry and cid not in reg_ids:
            raise InputError(
                f"operating_point[{cid!r}] carries 'v_set_pu', but that appliance is "
                "not an in-service generator with a voltage_regulation block (the "
                "setpoint would be ignored)."
            )
        if cid in reg_ids and ("q_var" in entry or "q_per_phase_var" in entry):
            _log.warning(
                "solve_power_flow: operating_point[%r] sets a reactive power on a "
                "VOLTAGE-REGULATING generator; it is ignored — the reactive power is "
                "solved from the voltage setpoint (bounded by q_min_var/q_max_var).",
                cid,
            )


def _as_real(value, rdt: torch.dtype, device) -> Tensor:
    """A 0-d real tensor from a float or a (possibly differentiable) tensor field.

    A python float is read AT the working precision (never through float32, which
    would quantise a setpoint like 1.02 to seven digits).
    """
    t = (
        value
        if isinstance(value, Tensor)
        else torch.as_tensor(value, dtype=rdt, device=device)
    )
    return t.to(dtype=rdt, device=device).reshape(())


def _setpoint(gen, operating_point, rdt: torch.dtype, device) -> Tensor:
    """Per-scenario voltage setpoint ``[*batch]`` (pu) of one regulating generator."""
    value = gen.voltage_regulation.v_set_pu
    if operating_point is not None:
        entry = operating_point.get(gen.id)
        if isinstance(entry, dict) and "v_set_pu" in entry:
            value = entry["v_set_pu"]
    return (
        _as_real(value, rdt, device)
        if not isinstance(value, Tensor)
        else value.to(dtype=rdt, device=device)
    )


def _limit(value, unbounded: float, rdt: torch.dtype, device) -> Tensor:
    """Reactive limit ``[*batch]`` (var); ``None`` becomes the unbounded sentinel."""
    if value is None:
        return torch.as_tensor(unbounded, dtype=rdt, device=device)
    if isinstance(value, Tensor):
        return value.to(dtype=rdt, device=device)
    return _as_real(value, rdt, device)


def _stack_batched(values: list[Tensor]) -> Tensor:
    """Stack per-generator ``[*batch]`` tensors into ``[*batch, K]`` (broadcasting)."""
    if len(values) > 1:
        values = list(torch.broadcast_tensors(*values))
    return torch.stack(values, dim=-1)


# ---------------------------------------------------------------------------
# per-group tensor math
# ---------------------------------------------------------------------------
def _lead(g: PVGroup, *tensors: Tensor) -> tuple[int, ...]:
    shapes = [tuple(t.shape[:-1]) for t in tensors]
    shapes += [
        tuple(g.v_set_pu.shape[:-1]),
        tuple(g.q_min.shape[:-1]),
        tuple(g.q_max.shape[:-1]),
        tuple(g.state.shape[:-1]),
    ]
    return tuple(torch.broadcast_shapes(*shapes))


def _gather(g: PVGroup, t: Tensor, lead: tuple[int, ...]) -> Tensor:
    """Gather ``t`` ``[*batch, N]`` at the group's rows -> ``[*lead, K, P]``."""
    n = t.shape[-1]
    k, p = g.rows.shape
    tb = t.broadcast_to(*lead, n)
    return tb.index_select(-1, g.flat_rows).reshape(*lead, k, p)


def _power_form(g: PVGroup, fc: Tensor, v: Tensor, lead) -> tuple[Tensor, Tensor]:
    """``(v_term, g_r)`` with ``g_r = conj(V) * F_c / v0`` ``[*lead, K, P]`` (amperes)."""
    v_t = _gather(g, v, lead)
    f_t = _gather(g, fc, lead)
    v0 = g.v0.to(dtype=v_t.real.dtype, device=v_t.device).reshape(-1, 1)  # [K, 1]
    return v_t, torch.conj(v_t) * f_t / v0


def _regulated_voltage(g: PVGroup, v_t: Tensor) -> Tensor:
    """Regulated magnitude ``[*lead, K]`` (positive sequence) from ``v_t``."""
    if g.seq is None:
        return v_t[..., 0].abs()
    seq = g.seq.to(dtype=v_t.dtype, device=v_t.device)
    return (v_t * seq).sum(-1).abs() / v_t.shape[-1]


def _setpoint_volts(g: PVGroup, rdt: torch.dtype, device) -> Tensor:
    return g.v_set_pu * g.v0.to(dtype=rdt, device=device)


def _transform_group(g: PVGroup, fc: Tensor, v: Tensor) -> Tensor:
    """Replace the group's residual rows with the power-form / setpoint pair."""
    lead = _lead(g, fc, v)
    k, p = g.rows.shape
    n = fc.shape[-1]
    v_t, gr = _power_form(g, fc, v, lead)
    rdt = v_t.real.dtype
    vset = _setpoint_volts(g, rdt, v_t.device)  # [*b, K]
    if g.per_phase:
        # Each phase holds its own magnitude; every imaginary row is a setpoint row.
        im_reg = (v_t.abs() ** 2 - (vset**2).unsqueeze(-1)) / (2.0 * vset).unsqueeze(-1)
    else:
        # One reactive power, split equally over the phases: the first phase carries
        # the setpoint row, the others the (reactive-power-free) equal-split rows.
        err = (_regulated_voltage(g, v_t) ** 2 - vset**2) / (2.0 * vset)  # [*b, K]
        im = gr.imag
        im_reg = err.unsqueeze(-1)
        if p > 1:
            im_reg = torch.cat(
                [im_reg.broadcast_to(*im.shape[:-1], 1), im[..., 1:] - im[..., :1]], -1
            )
    reg = (g.state == 0).unsqueeze(-1)  # [*b, K, 1]
    new_t = torch.complex(
        gr.real, torch.where(reg, im_reg.broadcast_to(gr.imag.shape), gr.imag)
    ).to(fc.dtype)
    idx = g.flat_rows.expand(*lead, k * p)
    return fc.broadcast_to(*lead, n).scatter(-1, idx, new_t.reshape(*lead, k * p))


def _required_q(g: PVGroup, fc: Tensor, v: Tensor) -> Tensor:
    """Solved total reactive injection ``[*lead, K]`` (var) from the raw residual.

    ``Q_e = Q_e,pinned - Im(conj(V_r) F_c,r)`` per element (the reactive mismatch the
    terminal's free reactive power absorbs), summed over the generator's phases.
    """
    lead = _lead(g, fc, v)
    v_t = _gather(g, v, lead)
    f_t = _gather(g, fc, lead)
    dq = (torch.conj(v_t) * f_t).imag.sum(-1)  # [*lead, K]
    return _pinned_q(g) - dq


def _pinned_q(g: PVGroup) -> Tensor:
    """Reactive power written into the operating point ``[*batch, K]`` (var).

    The active set, the limits and the setpoint may each carry their own batch shape
    (an unbatched limit against a per-scenario active set, say), so all four are
    broadcast to their common shape before the selection.
    """
    lo, hi, state, zero = torch.broadcast_tensors(
        g.q_min, g.q_max, g.state, torch.zeros_like(g.v_set_pu)
    )
    return torch.where(state > 0, hi, torch.where(state < 0, lo, zero))


def _switch_group(
    g: PVGroup, fc: Tensor, v: Tensor, hyst_pu: float, hyst_rel: float
) -> Tensor:
    """The group's next active set ``[*lead, K]`` int8 (off-tape decision)."""
    with torch.no_grad():
        lead = _lead(g, fc, v)
        v_t = _gather(g, v, lead)
        rdt = v_t.real.dtype
        q_req = _required_q(g, fc, v)  # [*lead, K]
        v0 = g.v0.to(dtype=rdt, device=v.device)
        v_pu = (
            (v_t.abs() / v0.reshape(-1, 1)).mean(-1)
            if g.per_phase
            else _regulated_voltage(g, v_t) / v0
        )  # [*lead, K]
        # Hysteresis from the FINITE limits only (an unbounded side must not widen
        # the band of the bounded one).
        finite = torch.zeros_like(g.q_max)
        for lim in (g.q_min, g.q_max):
            finite = torch.maximum(
                finite,
                torch.where(torch.isfinite(lim), lim.abs(), torch.zeros_like(lim)),
            )
        hyst_q = hyst_rel * finite
        state = g.state.broadcast_to(*lead, g.rows.shape[0])
        regulating = state == 0
        hit_max = regulating & (q_req > g.q_max + hyst_q)
        hit_min = regulating & (q_req < g.q_min - hyst_q)
        release = ((state > 0) & (v_pu > g.v_set_pu + hyst_pu)) | (
            (state < 0) & (v_pu < g.v_set_pu - hyst_pu)
        )
        hi = torch.ones_like(state)
        new_state = torch.where(
            hit_max,
            hi,
            torch.where(
                hit_min,
                -hi,
                torch.where(release, torch.zeros_like(state), state),
            ),
        )
        return new_state.to(torch.int8)


def active_power_mismatch(pv: PVTerminals, fc: Tensor, v: Tensor) -> Tensor:
    """Per-row mismatch magnitude a regulating terminal actually enforces ``[*b, N]``.

    The raw current mismatch at a regulating row is nonzero by construction (it is the
    reactive current the generator supplies), so the convergence diagnostics report the
    ACTIVE component ``|Re(conj(V) F_c)| / |V|`` there instead. Pinned terminals keep
    their full mismatch. Returns a tensor to SCATTER over the diagnostics' ``|F_c|``.
    """
    out = fc.abs()
    for g in pv.groups:
        lead = _lead(g, fc, v)
        k, p = g.rows.shape
        v_t = _gather(g, v, lead)
        f_t = _gather(g, fc, lead)
        active = (torch.conj(v_t) * f_t).real / v_t.abs().clamp_min(1e-30)
        reg = (g.state == 0).unsqueeze(-1).broadcast_to(active.shape)
        val = torch.where(reg, active.abs(), f_t.abs().broadcast_to(active.shape))
        idx = g.flat_rows.expand(*lead, k * p)
        out = out.broadcast_to(*lead, fc.shape[-1]).scatter(
            -1, idx, val.reshape(*lead, k * p)
        )
    return out


def _slice_group(g: PVGroup, i: int) -> PVGroup:
    def slc(t: Tensor) -> Tensor:
        return t[i] if t.ndim >= 2 and t.shape[0] > 1 else t

    return replace(
        g,
        v_set_pu=slc(g.v_set_pu),
        q_min=slc(g.q_min),
        q_max=slc(g.q_max),
        state=slc(g.state),
    )


__all__ = ["PVGroup", "PVTerminals", "collect_pv_terminals", "active_power_mismatch"]
