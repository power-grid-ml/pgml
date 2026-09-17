"""Differentiable, batched, per-frequency Y-bus assembly and current injections.

Public API
----------
- ``assemble_ybus`` — ``(grid, frequencies_hz, *, dtype, device, operating_point,
  param_overrides=None) -> YBus``
- ``build_injections`` — ``(grid, frequencies_hz, index, *, dtype, device,
  operating_point, param_overrides=None) -> Tensor``
- ``branch_currents`` — ``(grid, v, frequencies_hz, index, *, dtype, device,
  param_overrides=None) -> list[BranchCurrent]`` (per-branch terminal currents).
- ``node_phase_index`` / ``NodePhaseIndex`` (re-exported from ``.index``).

See ``assembly/CONTEXT.md`` for the frozen layout and stamp definitions.

Differentiability + GPU (CLAUDE.md): every Y / I entry is differentiable w.r.t.
R, L, G, C, length, tap, source Z and operating P, Q. The tape starts where a
python parameter is turned into a tensor (or, for gradcheck, where the caller
supplies a leaf tensor via ``param_overrides``). All tensor math is autograd-safe
torch with no ``.item()/.detach()/.numpy()``, no in-place on tracked leaves, no
hard-coded device, and no python loop over individual branches/phases on the
differentiable path (loops are only over the fixed set of component KINDS).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

from pgml.defaults import get as _cfg
from pgml.errors import InputError
from pgml.schemas.grid_schema import (
    GenericBranch,
    Grid,
    InjectionAppliance,
    Line,
    Load,
    LoadModel,
    Phase,
    ShuntAppliance,
    ShuntReactor,
    Source,
    Switch,
    Transformer,
    WindingConnection,
)

from ._control import resolve_injection_power
from ._fusion import (
    FusionMap,
    describe_unfusable,
    fused_branch_currents,
    log_fusion_summary,
    resolve_fusion,
    zero_impedance_branches,
)
from ._incidence import (
    build_incidence,
    cyclic_delta_incidence,
    group_appliances,
    used_rows,
)
from ._transformer import (
    block_incidence,
    group_key as _xfmr_group_key,
    harmonic_resistance_law,
    is_sequence_aware as _xfmr_is_sequence_aware,
    magnetizing_blocks,
    magnetizing_placement,
    resistance_scales_with_order,
    nominal_turns_ratio,
    resolve_vector_group,
    sequence_leakage_matrices,
    winding_leakage_block,
    zero_sequence_leakage,
)
from ._params import (
    const_z_shunt_admittance,
    phase_voltage_magnitude,
    resolve_operating_power,
)
from ._scatter import scatter_blocks_into
from ._symmetry import log_modeling_summary, resolve_asymmetric
from ._stamps import (
    _cdtype,
    _rdtype,
    pi_series_blocks,
    series_admittance_matrix,
    shunt_admittance_matrix,
)
from .index import NodePhaseIndex, node_phase_index

_log = logging.getLogger("pgml")


@dataclass(frozen=True)
class YBus:
    """Assembled nodal admittance system.

    Attributes
    ----------
    Y:
        Complex tensor ``[*batch, H, N, N]`` (or ``[N, N]`` if a single grid and a
        single frequency were requested and ``squeeze`` applies).
    index:
        The :class:`NodePhaseIndex` describing the compact row layout of ``Y`` — the
        grid's full layout, or the REDUCED one when ``fusion`` is set.
    frequencies_hz:
        Real tensor ``[H]`` of the absolute frequencies the Y was built at.
    fusion:
        The :class:`~pgml.assembly._fusion.FusionMap` applied, or ``None`` when the
        grid has no zero-impedance branch to collapse. When set, ``Y``'s rows are the
        fused ones (``index is fusion.index``, ``N == fusion.size``) and
        ``fusion.prolong`` maps a solution back to the grid's full row layout.
    """

    Y: Tensor
    index: NodePhaseIndex
    frequencies_hz: Tensor
    fusion: Optional[FusionMap] = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_freq_tensor(frequencies_hz, dtype: torch.dtype, device) -> Tensor:
    """Coerce the frequencies argument to a real 1-D tensor ``[H]``."""
    rdt = _rdtype(dtype)
    if isinstance(frequencies_hz, Tensor):
        f = frequencies_hz.to(dtype=rdt, device=device)
    else:
        f = torch.as_tensor(frequencies_hz, dtype=rdt, device=device)
    if f.ndim == 0:
        f = f.reshape(1)
    return f


def _override(param_overrides: Optional[dict], key: tuple, default: Tensor) -> Tensor:
    """Return a caller-supplied leaf tensor for ``key`` if present, else ``default``.

    ``param_overrides`` maps ``(component_kind, element_id, field_name)`` to a
    tensor of the SAME shape/meaning as the default. This is the hook a gradcheck
    test uses to inject differentiable leaves for R/L/C/Z without mutating the
    frozen schema. ``None`` disables overriding.
    """
    if param_overrides is None:
        return default
    val = param_overrides.get(key)
    if val is None:
        return default
    # Match BOTH dtype and device of the default, otherwise a leaf created on a
    # different device (e.g. a CPU gradcheck tensor while assembling on CUDA) would
    # be returned stale and crash downstream at the first stack/scatter. `.to()` is
    # differentiable, so the override stays a valid autograd leaf.
    if val.dtype != default.dtype or val.device != default.device:
        return val.to(dtype=default.dtype, device=default.device)
    return val


def _real_matrix(mat: Sequence[Sequence[float]], rdt: torch.dtype, device) -> Tensor:
    return torch.as_tensor(mat, dtype=rdt, device=device)


def _stack_or_empty(mats: list[Tensor], p: int, rdt: torch.dtype, device) -> Tensor:
    """Stack ``[P,P]`` matrices to ``[K,P,P]``; empty -> ``[0,P,P]``."""
    if mats:
        return torch.stack(mats, dim=0)
    return torch.zeros((0, p, p), dtype=rdt, device=device)


# ---------------------------------------------------------------------------
# branch-state masking (topology / switch-state batching)
# ---------------------------------------------------------------------------
def _branch_active(b, branch_states: Optional[dict]) -> bool:
    """Stamp this branch? ``branch_states`` OVERRIDES the static flags.

    A branch listed in ``branch_states`` is ALWAYS stamped — its (possibly
    batched, possibly zero) state scales the primitive block, so "open" is the
    state value 0, not an omitted stamp. Unlisted branches keep the static
    semantics: in service, and (for a :class:`Switch`) closed.
    """
    if branch_states and b.id in branch_states:
        return True
    if not b.in_service:
        return False
    if isinstance(b, Switch) and not b.closed:
        return False
    return True


def _group_states(
    group, branch_states: Optional[dict], rdt: torch.dtype, device
) -> Optional[Tensor]:
    """Per-branch state factors ``[*batch, K]`` for a stamp group (``None`` = all 1).

    Each listed branch contributes its state (float, 0-d, or ``[*batch]`` scenario
    tensor); unlisted group members contribute the neutral 1.0. Entries broadcast
    to a common leading batch before stacking, so one batched switch promotes the
    whole group (and thus ``Y``) to scenario-batched. Tensor states keep their
    autograd graph — a continuous state in ``[0, 1]`` is a differentiable
    topology parameter.
    """
    if not branch_states or not any(b.id in branch_states for b in group):
        return None
    vals = []
    for b in group:
        v = branch_states.get(b.id, 1.0)
        vt = (
            v.to(dtype=rdt, device=device)
            if isinstance(v, Tensor)
            else torch.as_tensor(float(v), dtype=rdt, device=device)
        )
        vals.append(vt)
    lead = torch.broadcast_shapes(*[t.shape for t in vals])
    return torch.stack([t.broadcast_to(lead) for t in vals], dim=-1)  # [*lead, K]


def _masked_block(block: Tensor, state: Optional[Tensor]) -> Tensor:
    """Scale a primitive block ``[H, K, M, M]`` by per-branch states ``[*b, K]``."""
    if state is None:
        return block
    return block * state[..., None, :, None, None].to(block.dtype)


# ---------------------------------------------------------------------------
# Branch-stamp registry (THE extension point for branch / device models)
# ---------------------------------------------------------------------------
# A "light" registry: one ordered list of the branch KINDS the assembler knows,
# each entry pairing a kind name with its primitive-block builder. Both the Y-bus
# assembly (:func:`_stamp_network`, which scatters every yielded block) and the
# branch-current derivation (:func:`branch_currents`, which multiplies each block
# with the terminal voltage) iterate THIS list — so adding a new branch model is a
# one-line registration, not an edit to either dispatch path.
#
# A builder is a generator with the uniform signature
# ``builder(grid, f, index, cdt, rdt, device, param_overrides, branch_states)``
# that YIELDS
# ``(group, block, rows, cols)``: ``group`` is the list of source branches, ``block``
# is the primitive admittance ``[H, K, M, M]`` (``M = 2P`` for a two-terminal series
# branch, ``M = P`` for a single-terminal shunt), and ``rows == cols`` are the global
# node-row indices ``[K, M]`` that block scatters into / gathers its voltage from.
#
# ``single_terminal=True`` marks a shunt-type branch (one terminal, ``M = P``): its
# block has no TO half, so :func:`branch_currents` emits it as ``to_node=None`` with
# an empty ``i_to`` instead of splitting the block into from/to halves.


@dataclass(frozen=True)
class _BranchStamp:
    """A registered branch kind: its name, primitive builder, and terminal arity."""

    kind: str
    builder: object  # generator(grid, f, index, cdt, rdt, device, param_overrides)
    single_terminal: bool


_BRANCH_STAMPS: list[_BranchStamp] = []


def _branch_stamp(kind: str, *, single_terminal: bool = False):
    """Register a ``_<kind>_block_groups`` generator as a branch-stamp builder.

    Decorates the primitive-block generator in place (returns it unchanged) and
    appends it to :data:`_BRANCH_STAMPS` in definition order. ``single_terminal``
    flags a one-terminal shunt branch (block ``[H, K, P, P]``, no TO half).
    """

    def register(builder):
        _BRANCH_STAMPS.append(
            _BranchStamp(kind=kind, builder=builder, single_terminal=single_terminal)
        )
        return builder

    return register


# ---------------------------------------------------------------------------
# Y-bus assembly
# ---------------------------------------------------------------------------
def assemble_ybus(
    grid: Grid,
    frequencies_hz,
    *,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    operating_point: Optional[dict] = None,
    param_overrides: Optional[dict] = None,
    symmetry: Optional[str] = None,
    branch_states: Optional[dict] = None,
    fusion: Optional[object] = None,
) -> YBus:
    """Assemble the LINEAR (const-Z) complex nodal admittance ``Y(f)``.

    This is the linear-model assembler: it equals
    :func:`assemble_network_ybus` (the passive network) PLUS the const-Z device
    shunts (loads/generators folded as constant impedance at nominal voltage) PLUS
    the source Norton (Thévenin) shunt. The nonlinear (const-P / ZIP) power flow
    instead uses :func:`assemble_network_ybus` + :func:`device_current_injections`
    (see ``assembly/CONTEXT.md``).

    The folded device shunt is the OPERATING POINT expressed as an admittance: its
    conductance ``P/|V0|^2`` is a resistance and stays flat with frequency, while its
    susceptance ``-Q/|V0|^2`` is the equivalent reactive element and scales like one
    (``* h`` where the device is capacitive, ``/ h`` where it is inductive) — the
    parallel R-L / R-C branch of the classical harmonic load model. The fold is exact
    at ``f0``; it is a MODEL of the device at other frequencies, derived from P and Q
    and not from a measured harmonic impedance. The harmonic power flow
    (:func:`pgml.solver.solve_harmonic_flow`) does not fold devices at all: it
    assembles :func:`assemble_network_ybus` and treats every device as a current
    source, so use that function for a harmonic network matrix.

    Parameters
    ----------
    grid:
        A materialised :class:`~pgml.schemas.grid_schema.Grid` (all ``type_ref``
        already expanded).
    frequencies_hz:
        1-D real tensor / sequence of ``H`` absolute frequencies (Hz). The
        harmonic order is ``h = f / grid.base_frequency_hz`` (need not be integer).
    dtype:
        Complex dtype of ``Y`` (``complex128`` for gradcheck; ``complex64`` ok).
    device:
        Target device; defaults to the device of ``frequencies_hz`` if it is a
        tensor, else CPU. No device is hard-coded.
    operating_point:
        Optional ``{appliance_id: {...}}`` overriding nameplate P/Q for the const-Z
        load/gen model (see :func:`resolve_operating_power`).
    param_overrides:
        Optional differentiability hook (see :func:`_override`): map
        ``(kind, id, field) -> leaf tensor`` to inject gradient-bearing parameters
        without editing the schema. Used by gradcheck tests.
    symmetry:
        Calculation-symmetry mode ``None`` / ``"auto"`` / ``"symmetric"`` /
        ``"asymmetric"`` (``None`` -> config ``calculation.symmetry``). Resolved ONCE
        via :func:`resolve_asymmetric`: ``True`` honors per-phase data, ``False``
        splits each total equally over the phases (ignoring per-phase data).
    branch_states:
        Optional topology / switch-state mask ``{branch_id: state}``. A listed
        branch is ALWAYS stamped (overriding ``in_service`` / ``closed``) and its
        primitive block is multiplied by the state — a python float, a 0-d tensor,
        or a ``[*batch]`` scenario tensor in ``[0, 1]`` (0 = open, 1 = in service;
        intermediate values scale the admittance continuously and stay
        differentiable). A batched state promotes ``Y`` to ``[*batch, H, N, N]``,
        so one assembly covers a whole batch of switch configurations.
    fusion:
        Exact bus fusion of zero-impedance branches. ``None`` (default) resolves the
        documented policy ``branch.zero_impedance``; ``False`` refuses a
        zero-impedance branch by name instead of collapsing it; a
        :class:`~pgml.assembly._fusion.FusionMap` uses that map (the way a solve shares
        one map across its orders), and :data:`~pgml.assembly._fusion.NO_FUSION` says
        that the resolution has already run and produced no map, which skips the branch
        walk. When fusion applies, ``Y`` is the REDUCED system ``P^T Y P`` and the
        returned ``index`` is the reduced layout.

    Returns
    -------
    YBus
        ``Y`` complex ``[H, N, N]`` (``[N, N]`` if ``H == 1`` and a scalar
        frequency was passed; ``[*batch, H, N, N]`` with batched
        ``branch_states``), the :class:`NodePhaseIndex` of its rows, the frequencies,
        and the :class:`~pgml.assembly._fusion.FusionMap` when one applied.
    """
    if device is None and isinstance(frequencies_hz, Tensor):
        device = frequencies_hz.device
    f = _as_freq_tensor(frequencies_hz, dtype, device)
    device = f.device
    h = f.shape[0]
    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)

    fused = resolve_fusion(
        grid, fusion, param_overrides=param_overrides, branch_states=branch_states
    )
    index = fused.index if fused is not None else node_phase_index(grid)
    n = index.size

    asymmetric = resolve_asymmetric(grid, operating_point, mode=symmetry)
    # Single INFO modeling summary for this entry point (logs once per assemble_ybus
    # call; solve_power_flow logs its own, and harmonic_flow delegates its log to
    # solve_power_flow — so no double logging across the public entry points).
    log_modeling_summary(grid, asymmetric=asymmetric)
    log_fusion_summary(fused)

    y = torch.zeros((h, n, n), dtype=cdt, device=device)

    # Passive network (shared with assemble_network_ybus).
    y = _stamp_network(
        grid, f, y, index, cdt, rdt, device, param_overrides, branch_states, fused
    )
    # Linear-model device folding: source Norton + const-Z loads/gens.
    y = _stamp_sources(grid, f, y, index, cdt, rdt, device, param_overrides)
    y = _stamp_const_z_loads(
        grid,
        f,
        y,
        index,
        cdt,
        rdt,
        device,
        operating_point,
        param_overrides,
        asymmetric,
    )

    scalar_freq = (
        not isinstance(frequencies_hz, Tensor)
        and not isinstance(frequencies_hz, (list, tuple))
        and h == 1
    )
    if scalar_freq and y.ndim == 3:
        y = y.reshape(n, n)
    return YBus(Y=y, index=index, frequencies_hz=f, fusion=fused)


def assemble_network_ybus(
    grid: Grid,
    frequencies_hz,
    *,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    param_overrides: Optional[dict] = None,
    branch_states: Optional[dict] = None,
    fusion: Optional[object] = None,
) -> YBus:
    """Assemble the PASSIVE-NETWORK nodal admittance ``Y_net(f)``.

    Contains ONLY the passive network: lines, transformers, switches, shunt
    reactors, generic branches, and :class:`ShuntAppliance` (a fixed linear
    shunt). It does NOT stamp loads/generators and does NOT fold the
    source as a Norton shunt — both are handled on the RHS by the nonlinear
    power-flow path (:func:`device_current_injections` and the slack handling in
    :func:`pgml.solver.solve_power_flow`).

    It is constant (voltage-independent) and differentiable w.r.t. network params,
    and uses the SAME compact :class:`NodePhaseIndex` layout and ``[*batch, H, N,
    N]`` shapes as :func:`assemble_ybus`.

    Parameters
    ----------
    grid, frequencies_hz, dtype, device, param_overrides, branch_states, fusion:
        Identical meaning to :func:`assemble_ybus` (no ``operating_point`` — there
        is no device folding here). ``branch_states`` masks/batches branch stamps;
        ``fusion`` collapses zero-impedance branches into single rows.

    Returns
    -------
    YBus
        ``Y`` complex ``[H, N, N]`` (``[N, N]`` for a scalar frequency;
        ``[*batch, H, N, N]`` with batched ``branch_states``), the
        :class:`NodePhaseIndex` of its rows, the frequencies, and the
        :class:`~pgml.assembly._fusion.FusionMap` when one applied.
    """
    if device is None and isinstance(frequencies_hz, Tensor):
        device = frequencies_hz.device
    f = _as_freq_tensor(frequencies_hz, dtype, device)
    device = f.device
    h = f.shape[0]
    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)

    fused = resolve_fusion(
        grid, fusion, param_overrides=param_overrides, branch_states=branch_states
    )
    index = fused.index if fused is not None else node_phase_index(grid)
    n = index.size

    y = torch.zeros((h, n, n), dtype=cdt, device=device)
    y = _stamp_network(
        grid, f, y, index, cdt, rdt, device, param_overrides, branch_states, fused
    )

    scalar_freq = (
        not isinstance(frequencies_hz, Tensor)
        and not isinstance(frequencies_hz, (list, tuple))
        and h == 1
    )
    if scalar_freq and y.ndim == 3:
        y = y.reshape(n, n)
    return YBus(Y=y, index=index, frequencies_hz=f, fusion=fused)


def _unfused_view(grid: Grid, fusion) -> Grid:
    """A shallow grid view without the fused branches (same nodes, same catalog).

    A fused branch has no primitive admittance to stamp — its physics is the row
    collapse itself — so the stamp builders never see it. Dropping it from a shallow
    ``model_copy`` keeps every parameter tensor (and its autograd identity) shared with
    the original grid.
    """
    if fusion is None or not fusion.fused_branch_ids:
        return grid
    drop = set(fusion.fused_branch_ids)
    return grid.model_copy(
        update={"branches": [b for b in grid.branches if int(b.id) not in drop]}
    )


def _stamp_network(
    grid,
    f,
    y,
    index,
    cdt,
    rdt,
    device,
    param_overrides,
    branch_states=None,
    fusion=None,
):
    """Accumulate every PASSIVE contribution into ``y`` (shared assembler core).

    Iterates the branch-stamp registry (:data:`_BRANCH_STAMPS`) — lines, switches,
    generic branches, shunt reactors, transformers — scattering every primitive
    block each builder yields, then adds the ShuntAppliance fixed shunt (an
    appliance, not a branch, so outside the registry). NO source Norton, NO
    load/generator folding. Because scatter-add is order-independent, the resulting
    ``y`` is identical regardless of the registration order.

    ``branch_states`` scales each listed branch's primitive block by its state
    (:func:`_group_states`) before scattering; a batched state promotes ``y`` to
    ``[*batch, H, N, N]`` through the scatter's broadcast. With a ``fusion`` map the
    builders run on the grid WITHOUT its fused branches and index through the reduced
    layout, which accumulates ``P^T Y P`` directly (the fused rows' scatter targets
    coincide) without ever forming the full system.
    """
    grid = _unfused_view(grid, fusion)
    for stamp in _BRANCH_STAMPS:
        for group, block, rows, cols in stamp.builder(
            grid, f, index, cdt, rdt, device, param_overrides, branch_states
        ):
            block = _masked_block(
                block, _group_states(group, branch_states, rdt, device)
            )
            y = scatter_blocks_into(y, block, rows, cols)
    y = _stamp_shunt_appliances(grid, f, y, index, cdt, rdt, device, param_overrides)
    return y


# ---- series-branch stamps -------------------------------------------------
def _series_terminal_indices(
    branches, index: NodePhaseIndex, device
) -> tuple[Tensor, Tensor]:
    """Global row indices for a list of series (2-terminal, P-phase) branches.

    Returns ``rows[K, 2P]`` == ``cols[K, 2P]`` (the primitive block is symmetric in
    its row/col mapping): first P entries are the from-terminal rows, next P the
    to-terminal rows.
    """
    idx_rows: list[list[int]] = []
    for b in branches:
        from_rows = [index.row(b.from_node, ph) for ph in b.from_phases]
        to_rows = [index.row(b.to_node, ph) for ph in b.to_phases]
        idx_rows.append(from_rows + to_rows)
    rows = torch.as_tensor(idx_rows, dtype=torch.int64, device=device)
    return rows, rows


def _is_sequence_aware(line) -> bool:
    """A line that selected the sequence-aware harmonic model (UNBALANCED studies)."""
    return line.harmonic_line_model == "sequence_aware"


@_branch_stamp("line")
def _line_block_groups(
    grid, f, index, cdt, rdt, device, param_overrides, branch_states=None
):
    """Yield ``(group, block, rows, cols)`` for every line group (all three paths).

    Every path produces a LUMPED pi branch — ``Z = z·length``, ``Y = y·length`` with the
    shunt split half to each terminal, no hyperbolic long-line correction and no
    distributed-parameter model. Frequency-domain steady state only: no standing or
    travelling waves. Accurate for distribution feeders over the harmonic range; split a
    long line into segments when the electrical length stops being small.

    Explicit-R/L/C lines go through the matrix path; lines carrying a
    ``conductor_geometry`` go through the Carson/Deri geometry path; lines tagged
    ``harmonic_line_model=sequence_aware`` go through the Z1/Z0 sequence-aware path
    (frequency-correct each sequence: earth return only in Z0). See
    ``geometry/sequence.py`` / ``apply_sequence_aware_harmonic_model``.

    This is the SHARED primitive builder registered in :data:`_BRANCH_STAMPS`: the
    Y-bus stamp scatters each block into the Y-bus, while :func:`branch_currents`
    multiplies it with the terminal voltage. Yielding (rather than scattering
    inline) keeps the two consumers in lockstep with no duplicated physics.
    """
    flow_lines = [
        b
        for b in grid.branches
        if isinstance(b, Line)
        and _branch_active(b, branch_states)
        and b.conductor_geometry is None
    ]
    rx_lines = [b for b in flow_lines if not _is_sequence_aware(b)]
    seq_lines = [b for b in flow_lines if _is_sequence_aware(b)]
    if rx_lines:
        yield from _line_rx_block_groups(
            rx_lines,
            f,
            index,
            cdt,
            rdt,
            device,
            param_overrides,
            f0=float(grid.base_frequency_hz),
        )
    if seq_lines:
        yield from _sequence_aware_block_groups(
            seq_lines, grid, f, index, cdt, rdt, device, param_overrides
        )
    yield from _geometry_block_groups(
        grid, f, index, cdt, rdt, device, param_overrides, branch_states
    )


def _sequence_aware_options(line) -> tuple:
    """DISCRETE sequence-aware model options of one line (the batching group key).

    ``(skin, x0_frequency, r0_includes_earth_return)``, each resolved from the line's
    typed fields and falling back to the modeling defaults (``pgml.defaults``). These
    select code paths, so they group lines; the numeric coefficients do not (see
    :func:`_earth_field`).
    """
    er = line.earth_return
    skin = line.harmonic_skin_effect
    if skin is None:
        skin = _cfg("line.harmonic_model.skin_effect")
    x0_frequency = getattr(er, "x0_frequency", None) or _cfg(
        "line.earth_return.x0_frequency"
    )
    r0_inc = getattr(er, "r0_includes_earth_return", None)
    if r0_inc is None:
        r0_inc = _cfg("line.zero_sequence.r0_includes_earth_return")
    nonnegative = getattr(er, "x0_nonnegative", None)
    if nonnegative is None:
        nonnegative = _cfg("line.earth_return.x0_nonnegative")
    return bool(skin), str(x0_frequency), bool(r0_inc), bool(nonnegative)


#: Modeling-default keys of the numeric earth-return coefficients, by field name.
_EARTH_DEFAULT_KEY = {
    "resistance_coeff_ohm_per_m_per_hz": (
        "line.earth_return.resistance_coeff_ohm_per_m_per_hz"
    ),
    "reactance_coeff_ohm_per_m_per_hz": (
        "line.earth_return.reactance_coeff_ohm_per_m_per_hz"
    ),
    "x0_exponent": "line.earth_return.x0_exponent",
}


def _earth_field(lines, field: str, rdt: torch.dtype, device) -> Tensor:
    """Stack one numeric earth-return coefficient over a line group -> ``[K]``.

    Each line's :class:`~pgml.schemas.grid_schema.EarthReturnModel` value is used when
    set (a python float OR a tensor, so gradients flow), else the modeling default.
    """
    vals = []
    for ln in lines:
        v = getattr(ln.earth_return, field, None) if ln.earth_return else None
        if v is None:
            v = _cfg(_EARTH_DEFAULT_KEY[field])
        vals.append(_geom_scalar(v, rdt, device))
    return torch.stack(vals, 0)


def _sequence_aware_block_groups(
    lines, grid, f, index, cdt, rdt, device, param_overrides
):
    """Yield ``(group, block, rows, cols)`` for sequence-aware 3-phase lines.

    Each line's reference-frequency phase matrix ``Z_abc(f0) = R + j·2πf0·L`` is
    decomposed into ``Z1(f0)``/``Z0(f0)`` (balanced/transposed), each sequence is
    frequency-corrected separately (``Z1``: ``X∝h`` + skin, NO earth; ``Z0``: conductor
    + Carson earth-return resistance), and the result recombined to ``Z_abc(h)`` — the
    model an asymmetric 4-wire study needs. Differentiable in R/L; the shunt ``C`` keeps
    the usual ``B∝h`` split.
    """
    from pgml.geometry.sequence import sequence_aware_phase_z, x0_sublinear_deficit

    f0 = float(grid.base_frequency_hz)
    two_pi_f0 = 2.0 * math.pi * f0
    two_pi_f = (2.0 * torch.pi) * f  # [H]
    deficit_ids: list[str] = []
    deficit_guarded = True

    # Group by the DISCRETE model options (skin flag, reactance law, R0 convention) so
    # each batched call shares them; the numeric earth coefficients stay per line and
    # are stacked into tensors, so a tensor coefficient keeps its gradient.
    by_opts: dict[tuple, list] = {}
    for ln in lines:
        by_opts.setdefault(_sequence_aware_options(ln), []).append(ln)

    for opts, group in by_opts.items():
        skin, x0_frequency, r0_includes_earth, x0_nonnegative = opts
        coeff = _earth_field(group, "resistance_coeff_ohm_per_m_per_hz", rdt, device)
        coeff_x = _earth_field(group, "reactance_coeff_ohm_per_m_per_hz", rdt, device)
        x0_exponent = _earth_field(group, "x0_exponent", rdt, device)
        r_list, l_list, c_list, len_list = [], [], [], []
        for ln in group:
            r_list.append(
                _override(
                    param_overrides,
                    ("line", ln.id, "series_resistance_ohm_per_m"),
                    _real_matrix(ln.series_resistance_ohm_per_m, rdt, device),
                )
            )
            l_list.append(
                _override(
                    param_overrides,
                    ("line", ln.id, "series_inductance_h_per_m"),
                    _real_matrix(ln.series_inductance_h_per_m, rdt, device),
                )
            )
            c_list.append(
                _real_matrix(ln.shunt_capacitance_f_per_m, rdt, device)
                if ln.shunt_capacitance_f_per_m is not None
                else torch.zeros((3, 3), dtype=rdt, device=device)
            )
            len_list.append(_geom_scalar(ln.length_m, rdt, device))
        r = torch.stack(r_list, 0)  # [K,3,3]
        ind = torch.stack(l_list, 0)
        c = torch.stack(c_list, 0)
        length = torch.stack(len_list, 0)  # [K]

        # Z_abc(f0) -> sequence (Z1, Z0) via the symmetric (transposed) decomposition.
        z_f0 = torch.complex(r, two_pi_f0 * ind).to(cdt)  # [K,3,3]
        diag = z_f0.diagonal(dim1=-2, dim2=-1)  # [K,3]
        zs = diag.mean(-1)  # [K] self
        zm = (z_f0.sum((-2, -1)) - diag.sum(-1)) / 6.0  # [K] mutual (6 off-diagonals)
        z1 = zs - zm  # [K]
        z0 = zs + 2.0 * zm
        if x0_frequency == "carson_sublinear":
            deficit = x0_sublinear_deficit(
                z0.imag,
                f0,
                f,
                earth_reactance_coeff=coeff_x,
                x0_exponent=x0_exponent,
            ).any(-1)  # [K]
            hit = [ln.id for ln, d in zip(group, deficit.tolist()) if d]
            deficit_ids.extend(hit)
            deficit_guarded = deficit_guarded and (x0_nonnegative or not hit)
        z_abc = sequence_aware_phase_z(
            z1.real,
            z1.imag,
            z0.real,
            z0.imag,
            f0,
            f,
            skin=skin,
            earth_resistance_coeff=coeff,
            earth_reactance_coeff=coeff_x,
            x0_frequency=x0_frequency,
            x0_nonnegative=x0_nonnegative,
            x0_exponent=x0_exponent,
            r0_includes_earth_return=r0_includes_earth,
        )  # [K,H,3,3]  Ω/m

        z_len = (z_abc * length[:, None, None, None]).to(cdt)  # [K,H,3,3]
        ys_adm = torch.linalg.inv(z_len).transpose(0, 1)  # [H,K,3,3]
        series_block = pi_series_blocks(ys_adm)

        c_len = (c * length[:, None, None]).to(cdt)  # [K,3,3]
        yc = (1j * two_pi_f).to(cdt)[:, None, None, None] * c_len[None]  # [H,K,3,3]
        half = 0.5 * yc
        zeros = torch.zeros_like(half)
        shunt_block = torch.cat(
            [torch.cat([half, zeros], dim=-1), torch.cat([zeros, half], dim=-1)], dim=-2
        )
        block = series_block + shunt_block
        rows, cols = _series_terminal_indices(group, index, device)
        yield group, block, rows, cols
    _warn_x0_sublinear_deficit(deficit_ids, len(lines), guarded=deficit_guarded)


#: Line sets already reported by :func:`_warn_x0_sublinear_deficit`, so that the
#: per-scenario and per-order assemblies of one study log the condition once.
_X0_DEFICIT_REPORTED: set[tuple] = set()


def _warn_x0_sublinear_deficit(line_ids, n_lines: int, *, guarded: bool) -> None:
    """Warn once per affected line set that ``carson_sublinear`` exhausted ``X0(h)``.

    The law subtracts the frequency decay of a deep-earth return from the stored
    ``X0``. Where the result is negative the stored ``X0`` never contained that earth
    term (a cable, or an ``X0`` derived from an ``X0/X1`` ratio), and the requested
    orders are assembled with a zero (guard on) or negative (guard off) zero-sequence
    reactance.
    """
    if not line_ids:
        return
    key = (tuple(sorted(line_ids)), guarded)
    if key in _X0_DEFICIT_REPORTED:
        return
    _X0_DEFICIT_REPORTED.add(key)
    _log.warning(
        "pgml: x0_frequency='carson_sublinear' leaves no zero-sequence reactance on "
        "%d of %d sequence-aware line(s) within the requested frequencies; X0(h) is "
        "%s there. The law presumes a stored X0 that contains the deep-earth return "
        "reactance (an overhead line); a cable or an X0 derived from an X0/X1 ratio "
        "does not. Use line.earth_return.x0_frequency='linear' (the default) for "
        "these lines, or give them a conductor_geometry. First affected: %s.",
        len(line_ids),
        n_lines,
        "clamped at zero" if guarded else "NEGATIVE (x0_nonnegative is off)",
        sorted(line_ids)[:5],
    )


def _geom_scalar(v, rdt: torch.dtype, device) -> Tensor:
    """Coerce a geometry field (python float OR tensor) to a 0-d real tensor (grad-safe)."""
    if isinstance(v, Tensor):
        return v.to(dtype=rdt, device=device).reshape(())
    return torch.as_tensor(float(v), dtype=rdt, device=device)


def _geom_conductor_arrays(ln, rdt, device):
    """Per-conductor arrays ``[Ncond]`` ordered phases-first (per from_phases) then neutrals."""
    geo = ln.conductor_geometry
    phase_conds = []
    for ph in ln.from_phases:
        c = next(
            (c for c in geo.conductors if not c.is_neutral and c.phase == ph), None
        )
        if c is None:
            raise ValueError(
                f"Line {ln.id}: conductor_geometry has no phase conductor for {ph!r} "
                f"(from_phases={ln.from_phases}); each phase needs a non-neutral conductor."
            )
        phase_conds.append(c)
    ordered = phase_conds + [c for c in geo.conductors if c.is_neutral]
    x = torch.stack([_geom_scalar(c.x_m, rdt, device) for c in ordered])
    y = torch.stack([_geom_scalar(c.y_m, rdt, device) for c in ordered])
    gmr = torch.stack([_geom_scalar(c.gmr_m, rdt, device) for c in ordered])
    rdc = torch.stack([_geom_scalar(c.r_dc_ohm_per_m, rdt, device) for c in ordered])
    rad = torch.stack([_geom_scalar(c.radius_m, rdt, device) for c in ordered])
    return x, y, gmr, rdc, rad


def _geometry_block_groups(
    grid, f, index, cdt, rdt, device, param_overrides, branch_states=None
):
    """Yield ``(group, block, rows, cols)`` for geometry (Carson/Deri) lines.

    The conductor internal-inductance model is read from the modeling defaults
    (``line.geometry.internal_inductance``) at assembly time, so a project-level
    defaults override applies without reimporting the package.
    """
    glines = [
        b
        for b in grid.branches
        if isinstance(b, Line)
        and _branch_active(b, branch_states)
        and b.conductor_geometry is not None
    ]
    if not glines:
        return
    from pgml.geometry.carson import line_constants

    by_key: dict[tuple[int, int, str], list] = {}
    for ln in glines:
        model = ln.conductor_geometry.internal_inductance
        if model is None:
            model = _cfg("line.geometry.internal_inductance")
        key = (len(ln.from_phases), len(ln.conductor_geometry.conductors), model)
        by_key.setdefault(key, []).append(ln)

    two_pi_f = (2.0 * torch.pi) * f  # [H]
    for (nph, _ncond, model), group in by_key.items():
        xs, ys_, gmrs, rdcs, rads, lengths, rhos = [], [], [], [], [], [], []
        for ln in group:
            cx, cy, cg, cr, cra = _geom_conductor_arrays(ln, rdt, device)
            xs.append(cx)
            ys_.append(cy)
            gmrs.append(cg)
            rdcs.append(cr)
            rads.append(cra)
            lengths.append(_geom_scalar(ln.length_m, rdt, device))
            rhos.append(
                _geom_scalar(ln.conductor_geometry.earth_resistivity_ohm_m, rdt, device)
            )
        X, Y = torch.stack(xs), torch.stack(ys_)  # [K, Ncond]
        GMR, RDC, RAD = torch.stack(gmrs), torch.stack(rdcs), torch.stack(rads)
        RHO, LEN = torch.stack(rhos), torch.stack(lengths)  # [K]

        z, c = line_constants(
            X,
            Y,
            GMR,
            RDC,
            RAD,
            RHO,
            f,
            nph,
            internal_inductance=model,
            power_frequency_band_hz=tuple(
                _cfg("line.geometry.power_frequency_band_hz")
            ),
        )  # Z[K,H,P,P] Ω/m, C[K,P,P] F/m
        z_len = (z * LEN[:, None, None, None]).to(cdt)
        ys_adm = torch.linalg.inv(z_len).transpose(0, 1)  # [H,K,P,P] series admittance
        series_block = pi_series_blocks(ys_adm)

        c_len = (c * LEN[:, None, None]).to(cdt)  # [K,P,P]
        yc = (1j * two_pi_f).to(cdt)[:, None, None, None] * c_len[None]  # [H,K,P,P]
        half = 0.5 * yc
        zeros = torch.zeros_like(half)
        shunt_block = torch.cat(
            [torch.cat([half, zeros], dim=-1), torch.cat([zeros, half], dim=-1)], dim=-2
        )

        block = series_block + shunt_block
        rows, cols = _series_terminal_indices(group, index, device)
        yield group, block, rows, cols


def _line_rx_block_groups(
    lines, f, index, cdt, rdt, device, param_overrides, *, f0: float
):
    """Yield ``(group, block, rows, cols)`` for explicit-R/L/C lines.

    Covers the ``naive`` and ``positive_sequence`` models and lines whose
    ``harmonic_line_model`` is still unresolved (assembled from their stored
    parameters). The series resistance at harmonic ``h`` is::

        R(h) = m(h) * (R - R_earth) + R_earth

    where ``R_earth`` holds the off-diagonal (mutual) entries and, on its diagonal, each
    row's mean mutual. On a multi-phase line the mutual resistance IS the earth-return
    term (Carson: it is common to the self and mutual entries), so the skin-effect
    multiplier ``m(h)`` scales the CONDUCTOR part only; a 1-phase line has no mutual and
    keeps ``R(h) = m(h) * R``. ``m(h)`` is the Bessel skin curve of the line's
    positive-sequence resistance for ``positive_sequence`` (differentiable in ``R``),
    ``1`` for ``naive``, and the line's ``resistance_frequency`` law otherwise.
    """
    by_opts: dict[tuple, list] = {}
    for ln in lines:
        by_opts.setdefault(_rx_options(ln), []).append(ln)
    for (p, model, skin), group in by_opts.items():
        r_list, l_list, g_list, c_list, mult_list, r_pm_list = [], [], [], [], [], []
        for ln in group:
            length = ln.length_m
            r_pm = _override(
                param_overrides,
                ("line", ln.id, "series_resistance_ohm_per_m"),
                _real_matrix(ln.series_resistance_ohm_per_m, rdt, device),
            )
            r0 = r_pm * length
            ind = (
                _override(
                    param_overrides,
                    ("line", ln.id, "series_inductance_h_per_m"),
                    _real_matrix(ln.series_inductance_h_per_m, rdt, device),
                )
                * length
            )
            cap = (
                _override(
                    param_overrides,
                    ("line", ln.id, "shunt_capacitance_f_per_m"),
                    _real_matrix(ln.shunt_capacitance_f_per_m, rdt, device),
                )
                * length
            )
            if ln.shunt_conductance_s_per_m is not None:
                cond = (
                    _override(
                        param_overrides,
                        ("line", ln.id, "shunt_conductance_s_per_m"),
                        _real_matrix(ln.shunt_conductance_s_per_m, rdt, device),
                    )
                    * length
                )
            else:
                cond = torch.zeros((p, p), dtype=rdt, device=device)
            r_pm_list.append(r_pm)
            r_list.append(r0)
            l_list.append(ind)
            g_list.append(cond)
            c_list.append(cap)
            if model is None:
                mult_list.append(_resistance_multiplier(ln, f, rdt, device))  # [H]
        r = torch.stack(r_list, 0)  # [K,P,P]
        ind = torch.stack(l_list, 0)
        g = torch.stack(g_list, 0)
        c = torch.stack(c_list, 0)
        r_earth = _mutual_resistance(r)  # [K,P,P] (zeros for P == 1)
        if model is None:
            rmult = torch.stack(mult_list, 1)[:, :, None, None]  # [H,K,1,1]
        elif model == "positive_sequence" and skin:
            from pgml.geometry.sequence import skin_resistance_multiplier

            r1 = _positive_sequence_resistance(torch.stack(r_pm_list, 0))  # [K]
            rmult = skin_resistance_multiplier(r1, f0, f)  # [K,H]
            rmult = rmult.transpose(0, 1)[:, :, None, None]  # [H,K,1,1]
        else:  # naive / positive_sequence without skin: R constant
            rmult = torch.ones((f.shape[0], 1, 1, 1), dtype=rdt, device=device)

        ys = series_admittance_matrix(
            r - r_earth, ind, f, cdt, r_mult=rmult, r_unscaled=r_earth
        )  # [H,K,P,P]
        series_block = pi_series_blocks(ys)  # [H,K,2P,2P]

        # Shunt admittance split half to each terminal diagonal block.
        y_sh = shunt_admittance_matrix(g, c, f, cdt)  # [H,K,P,P]
        half = 0.5 * y_sh
        zeros = torch.zeros_like(half)
        shunt_block = torch.cat(
            [torch.cat([half, zeros], dim=-1), torch.cat([zeros, half], dim=-1)],
            dim=-2,
        )  # [H,K,2P,2P]

        block = series_block + shunt_block
        rows, cols = _series_terminal_indices(group, index, device)
        yield group, block, rows, cols


def _rx_options(line) -> tuple:
    """``(n_phases, harmonic_line_model, skin)`` batching key of an explicit-R/L/C line."""
    model = line.harmonic_line_model
    skin = line.harmonic_skin_effect
    if model == "positive_sequence" and skin is None:
        skin = _cfg("line.harmonic_model.skin_effect")
    return len(line.from_phases), model, bool(skin)


def _mutual_resistance(r: Tensor) -> Tensor:
    """Earth-return part of a resistance matrix ``[K,P,P]``: the mutual entries.

    Off-diagonal entries are kept as they are; each diagonal entry becomes its row's
    mean mutual (for the circulant matrix of a transposed line that is exactly the
    mutual resistance ``Rg``). Returns zeros for a single-phase line, which has no
    mutual and therefore no earth-return component in its stored ``R``.
    """
    p = r.shape[-1]
    if p == 1:
        return torch.zeros_like(r)
    eye = torch.eye(p, dtype=r.dtype, device=r.device)
    off = r * (1.0 - eye)  # [K,P,P]
    row_mean = off.sum(-1) / (p - 1)  # [K,P]
    return off + row_mean.unsqueeze(-1) * eye


def _positive_sequence_resistance(r: Tensor) -> Tensor:
    """Positive-sequence resistance ``[K]`` of per-metre resistance matrices ``[K,P,P]``.

    ``mean(diagonal) - mean(off-diagonal)``: the mutual entries are the earth-return
    term, so the resistance a balanced (positive-sequence) current sees is the
    difference. Matches :func:`pgml.geometry.synthesis._line_representative_r1`.
    """
    p = r.shape[-1]
    diag = r.diagonal(dim1=-2, dim2=-1)  # [K,P]
    self_ = diag.mean(-1)  # [K]
    if p == 1:
        return self_
    mutual = (r.sum((-2, -1)) - diag.sum(-1)) / (p * (p - 1))  # [K]
    return self_ - mutual


def _resistance_multiplier(line, f, rdt, device) -> Tensor:
    """Per-frequency resistance multiplier m(f) ``[H]`` from ResistanceFrequencyModel.

    Supported multipliers:
    - ``constant`` -> its scalar value (the default; 1.0 means no skin effect).
    - ``analytic`` with ``law == "carson_skin_multiplier"`` -> the differentiable
      positive-sequence skin-effect curve (``pgml.geometry.sequence``), the Bessel
      ``I0/I1`` internal-resistance growth WITHOUT the earth-return floor; ``params``
      carry ``r1_ohm_per_m`` and ``f0_hz`` (see
      :func:`pgml.geometry.synthesis.positive_sequence_resistance_model`).
      Other analytic laws fall back to ``base_value``.
    - ``curve`` -> linear interpolation of the sampled multiplier (constant
      extrapolation outside the sampled band).
    Anything else -> 1.0 (no skin effect).
    """
    rfm = getattr(line, "resistance_frequency", None)
    if rfm is None:
        return torch.ones(f.shape[0], dtype=rdt, device=device)
    mult = rfm.multiplier
    kind = getattr(mult, "kind", None)
    if kind == "constant":
        return torch.full((f.shape[0],), float(mult.value), dtype=rdt, device=device)
    if kind == "analytic":
        if getattr(mult, "law", None) == "carson_skin_multiplier":
            from pgml.geometry.sequence import skin_resistance_multiplier

            m = skin_resistance_multiplier(
                mult.params["r1_ohm_per_m"], mult.params["f0_hz"], f
            )  # [H]
            return (float(mult.base_value) * m).to(dtype=rdt, device=device)
        return torch.full(
            (f.shape[0],), float(mult.base_value), dtype=rdt, device=device
        )
    if kind == "curve":
        fr = torch.as_tensor(mult.frequencies_hz, dtype=rdt, device=device)
        val = torch.as_tensor(mult.values, dtype=rdt, device=device)
        return _interp1d_constant_edges(f, fr, val)
    return torch.ones(f.shape[0], dtype=rdt, device=device)


def _interp1d_constant_edges(xq: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Piecewise-linear interp of ``fp`` over strictly increasing ``xp`` at ``xq`` ``[H]``.

    Constant extrapolation beyond the sampled endpoints. Pure torch (autograd-safe).
    """
    idx = torch.searchsorted(xp, xq).clamp(1, xp.shape[0] - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    t = ((xq - x0) / (x1 - x0)).clamp(0.0, 1.0)  # clamp -> constant extrapolation
    return y0 + t * (y1 - y0)


@_branch_stamp("switch")
def _switch_block_groups(
    grid, f, index, cdt, rdt, device, param_overrides, branch_states=None
):
    switches = [
        b
        for b in grid.branches
        if isinstance(b, Switch) and _branch_active(b, branch_states)
    ]
    if not switches:
        return
    by_p: dict[int, list] = {}
    for sw in switches:
        by_p.setdefault(len(sw.from_phases), []).append(sw)
    for p, group in by_p.items():
        eye = torch.eye(p, dtype=rdt, device=device)
        r_list, l_list, g_list, c_list = [], [], [], []
        for sw in group:
            r_list.append(
                _override(
                    param_overrides,
                    ("switch", sw.id, "resistance_ohm"),
                    torch.as_tensor(sw.resistance_ohm, dtype=rdt, device=device),
                )
                * eye
            )
            l_list.append(
                _override(
                    param_overrides,
                    ("switch", sw.id, "inductance_h"),
                    torch.as_tensor(sw.inductance_h, dtype=rdt, device=device),
                )
                * eye
            )
            g_list.append(
                torch.as_tensor(sw.shunt_conductance_s, dtype=rdt, device=device) * eye
            )
            c_list.append(
                torch.as_tensor(sw.shunt_capacitance_f, dtype=rdt, device=device) * eye
            )
        r = torch.stack(r_list, 0)
        ind = torch.stack(l_list, 0)
        g = torch.stack(g_list, 0)
        c = torch.stack(c_list, 0)
        ys = series_admittance_matrix(r, ind, f, cdt)
        series_block = pi_series_blocks(ys)
        y_sh = shunt_admittance_matrix(g, c, f, cdt)
        half = 0.5 * y_sh
        zeros = torch.zeros_like(half)
        shunt_block = torch.cat(
            [torch.cat([half, zeros], dim=-1), torch.cat([zeros, half], dim=-1)], dim=-2
        )
        block = series_block + shunt_block
        rows, cols = _series_terminal_indices(group, index, device)
        yield group, block, rows, cols


@_branch_stamp("generic_branch")
def _generic_branch_block_groups(
    grid, f, index, cdt, rdt, device, param_overrides, branch_states=None
):
    branches = [
        b
        for b in grid.branches
        if isinstance(b, GenericBranch) and _branch_active(b, branch_states)
    ]
    if not branches:
        return
    by_p: dict[int, list] = {}
    for gb in branches:
        by_p.setdefault(len(gb.from_phases), []).append(gb)
    for p, group in by_p.items():
        zeros_pp = torch.zeros((p, p), dtype=rdt, device=device)
        r_list, l_list, cfrom_list, cto_list = [], [], [], []
        for gb in group:
            r_list.append(
                _override(
                    param_overrides,
                    ("generic_branch", gb.id, "series_resistance_ohm"),
                    _real_matrix(gb.series_resistance_ohm, rdt, device),
                )
            )
            l_list.append(
                _override(
                    param_overrides,
                    ("generic_branch", gb.id, "series_inductance_h"),
                    _real_matrix(gb.series_inductance_h, rdt, device),
                )
            )
            cfrom_list.append(
                _real_matrix(gb.shunt_capacitance_from_f, rdt, device)
                if gb.shunt_capacitance_from_f is not None
                else zeros_pp
            )
            cto_list.append(
                _real_matrix(gb.shunt_capacitance_to_f, rdt, device)
                if gb.shunt_capacitance_to_f is not None
                else zeros_pp
            )
        r = torch.stack(r_list, 0)
        ind = torch.stack(l_list, 0)
        cfrom = torch.stack(cfrom_list, 0)
        cto = torch.stack(cto_list, 0)
        gzero = torch.zeros_like(cfrom)
        ys = series_admittance_matrix(r, ind, f, cdt)
        series_block = pi_series_blocks(ys)
        y_from = shunt_admittance_matrix(gzero, cfrom, f, cdt)  # [H,K,P,P]
        y_to = shunt_admittance_matrix(gzero, cto, f, cdt)
        zeros = torch.zeros_like(y_from)
        shunt_block = torch.cat(
            [torch.cat([y_from, zeros], dim=-1), torch.cat([zeros, y_to], dim=-1)],
            dim=-2,
        )
        block = series_block + shunt_block
        rows, cols = _series_terminal_indices(group, index, device)
        yield group, block, rows, cols


# ---- pure-shunt stamps ----------------------------------------------------
def _shunt_node_indices(elements, index, device, *, terminal: str = "from"):
    """Global row indices ``[K, P]`` for a list of single-terminal shunt stamps."""
    idx_rows: list[list[int]] = []
    for el in elements:
        if terminal == "from":
            node, phases = el.from_node, el.from_phases
        elif terminal == "node":
            node, phases = el.node, el.phases
        else:
            node, phases = el.to_node, el.to_phases
        idx_rows.append([index.row(node, ph) for ph in phases])
    rows = torch.as_tensor(idx_rows, dtype=torch.int64, device=device)
    return rows, rows


@_branch_stamp("shunt_reactor", single_terminal=True)
def _shunt_reactor_block_groups(
    grid, f, index, cdt, rdt, device, param_overrides, branch_states=None
):
    reactors = [
        b
        for b in grid.branches
        if isinstance(b, ShuntReactor) and _branch_active(b, branch_states)
    ]
    if not reactors:
        return
    # Group by phase count AND by whether the reactor has an inductive path: the
    # inductive term needs a matrix inverse, so it cannot share a batch with a
    # purely capacitive/conductive shunt.
    by_p: dict[tuple, list] = {}
    for sr in reactors:
        by_p.setdefault((len(sr.from_phases), sr.inductance_h is not None), []).append(
            sr
        )
    for (_p, inductive), group in by_p.items():
        g_list, c_list = [], []
        for sr in group:
            g_list.append(_real_matrix(sr.conductance_s, rdt, device))
            c_list.append(_real_matrix(sr.capacitance_f, rdt, device))
        g = torch.stack(g_list, 0)
        c = torch.stack(c_list, 0)
        ind = (
            torch.stack([_real_matrix(sr.inductance_h, rdt, device) for sr in group], 0)
            if inductive
            else None
        )
        block = shunt_admittance_matrix(g, c, f, cdt, ind=ind)  # [H,K,P,P]
        rows, cols = _shunt_node_indices(group, index, device, terminal="from")
        yield group, block, rows, cols


def _stamp_shunt_appliances(grid, f, y, index, cdt, rdt, device, param_overrides):
    """Stamp every in-service :class:`ShuntAppliance` (fixed linear shunt).

    Each per-phase element is ``y(h) = G + j·2π h f0 C + 1/(j·2π h f0 L)`` (the
    inductive term only where ``inductance_h`` is set — an inductive shunt's
    susceptance magnitude falls as ``1/h`` where a capacitive one rises as ``h``).
    WYE (default): each element connects its phase to ground (the historical diagonal
    stamp). DELTA: element ``k`` connects phase ``k`` to phase ``(k+1) % n`` cyclic,
    stamped ``M^T diag(y) M`` via the same :func:`cyclic_delta_incidence` convention the
    DELTA load uses (``+y`` on both leg diagonals, ``-y`` off-diagonal). Both are
    frequency-correct at every harmonic order and differentiable w.r.t. ``G``/``C``/``L``
    (the incidence ``M`` is a topology constant).
    """
    shunts = [
        a for a in grid.appliances if isinstance(a, ShuntAppliance) and a.in_service
    ]
    if not shunts:
        return y
    by_key: dict[tuple, list] = {}
    for sh in shunts:
        by_key.setdefault(
            (sh.connection, len(sh.phases), sh.inductance_h is not None), []
        ).append(sh)
    for (conn, p, inductive), group in by_key.items():
        rows, cols = _shunt_node_indices(group, index, device, terminal="node")
        if conn == WindingConnection.DELTA:
            m_c = cyclic_delta_incidence(p, rdt, device).to(cdt)  # [p, p]
            g = torch.stack(
                [
                    torch.as_tensor(sh.conductance_s, dtype=rdt, device=device)
                    for sh in group
                ],
                0,
            )  # [K, p]
            c = torch.stack(
                [
                    torch.as_tensor(sh.capacitance_f, dtype=rdt, device=device)
                    for sh in group
                ],
                0,
            )  # [K, p]
            two_pi_f = (2.0 * torch.pi) * f  # [H]
            b = two_pi_f[:, None, None] * c[None]  # [H, K, p] (B = 2*pi*f*C)
            g_b = g[None].expand_as(b)  # [H, K, p]
            y_elem = torch.complex(g_b.to(rdt), b.to(rdt)).to(cdt)  # [H, K, p]
            if inductive:
                ind = torch.stack(
                    [
                        torch.as_tensor(sh.inductance_h, dtype=rdt, device=device)
                        for sh in group
                    ],
                    0,
                )  # [K, p]
                z_l = torch.complex(
                    torch.zeros_like(b).to(rdt),
                    (two_pi_f[:, None, None] * ind[None]).to(rdt),
                ).to(cdt)  # j*2*pi*f*L  [H, K, p]
                y_elem = y_elem + 1.0 / z_l
            # Y_block = M^T diag(y_elem) M  -> [H, K, p, p].
            block = torch.einsum("ei,hke,ej->hkij", m_c, y_elem, m_c)
        else:  # WYE (phase-to-ground) — historical diagonal stamp
            g_list, c_list = [], []
            for sh in group:
                g_list.append(
                    torch.diag(
                        torch.as_tensor(sh.conductance_s, dtype=rdt, device=device)
                    )
                )
                c_list.append(
                    torch.diag(
                        torch.as_tensor(sh.capacitance_f, dtype=rdt, device=device)
                    )
                )
            g = torch.stack(g_list, 0)
            c = torch.stack(c_list, 0)
            ind = (
                torch.stack(
                    [
                        torch.diag(
                            torch.as_tensor(sh.inductance_h, dtype=rdt, device=device)
                        )
                        for sh in group
                    ],
                    0,
                )
                if inductive
                else None
            )
            block = shunt_admittance_matrix(g, c, f, cdt, ind=ind)
        y = scatter_blocks_into(y, block, rows, cols)
    return y


# ---- source Thevenin -> Norton shunt --------------------------------------
def _source_series_admittance(group, f, cdt, rdt, device, param_overrides):
    """Stacked source series admittance ``Y_s = Z_s^{-1}`` ``[H, K, P, P]``.

    One override-aware R/L stack for a group of same-phase-count sources. Shared
    by the shunt stamp and the Norton-EMF injection so the admittance the source
    presents to the network and the one driving its EMF cannot drift apart.
    """
    r_list, l_list = [], []
    for s in group:
        r_list.append(
            _override(
                param_overrides,
                ("source", s.id, "resistance_ohm"),
                _real_matrix(s.resistance_ohm, rdt, device),
            )
        )
        l_list.append(
            _override(
                param_overrides,
                ("source", s.id, "inductance_h"),
                _real_matrix(s.inductance_h, rdt, device),
            )
        )
    r = torch.stack(r_list, 0)  # [K,P,P]
    ind = torch.stack(l_list, 0)
    return series_admittance_matrix(r, ind, f, cdt)


def _stamp_sources(grid, f, y, index, cdt, rdt, device, param_overrides):
    sources = [a for a in grid.appliances if isinstance(a, Source) and a.in_service]
    if not sources:
        return y
    by_p: dict[int, list] = {}
    for s in sources:
        by_p.setdefault(len(s.phases), []).append(s)
    for _p, group in by_p.items():
        ys = _source_series_admittance(group, f, cdt, rdt, device, param_overrides)
        rows, cols = _shunt_node_indices(group, index, device, terminal="node")
        y = scatter_blocks_into(y, ys, rows, cols)
    return y


# ---- const-Z load / generator ---------------------------------------------
def _const_z_frequency_scaling(y_elem, f, f0: float, cdt) -> Tensor:
    """Scale a const-Z element admittance ``[*b, K, P]`` over frequency -> ``[*b, H, K, P]``.

    The conductance stays flat; the susceptance scales as the reactive element it
    represents at the operating point (``B*h`` where it is capacitive, ``B/h`` where it
    is inductive). The two branches are selected with ``clamp``, so the result is exact
    at ``h = 1`` and differentiable in the operating-point power.
    """
    rdt = _rdtype(cdt)
    h = (f / f0).to(rdt).reshape(-1)[:, None, None]  # [H,1,1]
    g = y_elem.real.unsqueeze(-3)  # [*b,1,K,P]
    b0 = y_elem.imag.unsqueeze(-3)  # [*b,1,K,P]
    b = torch.clamp(b0, min=0.0) * h + torch.clamp(b0, max=0.0) / h
    return torch.complex(g.expand_as(b), b).to(cdt)


def _stamp_const_z_loads(
    grid, f, y, index, cdt, rdt, device, operating_point, param_overrides, asymmetric
):
    """Fold each Load/Generator as a connection-aware const-Z shunt.

    The internal per-ELEMENT admittance at the operating point is
    ``y_elem = conj(P_k + jQ_k)/|V0|^2`` and is mapped to a nodal block
    ``M^T diag(y_elem) M`` via the terminal incidence ``M`` (``_incidence``):
    WYE-to-ground reduces to ``M = I`` (the historical diagonal stamp, reproduced
    exactly);
    WYE-with-neutral uses ``[I|-1]`` (the neutral row receives the phase return);
    DELTA-3 uses the circulant difference. ``asymmetric=False`` forces the equal split
    inside ``resolve_operating_power``. ``V0`` is L-N for WYE and L-L for DELTA
    (:func:`phase_voltage_magnitude`).

    Frequency dependence: the CONDUCTANCE ``G = P/|V0|^2`` is frequency-flat (a
    resistance), while the SUSCEPTANCE ``B = -Q/|V0|^2`` is the equivalent reactive
    element at the operating point and scales like it::

        B(h) = B(f0) * h    where B(f0) > 0  (capacitive, Q < 0 -> a fixed C)
        B(h) = B(f0) / h    where B(f0) < 0  (inductive,  Q > 0 -> a fixed L)

    which is the parallel R-L / R-C branch of the classical harmonic load model. The
    operating point is preserved EXACTLY at ``h = 1`` (both forms reduce to ``B(f0)``),
    so the fundamental load flow is unchanged; the split by the sign of ``B`` is
    built from ``clamp`` so it stays differentiable in P and Q. Frequencies must be
    positive (an inductive shunt diverges at DC).
    """
    loads = [
        a for a in grid.appliances if isinstance(a, InjectionAppliance) and a.in_service
    ]
    if not loads:
        return y
    node_map = {nd.id: nd for nd in grid.nodes}
    for grp in group_appliances(loads, node_map):
        is_delta = grp.connection == WindingConnection.DELTA
        m = build_incidence(grp, rdt, device)  # [n_elem, n_used] real
        m_c = m.to(cdt)
        elem_list = []
        for a in grp.appliances:
            node = node_map[a.node]
            v0 = phase_voltage_magnitude(
                node.u_rated_v, len(node.phases), line_to_line=is_delta
            )
            sign = 1.0 if isinstance(a, Load) else -1.0
            p_pp, q_pp = resolve_operating_power(
                a, operating_point, asymmetric=asymmetric
            )
            # Per-element admittance y_elem = conj(P+jQ)/|V0|^2  -> [*b, n_elem]
            # (a batched operating point carries leading scenario dims).
            y_elem = const_z_shunt_admittance(p_pp, q_pp, v0, sign, cdt, device)
            elem_list.append(y_elem)
        # Broadcast the per-device leading batch dims to a common shape before
        # stacking (a device without a batched override broadcasts its nominal),
        # then stack devices at -2: [*b, K, n_elem].
        lead = torch.broadcast_shapes(*[t.shape[:-1] for t in elem_list])
        y_elem_k = torch.stack(
            [t.broadcast_to(*lead, t.shape[-1]) for t in elem_list], -2
        )
        # Frequency-dependent element admittance: [*b, H, K, n_elem].
        y_elem_hk = _const_z_frequency_scaling(
            y_elem_k, f, float(grid.base_frequency_hz), cdt
        )
        # Y_block = M^T diag(y_elem) M  -> [*b, H, K, n_used, n_used].
        block = torch.einsum("ei,...ke,ej->...kij", m_c, y_elem_hk, m_c)
        rows = used_rows(grp, index, device)  # [K, n_used]
        y = scatter_blocks_into(y, block, rows, rows)
    return y


# ---- transformer (vector-group winding-incidence primitive) ---------------
@_branch_stamp("transformer")
def _transformer_block_groups(
    grid, f, index, cdt, rdt, device, param_overrides, branch_states=None
):
    """Yield ``(group, block, rows, cols)`` for every transformer group.

    Two-winding transformer primitive: vector-group winding-incidence block. The
    nodal block is built in the winding-voltage domain and mapped to the bus phase
    rows by a constant real incidence ``N`` (``Y_node = Nᵀ Y_winding N``), so the
    winding connections (wye / grounded-wye / delta) and the clock / vector group
    are modelled explicitly. A delta winding blocks the zero sequence (traps triplen
    / residual harmonics) and supplies the intrinsic ``√3`` magnitude and ``±30°``
    phase shift; the nominal turns ratio comes from the rated voltages + connections,
    so ``tap.ratio_magnitude`` is the OFF-NOMINAL tap only. See
    :mod:`pgml.assembly._transformer` for the full derivation.

    - Leakage admittance ``y_se = (R + jX(f))^-1`` (X(f)=2πfL), referred to the
      TO-side (LV) coil, carried through the incidence transform. A 3-phase unit whose
      ZERO-sequence leakage differs from its positive-sequence one (an explicit
      ``Transformer.zero_sequence``, or a non-unit ``transformer.zero_sequence.*``
      default) carries a per-phase leakage MATRIX instead of a scalar: the
      symmetric-component split ``Z_self=(Z0+2·Z1)/3``, ``Z_mutual=(Z0−Z1)/3``
      (:func:`pgml.assembly._transformer.sequence_leakage_matrices`). The zero-sequence
      PATH still comes from the winding topology — a delta or zigzag winding blocks it
      whatever the value is — so a YNyn three-limb core and a grounding zigzag now carry
      their true Z0, while ``Z0 == Z1`` keeps the scalar stamp (with the matrix form
      reproducing it to 3e-16 relative, measured on a YNyn unit).
    - Winding-resistance frequency law: ``R(f) = R · m(f) · (f/f0 if
      harmonic_xr_constant else 1)``. ``m(f)`` is the shared
      :class:`~pgml.schemas.grid_schema.ResistanceFrequencyModel` multiplier (constant,
      the Carson skin law, or a sampled curve — the same helper the line path uses);
      ``harmonic_xr_constant`` is OpenDSS's ``XRConst``, which holds X/R constant with
      frequency by scaling R with the order. Which transformers follow it is the
      documented ``transformer.harmonic_resistance.law`` choice (``element`` = the
      per-transformer flag, the default; ``constant`` / ``xr_constant`` force one law).
      Both default to no change (``m = 1``, ``XRConst = No``), i.e. X ∝ h at fixed R.
    - Magnetizing shunt ``y_m = G_m + jB_m`` added to a TERMINAL phase diagonal
      directly (outside the incidence transform), referred to the HV line voltage as
      stored. The terminal is the documented modeling choice
      ``transformer.magnetizing_placement``: ``from_terminal`` (the HV
      diagonal), ``to_terminal`` (the LV diagonal through the squared rated-voltage
      ratio — OpenDSS's own placement) or ``split`` (half on each —
      power-grid-model's). The three are different topologies: they differ in whether
      the magnetizing current sees a winding's leakage drop.

    Differentiable w.r.t. series R, L, the zero-sequence R0/L0 and the off-nominal tap
    magnitude; the discrete vector-group connections / clock select the constant
    incidence ``N``. ``param_overrides`` keys: ``("transformer", id,
    "series_resistance_ohm" | "series_inductance_h" | "tap_magnitude" |
    "zero_sequence_resistance_ohm" | "zero_sequence_inductance_h")``.

    Shared primitive builder for the Y-bus stamp (which scatters each block) and
    :func:`branch_currents` (which multiplies it with the terminal voltage). The
    block is the full ``[H, K, 2P, 2P]`` vector-group winding-incidence primitive
    plus the magnetizing shunt on the HV diagonal.
    """
    xfmrs = [
        b
        for b in grid.branches
        if isinstance(b, Transformer) and _branch_active(b, branch_states)
    ]
    if not xfmrs:
        return
    # Group transformers sharing one incidence N (same connection pair, clock, P).
    by_key: dict[tuple, list] = {}
    vgs: dict[int, object] = {}
    for t in xfmrs:
        p_t = len(t.from_phases)
        vg = resolve_vector_group(t, n_phases=p_t)
        vgs[id(t)] = vg
        key = _xfmr_group_key(vg, p_t, _xfmr_is_sequence_aware(t, p_t))
        by_key.setdefault(key, []).append(t)

    two_pi_f = (2.0 * torch.pi) * f  # [H]
    two_pi_f0 = 2.0 * torch.pi * float(grid.base_frequency_hz)
    placement = magnetizing_placement()
    resistance_law = harmonic_resistance_law()
    for (_fk, _tk, _clock, p, sequence_aware), group in by_key.items():
        vg0 = vgs[id(group[0])]

        yse_list, ratio_list, ym_list, nline_list = [], [], [], []
        zr_list, zl_list, rmult_list = [], [], []  # sequence-aware matrix path
        for t in group:
            vg = vgs[id(t)]
            r = _override(
                param_overrides,
                ("transformer", t.id, "series_resistance_ohm"),
                torch.as_tensor(t.series_resistance_ohm, dtype=rdt, device=device),
            )
            ell = _override(
                param_overrides,
                ("transformer", t.id, "series_inductance_h"),
                torch.as_tensor(t.series_inductance_h, dtype=rdt, device=device),
            )
            # Winding-resistance frequency law (the same `ResistanceFrequencyModel`
            # multiplier the line path uses) times the OpenDSS `XRConst` law: with
            # `harmonic_xr_constant` the resistance scales with the order so X/R stays
            # constant, instead of X growing with h at fixed R.
            rmult = _resistance_multiplier(t, f, rdt, device)  # [H]
            if resistance_scales_with_order(t, resistance_law):
                rmult = rmult * (f / float(grid.base_frequency_hz))
            if sequence_aware:
                # Zero-sequence leakage VALUE on the topology-derived zero-sequence
                # PATH: the per-phase leakage becomes the symmetric-component matrix,
                # carried through the same winding-incidence transform.
                r0_default, l0_default = zero_sequence_leakage(t, r, ell, two_pi_f0)
                r0 = _override(
                    param_overrides,
                    ("transformer", t.id, "zero_sequence_resistance_ohm"),
                    r0_default,
                )
                l0 = _override(
                    param_overrides,
                    ("transformer", t.id, "zero_sequence_inductance_h"),
                    l0_default,
                )
                r_mat, l_mat = sequence_leakage_matrices(r, ell, r0, l0, p)
                zr_list.append(r_mat)
                zl_list.append(l_mat)
                rmult_list.append(rmult)
            else:
                x = two_pi_f * ell  # [H]
                r_f = (r * rmult).to(rdt)  # [H]
                z = torch.complex(r_f.expand_as(x), x.to(rdt)).to(cdt)  # [H]
                yse_list.append(1.0 / z)  # [H]

            tap_mag = _override(
                param_overrides,
                ("transformer", t.id, "tap_magnitude"),
                torch.as_tensor(t.tap.ratio_magnitude, dtype=rdt, device=device),
            )
            u_from = torch.as_tensor(t.u_rated_from_v, dtype=rdt, device=device)
            u_to = torch.as_tensor(t.u_rated_to_v, dtype=rdt, device=device)
            # Rated LINE-voltage ratio: refers the magnetizing shunt (stored on the
            # from/HV side) to the to/LV terminal for the to_terminal / split placements.
            nline_list.append(u_from / u_to)
            if p == 1:
                # Single-phase / positive-sequence equivalent: the vector group is
                # folded into a complex line-to-line ratio (magnitude n_LL, exact
                # phase shift), the textbook off-nominal-tap pi. The exact shift
                # honours arbitrary phase-shifter angles that no 3-phase winding
                # topology can realise.
                theta = math.radians(vg.shift_exact_deg)
                rot = torch.complex(
                    torch.as_tensor(math.cos(theta), dtype=rdt, device=device),
                    torch.as_tensor(math.sin(theta), dtype=rdt, device=device),
                )
                ratio_list.append((u_from / u_to * tap_mag).to(cdt) * rot)  # scalar
            else:
                # Phase-domain coil turns ratio (real; √3 + clock come from N).
                ratio_list.append(
                    (nominal_turns_ratio(vg, u_from, u_to) * tap_mag).to(cdt)
                )

            gm = torch.as_tensor(t.magnetizing_conductance_s, dtype=rdt, device=device)
            lm = t.magnetizing_inductance_h
            lm_t = torch.as_tensor(
                lm if lm is not None else math.inf, dtype=rdt, device=device
            )
            bm = -1.0 / (two_pi_f * lm_t)  # [H]; lm=inf -> 0
            ym_list.append(torch.complex(gm.expand_as(bm), bm).to(cdt))  # [H]

        if sequence_aware:
            # [H,K,P,P] = ((R(f) + j*2*pi*f*L)^-1) per phase pair (matrix inverse).
            y_se = series_admittance_matrix(
                torch.stack(zr_list, 0),
                torch.stack(zl_list, 0),
                f,
                cdt,
                r_mult=torch.stack(rmult_list, 1)[:, :, None, None],  # [H,K,1,1]
            )
        else:
            y_se = torch.stack(yse_list, dim=1)  # [H,K]
        ratio = torch.stack(ratio_list, dim=0)  # [K]
        if p == 1:
            # The schema stores the leakage referred to the TO-side COIL; the
            # scalar pi consumes the line-to-line equivalent. They coincide for
            # a wye / zigzag TO winding; a delta TO coil carries 3x the
            # line-to-line impedance (y_LL = 3·y_coil) — the same identity the
            # 3-phase incidence realises through Mᵀ M.
            k_ll = 3.0 if vg0.to_side.kind == "delta" else 1.0
            block = _scalar_tap_blocks(k_ll * y_se, ratio)  # [H,K,2,2]
        else:
            n_block = block_incidence(vg0, p, rdt, device)  # [2P,2P]
            block = winding_leakage_block(y_se, ratio, n_block)  # [H,K,2P,2P]

        # Magnetizing shunt on a terminal diagonal (outside the incidence transform);
        # `transformer.magnetizing_placement` picks the terminal (see `_transformer`).
        ym = torch.stack(ym_list, dim=1)  # [H,K]
        n_line = torch.stack(nline_list, dim=0)  # [K]
        block = block + magnetizing_blocks(ym, n_line, p, placement)

        rows, cols = _series_terminal_indices(group, index, device)
        yield group, block, rows, cols


def _scalar_tap_blocks(y_se: Tensor, t: Tensor) -> Tensor:
    """Off-nominal complex-tap pi for a single-phase / positive-sequence unit.

    ``y_se`` ``[H, K]`` leakage admittance (LV-referred), ``t`` ``[K]`` complex
    ratio ``n·e^{jθ}``. Returns the ``[H, K, 2, 2]`` primitive::

        [[ y/|t|² , −y/conj(t) ],
         [ −y/t   ,     y      ]]
    """
    t_c = t[None, :]  # [1,K]
    y_ff = y_se / (t_c * torch.conj(t_c))  # HV-HV
    y_ft = -y_se / torch.conj(t_c)  # HV-LV
    y_tf = -y_se / t_c  # LV-HV
    y_tt = y_se  # LV-LV
    top = torch.stack([y_ff, y_ft], dim=-1)  # [H,K,2]
    bot = torch.stack([y_tf, y_tt], dim=-1)
    return torch.stack([top, bot], dim=-2)  # [H,K,2,2]


# ---------------------------------------------------------------------------
# branch terminal currents
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BranchCurrent:
    """Per-branch terminal currents derived from solved node voltages.

    The currents are computed from the SAME primitive admittance block ``Yprim``
    that the Y-bus stamp scatters into the global ``Y``: ``I_term = Yprim @ V_term``
    with ``V_term = concat(V[from_rows], V[to_rows])``. Sign convention matches
    :class:`~pgml.schemas.result_schema.BranchResult`: positive is current flowing
    INTO the branch at each terminal.

    Attributes
    ----------
    branch_id:
        Id of the source :class:`~pgml.schemas.grid_schema.BranchBase` branch.
    from_node, to_node:
        Terminal node ids. ``to_node`` is ``None`` for a single-terminal shunt
        (:class:`~pgml.schemas.grid_schema.ShuntReactor`).
    from_phases, to_phases:
        The branch's terminal phase tuples (``to_phases`` empty for a shunt).
    i_from:
        Complex ``[*batch, H, Pf]`` — current into the FROM terminal.
    i_to:
        Complex ``[*batch, H, Pt]`` — current into the TO terminal (all zeros for a
        single-terminal shunt).
    """

    branch_id: int
    from_node: int
    to_node: Optional[int]
    from_phases: tuple[Phase, ...]
    to_phases: tuple[Phase, ...]
    i_from: Tensor
    i_to: Tensor


def _terminal_currents_from_block(v: Tensor, block: Tensor, rows: Tensor) -> Tensor:
    """Per-branch terminal currents ``I_term = Yprim @ V_term`` for a group.

    Parameters
    ----------
    v:
        Complex node voltages ``[*batch, H, N]`` (H axis already present).
    block:
        Complex primitive blocks ``[H, K, M, M]`` (the matrix the stamp scatters).
    rows:
        int64 ``[K, M]`` global row indices of each branch's terminal slots
        (from-rows then to-rows for a series branch; from-rows only for a shunt).

    Returns
    -------
    Tensor
        Complex ``[*batch, H, K, M]`` terminal currents per branch.
    """
    k, m = rows.shape
    flat_rows = rows.reshape(-1)  # [K*M]
    v_term = v.index_select(-1, flat_rows)  # [*batch, H, K*M]
    v_term = v_term.reshape(*v.shape[:-1], k, m)  # [*batch, H, K, M]
    # I_term[..., k, i] = sum_j block[..., H, k, i, j] V_term[..., H, k, j]
    # (the block may carry leading scenario dims from batched branch states).
    return torch.einsum("...hkij,...hkj->...hki", block, v_term)


def branch_currents(
    grid: Grid,
    v: Tensor,
    frequencies_hz,
    index: NodePhaseIndex,
    *,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    param_overrides: Optional[dict] = None,
    branch_states: Optional[dict] = None,
    fusion: Optional[FusionMap] = None,
    i_inj: Optional[Tensor] = None,
) -> list[BranchCurrent]:
    """Per-branch terminal currents from solved node voltages.

    For every in-service :class:`~pgml.schemas.grid_schema.BranchBase` branch
    (Line — explicit / geometry / sequence-aware —, Transformer, Switch,
    GenericBranch, ShuntReactor), the terminal currents are computed from the SAME
    primitive admittance block ``Yprim`` that the Y-bus stamp scatters into the
    global ``Y``::

        V_term = concat(V[from_rows], V[to_rows])      ( = V[from_rows] for a shunt)
        I_term = Yprim @ V_term
        i_from = I_term[..., :Pf]    i_to = I_term[..., Pf:]

    Sign convention: positive current flows INTO the branch at each terminal
    (matching :class:`~pgml.schemas.result_schema.BranchResult`). Because the block
    is the exact stamp primitive, scattering ``(i_from, i_to)`` back to the node
    rows and summing over branches reproduces ``Y @ V`` at every row (this is the
    KCL consistency check).

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.schemas.grid_schema.Grid` (all ``type_ref``
        already expanded).
    v:
        Solved complex node voltages ``[*batch, H, N]`` (``N == index.size``,
        ``H == len(frequencies_hz)``). A missing batch dim is allowed; a missing H
        axis (``[*batch, N]`` / ``[N]``) is broadcast over H.
    frequencies_hz:
        1-D real tensor / sequence of ``H`` absolute frequencies (Hz), the same
        frequencies the voltages were solved at.
    index:
        The compact :class:`NodePhaseIndex` describing ``v``'s row layout.
    dtype:
        Complex working dtype (``complex128`` for gradcheck; ``complex64`` ok).
    device:
        Target device; defaults to ``v``'s device. No device is hard-coded.
    param_overrides:
        Optional differentiability hook (see :func:`assemble_ybus`): the same
        ``(kind, id, field) -> leaf tensor`` keys the stamps use, so gradients can
        flow to physical params without mutating the schema.
    branch_states:
        Optional topology / switch-state mask ``{branch_id: state}`` — MUST match
        the value the voltages were solved with (:func:`assemble_ybus`): each
        listed branch's primitive block is scaled by its state, so an open
        (state 0) branch reports zero current and a batched state yields
        per-scenario currents.
    fusion:
        The :class:`~pgml.assembly._fusion.FusionMap` the solve used, if any. A FUSED
        branch has no primitive block to multiply, so its current comes from
        Kirchhoff's law at the fused node instead: the currents of the fused branches
        meeting at a fused row carry exactly what the rest of the network leaves there.
        ``v`` and ``index`` must then be the FULL (unreduced) layout — which is what a
        result reports.
    i_inj:
        The nodal current injection ``[*batch, H, N]`` (or broadcastable) the solve
        used, in the full row layout, needed ONLY to recover the current through a
        fused branch: the devices and sources sitting on a fused node are part of that
        node's current balance. ``None`` assumes zero injection at the fused rows,
        which is exact for a fused node that carries no appliance and logs a warning
        when one does.

    Returns
    -------
    list[BranchCurrent]
        One :class:`BranchCurrent` per in-service branch, in ``grid.branches``
        order. ``i_from`` / ``i_to`` are complex ``[*batch, H, P]`` (the leading
        ``*batch`` matching ``v``'s broadcast; a scalar/length-1 frequency keeps a
        singleton H axis). For a fused branch ``i_to == -i_from`` exactly (an ideal
        conductor has no shunt path).
    """
    if device is None:
        device = v.device
    f = _as_freq_tensor(frequencies_hz, dtype, device)
    h = f.shape[0]
    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)
    n = index.size

    v = v.to(dtype=cdt, device=device)
    if v.shape[-1] != n:
        raise InputError(f"branch_currents: v last dim {v.shape[-1]} != N={n}")
    # Normalise to [*batch, H, N]: insert / broadcast a singleton H axis if absent.
    has_h = v.ndim >= 2 and v.shape[-2] == h
    if not has_h:
        v = v.unsqueeze(-2)  # [..., 1, N]
    if v.shape[-2] != h:
        v = v.expand(*v.shape[:-2], h, n)

    if fusion is not None and n != fusion.full_index.size:
        raise InputError(
            f"branch_currents: with a fusion map, v / index must be the FULL row "
            f"layout (N={fusion.full_index.size}); got N={n}. Prolong the reduced "
            "solution first (fusion.prolong)."
        )

    # Iterate the SAME branch-stamp registry the Y-bus assembly does: each builder
    # yields (group, block, rows, cols) — the exact primitive block the stamp
    # scatters — so the currents are consistent with Y to machine precision (KCL).
    # rows == cols here; the block is multiplied with the gathered terminal voltage.
    results: dict[int, BranchCurrent] = {}
    stamped = _unfused_view(grid, fusion)

    for stamp in _BRANCH_STAMPS:
        for group, block, rows, _cols in stamp.builder(
            stamped, f, index, cdt, rdt, device, param_overrides, branch_states
        ):
            block = _masked_block(
                block, _group_states(group, branch_states, rdt, device)
            )
            i_term = _terminal_currents_from_block(v, block, rows)  # [*b,H,K,M]
            if stamp.single_terminal:
                # One-terminal shunt: the whole block is the FROM current; no TO half.
                zero_to = torch.zeros_like(i_term[..., 0, 0:0])  # [*b,H,0]
                for k, b in enumerate(group):
                    results[b.id] = BranchCurrent(
                        branch_id=b.id,
                        from_node=b.from_node,
                        to_node=None,
                        from_phases=tuple(b.from_phases),
                        to_phases=(),
                        i_from=i_term[..., k, :],
                        i_to=zero_to,
                    )
                continue
            p = len(group[0].from_phases)
            i_from = i_term[..., :, :p]  # [*b,H,K,Pf]
            i_to = i_term[..., :, p:]  # [*b,H,K,Pt]
            for k, b in enumerate(group):
                results[b.id] = BranchCurrent(
                    branch_id=b.id,
                    from_node=b.from_node,
                    to_node=b.to_node,
                    from_phases=tuple(b.from_phases),
                    to_phases=tuple(b.to_phases),
                    i_from=i_from[..., k, :],
                    i_to=i_to[..., k, :],
                )

    if fusion is not None and fusion.fused_branch_ids:
        results.update(
            _fused_branch_results(
                grid,
                v,
                f,
                index,
                fusion,
                i_inj,
                cdt,
                rdt,
                device,
                dtype,
                param_overrides,
                branch_states,
            )
        )

    # Emit in grid.branches order over the stamped BranchBase branches (every
    # in-service branch, plus any branch listed in ``branch_states``).
    return [results[b.id] for b in grid.branches if b.id in results]


def _fused_branch_results(
    grid,
    v,
    f,
    index,
    fusion,
    i_inj,
    cdt,
    rdt,
    device,
    dtype,
    param_overrides,
    branch_states,
) -> dict[int, BranchCurrent]:
    """The :class:`BranchCurrent` of every FUSED branch, from KCL at the fused rows.

    The defect the fused branches have to carry is what the rest of the network leaves
    at each fused row::

        defect = i_inj - Y_network_without_fused_branches @ V

    built in the FULL row layout (hence one assembly of the unfused network — the fused
    rows' individual balances are exactly the information the reduced system sums away),
    then inverted per fused group by the structural maps of the fusion map. Sign
    convention as everywhere: positive current flows INTO the branch terminal, so the
    TO terminal of an ideal conductor carries ``-i_from``.
    """
    y_nf = assemble_network_ybus(
        _unfused_view(grid, fusion),
        f,
        dtype=dtype,
        device=device,
        param_overrides=param_overrides,
        branch_states=branch_states,
        fusion=False,
    ).Y  # [H, N, N] (or [*batch, H, N, N] with batched states), FULL layout
    if y_nf.ndim == 2:
        y_nf = y_nf.unsqueeze(0)
    defect = -torch.matmul(y_nf, v.unsqueeze(-1)).squeeze(-1)  # [*batch, H, N]
    if i_inj is not None:
        defect = defect + i_inj.to(dtype=cdt, device=device)
    elif _fused_rows_carry_injection(grid, fusion):
        _log.warning(
            "branch_currents: the fused node(s) %s carry an injecting appliance or a "
            "source, whose current is part of their current balance, but no i_inj was "
            "given; the current reported for the fused branch(es) omits it. Pass the "
            "solve's nodal injection as i_inj.",
            [nid for grp in fusion.node_groups() for nid, _ph in grp][:8],
        )
    recovered = fused_branch_currents(fusion, defect)
    out: dict[int, BranchCurrent] = {}
    by_id = {int(b.id): b for b in grid.branches}
    for bid, per_phase in recovered.items():
        b = by_id[bid]
        i_from = torch.stack(
            [per_phase[p] for p in range(len(b.from_phases))], dim=-1
        )  # [*batch, H, P]
        out[bid] = BranchCurrent(
            branch_id=bid,
            from_node=b.from_node,
            to_node=b.to_node,
            from_phases=tuple(b.from_phases),
            to_phases=tuple(b.to_phases),
            i_from=i_from,
            i_to=-i_from,
        )
    return out


def _fused_rows_carry_injection(grid, fusion) -> bool:
    """Does any fused node-phase row host an in-service injecting appliance / source?"""
    fused_nodes = {nid for grp in fusion.node_groups() for nid, _ph in grp}
    for a in grid.appliances:
        if not getattr(a, "in_service", True):
            continue
        if isinstance(a, (InjectionAppliance, Source)) and int(a.node) in fused_nodes:
            return True
    return False


# ---------------------------------------------------------------------------
# per-branch primitive stamps (the incidence structure of a branch in Y)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BranchStampBlock:
    """One branch's primitive admittance block and the Y rows it occupies.

    A branch enters the Y-bus ONLY as ``Y[rows, rows] += block`` (the stamp the
    assembly scatters, :func:`assemble_network_ybus`), so this pair is the complete
    description of that branch's contribution — the structure that makes a change
    of a single branch's admittance a LOW-RANK modification of ``Y`` (rank ≤ ``M``,
    ``M = 2P`` for a two-terminal branch and ``P`` for a single-terminal shunt).

    Attributes
    ----------
    branch_id:
        Id of the source :class:`~pgml.schemas.grid_schema.BranchBase` branch.
    kind:
        Registered stamp kind (``"line"``, ``"switch"``, ``"transformer"``,
        ``"generic_branch"``, ``"shunt_reactor"``).
    block:
        Complex ``[H, M, M]`` primitive admittance (per frequency), UNSCALED by any
        branch state — exactly the matrix a state of 1 stamps.
    rows:
        int64 ``[M]`` global node-phase rows the block occupies (from-terminal rows
        then to-terminal rows; from-terminal only for a single-terminal shunt).
    single_terminal:
        ``True`` for a one-terminal shunt branch (``M == P``, no TO half).
    """

    branch_id: int
    kind: str
    block: Tensor
    rows: Tensor
    single_terminal: bool


def branch_stamp_blocks(
    grid: Grid,
    frequencies_hz,
    branch_ids,
    index: NodePhaseIndex,
    *,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    param_overrides: Optional[dict] = None,
) -> list[BranchStampBlock]:
    """The primitive admittance stamp of each named branch, with its Y rows.

    Walks the SAME branch-stamp registry as the Y-bus assembly and
    :func:`branch_currents` (no stamp physics is re-derived) and returns, per
    requested branch, the primitive block and the global rows it scatters into. A
    branch is stamped regardless of its ``in_service`` / ``closed`` flags — the
    caller decides what a state of 0 means — and the returned block is the
    UNSCALED (state 1) primitive.

    This is the structural input to a low-rank admittance update
    (:mod:`pgml.solver.lowrank`): scaling branch ``b``'s stamp by ``s`` changes
    ``Y`` by ``(s − 1)`` times its block on ``rows``, a rank-``≤ M`` term.

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.schemas.grid_schema.Grid` (``type_ref`` expanded).
    frequencies_hz:
        1-D real tensor / sequence of ``H`` absolute frequencies (Hz), or a scalar.
    branch_ids:
        Iterable of branch ids to return stamps for. Every id must name a branch of
        ``grid`` that the registry stamps.
    index:
        The compact :class:`NodePhaseIndex` of the FULL grid — the row layout the
        returned ``rows`` refer to.
    dtype, device, param_overrides:
        As in :func:`assemble_ybus` (``device`` defaults to the frequency tensor's,
        else CPU; ``param_overrides`` injects differentiable parameter leaves).

    Returns
    -------
    list[BranchStampBlock]
        One entry per (branch, stamp group) in ``grid.branches`` order.
        Differentiable w.r.t. the branch parameters; device/dtype follow the
        arguments.
    """
    wanted = list(dict.fromkeys(int(b) for b in branch_ids))
    known = {b.id for b in grid.branches}
    missing = [b for b in wanted if b not in known]
    if missing:
        raise InputError(
            f"branch_stamp_blocks: unknown branch id(s) {missing} (not in grid.branches)."
        )
    if device is None and isinstance(frequencies_hz, Tensor):
        device = frequencies_hz.device
    f = _as_freq_tensor(frequencies_hz, dtype, device)
    device = f.device
    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)

    # Restrict the builders to the requested branches (a shallow view of the grid:
    # same nodes, same catalog, fewer branches) so the stamp work is O(len(wanted))
    # instead of O(all branches). Row indices still come from the FULL grid's index.
    wanted_set = set(wanted)
    sub = grid.model_copy(
        update={"branches": [b for b in grid.branches if b.id in wanted_set]}
    )
    # A zero-impedance branch has no primitive block, so it cannot be the rank-k term of
    # a low-rank update either; name it instead of returning an infinite stamp. The
    # static flags are forced on for the check, because every requested branch is
    # stamped whatever they say.
    forced = sub.model_copy(
        update={
            "branches": [
                b.model_copy(
                    update=(
                        {"in_service": True, "closed": True}
                        if hasattr(b, "closed")
                        else {"in_service": True}
                    )
                )
                for b in sub.branches
            ]
        }
    )
    ideal = [
        z
        for z in zero_impedance_branches(forced, param_overrides=param_overrides)
        if z.branch_id in wanted_set
    ]
    if ideal:
        raise InputError(
            "branch_stamp_blocks: "
            + describe_unfusable(ideal)
            + " A swept or low-rank-updated branch has to stay a stamped branch."
        )
    # Every requested branch is stamped, whatever its static flags say.
    active = {bid: 1.0 for bid in wanted}

    found: dict[int, BranchStampBlock] = {}
    for stamp in _BRANCH_STAMPS:
        for group, block, rows, cols in stamp.builder(
            sub, f, index, cdt, rdt, device, param_overrides, active
        ):
            if not torch.equal(rows, cols):
                raise InputError(
                    f"branch_stamp_blocks: the {stamp.kind!r} stamp scatters into "
                    "asymmetric (row != col) positions; a low-rank update needs the "
                    "symmetric row/col mapping every registered branch stamp uses."
                )
            for k, b in enumerate(group):
                if b.id in found:
                    raise InputError(
                        f"branch_stamp_blocks: branch {b.id} is stamped twice (as "
                        f"{found[b.id].kind!r} and {stamp.kind!r}); a branch must "
                        "yield exactly one primitive block for its contribution to Y "
                        "to be a single low-rank term."
                    )
                found[b.id] = BranchStampBlock(
                    branch_id=b.id,
                    kind=stamp.kind,
                    block=block.select(-3, k),  # [H, M, M]
                    rows=rows[k],  # [M]
                    single_terminal=stamp.single_terminal,
                )
    unstamped = [bid for bid in wanted if bid not in found]
    if unstamped:
        raise InputError(
            f"branch_stamp_blocks: branch id(s) {unstamped} have no registered "
            "primitive stamp (only Line / Transformer / Switch / GenericBranch / "
            "ShuntReactor branches carry one)."
        )
    return [found[b.id] for b in grid.branches if b.id in found]


# ---------------------------------------------------------------------------
# current injections
# ---------------------------------------------------------------------------
def build_injections(
    grid: Grid,
    frequencies_hz,
    index: NodePhaseIndex,
    *,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    operating_point: Optional[dict] = None,
    param_overrides: Optional[dict] = None,
) -> Tensor:
    """Norton current-source vector ``I(f)`` aligned to ``index``.

    Stamps only the source Norton current ``i_s = Y_s @ V_th`` at the source rows,
    where ``V_th`` is the per-phase Thevenin phasor ``u_ref * exp(j*u_angle)`` and
    ``Y_s = Z_s(f)^-1``. Harmonic current injections from load/generator spectra
    are computed separately by :mod:`pgml.solver.harmonic_flow`, not here — a grid
    with no in-service source contributes 0.

    Returns
    -------
    Tensor
        Complex ``[H, N]`` (``[N]`` if a scalar frequency was passed). A purely
        passive grid yields all zeros.
    """
    if device is None and isinstance(frequencies_hz, Tensor):
        device = frequencies_hz.device
    f = _as_freq_tensor(frequencies_hz, dtype, device)
    device = f.device
    h = f.shape[0]
    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)
    n = index.size

    i = torch.zeros((h, n), dtype=cdt, device=device)

    sources = [a for a in grid.appliances if isinstance(a, Source) and a.in_service]
    if sources:
        by_p: dict[int, list] = {}
        for s in sources:
            by_p.setdefault(len(s.phases), []).append(s)
        for _p, group in by_p.items():
            vth_list, row_list = [], []
            for s in group:
                u_ref = torch.as_tensor(s.u_ref_v, dtype=rdt, device=device)
                ang = torch.as_tensor(s.u_angle_deg, dtype=rdt, device=device) * (
                    math.pi / 180.0
                )
                vth_list.append(torch.polar(u_ref, ang).to(cdt))  # [P]
                row_list.append([index.row(s.node, ph) for ph in s.phases])
            vth = torch.stack(vth_list, 0)  # [K,P]
            ys = _source_series_admittance(group, f, cdt, rdt, device, param_overrides)
            # i_s = Y_s @ V_th  -> [H,K,P]
            i_s = torch.matmul(ys, vth[None, :, :, None].to(cdt)).squeeze(-1)
            rows = torch.as_tensor(row_list, dtype=torch.int64, device=device)  # [K,P]
            i = _scatter_injection(i, i_s, rows)

    scalar_freq = (
        not isinstance(frequencies_hz, Tensor)
        and not isinstance(frequencies_hz, (list, tuple))
        and h == 1
    )
    if scalar_freq:
        i = i.reshape(n)
    return i


def _scatter_injection(i: Tensor, values: Tensor, rows: Tensor) -> Tensor:
    """Accumulate ``values`` ``[*,K,P]`` into ``i`` ``[*,N]`` at ``rows`` ``[K,P]``.

    Uses ``index_add_`` along the last (node) axis: it accumulates duplicate
    targets, broadcasts the same row targets across leading batch/H dims, and is
    autograd-safe (gradients flow into ``values``).
    """
    if values.shape[-2] == 0:
        return i
    k, p = rows.shape
    flat_rows = rows.reshape(-1)  # [K*P] int64
    lead = values.shape[:-2]
    flat_vals = values.reshape(*lead, k * p)
    target_lead = torch.broadcast_shapes(i.shape[:-1], lead)
    n = i.shape[-1]
    i_full = i.broadcast_to(*target_lead, n).clone()
    flat_vals = flat_vals.broadcast_to(*target_lead, k * p)
    i_full.index_add_(-1, flat_rows, flat_vals)
    return i_full


# ---------------------------------------------------------------------------
# ZIP device current injections (voltage-dependent, nonlinear power flow)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _UncontrolledGroupPlan:
    """V-independent tensors for one incidence group of uncontrolled devices.

    Everything here depends only on the grid, the operating point, and the
    overrides — not on the voltage — so it is computed once per solve and reused
    across the fixed-point / Newton iterations (:func:`injections_from_plan`).
    """

    m_c: Tensor  # [n_elem, n_used] complex incidence
    rows: Tensor  # [K, n_used] int64 global rows (scatter targets)
    flat_rows: Tensor  # [K * n_used] int64 (gather index)
    n_used: int
    p_pp: Tensor  # [*pbatch, 1, K, n_elem] signed active power
    q_pp: Tensor  # [*qbatch, 1, K, n_elem] signed reactive power
    v0: Tensor  # [K, n_elem] nominal voltage magnitude
    zip_p: Tensor  # [K, 3] ZIP triples (z, i, p) for P
    zip_q: Tensor  # [K, 3] ZIP triples for Q


@dataclass(frozen=True)
class _ControlledAppliancePlan:
    """V-independent tensors for one inverter-controlled appliance."""

    control: object  # the schema InverterControl (drives resolve_injection_power)
    sign: float
    v0: object  # float or tensor nominal voltage magnitude
    p_avail: Tensor  # [*pb, n_elem] available power (native sign)
    m_c: Tensor  # [n_elem, n_used] complex incidence
    arow: Tensor  # [n_used] int64 global rows of THIS appliance
    n_used: int


@dataclass(frozen=True)
class InjectionPlan:
    """Precomputed :func:`device_current_injections` state (everything but ``V``).

    Resolving the operating point walks python lists of appliances, pydantic
    fields, and config defaults — cheap once, but the nonlinear solvers evaluate
    the injection at EVERY iteration (and Newton also inside its line search and
    Jacobian), where that python work dominated the solve time. The plan captures
    the V-independent tensors once (:func:`build_injection_plan`); each iteration
    then runs :func:`injections_from_plan` — pure tensor ops.

    The plan's tensors keep whatever autograd graph the inputs carry: built under
    ``torch.no_grad()`` it is a detached fast path (the fixed-point forward);
    built with gradients enabled the result stays differentiable w.r.t. the
    parameter leaves exactly like :func:`device_current_injections`.
    """

    h: int
    n: int
    cdt: torch.dtype
    device: torch.device
    uncontrolled: tuple[_UncontrolledGroupPlan, ...]
    controlled: tuple[_ControlledAppliancePlan, ...]


def build_injection_plan(
    grid: Grid,
    index: NodePhaseIndex,
    frequencies_hz,
    *,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    operating_point: Optional[dict] = None,
    param_overrides: Optional[dict] = None,
    symmetry: Optional[str] = None,
) -> InjectionPlan:
    """Precompute the V-independent part of :func:`device_current_injections`.

    Same parameters and resolution rules as :func:`device_current_injections`
    (operating-point overrides, per-phase splitting, ZIP coefficients, incidence
    grouping); returns the :class:`InjectionPlan` consumed by
    :func:`injections_from_plan`.
    """
    if device is None:
        device = torch.device("cpu")
    f = _as_freq_tensor(frequencies_hz, dtype, device)
    h = f.shape[0]
    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)
    n = index.size

    loads = [
        a for a in grid.appliances if isinstance(a, InjectionAppliance) and a.in_service
    ]
    if not loads:
        return InjectionPlan(
            h=h, n=n, cdt=cdt, device=device, uncontrolled=(), controlled=()
        )

    asymmetric = resolve_asymmetric(grid, operating_point, mode=symmetry)
    node_map = {nd.id: nd for nd in grid.nodes}
    uncontrolled = [a for a in loads if getattr(a, "control", None) is None]
    controlled = [a for a in loads if getattr(a, "control", None) is not None]

    group_plans: list[_UncontrolledGroupPlan] = []
    for grp in group_appliances(uncontrolled, node_map):
        n_elem = grp.n_elem
        is_delta = grp.connection == WindingConnection.DELTA
        m = build_incidence(grp, rdt, device)  # [n_elem, n_used] real
        m_c = m.to(cdt)

        p_list, q_list, v0_list, zipp_list, zipq_list = [], [], [], [], []
        for a in grp.appliances:
            kind = "load" if isinstance(a, Load) else "generator"
            sign = 1.0 if isinstance(a, Load) else -1.0
            node = node_map[a.node]
            v0 = phase_voltage_magnitude(
                node.u_rated_v, len(node.phases), line_to_line=is_delta
            )

            # Resolve total / per-phase honoring operating_point, keeping tensors.
            p_total, q_total = a.p_nom_w, a.q_nom_var
            p_per, q_per = a.p_nom_per_phase_w, a.q_nom_per_phase_var
            if operating_point is not None and a.id in operating_point:
                op = operating_point[a.id]
                if "p_per_phase_w" in op:
                    p_per = op["p_per_phase_w"]
                elif "p_w" in op:
                    p_total, p_per = op["p_w"], None
                if "q_per_phase_var" in op:
                    q_per = op["q_per_phase_var"]
                elif "q_var" in op:
                    q_total, q_per = op["q_var"], None
            if not asymmetric:
                # Symmetric calc: ignore the per-phase split, distribute the total
                # equally over the elements (power-grid-model rule).
                if p_per is not None:
                    p_total, p_per = _tensor_sum(p_per, rdt, device), None
                if q_per is not None:
                    q_total, q_per = _tensor_sum(q_per, rdt, device), None

            p_t = _override(
                param_overrides,
                (kind, a.id, "p_nom_per_phase_w"),
                _per_phase_power_tensor(p_total, p_per, n_elem, rdt, device),
            )
            q_t = _override(
                param_overrides,
                (kind, a.id, "q_nom_per_phase_var"),
                _per_phase_power_tensor(q_total, q_per, n_elem, rdt, device),
            )
            p_list.append(sign * p_t)  # [n_elem]
            q_list.append(sign * q_t)  # [n_elem]
            v0_t = (
                v0
                if isinstance(v0, Tensor)
                else torch.as_tensor(v0, dtype=rdt, device=device)
            )
            v0_list.append(v0_t.to(dtype=rdt, device=device).reshape(()).expand(n_elem))
            zp, zq = _zip_coeffs(a, rdt, device)
            zipp_list.append(zp)  # [3]
            zipq_list.append(zq)

        # Stack with K at dim -2 so any per-load batch dims stay leading and broadcast
        # against the [*b, H, K, n_elem] voltage tensor. A scenario sweep may vary only
        # SOME devices (e.g. loads but not generators), so the per-device entries can
        # carry different leading batch shapes; broadcast them to a common batch before
        # stacking (a device with no batched override broadcasts its nominal across the
        # batch) instead of failing the stack.
        p_lead = torch.broadcast_shapes(*[t.shape[:-1] for t in p_list])
        q_lead = torch.broadcast_shapes(*[t.shape[:-1] for t in q_list])
        p_pp = torch.stack(
            [t.broadcast_to(*p_lead, t.shape[-1]) for t in p_list], -2
        )  # [*pbatch, K, n_elem]
        q_pp = torch.stack([t.broadcast_to(*q_lead, t.shape[-1]) for t in q_list], -2)
        rows = used_rows(grp, index, device)  # [K, n_used] int64
        group_plans.append(
            _UncontrolledGroupPlan(
                m_c=m_c,
                rows=rows,
                flat_rows=rows.reshape(-1),
                n_used=grp.n_used,
                # Insert a singleton H axis so the powers broadcast over H at apply.
                p_pp=p_pp.unsqueeze(-3),  # [*pbatch, 1, K, n_elem]
                q_pp=q_pp.unsqueeze(-3),
                v0=torch.stack(v0_list, -2),  # [K, n_elem]
                zip_p=torch.stack(zipp_list, 0),  # [K, 3]
                zip_q=torch.stack(zipq_list, 0),
            )
        )

    controlled_plans: list[_ControlledAppliancePlan] = []
    for grp in group_appliances(controlled, node_map):
        is_delta = grp.connection == WindingConnection.DELTA
        m_c = build_incidence(grp, rdt, device).to(cdt)  # [n_elem, n_used]
        rows = used_rows(grp, index, device)  # [K, n_used]
        for ki, a in enumerate(grp.appliances):
            sign = 1.0 if isinstance(a, Load) else -1.0
            node = node_map[a.node]
            v0 = phase_voltage_magnitude(
                node.u_rated_v, len(node.phases), line_to_line=is_delta
            )
            p_list, _q_list = resolve_operating_power(
                a, operating_point, asymmetric=asymmetric
            )
            controlled_plans.append(
                _ControlledAppliancePlan(
                    control=a.control,
                    sign=sign,
                    v0=v0,
                    p_avail=_stack_elements(p_list, rdt, device),  # [*pb, n_elem]
                    m_c=m_c,
                    arow=rows[ki],  # [n_used]
                    n_used=grp.n_used,
                )
            )

    return InjectionPlan(
        h=h,
        n=n,
        cdt=cdt,
        device=device,
        uncontrolled=tuple(group_plans),
        controlled=tuple(controlled_plans),
    )


def flatten_plan_batch(
    plan: InjectionPlan, batch_shape: Sequence[int]
) -> InjectionPlan:
    """A copy of ``plan`` with each group's power leading batch flattened to one dim.

    The plan captures the operating point with its natural batch (e.g. a per-step
    profiled operating point is ``[B, T]``). The IFT backward builds a block-diagonal
    state Jacobian over a SINGLE flattened ``[B*T]`` scenario axis, so it evaluates the
    residual with a collapsed 1-D batch; this returns a plan whose power tensors have
    their leading ``batch_shape`` dims collapsed to ``prod(batch_shape)`` to match,
    while any BROADCAST (size-1) leading dims a group carries are preserved. A no-op
    for a group whose leading batch is scalar / already 1-D. The power tensors are
    detached constants of that differentiation, so the reshape is autograd-safe.
    """
    bshape = tuple(int(s) for s in batch_shape)
    ndim = len(bshape)
    flat = math.prod(bshape) if bshape else 1

    def _flat(power: Tensor, tail_ndim: int) -> Tensor:
        lead = tuple(power.shape[:-tail_ndim])
        # Only collapse a leading batch that MATCHES the flattened scenario batch
        # exactly (the operating-point batch); a broadcast placeholder or a smaller
        # rank is left to broadcast as-is.
        if lead == bshape and ndim > 1:
            return power.reshape(flat, *power.shape[-tail_ndim:])
        return power

    uncontrolled = tuple(
        _UncontrolledGroupPlan(
            m_c=g.m_c,
            rows=g.rows,
            flat_rows=g.flat_rows,
            n_used=g.n_used,
            p_pp=_flat(g.p_pp, 3),
            q_pp=_flat(g.q_pp, 3),
            v0=g.v0,
            zip_p=g.zip_p,
            zip_q=g.zip_q,
        )
        for g in plan.uncontrolled
    )
    controlled = tuple(
        _ControlledAppliancePlan(
            control=c.control,
            sign=c.sign,
            v0=c.v0,
            p_avail=_flat(c.p_avail, 1),
            m_c=c.m_c,
            arow=c.arow,
            n_used=c.n_used,
        )
        for c in plan.controlled
    )
    return InjectionPlan(
        h=plan.h,
        n=plan.n,
        cdt=plan.cdt,
        device=plan.device,
        uncontrolled=uncontrolled,
        controlled=controlled,
    )


def select_plan_batch(
    plan: InjectionPlan, rows: Tensor, *, batch_size: int
) -> InjectionPlan:
    """A copy of ``plan`` holding only the scenarios ``rows`` of a FLAT batch axis.

    The counterpart of :func:`flatten_plan_batch` for the other direction: where that
    collapses a multi-dimensional operating-point batch onto one axis, this picks a
    subset out of that one axis. A group whose power carries the full flat batch
    (``batch_size`` entries) is indexed; a group carrying a broadcast (size-1) or scalar
    batch is left alone, since it already applies to every scenario. ``rows`` is an
    int64 index tensor (a contiguous chunk, or a single scenario).

    Two consumers need it: the implicit-function backward, which builds the
    block-diagonal state Jacobian in batch chunks whose size a memory budget decides,
    and the criticality diagnostic, which analyses the single hardest scenario of a
    batched solve. Index-select keeps autograd history, so a differentiable plan stays
    differentiable.
    """
    bs = int(batch_size)
    rows = rows.to(dtype=torch.int64)

    def _sel(power: Tensor, tail_ndim: int) -> Tensor:
        lead = tuple(power.shape[:-tail_ndim])
        if len(lead) == 1 and int(lead[0]) == bs and bs > 1:
            return power.index_select(0, rows.to(power.device))
        return power

    uncontrolled = tuple(
        _UncontrolledGroupPlan(
            m_c=g.m_c,
            rows=g.rows,
            flat_rows=g.flat_rows,
            n_used=g.n_used,
            p_pp=_sel(g.p_pp, 3),
            q_pp=_sel(g.q_pp, 3),
            v0=g.v0,
            zip_p=g.zip_p,
            zip_q=g.zip_q,
        )
        for g in plan.uncontrolled
    )
    controlled = tuple(
        _ControlledAppliancePlan(
            control=c.control,
            sign=c.sign,
            v0=c.v0,
            p_avail=_sel(c.p_avail, 1),
            m_c=c.m_c,
            arow=c.arow,
            n_used=c.n_used,
        )
        for c in plan.controlled
    )
    return InjectionPlan(
        h=plan.h,
        n=plan.n,
        cdt=plan.cdt,
        device=plan.device,
        uncontrolled=uncontrolled,
        controlled=controlled,
    )


def injections_from_plan(plan: InjectionPlan, v: Tensor) -> Tensor:
    """Evaluate ``I_device(V)`` from a precomputed :class:`InjectionPlan`.

    The per-iteration half of :func:`device_current_injections`: pure tensor ops
    (gather, einsum, the ZIP law, scatter) with no python resolution work.
    ``v`` is complex ``[*batch, H, N]`` (or ``[*batch, N]`` / ``[N]``; a missing H
    axis is broadcast). Returns ``[*batch, H, N]`` exactly like
    :func:`device_current_injections`.

    With ``H == 1`` an explicit H axis is recognized only on a bare ``[1, N]``
    input; any deeper ``v`` is read as ``[*batch, N]``. A size-one dim at ``-2``
    is otherwise indistinguishable from a trailing scenario dim of one — an
    operating point batched ``[B, 1]`` produces exactly that, and reading its
    batch dim as H would mix the scenarios into each other.
    """
    h, n, cdt, device = plan.h, plan.n, plan.cdt, plan.device
    rdt = v.real.dtype if v.is_complex() else v.dtype

    v = v.to(dtype=cdt, device=device)
    if v.shape[-1] != n:
        raise ValueError(
            f"device_current_injections: v last dim {v.shape[-1]} != N={n}"
        )
    if v.ndim == 1:
        v = v.reshape(1, n)  # [1, N]; treated as [H=1, N] -> broadcast over H below
    has_h = v.ndim >= 2 and v.shape[-2] == h and (h > 1 or v.ndim == 2)
    if not has_h:
        v = v.unsqueeze(-2)  # [..., 1, N]
    batch_lead = v.shape[:-2]

    out = torch.zeros((*batch_lead, h, n), dtype=cdt, device=device)

    for g in plan.uncontrolled:
        k = g.rows.shape[0]
        # Gather USED-row voltages, then form the ELEMENT (terminal) voltages
        # V_term = M @ V_used.  v[*b, H, N] -> V_used[*b, H, K, n_used].
        v_used = v.index_select(-1, g.flat_rows)  # [*b, H, K*n_used]
        v_used = v_used.reshape(*batch_lead, h, k, g.n_used)
        # V_term[..., e] = sum_u M[e,u] V_used[..., u]  -> [*b,H,K,n_elem].
        vt = torch.einsum("eu,...ku->...ke", g.m_c, v_used)

        vmag = torch.abs(vt)  # [*b, H, K, n_elem] real
        ratio = vmag / g.v0  # |V_term| / |V0|  broadcasts [K,n_elem]

        # ZIP scaling per power component: z*ratio^2 + i*ratio + p.
        z_p, i_p, pp_p = g.zip_p[..., 0], g.zip_p[..., 1], g.zip_p[..., 2]  # [K]
        z_q, i_q, pp_q = g.zip_q[..., 0], g.zip_q[..., 1], g.zip_q[..., 2]
        scale_p = z_p[..., None] * ratio**2 + i_p[..., None] * ratio + pp_p[..., None]
        scale_q = z_q[..., None] * ratio**2 + i_q[..., None] * ratio + pp_q[..., None]

        s_eff = torch.complex(g.p_pp * scale_p, g.q_pp * scale_q).to(cdt)
        # i_elem = conj(S_eff) / conj(V_term). NOTE: terminal voltage is assumed
        # non-zero here (a converged PF never has a 0 V live terminal), so this
        # divide is UNGUARDED — unlike the otherwise-identical conj(vt) divide in
        # solver/harmonic_flow.py::_harmonic_injections, which DOES mask vt==0
        # because a gradcheck perturbation / dead harmonic terminal can hit zero.
        i_elem = torch.conj(s_eff) / torch.conj(vt)  # [*b,H,K,n_elem]
        # Nodal current at the used rows: I_used = M^T @ i_elem -> [*b,H,K,n_used].
        i_used = torch.einsum("eu,...ke->...ku", g.m_c, i_elem)
        out = _scatter_injection(out, i_used, g.rows)

    # --- inverter-controlled injections (voltage-dependent (P, Q) law) --------
    for c in plan.controlled:
        v_used = v.index_select(-1, c.arow).reshape(*batch_lead, h, c.n_used)
        vt = torch.einsum("eu,...u->...e", c.m_c, v_used)  # [*b, H, n_elem]
        v_pu = torch.abs(vt) / c.v0  # [*b, H, n_elem]

        p_eff, q_eff = resolve_injection_power(
            c.control, c.p_avail, v_pu, rdt=rdt, device=device
        )  # [*b, H, n_elem] native
        s_eff = torch.complex(c.sign * p_eff, c.sign * q_eff).to(cdt)
        # Guard the conj(vt) divide for a transiently/perturbed-to-zero terminal
        # (the iteration / a gradcheck step can reach 0 V) — mask the denominator,
        # then mask the result, keeping the gradient finite on live terminals.
        vtc = torch.conj(vt)
        safe = torch.where(vtc.abs() < 1e-300, torch.ones_like(vtc), vtc)
        i_elem = torch.where(
            vtc.abs() < 1e-300, torch.zeros_like(s_eff), torch.conj(s_eff) / safe
        )  # [*b, H, n_elem]
        i_used = torch.einsum("eu,...e->...u", c.m_c, i_elem)  # [*b, H, n_used]
        out = _scatter_injection(out, i_used.unsqueeze(-2), c.arow.unsqueeze(0))

    return out


def _zip_coeffs(appliance, rdt, device) -> tuple[Tensor, Tensor]:
    """Per-power-component ZIP coefficient pairs ``(zip_p[3], zip_q[3])`` = (z, i, p).

    Mapping per the frozen CONTEXT: const_power=(0,0,1), const_impedance=(1,0,0),
    const_current=(0,1,0); ``zip`` reads ``ZipCoefficients`` (z_*, i_*, p_*).
    Returned as real tensors ``[3]`` ordered (z, i, p) for P and for Q.
    """
    model = appliance.load_model
    if model == LoadModel.ZIP:
        zc = appliance.zip_coefficients
        zip_p = torch.as_tensor([zc.z_p, zc.i_p, zc.p_p], dtype=rdt, device=device)
        zip_q = torch.as_tensor([zc.z_q, zc.i_q, zc.p_q], dtype=rdt, device=device)
        return zip_p, zip_q
    if model == LoadModel.CONST_IMPEDANCE:
        coeff = torch.as_tensor([1.0, 0.0, 0.0], dtype=rdt, device=device)
    elif model == LoadModel.CONST_CURRENT:
        coeff = torch.as_tensor([0.0, 1.0, 0.0], dtype=rdt, device=device)
    else:  # CONST_POWER (default)
        coeff = torch.as_tensor([0.0, 0.0, 1.0], dtype=rdt, device=device)
    return coeff, coeff


def _per_phase_power_tensor(
    total, per_phase, n_phases: int, rdt: torch.dtype, device
) -> Tensor:
    """Autograd-safe per-phase power tensor ``[P]`` from a total or per-phase value.

    Preserves the graph when ``total`` / ``per_phase`` carry torch tensor leaves
    (tensor duality): a per-phase value is converted element-wise without
    ``torch.as_tensor`` over a list of tensors (which would detach); a total scalar
    is split equally across phases with a tensor-safe divide.
    """
    if per_phase is not None:
        parts = [
            x if isinstance(x, Tensor) else torch.as_tensor(x, dtype=rdt, device=device)
            for x in per_phase
        ]
        # Stack along the phase axis (last): supports per-load batch leading dims.
        return torch.stack([pp.to(dtype=rdt, device=device) for pp in parts], -1)
    # total split equally across phases (tensor-safe). ``total`` may be a scalar or
    # carry a leading scenario/batch shape -> per-phase tensor [*batch, P].
    tot = (
        total
        if isinstance(total, Tensor)
        else torch.as_tensor(total, dtype=rdt, device=device)
    )
    tot = tot.to(dtype=rdt, device=device)
    per = (tot / n_phases).unsqueeze(-1)  # [*batch, 1]
    return per.expand(*tot.shape, n_phases)  # [*batch, P]


def device_current_injections(
    grid: Grid,
    v: Tensor,
    index: NodePhaseIndex,
    frequencies_hz,
    *,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    operating_point: Optional[dict] = None,
    param_overrides: Optional[dict] = None,
    symmetry: Optional[str] = None,
) -> Tensor:
    """Voltage-dependent ZIP nodal current ``I_device(V)`` absorbed by loads/gens.

    For every in-service :class:`Load` / :class:`Generator`, the per-terminal
    (per phase) current drawn under the ZIP model is::

        S_eff(V) = S0 * ( z*(|Vt|/|V0|)**2 + i*(|Vt|/|V0|) + p )
        I_term   = conj(S_eff) / conj(Vt)

    with ``S0 = sign * (P + jQ)`` (sign +1 load / -1 generator, identical to the
    const-Z stamp), ``V0`` = NOMINAL line-to-neutral voltage (``u_rated``), and
    ``Vt`` = the terminal voltage gathered from ``v``. The ZIP triples
    ``(z, i, p)`` are taken per power component (P and Q independently) from the
    load model: const_power=(0,0,1), const_impedance=(1,0,0), const_current=
    (0,1,0), ``zip`` from :class:`ZipCoefficients`.

    The result has the SAME sign convention as the const-Z fold in
    :func:`assemble_ybus`: a const-impedance ZIP solve reproduces the linear
    ``assemble_ybus`` system exactly (``I_device = Y_devZ @ V``). The nonlinear
    nodal balance is ``Y_eff @ V = I_slack - I_device(V)``.

    Parameters
    ----------
    grid:
        Materialised grid.
    v:
        Complex terminal voltages ``[*batch, H, N]`` (or ``[*batch, N]`` / ``[N]``;
        a missing H axis is broadcast). Must be on ``device`` / convertible.
    index:
        The compact :class:`NodePhaseIndex` (must match ``v``'s row layout).
    frequencies_hz:
        Frequencies; only the count ``H`` matters here (the ZIP model is at f0 and
        frequency-independent) — used to align the output H axis.
    dtype, device:
        Complex dtype / target device. ``device`` defaults to ``v``'s device.
    operating_point:
        Optional override of nameplate P/Q (see :func:`resolve_operating_power`).
    param_overrides:
        Optional differentiability hook; keys ``("load"|"generator", id,
        "p_nom_per_phase_w"|"q_nom_per_phase_var")`` inject leaf per-phase P/Q.
    symmetry:
        Calculation-symmetry mode (``None`` -> config). Resolved internally to a bool
        (silently — no logging in the iteration). When the caller already resolved it,
        pass the resolved string through so the iteration stays consistent.

    Connection-aware: each Load/Generator's per-element current is
    ``i_elem = conj(S_eff(V_term)) / conj(V_term)`` with ``V_term = M @ V_used``
    (the WYE/DELTA/neutral incidence ``M``); the NODAL current is ``M^T @ i_elem``.
    WYE-to-ground (``M = I``) reduces to the historical per-phase form exactly.

    Returns
    -------
    Tensor
        Complex ``I_device`` ``[*batch, H, N]`` (matching ``v``'s batch broadcast).
        Purely passive grids -> zeros. Aligned to ``index``.
    """
    plan = build_injection_plan(
        grid,
        index,
        frequencies_hz,
        dtype=dtype,
        device=device if device is not None else v.device,
        operating_point=operating_point,
        param_overrides=param_overrides,
        symmetry=symmetry,
    )
    return injections_from_plan(plan, v)


def _stack_elements(vals, rdt: torch.dtype, device) -> Tensor:
    """Stack a per-element power list (mixed floats / batched tensors) -> ``[*b, n_elem]``.

    Broadcasts every entry to a common leading shape before stacking so a per-element
    list of scalars yields ``[n_elem]`` and a list of ``[*batch]`` tensors yields
    ``[*batch, n_elem]`` (graph-preserving for tensor leaves)."""
    ts = [
        v.to(dtype=rdt, device=device)
        if isinstance(v, Tensor)
        else torch.as_tensor(v, dtype=rdt, device=device)
        for v in vals
    ]
    lead = torch.broadcast_shapes(*[t.shape for t in ts])
    return torch.stack([t.broadcast_to(lead) for t in ts], dim=-1)


def _tensor_sum(per_phase, rdt: torch.dtype, device):
    """Autograd-safe sum of a per-phase list (mixed floats / tensors) -> 0-d tensor."""
    total = None
    for x in per_phase:
        xt = (
            x if isinstance(x, Tensor) else torch.as_tensor(x, dtype=rdt, device=device)
        )
        xt = xt.to(dtype=rdt, device=device)
        total = xt if total is None else total + xt
    return total


__all__ = [
    "YBus",
    "BranchCurrent",
    "BranchStampBlock",
    "InjectionPlan",
    "assemble_ybus",
    "assemble_network_ybus",
    "branch_currents",
    "branch_stamp_blocks",
    "build_injection_plan",
    "build_injections",
    "device_current_injections",
    "injections_from_plan",
    "node_phase_index",
    "NodePhaseIndex",
]
