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

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

from pgml.equations import registry
from pgml.errors import InputError
from pgml.schemas.grid_schema import (
    Generator,
    GenericBranch,
    Grid,
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

from ._incidence import build_incidence, group_appliances, used_rows
from ._transformer import (
    block_incidence,
    group_key as _xfmr_group_key,
    nominal_turns_ratio,
    resolve_vector_group,
    winding_leakage_block,
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

# ``equations`` is imported so its laws are registered; the assembly uses the
# same closed-form expressions inline (X=2*pi*f*L, B=2*pi*f*C) for vectorized
# matrix math, which the registry cannot express (matrix inverse lives here).
_ = registry  # keep the import meaningful / ensure laws are registered.


@dataclass(frozen=True)
class YBus:
    """Assembled nodal admittance system.

    Attributes
    ----------
    Y:
        Complex tensor ``[*batch, H, N, N]`` (or ``[N, N]`` if a single grid and a
        single frequency were requested and ``squeeze`` applies).
    index:
        The :class:`NodePhaseIndex` describing the compact row layout.
    frequencies_hz:
        Real tensor ``[H]`` of the absolute frequencies the Y was built at.
    """

    Y: Tensor
    index: NodePhaseIndex
    frequencies_hz: Tensor


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
# ``builder(grid, f, index, cdt, rdt, device, param_overrides)`` that YIELDS
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
) -> YBus:
    """Assemble the LINEAR (const-Z) complex nodal admittance ``Y(f)``.

    This is the **linear-model** assembler: it equals
    :func:`assemble_network_ybus` (the passive network) PLUS the const-Z device
    shunts (loads/generators folded as constant impedance at nominal voltage) PLUS
    the source Norton (Thévenin) shunt. The nonlinear (const-P / ZIP) power flow
    instead uses :func:`assemble_network_ybus` + :func:`device_current_injections`
    (see ``assembly/CONTEXT.md``). The IEEE33 / tiny-grid oracle tests keep using
    this function as their regression suite, so its behaviour is UNCHANGED.

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

    Returns
    -------
    YBus
        ``Y`` complex ``[H, N, N]`` (``[N, N]`` if ``H == 1`` and a scalar
        frequency was passed), the :class:`NodePhaseIndex`, and the frequencies.
    """
    if device is None and isinstance(frequencies_hz, Tensor):
        device = frequencies_hz.device
    f = _as_freq_tensor(frequencies_hz, dtype, device)
    device = f.device
    h = f.shape[0]
    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)

    index = node_phase_index(grid)
    n = index.size

    asymmetric = resolve_asymmetric(grid, operating_point, mode=symmetry)
    # Single INFO modeling summary for this entry point (logs once per assemble_ybus
    # call; solve_power_flow logs its own, and harmonic_flow delegates its log to
    # solve_power_flow — so no double logging across the public entry points).
    log_modeling_summary(grid, asymmetric=asymmetric)

    y = torch.zeros((h, n, n), dtype=cdt, device=device)

    # Passive network (shared with assemble_network_ybus).
    y = _stamp_network(grid, f, y, index, cdt, rdt, device, param_overrides)
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
    if scalar_freq:
        y = y.reshape(n, n)
    return YBus(Y=y, index=index, frequencies_hz=f)


def assemble_network_ybus(
    grid: Grid,
    frequencies_hz,
    *,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    param_overrides: Optional[dict] = None,
) -> YBus:
    """Assemble the PASSIVE-NETWORK nodal admittance ``Y_net(f)``.

    Contains ONLY the passive network: lines, transformers, switches, shunt
    reactors, generic branches, and :class:`ShuntAppliance` (a fixed linear
    shunt). It does **not** stamp loads/generators and does **not** fold the
    source as a Norton shunt — both are handled on the RHS by the nonlinear
    power-flow path (:func:`device_current_injections` and the slack handling in
    :func:`pgml.solver.solve_power_flow`).

    It is constant (voltage-independent) and differentiable w.r.t. network params,
    and uses the SAME compact :class:`NodePhaseIndex` layout and ``[*batch, H, N,
    N]`` shapes as :func:`assemble_ybus`.

    Parameters
    ----------
    grid, frequencies_hz, dtype, device, param_overrides:
        Identical meaning to :func:`assemble_ybus` (no ``operating_point`` — there
        is no device folding here).

    Returns
    -------
    YBus
        ``Y`` complex ``[H, N, N]`` (``[N, N]`` for a scalar frequency), the
        :class:`NodePhaseIndex`, and the frequencies.
    """
    if device is None and isinstance(frequencies_hz, Tensor):
        device = frequencies_hz.device
    f = _as_freq_tensor(frequencies_hz, dtype, device)
    device = f.device
    h = f.shape[0]
    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)

    index = node_phase_index(grid)
    n = index.size

    y = torch.zeros((h, n, n), dtype=cdt, device=device)
    y = _stamp_network(grid, f, y, index, cdt, rdt, device, param_overrides)

    scalar_freq = (
        not isinstance(frequencies_hz, Tensor)
        and not isinstance(frequencies_hz, (list, tuple))
        and h == 1
    )
    if scalar_freq:
        y = y.reshape(n, n)
    return YBus(Y=y, index=index, frequencies_hz=f)


def _stamp_network(grid, f, y, index, cdt, rdt, device, param_overrides):
    """Accumulate every PASSIVE contribution into ``y`` (shared assembler core).

    Iterates the branch-stamp registry (:data:`_BRANCH_STAMPS`) — lines, switches,
    generic branches, shunt reactors, transformers — scattering every primitive
    block each builder yields, then adds the ShuntAppliance fixed shunt (an
    appliance, not a branch, so outside the registry). NO source Norton, NO
    load/generator folding. Because scatter-add is order-independent, the resulting
    ``y`` is identical regardless of the registration order.
    """
    for stamp in _BRANCH_STAMPS:
        for _group, block, rows, cols in stamp.builder(
            grid, f, index, cdt, rdt, device, param_overrides
        ):
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
    """A line opted into the sequence-aware harmonic model (UNBALANCED studies)."""
    return (line.tags or {}).get("harmonic_line_model") == "sequence_aware"


@_branch_stamp("line")
def _line_block_groups(grid, f, index, cdt, rdt, device, param_overrides):
    """Yield ``(group, block, rows, cols)`` for every line group (all three paths).

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
        if isinstance(b, Line) and b.in_service and b.conductor_geometry is None
    ]
    rx_lines = [b for b in flow_lines if not _is_sequence_aware(b)]
    seq_lines = [b for b in flow_lines if _is_sequence_aware(b)]
    if rx_lines:
        yield from _line_rx_block_groups(
            rx_lines, f, index, cdt, rdt, device, param_overrides
        )
    if seq_lines:
        yield from _sequence_aware_block_groups(
            seq_lines, grid, f, index, cdt, rdt, device, param_overrides
        )
    yield from _geometry_block_groups(grid, f, index, cdt, rdt, device, param_overrides)


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
    from pgml.geometry.sequence import CARSON_EARTH_R_PER_HZ, sequence_aware_phase_z

    f0 = float(grid.base_frequency_hz)
    two_pi_f0 = 2.0 * math.pi * f0
    two_pi_f = (2.0 * torch.pi) * f  # [H]

    # Group by (skin, earth coeff) so each batched call shares its model options.
    by_opts: dict[tuple, list] = {}
    for ln in lines:
        if len(ln.from_phases) != 3:
            raise ValueError(
                f"Line {ln.id}: harmonic_line_model=sequence_aware requires a 3-phase "
                f"line (got {len(ln.from_phases)} phases); it is a Z1/Z0 model."
            )
        tags = ln.tags or {}
        skin = tags.get("seq_skin", "true") == "true"
        coeff = float(tags.get("seq_earth_coeff", CARSON_EARTH_R_PER_HZ))
        by_opts.setdefault((skin, coeff), []).append(ln)

    for (skin, coeff), group in by_opts.items():
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
        z_abc = sequence_aware_phase_z(
            z1.real,
            z1.imag,
            z0.real,
            z0.imag,
            f0,
            f,
            skin=skin,
            earth_resistance_coeff=coeff,
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


def _geometry_block_groups(grid, f, index, cdt, rdt, device, param_overrides):
    """Yield ``(group, block, rows, cols)`` for geometry (Carson/Deri) lines."""
    glines = [
        b
        for b in grid.branches
        if isinstance(b, Line) and b.in_service and b.conductor_geometry is not None
    ]
    if not glines:
        return
    from pgml.geometry.carson import line_constants

    by_key: dict[tuple[int, int], list] = {}
    for ln in glines:
        key = (len(ln.from_phases), len(ln.conductor_geometry.conductors))
        by_key.setdefault(key, []).append(ln)

    two_pi_f = (2.0 * torch.pi) * f  # [H]
    for (nph, _ncond), group in by_key.items():
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
            X, Y, GMR, RDC, RAD, RHO, f, nph
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


def _line_rx_block_groups(lines, f, index, cdt, rdt, device, param_overrides):
    by_p: dict[int, list] = {}
    for ln in lines:
        by_p.setdefault(len(ln.from_phases), []).append(ln)
    for p, group in by_p.items():
        r_list, l_list, g_list, c_list, mult_list = [], [], [], [], []
        for ln in group:
            length = ln.length_m
            r0 = (
                _override(
                    param_overrides,
                    ("line", ln.id, "series_resistance_ohm_per_m"),
                    _real_matrix(ln.series_resistance_ohm_per_m, rdt, device),
                )
                * length
            )
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
            r_list.append(r0)
            l_list.append(ind)
            g_list.append(cond)
            c_list.append(cap)
            mult_list.append(_resistance_multiplier(ln, f, rdt, device))  # [H]
        r = torch.stack(r_list, 0)  # [K,P,P]
        ind = torch.stack(l_list, 0)
        g = torch.stack(g_list, 0)
        c = torch.stack(c_list, 0)
        rmult = torch.stack(mult_list, 1)  # [H,K]
        rmult = rmult[:, :, None, None]  # [H,K,1,1]

        ys = series_admittance_matrix(r, ind, f, cdt, r_mult=rmult)  # [H,K,P,P]
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


def _resistance_multiplier(line, f, rdt, device) -> Tensor:
    """Per-frequency resistance multiplier m(f) ``[H]`` from ResistanceFrequencyModel.

    Supported multipliers:
    - ``constant`` -> its scalar value (M1 default).
    - ``analytic`` with ``law == "carson_skin_multiplier"`` -> the differentiable
      positive-sequence skin-effect curve (``pgml.geometry.sequence``), the Bessel
      ``I0/I1`` internal-resistance growth WITHOUT the earth-return floor; ``params``
      carry ``r1_ohm_per_m`` and ``f0_hz`` (see ``synthesize_positive_sequence_*``).
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
def _switch_block_groups(grid, f, index, cdt, rdt, device, param_overrides):
    switches = [
        b for b in grid.branches if isinstance(b, Switch) and b.in_service and b.closed
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
def _generic_branch_block_groups(grid, f, index, cdt, rdt, device, param_overrides):
    branches = [
        b for b in grid.branches if isinstance(b, GenericBranch) and b.in_service
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
def _shunt_reactor_block_groups(grid, f, index, cdt, rdt, device, param_overrides):
    reactors = [
        b for b in grid.branches if isinstance(b, ShuntReactor) and b.in_service
    ]
    if not reactors:
        return
    by_p: dict[int, list] = {}
    for sr in reactors:
        by_p.setdefault(len(sr.from_phases), []).append(sr)
    for p, group in by_p.items():
        g_list, c_list = [], []
        for sr in group:
            g_list.append(_real_matrix(sr.conductance_s, rdt, device))
            c_list.append(_real_matrix(sr.capacitance_f, rdt, device))
        g = torch.stack(g_list, 0)
        c = torch.stack(c_list, 0)
        block = shunt_admittance_matrix(g, c, f, cdt)  # [H,K,P,P]
        rows, cols = _shunt_node_indices(group, index, device, terminal="from")
        yield group, block, rows, cols


def _stamp_shunt_appliances(grid, f, y, index, cdt, rdt, device, param_overrides):
    shunts = [
        a for a in grid.appliances if isinstance(a, ShuntAppliance) and a.in_service
    ]
    if not shunts:
        return y
    by_p: dict[int, list] = {}
    for sh in shunts:
        by_p.setdefault(len(sh.phases), []).append(sh)
    for p, group in by_p.items():
        g_list, c_list = [], []
        for sh in group:
            g_list.append(
                torch.diag(torch.as_tensor(sh.conductance_s, dtype=rdt, device=device))
            )
            c_list.append(
                torch.diag(torch.as_tensor(sh.capacitance_f, dtype=rdt, device=device))
            )
        g = torch.stack(g_list, 0)
        c = torch.stack(c_list, 0)
        block = shunt_admittance_matrix(g, c, f, cdt)
        rows, cols = _shunt_node_indices(group, index, device, terminal="node")
        y = scatter_blocks_into(y, block, rows, cols)
    return y


# ---- source Thevenin -> Norton shunt --------------------------------------
def _stamp_sources(grid, f, y, index, cdt, rdt, device, param_overrides):
    sources = [a for a in grid.appliances if isinstance(a, Source) and a.in_service]
    if not sources:
        return y
    by_p: dict[int, list] = {}
    for s in sources:
        by_p.setdefault(len(s.phases), []).append(s)
    for p, group in by_p.items():
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
        r = torch.stack(r_list, 0)
        ind = torch.stack(l_list, 0)
        ys = series_admittance_matrix(r, ind, f, cdt)  # [H,K,P,P]  Y_s = Z_s^-1
        rows, cols = _shunt_node_indices(group, index, device, terminal="node")
        y = scatter_blocks_into(y, ys, rows, cols)
    return y


# ---- const-Z load / generator ---------------------------------------------
def _stamp_const_z_loads(
    grid, f, y, index, cdt, rdt, device, operating_point, param_overrides, asymmetric
):
    """Fold each Load/Generator as a connection-aware const-Z shunt.

    The internal per-ELEMENT admittance ``y_elem = conj(P_k + jQ_k)/|V0|^2`` is
    mapped to a nodal block ``M^T diag(y_elem) M`` via the terminal incidence ``M``
    (``_incidence``): WYE-to-ground reduces to ``M = I`` (the historical diagonal
    stamp, bit-exact); WYE-with-neutral uses ``[I|-1]`` (the neutral row receives
    the phase return); DELTA-3 uses the circulant difference. ``asymmetric=False``
    forces the equal split inside ``resolve_operating_power``. ``V0`` is L-N for WYE
    and L-L for DELTA (:func:`phase_voltage_magnitude`).
    """
    loads = [
        a for a in grid.appliances if isinstance(a, (Load, Generator)) and a.in_service
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
            # Per-element admittance y_elem = conj(P+jQ)/|V0|^2  -> [n_elem].
            y_elem = const_z_shunt_admittance(p_pp, q_pp, v0, sign, cdt, device)
            elem_list.append(y_elem)  # [n_elem]
        y_elem_k = torch.stack(elem_list, 0)  # [K, n_elem]
        # Y_block = M^T diag(y_elem) M  -> [K, n_used, n_used].
        block = torch.einsum("ei,ke,ej->kij", m_c, y_elem_k, m_c)
        block = block[None].expand(f.shape[0], *block.shape)  # [H,K,n_used,n_used]
        rows = used_rows(grp, index, device)  # [K, n_used]
        y = scatter_blocks_into(y, block, rows, rows)
    return y


# ---- transformer (vector-group winding-incidence primitive) ---------------
@_branch_stamp("transformer")
def _transformer_block_groups(grid, f, index, cdt, rdt, device, param_overrides):
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

    - Leakage admittance ``y_se = (R + jX(f))^-1`` (scalar per phase, X(f)=2πfL),
      referred to the TO-side (LV) coil, carried through the incidence transform.
    - Magnetizing shunt ``y_m = G_m + jB_m`` added to the HV terminal phase
      diagonal directly (referred to the HV line voltage).

    Differentiable w.r.t. series R, L and the off-nominal tap magnitude; the
    discrete vector-group connections / clock select the constant incidence ``N``.

    Shared primitive builder for the Y-bus stamp (which scatters each block) and
    :func:`branch_currents` (which multiplies it with the terminal voltage). The
    block is the full ``[H, K, 2P, 2P]`` vector-group winding-incidence primitive
    plus the magnetizing shunt on the HV diagonal.
    """
    xfmrs = [b for b in grid.branches if isinstance(b, Transformer) and b.in_service]
    if not xfmrs:
        return
    # Group transformers sharing one incidence N (same connection pair, clock, P).
    by_key: dict[tuple, list] = {}
    vgs: dict[int, object] = {}
    for t in xfmrs:
        vg = resolve_vector_group(t)
        vgs[id(t)] = vg
        by_key.setdefault(_xfmr_group_key(vg, len(t.from_phases)), []).append(t)

    two_pi_f = (2.0 * torch.pi) * f  # [H]
    for (_fk, _tk, _ct, p), group in by_key.items():
        vg0 = vgs[id(group[0])]
        eye_p = torch.eye(p, dtype=cdt, device=device)

        yse_list, ratio_list, ym_list = [], [], []
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
            x = two_pi_f * ell  # [H]
            z = torch.complex(r.to(rdt).expand_as(x), x.to(rdt)).to(cdt)  # [H]
            yse_list.append(1.0 / z)  # [H]

            tap_mag = _override(
                param_overrides,
                ("transformer", t.id, "tap_magnitude"),
                torch.as_tensor(t.tap.ratio_magnitude, dtype=rdt, device=device),
            )
            u_from = torch.as_tensor(t.u_rated_from_v, dtype=rdt, device=device)
            u_to = torch.as_tensor(t.u_rated_to_v, dtype=rdt, device=device)
            if p == 1:
                # Single-phase / positive-sequence equivalent: the vector group is
                # folded into a complex line-to-line ratio (magnitude n_LL, clock
                # phase shift), the textbook off-nominal-tap pi.
                theta = math.radians(vg.shift_deg)
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

        y_se = torch.stack(yse_list, dim=1)  # [H,K]
        ratio = torch.stack(ratio_list, dim=0)  # [K]
        if p == 1:
            block = _scalar_tap_blocks(y_se, ratio)  # [H,K,2,2]
        else:
            n_block = block_incidence(vg0, p, rdt, device)  # [2P,2P]
            block = winding_leakage_block(y_se, ratio, n_block)  # [H,K,2P,2P]

        # Magnetizing shunt on the HV terminal diagonal (outside the incidence).
        ym = torch.stack(ym_list, dim=1)  # [H,K]
        ym_hv = ym[:, :, None, None] * eye_p  # [H,K,P,P]
        ym_full = torch.nn.functional.pad(ym_hv, (0, p, 0, p))  # [H,K,2P,2P]
        block = block + ym_full

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
    # I_term[..., k, i] = sum_j block[H, k, i, j] V_term[..., H, k, j].
    return torch.einsum("hkij,...hkj->...hki", block, v_term)


def branch_currents(
    grid: Grid,
    v: Tensor,
    frequencies_hz,
    index: NodePhaseIndex,
    *,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
    param_overrides: Optional[dict] = None,
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

    Returns
    -------
    list[BranchCurrent]
        One :class:`BranchCurrent` per in-service branch, in ``grid.branches``
        order. ``i_from`` / ``i_to`` are complex ``[*batch, H, P]`` (the leading
        ``*batch`` matching ``v``'s broadcast; a scalar/length-1 frequency keeps a
        singleton H axis).
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

    # Iterate the SAME branch-stamp registry the Y-bus assembly does: each builder
    # yields (group, block, rows, cols) — the exact primitive block the stamp
    # scatters — so the currents are consistent with Y to machine precision (KCL).
    # rows == cols here; the block is multiplied with the gathered terminal voltage.
    results: dict[int, BranchCurrent] = {}

    for stamp in _BRANCH_STAMPS:
        for group, block, rows, _cols in stamp.builder(
            grid, f, index, cdt, rdt, device, param_overrides
        ):
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

    # Emit in grid.branches order over the in-service BranchBase branches.
    return [results[b.id] for b in grid.branches if b.in_service and b.id in results]


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

    M1 content: the source Norton current ``i_s = Y_s @ V_th`` stamped at the
    source rows, where ``V_th`` is the per-phase Thevenin phasor
    ``u_ref * exp(j*u_angle)`` and ``Y_s = Z_s(f)^-1``. Harmonic current sources
    from load/generator spectra are a later milestone (return 0 contribution).

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
        for p, group in by_p.items():
            r_list, l_list, vth_list, row_list = [], [], [], []
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
                u_ref = torch.as_tensor(s.u_ref_v, dtype=rdt, device=device)
                ang = torch.as_tensor(s.u_angle_deg, dtype=rdt, device=device) * (
                    math.pi / 180.0
                )
                vth_list.append(torch.polar(u_ref, ang).to(cdt))  # [P]
                row_list.append([index.row(s.node, ph) for ph in s.phases])
            r = torch.stack(r_list, 0)  # [K,P,P]
            ind = torch.stack(l_list, 0)
            vth = torch.stack(vth_list, 0)  # [K,P]
            ys = series_admittance_matrix(r, ind, f, cdt)  # [H,K,P,P]
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
    if device is None:
        device = v.device
    f = _as_freq_tensor(frequencies_hz, dtype, device)
    h = f.shape[0]
    cdt = _cdtype(dtype)
    rdt = _rdtype(dtype)
    n = index.size

    v = v.to(dtype=cdt, device=device)
    # Normalise v to leading shape [*batch, H, N]: insert an H axis if absent.
    if v.shape[-1] != n:
        raise ValueError(
            f"device_current_injections: v last dim {v.shape[-1]} != N={n}"
        )
    if v.ndim == 1:
        v = v.reshape(1, n)  # [1, N]; treated as [H=1, N] -> broadcast over H below
    # If there is no explicit H axis matching h, broadcast a singleton H in.
    has_h = v.ndim >= 2 and v.shape[-2] == h
    if not has_h:
        v = v.unsqueeze(-2)  # [..., 1, N]
    batch_lead = v.shape[:-2]

    out = torch.zeros((*batch_lead, h, n), dtype=cdt, device=device)

    loads = [
        a for a in grid.appliances if isinstance(a, (Load, Generator)) and a.in_service
    ]
    if not loads:
        return out

    asymmetric = resolve_asymmetric(grid, operating_point, mode=symmetry)

    node_map = {nd.id: nd for nd in grid.nodes}

    for grp in group_appliances(loads, node_map):
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

        k = len(p_list)
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
        v0 = torch.stack(v0_list, -2)  # [K, n_elem]
        zip_p = torch.stack(zipp_list, 0)  # [K, 3]
        zip_q = torch.stack(zipq_list, 0)
        rows = used_rows(grp, index, device)  # [K, n_used] int64

        # Insert a singleton H axis into power tensors so they broadcast over H.
        p_pp = p_pp.unsqueeze(-3)  # [*pbatch, 1, K, n_elem]
        q_pp = q_pp.unsqueeze(-3)

        # Gather USED-row voltages, then form the ELEMENT (terminal) voltages
        # V_term = M @ V_used.  v[*b, H, N] -> V_used[*b, H, K, n_used].
        flat_rows = rows.reshape(-1)  # [K*n_used]
        v_used = v.index_select(-1, flat_rows)  # [*b, H, K*n_used]
        v_used = v_used.reshape(*batch_lead, h, k, grp.n_used)  # [*b,H,K,n_used]
        # V_term[..., e] = sum_u M[e,u] V_used[..., u]  -> [*b,H,K,n_elem].
        vt = torch.einsum("eu,...ku->...ke", m_c, v_used)

        vmag = torch.abs(vt)  # [*b, H, K, n_elem] real
        ratio = vmag / v0  # |V_term| / |V0|  broadcasts [K,n_elem]

        # ZIP scaling per power component: z*ratio^2 + i*ratio + p.
        z_p, i_p, pp_p = zip_p[..., 0], zip_p[..., 1], zip_p[..., 2]  # [K]
        z_q, i_q, pp_q = zip_q[..., 0], zip_q[..., 1], zip_q[..., 2]
        scale_p = z_p[..., None] * ratio**2 + i_p[..., None] * ratio + pp_p[..., None]
        scale_q = z_q[..., None] * ratio**2 + i_q[..., None] * ratio + pp_q[..., None]

        s_eff = torch.complex(p_pp * scale_p, q_pp * scale_q).to(cdt)  # [*b,H,K,n_elem]
        # i_elem = conj(S_eff) / conj(V_term). NOTE: terminal voltage is assumed
        # non-zero here (a converged PF never has a 0 V live terminal), so this
        # divide is UNGUARDED — unlike the otherwise-identical conj(vt) divide in
        # solver/harmonic_flow.py::_harmonic_injections, which DOES mask vt==0
        # because a gradcheck perturbation / dead harmonic terminal can hit zero.
        i_elem = torch.conj(s_eff) / torch.conj(vt)  # [*b,H,K,n_elem]
        # Nodal current at the used rows: I_used = M^T @ i_elem -> [*b,H,K,n_used].
        i_used = torch.einsum("eu,...ke->...ku", m_c, i_elem)

        out = _scatter_injection(out, i_used, rows)

    return out


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
    "assemble_ybus",
    "assemble_network_ybus",
    "branch_currents",
    "build_injections",
    "device_current_injections",
    "node_phase_index",
    "NodePhaseIndex",
]
