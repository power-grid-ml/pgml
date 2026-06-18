"""Differentiable, batched, per-frequency Y-bus assembly and current injections.

Public API
----------
- ``assemble_ybus(grid, frequencies_hz, *, dtype, device, operating_point,
   param_overrides=None) -> YBus``
- ``build_injections(grid, frequencies_hz, index, *, dtype, device,
   operating_point, param_overrides=None) -> Tensor``
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
from pgml.schemas.grid_schema import (
    Generator,
    GenericBranch,
    Grid,
    Line,
    Load,
    LoadModel,
    ShuntAppliance,
    ShuntReactor,
    Source,
    Switch,
    Transformer,
)

from ._params import (
    const_z_shunt_admittance,
    phase_voltage_magnitude,
    resolve_operating_power,
)
from ._scatter import scatter_blocks_into
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

    y = torch.zeros((h, n, n), dtype=cdt, device=device)

    # Passive network (shared with assemble_network_ybus).
    y = _stamp_network(grid, f, y, index, cdt, rdt, device, param_overrides)
    # Linear-model device folding: source Norton + const-Z loads/gens.
    y = _stamp_sources(grid, f, y, index, cdt, rdt, device, param_overrides)
    y = _stamp_const_z_loads(
        grid, f, y, index, cdt, rdt, device, operating_point, param_overrides
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

    Lines, switches, generic branches, shunt reactors, transformers, and
    ShuntAppliance. NO source Norton, NO load/generator folding.
    """
    y = _stamp_lines(grid, f, y, index, cdt, rdt, device, param_overrides)
    y = _stamp_switches(grid, f, y, index, cdt, rdt, device, param_overrides)
    y = _stamp_generic_branches(grid, f, y, index, cdt, rdt, device, param_overrides)
    y = _stamp_shunt_reactors(grid, f, y, index, cdt, rdt, device, param_overrides)
    y = _stamp_transformers(grid, f, y, index, cdt, rdt, device, param_overrides)
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


def _stamp_lines(grid, f, y, index, cdt, rdt, device, param_overrides):
    # Explicit-R/L/C lines go through the matrix path; lines carrying a
    # `conductor_geometry` go through the Carson/Deri geometry path.
    rx_lines = [
        b
        for b in grid.branches
        if isinstance(b, Line) and b.in_service and b.conductor_geometry is None
    ]
    if rx_lines:
        y = _stamp_line_groups(rx_lines, f, y, index, cdt, rdt, device, param_overrides)
    y = _stamp_geometry_lines(grid, f, y, index, cdt, rdt, device, param_overrides)
    return y


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
        c = next((c for c in geo.conductors if not c.is_neutral and c.phase == ph), None)
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


def _stamp_geometry_lines(grid, f, y, index, cdt, rdt, device, param_overrides):
    """Stamp lines whose impedance comes from conductor geometry (Carson/Deri)."""
    glines = [
        b
        for b in grid.branches
        if isinstance(b, Line) and b.in_service and b.conductor_geometry is not None
    ]
    if not glines:
        return y
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
        y = scatter_blocks_into(y, block, rows, cols)
    return y


def _stamp_line_groups(lines, f, y, index, cdt, rdt, device, param_overrides):
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
        y = scatter_blocks_into(y, block, rows, cols)
    return y


def _resistance_multiplier(line, f, rdt, device) -> Tensor:
    """Per-frequency resistance multiplier m(f) ``[H]`` from ResistanceFrequencyModel.

    Only the ``constant`` multiplier is wired in M1 (returns its scalar value);
    analytic/curve/equation laws are a Phase-2 (geometry/skin-effect) concern and
    fall back to the constant value if present, else 1.0.
    """
    rfm = getattr(line, "resistance_frequency", None)
    val = 1.0
    if rfm is not None:
        mult = rfm.multiplier
        if getattr(mult, "kind", None) == "constant":
            val = mult.value
        elif getattr(mult, "kind", None) == "analytic":
            val = mult.base_value
    return torch.full((f.shape[0],), float(val), dtype=rdt, device=device)


def _stamp_switches(grid, f, y, index, cdt, rdt, device, param_overrides):
    switches = [
        b for b in grid.branches if isinstance(b, Switch) and b.in_service and b.closed
    ]
    if not switches:
        return y
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
        y = scatter_blocks_into(y, block, rows, cols)
    return y


def _stamp_generic_branches(grid, f, y, index, cdt, rdt, device, param_overrides):
    branches = [
        b for b in grid.branches if isinstance(b, GenericBranch) and b.in_service
    ]
    if not branches:
        return y
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
        y = scatter_blocks_into(y, block, rows, cols)
    return y


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


def _stamp_shunt_reactors(grid, f, y, index, cdt, rdt, device, param_overrides):
    reactors = [
        b for b in grid.branches if isinstance(b, ShuntReactor) and b.in_service
    ]
    if not reactors:
        return y
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
        y = scatter_blocks_into(y, block, rows, cols)
    return y


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
    grid, f, y, index, cdt, rdt, device, operating_point, param_overrides
):
    loads = [
        a for a in grid.appliances if isinstance(a, (Load, Generator)) and a.in_service
    ]
    if not loads:
        return y
    node_map = {nd.id: nd for nd in grid.nodes}
    by_p: dict[int, list] = {}
    for a in loads:
        by_p.setdefault(len(a.phases), []).append(a)
    for p, group in by_p.items():
        diag_list = []
        for a in group:
            node = node_map[a.node]
            u_ln = phase_voltage_magnitude(node.u_rated_v, len(node.phases))
            sign = 1.0 if isinstance(a, Load) else -1.0
            p_pp, q_pp = resolve_operating_power(a, operating_point)
            y_pp = const_z_shunt_admittance(p_pp, q_pp, u_ln, sign, cdt, device)  # [P]
            diag_list.append(torch.diag_embed(y_pp))  # [P,P]
        block = torch.stack(diag_list, 0)  # [K,P,P]
        block = block[None].expand(f.shape[0], *block.shape)  # [H,K,P,P]
        rows, cols = _shunt_node_indices(group, index, device, terminal="node")
        y = scatter_blocks_into(y, block, rows, cols)
    return y


# ---- transformer (in-phase ratio + leakage pi; vector-group deferred) -----
def _stamp_transformers(grid, f, y, index, cdt, rdt, device, param_overrides):
    """Two-winding transformer stamp: in-phase complex-tap leakage pi.

    Implemented (M1)
    ----------------
    - Per-phase series leakage admittance ``y_se = (R + jX(f))^-1`` (scalar per
      phase, X(f)=2*pi*f*L) with the magnetizing shunt ``y_m = G_m + jB_m`` added
      to the HV-side diagonal.
    - Off-nominal complex tap ``t = ratio_magnitude * exp(j*shift_deg)`` applied as
      the standard two-port leakage-pi primitive:
          Y_ff = y_se / |t|^2,  Y_ft = -y_se / conj(t),
          Y_tf = -y_se / t,     Y_tt = y_se
      (the textbook off-nominal-tap pi; with ``shift_deg=0`` this is the real-ratio
      transformer). This handles the in-phase ratio AND a uniform per-phase phase
      shift differentiably w.r.t. R, L, and the tap.

    Deferred to M2 (marked, not implemented)
    ----------------------------------------
    - Full vector-group phase-domain coupling (Dyn / Yd connection matrices that
      mix phases), zero-sequence path from winding connection, and neutral
      grounding impedance. These require the connection-dependent incidence
      matrices; M1 treats the transformer as a per-phase (diagonal) coupled
      two-port using ``tap.shift_deg`` as a uniform phase shift only.
    """
    xfmrs = [b for b in grid.branches if isinstance(b, Transformer) and b.in_service]
    if not xfmrs:
        return y
    by_p: dict[int, list] = {}
    for t in xfmrs:
        by_p.setdefault(len(t.from_phases), []).append(t)
    for p, group in by_p.items():
        eye = torch.eye(p, dtype=rdt, device=device)
        r_list, l_list, gm_list, lm_list, tap_mag_list, tap_shift_list = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for t in group:
            r_list.append(
                _override(
                    param_overrides,
                    ("transformer", t.id, "series_resistance_ohm"),
                    torch.as_tensor(t.series_resistance_ohm, dtype=rdt, device=device),
                )
                * eye
            )
            l_list.append(
                _override(
                    param_overrides,
                    ("transformer", t.id, "series_inductance_h"),
                    torch.as_tensor(t.series_inductance_h, dtype=rdt, device=device),
                )
                * eye
            )
            gm_list.append(
                torch.as_tensor(t.magnetizing_conductance_s, dtype=rdt, device=device)
            )
            # B_m(h) = -1/(2*pi*f*L_m); store L_m (or +inf -> 0 susceptance).
            lm = t.magnetizing_inductance_h
            lm_list.append(
                torch.as_tensor(
                    lm if lm is not None else math.inf, dtype=rdt, device=device
                )
            )
            tap_mag_list.append(
                _override(
                    param_overrides,
                    ("transformer", t.id, "tap_magnitude"),
                    torch.as_tensor(t.tap.ratio_magnitude, dtype=rdt, device=device),
                )
            )
            tap_shift_list.append(
                _override(
                    param_overrides,
                    ("transformer", t.id, "tap_shift_deg"),
                    torch.as_tensor(t.tap.shift_deg, dtype=rdt, device=device),
                )
            )
        r = torch.stack(r_list, 0)  # [K,P,P]
        ind = torch.stack(l_list, 0)
        ys = series_admittance_matrix(r, ind, f, cdt)  # [H,K,P,P] leakage admittance

        # magnetizing shunt (scalar per transformer) on the HV diagonal.
        gm = torch.stack(gm_list, 0)  # [K]
        lm = torch.stack(lm_list, 0)  # [K]
        two_pi_f = (2.0 * torch.pi) * f  # [H]
        bm = -1.0 / (two_pi_f[:, None] * lm[None])  # [H,K]; lm=inf -> 0
        ym_scalar = torch.complex(
            gm[None].expand_as(bm).to(_rdtype(cdt)), bm.to(_rdtype(cdt))
        ).to(cdt)  # [H,K]
        ym = ym_scalar[:, :, None, None] * eye.to(cdt)  # [H,K,P,P]

        # complex tap t = mag * exp(j*shift)
        mag = torch.stack(tap_mag_list, 0)  # [K]
        shift = torch.stack(tap_shift_list, 0) * (math.pi / 180.0)  # [K] radians
        t = torch.polar(mag.to(_rdtype(cdt)), shift.to(_rdtype(cdt))).to(cdt)  # [K]
        t = t[None, :, None, None]  # [1,K,1,1]
        t_conj = torch.conj(t)
        abs_t2 = (mag * mag).to(cdt)[None, :, None, None]

        y_ff = ys / abs_t2 + ym  # HV-HV
        y_ft = -ys / t_conj  # HV-LV
        y_tf = -ys / t  # LV-HV
        y_tt = ys  # LV-LV
        block = torch.cat(
            [torch.cat([y_ff, y_ft], dim=-1), torch.cat([y_tf, y_tt], dim=-1)],
            dim=-2,
        )  # [H,K,2P,2P]
        rows, cols = _series_terminal_indices(group, index, device)
        y = scatter_blocks_into(y, block, rows, cols)
    return y


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
) -> Tensor:
    """Voltage-dependent ZIP nodal current ``I_device(V)`` absorbed by loads/gens.

    For every in-service :class:`Load` / :class:`Generator`, the per-terminal
    (per phase) current drawn under the ZIP model is

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
        Optional differentiability hook; keys ``("load"|"generator", id, "p_nom_w"
        |"q_nom_var"|"p_nom_per_phase_w"|"q_nom_per_phase_var")`` inject leaf P/Q.

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

    node_map = {nd.id: nd for nd in grid.nodes}
    by_p: dict[int, list] = {}
    for a in loads:
        by_p.setdefault(len(a.phases), []).append(a)

    for p, group in by_p.items():
        p_list, q_list, v0_list, zipp_list, zipq_list, row_list = [], [], [], [], [], []
        for a in group:
            kind = "load" if isinstance(a, Load) else "generator"
            sign = 1.0 if isinstance(a, Load) else -1.0
            node = node_map[a.node]
            u_ln = phase_voltage_magnitude(node.u_rated_v, len(node.phases))

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

            p_t = _override(
                param_overrides,
                (kind, a.id, "p_nom_per_phase_w"),
                _per_phase_power_tensor(p_total, p_per, p, rdt, device),
            )
            q_t = _override(
                param_overrides,
                (kind, a.id, "q_nom_per_phase_var"),
                _per_phase_power_tensor(q_total, q_per, p, rdt, device),
            )
            p_list.append(sign * p_t)  # [P]
            q_list.append(sign * q_t)  # [P]
            u_ln_t = (
                u_ln
                if isinstance(u_ln, Tensor)
                else torch.as_tensor(u_ln, dtype=rdt, device=device)
            )
            v0_list.append(
                u_ln_t.to(dtype=rdt, device=device).reshape(()).expand(p)
            )  # [P]
            zp, zq = _zip_coeffs(a, rdt, device)
            zipp_list.append(zp)  # [3]
            zipq_list.append(zq)
            row_list.append([index.row(a.node, ph) for ph in a.phases])

        k = len(p_list)
        # Stack with K at dim -2 so any per-load batch dims stay leading and
        # broadcast against the [*b, H, K, P] voltage tensor: [*pbatch, K, P].
        p_pp = torch.stack(p_list, -2)  # [*pbatch, K, P]
        q_pp = torch.stack(q_list, -2)
        v0 = torch.stack(v0_list, -2)  # [K, P] (no power batch)
        zip_p = torch.stack(zipp_list, 0)  # [K, 3]
        zip_q = torch.stack(zipq_list, 0)
        rows = torch.as_tensor(row_list, dtype=torch.int64, device=device)  # [K,P]

        # Insert a singleton H axis into power tensors so they broadcast over H:
        # [*pbatch, K, P] -> [*pbatch, 1, K, P].
        p_pp = p_pp.unsqueeze(-3)
        q_pp = q_pp.unsqueeze(-3)

        # Gather terminal voltages Vt at these rows: v[*batch, H, N] -> [*b, H, K, P].
        flat_rows = rows.reshape(-1)  # [K*P]
        vt = v.index_select(-1, flat_rows)  # [*b, H, K*P]
        vt = vt.reshape(*batch_lead, h, k, p)  # [*b, H, K, P]

        vmag = torch.abs(vt)  # [*b, H, K, P] real
        ratio = vmag / v0  # |Vt| / |V0|  -> broadcasts [K,P] over [*b,H,K,P]

        # ZIP scaling per power component: z*ratio^2 + i*ratio + p.
        z_p, i_p, pp_p = zip_p[..., 0], zip_p[..., 1], zip_p[..., 2]  # [K]
        z_q, i_q, pp_q = zip_q[..., 0], zip_q[..., 1], zip_q[..., 2]
        scale_p = (
            z_p[..., None] * ratio**2 + i_p[..., None] * ratio + pp_p[..., None]
        )  # [*b,H,K,P]
        scale_q = z_q[..., None] * ratio**2 + i_q[..., None] * ratio + pp_q[..., None]

        s_eff = torch.complex(p_pp * scale_p, q_pp * scale_q).to(cdt)  # [*b,H,K,P]
        # I_term = conj(S_eff) / conj(Vt).
        i_term = torch.conj(s_eff) / torch.conj(vt)  # [*b,H,K,P]

        out = _scatter_injection(out, i_term, rows)

    return out


__all__ = [
    "YBus",
    "assemble_ybus",
    "assemble_network_ybus",
    "build_injections",
    "device_current_injections",
    "node_phase_index",
    "NodePhaseIndex",
]
