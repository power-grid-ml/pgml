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
    """Assemble the complex nodal admittance ``Y(f)`` for a materialised ``grid``.

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

    y = _stamp_lines(grid, f, y, index, cdt, rdt, device, param_overrides)
    y = _stamp_switches(grid, f, y, index, cdt, rdt, device, param_overrides)
    y = _stamp_generic_branches(grid, f, y, index, cdt, rdt, device, param_overrides)
    y = _stamp_shunt_reactors(grid, f, y, index, cdt, rdt, device, param_overrides)
    y = _stamp_transformers(grid, f, y, index, cdt, rdt, device, param_overrides)
    y = _stamp_sources(grid, f, y, index, cdt, rdt, device, param_overrides)
    y = _stamp_shunt_appliances(grid, f, y, index, cdt, rdt, device, param_overrides)
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
    lines = [b for b in grid.branches if isinstance(b, Line) and b.in_service]
    if not lines:
        return y
    # Group by phase count so each group stacks into a rectangular tensor.
    return _stamp_line_groups(lines, f, y, index, cdt, rdt, device, param_overrides)


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


__all__ = [
    "YBus",
    "assemble_ybus",
    "build_injections",
    "node_phase_index",
    "NodePhaseIndex",
]
