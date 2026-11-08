"""Transformer star points as terminals: neutral rows, earthing impedances, centre taps.

The vector-group stamp of :mod:`pgml.assembly._transformer` maps each winding's coils to
the bus PHASE rows and puts a wye winding's star point either at the earth reference
(grounded) or eliminates it by the zero-sequence projection (ungrounded). This module
extends the same winding-incidence form ``Y = Nᵀ·Y_coil·N`` with the star point as a
terminal of its own, which covers three cases the phase-row form cannot express:

- the star point is the node's ``Phase.N`` row (bonded to an explicit neutral
  conductor), optionally earthed through a finite impedance on that row;
- the star point is an internal point earthed through a finite impedance (a neutral
  earthing resistor, the earth electrode of a single-wire earth-return primary); the
  point is eliminated exactly by a Schur complement (Kron reduction);
- a single-phase unit with a centre-tapped secondary (split phase, X1-N-X2): three
  windings on one core in their star (T) equivalent, the centre tap being the
  secondary's star point.

Coil domain and incidence
-------------------------
Every winding coil ``c`` spans two terminals; its voltage is ``e = N·v_ext`` with
``v_ext`` the extended terminal voltages ``[from rows, (from star), to rows, (to
star)]``. A wye coil ``k`` contributes ``+B[k, :]`` on the phase terminals and
``−B[k, :]·1`` on the star terminal, where ``B`` is the side's signed phase permutation
selected by the clock (:func:`pgml.assembly._transformer._incidence_pair`); a delta coil
is the circulant difference ``M``. A solidly earthed star point has no terminal (its
voltage is the reference), which reproduces the vector-group stamp's ``I`` block. An
internal star point is eliminated after its earthing admittance ``y_g`` is added
(``y_g = 0`` for an ungrounded wye, which reproduces the projection ``P = I − 11ᵀ/3``).

Coil primitives
---------------
- Three-phase and per-coil units: the two-winding primitive of
  :func:`pgml.assembly._transformer.winding_leakage_block` (leakage referred to the
  to-side coil, coil turns ratio ``τ``), including the sequence-aware leakage matrix.
- Centre tap: with the star branches ``z_H`` (primary) and ``z_X`` (each half), both
  referred to the full secondary coil, the referred coil admittance of three windings on
  an ideal core is ``Y' = D − d·dᵀ/Σd`` (``D = diag(d)``, ``d = (1/z_H, 1/z_X, 1/z_X)``);
  the coil-domain primitive is ``T·Y'·T`` with ``T = diag(1/τ, 2, 2)`` (``τ`` the
  primary-to-full-secondary coil ratio, a half coil is rated half the full coil).

Everything here except the tensor math in :func:`centre_tap_coil_primitive`,
:func:`earthing_admittance` and :func:`kron_eliminate` is constant topology (python
tuples, no tensors, no autograd).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from pgml.errors import ModelingError
from pgml.schemas.grid_schema import Phase, WindingConnection

from ._transformer import (
    SideConn,
    VectorGroup,
    _classify,
    _grounding_is_finite,
    _incidence_pair,
    resolve_vector_group,
)

#: Star-point kinds of one winding side.
#: ``none`` (delta / zigzag: no star point terminal), ``earth`` (solidly earthed, no
#: terminal), ``internal`` (eliminated after its earthing admittance), ``row`` (the
#: node's ``Phase.N`` row).
STAR_KINDS = ("none", "earth", "internal", "row")


def uses_terminal_model(t) -> bool:
    """True when a transformer needs the star-point terminal model of this module.

    That is the case for a centre-tapped unit, a winding with its star point on a
    ``Phase.N`` row, a finite earthing impedance on either side, unequal from/to phase
    counts and a single phase-to-phase (two-phase delta) winding. Every other unit is
    stamped by the vector-group model unchanged.
    """
    if getattr(t, "centre_tap", None) is not None:
        return True
    if Phase.N in t.from_phases or Phase.N in t.to_phases:
        return True
    if len(t.from_phases) != len(t.to_phases):
        return True
    if _grounding_is_finite(t.from_grounding) or _grounding_is_finite(t.to_grounding):
        return True
    for conn, phases in (
        (t.from_connection, t.from_phases),
        (t.to_connection, t.to_phases),
    ):
        if conn == WindingConnection.DELTA and len(phases) == 2:
            return True
    return False


@dataclass(frozen=True)
class TerminalLayout:
    """Constant topology of one transformer in the terminal model.

    ``model`` is ``three_phase`` (a clocked vector group on three coils per side),
    ``coils`` (``P`` independent coil pairs, clock 0 or 6) or ``centre_tap``.
    ``incidence`` is the real ``[C, T]`` coil-to-extended-terminal incidence;
    ``internal`` the extended slots eliminated after stamping; ``from_star`` /
    ``to_star`` the slot index of each side's star point (``None`` without a
    terminal) and ``*_earthed`` whether an earthing admittance is stamped there.
    ``phase_slots`` are the extended slots of the phase terminals (three-phase
    magnetizing placement). ``n_coils`` is the coil count of the from side.
    """

    model: str
    n_coils: int
    incidence: tuple
    internal: tuple
    from_star: Optional[int]
    to_star: Optional[int]
    from_earthed: bool
    to_earthed: bool
    phase_slots: tuple

    @property
    def key(self) -> tuple:
        """Hashable grouping key: units sharing it stack into one batched block."""
        return (
            self.model,
            self.n_coils,
            self.incidence,
            self.internal,
            self.from_star,
            self.to_star,
            self.from_earthed,
            self.to_earthed,
            self.phase_slots,
        )


@dataclass(frozen=True)
class _Side:
    live: tuple  # phase terminals (no Phase.N), in tuple order
    has_row: bool  # Phase.N listed (star point on the neutral row)
    kind: str  # incidence kind of _transformer._classify
    star: str  # one of STAR_KINDS
    earthed: bool  # an earthing admittance is stamped on the star terminal


def _side(t, side: str, conn: WindingConnection) -> _Side:
    phases = tuple(getattr(t, f"{side}_phases"))
    grounding = getattr(t, f"{side}_grounding")
    has_row = Phase.N in phases
    live = tuple(p for p in phases if p != Phase.N)
    kind = _classify(conn).kind
    finite = _grounding_is_finite(grounding)
    where = f"transformer {t.id} {side} side"
    if kind in ("zigzag", "zigzag_grounded"):
        if has_row or finite:
            raise ModelingError(
                f"{where}: a zigzag star point on a neutral row or behind an earthing "
                "impedance is not modelled (solid earthing only)."
            )
        return _Side(live, False, kind, "none", False)
    if kind == "delta":
        if has_row or finite:
            raise ModelingError(
                f"{where}: a delta winding has no star point; drop Phase.N from the "
                "phases and the GroundingImpedance."
            )
        return _Side(live, False, kind, "none", False)
    if kind == "wye" and finite:
        raise ModelingError(
            f"{where}: an earthing impedance on an ungrounded (wye) winding; use "
            "wye_grounded to earth the star point through it."
        )
    if has_row:
        if kind == "wye_grounded" and not finite:
            raise ModelingError(
                f"{where}: the star point is the neutral row and wye_grounded, but "
                "the row cannot be earthed solidly (it would be the reference "
                "itself). Give a finite GroundingImpedance (the earth resistance at "
                "the unit), use wye to bond the star point to the neutral only, or "
                "drop Phase.N to earth the star point at the reference."
            )
        return _Side(live, True, kind, "row", kind == "wye_grounded")
    if kind == "wye_grounded":
        if finite:
            return _Side(live, False, kind, "internal", True)
        return _Side(live, False, kind, "earth", False)
    return _Side(live, False, kind, "internal", False)  # floating wye star point


def _slots(t, sf: _Side, st: _Side) -> tuple[dict, int]:
    """Extended slot index of every from/to terminal and star point."""
    slot: dict = {}
    pos = 0
    for p in t.from_phases:
        slot[("from", p)] = pos
        pos += 1
    if sf.star == "internal":
        slot[("from", "star")] = pos
        pos += 1
    elif sf.star == "row":
        slot[("from", "star")] = slot[("from", Phase.N)]
    for p in t.to_phases:
        slot[("to", p)] = pos
        pos += 1
    if st.star == "internal":
        slot[("to", "star")] = pos
        pos += 1
    elif st.star == "row":
        slot[("to", "star")] = slot[("to", Phase.N)]
    return slot, pos


def _wye_rows(side: str, sd: _Side, b_mat, slot: dict, n_ext: int) -> list:
    """Coil rows of a wye side: ``+B`` on the phases, ``−B·1`` on the star point."""
    rows = []
    for k in range(len(b_mat)):
        row = [0.0] * n_ext
        for j, ph in enumerate(sd.live):
            row[slot[(side, ph)]] += b_mat[k][j]
        if sd.star in ("internal", "row"):
            row[slot[(side, "star")]] -= sum(b_mat[k])
        rows.append(row)
    return rows


def _matrix_rows(side: str, sd: _Side, a_mat, slot: dict, n_ext: int) -> list:
    rows = []
    for k in range(len(a_mat)):
        row = [0.0] * n_ext
        for j, ph in enumerate(sd.live):
            row[slot[(side, ph)]] += a_mat[k][j]
        rows.append(row)
    return rows


def _single_coil_rows(side: str, sd: _Side, slot: dict, n_ext: int, sign: float):
    """One coil per live phase (wye) or one phase-to-phase coil (two-phase delta)."""
    if sd.kind == "delta":
        if len(sd.live) != 2:
            raise ModelingError(
                "a delta winding outside the three-phase vector group must be one "
                f"phase-to-phase coil on two phases (got {len(sd.live)})."
            )
        row = [0.0] * n_ext
        row[slot[(side, sd.live[0])]] = sign
        row[slot[(side, sd.live[1])]] = -sign
        return [row]
    if sd.kind not in ("wye", "wye_grounded"):
        raise ModelingError(f"{sd.kind} windings are only modelled with three phases.")
    eye = [
        [sign if i == j else 0.0 for j in range(len(sd.live))]
        for i in range(len(sd.live))
    ]
    return _wye_rows(side, sd, eye, slot, n_ext)


def _connections(t) -> tuple[WindingConnection, WindingConnection]:
    if t.from_connection is None or t.to_connection is None:
        vg = resolve_vector_group(t, check_grounding=False)
        return (
            WindingConnection(vg.from_side.kind),
            WindingConnection(vg.to_side.kind),
        )
    return t.from_connection, t.to_connection


def _polarity_clock(t) -> float:
    """``+1`` (clock 0) or ``−1`` (clock 6) of a unit without a three-phase group."""
    shift = float(t.tap.shift_deg) % 360.0
    if min(abs(shift), abs(shift - 360.0)) < 1.0e-3:
        return 1.0
    if abs(shift - 180.0) < 1.0e-3:
        return -1.0
    raise ModelingError(
        f"transformer {t.id}: a unit without a three-phase vector group has clock 0 "
        f"or 6 only (tap.shift_deg 0 or 180), got {shift}."
    )


def resolve_terminal_layout(t) -> TerminalLayout:
    """The constant topology of a transformer in the terminal model.

    Raises :class:`~pgml.errors.ModelingError` for a combination that has no
    physical meaning or is not modelled (a delta or zigzag star point, a solidly
    earthed neutral row, unequal coil counts, a clock no topology realises).
    """
    from_conn, to_conn = _connections(t)
    sf = _side(t, "from", from_conn)
    st = _side(t, "to", to_conn)
    slot, n_ext = _slots(t, sf, st)

    if getattr(t, "centre_tap", None) is not None:
        model = "centre_tap"
        rows = _single_coil_rows("from", sf, slot, n_ext, 1.0)
        if len(rows) != 1:
            raise ModelingError(
                f"transformer {t.id}: a centre-tap primary is one winding."
            )
        x1, x2 = st.live
        for outer, sign in ((x1, 1.0), (x2, -1.0)):
            row = [0.0] * n_ext
            row[slot[("to", outer)]] = sign
            if st.star in ("internal", "row"):
                row[slot[("to", "star")]] = -sign
            rows.append(row)
        n_coils = 1
    elif len(sf.live) == 3 and len(st.live) == 3:
        model = "three_phase"
        vg = resolve_vector_group(t, n_phases=3, check_grounding=False)
        topo = VectorGroup(
            SideConn(_topology_kind(vg.from_side.kind), vg.from_side.grounded),
            SideConn(_topology_kind(vg.to_side.kind), vg.to_side.grounded),
            vg.clock,
        )
        a_from, a_to = _incidence_pair(topo)
        rows = []
        for side, sd, a_mat in (("from", sf, a_from), ("to", st, a_to)):
            if sd.kind in ("wye", "wye_grounded"):
                rows += _wye_rows(side, sd, a_mat, slot, n_ext)
            else:
                rows += _matrix_rows(side, sd, a_mat, slot, n_ext)
        n_coils = 3
    else:
        model = "coils"
        sign = _polarity_clock(t)
        rows_f = _single_coil_rows("from", sf, slot, n_ext, 1.0)
        rows_t = _single_coil_rows("to", st, slot, n_ext, sign)
        if len(rows_f) != len(rows_t):
            raise ModelingError(
                f"transformer {t.id}: {len(rows_f)} from-side coil(s) cannot pair "
                f"with {len(rows_t)} to-side coil(s)."
            )
        rows = rows_f + rows_t
        n_coils = len(rows_f)

    internal = tuple(
        slot[(side, "star")]
        for side, sd in (("from", sf), ("to", st))
        if sd.star == "internal"
    )
    phase_slots = tuple(slot[("from", p)] for p in sf.live) + tuple(
        slot[("to", p)] for p in st.live
    )
    return TerminalLayout(
        model=model,
        n_coils=n_coils,
        incidence=tuple(tuple(r) for r in rows),
        internal=internal,
        from_star=slot.get(("from", "star")),
        to_star=slot.get(("to", "star")),
        from_earthed=sf.earthed,
        to_earthed=st.earthed,
        phase_slots=phase_slots,
    )


def _topology_kind(kind: str) -> str:
    """Signed-permutation topology of a side (a wye star point is a terminal here)."""
    return "wye_grounded" if kind == "wye" else kind


def earthing_admittance(
    r_ohm: Tensor, x_ohm: Tensor, f: Tensor, f0: float, cdt: torch.dtype
) -> Tensor:
    """Earthing admittance ``1/(R + j·(f/f0)·X)`` ``[H, K]`` from ``[K]`` R and X.

    The resistance is frequency-independent; the reactance (given at ``f0``) scales
    with the harmonic order. Differentiable in ``r_ohm`` and ``x_ohm``.
    """
    h = (f / f0)[:, None]
    z = torch.complex(r_ohm[None, :].expand(h.shape[0], -1), h * x_ohm[None, :])
    return 1.0 / z.to(cdt)


def centre_tap_coil_primitive(z_primary: Tensor, z_half: Tensor, tau: Tensor) -> Tensor:
    """Coil-domain primitive ``[H, K, 3, 3]`` of a centre-tapped unit (coils H, X1, X2).

    ``z_primary`` ``[H, K]`` is the primary star branch referred to the full secondary
    coil, ``z_half`` ``[H, K]`` one half winding's branch in its own coil ohms, ``tau``
    ``[K]`` the primary-to-full-secondary coil turns ratio (off-nominal tap included).
    The three windings meet at an ideal core node (no magnetizing branch here), so
    the referred primitive is ``D − d·dᵀ/Σd`` and the coil quantities follow through
    ``T = diag(1/τ, 2, 2)``. Device and dtype follow ``z_primary``.
    """
    y_h = 1.0 / z_primary
    y_x = 1.0 / (4.0 * z_half)  # half branch referred to the full secondary coil
    d = torch.stack([y_h, y_x, y_x], dim=-1)  # [H,K,3]
    total = d.sum(dim=-1)[..., None, None]
    y_ref = torch.diag_embed(d) - d[..., :, None] * d[..., None, :] / total
    one = torch.ones_like(tau)
    t_vec = torch.stack([1.0 / tau, 2.0 * one, 2.0 * one], dim=-1).to(y_ref.dtype)
    return y_ref * t_vec[None, :, :, None] * t_vec[None, :, None, :]


def kron_eliminate(block: Tensor, internal: tuple) -> Tensor:
    """Eliminate the ``internal`` slots of ``block`` ``[..., T, T]`` exactly.

    Schur complement ``Y_kk − Y_ki·Y_ii⁻¹·Y_ik`` (a batched solve, differentiable);
    the internal points carry no external current.
    """
    if not internal:
        return block
    n = block.shape[-1]
    keep = [i for i in range(n) if i not in internal]
    ki = torch.as_tensor(keep, dtype=torch.int64, device=block.device)
    ii = torch.as_tensor(list(internal), dtype=torch.int64, device=block.device)
    y_kk = block.index_select(-2, ki).index_select(-1, ki)
    y_ki = block.index_select(-2, ki).index_select(-1, ii)
    y_ik = block.index_select(-2, ii).index_select(-1, ki)
    y_ii = block.index_select(-2, ii).index_select(-1, ii)
    return y_kk - y_ki @ torch.linalg.solve(y_ii, y_ik)


__all__ = [
    "STAR_KINDS",
    "TerminalLayout",
    "centre_tap_coil_primitive",
    "earthing_admittance",
    "kron_eliminate",
    "resolve_terminal_layout",
    "uses_terminal_model",
]
