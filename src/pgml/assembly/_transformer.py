"""Vector-group two-winding transformer primitive (winding-incidence form).

A two-winding three-phase transformer couples two windings whose terminals connect
to the bus phases through a connection topology (wye / grounded-wye / delta). The
nodal admittance is built in the WINDING-VOLTAGE domain and mapped to the bus
phase rows by a constant real incidence ``N`` — exactly the pattern used for
connection-aware loads in :mod:`pgml.assembly._incidence`::

    Y_node = Nᵀ · Y_winding · N

with the per-phase-pair winding primitive (leakage admittance ``y`` referred to the
TO-side / LV winding coil, off-nominal-and-base turns ratio ``τ``)::

    Y_winding = [[ (y/τ²)·I_P ,  −(y/τ)·I_P ],
                 [ −(y/τ)·I_P ,    y·I_P    ]]      (shape [H, K, 2P, 2P])

and the block-diagonal incidence ``N = blockdiag(N_hv, N_lv)`` (shape ``[2P, 2P]``).

Per-side incidence (``P == 3``):

- ``wye_grounded`` -> ``I_3``. The neutral is solidly grounded to the system
  reference, so each phase coil returns to ground; zero-sequence current has a
  path. Reduces the self block to ``y·I`` — identical to the historical diagonal
  stamp for the positive sequence.
- ``delta`` -> the circulant difference matrix ``M`` (or its transpose, selecting
  the clock / vector group). ``M·[1,1,1]ᵀ = 0``, so a delta winding BLOCKS the
  zero sequence (it circulates inside the delta loop): triplen / residual
  harmonics injected on the wye side do not propagate through the delta side. The
  delta also injects the intrinsic ``√3`` magnitude and ``±30°`` clock shift
  (``M·V⁺ = √3·∠+30°``), so the nominal ratio and vector-group phase shift come
  from the connection + rated voltages, NOT from an explicit complex tap.
- ``wye`` (ungrounded) -> the zero-sequence projection ``P = I − (1/3)·11ᵀ``.
  ``P`` is idempotent, so ``Pᵀ(y·I)P = y·P``, the textbook ``Y_II`` self block; an
  ungrounded-wye neutral floats and blocks the zero sequence with no separate
  Kron reduction.

The turns ratio ``τ`` is the ratio of rated COIL voltages (a delta coil is rated
at the line-to-line voltage, a wye coil at the line-to-neutral voltage), times the
off-nominal tap magnitude::

    coil_rated = u_rated            (delta winding)
    coil_rated = u_rated / √3       (wye winding)
    τ = (coil_rated_from / coil_rated_to) · tap.ratio_magnitude

For a Dyn 20 kV / 0.4 kV unit, ``τ = 20000 / (400/√3) = √3·(20000/400) = √3·n_LL``.
The ``√3`` then cancels against ``M`` so the assembled positive-sequence block is
identical to ``y/n_LL²`` (HV self) and ``y/n_LL`` (coupling) — the historical
off-nominal-tap pi — while the zero sequence is now modelled correctly.

The magnetizing (core-loss) shunt ``y_m`` is added to the HV terminal phase
diagonal directly (referred to the HV line voltage), outside the leakage
incidence transform — unchanged from the historical stamp.

All matrices ``N`` are real, constant topology (no autograd through ``N``);
``Y_winding`` carries the differentiable ``y`` and ``τ`` so gradients flow to the
series R / L and the tap. Vectorised over the K transformers of a group (shared
incidence) and over the H frequencies — no Python loop over individual units on
the tape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from pgml import config
from pgml.errors import ModelingError
from pgml.schemas.grid_schema import WindingConnection

# The delta circulant difference matrix (Kersting's [D]); element k spans phase k
# and phase (k+1) % 3. Its transpose selects the opposite clock parity. Rows and
# columns sum to zero -> a delta winding blocks the zero sequence.
_DELTA_M = ((1.0, -1.0, 0.0), (0.0, 1.0, -1.0), (-1.0, 0.0, 1.0))


@dataclass(frozen=True)
class SideConn:
    """A resolved per-winding connection: incidence KIND and grounded flag."""

    kind: str  # "wye_grounded" | "wye" | "delta"
    grounded: bool


@dataclass(frozen=True)
class VectorGroup:
    """Resolved vector group of a two-winding transformer.

    ``clock`` is the IEC clock number; the canonical HV->LV complex ratio carries
    the phase angle ``shift_deg = clock·30`` (mod 360) — the same convention the
    pandapower / MATPOWER off-nominal tap uses (``shift_degree = clock·30``, with
    a positive angle making the LV phasor LAG the HV one). ``Dyn1`` -> 30° (LV
    lags), ``Dyn11`` -> 330° ≡ −30° (LV leads).
    """

    from_side: SideConn
    to_side: SideConn
    clock: int

    @property
    def shift_deg(self) -> float:
        """Canonical HV->LV phase-shift angle ``clock·30`` (mod 360)."""
        return float((self.clock * 30) % 360)

    @property
    def is_delta_wye(self) -> bool:
        """True iff exactly one winding is delta (a Dyn / Yd phase-shifting group)."""
        return (self.from_side.kind == "delta") != (self.to_side.kind == "delta")

    @property
    def clock_transpose(self) -> bool:
        """Whether the HV delta incidence uses ``Mᵀ`` (vs ``M``).

        Pinned against the scalar off-nominal-tap pi (and a live OpenDSS export):
        ``Mᵀ`` reproduces the positive-sequence coupling of a +30° tap (LV lags,
        clock 1 / Dyn1); ``M`` that of a −30° tap (LV leads, clock 11 / Dyn11).
        So the transpose is taken when the shift angle lies in ``(0°, 180°)``.
        """
        return math.sin(math.radians(self.shift_deg)) > 0.0


def _classify(conn: WindingConnection) -> SideConn:
    """Map a :class:`WindingConnection` to the incidence kind used here."""
    if conn == WindingConnection.WYE_GROUNDED:
        return SideConn("wye_grounded", grounded=True)
    if conn == WindingConnection.WYE:
        return SideConn("wye", grounded=False)
    if conn == WindingConnection.DELTA:
        return SideConn("delta", grounded=False)
    raise ModelingError(
        f"transformer winding connection {conn!r} is not modelled "
        "(zigzag windings are not supported yet)."
    )


def resolve_vector_group(t) -> VectorGroup:
    """Resolve a transformer's vector group: explicit connections + clock, else config.

    ``from_connection`` / ``to_connection`` come from the schema when set, else the
    config defaults ``transformer.vector_group.{from,to}`` (Dyn11). The clock is read
    from ``tap.shift_deg`` (rounded to the nearest 30°) when the connections are
    explicit, else ``transformer.vector_group.clock``.
    """
    if t.from_connection is not None and t.to_connection is not None:
        from_conn, to_conn = t.from_connection, t.to_connection
        clock = int(round(float(t.tap.shift_deg) / 30.0)) % 12
    else:
        from_conn = WindingConnection(config.get("transformer.vector_group.from"))
        to_conn = WindingConnection(config.get("transformer.vector_group.to"))
        clock = int(config.get("transformer.vector_group.clock"))
    vg = VectorGroup(_classify(from_conn), _classify(to_conn), clock)
    # The phase-domain delta incidence here realises only the ±30° (clock 1 / 11)
    # delta-wye pairing and the in-phase (clock 0) groups; other clocks need a
    # cyclic phase permutation of the winding pairing (not yet modelled).
    if vg.is_delta_wye and clock not in (1, 11):
        raise ModelingError(
            f"delta-wye transformer clock {clock} is not modelled (only Dyn1 / "
            "Dyn11, i.e. clock 1 or 11, are supported in the phase-domain stamp)."
        )
    if not vg.is_delta_wye and clock % 6 != 0:
        raise ModelingError(
            f"transformer clock {clock} for a non-phase-shifting group is not "
            "modelled (only clock 0 / 6 are supported for wye-wye / delta-delta)."
        )
    return vg


def group_key(vg: VectorGroup, p: int) -> tuple:
    """Hashable key grouping transformers that share one incidence ``N``."""
    return (vg.from_side.kind, vg.to_side.kind, vg.clock_transpose, p)


def side_incidence(
    side: SideConn, p: int, clock_transpose: bool, rdt, device
) -> Tensor:
    """Real per-winding incidence ``N_side`` ``[P, P]`` (constant topology).

    ``wye_grounded`` -> ``I_P``; ``wye`` -> ``I − 11ᵀ/P`` (zero-seq projection);
    ``delta`` -> the circulant ``M`` (``clock_transpose`` swaps to ``Mᵀ``).
    """
    eye = torch.eye(p, dtype=rdt, device=device)
    if side.kind == "wye_grounded":
        return eye
    if side.kind == "wye":
        if p != 3:
            raise ModelingError(
                "ungrounded-wye transformer winding is only modelled for 3 phases."
            )
        ones = torch.ones((p, p), dtype=rdt, device=device)
        return eye - ones / p
    if p != 3:
        raise ModelingError("delta transformer winding is only modelled for 3 phases.")
    m = torch.as_tensor(_DELTA_M, dtype=rdt, device=device)
    return m.t().contiguous() if clock_transpose else m


def block_incidence(vg: VectorGroup, p: int, rdt, device) -> Tensor:
    """Block-diagonal incidence ``N = blockdiag(N_hv, N_lv)`` ``[2P, 2P]``.

    The clock transpose is passed to BOTH sides; ``side_incidence`` only consumes it
    for a delta winding (ignored for wye / grounded-wye), so it correctly selects the
    clock orientation whether the delta is on the HV (Dyn) or LV (Yd) side.
    """
    n_hv = side_incidence(vg.from_side, p, vg.clock_transpose, rdt, device)
    n_lv = side_incidence(vg.to_side, p, vg.clock_transpose, rdt, device)
    n = torch.zeros((2 * p, 2 * p), dtype=rdt, device=device)
    n[:p, :p] = n_hv
    n[p:, p:] = n_lv
    return n


def nominal_turns_ratio(vg: VectorGroup, u_from: Tensor, u_to: Tensor) -> Tensor:
    """Nominal COIL turns ratio ``τ`` from rated LINE voltages + connections.

    A delta coil is rated at the line-to-line voltage; a wye coil at the
    line-to-neutral voltage ``u_rated/√3``. ``τ = coil_from / coil_to``.
    """
    sqrt3 = math.sqrt(3.0)
    coil_from = u_from if vg.from_side.kind == "delta" else u_from / sqrt3
    coil_to = u_to if vg.to_side.kind == "delta" else u_to / sqrt3
    return coil_from / coil_to


def winding_leakage_block(y_se: Tensor, tau: Tensor, n_block: Tensor) -> Tensor:
    """Leakage nodal block ``Nᵀ·Y_winding·N`` ``[H, K, 2P, 2P]`` (complex).

    ``y_se`` ``[H, K]`` is the leakage admittance referred to the TO-side coil;
    ``tau`` ``[K]`` the coil turns ratio; ``n_block`` ``[2P, 2P]`` the (real,
    constant) block incidence.
    """
    p = n_block.shape[0] // 2
    cdt = y_se.dtype
    tau_c = tau.to(cdt)[None, :]  # [1,K]
    a = y_se / (tau_c * tau_c)  # [H,K]  HV-HV
    b = -(y_se / tau_c)  # [H,K]         HV-LV / LV-HV
    d = y_se  # [H,K]                    LV-LV

    eye = torch.eye(p, dtype=cdt, device=y_se.device)
    top = torch.cat([a[..., None, None] * eye, b[..., None, None] * eye], dim=-1)
    bot = torch.cat([b[..., None, None] * eye, d[..., None, None] * eye], dim=-1)
    y_w = torch.cat([top, bot], dim=-2)  # [H,K,2P,2P]

    nt = n_block.t().to(cdt)
    n_c = n_block.to(cdt)
    return nt @ y_w @ n_c  # broadcast over [H, K]


__all__ = [
    "SideConn",
    "VectorGroup",
    "resolve_vector_group",
    "group_key",
    "side_incidence",
    "block_incidence",
    "nominal_turns_ratio",
    "winding_leakage_block",
]
