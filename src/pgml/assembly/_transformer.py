"""Vector-group two-winding transformer primitive (winding-incidence form).

A two-winding three-phase transformer couples two windings whose terminals connect
to the bus phases through a connection topology (wye / grounded-wye / delta /
zigzag). The nodal admittance is built in the WINDING-VOLTAGE domain and mapped to
the bus phase rows by a constant real incidence ``N`` — exactly the pattern used
for connection-aware loads in :mod:`pgml.assembly._incidence`::

    Y_node = Nᵀ · Y_winding · N

with the per-phase-pair winding primitive (leakage admittance ``y`` referred to the
TO-side / LV winding coil, off-nominal-and-base turns ratio ``τ``)::

    Y_winding = [[ (y/τ²)·I_P ,  −(y/τ)·I_P ],
                 [ −(y/τ)·I_P ,    y·I_P    ]]      (shape [H, K, 2P, 2P])

and the block-diagonal incidence ``N = blockdiag(N_hv, N_lv)`` (shape ``[2P, 2P]``).

Per-side building blocks (``P == 3``):

- ``wye_grounded`` -> ``I_3``. The neutral is solidly grounded to the system
  reference, so each phase coil returns to ground; zero-sequence current has a
  path. Reduces the self block to ``y·I`` — identical to the historical diagonal
  stamp for the positive sequence.
- ``delta`` -> the circulant difference matrix ``M`` (or its transpose, selecting
  the clock orientation). ``M·[1,1,1]ᵀ = 0``, so a delta winding BLOCKS the zero
  sequence (it circulates inside the delta loop): triplen / residual harmonics
  injected on the wye side do not propagate through the delta side. The delta
  also injects the intrinsic ``√3`` magnitude and ``±30°`` clock shift
  (``M·V⁺ = √3·∠+30°``), so the nominal ratio and vector-group phase shift come
  from the connection + rated voltages, NOT from an explicit complex tap.
- ``wye`` (ungrounded) -> the zero-sequence projection ``P = I − (1/3)·11ᵀ``.
  ``P`` is idempotent, so ``Pᵀ(y·I)P = y·P``, the textbook ``Y_II`` self block; an
  ungrounded-wye neutral floats and blocks the zero sequence with no separate
  Kron reduction.
- ``zigzag`` / ``zigzag_grounded`` (interconnected star): each phase leg is two
  half-coils in series opposition on ADJACENT core limbs, so the winding couples
  to the limb fluxes through the normalised circulant ``Z = (I − C)/√3`` (or its
  transpose), where ``C`` is the cyclic phase permutation. Because the limb flux
  is the quantity shared with the other winding, the zigzag topology appears in
  the OTHER side's incidence block (``Ñ_other = Z · N_other``) while the
  zigzag's own block keeps the plain star topology (``I`` grounded, ``P``
  ungrounded). Consequences, all verified numerically:

  * ``Z·V⁺ = 1·∠±30°`` — a zigzag winding shifts the clock by ±1 exactly like a
    delta; the ``1/√3`` normalisation keeps the leakage referral basis at the
    physical line-to-neutral quantity (its effective coil rating is
    ``u_rated/√3``, see :func:`nominal_turns_ratio`).
  * ``Z·[1,1,1]ᵀ = 0`` — zero-sequence MMF cancels per limb, so zero sequence
    cannot TRANSFER through a zigzag winding (the coupling and far-side self
    blocks lose their zero sequence).
  * a grounded zigzag's own self block stays ``y·I`` — the winding presents a
    LOW-impedance zero-sequence path to ground on its own side (the classic
    grounding-transformer property). The path's value equals the positive-
    sequence leakage here; the true zero-sequence leakage of a zigzag (set by
    the half-coil geometry) is typically smaller and would need an explicit
    zero-sequence override, which is not consumed yet.

Clock realisation. The IEC clock number ``c`` (LV lags HV by ``c·30°``) is
realised entirely by constant topology: each delta / zigzag winding contributes
its intrinsic ``±30°`` (the ``M`` vs ``Mᵀ`` / ``Z`` vs ``Zᵀ`` orientation), a
cyclic permutation ``C^m`` of the TO-side bus connection contributes ``m·(−120°)``
(``±4`` clock steps), and a reversed TO-winding polarity (``−1`` on the LV
incidence) contributes ``180°`` (``6`` steps). The clock PARITY is therefore fixed
by the pairing — an odd number of shifting windings (Dy, Yd, Yz, Zy) admits odd
clocks only, an even number (Yy, Dd, Dz, Zd) even clocks only — and every clock of
the right parity is reachable. The concrete orientation / permutation / polarity
combination for a requested clock is selected by matching the realised
positive-sequence rotation of the candidate incidence against ``c·30°``
(:func:`block_incidence`); the selection is exact and deterministic, and
degenerate candidates that realise the same clock produce the identical nodal
block (a real circulant is fully determined by its zero- and positive-sequence
eigenvalues). The positive-sequence coupling equals the scalar off-nominal-tap pi
``Y_ft = −y_se/conj(t)``, ``t = n·e^{j·c·30°}`` — the same convention pandapower /
MATPOWER use (positive shift = LV lags HV), pinned against a live OpenDSS export.

The turns ratio ``τ`` is the ratio of rated COIL voltages (a delta coil is rated
at the line-to-line voltage; wye and zigzag coils at the line-to-neutral voltage
— the ``1/√3`` zigzag normalisation makes its effective rating line-to-neutral
even though the physical half-coils are each rated ``u_rated/3``), times the
off-nominal tap magnitude::

    coil_rated = u_rated            (delta winding)
    coil_rated = u_rated / √3       (wye and zigzag windings)
    τ = (coil_rated_from / coil_rated_to) · tap.ratio_magnitude

For a Dyn 20 kV / 0.4 kV unit, ``τ = 20000 / (400/√3) = √3·(20000/400) = √3·n_LL``.
The ``√3`` then cancels against ``M`` so the assembled positive-sequence block is
identical to ``y/n_LL²`` (HV self) and ``y/n_LL`` (coupling) — the historical
off-nominal-tap pi — while the zero sequence is now modelled correctly.

The magnetizing (core-loss) shunt ``y_m`` is added to the HV terminal phase
diagonal directly (referred to the HV line voltage), outside the leakage
incidence transform.

All matrices ``N`` are real, constant topology (no autograd through ``N``);
``Y_winding`` carries the differentiable ``y`` and ``τ`` so gradients flow to the
series R / L and the tap. Vectorised over the K transformers of a group (shared
incidence) and over the H frequencies — no Python loop over individual units on
the tape.

References: Chen/Dillon generalized transformer model (Arrillaga & Watson,
*Computer Modelling of Electrical Power Systems*; Bazrafshan & Gatsis,
arXiv:1705.06782), extended with the zigzag limb-domain incidence.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from pgml import defaults
from pgml.errors import ModelingError
from pgml.schemas.grid_schema import WindingConnection

# The delta circulant difference matrix (Kersting's [D]); element k spans phase k
# and phase (k+1) % 3. Its transpose selects the opposite clock parity. Rows and
# columns sum to zero -> a delta winding blocks the zero sequence.
_DELTA_M = ((1.0, -1.0, 0.0), (0.0, 1.0, -1.0), (-1.0, 0.0, 1.0))

# Cyclic phase permutation (C·v)_i = v_{(i+1) mod 3}: rotates the positive
# sequence by −120° per application; leaves the zero sequence untouched.
_CYCLIC_C = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))

# Positive-sequence unit phasor set (phase A reference, ABC rotation).
_V_POS = (1.0 + 0.0j, cmath.exp(-2j * math.pi / 3), cmath.exp(2j * math.pi / 3))

_SHIFTING_KINDS = ("delta", "zigzag", "zigzag_grounded")
_ZIGZAG_KINDS = ("zigzag", "zigzag_grounded")


@dataclass(frozen=True)
class SideConn:
    """A resolved per-winding connection: incidence KIND and grounded flag."""

    kind: str  # "wye_grounded" | "wye" | "delta" | "zigzag_grounded" | "zigzag"
    grounded: bool

    @property
    def is_shifting(self) -> bool:
        """True when the winding contributes an intrinsic ±30° (delta / zigzag)."""
        return self.kind in _SHIFTING_KINDS

    @property
    def is_zigzag(self) -> bool:
        return self.kind in _ZIGZAG_KINDS


@dataclass(frozen=True)
class VectorGroup:
    """Resolved vector group of a two-winding transformer.

    ``clock`` is the IEC clock number; the canonical HV->LV complex ratio carries
    the phase angle ``shift_deg = clock·30`` (mod 360) — the same convention the
    pandapower / MATPOWER off-nominal tap uses (``shift_degree = clock·30``, with
    a positive angle making the LV phasor LAG the HV one). ``Dyn1`` -> 30° (LV
    lags), ``Dyn11`` -> 330° ≡ −30° (LV leads).

    ``shift_raw_deg`` preserves the source's exact phase-shift angle. It equals
    ``clock·30`` for every genuine vector group, but a positive-sequence dataset
    (e.g. a MATPOWER phase shifter) may carry an arbitrary angle; the single-phase
    equivalent stamp honours it exactly, while the 3-phase stamp requires a true
    clock (see :func:`resolve_vector_group`).
    """

    from_side: SideConn
    to_side: SideConn
    clock: int
    shift_raw_deg: Optional[float] = None

    @property
    def shift_deg(self) -> float:
        """Canonical HV->LV phase-shift angle ``clock·30`` (mod 360)."""
        return float((self.clock * 30) % 360)

    @property
    def shift_exact_deg(self) -> float:
        """Exact source phase shift; falls back to the canonical clock angle."""
        return self.shift_deg if self.shift_raw_deg is None else self.shift_raw_deg

    @property
    def parity_odd(self) -> bool:
        """True iff an odd number of windings is delta / zigzag (odd clocks only)."""
        return self.from_side.is_shifting != self.to_side.is_shifting


def _classify(conn: WindingConnection) -> SideConn:
    """Map a :class:`WindingConnection` to the incidence kind used here."""
    if conn == WindingConnection.WYE_GROUNDED:
        return SideConn("wye_grounded", grounded=True)
    if conn == WindingConnection.WYE:
        return SideConn("wye", grounded=False)
    if conn == WindingConnection.DELTA:
        return SideConn("delta", grounded=False)
    if conn == WindingConnection.ZIGZAG_GROUNDED:
        return SideConn("zigzag_grounded", grounded=True)
    if conn == WindingConnection.ZIGZAG:
        return SideConn("zigzag", grounded=False)
    raise ModelingError(f"transformer winding connection {conn!r} is not modelled.")


def _grounding_is_finite(grounding) -> bool:
    """True when a nonzero neutral grounding impedance is set (r=x=0 = solid)."""
    if grounding is None:
        return False
    return float(grounding.r_ohm) != 0.0 or float(grounding.x_ohm) != 0.0


def resolve_vector_group(t, n_phases: Optional[int] = None) -> VectorGroup:
    """Resolve a transformer's vector group: explicit connections + clock, else defaults.

    ``from_connection`` / ``to_connection`` come from the schema when set, else the
    modeling defaults ``transformer.vector_group.{from,to}`` (Dyn11). The clock is
    read from ``tap.shift_deg`` (rounded to the nearest 30°) when the connections
    are explicit, else ``transformer.vector_group.clock``.

    Validation (fail loud, never silently wrong):

    - The clock parity must match the winding pairing: an odd number of delta /
      zigzag windings admits odd clocks only (Dy, Yd, Yz, Zy), an even number
      even clocks only (Yy, Dd, Dz, Zd). Any clock of the right parity is
      modelled.
    - A ``tap.shift_deg`` that is not a multiple of 30° is a positive-sequence
      phase-shifter angle, not a vector group. It is exact in the single-phase
      equivalent stamp (``n_phases == 1``) and rejected in the phase-domain stamp
      (``n_phases > 1``), where the shift is realised by constant winding
      topology. Pass ``n_phases`` to enable this check; ``None`` skips it.
    - Zigzag-zigzag pairings are rejected (no such standard unit; the limb-domain
      elimination needs one plain-star or delta winding).
    - Windings are stamped solidly grounded: a finite neutral grounding impedance
      would alter the zero-sequence path, so it is rejected rather than silently
      ignored.
    """
    for side in ("from", "to"):
        if _grounding_is_finite(getattr(t, f"{side}_grounding", None)):
            raise ModelingError(
                f"transformer {t.id}: a finite {side}-side neutral grounding "
                "impedance is not modelled (windings are stamped solidly "
                "grounded); remove the GroundingImpedance or set r_ohm = "
                "x_ohm = 0."
            )
    if t.from_connection is not None and t.to_connection is not None:
        from_conn, to_conn = t.from_connection, t.to_connection
        # `tap.shift_deg` is a plain float by schema (a discrete clock selector,
        # never a gradient leaf); the shift is realised by the constant incidence.
        shift = float(t.tap.shift_deg)
        clock = int(round(shift / 30.0)) % 12
        off_clock = abs(shift - round(shift / 30.0) * 30.0)
        if off_clock > 1.0e-3 and n_phases is not None and n_phases > 1:
            raise ModelingError(
                f"transformer {t.id}: tap.shift_deg = {shift} is not a multiple "
                "of 30° — an arbitrary phase-shifter angle is only representable "
                "in the single-phase positive-sequence equivalent; the phase-"
                "domain stamp realises the shift by winding topology (clock·30°)."
            )
        vg = VectorGroup(_classify(from_conn), _classify(to_conn), clock, shift)
    else:
        from_conn = WindingConnection(defaults.get("transformer.vector_group.from"))
        to_conn = WindingConnection(defaults.get("transformer.vector_group.to"))
        clock = int(defaults.get("transformer.vector_group.clock"))
        vg = VectorGroup(_classify(from_conn), _classify(to_conn), clock)

    if vg.from_side.is_zigzag and vg.to_side.is_zigzag:
        raise ModelingError(
            f"transformer {t.id}: zigzag-zigzag winding pairing is not modelled."
        )
    if vg.parity_odd != (clock % 2 == 1):
        pairing = f"{vg.from_side.kind} / {vg.to_side.kind}"
        allowed = (
            "odd (1, 3, 5, 7, 9, 11)" if vg.parity_odd else "even (0, 2, 4, 6, 8, 10)"
        )
        raise ModelingError(
            f"transformer {t.id}: clock {clock} is inconsistent with the "
            f"{pairing} pairing — each delta or zigzag winding contributes an "
            f"intrinsic ±30°, so only {allowed} clock numbers exist for this "
            "pairing."
        )
    return vg


def group_key(vg: VectorGroup, p: int) -> tuple:
    """Hashable key grouping transformers that share one incidence ``N``."""
    return (vg.from_side.kind, vg.to_side.kind, vg.clock, p)


# ---------------------------------------------------------------------------
# Constant-topology incidence construction (pure Python; no tensors, no tape)
# ---------------------------------------------------------------------------
def _mat_mul(a, b):
    """3x3 dense product of tuple-matrices."""
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _mat_scale(a, s: float):
    return tuple(tuple(s * a[i][j] for j in range(3)) for i in range(3))


def _pos_seq_eigenvalue(a) -> complex:
    """Eigenvalue of a real circulant-algebra matrix on the positive sequence.

    Every matrix built here (products of ``I``, ``C``, ``M``, the wye projection
    and scalars) is circulant, so ``V⁺`` is an exact eigenvector; the eigenvalue
    is read off the first component of ``A·V⁺``.
    """
    return sum(a[0][j] * _V_POS[j] for j in range(3))


def _identity3():
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _wye_projection3():
    third = 1.0 / 3.0
    return tuple(
        tuple((1.0 if i == j else 0.0) - third for j in range(3)) for i in range(3)
    )


def _transpose(a):
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))


def _cyclic_power(m: int):
    out = _identity3()
    for _ in range(m % 3):
        out = _mat_mul(_CYCLIC_C, out)
    return out


def _star_topology(kind: str):
    """Own-side bus topology of a star-type winding (identity or projection)."""
    return _wye_projection3() if kind in ("wye", "zigzag") else _identity3()


def _side_candidates(side: SideConn):
    """Candidate (own_topology, mirror_factor) pairs for one winding.

    ``own_topology`` right-composes with the side's bus voltages; a zigzag's
    circulant ``Z = I − C`` couples through the LIMB fluxes and therefore
    left-multiplies the OTHER side's incidence block (``mirror_factor``).
    Delta and zigzag orientations (transpose choices) carry the intrinsic ±30°;
    both are offered and the clock match selects one.
    """
    if side.kind == "delta":
        return [(_DELTA_M, None), (_transpose(_DELTA_M), None)]
    if side.is_zigzag:
        # Normalised limb-difference topology Z = (I − C)/√3: positive-sequence
        # gain 1∠±30°, so the TO-coil leakage referral basis stays the physical
        # line-to-neutral quantity whichever side is zigzag (paired with the
        # u_rated/√3 coil rating in `nominal_turns_ratio`).
        inv_sqrt3 = 1.0 / math.sqrt(3.0)
        z = tuple(
            tuple((i - c) * inv_sqrt3 for i, c in zip(row_i, row_c))
            for row_i, row_c in zip(_identity3(), _CYCLIC_C)
        )
        own = _star_topology(side.kind)
        return [(own, z), (own, _transpose(z))]
    return [(_star_topology(side.kind), None)]


def _incidence_pair(vg: VectorGroup) -> tuple[tuple, tuple]:
    """Select the (N_from, N_to) 3x3 tuple-matrices realising ``vg.clock``.

    Enumerates the constant-topology candidates (delta / zigzag orientation ×
    cyclic permutation ``C^m`` × polarity ``(−1)^s`` on the TO side) and returns
    the first whose positive-sequence rotation equals ``clock·30°``. The realised
    coupling rotation of a candidate is ``arg(conj(λ⁺(N_from))·λ⁺(N_to))``
    because all blocks share the positive-sequence eigenvector. Degenerate
    matches produce the identical nodal block, so first-match is deterministic
    physics, not an arbitrary choice.
    """
    target = cmath.exp(1j * math.radians(vg.clock * 30.0))
    for from_topo, from_mirror in _side_candidates(vg.from_side):
        for to_topo, to_mirror in _side_candidates(vg.to_side):
            n_from_base = (
                from_topo if to_mirror is None else _mat_mul(to_mirror, from_topo)
            )
            n_to_base = (
                to_topo if from_mirror is None else _mat_mul(from_mirror, to_topo)
            )
            lam_from = _pos_seq_eigenvalue(n_from_base)
            for m in range(3):
                for s in (1.0, -1.0):
                    n_to = _mat_scale(_mat_mul(n_to_base, _cyclic_power(m)), s)
                    lam_to = _pos_seq_eigenvalue(n_to)
                    mu = lam_from.conjugate() * lam_to
                    if abs(mu) < 1.0e-9:
                        continue
                    if abs(mu / abs(mu) - target) < 1.0e-9:
                        return n_from_base, n_to
    raise ModelingError(
        f"transformer clock {vg.clock} is not realisable for the "
        f"{vg.from_side.kind} / {vg.to_side.kind} pairing."
    )


def block_incidence(vg: VectorGroup, p: int, rdt, device) -> Tensor:
    """Block-diagonal incidence ``N = blockdiag(N_hv, N_lv)`` ``[2P, 2P]``.

    For ``P == 3`` the per-side blocks realise the full vector group: winding
    topology, delta / zigzag orientation, cyclic phase permutation and polarity,
    selected so the positive-sequence rotation equals ``clock·30°`` (see
    :func:`_incidence_pair`). For other phase counts only plain grounded-wye
    windings with clock 0 (identity) or clock 6 (reversed polarity) are modelled.
    """
    if p == 3:
        n_from, n_to = _incidence_pair(vg)
        n = torch.zeros((2 * p, 2 * p), dtype=rdt, device=device)
        n[:p, :p] = torch.as_tensor(n_from, dtype=rdt, device=device)
        n[p:, p:] = torch.as_tensor(n_to, dtype=rdt, device=device)
        return n
    if vg.from_side.kind != "wye_grounded" or vg.to_side.kind != "wye_grounded":
        raise ModelingError(
            "delta / zigzag / ungrounded-wye transformer windings are only "
            f"modelled for 3 phases (got {p})."
        )
    if vg.clock not in (0, 6):
        raise ModelingError(
            f"transformer clock {vg.clock} needs a 3-phase winding topology "
            f"(got {p} phases; only clock 0 / 6 exist without one)."
        )
    eye = torch.eye(p, dtype=rdt, device=device)
    n = torch.zeros((2 * p, 2 * p), dtype=rdt, device=device)
    n[:p, :p] = eye
    n[p:, p:] = -eye if vg.clock == 6 else eye
    return n


def nominal_turns_ratio(vg: VectorGroup, u_from: Tensor, u_to: Tensor) -> Tensor:
    """Nominal COIL turns ratio ``τ`` from rated LINE voltages + connections.

    A delta coil is rated at the line-to-line voltage; wye and zigzag coils at
    the line-to-neutral voltage ``u_rated/√3`` (the zigzag topology is the
    NORMALISED limb difference ``(I − C)/√3`` with unit positive-sequence gain,
    so its effective coil rating is line-to-neutral like a wye even though the
    physical half-coils are each rated ``u_rated/3``). ``τ = coil_from /
    coil_to``.
    """

    def coil(kind: str, u):
        return u if kind == "delta" else u / math.sqrt(3.0)

    return coil(vg.from_side.kind, u_from) / coil(vg.to_side.kind, u_to)


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
    "block_incidence",
    "nominal_turns_ratio",
    "winding_leakage_block",
]
