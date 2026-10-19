"""Frequency scan of the driving-point and transfer impedance seen from a bus.

:func:`impedance_scan` evaluates ``Z(f) = Y(f)^-1`` of the linear network model on a grid
of frequencies and keeps the rows and columns of the requested buses: per bus the
phase-domain driving-point matrix, its positive- and zero-sequence impedance, and
optionally the full transfer matrix between all requested node-phases. Peaks of the
magnitude are parallel resonances, dips are series resonances
(:meth:`ImpedanceScan.resonances`).

Network model
-------------
By default ``Y(f)`` is the matrix the harmonic power flow solves at its harmonic orders
(:func:`pgml.solver.assemble_harmonic_ybus`), evaluated at every scan frequency: the
passive network with its frequency-dependent branch models, the source as its Norton
(short-circuit) impedance, every explicit Generator / Storage ``harmonic_impedance``, and
each load's harmonic Norton shunt selected by ``load_shunt`` (``config.load_shunt``, else
the modeling default ``appliance.harmonic_shunt.model``, shipped as ``"opendss"``: the
operating-point admittance split into a parallel and a series R-L branch, with each
device's own ``harmonic_model`` overriding it). At an integer order the scan's ``Y`` is
the harmonic flow's ``Y(h)`` for the same options, so the impedance a harmonic current
source sees in :func:`pgml.simulate` is the scanned one (exactly so for constant-power
devices, see below). Generation-sign devices follow
``appliance.harmonic_shunt.generation_model`` (shipped ``"none"``: a pure current source,
open in the scan) unless they carry an explicit ``harmonic_impedance``.

The scan has no fundamental solution, so the shunt of a device is built from its power at
its RATED terminal voltage (``load_shunt_basis="operating_point"`` takes the powers of
``operating_point`` / ``config.operating_point``, ``"nameplate"`` the stored ones). For a
constant-power device that is exactly the harmonic flow's shunt; for a voltage-dependent
(ZIP) or inverter-controlled device the flow uses the power at the solved voltage
instead, and a WARNING names such devices.

``load_shunt="const_z"`` selects the simpler model of :func:`pgml.assembly.assemble_ybus`
instead: every load folded as a constant impedance at its operating point, conductance
flat with frequency and susceptance as the equivalent L or C (the parallel R-L / R-C
branch, the ``series_rl_fraction = 0`` limit of the ``"opendss"`` shunt and the most
heavily damped of the models). ``"none"`` removes every derived device shunt (explicit
harmonic impedances stay).

The source is always a Norton impedance, whatever ``SimulationConfig.slack`` says: an
ideal slack is a short circuit and would make every scan zero at the source bus. This is
the source model the harmonic power flow uses at every order above the fundamental.

Numerics
--------
``Y(f)`` is assembled for a chunk of frequencies at a time, factored once per frequency
with :func:`pgml.solver.lu_factor_system` and solved for the unit-current columns of the
requested rows only (``Y X = E``), never inverted in full. The chunk size follows the
memory budget ``solver.harmonic.system_budget_mb``. Grids with zero-impedance branches are
scanned on the fused system; fused node-phases share one row and therefore one impedance.
A frequency at which ``Y`` is numerically singular is named in a WARNING: a lower bound
of the 1-norm condition number of the equilibrated system, read from the solved columns
at no measurable extra cost, times the unit roundoff exceeds 1, so no digit of the
impedances there is guaranteed, finite as they are. The typical cause is a part of the
network without a path to the source that is held to ground only by a very small shunt.
The bound sees the conditioning through the requested columns only, so a singular part
that no requested bus couples to (and that therefore does not affect the scanned
impedances) is not reported. A regular feeder stays far below the thresholds 4.5e15
(complex128) and 8.4e6 (complex64): the bound on a 200-node three-phase feeder measures
3.5e4 at orders 1 to 50, the full condition estimate
(:func:`pgml.solver.estimate_condition`) 1.6e5.

Differentiability
-----------------
Every impedance tensor of an :class:`ImpedanceScan` (``z_phase_ohm``,
``z_positive_ohm``, ``z_zero_ohm``, ``z_transfer_ohm``) is differentiable with respect to
the tensor-valued grid parameters it was assembled from (line and transformer R/L/C,
source impedance, operating-point power), through the assembly and the linear-solve
adjoint. :meth:`ImpedanceScan.resonances` and :meth:`ImpedanceScan.to_dict` read values
off the tape. A scan whose grid carries tracked parameters keeps every frequency chunk's
``Y`` and factorization alive on the autograd graph, so the memory budget bounds only the
working set of an untracked scan.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from pgml import defaults
from pgml.assembly import assemble_ybus, node_phase_index
from pgml.assembly._load_shunt import resolve_shunt_basis, resolve_shunt_model_name
from pgml.errors import ComputationError, InputError
from pgml.geometry import phase_to_sequence
from pgml.schemas.grid_schema import (
    Grid,
    InjectionAppliance,
    Load,
    Phase,
    Source,
)
from pgml.simulation import SimulationConfig
from pgml.solver import lu_factor_system, solve_factored
from pgml.solver.harmonic_flow import _harmonic_admittance

from ._common import check_study, resolve_complex_dtype

_log = logging.getLogger("pgml")

#: Highest harmonic order of the default frequency grid.
DEFAULT_MAX_ORDER = 50
#: Order step of the default frequency grid (``base_frequency_hz / 10``: 5 Hz at 50 Hz).
DEFAULT_ORDER_STEP = 0.1

#: Default relative prominence a resonance must have (:meth:`ImpedanceScan.resonances`).
DEFAULT_MIN_PROMINENCE = 0.05
#: Load models of the scan: the harmonic flow's device shunts plus the const-Z fold.
_LOAD_SHUNT_MODELS = ("opendss", "motor", "none", "const_z")

_ABC = (Phase.A, Phase.B, Phase.C)
_COMPONENTS = ("positive", "zero", "a", "b", "c", "n")
#: Why a scan fails when ``Y(f)`` is singular (a zero pivot or a non-finite solve).
_SINGULAR_MESSAGE = (
    "the scanned impedance is not finite: Y(f) is singular at some frequency. "
    "Every part of the network needs a path to the source or to a shunt."
)

#: Estimated condition number times unit roundoff above which a frequency is reported as
#: numerically singular (no digit of the solution guaranteed).
_SINGULAR_ERROR_BOUND = 1.0


@dataclass(frozen=True)
class Resonance:
    """One resonance of a driving-point impedance magnitude.

    Attributes
    ----------
    kind:
        ``"parallel"`` for a local maximum of ``|Z|`` (the network presents a high
        impedance; a harmonic current source there produces a large voltage) or
        ``"series"`` for a local minimum (a low-impedance path that attracts harmonic
        current).
    frequency_hz:
        Frequency of the extremum [Hz], refined between the scan points (see
        :meth:`ImpedanceScan.resonances`).
    order:
        ``frequency_hz / base_frequency_hz``.
    z_ohm:
        ``|Z|`` at the refined frequency [Ω] (the interpolating parabola's vertex
        value).
    prominence:
        Relative prominence of the extremum, in ``(0, 1]``: how far it stands out of the
        surrounding curve as a fraction of the local level (see
        :meth:`ImpedanceScan.resonances`). ``1`` for a lossless series resonance
        (``|Z| = 0`` at the dip).
    """

    kind: str
    frequency_hz: float
    order: float
    z_ohm: float
    prominence: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready ``{"kind", "frequency_hz", "order", "z_ohm", "prominence"}``."""
        return {
            "kind": self.kind,
            "frequency_hz": self.frequency_hz,
            "order": self.order,
            "z_ohm": self.z_ohm,
            "prominence": self.prominence,
        }


@dataclass(frozen=True, eq=False)
class ImpedanceScan:
    """Result of :func:`impedance_scan`: impedances seen from each requested bus vs frequency.

    All tensors live on the CPU. Impedances are complex in the dtype the scan ran at and
    in ohms; they stay on the autograd tape when the grid carried tracked parameters.

    Attributes
    ----------
    frequencies_hz:
        Real tensor ``[F]`` of the scan frequencies [Hz], strictly increasing.
    base_frequency_hz:
        The grid's fundamental frequency [Hz]; ``order = frequency / base``.
    node_ids:
        The scanned node ids, in the order of every per-node field.
    phases:
        Per node, the phases of its rows (the grid's ``Node.phases`` order), fixing the
        row and column order of its driving-point matrix.
    z_phase_ohm:
        Per node, the phase-domain driving-point matrix, complex ``[F, P, P]`` with
        ``P = len(phases[k])``: entry ``(i, j)`` is the voltage at phase ``i`` per unit
        current injected into phase ``j`` of the same node, every other bus open.
    z_positive_ohm, z_zero_ohm:
        Complex ``[n, F]`` positive- and zero-sequence driving-point impedance, the
        diagonal of ``A^-1 Z_abc A`` (:func:`pgml.geometry.phase_to_sequence`) over the
        node's a, b, c block. ``NaN`` for a node that does not carry all three of a, b, c.
    transfer_rows:
        ``((node_id, phase), ...)`` labelling the rows and columns of
        ``z_transfer_ohm``: the requested nodes' rows, node by node, or ``None``.
    z_transfer_ohm:
        Complex ``[F, R, R]`` transfer matrix between all requested node-phases (the
        driving-point matrices are its diagonal blocks), or ``None`` unless the scan was
        run with ``transfer=True``.
    """

    frequencies_hz: Tensor
    base_frequency_hz: float
    node_ids: tuple[int, ...]
    phases: tuple[tuple[Phase, ...], ...]
    z_phase_ohm: tuple[Tensor, ...]
    z_positive_ohm: Tensor
    z_zero_ohm: Tensor
    transfer_rows: Optional[tuple[tuple[int, Phase], ...]] = None
    z_transfer_ohm: Optional[Tensor] = None

    # -- lookups ---------------------------------------------------------------
    def _position(self, node_id: int) -> int:
        try:
            return self.node_ids.index(int(node_id))
        except ValueError:
            raise InputError(
                f"node {node_id} is not part of this impedance scan "
                f"(scanned nodes: {list(self.node_ids)})."
            ) from None

    @property
    def orders(self) -> Tensor:
        """Harmonic order of every scan point, real ``[F]``."""
        return self.frequencies_hz / self.base_frequency_hz

    def driving_point(self, node_id: int) -> Tensor:
        """Phase-domain driving-point matrix of ``node_id``, complex ``[F, P, P]`` [Ω]."""
        return self.z_phase_ohm[self._position(node_id)]

    def impedance(self, node_id: int, component: Optional[str] = None) -> Tensor:
        """One driving-point impedance curve of ``node_id``, complex ``[F]`` [Ω].

        Parameters
        ----------
        node_id:
            A scanned node.
        component:
            ``"positive"`` / ``"zero"`` for the sequence impedance (nodes with a, b, c),
            or a phase ``"a"``, ``"b"``, ``"c"``, ``"n"`` for that phase's self
            impedance (the diagonal entry of the phase matrix). ``None`` selects
            ``"positive"`` on a node with a, b, c and the self impedance of the node's
            first phase otherwise.
        """
        k = self._position(node_id)
        phases = self.phases[k]
        if component is None:
            component = "positive" if set(_ABC) <= set(phases) else phases[0].value
        component = str(component).lower()
        if component not in _COMPONENTS:
            raise InputError(
                f"component must be one of {list(_COMPONENTS)}, got {component!r}."
            )
        if component in ("positive", "zero"):
            if not set(_ABC) <= set(phases):
                raise InputError(
                    f"node {node_id} has phases {[p.value for p in phases]}; sequence "
                    "impedances need all of a, b, c."
                )
            return (
                self.z_positive_ohm if component == "positive" else self.z_zero_ohm
            )[k]
        phase = Phase(component)
        if phase not in phases:
            raise InputError(
                f"node {node_id} has no phase {component!r} "
                f"(phases {[p.value for p in phases]})."
            )
        p = phases.index(phase)
        return self.z_phase_ohm[k][:, p, p]

    def resonances(
        self,
        node_id: int,
        component: Optional[str] = None,
        *,
        min_prominence: float = DEFAULT_MIN_PROMINENCE,
    ) -> list[Resonance]:
        """Parallel (peaks of ``|Z|``) and series (dips) resonances of one curve.

        The interior local extrema of ``|Z(f)|`` of :meth:`impedance`
        ``(node_id, component)`` whose relative prominence reaches ``min_prominence``,
        sorted by frequency. The scan's end points are never extrema. A run of equal
        samples (a plateau) is one extremum, placed at the middle sample of the run.

        Prominence is measured against the surrounding curve, not against the next
        sample, so noise on the flank of a resonance neither creates a resonance nor
        hides one. For a peak of height ``|Z|_p``: on each side, the lowest ``|Z|``
        between the peak and the nearest higher sample (or the scan's end); the higher
        of the two is the peak's base, and the relative prominence is
        ``(|Z|_p - base) / |Z|_p``. For a dip of depth ``|Z|_d`` the roles swap: on each
        side the highest ``|Z|`` up to the nearest lower sample, the lower of the two is
        the base, and the relative prominence is ``(base - |Z|_d) / base``. Both lie in
        ``[0, 1]``; a lossless series resonance (``|Z|_d = 0``) has prominence ``1``.
        The default ``0.05`` keeps resonances that rise or fall at least 5 % out of their
        surroundings and rejects the rounding noise of a flat curve (about ``1e-15``
        relative in complex128, ``1e-7`` in complex64) by many orders of magnitude.

        The frequency of an extremum is refined between the scan points by the vertex of
        the parabola through the extremum and its two neighbours (clamped to that
        interval), fitted to ``1/|Z|^2`` at a peak and to ``|Z|^2`` at a dip: near a
        resonance of a lightly damped circuit those quantities are quadratic in frequency
        (the Lorentzian shape), so the estimate is not limited to the scan step. A
        resonance narrower than the scan step can be missed or misplaced: scan finer
        around it.

        Parameters
        ----------
        node_id, component:
            As :meth:`impedance`.
        min_prominence:
            Smallest relative prominence reported, in ``[0, 1]``. ``0`` reports every
            strict extremum, including those of rounding noise.

        Returns
        -------
        list of Resonance
            Empty when the curve has no extremum of the required prominence.
        """
        if not (0.0 <= float(min_prominence) <= 1.0):
            raise InputError(
                f"min_prominence must lie in [0, 1], got {min_prominence!r}."
            )
        z = self.impedance(node_id, component)
        with torch.no_grad():
            mag = z.abs().to(torch.float64).numpy()
            f = self.frequencies_hz.to(torch.float64).numpy()
        apex, is_peak = _extrema(mag)
        if apex.size == 0:
            return []
        prominence = _relative_prominence(mag, apex, is_peak)
        keep = prominence >= float(min_prominence)
        apex, is_peak, prominence = apex[keep], is_peak[keep], prominence[keep]
        if apex.size == 0:
            return []
        f_ref, z_ref = _refine(f, mag, apex, is_peak)
        base = float(self.base_frequency_hz)
        return [
            Resonance(
                kind="parallel" if pk else "series",
                frequency_hz=float(fr),
                order=float(fr) / base,
                z_ohm=float(zr),
                prominence=float(pr),
            )
            for pk, fr, zr, pr in zip(is_peak.tolist(), f_ref, z_ref, prominence)
        ]

    # -- serialization -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict of the scan (plain lists; complex values split into re/im).

        Shape::

            {
              "study": "impedance_scan",
              "base_frequency_hz": float,
              "frequencies_hz": [F],
              "nodes": [
                {"node_id": int, "phases": ["a", ...],
                 "z_phase_re_ohm": [F][P][P], "z_phase_im_ohm": [F][P][P],
                 "z_positive_re_ohm": [F] | None, "z_positive_im_ohm": [F] | None,
                 "z_zero_re_ohm": [F] | None, "z_zero_im_ohm": [F] | None},
                ...
              ],
              "transfer": None | {
                "rows": [{"node_id": int, "phase": "a"}, ...],
                "z_re_ohm": [F][R][R], "z_im_ohm": [F][R][R]
              }
            }

        Sequence impedances are ``None`` for a node without a, b, c. Values are read off
        the autograd tape.
        """
        with torch.no_grad():
            nodes = []
            for k, nid in enumerate(self.node_ids):
                zp = self.z_phase_ohm[k]
                entry: dict[str, Any] = {
                    "node_id": int(nid),
                    "phases": [p.value for p in self.phases[k]],
                    "z_phase_re_ohm": zp.real.tolist(),
                    "z_phase_im_ohm": zp.imag.tolist(),
                }
                has_seq = set(_ABC) <= set(self.phases[k])
                for name, zs in (
                    ("positive", self.z_positive_ohm[k]),
                    ("zero", self.z_zero_ohm[k]),
                ):
                    entry[f"z_{name}_re_ohm"] = zs.real.tolist() if has_seq else None
                    entry[f"z_{name}_im_ohm"] = zs.imag.tolist() if has_seq else None
                nodes.append(entry)
            transfer = None
            if self.z_transfer_ohm is not None and self.transfer_rows is not None:
                transfer = {
                    "rows": [
                        {"node_id": int(n), "phase": p.value}
                        for n, p in self.transfer_rows
                    ],
                    "z_re_ohm": self.z_transfer_ohm.real.tolist(),
                    "z_im_ohm": self.z_transfer_ohm.imag.tolist(),
                }
            return {
                "study": "impedance_scan",
                "base_frequency_hz": float(self.base_frequency_hz),
                "frequencies_hz": self.frequencies_hz.tolist(),
                "nodes": nodes,
                "transfer": transfer,
            }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImpedanceScan":
        """Rebuild a scan from :meth:`to_dict` output (float64 / complex128, CPU).

        Raises :class:`~pgml.errors.InputError` unless ``data["study"]`` is
        ``"impedance_scan"``.
        """
        check_study(data, "impedance_scan")
        f = torch.as_tensor(data["frequencies_hz"], dtype=torch.float64)
        n_f = f.shape[0]
        node_ids, phases, z_phase, z_pos, z_zero = [], [], [], [], []
        nan = torch.full((n_f,), complex(math.nan, math.nan), dtype=torch.complex128)
        for entry in data["nodes"]:
            node_ids.append(int(entry["node_id"]))
            phases.append(tuple(Phase(p) for p in entry["phases"]))
            z_phase.append(
                _complex(entry["z_phase_re_ohm"], entry["z_phase_im_ohm"]).reshape(
                    n_f, len(entry["phases"]), len(entry["phases"])
                )
            )
            for name, out in (("positive", z_pos), ("zero", z_zero)):
                re = entry.get(f"z_{name}_re_ohm")
                out.append(
                    nan if re is None else _complex(re, entry[f"z_{name}_im_ohm"])
                )
        transfer = data.get("transfer")
        rows = z_t = None
        if transfer is not None:
            rows = tuple(
                (int(r["node_id"]), Phase(r["phase"])) for r in transfer["rows"]
            )
            z_t = _complex(transfer["z_re_ohm"], transfer["z_im_ohm"]).reshape(
                n_f, len(rows), len(rows)
            )
        empty = torch.zeros((0, n_f), dtype=torch.complex128)
        return cls(
            frequencies_hz=f,
            base_frequency_hz=float(data["base_frequency_hz"]),
            node_ids=tuple(node_ids),
            phases=tuple(phases),
            z_phase_ohm=tuple(z_phase),
            z_positive_ohm=torch.stack(z_pos) if z_pos else empty,
            z_zero_ohm=torch.stack(z_zero) if z_zero else empty,
            transfer_rows=rows,
            z_transfer_ohm=z_t,
        )


def impedance_scan(
    grid: Grid,
    nodes: Optional[Sequence[int]] = None,
    *,
    frequencies_hz=None,
    config: Optional[SimulationConfig] = None,
    operating_point: Optional[dict] = None,
    load_shunt: Optional[Literal["opendss", "motor", "none", "const_z"]] = None,
    load_shunt_basis: Optional[Literal["operating_point", "nameplate"]] = None,
    transfer: bool = False,
    fold_generation: bool = False,
    dtype: Union[str, torch.dtype] = torch.complex128,
    device: Optional[Union[str, torch.device]] = None,
) -> ImpedanceScan:
    """Driving-point (and optionally transfer) impedance seen from buses vs frequency.

    Computes ``Z(f) = Y(f)^-1`` restricted to the requested buses' node-phase rows. By
    default ``Y(f)`` is the harmonic power flow's system matrix with the same device
    shunt (``load_shunt``) the flow uses, so a scan agrees with :func:`pgml.simulate`
    for the same :class:`~pgml.SimulationConfig` out of the box (module docstring).

    Parameters
    ----------
    grid:
        A materialised :class:`~pgml.Grid` with at least one in-service
        :class:`~pgml.schemas.grid_schema.Source` of nonzero impedance. Not modified.
    nodes:
        Node ids to scan, in the order of the result. ``None`` scans every node in
        ``grid.nodes`` order (the cost grows with the number of scanned rows: one
        right-hand side per row).
    frequencies_hz:
        1-D sequence or real tensor of strictly increasing positive frequencies [Hz].
        ``None`` uses orders 1 to :data:`~pgml.studies.impedance.DEFAULT_MAX_ORDER` in
        steps of :data:`~pgml.studies.impedance.DEFAULT_ORDER_STEP` (491 points: 50 to
        2500 Hz in 5 Hz steps on a 50 Hz grid), which resolves the resonances of
        distribution feeders and capacitor banks; :meth:`ImpedanceScan.resonances`
        refines extrema between the points.
    config:
        Optional :class:`~pgml.SimulationConfig`. ``symmetry``, ``operating_point``,
        ``load_shunt`` and ``load_shunt_basis`` are read; ``slack`` is ignored (the
        source is always its Norton impedance) and so are the harmonic orders and the
        convergence settings.
    operating_point:
        ``{appliance_id: {"p_w", "q_var"} | {"p_per_phase_w", "q_per_phase_var"}}``
        powers the device shunts are built from. Replaces ``config.operating_point``
        when given; ``None`` falls back to it, then to the nameplate. Not modified. A
        batched (scenario) operating point is not supported.
    load_shunt:
        Load model in ``Y(f)``: ``"opendss"`` (the operating-point admittance split
        into a parallel and a series R-L branch), ``"motor"`` (fixed blocked-rotor
        series reactance) or ``"none"`` (no derived device shunt), exactly as the
        harmonic power flow's option of the same name, with each device's own
        ``harmonic_model`` overriding it; or ``"const_z"``, the constant-impedance fold
        of :func:`pgml.assembly.assemble_ybus` (parallel R-L / R-C branch). ``None``
        takes ``config.load_shunt``, else the modeling default
        ``appliance.harmonic_shunt.model`` (``"opendss"``), the flow's own default.
    load_shunt_basis:
        Which powers the shunts are built from: ``"operating_point"`` (the powers of
        ``operating_point``, else the nameplate) or ``"nameplate"`` (the stored powers,
        ``operating_point`` ignored); both at the device's rated terminal voltage.
        ``None`` takes ``config.load_shunt_basis``, else the modeling default
        ``appliance.harmonic_shunt.basis``.
    transfer:
        Also return the transfer matrix between every pair of requested node-phases
        (``z_transfer_ohm``, ``[F, R, R]``). The solve is the same; only the stored
        result grows.
    fold_generation:
        ``load_shunt="const_z"`` only: fold generators and storage as constant
        impedances like loads. ``False`` (default) leaves them out of the scan (open
        circuit), because their folded image has a negative conductance. The harmonic
        shunt models apply the policy ``appliance.harmonic_shunt.generation_model``
        instead, so ``True`` with another model raises.
    dtype:
        Complex working dtype, ``torch.complex128`` (default) or ``torch.complex64``,
        or their names (``"complex128"``, ``"complex64"``).
    device:
        Device to assemble and solve on; ``None`` uses the device of a tensor
        ``frequencies_hz``, else the CPU. The result is moved to the CPU.

    Returns
    -------
    ImpedanceScan
        Frequencies, node ids and phases, per-node phase matrices ``[F, P, P]``,
        sequence impedances ``[n, F]`` and the optional transfer matrix.

    Raises
    ------
    InputError
        Unknown or repeated node ids, an unknown ``dtype``, ``load_shunt`` or
        ``load_shunt_basis``, ``fold_generation`` with a harmonic shunt model, invalid
        frequencies, no in-service source, a source with zero impedance, or a batched
        operating point.
    ComputationError
        ``Y(f)`` is singular at some frequency: a zero pivot in the factorization or a
        non-finite impedance (a part of the network without a path to the source or to
        a shunt). Which of the two a matrix singular only to working precision produces
        depends on the factorization backend; a nearly singular but factorizable ``Y``
        is reported by a WARNING instead.

    Notes
    -----
    Device and dtype: assembly, factorization and solve run on ``device`` in ``dtype``
    (CPU and CUDA); frequencies are chunked against ``solver.harmonic.system_budget_mb``
    and each chunk is one batched factorization with ``R`` right-hand sides per
    frequency, ``R`` the number of requested rows. The returned impedances are
    differentiable with respect to the grid's tensor-valued parameters.

    Examples
    --------
    >>> from pgml.studies import impedance_scan
    >>> scan = impedance_scan(grid, nodes=[18])                  # doctest: +SKIP
    >>> scan.impedance(18).abs()       # |Z1(f)| [ohm], shape [F]  # doctest: +SKIP
    >>> scan.resonances(18)                                      # doctest: +SKIP
    """
    cfg = config if config is not None else SimulationConfig()
    f0 = float(grid.base_frequency_hz)
    if device is None and isinstance(frequencies_hz, Tensor):
        device = frequencies_hz.device
    device = torch.device(device) if device is not None else torch.device("cpu")
    dtype = resolve_complex_dtype(dtype)
    rdt = torch.float64 if dtype == torch.complex128 else torch.float32
    model = _resolve_load_model(
        load_shunt if load_shunt is not None else cfg.load_shunt
    )
    basis = resolve_shunt_basis(
        load_shunt_basis if load_shunt_basis is not None else cfg.load_shunt_basis
    )
    if fold_generation and model != "const_z":
        raise InputError(
            "fold_generation applies to load_shunt='const_z' only; the harmonic shunt "
            f"model {model!r} follows appliance.harmonic_shunt.generation_model."
        )

    freqs = _frequency_grid(frequencies_hz, f0, rdt, device)
    node_ids = _resolve_nodes(grid, nodes)
    _check_sources(grid)
    op = operating_point if operating_point is not None else cfg.operating_point
    if basis == "nameplate":
        op = None
    if model == "const_z":
        op = _fold_operating_point(grid, op, fold_generation)

    full = node_phase_index(grid)
    node_by_id = {int(n.id): n for n in grid.nodes}
    phases = tuple(tuple(node_by_id[nid].phases) for nid in node_ids)
    block = _block_index(phases, device)  # [n, P_max]
    rows_full = [r for nid in node_ids for r in full.rows(nid)]
    rows_full_t = torch.as_tensor(rows_full, dtype=torch.int64, device=device)
    n_req = len(rows_full)

    itemsize = torch.empty((), dtype=dtype).element_size()
    budget = float(defaults.get("solver.harmonic.system_budget_mb")) * 1024.0**2
    # Charged per frequency: the dense Y and its factorization, the R solution columns
    # and their unit right-hand sides, and the kept [R, R] block.
    per_freq = itemsize * (2 * full.size**2 + 2 * full.size * n_req + n_req**2)
    chunk = max(1, int(budget // per_freq))

    fusion = None
    block_parts, transfer_parts, cond_parts = [], [], []
    for start in range(0, freqs.shape[0], chunk):
        f_chunk = freqs[start : start + chunk]
        if model == "const_z":
            ybus = assemble_ybus(
                grid,
                f_chunk,
                dtype=dtype,
                device=device,
                operating_point=op,
                symmetry=cfg.symmetry,
                fusion=fusion,
            )
            y, fusion = ybus.Y, ybus.fusion
        else:
            orders = [float(x) / f0 for x in f_chunk.tolist()]
            y, _, fusion = _harmonic_admittance(
                grid,
                f_chunk,
                orders,
                v1=None,
                operating_point=op,
                load_shunt=model,
                load_shunt_basis=basis,
                symmetry=cfg.symmetry,
                dtype=dtype,
                device=device,
                fusion=fusion,
                warn=start == 0,
            )
        if y.ndim != 3:
            raise InputError(
                "impedance_scan takes one operating point; a batched operating point "
                f"promotes Y to {tuple(y.shape)}. Scan each scenario separately."
            )
        rows = (
            fusion.row_to_reduced.to(device)[rows_full_t]
            if fusion is not None
            else rows_full_t
        )
        try:
            z_rows, cond = _solve_columns(y, rows)  # [F_chunk, R, R], [F_chunk]
        except (ComputationError, RuntimeError) as exc:
            # An exactly zero pivot: whether rounding leaves a tiny pivot (a finite,
            # ill-conditioned solve reported below) or none at all depends on the
            # factorization backend, so both end in the same diagnosis. A single
            # matrix's zero pivot arrives as torch's RuntimeError, a stack's as a
            # ComputationError.
            raise ComputationError(_SINGULAR_MESSAGE) from exc
        cond_parts.append(cond)
        # Driving-point blocks [F_chunk, n, P_max, P_max]; padding is dropped later.
        block_parts.append(z_rows[:, block[:, :, None], block[:, None, :]])
        if transfer:
            transfer_parts.append(z_rows)
    blocks = torch.cat(block_parts, dim=0)
    z_transfer = torch.cat(transfer_parts, dim=0) if transfer else None

    with torch.no_grad():
        finite = bool(torch.isfinite(torch.view_as_real(blocks)).all())
    if not finite:
        raise ComputationError(_SINGULAR_MESSAGE)
    _warn_ill_conditioned(torch.cat(cond_parts), freqs, dtype)

    return _build_result(node_ids, phases, freqs, blocks, z_transfer, f0)


# ---- private helpers -----------------------------------------------------------
def _frequency_grid(frequencies_hz, f0: float, rdt, device) -> Tensor:
    """Validated 1-D frequency tensor ``[F]`` on ``device`` (the default grid if None)."""
    if frequencies_hz is None:
        n_steps = int(round((DEFAULT_MAX_ORDER - 1) / DEFAULT_ORDER_STEP))
        orders = 1.0 + DEFAULT_ORDER_STEP * torch.arange(
            n_steps + 1, dtype=torch.float64
        )
        return (orders * f0).to(dtype=rdt, device=device)
    if isinstance(frequencies_hz, Tensor):
        with torch.no_grad():
            f = frequencies_hz.to(dtype=rdt, device=device)
    else:
        f = torch.as_tensor(frequencies_hz, dtype=rdt, device=device)
    f = f.reshape(-1) if f.ndim == 0 else f
    if f.ndim != 1 or f.shape[0] == 0:
        raise InputError("frequencies_hz must be a non-empty 1-D sequence.")
    with torch.no_grad():
        ok = bool(torch.isfinite(f).all()) and bool((f > 0).all())
        increasing = f.shape[0] == 1 or bool((f[1:] > f[:-1]).all())
    if not ok:
        raise InputError("frequencies_hz must be finite and positive.")
    if not increasing:
        raise InputError("frequencies_hz must be strictly increasing.")
    return f


def _resolve_nodes(grid: Grid, nodes: Optional[Sequence[int]]) -> tuple[int, ...]:
    """Requested node ids as a tuple; every node when ``nodes`` is None."""
    known = [int(n.id) for n in grid.nodes]
    if nodes is None:
        return tuple(known)
    ids = tuple(int(n) for n in nodes)
    if not ids:
        raise InputError("nodes must name at least one node (or be None for all).")
    missing = sorted(set(ids) - set(known))
    if missing:
        raise InputError(f"unknown node ids {missing}.")
    if len(set(ids)) != len(ids):
        raise InputError(f"node ids must not repeat, got {list(ids)}.")
    return ids


def _check_sources(grid: Grid) -> None:
    """Require an in-service source whose impedance is not zero (it would short Y)."""
    sources = [a for a in grid.appliances if isinstance(a, Source) and a.in_service]
    if not sources:
        raise InputError(
            "impedance_scan needs an in-service Source: the scan represents the "
            "upstream network by the source's Norton impedance."
        )
    with torch.no_grad():
        for s in sources:
            r = torch.as_tensor(s.resistance_ohm, dtype=torch.float64)
            ind = torch.as_tensor(s.inductance_h, dtype=torch.float64)
            if bool((r == 0).all()) and bool((ind == 0).all()):
                raise InputError(
                    f"Source {s.id} has zero impedance: an ideal source short-circuits "
                    "the scan. Give it its short-circuit impedance (R and L)."
                )


def _fold_operating_point(
    grid: Grid, operating_point: Optional[dict], fold_generation: bool
) -> Optional[dict]:
    """Operating point of the load fold; generation zeroed unless it is folded.

    Returns a new dict (the caller's is never modified). A device with zero power folds
    to a zero admittance, i.e. it is open in the scan.
    """
    if fold_generation:
        return operating_point
    generation = [
        a.id
        for a in grid.appliances
        if isinstance(a, InjectionAppliance)
        and not isinstance(a, Load)
        and a.in_service
    ]
    if not generation:
        return operating_point
    op = dict(operating_point) if operating_point is not None else {}
    for aid in generation:
        op[aid] = {"p_w": 0.0, "q_var": 0.0}
    return op


def _resolve_load_model(name: Optional[str]) -> str:
    """Validated scan load model (``None`` -> ``appliance.harmonic_shunt.model``)."""
    if name == "const_z":
        return "const_z"
    if name is not None and name not in _LOAD_SHUNT_MODELS:
        raise InputError(
            f"load_shunt must be one of {list(_LOAD_SHUNT_MODELS)}, got {name!r}."
        )
    return resolve_shunt_model_name(name)


def _solve_columns(y: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
    """``Z[f, i, j] = (Y(f)^-1)[rows[i], rows[j]]`` for ``Y`` ``[F, M, M]`` -> ``[F, R, R]``.

    One factorization per frequency, ``R`` unit-current right-hand sides solved as
    multiple columns of that factorization; the inverse is never formed.

    Also returns, real ``[F]`` and off the tape, a lower bound of the 1-norm condition
    number of the equilibrated matrix ``A = D Y D``, ``D = diag(|Y_ii|^-1/2)`` (the van
    der Sluis scaling the factorization applies), read from the solved columns
    themselves: ``max_j ||A^-1 e_j||_1`` over the requested columns ``j``, with
    ``A^-1 e_j = D^-1 (Y^-1 e_j) / d_j``, times ``||A||_1 >= max |A_jj| = 1``. It costs
    no extra solve and no pass over ``Y``; dropping ``||A||_1`` (about 2 for a network
    admittance, whose off-diagonal magnitudes sum to about its diagonal) only weakens the
    bound by that factor.
    """
    n_f, m = y.shape[0], y.shape[-1]
    n_r = rows.shape[0]
    fac = lu_factor_system(y)
    e = torch.zeros((n_r, 1, m), dtype=y.dtype, device=y.device)
    e[torch.arange(n_r, device=y.device), 0, rows] = 1.0
    x = solve_factored(fac, e.expand(n_r, n_f, m))  # [R (column j), F, M]
    with torch.no_grad():
        diag = y.diagonal(dim1=-2, dim2=-1).abs()  # [F, M]
        d = diag.clamp_min(torch.finfo(diag.dtype).tiny).rsqrt()
        col = (x.abs() / d).sum(dim=-1) / d[:, rows].transpose(0, 1)  # [R, F]
        cond = col.amax(dim=0)
        cond = torch.where(torch.isfinite(cond), cond, torch.full_like(cond, math.inf))
    return x[:, :, rows].permute(1, 2, 0), cond  # [F, R (row i), R (column j)]


def _warn_ill_conditioned(cond: Tensor, freqs: Tensor, dtype: torch.dtype) -> None:
    """WARNING naming the frequencies at which ``Y`` is numerically singular."""
    eps = torch.finfo(torch.float64 if dtype == torch.complex128 else torch.float32).eps
    with torch.no_grad():
        bad = (cond.to(torch.float64) * eps > _SINGULAR_ERROR_BOUND).cpu()
        if not bool(bad.any()):
            return
        f_bad = freqs.cpu()[bad].tolist()
        worst = float(cond.to(torch.float64).max().cpu())
    _log.warning(
        "impedance_scan: Y(f) is numerically singular at %d frequency(ies) %s Hz "
        "(estimated condition number up to %.3g at %s): no digit of the impedances "
        "there is guaranteed. A part of the network without a path to the source is "
        "held only by a very small shunt.",
        len(f_bad),
        [round(x, 6) for x in f_bad[:10]],
        worst,
        dtype,
    )


def _block_index(phases: tuple[tuple[Phase, ...], ...], device) -> Tensor:
    """Positions of each node's rows within the requested rows, int64 ``[n, P_max]``.

    Nodes with fewer than ``P_max`` phases are padded with their first position; the
    padded entries are sliced away when the per-node matrices are built.
    """
    offsets = np.cumsum([0] + [len(p) for p in phases])
    p_max = max(len(p) for p in phases)
    idx = [
        [int(offsets[k]) + min(j, len(p) - 1) for j in range(p_max)]
        for k, p in enumerate(phases)
    ]
    return torch.as_tensor(idx, dtype=torch.int64, device=device)


def _build_result(node_ids, phases, freqs, blocks, z_transfer, f0) -> ImpedanceScan:
    """Per-node result fields from the padded driving-point blocks ``[F, n, P_max, P_max]``."""
    blocks = blocks.cpu()
    z_phase = tuple(blocks[:, k, : len(p), : len(p)] for k, p in enumerate(phases))

    # Sequence impedances of every node with a, b, c: one batched transform.
    seq_nodes = [k for k, p in enumerate(phases) if set(_ABC) <= set(p)]
    n_f = blocks.shape[0]
    nan = complex(math.nan, math.nan)
    z_pos = torch.full((len(node_ids), n_f), nan, dtype=blocks.dtype)
    z_zero = torch.full((len(node_ids), n_f), nan, dtype=blocks.dtype)
    if seq_nodes:
        k_idx = torch.as_tensor(seq_nodes, dtype=torch.int64)
        abc = torch.as_tensor(
            [[phases[k].index(ph) for ph in _ABC] for k in seq_nodes], dtype=torch.int64
        )  # [K, 3] positions within each node's block
        z_abc = blocks[
            :, k_idx[:, None, None], abc[:, :, None], abc[:, None, :]
        ]  # [F, K, 3, 3]
        z_seq = phase_to_sequence(z_abc)
        z_zero = z_zero.index_put((k_idx,), z_seq[..., 0, 0].transpose(0, 1))
        z_pos = z_pos.index_put((k_idx,), z_seq[..., 1, 1].transpose(0, 1))

    rows = None
    if z_transfer is not None:
        rows = tuple((nid, ph) for nid, p in zip(node_ids, phases) for ph in p)
        z_transfer = z_transfer.cpu()
    return ImpedanceScan(
        frequencies_hz=freqs.cpu(),
        base_frequency_hz=f0,
        node_ids=tuple(node_ids),
        phases=phases,
        z_phase_ohm=z_phase,
        z_positive_ohm=z_pos,
        z_zero_ohm=z_zero,
        transfer_rows=rows,
        z_transfer_ohm=z_transfer,
    )


def _extrema(mag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interior extrema of a sampled curve ``[F]``: apex indices and a peak mask.

    Runs of equal samples are collapsed first, so a plateau whose neighbours both lie
    below (above) it is one peak (dip), placed at the middle sample of the run. The
    scan's end points are never extrema.
    """
    if mag.size < 3:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    # Start index of every run of equal values, and the run lengths.
    starts = np.flatnonzero(np.r_[True, mag[1:] != mag[:-1]])
    lengths = np.diff(np.r_[starts, mag.size])
    level = mag[starts]
    if level.size < 3:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    d = np.diff(level)
    peak = (d[:-1] > 0) & (d[1:] < 0)
    dip = (d[:-1] < 0) & (d[1:] > 0)
    run = np.flatnonzero(peak | dip) + 1
    apex = starts[run] + (lengths[run] - 1) // 2
    return apex.astype(np.int64), peak[run - 1]


def _relative_prominence(
    mag: np.ndarray, apex: np.ndarray, is_peak: np.ndarray
) -> np.ndarray:
    """Relative prominence ``[K]`` of each extremum (method of ``resonances``).

    A dip is a peak of ``-|Z|``: its base is the lower of the two side maxima up to the
    nearest lower sample, and the prominence is normalised by that base instead of by
    the apex.
    """
    out = np.zeros(apex.size)
    n = mag.size
    for k, (i, pk) in enumerate(zip(apex.tolist(), is_peak.tolist())):
        y = mag if pk else -mag
        top = y[i]
        sides = []
        for step in (-1, 1):
            j, lowest = i + step, top
            while 0 <= j < n and y[j] <= top:
                lowest = min(lowest, y[j])
                j += step
            sides.append(lowest)
        base = max(sides)  # the higher base of y (for a dip: the lower side maximum)
        if pk:
            out[k] = (top - base) / top if top > 0.0 else 0.0
        else:
            level = -base  # the lower of the two shoulders of |Z|
            out[k] = (level - mag[i]) / level if level > 0.0 else 0.0
    return out


def _refine(
    f: np.ndarray, mag: np.ndarray, apex: np.ndarray, is_peak: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Refined frequency and ``|Z|`` of each extremum (parabolic vertex, see above).

    ``1/|Z|^2`` is formed only around peaks, whose samples are positive; a degenerate
    fit (a non-finite vertex) falls back to the apex sample.
    """
    f_ref = f[apex].copy()
    z_ref = mag[apex].copy()
    pk = np.flatnonzero(is_peak)
    if pk.size:
        idx = apex[pk]
        window = np.stack([mag[idx - 1], mag[idx], mag[idx + 1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.zeros_like(mag)
            sel = np.r_[idx - 1, idx, idx + 1]
            inv[sel] = 1.0 / np.square(mag[sel])
            xv, yv = _parabolic_vertex(f, inv, idx)
            zv = 1.0 / np.sqrt(yv)
        good = np.isfinite(xv) & np.isfinite(zv) & (window > 0.0).all(axis=0)
        f_ref[pk] = np.where(good, xv, f[idx])
        z_ref[pk] = np.where(good, zv, mag[idx])
    dp = np.flatnonzero(~is_peak)
    if dp.size:
        idx = apex[dp]
        xv, yv = _parabolic_vertex(f, np.square(mag), idx)
        f_ref[dp] = xv
        z_ref[dp] = np.sqrt(np.maximum(yv, 0.0))
    return f_ref, z_ref


def _parabolic_vertex(
    x: np.ndarray, y: np.ndarray, idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vertex ``(x*, y*)`` of the parabola through points ``idx - 1, idx, idx + 1``.

    Works on a non-uniform ``x``; ``x*`` is clamped to ``[x[idx-1], x[idx+1]]`` and a
    degenerate (collinear) triple returns the middle point.
    """
    x0, x1, x2 = x[idx - 1], x[idx], x[idx + 1]
    y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]
    d01 = (y1 - y0) / (x1 - x0)
    d12 = (y2 - y1) / (x2 - x1)
    a = (d12 - d01) / (x2 - x0)
    b = d01 - a * (x0 + x1)
    safe = np.abs(a) > 0
    a_s = np.where(safe, a, 1.0)
    xv = np.where(safe, -b / (2.0 * a_s), x1)
    xv = np.clip(xv, x0, x2)
    # Newton form of the interpolating parabola.
    yv = np.where(safe, y0 + d01 * (xv - x0) + a * (xv - x0) * (xv - x1), y1)
    return xv, yv


def _complex(re, im) -> Tensor:
    """complex128 tensor from nested real / imaginary lists."""
    return torch.complex(
        torch.as_tensor(re, dtype=torch.float64),
        torch.as_tensor(im, dtype=torch.float64),
    )


__all__ = [
    "DEFAULT_MAX_ORDER",
    "DEFAULT_MIN_PROMINENCE",
    "DEFAULT_ORDER_STEP",
    "ImpedanceScan",
    "Resonance",
    "impedance_scan",
]
