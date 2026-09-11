"""High-level simulation entry point — the blessed external API.

``simulate(grid, config)`` is the front door for external consumers (dashboards, a
REST API, analysis scripts). It dispatches to the differentiable solvers and
returns a :class:`SolvedState` — a complete, lazily-derived snapshot of the solved grid
from which any quantity can be extracted:

- node voltage phasors (eager — the direct solve output),
- branch terminal currents and power flows (computed on access as tracked tensor ops),
- per-node harmonic spectra and THD,

all differentiable (gradients flow ``grid params -> SolvedState accessors``) so the ML
layer can use it directly, plus :meth:`SolvedState.to_result_set` which materialises the
serializable :mod:`pgml.schemas.result_schema` records (JSON for REST / dashboard, or to
persist for training).

Design split (see also the README):

- :class:`SimulationConfig` is the serializable *definition* of WHAT to simulate
  (calculation type, harmonic orders, slack, symmetry, operating point, tolerances). It
  is a pydantic model — a clean JSON body for REST.
- Device and dtype are *execution* concerns (WHERE / precision), passed as keyword
  arguments to :func:`simulate`, NOT part of the serializable config.

For batched training-data generation use :func:`pgml.scenarios.run_scenarios` instead;
for raw differentiable tensors at minimal overhead use :mod:`pgml.solver` directly.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from .errors import ConvergenceError, InputError
from .schemas.grid_schema import Grid, Phase
from .schemas.result_schema import (
    BranchResult,
    NodeResult,
    ResultSet,
    SolverDiagnostics,
)
from .solver import solve_harmonic_flow, solve_power_flow

Calculation = Literal["power_flow", "harmonic"]
Slack = Literal["ideal", "norton"]
Symmetry = Literal["auto", "symmetric", "asymmetric"]
_DTYPES = {"complex128": torch.complex128, "complex64": torch.complex64}


class SimulationConfig(BaseModel):
    """Serializable definition of WHAT to simulate (the REST-friendly run spec).

    Device / dtype / working precision are execution concerns and live on the
    :func:`simulate` call, not here. ``harmonic_orders`` is used only when
    ``calculation == "harmonic"``. The convergence tolerances are PER UNIT, so one
    config means the same thing on any voltage level; ``None`` resolves the documented
    defaults in ``pgml/data/defaults.yaml``. ``enforce_q_limits`` is a MODELING choice
    (what a regulating generator is allowed to do), so it belongs in the config even
    though it reaches the solver as a keyword.
    """

    model_config = ConfigDict(extra="forbid")

    calculation: Calculation = "harmonic"
    harmonic_orders: list[float] = Field(
        default_factory=lambda: [1, 3, 5, 7, 9, 11, 13],
        description="Integer harmonic orders h = f/f0 to solve (harmonic calculation "
        "only); non-integer (interharmonic) orders are rejected.",
    )
    slack: Slack = "ideal"
    symmetry: Optional[Symmetry] = Field(
        default=None,
        description="Per-phase vs balanced load modeling; None resolves from config.",
    )
    operating_point: Optional[dict] = Field(
        default=None, description="Per-appliance P/Q override; None = nameplate."
    )
    include_load_shunt: bool = False
    tol: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="PRIMARY convergence tolerance of the nonlinear fundamental: the "
        "largest nodal apparent-power mismatch in PER UNIT of s_base_va (what "
        "pandapower and power-grid-model converge on). None = the documented default "
        "solver.convergence.mismatch_pu (1e-8 pu).",
    )
    tol_update_pu: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="SECONDARY convergence tolerance: the largest per-row voltage "
        "update in PER UNIT of the node's line-to-neutral rated voltage; both criteria "
        "must hold. None = the documented default solver.convergence.update_pu "
        "(1e-8 pu).",
    )
    s_base_va: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Apparent-power base of the per-unit power mismatch. None = the "
        "documented default solver.convergence.s_base_va (1e6 VA).",
    )
    enforce_q_limits: Optional[bool] = Field(
        default=None,
        description="Whether a voltage-regulating generator's reactive limits bound "
        "its output (PV-to-PQ switching at the fundamental). None = the documented "
        "default appliance.generator.enforce_q_limits. False reproduces pandapower "
        "runpp's own default. Inert on a grid with no regulating generator.",
    )
    max_iter: int = Field(default=100, gt=0)

    @model_validator(mode="after")
    def _check(self) -> "SimulationConfig":
        if self.calculation == "harmonic":
            if not self.harmonic_orders:
                raise ValueError(
                    "harmonic calculation needs at least one harmonic order"
                )
            if any(o <= 0 for o in self.harmonic_orders):
                raise ValueError("harmonic_orders must be positive")
            if any(float(o) != int(o) for o in self.harmonic_orders):
                raise ValueError(
                    "harmonic_orders must be integer multiples of the fundamental; "
                    "interharmonics are not supported (spectra and the per-order "
                    "assembly are defined for integer orders only)."
                )
        return self


def _resolve_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _DTYPES[str(dtype)]
    except KeyError:
        raise InputError(
            f"dtype must be one of {sorted(_DTYPES)} or a torch complex dtype, got {dtype!r}"
        ) from None


class SolvedState:
    """A complete, lazily-derived view of a solved grid.

    ``v`` (the node voltage phasors, shape ``[*batch, H, N]`` complex) is eager — it is
    the direct, differentiable solve output. Currents, power flows, spectra and THD are
    derived ON ACCESS as tracked tensor operations on ``v``, so a caller that only reads
    voltages never pays for (or back-propagates through) the current/power derivation.

    The state holds the grid BY REFERENCE (a parameter deep-copy would sever the
    autograd identity of tensor-valued fields), and the lazy branch quantities are
    recomputed from it — with the same ``param_overrides`` the solve used, so voltage
    and currents always describe one consistent network. Do not mutate the grid's
    parameters between solving and reading lazy accessors; re-solve instead.

    Tensor accessors stay on the autograd tape (for ML / parameter recovery);
    :meth:`to_result_set` produces the detached, JSON-serializable records.
    """

    def __init__(
        self,
        *,
        grid: Grid,
        config: SimulationConfig,
        v: Tensor,
        frequencies_hz: Tensor,
        index,
        converged: bool,
        iterations: int,
        residual: float,
        dtype: torch.dtype,
        device: Optional[torch.device],
        param_overrides: Optional[dict] = None,
    ) -> None:
        self.grid = grid
        self.config = config
        self.v = v  # [*batch, H, N] complex, differentiable
        self.frequencies_hz = frequencies_hz  # [H]
        self.index = index
        self.converged = converged
        self.iterations = iterations
        self.residual = residual
        self.dtype = dtype
        self.device = device
        self.param_overrides = param_overrides

    # -- node quantities ---------------------------------------------------- #
    def node_voltages(self) -> Tensor:
        """All node voltage phasors ``[*batch, H, N]`` (the eager solve output)."""
        return self.v

    def voltage(self, node_id: int, phase: Phase) -> Tensor:
        """Voltage phasor at one ``(node, phase)`` across orders ``[*batch, H]``."""
        return self.v[..., :, self.index.row(node_id, phase)]

    def spectrum_at(self, node_id: int, phase: Phase) -> Tensor:
        """Harmonic spectrum at a ``(node, phase)``: voltage phasor per order ``[*batch, H]``."""
        return self.voltage(node_id, phase)

    def thd(self, node_id: int, phase: Phase) -> Tensor:
        """Voltage THD at a ``(node, phase)``: ``sqrt(sum_{h>1}|V_h|^2)/|V_1|``.

        Computed over the REQUESTED orders only — the orders in
        ``config.harmonic_orders``, which is what ``v`` holds. This is the IEC
        definition restricted to the solved spectrum, so the value depends on which
        orders the caller asked for: a solve of ``[1, 5, 7]`` reports the THD of those
        two harmonics, not of the full spectrum up to order 40 that a standard
        measurement would cover. Ask for every order that carries energy (the default
        ``[1, 3, 5, 7, 9, 11, 13]`` covers the dominant ones of a converter spectrum)
        when the number is to be compared with a measurement or a limit.

        Requires the fundamental (order 1) to be among the solved orders.
        """
        orders = [
            float(o) for o in self.frequencies_hz / float(self.grid.base_frequency_hz)
        ]
        try:
            i1 = orders.index(1.0)
        except ValueError:
            raise InputError(
                "thd requires order 1 (the fundamental) to be solved"
            ) from None
        vh = self.voltage(node_id, phase)  # [*batch, H]
        mag = vh.abs()
        harmonics_sq = mag.pow(2).sum(dim=-1) - mag[..., i1].pow(2)
        return torch.sqrt(harmonics_sq.clamp_min(0.0)) / mag[..., i1]

    # -- branch quantities (lazy) ------------------------------------------- #
    def branch_currents(self):
        """Per-branch terminal currents (list of ``BranchCurrent``); differentiable.

        Lazily reuses the assembly's primitive blocks (``Y_prim @ V_terminal``),
        applying the same ``param_overrides`` the voltage solve used.
        """
        from .assembly import branch_currents as _branch_currents

        return _branch_currents(
            self.grid,
            self.v,
            self.frequencies_hz,
            self.index,
            dtype=self.dtype,
            device=self.device,
            param_overrides=self.param_overrides,
        )

    def branch_flows(self):
        """Per-branch complex power flows ``S = V ⊙ conj(I)`` at each terminal.

        Returns a list of ``(branch_id, s_from, s_to)`` with ``s_*`` shaped
        ``[*batch, H, P]`` (complex); ``P = V·conj(I)`` per phase, differentiable.
        """
        flows = []
        for bc in self.branch_currents():
            v_from = self._gather(bc.from_node, bc.from_phases)
            s_from = v_from * torch.conj(bc.i_from)
            if bc.to_node is None:
                s_to = torch.zeros_like(s_from)
            else:
                v_to = self._gather(bc.to_node, bc.to_phases)
                s_to = v_to * torch.conj(bc.i_to)
            flows.append((bc.branch_id, s_from, s_to))
        return flows

    def _gather(self, node_id: int, phases) -> Tensor:
        rows = [self.index.row(node_id, ph) for ph in phases]
        return self.v[..., :, rows]  # [*batch, H, P]

    # -- serialization ------------------------------------------------------ #
    def to_result_set(
        self,
        *,
        result_set_id: int = 0,
        grid_id: Optional[int] = None,
        description: Optional[str] = None,
        include_branches: bool = True,
        include_power: bool = True,
    ) -> "ResultBundle":
        """Materialise the serializable :class:`ResultBundle` (JSON for REST / persist).

        Detaches tensors to plain floats (the serialized projection is not
        differentiable — use the tensor accessors for autograd). Single (unbatched)
        solve only; for batched scenarios use ``pgml.scenarios`` persistence.
        """
        if self.v.dim() != 2:
            raise InputError(
                "to_result_set serializes a single (unbatched) solve [H, N]; for batched "
                "scenarios use pgml.scenarios.run_scenarios + write_dataset."
            )
        f0 = float(self.grid.base_frequency_hz)
        freqs = [float(f) for f in self.frequencies_hz]
        node_phases = {int(n.id): tuple(n.phases) for n in self.grid.nodes}

        nodes: list[NodeResult] = []
        for nid, phases in node_phases.items():
            rows = [self.index.row(nid, ph) for ph in phases]
            for h, fhz in enumerate(freqs):
                vrow = self.v[h, rows].detach()
                nodes.append(
                    NodeResult(
                        result_set_id=result_set_id,
                        node_id=nid,
                        frequency_hz=fhz,
                        step=0,
                        phases=phases,
                        v_re=tuple(float(x) for x in vrow.real),
                        v_im=tuple(float(x) for x in vrow.imag),
                    )
                )

        branches: list[BranchResult] = []
        if include_branches:
            flows = (
                {bid: (sf, st) for bid, sf, st in self.branch_flows()}
                if include_power
                else {}
            )
            for bc in self.branch_currents():
                for h, fhz in enumerate(freqs):
                    i_from = bc.i_from[h].detach()
                    i_to = bc.i_to[h].detach()
                    kw = {}
                    if include_power and bc.branch_id in flows:
                        sf = flows[bc.branch_id][0][h].detach()
                        st = flows[bc.branch_id][1][h].detach()
                        kw = {
                            "p_from_w": tuple(float(x) for x in sf.real),
                            "q_from_var": tuple(float(x) for x in sf.imag),
                            "s_from_va": tuple(float(abs(x)) for x in sf),
                            "p_to_w": tuple(float(x) for x in st.real),
                            "q_to_var": tuple(float(x) for x in st.imag),
                            "s_to_va": tuple(float(abs(x)) for x in st),
                        }
                    branches.append(
                        BranchResult(
                            result_set_id=result_set_id,
                            branch_id=bc.branch_id,
                            branch_kind=self._branch_kind(bc.branch_id),
                            frequency_hz=fhz,
                            step=0,
                            from_phases=tuple(bc.from_phases),
                            to_phases=tuple(bc.to_phases),
                            i_from_re=tuple(float(x) for x in i_from.real),
                            i_from_im=tuple(float(x) for x in i_from.imag),
                            i_to_re=tuple(float(x) for x in i_to.real),
                            i_to_im=tuple(float(x) for x in i_to.imag),
                            **kw,
                        )
                    )

        return ResultBundle(
            result_set=ResultSet(
                id=result_set_id,
                description=description,
                grid_id=grid_id,
                base_frequency_hz=f0,
            ),
            diagnostics=SolverDiagnostics(
                result_set_id=result_set_id,
                step=0,
                converged=self.converged,
                iterations=self.iterations,
                residual_norm=self.residual,
            ),
            nodes=nodes,
            branches=branches,
        )

    def _branch_kind(self, branch_id: int) -> str:
        for b in self.grid.branches:
            if b.id == branch_id:
                return b.component
        return "line"


class ResultBundle(BaseModel):
    """JSON-serializable projection of a :class:`SolvedState` (REST / persistence).

    Composes the frozen :mod:`pgml.schemas.result_schema` records: the ``result_set``
    header, ``diagnostics``, and the per-(node|branch, order) ``nodes`` / ``branches``
    rows. ``model_dump_json()`` yields the REST body; persist it for training data.
    """

    model_config = ConfigDict(arbitrary_types_allowed=False)

    result_set: ResultSet
    diagnostics: SolverDiagnostics
    nodes: list[NodeResult]
    branches: list[BranchResult]


def simulate(
    grid: Grid,
    config: Optional[SimulationConfig] = None,
    *,
    device: Optional[str | torch.device] = None,
    dtype: str | torch.dtype = "complex128",
    precision: str = "full",
    param_overrides: Optional[dict] = None,
    harmonic_injection: Optional[dict] = None,
    node_sources: Optional[Sequence] = None,
    strict: bool = True,
    on_disconnected: str = "raise",
    linear_solver: str = "auto",
    block_rows: Optional[Sequence[Tensor]] = None,
    equilibrate: Optional[str] = None,
) -> SolvedState:
    """Run a simulation and return the differentiable :class:`SolvedState`.

    ``config`` (a :class:`SimulationConfig`, default = harmonic with standard orders)
    is the serializable definition of WHAT to compute; ``device`` / ``dtype`` /
    ``precision`` are the execution concerns. Gradients flow from ``grid`` parameters
    through the result's tensor accessors. ``param_overrides`` (per-parameter tensor
    substitution, see :func:`pgml.assembly.assemble_ybus`) applies to BOTH calculations:
    the state threads it into its lazy branch quantities so voltage and currents
    describe the same overridden network, and the harmonic calculation applies it to
    every order's admittance and device power.

    ``precision`` selects the working precision of the linear algebra: ``"full"``
    (default) factors at ``dtype``, ``"mixed"`` factors at complex64 and refines
    against complex128 residuals (see :func:`pgml.solver.solve_power_flow`) — the
    recommended recipe for throughput on an ill-conditioned SI-unit feeder, since a
    plain complex64 run of such a network keeps only a handful of digits.

    ``on_disconnected`` decides what the pre-solve connectivity check does with
    (node, phase) rows that have no galvanic path to an in-service source — an open
    switch, a line taken out of service, or a grid without a source. It applies to
    BOTH calculations:

    - ``"raise"`` (default) — raise :class:`~pgml.errors.ConnectivityError` naming the
      de-energized nodes, the separating branches, and the concrete fixes.
    - ``"zero"`` — solve the energized sub-grid and report exactly 0 V on the
      de-energized rows at every order (a de-energized conductor carries no voltage).
      The state keeps the FULL grid's row layout, so
      :meth:`SolvedState.node_voltages` / :meth:`SolvedState.voltage` /
      :meth:`SolvedState.branch_currents` / :meth:`SolvedState.branch_flows` stay
      addressable by the original node ids and report 0 on everything the island
      contains. This is the mode an operational tool wants when an element is
      switched out.
    - ``"ignore"`` — skip the check; a de-energized area then surfaces as a singular
      factorization or as non-convergence.

    :meth:`SolvedState.thd` is undefined on a de-energized row (its fundamental
    magnitude is 0); read it at energized nodes.

    ``linear_solver`` / ``block_rows`` select the inner factorization backend (see
    :func:`pgml.solver.solve_power_flow`) — execution concerns like device and dtype,
    not part of the serializable config. ``linear_solver="block"`` with the row
    partition of an independent-grid ensemble
    (``pgml.multigrid.MergedGrid.block_rows()``) factors each member on its own. Both
    apply to the harmonic calculation as well, where they select the backend of the
    fundamental solve AND of every per-order solve.

    ``equilibrate`` is the diagonal equilibration of every factored system (``None`` =
    the documented default ``solver.equilibration.mode``, ``"off"`` to factor each
    matrix as assembled). It changes the conditioning of the factorizations, not the
    result: see :func:`pgml.solver.solve_power_flow`.

    Raises :class:`~pgml.errors.ConvergenceError` if the nonlinear power flow does not
    converge (``strict=True``, the default); pass ``strict=False`` to return the
    (non-converged) state with ``converged=False`` instead.
    """
    config = config or SimulationConfig()
    cdt = _resolve_dtype(dtype)
    dev = torch.device(device) if isinstance(device, str) else device

    if config.calculation == "power_flow":
        pf = solve_power_flow(
            grid,
            slack=config.slack,
            tol=config.tol,
            tol_update_pu=config.tol_update_pu,
            s_base_va=config.s_base_va,
            enforce_q_limits=config.enforce_q_limits,
            max_iter=config.max_iter,
            dtype=cdt,
            precision=precision,
            device=dev,
            operating_point=config.operating_point,
            param_overrides=param_overrides,
            symmetry=config.symmetry,
            linear_solver=linear_solver,
            block_rows=block_rows,
            on_disconnected=on_disconnected,
            equilibrate=equilibrate,
        )
        v = pf.v.unsqueeze(-2)  # [*batch, 1, N]
        freqs = torch.tensor(
            [float(grid.base_frequency_hz)], dtype=v.real.dtype, device=v.device
        )
        index, converged, iterations, residual = (
            pf.index,
            pf.converged,
            pf.iterations,
            float(pf.residual),
        )
        pf_diag = pf.diagnostics
    else:
        hf = solve_harmonic_flow(
            grid,
            config.harmonic_orders,
            slack=config.slack,
            operating_point=config.operating_point,
            harmonic_injection=harmonic_injection,
            node_sources=node_sources,
            include_load_shunt=config.include_load_shunt,
            tol=config.tol,
            tol_update_pu=config.tol_update_pu,
            s_base_va=config.s_base_va,
            enforce_q_limits=config.enforce_q_limits,
            max_iter=config.max_iter,
            dtype=cdt,
            precision=precision,
            device=dev,
            symmetry=config.symmetry,
            on_disconnected=on_disconnected,
            param_overrides=param_overrides,
            linear_solver=linear_solver,
            block_rows=block_rows,
            equilibrate=equilibrate,
        )
        v, freqs, index = hf.v, hf.frequencies_hz, hf.index
        converged, iterations, residual = (
            hf.pf.converged,
            hf.pf.iterations,
            float(hf.pf.residual),
        )
        pf_diag = hf.pf.diagnostics

    if strict and not converged:
        diag: dict = {"calculation": config.calculation, "max_iter": config.max_iter}
        cause = ""
        if pf_diag is not None:
            diag.update(pf_diag.as_dict())
            cause = f" — {pf_diag.likely_cause}" if pf_diag.likely_cause else ""
        raise ConvergenceError(
            f"{config.calculation} did not converge in {iterations} iterations "
            f"(residual {residual:.3e}){cause}",
            iterations=iterations,
            residual=residual,
            diagnostics=diag,
        )

    return SolvedState(
        grid=grid,
        config=config,
        v=v,
        frequencies_hz=freqs,
        index=index,
        converged=converged,
        iterations=iterations,
        residual=residual,
        dtype=cdt,
        device=dev,
        param_overrides=param_overrides,
    )


def simulate_serializable(
    grid: Grid, config: Optional[SimulationConfig] = None, **kwargs
) -> ResultBundle:
    """Convenience wrapper: :func:`simulate` then :meth:`SolvedState.to_result_set`.

    Returns the JSON-serializable :class:`ResultBundle` directly — the entry point a
    REST handler or a file-persistence step calls. Keyword arguments are forwarded to
    :func:`simulate` (e.g. ``device``, ``dtype``, ``on_disconnected``).
    """
    return simulate(grid, config, **kwargs).to_result_set(
        grid_id=getattr(grid, "id", None)
    )


__all__ = [
    "SimulationConfig",
    "SolvedState",
    "ResultBundle",
    "simulate",
    "simulate_serializable",
]
