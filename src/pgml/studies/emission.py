"""Spread of inverter emission through a grid: harmonics, interharmonics, supraharmonics.

:func:`emission_flow` solves the linear network at an arbitrary frequency grid with each
modelled inverter as its Norton equivalent ``i = I_N - Y v``
(:class:`pgml.inverter.GridFollowingInverter`), on top of the solved fundamental
operating point:

A modelled inverter is a Generator or Storage with an ``inverter_model`` block in the grid
(:class:`~pgml.schemas.grid_schema.InverterModel`) or an entry of the ``inverters``
argument, which overrides the block. Every other device keeps the simple injection mode.

1. the fundamental power flow gives every device's operating point and terminal state;
2. each inverter evaluates ``I_N(f)`` and ``Y(f)`` at that state (control admittance,
   filter, PWM residues, dead time, MPPT lines, DC-link ripple);
3. the network matrix is the harmonic flow's own ``Y(f)`` (passive network, source
   Norton shunt, load harmonic shunts, explicit DER impedances; evaluated at any
   frequency, as the impedance scan does) plus each inverter's ``Y(f)``;
4. the right-hand side holds the inverter Norton currents, and at integer orders the
   spectrum injections of every other device and the background distortion
   (``node_sources``), exactly as :func:`pgml.solve_harmonic_flow` forms them. Without
   inverter models the study reproduces the harmonic flow at integer orders.

Aggregation
-----------
Each inverter's sources are split into components locked to its fundamental (dead time,
DC-link ripple) and components without a common phase reference between units (PWM
residues follow free-running carriers, MPPT lines independent clocks). Two results are
returned:

- ``v``: the coherent solution with every source at its stated phase;
- ``v_rms``: the expected magnitude ``sqrt(E|V|^2)`` when the unlocked components of
  different units have independent uniformly distributed phases,
  ``E|V|^2 = |V_locked|^2 + sum_d |V_unlocked,d|^2`` (the cross terms of independent
  uniform phases average to zero). This is the general summation law with exponent 2
  (IEC TR 61000-3-6 uses that exponent above order 10); the locked part adds as phasors.

Primary and secondary emission
------------------------------
Following Bollen and Rönnberg (2016), the primary emission of a device is the current
at its terminals driven by its own sources (all other sources zero), the secondary
emission the part driven by everything else. Both are returned per inverter
(``device_primary``, ``device_current - device_primary``).

Stability indicator
-------------------
For each inverter the network seen from its terminals (everything else, including the
other devices' admittances) is ``Z_g = (Z_tt^-1 - Y)^-1``, with ``Z_tt`` the driving-point
impedance of the full system at the device. ``stability_margin`` is the smallest singular
value of ``I + Z_g Y = Z_g Z_tt^-1`` per frequency, the distance of the impedance ratio
from ``-1`` (Sun 2011's ``Z_g / Z_o`` criterion read per frequency). It is a margin
indicator, not a Nyquist test: a small value flags a frequency where the steady-state
result may have no physical meaning.

The study is differentiable in every tensor-valued grid and inverter parameter and in the
operating point; the frequency grid is chunked against ``solver.harmonic.system_budget_mb``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor

from pgml import defaults
from pgml.assembly import node_phase_index
from pgml.assembly._incidence import build_incidence, group_appliances, used_rows
from pgml.assembly._load_shunt import (
    generation_shunt_is_neglected,
    resolve_shunt_basis,
    resolve_shunt_model_name,
)
from pgml.assembly._scatter import scatter_blocks_into
from pgml.assembly._stamps import _cdtype, _rdtype
from pgml.assembly._symmetry import resolve_asymmetric
from pgml.errors import InputError
from pgml.inverter import (
    GridFollowingInverter,
    InverterState,
    NortonEquivalent,
    from_schema,
)
from pgml.assembly._symmetry import resolve_connection
from pgml.schemas.grid_schema import Generator, Grid, Phase, Storage, WindingConnection
from pgml.solver import lu_factor_system, solve_factored, solve_power_flow
from pgml.solver.harmonic_flow import (
    _apply_node_sources,
    _effective_element_power,
    _element_terminal_voltage,
    _harmonic_admittance,
    harmonic_injections,
)

from ._common import check_study, resolve_complex_dtype


@dataclass(frozen=True)
class EmissionFlow:
    """Result of :func:`emission_flow`.

    Attributes
    ----------
    frequencies_hz:
        Real ``[F]``, ascending, without the fundamental.
    node_ids, phases:
        Row labels of the voltage axis (the grid's full node-phase layout).
    v:
        Complex ``[*batch, F, N]`` coherent node-phase voltages (RMS phasors).
    v_rms:
        Real ``[*batch, F, N]`` expected magnitude with random-phase unlocked sources.
    v_locked:
        Complex ``[*batch, F, N]`` part driven by locked sources only.
    device_current, device_primary:
        ``{device_id: [*batch, F, E]}`` coherent terminal current leaving each inverter
        and its primary part.
    norton:
        ``{device_id: NortonEquivalent}`` as evaluated at the solved operating point.
    stability_margin:
        ``{device_id: [*batch, F]}`` (see module docstring), or empty.
    f1_hz:
        The fundamental frequency.
    v1:
        Complex ``[*batch, N]`` solved fundamental node-phase voltages (RMS phasors), or
        ``None`` for a result restored without them.
    """

    frequencies_hz: Tensor
    node_ids: tuple
    phases: tuple
    v: Tensor
    v_rms: Tensor
    v_locked: Tensor
    device_current: dict = field(default_factory=dict)
    device_primary: dict = field(default_factory=dict)
    norton: dict = field(default_factory=dict)
    stability_margin: dict = field(default_factory=dict)
    f1_hz: float = 50.0
    v1: Optional[Tensor] = None

    def rows(self, node_id: int, phase=None) -> list[int]:
        """Row indices of ``node_id`` (all its phases, or one)."""
        ph = (
            None
            if phase is None
            else Phase(str(getattr(phase, "value", phase)).lower()).value
        )
        return [
            k
            for k, (nid, p) in enumerate(zip(self.node_ids, self.phases))
            if nid == node_id and (ph is None or p == ph)
        ]

    def band_rms(self, f_lo_hz: float, f_hi_hz: float, *, kind: str = "rms") -> Tensor:
        """RMS over the lines in ``[f_lo, f_hi)`` per node-phase, ``[*batch, N]``.

        ``kind="rms"`` sums the random-phase expectations ``v_rms**2`` (lines at
        different frequencies add in power), ``"coherent"`` uses ``|v|``.
        """
        mask = (self.frequencies_hz >= f_lo_hz) & (self.frequencies_hz < f_hi_hz)
        x = self.v_rms if kind == "rms" else self.v.abs()
        return torch.sqrt((x[..., mask, :] ** 2).sum(dim=-2))

    def compatibility(self, node_id: int, phase=None, *, network: str = "lv") -> dict:
        """Solved voltages against the compatibility levels of IEC 61000-2-2 / -2-12.

        For the row of ``node_id`` / ``phase`` (the first row of the node when ``phase``
        is ``None``): per integer harmonic order the ratio of ``v_rms`` to the order's
        level (:func:`pgml.scenarios.iec61000_2_2_harmonic_levels`; ``network`` ``"lv"`` or
        ``"mv"``), and for a low-voltage network per 200 Hz band from 2 kHz up to the
        highest solved frequency the ratio of the band rms of ``v_rms`` to the band level
        (:func:`pgml.scenarios.iec61000_2_2_band_level_v`). Voltages are phase to ground
        rows of the solve, the compatibility levels are stated phase to neutral; the two
        agree for a solidly earthed neutral at the node. A band without a solved line
        reads 0. Returns ``{"harmonic_order", "harmonic_ratio" [*batch, H],
        "band_centre_hz", "band_ratio" [*batch, B], "max_ratio" [*batch]}``.
        """
        from pgml.scenarios.iec61000_2_2 import (
            iec61000_2_2_band_level_v,
            iec61000_2_2_harmonic_levels,
        )

        if self.v1 is None:
            raise InputError("compatibility needs the fundamental voltages (v1).")
        row = self.rows(node_id, phase)[0]
        u1 = self.v1[..., row].abs()  # [*batch]
        f = self.frequencies_hz
        levels = iec61000_2_2_harmonic_levels(network)
        orders, ratios = [], []
        for h, level in levels.items():
            k = torch.nonzero((f - h * self.f1_hz).abs() < 1e-6 * h * self.f1_hz)
            if k.numel():
                orders.append(h)
                ratios.append(self.v_rms[..., int(k[0, 0]), row] / (u1 * level))
        harm = torch.stack(ratios, dim=-1) if ratios else torch.zeros((*u1.shape, 0))
        centres, band = [], []
        if network == "lv":
            width = 200.0
            start = 2100.0 if self.f1_hz != 60.0 else 2500.0
            n_bands = int((float(f.max()) - start + width / 2) // width) + 1
            for j in range(max(0, n_bands)):
                centre = start + j * width
                limit = iec61000_2_2_band_level_v(
                    centre, 1.0, f1_hz=self.f1_hz
                )  # per unit of u1 below 9 kHz, absolute volts above
                if limit is None:
                    continue
                mask = (f >= centre - width / 2) & (f < centre + width / 2)
                rms = torch.sqrt((self.v_rms[..., mask, row] ** 2).sum(-1))
                scale = u1 if centre < 9000.0 else torch.ones_like(u1)
                centres.append(centre)
                band.append(rms / (scale * limit))
        bands = torch.stack(band, dim=-1) if band else torch.zeros((*u1.shape, 0))
        both = torch.cat([harm, bands], dim=-1)
        return {
            "harmonic_order": orders,
            "harmonic_ratio": harm,
            "band_centre_hz": centres,
            "band_ratio": bands,
            "max_ratio": both.amax(dim=-1) if both.shape[-1] else torch.zeros_like(u1),
        }

    def to_dict(self) -> dict:
        """JSON-ready: frequencies, labels and the voltages (``*_re_v`` / ``*_im_v``)."""
        with torch.no_grad():
            out = {
                "study": "emission_flow",
                "f1_hz": self.f1_hz,
                "frequencies_hz": self.frequencies_hz.cpu().tolist(),
                "node_ids": list(self.node_ids),
                "phases": list(self.phases),
                "v_re_v": self.v.real.cpu().tolist(),
                "v_im_v": self.v.imag.cpu().tolist(),
                "v_rms_v": self.v_rms.cpu().tolist(),
                "v_locked_re_v": self.v_locked.real.cpu().tolist(),
                "v_locked_im_v": self.v_locked.imag.cpu().tolist(),
                "device_current_re_a": {
                    str(k): x.real.cpu().tolist()
                    for k, x in self.device_current.items()
                },
                "device_current_im_a": {
                    str(k): x.imag.cpu().tolist()
                    for k, x in self.device_current.items()
                },
                "device_primary_re_a": {
                    str(k): x.real.cpu().tolist()
                    for k, x in self.device_primary.items()
                },
                "device_primary_im_a": {
                    str(k): x.imag.cpu().tolist()
                    for k, x in self.device_primary.items()
                },
                "stability_margin": {
                    str(k): x.cpu().tolist() for k, x in self.stability_margin.items()
                },
            }
            if self.v1 is not None:
                out["v1_re_v"] = self.v1.real.cpu().tolist()
                out["v1_im_v"] = self.v1.imag.cpu().tolist()
        return out

    @classmethod
    def from_dict(cls, data: Mapping) -> "EmissionFlow":
        """Inverse of :meth:`to_dict` (the Norton equivalents are not serialized)."""
        check_study(data, "emission_flow")

        def cplx(re, im):
            return torch.complex(
                torch.as_tensor(re, dtype=torch.float64),
                torch.as_tensor(im, dtype=torch.float64),
            )

        return cls(
            frequencies_hz=torch.as_tensor(data["frequencies_hz"], dtype=torch.float64),
            node_ids=tuple(int(x) for x in data["node_ids"]),
            phases=tuple(data["phases"]),
            v=cplx(data["v_re_v"], data["v_im_v"]),
            v_rms=torch.as_tensor(data["v_rms_v"], dtype=torch.float64),
            v_locked=cplx(data["v_locked_re_v"], data["v_locked_im_v"]),
            device_current={
                int(k): cplx(v, data["device_current_im_a"][k])
                for k, v in data["device_current_re_a"].items()
            },
            device_primary={
                int(k): cplx(v, data["device_primary_im_a"][k])
                for k, v in data["device_primary_re_a"].items()
            },
            stability_margin={
                int(k): torch.as_tensor(v, dtype=torch.float64)
                for k, v in data["stability_margin"].items()
            },
            f1_hz=float(data["f1_hz"]),
            v1=cplx(data["v1_re_v"], data["v1_im_v"]) if "v1_re_v" in data else None,
        )


@dataclass
class _Device:
    """An inverter bound to its grid appliance, incidence and solved fundamental state."""

    appliance: object
    model: GridFollowingInverter
    state: InverterState
    m: Tensor  # complex incidence [E, U]
    rows: Tensor  # full-layout rows [U]
    v1: Tensor  # [*batch, E]
    i1: Tensor  # [*batch, E] leaving the device
    norton: Optional[NortonEquivalent] = None


def emission_frequencies(
    grid: Grid,
    inverters: Optional[Mapping[int, GridFollowingInverter]] = None,
    *,
    harmonic_orders: Optional[Sequence[int]] = None,
    frequencies_hz: Optional[Sequence[float]] = None,
    f_max_hz: Optional[float] = None,
) -> list[float]:
    """The frequency grid :func:`emission_flow` solves: every model line plus extras.

    The union of each inverter's emission lines (the grid's ``inverter_model`` blocks and
    ``inverters``, :meth:`GridFollowingInverter.emission_frequencies`), the integer ``harmonic_orders`` and any explicit
    ``frequencies_hz``, without the fundamental, ascending, at most ``f_max_hz``.
    """
    f0 = float(grid.base_frequency_hz)
    lines: set = set()
    for model in _collect_models(grid, inverters).values():
        lines.update(model.emission_frequencies(f0, f_max_hz))
    for h in harmonic_orders or ():
        lines.add(round(float(h) * f0, 6))
    for f in frequencies_hz or ():
        lines.add(round(float(f), 6))
    out = sorted(f for f in lines if f > 0 and abs(f - f0) > 1e-6)
    if f_max_hz is not None:
        out = [f for f in out if f <= f_max_hz]
    if not out:
        raise InputError("emission_flow: the frequency grid is empty.")
    return out


def _collect_models(grid: Grid, inverters) -> dict:
    """Inverter models of the study: the grid's ``inverter_model`` blocks, overridden by
    the explicit ``inverters`` mapping."""
    models = {
        int(a.id): from_schema(a.inverter_model)
        for a in grid.appliances
        if isinstance(a, (Generator, Storage))
        and a.in_service
        and getattr(a, "inverter_model", None) is not None
    }
    models.update({int(k): v for k, v in (inverters or {}).items()})
    return models


def emission_flow(
    grid: Grid,
    inverters: Optional[Mapping[int, GridFollowingInverter]] = None,
    *,
    states: Optional[Mapping[int, InverterState]] = None,
    harmonic_orders: Optional[Sequence[int]] = None,
    frequencies_hz: Optional[Sequence[float]] = None,
    f_max_hz: Optional[float] = None,
    operating_point: Optional[dict] = None,
    harmonic_injection: Optional[dict] = None,
    node_sources: Optional[Sequence] = None,
    load_shunt: Optional[str] = None,
    load_shunt_basis: Optional[str] = None,
    slack: str = "norton",
    method: str = "current_injection",
    stability: bool = True,
    admittance: str = "model",
    dtype="complex128",
    device: Optional[torch.device] = None,
) -> EmissionFlow:
    """Solve the emission spread with inverter Norton models at arbitrary frequencies.

    Parameters
    ----------
    grid:
        The grid. Every key of ``inverters`` must be an in-service Generator or Storage
        id whose number of phases matches the model (1, or 3 as phases ``a, b, c``) and
        that carries no ``harmonic_impedance`` (the model replaces it).
    inverters:
        ``{device_id: GridFollowingInverter}``, added to (and overriding) the devices
        whose ``inverter_model`` block is set in the grid. The device's ``spectrum`` is
        ignored: its emission comes from the model.
    states:
        ``{device_id: InverterState}``. A missing entry takes the ``u_dc_v`` of the
        device's ``inverter_model`` block (bridge switching, relay closed) and raises
        without one. ``u_dc_v``, ``bridge_on`` and ``connected`` may carry the scenario
        batch of ``operating_point``.
    harmonic_orders, frequencies_hz, f_max_hz:
        Extend / cap the frequency grid (:func:`emission_frequencies`). The other
        devices' spectra and ``node_sources`` act at the integer orders of the grid; a
        voltage-kind node source's Thevenin shunt is therefore stamped at those orders
        only, as in :func:`pgml.solve_harmonic_flow`, and the network has no background
        source at other frequencies.
    operating_point, harmonic_injection, node_sources:
        As in :func:`pgml.solve_harmonic_flow` (operating-point overrides, other devices'
        spectra, background distortion).
    load_shunt, load_shunt_basis, method:
        As in :func:`pgml.solve_harmonic_flow` (load harmonic shunts, fundamental
        solver).
    slack:
        Slack model of the fundamental solve, ``"norton"`` (default) or ``"ideal"``. The
        inverter sources follow the fundamental terminal voltage (the modulation index
        sets the PWM residues), so the default keeps the source's Thevenin impedance at
        the fundamental, as the study already does at every other frequency. ``"ideal"``
        pins the source bus instead (the default of :func:`pgml.solve_harmonic_flow`); on
        a weak single-phase supply it put the modulation index 1.7 % low and single
        carrier lines 3 to 9 % off against a switching simulation. Pass the same slack to
        :func:`pgml.solve_harmonic_flow` when comparing the two.
    stability:
        Compute the per-device stability indicator (one extra right-hand side per
        element).
    admittance:
        Which inverter admittance the network sees, for separating the mechanisms
        that shape the spread: ``"model"`` (default, the closed-loop admittance through
        the filter), ``"passive"`` (the filter with the bridge shorted, i.e. the control's
        contribution removed) or ``"none"`` (pure current sources, the classic harmonic
        current-source model). The Norton currents are the same in all three; they are
        short-circuit currents, so the terminal current of a unit changes with the mode.
    dtype, device:
        Complex working dtype (``"complex128"`` / ``"complex64"`` or the torch dtype)
        and device.

    Returns
    -------
    EmissionFlow
    """
    if admittance not in ("model", "passive", "none"):
        raise InputError(
            f"emission_flow: admittance must be 'model', 'passive' or 'none', got "
            f"{admittance!r}."
        )
    cdt = resolve_complex_dtype(dtype)
    rdt = _rdtype(cdt)
    cdt = _cdtype(cdt)
    f0 = float(grid.base_frequency_hz)
    shunt = resolve_shunt_model_name(load_shunt)
    basis = resolve_shunt_basis(load_shunt_basis)
    states = dict(states or {})
    by_id = {int(a.id): a for a in grid.appliances}
    inverters = _collect_models(grid, inverters)
    for dev_id in inverters:
        spec = getattr(by_id.get(int(dev_id)), "inverter_model", None)
        if int(dev_id) not in states and spec is not None and spec.u_dc_v is not None:
            states[int(dev_id)] = InverterState(u_dc_v=spec.u_dc_v)
    node_map = {n.id: n for n in grid.nodes}

    for dev_id in inverters:
        a = by_id.get(int(dev_id))
        if a is None or not isinstance(a, (Generator, Storage)) or not a.in_service:
            raise InputError(
                f"emission_flow: inverter id {dev_id} is not an in-service Generator or "
                "Storage of the grid."
            )
        if a.harmonic_impedance is not None:
            raise InputError(
                f"emission_flow: device {dev_id} carries a harmonic_impedance; the "
                "inverter model replaces it, remove one of the two."
            )
        if shunt != "none" and not generation_shunt_is_neglected(a):
            raise InputError(
                f"emission_flow: device {dev_id} would also receive a derived harmonic "
                "shunt (appliance.harmonic_shunt.generation_model or its harmonic_model); "
                "the inverter model already holds its admittance."
            )
        if int(dev_id) not in states:
            raise InputError(f"emission_flow: no InverterState for device {dev_id}.")
        model = inverters[dev_id]
        if len(a.phases) != model.n_elements:
            raise InputError(
                f"emission_flow: device {dev_id} has {len(a.phases)} phase(s), its model "
                f"{model.n_elements} element(s)."
            )
        if model.n_elements == 3 and resolve_connection(a) == WindingConnection.DELTA:
            raise InputError(
                f"emission_flow: three-phase device {dev_id} is DELTA-connected; the "
                "sequence-domain model is per WYE element."
            )
        if model.n_elements == 3 and tuple(a.phases) != (Phase.A, Phase.B, Phase.C):
            raise InputError(
                f"emission_flow: three-phase device {dev_id} must list phases (a, b, c) "
                "in order for the sequence-domain model."
            )

    freqs = emission_frequencies(
        grid,
        inverters,
        harmonic_orders=harmonic_orders,
        frequencies_hz=frequencies_hz,
        f_max_hz=f_max_hz,
    )
    f_t = torch.as_tensor(freqs, dtype=rdt, device=device or "cpu")
    if device is None:
        device = f_t.device

    # 1. Fundamental operating point.
    pf = solve_power_flow(
        grid,
        slack=slack,
        method=method,
        operating_point=operating_point,
        dtype=cdt,
        device=device,
    )
    v1 = pf.v
    op = pf.resolved_operating_point(operating_point)
    full = node_phase_index(grid)
    asymmetric = resolve_asymmetric(grid, op, mode=None)

    # 2. Inverter Norton equivalents at the solved state.
    devices: dict[int, _Device] = {}
    for dev_id, model in inverters.items():
        a = by_id[int(dev_id)]
        grp = group_appliances([a], node_map)[0]
        m = build_incidence(grp, rdt, device).to(cdt)
        rows = used_rows(grp, full, device)[0]
        vt = _element_terminal_voltage(m, rows, v1)
        s_el = _effective_element_power(
            a, grp, node_map, vt, op, None, asymmetric, cdt, rdt, device
        )
        vtc = torch.conj(vt)
        safe = torch.where(vtc.abs() < 1e-300, torch.ones_like(vtc), vtc)
        i_abs = torch.where(
            vtc.abs() < 1e-300, torch.zeros_like(s_el), torch.conj(s_el) / safe
        )
        d = _Device(a, model, states[int(dev_id)], m, rows, vt, -i_abs)
        nor = model.norton(f_t, f1_hz=f0, v1=vt, i1=-i_abs, state=d.state)
        if admittance == "passive":
            nor = dataclasses.replace(nor, admittance=nor.admittance_passive)
        elif admittance == "none":
            nor = dataclasses.replace(nor, admittance=torch.zeros_like(nor.admittance))
        d.norton = nor
        devices[int(dev_id)] = d

    # Integer orders of the grid: other devices' spectra and the background act there.
    int_pos = [k for k, f in enumerate(freqs) if abs(f / f0 - round(f / f0)) < 1e-9]
    int_orders = [int(round(freqs[k] / f0)) for k in int_pos]
    silence = {int(k): {} for k in inverters}
    injection = {**(harmonic_injection or {}), **silence}

    n_full = full.size
    batch = torch.broadcast_shapes(
        v1.shape[:-1],
        *[d.norton.source_locked.shape[:-2] for d in devices.values()],
        *[d.norton.admittance.shape[:-3] for d in devices.values()],
    )
    n_f = len(freqs)
    other = torch.zeros((*batch, n_f, n_full), dtype=cdt, device=device)
    bg_diag = None
    if int_orders:
        i_int = harmonic_injections(
            grid,
            v1,
            int_orders,
            operating_point=op,
            harmonic_injection=injection,
            dtype=cdt,
            device=device,
        )  # [*b, Hi, N]
        pos = torch.as_tensor(int_pos, dtype=torch.int64, device=device)
        if node_sources:
            zero_y = torch.zeros(
                (len(int_orders), n_full, n_full), dtype=cdt, device=device
            )
            y_bg, i_int = _apply_node_sources(
                node_sources,
                grid,
                v1,
                full,
                int_orders,
                zero_y,
                i_int,
                cdt,
                rdt,
                device,
            )
            bg_diag = torch.zeros(
                (*y_bg.shape[:-3], n_f, n_full), dtype=cdt, device=device
            )
            bg_diag = bg_diag.index_add(-2, pos, y_bg.diagonal(dim1=-2, dim2=-1))
        other = other.index_add(
            -2, pos, i_int.broadcast_to((*batch, len(int_orders), n_full))
        )

    def nodal(d: _Device, x: Tensor) -> Tensor:
        """Element currents ``[*b, F, E]`` injected -> full nodal vector ``[*b, F, N]``."""
        u = torch.einsum("eu,...e->...u", d.m, x)
        out = torch.zeros((*u.shape[:-1], n_full), dtype=cdt, device=device)
        return out.index_add(-1, d.rows, u)

    ids = list(devices)
    rhs_full = [other.broadcast_to((*batch, n_f, n_full))]
    for i in ids:
        d = devices[i]
        rhs_full.append(
            nodal(d, d.norton.source_locked).broadcast_to((*batch, n_f, n_full))
        )
        rhs_full.append(
            nodal(d, d.norton.source_unlocked).broadcast_to((*batch, n_f, n_full))
        )
    unit_cols: list[tuple[int, int]] = []
    if stability:
        for i in ids:
            d = devices[i]
            e = d.m.shape[0]
            for k in range(e):
                x = torch.zeros((n_f, e), dtype=cdt, device=device)
                x[:, k] = 1.0
                rhs_full.append(nodal(d, x).broadcast_to((*batch, n_f, n_full)))
                unit_cols.append((i, k))
    rhs_full_t = torch.stack(rhs_full, dim=0)  # [R, *b, F, N]
    n_rhs = rhs_full_t.shape[0]

    unmodelled_generation = any(
        isinstance(a, (Generator, Storage))
        and a.in_service
        and int(a.id) not in devices
        for a in grid.appliances
    )

    # 3./4. Network matrix per frequency chunk, inverter admittances stamped, solve.
    itemsize = torch.empty((), dtype=cdt).element_size()
    budget = float(defaults.get("solver.harmonic.system_budget_mb")) * 1024.0**2
    n_b = max(1, int(torch.Size(batch).numel()))
    per_freq = itemsize * n_b * (2 * n_full**2 + 2 * n_rhs * n_full)
    chunk = max(1, int(budget // per_freq))
    fusion = None
    v_parts = []
    for start in range(0, n_f, chunk):
        sl = slice(start, min(start + chunk, n_f))
        f_chunk = f_t[sl]
        orders = [float(x) / f0 for x in freqs[sl]]
        y, index, fusion = _harmonic_admittance(
            grid,
            f_chunk,
            orders,
            v1=v1,
            operating_point=op,
            load_shunt=shunt,
            load_shunt_basis=basis,
            symmetry=None,
            dtype=cdt,
            device=device,
            fusion=fusion,
            # The generation-shunt notice would name the modelled inverters, whose
            # admittance is the model's; it stays on for unmodelled generators.
            warn=start == 0 and unmodelled_generation,
        )
        to_red = None if fusion is None else fusion.row_to_reduced.to(device)
        if bg_diag is not None:
            diag = bg_diag[..., sl, :]
            if fusion is not None:
                diag = fusion.restrict(diag)
            y = y + torch.diag_embed(diag)
        for i in ids:
            d = devices[i]
            y_el = d.norton.admittance[..., sl, :, :]
            block = torch.einsum("eu,...ef,fv->...uv", d.m, y_el, d.m)
            rows = d.rows if to_red is None else to_red[d.rows]
            y = scatter_blocks_into(
                y, block.unsqueeze(-3), rows.unsqueeze(0), rows.unsqueeze(0)
            )
        rhs = rhs_full_t[..., sl, :]
        if fusion is not None:
            rhs = fusion.restrict(rhs)
        fac = lu_factor_system(y)
        x = solve_factored(
            fac, rhs.broadcast_to((n_rhs, *batch, rhs.shape[-2], rhs.shape[-1]))
        )
        if fusion is not None:
            x = fusion.prolong(x)
        v_parts.append(x)
    x_all = torch.cat(v_parts, dim=-2)  # [R, *b, F, N]

    # 5. Aggregate.
    v_other = x_all[0]
    v_locked = v_other
    v = v_other
    power = torch.zeros(v_other.shape, dtype=rdt, device=device)
    for j, i in enumerate(ids):
        v_l, v_u = x_all[1 + 2 * j], x_all[2 + 2 * j]
        v_locked = v_locked + v_l
        v = v + v_l + v_u
        power = power + v_u.abs() ** 2
    p_total = v_locked.abs() ** 2 + power
    # sqrt has an infinite derivative at 0 (frequencies where nothing emits).
    safe = torch.where(p_total > 0, p_total, torch.ones_like(p_total))
    v_rms = torch.where(p_total > 0, torch.sqrt(safe), torch.zeros_like(p_total))

    def element_v(d: _Device, x: Tensor) -> Tensor:
        return torch.einsum("eu,...u->...e", d.m, x[..., d.rows])

    current, primary, margin = {}, {}, {}
    for j, i in enumerate(ids):
        d = devices[i]
        y_el = d.norton.admittance
        src = d.norton.source
        v_t = element_v(d, v)
        current[i] = src - torch.einsum("...ef,...f->...e", y_el, v_t)
        v_own = element_v(d, x_all[1 + 2 * j] + x_all[2 + 2 * j])
        primary[i] = src - torch.einsum("...ef,...f->...e", y_el, v_own)
    if stability:
        col = 1 + 2 * len(ids)
        for i in ids:
            d = devices[i]
            e = d.m.shape[0]
            z_tt = torch.stack(
                [element_v(d, x_all[col + k]) for k in range(e)], dim=-1
            )  # [*b, F, E (response), E (injection)]
            col += e
            y_el = d.norton.admittance
            eye = torch.eye(e, dtype=cdt, device=device)
            # Z_g = (Z_tt^-1 - Y)^-1 and I + Z_g Y = Z_g Z_tt^-1.
            z_inv = torch.linalg.inv(z_tt)
            z_g = torch.linalg.inv(z_inv - y_el)
            ratio = eye + z_g @ y_el
            margin[i] = torch.linalg.svdvals(ratio)[..., -1]

    return EmissionFlow(
        frequencies_hz=f_t,
        node_ids=tuple(full.node_id_of(r) for r in range(n_full)),
        phases=tuple(full.phase_of(r).value for r in range(n_full)),
        v=v,
        v_rms=v_rms,
        v_locked=v_locked,
        device_current=current,
        device_primary=primary,
        norton={i: devices[i].norton for i in ids},
        stability_margin=margin,
        f1_hz=f0,
        v1=v1,
    )
