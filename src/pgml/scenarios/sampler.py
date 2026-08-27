"""Reproducible scenario sampler: ScenarioConfig -> batched operating points.

Turns a `ScenarioConfig` (+ its seed) into a batch of ``B = n_samples`` realized
operating points, as the batched ``operating_point`` override the solver consumes
(``{appliance_id: {"p_w": Tensor[B], "q_var": Tensor[B]}}`` for balanced specs, or
``{appliance_id: {"p_per_phase_w": [Tensor[B], ...], ...}}`` for per-phase specs),
plus the raw sampled values per parameter (useful as ML inputs/labels). Deterministic:
same config+seed -> identical output.

All three methods produce unit-cube samples ``U in [0,1]^(B, D)`` then map each column
through that parameter's ``distribution.icdf`` — so QMC (Sobol/LHS) and plain RNG share
one transform path. ``D`` is laid out as: one column per declared
:class:`~pgml.scenarios.config.LatentFactor`, then per spec a block of base columns
(component-level draws) followed by per-phase columns (independent / small-imbalance).

Correlation (the realistic middle ground between ``per="each"`` and ``per="shared"``)
is a single-factor Gaussian copula in normal-score space; per-phase ``symmetry`` writes
per-phase overrides that promote the solve to asymmetric automatically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch import Tensor

from pgml.errors import InputError
from pgml.schemas.grid_schema import (
    Grid,
    InjectionAppliance,
    Source,
    StaticSpectrum,
)

from .config import (
    CartesianConfig,
    CoherentSpectrumConfig,
    NodeInjectionSweepConfig,
    ParameterSpec,
    Perturbation,
    ScenarioConfig,
    SpectrumSweepConfig,
)
from .en50160 import en50160_limit
from .iec61000_3_2 import iec61000_3_2_device_caps

# A built batch may originate from a random/QMC config or a cartesian config.
_AnyConfig = "ScenarioConfig | CartesianConfig"

_U_EPS = 1e-7  # clamp unit samples off {0,1} so Gaussian-tail icdf stays finite.
_SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class SampledScenarios:
    """Realized batch of operating points.

    Attributes
    ----------
    operating_point:
        ``{appliance_id: {...}}`` — pass straight to ``solve_power_flow`` /
        ``solve_harmonic_flow`` as ``operating_point=...``. Each entry holds totals
        (``p_w`` / ``q_var``, balanced specs) and/or per-phase overrides
        (``p_per_phase_w`` / ``q_per_phase_var``, asymmetric specs; per-phase keys
        auto-promote the solve to asymmetric).
    harmonic_injection:
        ``{appliance_id: {order: (mag_pu, phase_deg)}}`` (mag/phase are ``[B]`` tensors)
        — pass to ``solve_harmonic_flow(harmonic_injection=...)``. Built from ``h_mag``
        / ``h_phase`` specs, seeded from each device's stored ``StaticSpectrum`` so
        unspecified orders survive. Empty unless the config has harmonic fields.
    samples:
        ``{parameter_name: Tensor}`` — the raw sampled values (the reproducible ML
        input record). Shape ``[B, d]`` for component-level specs (d = #matched
        components for ``per="each"``/correlated, else 1), ``[B, n_comp, n_phase]``
        for ``symmetry="independent"`` specs, or ``[B, n_eff, n_orders]`` for harmonic
        specs. Full per-phase / per-order detail is always in ``operating_point`` /
        ``harmonic_injection``. An ``h_mag`` spec additionally records what it actually
        injected — ``<name>_mag`` / ``<name>_phase`` ``[B, n_dev, n_orders]`` (per unit
        of the device's own fundamental current / degrees) and ``<name>_device_ids``
        ``[n_dev]`` — since the raw draw is only a fraction of a per-device emission
        reference.
    n_samples:
        Batch size ``B``.
    config:
        The originating config (provenance / reproducibility).
    node_sources:
        Realized :class:`~pgml.solver.NodeHarmonicSource` entries — pass to
        ``solve_harmonic_flow(node_sources=...)``. Holds the upstream background of
        :class:`~pgml.scenarios.BackgroundHarmonicConfig` when one is configured, with
        ``[B]`` / ``[B, T]`` magnitude and phase tensors; empty otherwise.
    perturbations:
        Ground-truth :class:`~pgml.schemas.scenario_schema.ParameterPerturbation` rows
        for a :func:`~pgml.scenarios.perturbation.perturbation_sweep` (which scenario
        perturbed which component, nominal vs perturbed value); empty otherwise.
    """

    operating_point: dict
    samples: dict
    n_samples: int
    config: (
        "ScenarioConfig | CartesianConfig | CoherentSpectrumConfig | Perturbation | "
        "SpectrumSweepConfig | NodeInjectionSweepConfig"
    )
    harmonic_injection: dict = field(default_factory=dict)
    node_sources: list = field(default_factory=list)
    perturbations: list = field(default_factory=list)


class _Nominal(NamedTuple):
    """Nominal nameplate power of one Load/Generator (totals + per-phase split).

    Entries keep the schema's float/tensor duality: a tensor-valued nameplate
    (``p_nom_w`` as an autograd leaf) passes through UNTOUCHED so a
    ``mode="scale"`` operating point stays differentiable w.r.t. the grid's own
    rated power.
    """

    p_total: object  # float or 0-d array-like
    q_total: object
    p_pp: list  # per-phase active nominal (nameplate split or total / n)
    q_pp: list  # per-phase reactive nominal
    n: int  # phase count


def _unit_samples(b: int, d: int, method: str, seed: int) -> Tensor:
    """``[b, d]`` quasi-/pseudo-random samples in ``[0, 1)`` (float64, deterministic)."""
    if method == "sobol":
        eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
        return eng.draw(b, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    if method == "independent":
        return torch.rand((b, d), generator=gen, dtype=torch.float64)
    if method == "lhs":
        # Latin hypercube: per dim a random permutation of the b strata + jitter.
        u = torch.empty((b, d), dtype=torch.float64)
        strata = torch.arange(b, dtype=torch.float64)
        for j in range(d):
            perm = torch.randperm(b, generator=gen)
            jitter = torch.rand(b, generator=gen, dtype=torch.float64)
            u[:, j] = (strata[perm] + jitter) / b
        return u
    raise InputError(f"Unknown sampling method {method!r}.")


def _norm_icdf(u: Tensor) -> Tensor:
    """Standard-normal quantile ``Phi^{-1}(u)`` (clamped off {0,1})."""
    u = u.clamp(_U_EPS, 1.0 - _U_EPS)
    return _SQRT2 * torch.erfinv(2.0 * u - 1.0)


def _norm_cdf(z: Tensor) -> Tensor:
    """Standard-normal CDF ``Phi(z)``."""
    return 0.5 * (1.0 + torch.erf(z / _SQRT2))


class _SpecLayout(NamedTuple):
    spec: ParameterSpec
    ids: list  # matched component ids
    nph: list  # phase count per matched component
    base_off: int  # first unit-cube column of the base block
    base_dim: int  # #base columns
    phase_off: int  # first unit-cube column of the per-phase block
    phase_dim: int  # #per-phase columns


class _HarmLayout(NamedTuple):
    spec: ParameterSpec
    ids: list  # matched component ids
    off: int  # first unit-cube column
    dim: int  # #columns (= n_eff * n_orders)
    n_eff: int  # #independent component draws (n_comp for each, 1 for shared)


def _spec_dims(spec: ParameterSpec, n_comp: int, nph: list) -> tuple[int, int]:
    """``(base_dim, phase_dim)`` unit-cube columns this spec consumes."""
    correlated = spec.correlation is not None
    per_each = correlated or spec.per == "each"
    if spec.symmetry == "balanced":
        return (n_comp if per_each else 1), 0
    if spec.symmetry == "small_imbalance":
        # base (component-level, like balanced) + one perturbation per phase per comp.
        return (n_comp if per_each else 1), sum(nph)
    # independent: no base; one independent draw per phase (per comp if per="each").
    return 0, (sum(nph) if spec.per == "each" else nph[0])


def _reject_overlapping_writers(writers) -> None:
    """Raise when two specs/axes vary the same field of the same component.

    ``writers`` yields ``(spec_name, component_id, field_key)`` triples. Without
    this guard the last writer would silently win in ``operating_point`` while
    BOTH values stay recorded in ``samples`` — the persisted record would no
    longer match the operating point that was actually solved.
    """
    seen: dict[tuple, str] = {}
    for name, cid, key in writers:
        k = (cid, key)
        if k in seen:
            raise InputError(
                f"Parameters {seen[k]!r} and {name!r} both vary {key!r} of "
                f"component {cid}; overlapping writers on one field are "
                "ambiguous (last-writer-wins would desync the recorded samples "
                "from the realized operating point). Narrow the selectors."
            )
        seen[k] = name


def _resolve(grid: Grid, config: ScenarioConfig):
    """``(factor_index, op_layouts, harm_layouts, total_dim)`` for the unit-cube layout.

    Columns: one per declared factor, then per power spec a base + per-phase block, then
    per harmonic spec a ``n_eff * n_orders`` block. Power and harmonic specs share one
    QMC cube (better space-filling across power and spectrum together).
    """
    if not config.parameters:
        raise InputError("ScenarioConfig has no parameters / sampling dimensions.")

    # Load/Generator/Storage carry a nominal P/Q; a Source is selectable too (u_ref).
    by_id = {
        a.id: a for a in grid.appliances if isinstance(a, (InjectionAppliance, Source))
    }
    declared = {f.name for f in config.factors}

    dim = 0
    factor_index: dict[str, int] = {}
    for f in config.factors:
        factor_index[f.name] = dim
        dim += 1

    op_layouts: list[_SpecLayout] = []
    harm_layouts: list[_HarmLayout] = []
    for spec in config.parameters:
        ids = spec.selector.resolve(grid)
        if not ids:
            raise InputError(
                f"Parameter {spec.name!r} selector matched no in-service components."
            )
        if spec.is_harmonic:
            n_eff = len(ids) if spec.per == "each" else 1
            block = n_eff * len(spec.orders)
            harm_layouts.append(_HarmLayout(spec, ids, dim, block, n_eff))
            dim += block
            continue
        if spec.correlation is not None and spec.correlation.factor not in declared:
            raise InputError(
                f"Parameter {spec.name!r} correlation references undeclared factor "
                f"{spec.correlation.factor!r}; add it to ScenarioConfig.factors."
            )
        nph = [len(by_id[i].phases) for i in ids]
        if spec.symmetry == "independent" and len(set(nph)) > 1:
            raise InputError(
                f"Parameter {spec.name!r} symmetry='independent' matches components "
                f"with differing phase counts {sorted(set(nph))}; split into one spec "
                "per phase count."
            )
        base_dim, phase_dim = _spec_dims(spec, len(ids), nph)
        op_layouts.append(
            _SpecLayout(spec, ids, nph, dim, base_dim, dim + base_dim, phase_dim)
        )
        dim += base_dim + phase_dim

    writers = [
        (lay.spec.name, cid, fld)
        for lay in op_layouts
        for cid in lay.ids
        for fld in (("p", "q") if lay.spec.field == "pq" else (lay.spec.field,))
    ]
    writers += [
        (lay.spec.name, cid, (lay.spec.field, order))
        for lay in harm_layouts
        for cid in lay.ids
        for order in lay.spec.orders
    ]
    _reject_overlapping_writers(writers)

    return factor_index, op_layouts, harm_layouts, dim


def _draw(spec: ParameterSpec, u: Tensor, factor_z: dict) -> Tensor:
    """Map unit-cube columns through the spec's marginal, mixing in its latent factor.

    Without ``correlation`` this is the plain inverse CDF. With it, the columns become the
    IDIOSYNCRATIC part of a single-factor Gaussian copula
    (``Z = sqrt(rho)*Z_factor + sqrt(1-rho)*eps``), so every matched draw keeps the spec's
    marginal while co-moving with the factor. Shape-agnostic in ``u``: the factor is
    broadcast over whatever axes follow the batch, which is what lets a per-COMPONENT draw
    and a per-PHASE draw share one implementation.
    """
    if spec.correlation is None:
        return spec.distribution.icdf(u)
    zf = factor_z[spec.correlation.factor]  # [B]
    while zf.dim() < u.dim():
        zf = zf.unsqueeze(-1)
    rho = spec.correlation.rho
    z = math.sqrt(rho) * zf + math.sqrt(1.0 - rho) * _norm_icdf(u)
    return spec.distribution.icdf(_norm_cdf(z))


def _component_base(
    spec: ParameterSpec, base_u: Tensor, factor_z: dict, n_comp: int
) -> tuple[Tensor, Tensor]:
    """Component-level values: ``(write [B, n_comp], record [B, n_comp] or [B, 1])``."""
    dist = spec.distribution
    if spec.correlation is not None:
        vals = _draw(spec, base_u, factor_z)  # [B, n_comp], marginal preserved
        return vals, vals
    if spec.per == "shared":
        v = dist.icdf(base_u[:, 0]).unsqueeze(-1)  # [B, 1]
        return v.expand(-1, n_comp), v  # write broadcasts, record is the one draw
    vals = dist.icdf(base_u)  # [B, n_comp]
    return vals, vals


def _scalar_passthrough(x):
    """A python float for plain numbers; array-likes (tensors) pass UNTOUCHED.

    ``float(tensor)`` would silently detach an autograd leaf — the exact
    gradient break the schema's float/tensor duality exists to prevent.
    """
    return x if hasattr(x, "detach") or hasattr(x, "__array__") else float(x)


def _nominal(grid: Grid) -> dict:
    """``{id: _Nominal}`` per in-service injection appliance (totals + per-phase)."""
    out: dict[int, _Nominal] = {}
    for a in grid.appliances:
        if not isinstance(a, InjectionAppliance):
            continue
        n = len(a.phases)
        p_total = _scalar_passthrough(a.p_nom_w)
        q_total = _scalar_passthrough(a.q_nom_var)
        # Fresh `/ n` per slot: a `[x] * n` literal would alias one autograd
        # node into every phase (see assembly._params.resolve_operating_power).
        p_pp = (
            [_scalar_passthrough(x) for x in a.p_nom_per_phase_w]
            if a.p_nom_per_phase_w is not None
            else [p_total / n for _ in range(n)]
        )
        q_pp = (
            [_scalar_passthrough(x) for x in a.q_nom_per_phase_var]
            if a.q_nom_per_phase_var is not None
            else [q_total / n for _ in range(n)]
        )
        out[a.id] = _Nominal(p_total, q_total, p_pp, q_pp, n)
    return out


def _apply(
    entry: dict, spec: ParameterSpec, col: Tensor, p_nom: float, q_nom: float
) -> None:
    """Write a sampled per-scenario TOTAL column ``[B]`` (balanced / cartesian path)."""
    if spec.field in ("p", "pq"):
        entry["p_w"] = col * p_nom if spec.mode == "scale" else col
    if spec.field in ("q", "pq"):
        entry["q_var"] = col * q_nom if spec.mode == "scale" else col


def _write_per_phase(entry: dict, field_: str, p_pp: list, q_pp: list) -> None:
    """Write per-phase override lists, only for the fields the spec varies."""
    if field_ in ("p", "pq"):
        entry["p_per_phase_w"] = p_pp
    if field_ in ("q", "pq"):
        entry["q_per_phase_var"] = q_pp


def _stored_spectrum(appliance) -> dict:
    """``{order: [mag_pu, phase_deg]}`` from a device's stored ``StaticSpectrum``."""
    spec = getattr(appliance, "spectrum", None)
    if isinstance(spec, StaticSpectrum):
        return {
            c.order: [c.magnitude_pu, c.phase_deg] for c in spec.spectrum.components
        }
    return {}


def _coefficient_column(value, b: int, like: Tensor) -> Tensor:
    """One realized injection coefficient as a ``[B]`` real column.

    A coefficient is either a per-scenario ``[B]`` tensor (a spec drew it) or a scalar
    carried over from the device's stored spectrum (a float, or a 0-d array-like under
    the schema's float/tensor duality); the scalar broadcasts across the batch. ``like``
    supplies dtype + device, and the cast keeps the value on the autograd tape.
    """
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    tensor = tensor.to(dtype=like.dtype, device=like.device)
    return tensor.expand(b) if tensor.ndim == 0 else tensor


def _record_realized_injections(
    built: dict, harm_layouts: list, u: Tensor, samples: dict
) -> None:
    """Record the REALIZED per-device injection of every ``h_mag`` spec into ``samples``.

    ``samples[<spec.name>]`` holds the raw draw, which for a referenced spec is a
    FRACTION of a per-order emission limit — the magnitude that reaches the solver only
    exists once that per-device reference has been applied. These columns hold what was
    actually injected, so a persisted dataset is auditable without re-deriving the
    reference table it was generated against:

    - ``"<spec>_mag"`` ``[B, n_dev, n_ord]`` — magnitude in per unit of the device's own
      fundamental current (post-reference, the value the solver scales ``|I_1|`` by);
    - ``"<spec>_phase"`` ``[B, n_dev, n_ord]`` — the phase in degrees that goes with it
      (an ``h_phase`` spec's draw where one covers the device and order, otherwise the
      angle seeded from the device's stored spectrum);
    - ``"<spec>_device_ids"`` ``[n_dev]`` — the device ids of the middle axis.

    The device axis is the spec's full matched set even for ``per="shared"``: one shared
    draw still realizes as a different magnitude per device, because the emission
    reference is per device.
    """
    b = u.shape[0]
    for lay in harm_layouts:
        spec = lay.spec
        if spec.field != "h_mag":
            continue
        mags, phases = [], []
        for cid in lay.ids:
            dev = built[cid]
            mags.append(
                torch.stack(
                    [_coefficient_column(dev[o][0], b, u) for o in spec.orders], dim=-1
                )
            )
            phases.append(
                torch.stack(
                    [_coefficient_column(dev[o][1], b, u) for o in spec.orders], dim=-1
                )
            )
        samples[f"{spec.name}_mag"] = torch.stack(mags, dim=1)  # [B, n_dev, n_ord]
        samples[f"{spec.name}_phase"] = torch.stack(phases, dim=1)
        samples[f"{spec.name}_device_ids"] = torch.tensor(lay.ids, dtype=torch.long)


def _harmonic_injections(
    grid: Grid, harm_layouts: list, u: Tensor, samples: dict
) -> dict:
    """Build ``{id: {order: (mag[B], phase[B])}}`` from the harmonic specs.

    Each device's injection is seeded from its stored ``StaticSpectrum`` (so orders the
    config does not vary survive), then ``h_mag`` / ``h_phase`` specs overwrite their
    orders. ``h_mag`` magnitude is the sampled value times a per-order reference limit
    (``harmonic_reference="iec61000-3-2"`` -- a PER-DEVICE IEC 61000-3-2 current-emission
    fraction; ``"en50160"`` -- the per-order DIN EN 50160 voltage-compatibility level),
    the stored magnitude (``mode="scale"``), or the sampled value directly (absolute pu).

    ``samples`` records both the raw draw (``<spec.name>``) and the realized
    post-reference injection (:func:`_record_realized_injections`).
    """
    by_id = {a.id: a for a in grid.appliances if isinstance(a, InjectionAppliance)}
    # building store: {id: {order: [mag, phase]}}, seeded from stored spectra.
    built: dict[int, dict] = {}
    # pristine stored spectra (never mutated; `mode="scale"` references these).
    stored_cache: dict[int, dict] = {}

    def _stored(cid: int) -> dict:
        if cid not in stored_cache:
            stored_cache[cid] = _stored_spectrum(by_id[cid])
        return stored_cache[cid]

    def _dev(cid: int) -> dict:
        if cid not in built:
            built[cid] = {o: list(mp) for o, mp in _stored(cid).items()}
        return built[cid]

    for lay in harm_layouts:
        spec = lay.spec
        n_orders = len(spec.orders)
        block = u[:, lay.off : lay.off + lay.dim]  # [B, n_eff * n_orders]
        vals = spec.distribution.icdf(block).reshape(-1, lay.n_eff, n_orders)
        samples[spec.name] = vals  # [B, n_eff, n_orders]
        # IEC 61000-3-2 caps are PER DEVICE (from nominal P + node voltage); build once
        # per spec. EN 50160 caps are global per-order (looked up inline below).
        iec_caps = (
            iec61000_3_2_device_caps(
                grid, lay.ids, spec.orders, emission_class=spec.emission_class
            )
            if spec.harmonic_reference == "iec61000-3-2"
            else {}
        )
        for j, cid in enumerate(lay.ids):
            comp = vals[:, j if spec.per == "each" else 0, :]  # [B, n_orders]
            dev, stored = _dev(cid), _stored(cid)
            for o, order in enumerate(spec.orders):
                v = comp[:, o]  # [B]
                slot = dev.setdefault(order, [0.0, 0.0])
                if spec.field == "h_phase":
                    slot[1] = v
                elif spec.harmonic_reference == "iec61000-3-2":
                    slot[0] = v * iec_caps[cid][order]
                elif spec.harmonic_reference == "en50160":
                    slot[0] = v * en50160_limit(order)
                elif spec.mode == "scale":
                    if order not in stored:
                        raise InputError(
                            f"Parameter {spec.name!r} h_mag mode='scale' for order "
                            f"{order} on device {cid}, which has no stored spectrum "
                            "magnitude; use mode='absolute' or harmonic_reference."
                        )
                    slot[0] = v * stored[order][0]
                else:
                    slot[0] = v

    # After every spec has written: the realized (post-reference) magnitude and the
    # phase it pairs with, per device and order.
    _record_realized_injections(built, harm_layouts, u, samples)
    return {cid: {o: tuple(mp) for o, mp in d.items()} for cid, d in built.items()}


def sample(grid: Grid, config: ScenarioConfig) -> SampledScenarios:
    """Sample ``config.n_samples`` realized operating points from ``grid`` (reproducible)."""
    factor_index, op_layouts, harm_layouts, dim = _resolve(grid, config)
    b = config.n_samples
    u = _unit_samples(b, dim, config.method, config.seed)  # [B, D] in [0,1)

    factor_z = {name: _norm_icdf(u[:, idx]) for name, idx in factor_index.items()}
    nominal = _nominal(grid)
    operating_point: dict = {}
    samples: dict = {}

    for lay in op_layouts:
        spec = lay.spec
        scale = spec.mode == "scale"
        base_u = u[:, lay.base_off : lay.base_off + lay.base_dim]
        phase_u = u[:, lay.phase_off : lay.phase_off + lay.phase_dim]

        if spec.symmetry == "independent":
            records = []  # per comp [B, nph]
            pcol = 0
            for j, cid in enumerate(lay.ids):
                rec, nph = nominal[cid], lay.nph[j]
                pp_p, pp_q, comp_rec = [], [], []
                for ph in range(nph):
                    col = phase_u[:, pcol] if spec.per == "each" else phase_u[:, ph]
                    pcol += 1 if spec.per == "each" else 0
                    # per-PHASE draw, correlated through the spec's factor when it has one:
                    # independent symmetry has no component-level base to couple instead.
                    v = _draw(spec, col, factor_z)  # [B]
                    comp_rec.append(v)
                    pp_p.append(v * rec.p_pp[ph] if scale else v)
                    pp_q.append(v * rec.q_pp[ph] if scale else v)
                _write_per_phase(
                    operating_point.setdefault(cid, {}), spec.field, pp_p, pp_q
                )
                records.append(torch.stack(comp_rec, dim=-1))  # [B, nph]
            samples[spec.name] = torch.stack(records, dim=1)  # [B, n_comp, nph]
            continue

        write_vals, record_vals = _component_base(spec, base_u, factor_z, len(lay.ids))
        samples[spec.name] = record_vals

        if spec.symmetry == "balanced":
            for j, cid in enumerate(lay.ids):
                entry = operating_point.setdefault(cid, {})
                if spec.is_source_voltage:
                    # The slack-voltage scale is the sampled value itself (mode='scale'):
                    # the ideal-slack solve multiplies it onto the Source's u_ref_v.
                    entry["u_ref_scale"] = write_vals[:, j]
                    continue
                rec = nominal[cid]
                _apply(entry, spec, write_vals[:, j], rec.p_total, rec.q_total)
        else:  # small_imbalance: balanced base * (1 + small per-phase perturbation)
            pcol = 0
            for j, cid in enumerate(lay.ids):
                rec, nph = nominal[cid], lay.nph[j]
                s = write_vals[:, j]  # [B]
                pp_p, pp_q = [], []
                for ph in range(nph):
                    delta = spec.imbalance * _norm_icdf(phase_u[:, pcol])  # [B]
                    pcol += 1
                    scale_ph = s * (1.0 + delta)
                    pp_p.append(scale_ph * rec.p_pp[ph] if scale else scale_ph / nph)
                    pp_q.append(scale_ph * rec.q_pp[ph] if scale else scale_ph / nph)
                _write_per_phase(
                    operating_point.setdefault(cid, {}), spec.field, pp_p, pp_q
                )

    harmonic_injection = (
        _harmonic_injections(grid, harm_layouts, u, samples) if harm_layouts else {}
    )
    return SampledScenarios(
        operating_point=operating_point,
        samples=samples,
        n_samples=b,
        config=config,
        harmonic_injection=harmonic_injection,
    )


def cartesian_sample(grid: Grid, config: CartesianConfig) -> SampledScenarios:
    """Cartesian product of axis levels (pgm-style grid sweep; deterministic).

    ``B = prod(len(axis.values))``. Each axis applies its level to ALL components its
    selector matches (list multiple axes with id selectors for per-component sweeps).
    """
    resolved = [(ax, ax.selector.resolve(grid)) for ax in config.axes]
    for ax, ids in resolved:
        if not ids:
            raise InputError(
                f"Cartesian axis {ax.name!r} matched no in-service components."
            )
    _reject_overlapping_writers(
        (ax.name, cid, fld)
        for ax, ids in resolved
        for cid in ids
        for fld in (("p", "q") if ax.field == "pq" else (ax.field,))
    )
    levels = [torch.tensor(ax.values, dtype=torch.float64) for ax, _ in resolved]
    combos = torch.cartesian_prod(*levels)  # [B, n_axes] (or [B] for a single axis)
    if combos.ndim == 1:
        combos = combos.unsqueeze(-1)

    nominal = _nominal(grid)
    operating_point: dict = {}
    samples: dict = {}
    for i, (ax, ids) in enumerate(resolved):
        col = combos[:, i]  # [B]
        samples[ax.name] = col
        for cid in ids:
            rec = nominal[cid]
            _apply(
                operating_point.setdefault(cid, {}), ax, col, rec.p_total, rec.q_total
            )

    return SampledScenarios(
        operating_point=operating_point,
        samples=samples,
        n_samples=combos.shape[0],
        config=config,
    )


__all__ = ["SampledScenarios", "sample", "cartesian_sample"]
