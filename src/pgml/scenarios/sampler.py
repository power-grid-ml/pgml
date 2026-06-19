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

from pgml.schemas.grid_schema import Generator, Grid, Load, StaticSpectrum

from .config import (
    CartesianConfig,
    CoherentSpectrumConfig,
    ParameterSpec,
    ScenarioConfig,
)
from .en50160 import en50160_limit

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
        ``harmonic_injection``.
    n_samples:
        Batch size ``B``.
    config:
        The originating :class:`ScenarioConfig` (provenance / reproducibility).
    """

    operating_point: dict
    samples: dict
    n_samples: int
    config: "ScenarioConfig | CartesianConfig | CoherentSpectrumConfig"
    harmonic_injection: dict = field(default_factory=dict)


class _Nominal(NamedTuple):
    """Nominal nameplate power of one Load/Generator (totals + per-phase split)."""

    p_total: float
    q_total: float
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
    raise ValueError(f"Unknown sampling method {method!r}.")


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


def _resolve(grid: Grid, config: ScenarioConfig):
    """``(factor_index, op_layouts, harm_layouts, total_dim)`` for the unit-cube layout.

    Columns: one per declared factor, then per power spec a base + per-phase block, then
    per harmonic spec a ``n_eff * n_orders`` block. Power and harmonic specs share one
    QMC cube (better space-filling across power and spectrum together).
    """
    if not config.parameters:
        raise ValueError("ScenarioConfig has no parameters / sampling dimensions.")

    by_id = {a.id: a for a in grid.appliances if isinstance(a, (Load, Generator))}
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
            raise ValueError(
                f"Parameter {spec.name!r} selector matched no in-service components."
            )
        if spec.is_harmonic:
            n_eff = len(ids) if spec.per == "each" else 1
            block = n_eff * len(spec.orders)
            harm_layouts.append(_HarmLayout(spec, ids, dim, block, n_eff))
            dim += block
            continue
        if spec.correlation is not None and spec.correlation.factor not in declared:
            raise ValueError(
                f"Parameter {spec.name!r} correlation references undeclared factor "
                f"{spec.correlation.factor!r}; add it to ScenarioConfig.factors."
            )
        nph = [len(by_id[i].phases) for i in ids]
        if spec.symmetry == "independent" and len(set(nph)) > 1:
            raise ValueError(
                f"Parameter {spec.name!r} symmetry='independent' matches components "
                f"with differing phase counts {sorted(set(nph))}; split into one spec "
                "per phase count."
            )
        base_dim, phase_dim = _spec_dims(spec, len(ids), nph)
        op_layouts.append(
            _SpecLayout(spec, ids, nph, dim, base_dim, dim + base_dim, phase_dim)
        )
        dim += base_dim + phase_dim

    return factor_index, op_layouts, harm_layouts, dim


def _component_base(
    spec: ParameterSpec, base_u: Tensor, factor_z: dict, n_comp: int
) -> tuple[Tensor, Tensor]:
    """Component-level values: ``(write [B, n_comp], record [B, n_comp] or [B, 1])``."""
    dist = spec.distribution
    if spec.correlation is not None:
        zf = factor_z[spec.correlation.factor].unsqueeze(-1)  # [B, 1]
        rho = spec.correlation.rho
        eps = _norm_icdf(base_u)  # [B, n_comp]
        z = math.sqrt(rho) * zf + math.sqrt(1.0 - rho) * eps
        vals = dist.icdf(_norm_cdf(z))  # [B, n_comp], marginal preserved
        return vals, vals
    if spec.per == "shared":
        v = dist.icdf(base_u[:, 0]).unsqueeze(-1)  # [B, 1]
        return v.expand(-1, n_comp), v  # write broadcasts, record is the one draw
    vals = dist.icdf(base_u)  # [B, n_comp]
    return vals, vals


def _nominal(grid: Grid) -> dict:
    """``{id: _Nominal}`` for every in-service Load/Generator (totals + per-phase)."""
    out: dict[int, _Nominal] = {}
    for a in grid.appliances:
        if not isinstance(a, (Load, Generator)):
            continue
        n = len(a.phases)
        p_total, q_total = float(a.p_nom_w), float(a.q_nom_var)
        p_pp = (
            [float(x) for x in a.p_nom_per_phase_w]
            if a.p_nom_per_phase_w is not None
            else [p_total / n] * n
        )
        q_pp = (
            [float(x) for x in a.q_nom_per_phase_var]
            if a.q_nom_per_phase_var is not None
            else [q_total / n] * n
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


def _harmonic_injections(
    grid: Grid, harm_layouts: list, u: Tensor, samples: dict
) -> dict:
    """Build ``{id: {order: (mag[B], phase[B])}}`` from the harmonic specs.

    Each device's injection is seeded from its stored ``StaticSpectrum`` (so orders the
    config does not vary survive), then ``h_mag`` / ``h_phase`` specs overwrite their
    orders. ``h_mag`` magnitude is the sampled value times the per-order EN 50160 limit
    (``harmonic_reference="en50160"``), the stored magnitude (``mode="scale"``), or the
    sampled value directly (absolute pu).
    """
    by_id = {a.id: a for a in grid.appliances if isinstance(a, (Load, Generator))}
    # building store: {id: {order: [mag, phase]}}, seeded from stored spectra.
    built: dict[int, dict] = {}

    def _dev(cid: int) -> dict:
        if cid not in built:
            built[cid] = _stored_spectrum(by_id[cid])
        return built[cid]

    for lay in harm_layouts:
        spec = lay.spec
        n_orders = len(spec.orders)
        block = u[:, lay.off : lay.off + lay.dim]  # [B, n_eff * n_orders]
        vals = spec.distribution.icdf(block).reshape(-1, lay.n_eff, n_orders)
        samples[spec.name] = vals  # [B, n_eff, n_orders]
        for j, cid in enumerate(lay.ids):
            comp = vals[:, j if spec.per == "each" else 0, :]  # [B, n_orders]
            dev, stored = _dev(cid), _stored_spectrum(by_id[cid])
            for o, order in enumerate(spec.orders):
                v = comp[:, o]  # [B]
                slot = dev.setdefault(order, [0.0, 0.0])
                if spec.field == "h_phase":
                    slot[1] = v
                elif spec.harmonic_reference == "en50160":
                    slot[0] = v * en50160_limit(order)
                elif spec.mode == "scale":
                    if order not in stored:
                        raise ValueError(
                            f"Parameter {spec.name!r} h_mag mode='scale' for order "
                            f"{order} on device {cid}, which has no stored spectrum "
                            "magnitude; use mode='absolute' or harmonic_reference."
                        )
                    slot[0] = v * stored[order][0]
                else:
                    slot[0] = v

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
                    v = spec.distribution.icdf(col)  # [B]
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
                rec = nominal[cid]
                _apply(
                    operating_point.setdefault(cid, {}),
                    spec,
                    write_vals[:, j],
                    rec.p_total,
                    rec.q_total,
                )
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
            raise ValueError(
                f"Cartesian axis {ax.name!r} matched no in-service components."
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
