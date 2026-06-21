"""Persist a :class:`~pgml.scenarios.run.ScenarioResult` to parquet (training data).

Two interchangeable on-disk LAYOUTS for the node voltages, written to a dataset
DIRECTORY alongside a self-describing sidecar:

- ``layout="long"`` — a tidy table, one row per ``(scenario, step, frequency, node-phase)``
  with ``v_re`` / ``v_im`` (the ``result_schema`` phasor convention). Canonical, joins
  cleanly by id, great for DuckDB/polars analysis; larger on disk.
- ``layout="wide"`` — one row per ``(scenario, step)`` with ``v_re`` / ``v_im`` flattened
  over ``[H*N]`` as array columns. Compact and fast to reload straight into a tensor —
  the training-loop cache.

Both layouts carry the SAME ``meta.json`` (the serialized config + seed + node-phase
index + frequencies + dims — reproducibility is paramount) and the SAME
``samples.parquet`` (the realized sampled inputs), so a dataset is fully self-describing
and ``read_dataset`` returns an identical tensor regardless of layout.

This is result I/O, not the differentiable core: tensors are detached and moved to CPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
from torch import Tensor

from ..errors import InputError
from . import config as _cfg
from .run import ScenarioResult

_VOLTAGES = "voltages.parquet"
_SAMPLES = "samples.parquet"
_META = "meta.json"

# config types we can reconstruct on read (by class name).
_CONFIG_TYPES = {
    "ScenarioConfig": _cfg.ScenarioConfig,
    "CartesianConfig": _cfg.CartesianConfig,
    "CoherentSpectrumConfig": _cfg.CoherentSpectrumConfig,
    "Perturbation": _cfg.Perturbation,
    "SpectrumSweepConfig": _cfg.SpectrumSweepConfig,
    "NodeInjectionSweepConfig": _cfg.NodeInjectionSweepConfig,
}


@dataclass(frozen=True)
class LoadedDataset:
    """A dataset reloaded by :func:`read_dataset`.

    Attributes
    ----------
    v:
        Complex node voltages, reshaped to the ORIGINAL ``ScenarioResult.v`` shape
        (``[B, N]`` / ``[B, H, N]`` / ``[B, T, H, N]``).
    samples:
        The realized sampled inputs ``{name: Tensor}`` (per-scenario records restored to
        their original shapes; shared records from the sidecar).
    frequencies_hz:
        ``[H]`` real tensor, or ``None`` for a power-flow dataset.
    node_ids, phase_codes:
        int64 ``[N]`` row layout of ``v`` (mirrors :class:`NodePhaseIndex`).
    config:
        The reconstructed config object (``ScenarioConfig`` / … / ``Perturbation``) when
        its type is known, else the raw config dict.
    perturbations:
        Ground-truth ``ParameterPerturbation`` rows as dicts (empty if none).
    meta:
        The full sidecar dict (config_json, seed, dims, layout, …).
    """

    v: Tensor
    samples: dict
    frequencies_hz: Optional[Tensor]
    node_ids: Tensor
    phase_codes: Tensor
    config: object
    perturbations: list
    meta: dict


def _np(t: Tensor, dtype=np.float64) -> np.ndarray:
    """Detached CPU numpy view (result I/O, not the differentiable path)."""
    return t.detach().to("cpu").numpy().astype(dtype, copy=False)


def _canonical(v: Tensor) -> tuple[Tensor, int, int, int, int]:
    """Reshape ``v`` to ``[B, T, H, N]`` (inserting singleton T/H by ndim)."""
    if v.ndim == 2:  # [B, N] power flow
        b, n = v.shape
        return v.reshape(b, 1, 1, n), b, 1, 1
    if v.ndim == 3:  # [B, H, N] harmonic
        b, h, n = v.shape
        return v.reshape(b, 1, h, n), b, 1, h
    if v.ndim == 4:  # [B, T, H, N] coherent
        b, t, h, n = v.shape
        return v, b, t, h
    raise InputError(f"ScenarioResult.v must be 2-4 dims, got {v.ndim}.")


def _long_dataframe(vre, vim, dims, node_ids, phase_codes, freqs) -> "pl.DataFrame":
    """The tidy table: one row per (scenario, step, frequency, node-phase)."""
    b, t, h, n = dims
    n_rows = b * t * h * n
    scenario = np.repeat(np.arange(b), t * h * n)
    step = np.tile(np.repeat(np.arange(t), h * n), b)
    freq_idx = np.tile(np.repeat(np.arange(h), n), b * t)
    row = np.tile(np.arange(n), b * t * h)
    freq_hz = freqs[freq_idx] if freqs is not None else np.full(n_rows, np.nan)
    return pl.DataFrame(
        {
            "scenario": scenario.astype(np.int64),
            "step": step.astype(np.int64),
            "freq_idx": freq_idx.astype(np.int64),
            "frequency_hz": freq_hz.astype(np.float64),
            "node_id": node_ids[row].astype(np.int64),
            "phase_code": phase_codes[row].astype(np.int64),
            "row": row.astype(np.int64),
            "v_re": vre.reshape(-1).astype(np.float64),
            "v_im": vim.reshape(-1).astype(np.float64),
        }
    )


def _write_long(path: Path, vre, vim, dims, node_ids, phase_codes, freqs, comp) -> None:
    _long_dataframe(vre, vim, dims, node_ids, phase_codes, freqs).write_parquet(
        path / _VOLTAGES, compression=comp
    )


def _write_wide(path: Path, vre, vim, dims, comp) -> None:
    b, t, h, n = dims
    scenario = np.repeat(np.arange(b), t).astype(np.int64)
    step = np.tile(np.arange(t), b).astype(np.int64)
    re2d = vre.reshape(b * t, h * n)  # one flattened [H*N] vector per (scenario, step)
    im2d = vim.reshape(b * t, h * n)
    pl.DataFrame(
        {
            "scenario": scenario,
            "step": step,
            "v_re": _array_col(re2d),
            "v_im": _array_col(im2d),
        }
    ).write_parquet(path / _VOLTAGES, compression=comp)


def _array_col(arr2d: np.ndarray) -> pl.Series:
    """A fixed-size-list (``Array``) polars column from a 2-D numpy array (int/float)."""
    if np.issubdtype(arr2d.dtype, np.integer):
        return pl.Series(
            values=arr2d.astype(np.int64), dtype=pl.Array(pl.Int64, arr2d.shape[1])
        )
    return pl.Series(
        values=arr2d.astype(np.float64), dtype=pl.Array(pl.Float64, arr2d.shape[1])
    )


def _write_samples(samples: dict, b: int, path: Path, comp: str) -> dict:
    """Write per-scenario (B-leading) samples; return shapes/dtypes + shared (non-B)."""
    cols: dict = {"scenario": np.arange(b, dtype=np.int64)}
    shapes: dict = {}
    dtypes: dict = {}
    shared: dict = {}
    for name, t in samples.items():
        arr = t.detach().to("cpu").numpy()  # NATURAL dtype (preserve int vs float)
        if arr.ndim >= 1 and arr.shape[0] == b:
            flat = arr.reshape(b, -1)
            cols[name] = _array_col(flat) if flat.shape[1] > 1 else flat[:, 0]
            shapes[name] = list(arr.shape[1:])  # trailing shape per scenario
            dtypes[name] = str(t.dtype)
        else:
            shared[name] = {"data": arr.tolist(), "dtype": str(t.dtype)}
    if len(cols) > 1:
        pl.DataFrame(cols).write_parquet(path / _SAMPLES, compression=comp)
    return {"sample_shapes": shapes, "sample_dtypes": dtypes, "shared_samples": shared}


def write_dataset(
    result: ScenarioResult,
    path,
    *,
    layout: str = "wide",
    compression: str = "zstd",
    also_csv: bool = False,
) -> Path:
    """Write a :class:`ScenarioResult` to a parquet dataset directory.

    Parameters
    ----------
    result:
        The batched result to persist.
    path:
        Target dataset DIRECTORY (created if missing).
    layout:
        ``"wide"`` (compact tensor cache, default) or ``"long"`` (tidy table).
    compression:
        Parquet codec (default ``"zstd"``).
    also_csv:
        Additionally write the tidy long voltages to ``voltages.csv`` (regardless of
        ``layout``) for manual inspection — not the primary format, just a convenience.

    Returns
    -------
    Path
        The dataset directory.
    """
    if layout not in ("wide", "long"):
        raise InputError(f"layout must be 'wide' or 'long', got {layout!r}.")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    v4, b, t, h = _canonical(result.v)
    n = v4.shape[-1]
    vre, vim = _np(v4.real), _np(v4.imag)
    node_ids = _np(result.index.node_ids, np.int64)
    phase_codes = _np(result.index.phase_codes, np.int64)
    freqs = _np(result.frequencies_hz) if result.frequencies_hz is not None else None

    if layout == "long":
        _write_long(
            path, vre, vim, (b, t, h, n), node_ids, phase_codes, freqs, compression
        )
    else:
        _write_wide(path, vre, vim, (b, t, h, n), compression)
    if also_csv:
        _long_dataframe(vre, vim, (b, t, h, n), node_ids, phase_codes, freqs).write_csv(
            path / "voltages.csv"
        )

    sampled = result.sampled
    sample_meta = _write_samples(sampled.samples, b, path, compression)

    cfg = sampled.config
    meta = {
        "layout": layout,
        "calculation": "harmonic"
        if result.frequencies_hz is not None
        else "power_flow",
        "v_shape": list(result.v.shape),
        "v_dtype": str(result.v.dtype),
        "dims": {"B": b, "T": t, "H": h, "N": n},
        "n_samples": sampled.n_samples,
        "frequencies_hz": freqs.tolist() if freqs is not None else None,
        "node_ids": node_ids.tolist(),
        "phase_codes": phase_codes.tolist(),
        "config_type": type(cfg).__name__,
        "config_json": cfg.model_dump_json(),
        "seed": getattr(cfg, "seed", None),
        "perturbations": [p.model_dump() for p in sampled.perturbations],
        **sample_meta,
    }
    (path / _META).write_text(json.dumps(meta), encoding="utf-8")
    return path


def read_dataset(path) -> LoadedDataset:
    """Reload a dataset written by :func:`write_dataset` (layout-agnostic).

    Both ``"wide"`` and ``"long"`` layouts produce an identical
    :class:`LoadedDataset`; the layout is recorded in ``meta.json`` and handled
    transparently.

    Parameters
    ----------
    path:
        Dataset directory written by :func:`write_dataset` (must contain
        ``voltages.parquet``, ``meta.json``, and optionally ``samples.parquet``).

    Returns
    -------
    LoadedDataset
        Restored dataset with ``v`` shaped to the original
        ``ScenarioResult.v`` shape (``[B, N]`` / ``[B, H, N]`` / ``[B, T, H, N]``),
        reconstructed config, and ground-truth perturbation rows.
    """
    path = Path(path)
    meta = json.loads((path / _META).read_text(encoding="utf-8"))
    b, t, h, n = (meta["dims"][k] for k in ("B", "T", "H", "N"))

    df = pl.read_parquet(path / _VOLTAGES)
    if meta["layout"] == "long":
        df = df.sort(["scenario", "step", "freq_idx", "row"])
        re = df["v_re"].to_numpy().reshape(b, t, h, n)
        im = df["v_im"].to_numpy().reshape(b, t, h, n)
    else:
        df = df.sort(["scenario", "step"])
        re = np.stack(df["v_re"].to_numpy()).reshape(b, t, h, n)
        im = np.stack(df["v_im"].to_numpy()).reshape(b, t, h, n)
    v = torch.complex(
        torch.from_numpy(np.array(re)),  # np.array -> writable contiguous copy
        torch.from_numpy(np.array(im)),
    ).to(getattr(torch, meta["v_dtype"].replace("torch.", "")))
    v = v.reshape(*meta["v_shape"])

    samples: dict = {}
    spath = path / _SAMPLES
    if spath.is_file():
        sdf = pl.read_parquet(spath).sort("scenario")
        for name, shape in meta["sample_shapes"].items():
            col = sdf[name].to_numpy()
            arr = np.stack(col) if col.dtype == object or col.ndim > 1 else col
            tdt = getattr(torch, meta["sample_dtypes"][name].replace("torch.", ""))
            samples[name] = torch.from_numpy(np.array(arr)).reshape(b, *shape).to(tdt)
    for name, rec in meta["shared_samples"].items():
        tdt = getattr(torch, rec["dtype"].replace("torch.", ""))
        samples[name] = torch.tensor(rec["data"]).to(tdt)

    freqs = (
        torch.tensor(meta["frequencies_hz"], dtype=torch.float64)
        if meta["frequencies_hz"] is not None
        else None
    )
    cfg_cls = _CONFIG_TYPES.get(meta["config_type"])
    config = (
        cfg_cls.model_validate_json(meta["config_json"])
        if cfg_cls is not None
        else json.loads(meta["config_json"])
    )
    return LoadedDataset(
        v=v,
        samples=samples,
        frequencies_hz=freqs,
        node_ids=torch.tensor(meta["node_ids"], dtype=torch.int64),
        phase_codes=torch.tensor(meta["phase_codes"], dtype=torch.int64),
        config=config,
        perturbations=meta["perturbations"],
        meta=meta,
    )


__all__ = ["LoadedDataset", "write_dataset", "read_dataset"]
