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

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import polars as pl
import torch
from torch import Tensor

from ..errors import InputError
from ..schemas import SCHEMA_VERSION
from . import config as _cfg
from .run import ScenarioResult

_log = logging.getLogger("pgml")


def _check_schema_version(stored: Optional[str]) -> None:
    """Validate a dataset's stored ``schema_version`` against the current contract.

    A MAJOR-version mismatch is incompatible (raises); a minor/patch drift or a missing
    version (written before versioning) is read best-effort with a warning. The schema major
    tracks the library major during pre-1.0 development (see ``pgml.schemas.SCHEMA_VERSION``).
    """
    if stored is None:
        _log.warning(
            "dataset has no schema_version (written before versioning); current is %s — "
            "fields may have drifted.",
            SCHEMA_VERSION,
        )
        return
    if stored == SCHEMA_VERSION:
        return
    if stored.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        raise InputError(
            f"dataset schema_version {stored!r} is incompatible with the current schema "
            f"{SCHEMA_VERSION!r} (major-version mismatch)."
        )
    _log.warning(
        "dataset schema_version %s differs from the current %s (minor/patch drift; reading "
        "best-effort).",
        stored,
        SCHEMA_VERSION,
    )


_VOLTAGES = "voltages.parquet"
_SAMPLES = "samples.parquet"
_META = "meta.json"

#: The config classes :func:`read_dataset` reconstructs without being told, keyed by the
#: ``config_type`` string :func:`write_dataset` records. A dataset whose config class is
#: defined elsewhere is reconstructed by passing ``config_types=`` (the class name the
#: producer wrote, mapped to the class), so reading never imports a foreign module on a
#: guess.
SCENARIO_CONFIG_TYPES = {
    "ScenarioConfig": _cfg.ScenarioConfig,
    "CartesianConfig": _cfg.CartesianConfig,
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
        The full sidecar dict (``config_json``, ``seed``, ``dims``, ``layout``, the time
        axis ``n_steps`` / ``step_size_s`` / ``t0_unix_s``, the config owner
        ``config_module`` / ``config_class``, …) including the generation provenance
        (``config_hash``, ``provenance``, ``standards``, and the producer's own
        ``extra_provenance``). A dataset written before a key was stamped simply lacks it
        — reading is unaffected.
    converged:
        ``True`` iff every scenario's solve converged; ``None`` for a dataset written
        before convergence metadata was persisted (validity unknown).
    failed_scenarios:
        Scenario indices (along ``B``) whose solve did NOT converge — their stored
        voltages are best-effort iterates, not solutions. Consumers building training
        data must drop (or explicitly keep) these rows.
    """

    v: Tensor
    samples: dict
    frequencies_hz: Optional[Tensor]
    node_ids: Tensor
    phase_codes: Tensor
    config: object
    perturbations: list
    meta: dict
    converged: Optional[bool] = None
    failed_scenarios: tuple[int, ...] = ()


def _np(t: Tensor, dtype=np.float64) -> np.ndarray:
    """Detached CPU numpy view (result I/O, not the differentiable path)."""
    return t.detach().to("cpu").numpy().astype(dtype, copy=False)


def _canonical(
    v: Tensor, *, n_samples: int, n_steps: int, n_freq: int, n_rows: int
) -> tuple[Tensor, int, int, int]:
    """Reshape ``v`` to ``[B, T, H, N]`` from the DECLARED batch and index dimensions.

    The rank of ``v`` alone is ambiguous: ``[B, T, N]`` (a sequence power flow) and
    ``[B, H, N]`` (a snapshot harmonic run) are the same rank and would record one axis as
    the other. The batch declares ``B`` and ``T``, the result index declares ``N``, and the
    frequency vector declares ``H``, so the layout is read off the declarations and only
    checked against the element count.
    """
    dims = (int(n_samples), int(n_steps), int(n_freq), int(n_rows))
    expected = dims[0] * dims[1] * dims[2] * dims[3]
    if v.numel() != expected:
        raise InputError(
            f"ScenarioResult.v has {v.numel()} elements, which does not match the "
            f"declared layout B={dims[0]}, T={dims[1]}, H={dims[2]}, N={dims[3]} "
            f"({expected} elements); v has shape {tuple(v.shape)}."
        )
    return v.reshape(dims), dims[0], dims[1], dims[2]


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


def _real_array(name: str, t: Tensor) -> np.ndarray:
    """A detached numpy view of one sample record, in its NATURAL dtype (int vs float)."""
    arr = t.detach().to("cpu").numpy()
    if np.issubdtype(arr.dtype, np.complexfloating):
        # A float64 cast would silently DROP the imaginary part (and the
        # shared-sample JSON path cannot encode complex at all).
        raise InputError(
            f"sample column {name!r} is complex-valued; the parquet sample "
            "layout stores real columns only — split into real/imag (or "
            "magnitude/phase) columns before persisting."
        )
    return arr


def _write_samples(
    samples: dict, shared_records: dict, b: int, path: Path, comp: str
) -> dict:
    """Write the per-scenario sample columns; return shapes/dtypes + the shared block.

    ``shared_records`` are the records the batch DECLARED as batch-shared
    (:attr:`~pgml.scenarios.SampledScenarios.shared_samples`) and go to the sidecar
    regardless of their shape. An undeclared record is classified by its leading
    dimension, which is what a dataset written before the declaration existed relies on;
    a record of length ``B`` that is not per-scenario (a device-id column on a grid with
    as many devices as scenarios) can only be told apart by declaring it.
    """
    cols: dict = {"scenario": np.arange(b, dtype=np.int64)}
    shapes: dict = {}
    dtypes: dict = {}
    shared: dict = {}
    for name, t in shared_records.items():
        arr = _real_array(name, t)
        shared[name] = {"data": arr.tolist(), "dtype": str(t.dtype)}
    for name, t in samples.items():
        arr = _real_array(name, t)
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


def _step_size_s(config) -> Optional[float]:
    """The sequence step size in seconds, when the config declares one."""
    value = getattr(config, "step_size_s", None)
    return None if value is None else float(value)


def _t0_unix_s(sampled) -> Optional[float]:
    """Absolute epoch seconds of the first step, from the batch's own time record.

    Recorded so a consumer reads the dataset's time axis from the sidecar instead of
    re-deriving it from a generator-specific config field.
    """
    record = sampled.shared_samples.get(
        "time_unix_s", sampled.samples.get("time_unix_s")
    )
    if record is None or record.numel() == 0:
        return None
    return float(record.detach().to("cpu").reshape(-1)[0])


#: Length of the printable config fingerprint. 16 hex digits (64 bits) make an
#: accidental collision between two configs of one study impossible in practice while
#: staying short enough to read off a log line.
_HASH_CHARS = 16


def config_hash(config) -> str:
    """Stable short fingerprint of a scenario configuration.

    ``config`` is a pydantic scenario config (or its already-serialized JSON string).
    Two configs producing the same fingerprint describe the same batch — which is what
    makes a previously generated dataset reusable — and any changed field, including the
    seed and the sample count, changes it. Recorded in ``meta.json`` as ``config_hash``.

    Returns
    -------
    str
        16 lower-case hex characters.
    """
    payload = config if isinstance(config, str) else config.model_dump_json()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_HASH_CHARS]


def generation_provenance() -> dict:
    """Everything a generated dataset should record beyond its config and seed.

    Two datasets written from a byte-identical config and seed can still differ
    numerically — because the code changed, or because a ``PGML_*`` environment override
    replaced a standards table. Returns a JSON-ready dict with the code provenance
    (commit + dirty flag + versions) and the active EN 50160 / IEC 61000-3-2 tables with
    their content hashes. A generator with its own versioned inputs (a calibrated recipe,
    a device library) adds them through ``write_dataset(provenance=...)``.
    """
    from ..provenance import code_provenance
    from .en50160 import en50160_provenance
    from .iec61000_3_2 import iec61000_3_2_provenance

    return {
        "provenance": code_provenance(),
        "standards": {
            "en50160": en50160_provenance(),
            "iec61000_3_2": iec61000_3_2_provenance(),
        },
    }


def write_dataset(
    result: ScenarioResult,
    path,
    *,
    layout: str = "wide",
    compression: str = "zstd",
    also_csv: bool = False,
    provenance: Optional[dict] = None,
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
    provenance:
        The caller's own generation stamps, recorded in ``meta.json`` under
        ``extra_provenance`` (JSON-serializable values only). This is where a generator
        records what its config does not capture — the version of a calibrated recipe or
        of a device library whose defaults would change the data without changing the
        config. :func:`generation_provenance` covers what pgml itself contributes.

    Returns
    -------
    Path
        The dataset directory.
    """
    if layout not in ("wide", "long"):
        raise InputError(f"layout must be 'wide' or 'long', got {layout!r}.")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    node_ids = _np(result.index.node_ids, np.int64)
    phase_codes = _np(result.index.phase_codes, np.int64)
    freqs = _np(result.frequencies_hz) if result.frequencies_hz is not None else None
    v4, b, t, h = _canonical(
        result.v,
        n_samples=result.sampled.n_samples,
        n_steps=result.sampled.n_steps,
        n_freq=1 if freqs is None else len(freqs),
        n_rows=len(node_ids),
    )
    n = v4.shape[-1]
    vre, vim = _np(v4.real), _np(v4.imag)

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
    sample_meta = _write_samples(
        sampled.samples, sampled.shared_samples, b, path, compression
    )

    cfg = sampled.config
    # Reproducibility provenance: the schema contract version + the environment that
    # produced the float results (config+seed fixes the INPUTS; these tie the stored
    # voltages to the code/precision that computed them — they can differ across versions).
    try:
        import pgml

        _pgml_version = pgml.__version__
    except Exception:  # pragma: no cover - defensive
        _pgml_version = None
    config_json = None if cfg is None else cfg.model_dump_json()
    meta = {
        "schema_version": SCHEMA_VERSION,
        "pgml_version": _pgml_version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "layout": layout,
        "calculation": "harmonic"
        if result.frequencies_hz is not None
        else "power_flow",
        "v_shape": list(result.v.shape),
        "v_dtype": str(result.v.dtype),
        "dims": {"B": b, "T": t, "H": h, "N": n},
        "n_samples": sampled.n_samples,
        "n_steps": int(sampled.n_steps),
        "step_size_s": _step_size_s(cfg),
        "t0_unix_s": _t0_unix_s(sampled),
        "frequencies_hz": freqs.tolist() if freqs is not None else None,
        "node_ids": node_ids.tolist(),
        "phase_codes": phase_codes.tolist(),
        "config_type": None if cfg is None else type(cfg).__name__,
        "config_module": None if cfg is None else type(cfg).__module__,
        "config_class": None if cfg is None else type(cfg).__qualname__,
        "config_json": config_json,
        "config_hash": None if config_json is None else config_hash(config_json),
        "seed": getattr(cfg, "seed", None),
        "perturbations": [p.model_dump() for p in sampled.perturbations],
        "converged": bool(result.converged),
        "failed_scenarios": [int(i) for i in result.failed_states],
        "extra_provenance": dict(provenance or {}),
        **generation_provenance(),
        **sample_meta,
    }
    if result.failed_states:
        _log.warning(
            "write_dataset: %d/%d scenario(s) did NOT converge %s — their voltages "
            "are best-effort iterates. The indices are persisted in meta.json "
            "('failed_scenarios'); filter them before training.",
            len(result.failed_states),
            b,
            list(result.failed_states[:10]),
        )
    (path / _META).write_text(json.dumps(meta), encoding="utf-8")
    return path


def read_dataset(
    path, *, config_types: Optional[Mapping[str, type]] = None
) -> LoadedDataset:
    """Reload a dataset written by :func:`write_dataset` (layout-agnostic).

    Both ``"wide"`` and ``"long"`` layouts produce an identical
    :class:`LoadedDataset`; the layout is recorded in ``meta.json`` and handled
    transparently.

    Parameters
    ----------
    path:
        Dataset directory written by :func:`write_dataset` (must contain
        ``voltages.parquet``, ``meta.json``, and optionally ``samples.parquet``).
    config_types:
        Additional ``{class_name: class}`` entries for reconstructing the stored config,
        merged over :data:`SCENARIO_CONFIG_TYPES`. A dataset generated by a config class
        this package does not define reads back as a plain dict unless its class is named
        here; reconstruction is explicit because reading a dataset must not import a
        module chosen by the file's contents. The sidecar records
        ``config_module`` / ``config_class`` so the owner is identifiable.
        A stored config that its class no longer accepts (the dataset predates a change
        of that class) also reads back as a dict, with a warning, so the voltages and
        samples stay readable.

    Returns
    -------
    LoadedDataset
        Restored dataset with ``v`` shaped to the original
        ``ScenarioResult.v`` shape (``[B, N]`` / ``[B, H, N]`` / ``[B, T, H, N]``),
        reconstructed config, and ground-truth perturbation rows.
    """
    path = Path(path)
    meta = json.loads((path / _META).read_text(encoding="utf-8"))
    _check_schema_version(meta.get("schema_version"))
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
        # Build AT the recorded dtype: torch.tensor infers float32 for a python-float
        # list, which would truncate a large-magnitude float64 shared sample (e.g.
        # ``time_unix_s`` epoch seconds ~1.7e9) before the cast. The JSON decimals are
        # exact float64, so constructing at float64 restores them bit-for-bit.
        samples[name] = torch.tensor(rec["data"], dtype=tdt)

    freqs = (
        torch.tensor(meta["frequencies_hz"], dtype=torch.float64)
        if meta["frequencies_hz"] is not None
        else None
    )
    known = {**SCENARIO_CONFIG_TYPES, **(config_types or {})}
    cfg_cls = known.get(meta["config_type"])
    config_json = meta["config_json"]
    if config_json is None:
        config = None
    elif cfg_cls is not None:
        try:
            config = cfg_cls.model_validate_json(config_json)
        except ValueError as exc:
            # The class changed since the dataset was written (a removed field value, a
            # tightened constraint). The data is still valid, so hand the config back
            # as stored instead of refusing the whole dataset.
            _log.warning(
                "The stored %s of %s no longer validates (%s); returning it as a dict.",
                meta["config_type"],
                path,
                str(exc).splitlines()[0],
            )
            config = json.loads(config_json)
    else:
        config = json.loads(config_json)
    converged = meta.get("converged")
    return LoadedDataset(
        v=v,
        samples=samples,
        frequencies_hz=freqs,
        node_ids=torch.tensor(meta["node_ids"], dtype=torch.int64),
        phase_codes=torch.tensor(meta["phase_codes"], dtype=torch.int64),
        config=config,
        perturbations=meta["perturbations"],
        meta=meta,
        converged=bool(converged) if converged is not None else None,
        failed_scenarios=tuple(int(i) for i in meta.get("failed_scenarios", ())),
    )


__all__ = [
    "LoadedDataset",
    "SCENARIO_CONFIG_TYPES",
    "write_dataset",
    "read_dataset",
    "config_hash",
    "generation_provenance",
]
