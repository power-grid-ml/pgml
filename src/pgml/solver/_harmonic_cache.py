"""Bounded, value-validated preparation for repeated harmonic studies."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor


def _snapshot(value):
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(v) for v in value)
    return deepcopy(value)


def _equal(saved, current):
    if isinstance(saved, Tensor):
        return (
            isinstance(current, Tensor)
            and saved.dtype == current.dtype
            and saved.device == current.device
            and saved.shape == current.shape
            and torch.equal(saved, current)
        )
    if type(saved) is not type(current):
        return False
    if hasattr(saved, "__array__"):
        return (
            saved.dtype == current.dtype
            and saved.shape == current.shape
            and bool((saved == current).all())
        )
    if isinstance(saved, dict):
        return saved.keys() == current.keys() and all(
            _equal(v, current[k]) for k, v in saved.items()
        )
    if isinstance(saved, (tuple, list)):
        return len(saved) == len(current) and all(
            _equal(a, b) for a, b in zip(saved, current)
        )
    return saved == current


def _nbytes(value, seen: set, visited: set) -> int:
    """Tensor storage reachable from ``value``, counting each storage once.

    Walks containers and object attributes, so a factorization or a prepared system is
    measured through whatever it happens to hold. ``visited`` carries object ids to
    keep a cyclic object graph (a grid referencing its own components) finite.
    """
    if isinstance(value, Tensor):
        storage = value.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key in seen or key[0] == 0:
            return 0
        seen.add(key)
        return key[1]
    if isinstance(value, (str, bytes, int, float, complex, bool, type(None))):
        return 0
    if id(value) in visited:
        return 0
    visited.add(id(value))
    if isinstance(value, dict):
        return sum(_nbytes(v, seen, visited) for v in value.values())
    if isinstance(value, (tuple, list, set, frozenset)):
        return sum(_nbytes(v, seen, visited) for v in value)
    fields = getattr(value, "__dict__", None)
    if fields is not None:
        return sum(_nbytes(v, seen, visited) for v in fields.values())
    slots = getattr(type(value), "__slots__", ())
    return sum(
        _nbytes(getattr(value, name), seen, visited)
        for name in slots
        if hasattr(value, name)
    )


def _requires_grad(value):
    if isinstance(value, Tensor):
        return value.requires_grad
    if isinstance(value, dict):
        return any(_requires_grad(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_requires_grad(v) for v in value)
    return False


class HarmonicFlowSystem:
    """Lazy preparation state for repeated harmonic solves; not thread-safe.

    Pass an instance as ``system`` to :func:`solve_harmonic_flow`, or warm it with
    :func:`prepare_harmonic_flow`. Each call still solves the fundamental and
    evaluates device admittances and injections from the CURRENT operating point.
    Network assembly is reused only with identical grid data, defaults, overrides,
    branch states, frequencies and row layout. Numerical harmonic factors are
    reused only when the evaluated matrix and numerical options are exactly equal.
    Changed inputs automatically rebuild the affected entries, including in-place
    edits; the fundamental-only network fingerprint is not a harmonic cache key.

    Stores at most one fundamental preparation, one harmonic network matrix and
    one harmonic factorization, independent of the number of calls/chunks. Low-rank
    device-shunt updates are rebuilt on each call while their base can be reused.
    Differentiable harmonic matrices bypass numerical-factor caching; parameter
    gradients also bypass network caching. RHS-only gradients retain factor reuse.
    No autograd graph is retained by the harmonic entries.

    ``cache_batched_factors=False`` retains only scenario-independent harmonic
    factors. Use it for streams of different scenario chunks: snapshotting and
    retaining a full scenario matrix is unnecessary when the next chunk changes
    its admittances. The default also caches a repeated identical scenario batch.
    Retention adds a matrix snapshot and factors to the live solve workspace;
    one entry may still be large. This is bounded by entry count, not a byte cap.

    ``stats`` returns hit/miss/bypass counts for these three entries. Counts refer
    to calls, not individual matrices in a frequency/scenario batch. ``nbytes()``
    reports the tensor storage the entries currently hold, so a caller can size the
    retention against its own memory budget. ``clear()`` releases the entries and
    resets the counters. Cache lookup includes exact tensor comparisons, which can
    synchronize a CUDA device.

    Give each worker thread its OWN instance, reused sequentially by that worker.
    A shared instance has no locking around lookup/build/publication, ``clear()``,
    counters or use of backend factor handles; concurrent use is unsupported.
    Racing calls can duplicate work, evict one another's entries and lose counter
    updates. Correct concurrent backend solves are not guaranteed. Alternatively,
    serialize the ENTIRE consuming solve (and ``clear()``) with a caller-owned
    lock, not just preparation. Separate instances do not protect a shared grid,
    tensor inputs or defaults from concurrent mutation: keep those read-only or
    independently owned while workers solve.
    """

    def __init__(self, *, cache_batched_factors: bool = True):
        self.cache_batched_factors = cache_batched_factors
        self._entries = {}
        self._stats = {}

    @property
    def stats(self) -> dict[str, int]:
        """Copy of preparation hit/miss/bypass counters."""
        return dict(self._stats)

    def nbytes(self) -> int:
        """Bytes of tensor storage retained by the preparation, counting each once.

        Covers both halves of every entry: the value (a matrix, a factorization, a
        prepared fundamental system) and the key snapshot the validity check compares
        against, which for a numerical factor is a copy of the matrix itself. Shared
        storage is counted once, so a view of a retained tensor adds nothing. Python
        object overhead and non-tensor keys are not included.
        """
        return _nbytes(self._entries, set(), set())

    def clear(self) -> None:
        """Release all retained preparations and reset counters."""
        self._entries.clear()
        self._stats.clear()

    def _get(self, name, key, build, *, cacheable=True):
        def count(outcome):
            label = f"{name}_{outcome}"
            self._stats[label] = self._stats.get(label, 0) + 1

        if not cacheable:
            count("bypasses")
            return build()
        entry = self._entries.get(name)
        if entry is not None and _equal(entry[0], key):
            count("hits")
            return entry[1]
        count("misses")
        # Release an obsolete large matrix/factor pair before building its
        # replacement. A failed build leaves an empty slot, never partial state.
        self._entries.pop(name, None)
        del entry
        # Disable inference tensors as well as autograd so a later RHS-only
        # backward can use preparations built inside inference_mode().
        with torch.inference_mode(False), torch.no_grad():
            saved = _snapshot(key)
            value = build()
        self._entries[name] = (saved, value)
        return value
