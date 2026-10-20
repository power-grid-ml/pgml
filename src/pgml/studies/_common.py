"""Argument and serialization checks shared by the planning studies."""

from __future__ import annotations

from typing import Any, Mapping, Union

import torch

from pgml.errors import InputError

_COMPLEX_DTYPES = {"complex128": torch.complex128, "complex64": torch.complex64}


def resolve_complex_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """The complex working dtype named by ``dtype``.

    Accepts ``torch.complex64`` / ``torch.complex128`` or their names (``"complex64"``,
    ``"complex128"``, also with a ``"torch."`` prefix); anything else raises
    :class:`~pgml.errors.InputError`.
    """
    if isinstance(dtype, torch.dtype):
        if dtype in _COMPLEX_DTYPES.values():
            return dtype
    elif isinstance(dtype, str):
        resolved = _COMPLEX_DTYPES.get(dtype.removeprefix("torch."))
        if resolved is not None:
            return resolved
    raise InputError(
        f"dtype must be torch.complex64 or torch.complex128 (or their names "
        f"{sorted(_COMPLEX_DTYPES)}), got {dtype!r}."
    )


def check_study(data: Mapping[str, Any], expected: str) -> None:
    """Refuse a result dict whose ``"study"`` key does not name ``expected``."""
    found = data.get("study") if isinstance(data, Mapping) else None
    if found != expected:
        raise InputError(
            f'expected a {expected!r} result dict ("study": {expected!r}), '
            f'found "study": {found!r}.'
        )
