"""Shared helpers for fundamental positive-sequence exporters."""

from __future__ import annotations

from typing import Any

import numpy as np

from pgml.errors import ModelingError


class UnsupportedGridError(ModelingError):
    """A grid uses an element or option that an exporter cannot represent."""


def scalar(value: Any) -> float:
    """Return one scalar from a scalar-like schema field."""
    array = np.asarray(detached(value), dtype=float)
    if array.ndim == 0:
        return float(array)
    return float(array.reshape(-1)[0])


def detached(value: Any) -> Any:
    """Convert an optional autograd tensor to host data for an external library."""
    detach = getattr(value, "detach", None)
    return detach().cpu().numpy() if callable(detach) else value


def positive_sequence(value: Any) -> float:
    """Reduce a transposed phase matrix to its positive-sequence scalar."""
    if value is None:
        return 0.0
    array = np.asarray(detached(value), dtype=float)
    if array.ndim == 0:
        return float(array)
    if array.ndim == 1:
        return float(array[0])
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise UnsupportedGridError(
            f"positive-sequence reduction requires a square matrix, got {array.shape}"
        )
    count = array.shape[0]
    if count not in (1, 3):
        raise UnsupportedGridError(
            f"positive-sequence reduction supports 1x1 or 3x3 matrices, got {array.shape}"
        )
    if count == 1:
        return float(array[0, 0])
    diagonal = float(np.trace(array) / count)
    mutual = float((array.sum() - np.trace(array)) / (count * (count - 1)))
    return diagonal - mutual


def check_matrix_shape(value: Any, description: str) -> None:
    """Reject a non-square phase matrix."""
    if value is None:
        return
    array = np.asarray(detached(value), dtype=float)
    if array.ndim >= 2 and (
        array.ndim != 2
        or array.shape[0] != array.shape[1]
        or array.shape[0] not in (1, 3)
    ):
        raise UnsupportedGridError(
            f"{description}: expected a 1x1 or 3x3 phase matrix, got {array.shape}"
        )


def is_coupled(value: Any) -> bool:
    """Return whether a phase matrix has non-zero mutual terms."""
    if value is None:
        return False
    array = np.asarray(detached(value), dtype=float)
    if array.ndim < 2 or array.shape[0] < 2:
        return False
    off_diagonal = array - np.diag(np.diag(array))
    return bool(np.any(np.abs(off_diagonal) > 1e-18))


def phase_totals(appliance: Any) -> tuple[float, float]:
    """Return total P/Q, folding explicit per-phase values when present."""
    p_per_phase = getattr(appliance, "p_nom_per_phase_w", None)
    if p_per_phase is not None:
        p_w = float(np.sum(np.asarray(detached(p_per_phase), dtype=float)))
    else:
        p_w = scalar(appliance.p_nom_w)
    q_per_phase = getattr(appliance, "q_nom_per_phase_var", None)
    if q_per_phase is not None:
        q_var = float(np.sum(np.asarray(detached(q_per_phase), dtype=float)))
    else:
        q_var = scalar(appliance.q_nom_var)
    return p_w, q_var


def has_unbalanced_power(appliance: Any) -> bool:
    """Return whether explicit per-phase P or Q values are unbalanced."""
    for name in ("p_nom_per_phase_w", "q_nom_per_phase_var"):
        value = getattr(appliance, name, None)
        if value is None:
            continue
        array = np.asarray(detached(value), dtype=float).reshape(-1)
        if array.size > 1 and not np.allclose(array, array[0]):
            return True
    return False


def harmonic_fields(appliance: Any) -> list[str]:
    """Name populated harmonic-only fields ignored by a fundamental exporter."""
    populated = []
    for name in (
        "spectrum",
        "spectrum_per_phase",
        "harmonic_model",
        "harmonic_impedance",
    ):
        value = getattr(appliance, name, None)
        if value is not None and value != {}:
            populated.append(name)
    return populated


def shunt_positive_sequence(
    appliance: Any, angular_frequency: float
) -> tuple[float, float]:
    """Return fundamental positive-sequence ``(G, B)`` for a shunt appliance."""
    conductance = np.asarray(detached(appliance.conductance_s), dtype=float).reshape(-1)
    capacitance = np.asarray(detached(appliance.capacitance_f), dtype=float).reshape(-1)
    susceptance = angular_frequency * capacitance
    if appliance.inductance_h is not None:
        inductance = np.asarray(detached(appliance.inductance_h), dtype=float).reshape(
            -1
        )
        susceptance = susceptance - 1.0 / (angular_frequency * inductance)
    multiplier = (
        3.0
        if getattr(appliance.connection, "value", appliance.connection) == "delta"
        else 1.0
    )
    return multiplier * float(conductance.mean()), multiplier * float(
        susceptance.mean()
    )


def shunt_is_unbalanced(appliance: Any) -> bool:
    """Return whether a shunt appliance has unequal connection-element values."""
    for name in ("conductance_s", "capacitance_f", "inductance_h"):
        value = getattr(appliance, name, None)
        if value is None:
            continue
        array = np.asarray(detached(value), dtype=float).reshape(-1)
        if array.size > 1 and not np.allclose(array, array[0]):
            return True
    return False


def validate_balanced_grid_phases(grid: Any) -> None:
    """Require one consistent positive-sequence-compatible phase layout."""
    single = ("a",)
    three = ("a", "b", "c")
    node_layouts = {tuple(phase.value for phase in node.phases) for node in grid.nodes}
    if node_layouts not in ({single}, {three}):
        raise UnsupportedGridError(
            "balanced export requires every node to use the same (A,) or (A,B,C) "
            f"layout; found {sorted(node_layouts)}"
        )
    expected = next(iter(node_layouts))
    for branch in grid.branches:
        for terminal, phases in (
            ("from", branch.from_phases),
            ("to", branch.to_phases),
        ):
            layout = tuple(phase.value for phase in phases)
            if layout != expected:
                raise UnsupportedGridError(
                    f"branch {branch.id} {terminal} phases {layout} do not match the "
                    f"balanced node layout {expected}"
                )
    for appliance in grid.appliances:
        layout = tuple(phase.value for phase in appliance.phases)
        if layout != expected:
            raise UnsupportedGridError(
                f"appliance {appliance.id} phases {layout} do not match the balanced "
                f"node layout {expected}"
            )


__all__ = [
    "UnsupportedGridError",
    "check_matrix_shape",
    "detached",
    "harmonic_fields",
    "has_unbalanced_power",
    "is_coupled",
    "phase_totals",
    "positive_sequence",
    "scalar",
    "shunt_is_unbalanced",
    "shunt_positive_sequence",
    "validate_balanced_grid_phases",
]
