from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class MeasurementTokenSchema:
    """
    Defines the numeric layout of a measurement token.

    Minimal first implementation:
    value_real, value_imag
    """
    value_dim: int = 2


@dataclass(frozen=True)
class DeviceParameterSchema:
    """
    Canonical token names for per-device scalar parameters.

    Tokens will be stored as [value] + metadata instead of wide fixed vectors
    in the future model. For the first dataset implementation they will still
    be assembled into padded token arrays.
    """
    load_param_names: List[str] = None
    generator_param_names: List[str] = None
    vsource_param_names: List[str] = None
    injected_param_names: List[str] = None

    def __post_init__(self):
        object.__setattr__(self, "load_param_names", self.load_param_names or [
            "p1", "q1", "p2", "q2", "p3", "q3"
        ])
        object.__setattr__(self, "generator_param_names", self.generator_param_names or [
            "p1", "q1", "p2", "q2", "p3", "q3"
        ])
        object.__setattr__(self, "vsource_param_names", self.vsource_param_names or [
            "pu1", "pu2", "pu3"
        ])
        object.__setattr__(self, "injected_param_names", self.injected_param_names or [
            "sc1_mva"
        ])


DEVICE_TYPE_MAP: Dict[str, int] = {
    "load": 0,
    "generator": 1,
    "vsource": 2,
    "injected": 3,
}

NODE_MEASUREMENT_TYPE_MAP: Dict[str, int] = {
    "voltage": 0,
}

EDGE_MEASUREMENT_TYPE_MAP: Dict[str, int] = {
    "current": 0,
    "power": 1,
}

DEVICE_TOKEN_TYPE_MAP: Dict[str, int] = {
    "param": 0,
    "spectrum": 1,
}