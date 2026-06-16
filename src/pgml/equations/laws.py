"""M1 scalar / elementwise physical laws for the load-flow milestone.

Each law is a residual-form :class:`~pgml.equations.registry.Equation`
``0 = a - b`` registered with the module-level ``registry``. Every law's torch
evaluator is autograd-safe (``torch.*`` only), broadcasts over arbitrary tensor
shapes, and runs unchanged on CPU and CUDA.

Registered laws (M1)
--------------------
- ``reactance_from_inductance``      ``X = 2*pi*f*L``
- ``susceptance_from_capacitance``   ``B = 2*pi*f*C``
- ``series_admittance_scalar``       ``y = 1 / (R + 1j*X)``
- ``shunt_admittance_scalar``        ``y = G + 1j*B``
- ``skin_effect_multiplier``         ``R_eff = R0 * m``
- ``seq_to_phase_self``              ``Z_self = (Z0 + 2*Z1) / 3``
- ``seq_to_phase_mutual``            ``Z_mutual = (Z0 - Z1) / 3``

``f`` is the ABSOLUTE frequency in Hz; the harmonic order ``h = f / f0`` is a
caller concern (assembly multiplies f0 by h before calling these).
"""

from __future__ import annotations

import sympy

from .registry import Equation, SymbolMeta, registry

# ---------------------------------------------------------------------------
# Symbols (real-valued physical quantities; complex appears via the imaginary
# unit sympy.I in the residual, which lambdify maps to a complex literal).
# ---------------------------------------------------------------------------
_f = sympy.Symbol("f", real=True)
_L = sympy.Symbol("L", real=True)
_C = sympy.Symbol("C", real=True)
_R = sympy.Symbol("R", real=True)
_G = sympy.Symbol("G", real=True)
_X = sympy.Symbol("X", real=True)
_B = sympy.Symbol("B", real=True)
_R0 = sympy.Symbol("R0", real=True)
_m = sympy.Symbol("m", real=True)
_R_eff = sympy.Symbol("R_eff", real=True)
_y = sympy.Symbol("y")  # admittance result symbol (complex-valued)
_Z0 = sympy.Symbol("Z0")  # sequence impedances (may be complex)
_Z1 = sympy.Symbol("Z1")
_Z_self = sympy.Symbol("Z_self")
_Z_mutual = sympy.Symbol("Z_mutual")

_TWO_PI = 2 * sympy.pi


def _register_all() -> None:
    """Register every M1 law (idempotent-safe via duplicate-id guard upstream)."""
    registry.register(
        Equation(
            id="reactance_from_inductance",
            description="Inductive reactance at absolute frequency f: X = 2*pi*f*L.",
            residual=_X - _TWO_PI * _f * _L,
            symbols={
                "X": SymbolMeta(unit="Ohm", description="Reactance at f."),
                "f": SymbolMeta(unit="Hz", description="Absolute frequency."),
                "L": SymbolMeta(
                    unit="H",
                    schema_field="Line.series_inductance_h_per_m",
                    description="Inductance.",
                ),
            },
            tags=("series", "reactive", "frequency"),
        )
    )

    registry.register(
        Equation(
            id="susceptance_from_capacitance",
            description="Capacitive susceptance at absolute frequency f: B = 2*pi*f*C.",
            residual=_B - _TWO_PI * _f * _C,
            symbols={
                "B": SymbolMeta(unit="S", description="Susceptance at f."),
                "f": SymbolMeta(unit="Hz", description="Absolute frequency."),
                "C": SymbolMeta(
                    unit="F",
                    schema_field="Line.shunt_capacitance_f_per_m",
                    description="Capacitance.",
                ),
            },
            tags=("shunt", "reactive", "frequency"),
        )
    )

    registry.register(
        Equation(
            id="series_admittance_scalar",
            description="Scalar series admittance y = 1 / (R + j*X).",
            residual=_y - 1 / (_R + sympy.I * _X),
            symbols={
                "y": SymbolMeta(unit="S", description="Series admittance (complex)."),
                "R": SymbolMeta(unit="Ohm", description="Series resistance."),
                "X": SymbolMeta(unit="Ohm", description="Series reactance."),
            },
            tags=("series", "admittance"),
        )
    )

    registry.register(
        Equation(
            id="shunt_admittance_scalar",
            description="Scalar shunt admittance y = G + j*B.",
            residual=_y - (_G + sympy.I * _B),
            symbols={
                "y": SymbolMeta(unit="S", description="Shunt admittance (complex)."),
                "G": SymbolMeta(unit="S", description="Shunt conductance."),
                "B": SymbolMeta(unit="S", description="Shunt susceptance."),
            },
            tags=("shunt", "admittance"),
        )
    )

    registry.register(
        Equation(
            id="skin_effect_multiplier",
            description="Effective resistance with a frequency multiplier: R_eff = R0*m.",
            residual=_R_eff - _R0 * _m,
            symbols={
                "R_eff": SymbolMeta(
                    unit="Ohm", description="Effective resistance at f."
                ),
                "R0": SymbolMeta(
                    unit="Ohm", description="Reference-frequency resistance."
                ),
                "m": SymbolMeta(
                    unit="pu",
                    schema_field="ResistanceFrequencyModel.multiplier",
                    description="Per-unit resistance multiplier (1.0 = none).",
                ),
            },
            tags=("series", "resistance", "skin_effect"),
        )
    )

    registry.register(
        Equation(
            id="seq_to_phase_self",
            description="Sequence->phase self term: Z_self = (Z0 + 2*Z1) / 3.",
            residual=_Z_self - (_Z0 + 2 * _Z1) / 3,
            symbols={
                "Z_self": SymbolMeta(
                    unit="Ohm", description="Phase-domain self impedance."
                ),
                "Z0": SymbolMeta(unit="Ohm", description="Zero-sequence impedance."),
                "Z1": SymbolMeta(
                    unit="Ohm", description="Positive-sequence impedance."
                ),
            },
            tags=("sequence", "transform"),
        )
    )

    registry.register(
        Equation(
            id="seq_to_phase_mutual",
            description="Sequence->phase mutual term: Z_mutual = (Z0 - Z1) / 3.",
            residual=_Z_mutual - (_Z0 - _Z1) / 3,
            symbols={
                "Z_mutual": SymbolMeta(
                    unit="Ohm", description="Phase-domain mutual impedance."
                ),
                "Z0": SymbolMeta(unit="Ohm", description="Zero-sequence impedance."),
                "Z1": SymbolMeta(
                    unit="Ohm", description="Positive-sequence impedance."
                ),
            },
            tags=("sequence", "transform"),
        )
    )


_register_all()


__all__ = []
