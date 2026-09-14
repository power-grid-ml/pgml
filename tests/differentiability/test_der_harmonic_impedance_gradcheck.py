"""Differentiability of explicit DER harmonic impedance and voltage sources."""

from __future__ import annotations

import torch

from pgml.schemas import (
    Generator,
    Grid,
    HarmonicComponent,
    HarmonicImpedance,
    Line,
    Node,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
    WindingConnection,
)
from pgml.solver import harmonic_injections, solve_harmonic_flow


def _grid(resistance, inductance, *, reference="internal_voltage") -> Grid:
    phases = (Phase.A,)
    spectrum = StaticSpectrum(
        spectrum=SpectrumPoint(
            components=[
                HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=5.0),
                HarmonicComponent(order=5, magnitude_pu=0.08, phase_deg=23.0),
            ]
        )
    )
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=230.0, phases=phases),
            Node(id=2, u_rated_v=230.0, phases=phases),
        ],
        branches=[
            Line(
                id=20,
                from_node=1,
                to_node=2,
                from_phases=phases,
                to_phases=phases,
                length_m=100.0,
                series_resistance_ohm_per_m=[[1.0e-3]],
                series_inductance_h_per_m=[[1.0e-6]],
                shunt_capacitance_f_per_m=[[0.0]],
            )
        ],
        appliances=[
            Source(
                id=10,
                node=1,
                phases=phases,
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[1.0e-3]],
            ),
            Generator(
                id=30,
                node=2,
                phases=phases,
                p_nom_w=2_000.0,
                q_nom_var=300.0,
                spectrum=spectrum,
                harmonic_impedance=HarmonicImpedance(
                    resistance_ohm=resistance,
                    inductance_h=inductance,
                    spectrum_reference=reference,
                ),
            ),
        ],
    )


def _solve(resistance, inductance):
    return solve_harmonic_flow(
        _grid(resistance, inductance),
        [1, 5],
        load_shunt="none",
        dtype=torch.complex128,
    ).v.reshape(-1)


def test_gradcheck_internal_voltage_impedance():
    resistance = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    inductance = torch.tensor(1.5e-3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        _solve,
        (resistance, inductance),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )


def test_explicit_impedance_supersedes_derived_generator_shunt():
    grid = _grid(0.4, 1.5e-3, reference="current")
    none = solve_harmonic_flow(grid, [1, 5], load_shunt="none").v
    derived_requested = solve_harmonic_flow(grid, [1, 5], load_shunt="opendss").v
    torch.testing.assert_close(none, derived_requested, rtol=1e-12, atol=1e-12)


def test_opendss_wye_source_uses_phase_to_neutral_voltage():
    """A common phase/neutral displacement cannot change the WYE internal source."""
    abc = (Phase.A, Phase.B, Phase.C)
    spectrum = StaticSpectrum(
        spectrum=SpectrumPoint(
            components=[
                HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
                HarmonicComponent(order=3, magnitude_pu=0.1, phase_deg=0.0),
            ]
        )
    )
    grid = Grid(
        base_frequency_hz=50.0,
        nodes=[Node(id=1, u_rated_v=400.0, phases=(*abc, Phase.N))],
        appliances=[
            Generator(
                id=30,
                node=1,
                phases=abc,
                connection=WindingConnection.WYE,
                p_nom_w=0.0,
                q_nom_var=0.0,
                spectrum=spectrum,
                harmonic_impedance=HarmonicImpedance(
                    resistance_ohm=0.4,
                    inductance_h=1.5e-3,
                    spectrum_reference="opendss_voltage",
                    frequency_model="opendss_admittance",
                ),
            )
        ],
    )
    v1 = torch.tensor(
        [230.0 + 0j, -115.0 - 199.18584287j, -115.0 + 199.18584287j, 7.0 + 2.0j],
        dtype=torch.complex128,
    )
    displaced = v1 + torch.tensor(
        [11.0 - 4.0j, 11.0 - 4.0j, 11.0 - 4.0j, 11.0 - 4.0j],
        dtype=torch.complex128,
    )
    base = harmonic_injections(grid, v1, [3])
    shifted = harmonic_injections(grid, displaced, [3])
    torch.testing.assert_close(base, shifted, rtol=1e-12, atol=1e-12)
