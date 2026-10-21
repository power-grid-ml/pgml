"""Small LV grids and inverter parameter sets for the inverter-model tests."""

from __future__ import annotations

import math

from pgml.inverter import (
    CurrentController,
    DCLinkControl,
    GridFollowingInverter,
    InverterState,
    LCLFilter,
    PerturbObserveMPPT,
    PhaseLockedLoop,
    PWMModulator,
    SeriesStage,
    ShuntStage,
)
from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    HarmonicComponent,
    Line,
    Load,
    Node,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
)

ABC = (Phase.A, Phase.B, Phase.C)
F0 = 50.0
U_LL = 400.0


def _diag(x: float, n: int = 3) -> list[list[float]]:
    return [[x if i == j else 0.0 for j in range(n)] for i in range(n)]


def _cable(lid: int, a: int, b: int, length_m: float) -> Line:
    return Line(
        id=lid,
        from_node=a,
        to_node=b,
        from_phases=ABC,
        to_phases=ABC,
        length_m=length_m,
        series_resistance_ohm_per_m=_diag(0.206e-3),
        series_inductance_h_per_m=_diag(0.256e-6),
        shunt_capacitance_f_per_m=_diag(0.25e-9),
        harmonic_line_model="naive",
    )


def lv_feeder(
    *, p_pv3_w: float = 9000.0, p_pv1_w: float = 4000.0, load_w: float = 6000.0
) -> Grid:
    """Source 1 -- cable -- 2 -- cable -- 3 -- cable -- 4, 400 V, with PV inverters.

    Device ids: 101 a three-phase PV inverter at node 3, 102 and 103 single-phase PV
    inverters on phases a and b at node 4, 201 to 203 balanced loads with a 5 % fifth
    and 3 % seventh harmonic current, 1 the source (Thevenin 250 kVA-transformer-like).
    """
    u_ln = U_LL / math.sqrt(3.0)
    spectrum = StaticSpectrum(
        spectrum=SpectrumPoint(
            components=[
                HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
                HarmonicComponent(order=5, magnitude_pu=0.05, phase_deg=30.0),
                HarmonicComponent(order=7, magnitude_pu=0.03, phase_deg=-60.0),
            ]
        )
    )
    loads = [
        Load(
            id=200 + k,
            node=k + 1,
            phases=ABC,
            p_nom_w=load_w,
            q_nom_var=0.3 * load_w,
            spectrum=spectrum,
        )
        for k in (1, 2, 3)
    ]
    return Grid(
        base_frequency_hz=F0,
        nodes=[Node(id=i, u_rated_v=U_LL, phases=ABC) for i in (1, 2, 3, 4)],
        branches=[
            _cable(11, 1, 2, 120.0),
            _cable(12, 2, 3, 150.0),
            _cable(13, 3, 4, 100.0),
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ABC,
                u_ref_v=(u_ln, u_ln, u_ln),
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=_diag(0.01),
                inductance_h=_diag(30e-6),
            ),
            *loads,
            Generator(id=101, node=3, phases=ABC, p_nom_w=p_pv3_w, q_nom_var=0.0),
            Generator(
                id=102, node=4, phases=(Phase.A,), p_nom_w=p_pv1_w, q_nom_var=0.0
            ),
            Generator(
                id=103, node=4, phases=(Phase.B,), p_nom_w=p_pv1_w, q_nom_var=0.0
            ),
        ],
    )


def single_phase_inverter(**overrides) -> GridFollowingInverter:
    """A 4.6 kW single-phase unit: unipolar PWM at 16 kHz carrier, PR control, SOGI-like PLL."""
    kw = dict(
        filter=LCLFilter(
            l1_h=1.5e-3, r1_ohm=0.05, c_f=4.7e-6, r_d_ohm=1.0, l2_h=0.5e-3, r2_ohm=0.05
        ),
        control=CurrentController(
            "pr",
            kp=6.0,
            resonant={1: 500.0, 3: 150.0, 5: 100.0, 7: 80.0},
            resonant_bandwidth_rad_s=5.0,
            sampling_hz=32000.0,
            feedforward_gain=1.0,
            feedforward_cutoff_hz=1000.0,
        ),
        pwm=PWMModulator(
            "single_phase_unipolar", carrier_hz=16000.0, carrier_groups=3, sidebands=10
        ),
        pll=PhaseLockedLoop(kp=0.5, ki=40.0),
        emc=(
            SeriesStage(l_h=20e-6, r_ohm=0.01),
            ShuntStage(c_f=1.0e-6, c_ground_f=4.7e-9),
        ),
        dead_time_s=1.0e-6,
        dc_link=DCLinkControl(c_dc_f=1100e-6, kp=0.05, ki=2.0, sampling_hz=32000.0),
        mppt=PerturbObserveMPPT(v_step_v=4.0, rate_hz=5.0, n_terms=6),
        harmonic_orders=tuple(range(2, 41)),
    )
    kw.update(overrides)
    return GridFollowingInverter(**kw)


def three_phase_inverter(**overrides) -> GridFollowingInverter:
    """A 10 kW three-phase unit: two-level PWM at 10 kHz, dq PI control and SRF-PLL."""
    kw = dict(
        filter=LCLFilter(
            l1_h=2.0e-3, r1_ohm=0.05, c_f=10e-6, r_d_ohm=0.5, l2_h=0.5e-3, r2_ohm=0.03
        ),
        control=CurrentController(
            "pi_dq", kp=10.0, ki=2000.0, sampling_hz=20000.0, active_damping_kc=4.0
        ),
        pwm=PWMModulator(
            "three_phase", carrier_hz=10000.0, carrier_groups=3, sidebands=10
        ),
        pll=PhaseLockedLoop(kp=0.3, ki=30.0),
        emc=(ShuntStage(c_f=2.2e-6, c_ground_f=10e-9),),
        dead_time_s=1.5e-6,
        dc_link=DCLinkControl(c_dc_f=1500e-6, kp=0.05, ki=2.0, sampling_hz=20000.0),
        mppt=PerturbObserveMPPT(v_step_v=5.0, rate_hz=10.0, n_terms=6),
        harmonic_orders=tuple(range(2, 41)),
    )
    kw.update(overrides)
    return GridFollowingInverter(**kw)


def default_states() -> dict:
    return {
        101: InverterState(u_dc_v=700.0),
        102: InverterState(u_dc_v=400.0),
        103: InverterState(u_dc_v=400.0, carrier_angle_rad=1.0),
    }


def default_models() -> dict:
    return {
        101: three_phase_inverter(),
        102: single_phase_inverter(),
        103: single_phase_inverter(),
    }
