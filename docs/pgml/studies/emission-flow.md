# Emission flow

{func}`pgml.studies.emission_flow` answers how the emission of inverters spreads through a
grid: which harmonic, interharmonic and supraharmonic voltages appear at each node when every
modelled unit injects the current its physical model predicts, and how much of a unit's
terminal current it produces itself rather than draws from the rest of the grid. The inverter
models are those of {mod}`pgml.inverter`; the derivation of the model and its validation are
on {doc}`../modeling/inverter-emission`.

The study solves the fundamental power flow (with the source's Thevenin impedance,
`slack="norton"`, by default; the inverter sources follow the terminal voltage), evaluates each inverter as a frequency-domain
Norton equivalent `i = I_N - Y v` at its operating point, and solves the linear network on a
frequency grid that holds every emission line of the models plus any requested frequency.
Frequencies need not be integer multiples of the fundamental, which is the difference to
{func}`pgml.solve_harmonic_flow`. Without inverter models the study reproduces the harmonic
flow at the integer orders.

## A single-phase unit on a weak feeder

```python
import math

import torch

from pgml.inverter import (
    CurrentController, DCLinkControl, GridFollowingInverter, InverterState,
    LCLFilter, PerturbObserveMPPT, PhaseLockedLoop, PWMModulator, SeriesStage, ShuntStage,
)
from pgml.schemas import Generator, Grid, Line, Node, Phase, Source
from pgml.studies import emission_flow

A = (Phase.A,)
grid = Grid(
    base_frequency_hz=50.0,
    nodes=[Node(id=i, u_rated_v=230.0, phases=A) for i in (1, 2)],
    branches=[Line(id=10, from_node=1, to_node=2, from_phases=A, to_phases=A,
                   length_m=1.0, series_resistance_ohm_per_m=[[0.25]],
                   series_inductance_h_per_m=[[0.12 / (2 * math.pi * 50.0)]],
                   shunt_capacitance_f_per_m=[[0.0]], harmonic_line_model="naive")],
    appliances=[
        Source(id=1, node=1, phases=A, u_ref_v=(230.0,), u_angle_deg=(0.0,),
               resistance_ohm=[[1e-3]], inductance_h=[[1e-6]]),
        Generator(id=7, node=2, phases=A, p_nom_w=3000.0, q_nom_var=0.0),
    ],
)

inverter = GridFollowingInverter(
    filter=LCLFilter(l1_h=1.5e-3, r1_ohm=0.05, c_f=4.7e-6, r_d_ohm=1.0,
                     l2_h=0.5e-3, r2_ohm=0.05),
    control=CurrentController(
        "pr", kp=6.0, resonant={1: 500.0, 3: 150.0, 5: 100.0, 7: 80.0},
        resonant_bandwidth_rad_s=5.0, sampling_hz=32000.0,
        feedforward_gain=1.0, feedforward_cutoff_hz=1000.0),
    pwm=PWMModulator("single_phase_unipolar", carrier_hz=16000.0,
                     carrier_groups=4, sidebands=10),
    pll=PhaseLockedLoop(kp=0.5, ki=40.0),
    emc=(SeriesStage(l_h=20e-6, r_ohm=0.01), ShuntStage(c_f=1.0e-6, c_ground_f=4.7e-9)),
    dead_time_s=1.0e-6,
    dc_link=DCLinkControl(c_dc_f=1100e-6, kp=0.05, ki=2.0, sampling_hz=32000.0),
    mppt=PerturbObserveMPPT(v_step_v=4.0, rate_hz=5.0, max_line_hz=200.0),
    harmonic_orders=tuple(range(2, 41)),
)


states = {7: InverterState(u_dc_v=400.0)}
res = emission_flow(grid, {7: inverter}, states=states, f_max_hz=40000.0)

row = res.rows(2)[0]                          # node 2, phase a
k = int(torch.argmax(res.v_rms[:, row]))
print(f"largest line {res.frequencies_hz[k]:.1f} Hz, {res.v_rms[k, row] * 1e3:.1f} mV")
print(f"smallest stability margin {float(res.stability_margin[7].min()):.2f}")
```

```text
largest line 450.0 Hz, 269.8 mV
smallest stability margin 0.90
```

Instead of the `inverters` argument, the model can be stored in the grid as the
device's `inverter_model` block (see {doc}`../modeling/inverter-emission`); `emission_flow`
then finds it there, and an argument entry overrides a stored block. The inverter dictionary
is keyed by the id of an in-service `Generator` or `Storage`; the
device's own `spectrum` is ignored for it. `states` carries one
{class}`~pgml.inverter.InverterState` per modelled unit (DC-link voltage, whether the bridge
switches and the relay is closed, the carrier phase). `f_max_hz` caps the frequency grid;
`harmonic_orders` and `frequencies_hz` add integer orders or arbitrary frequencies.
{func}`~pgml.studies.emission_frequencies` returns the grid the study will solve.

## Reading the result

The result is an {class}`~pgml.studies.EmissionFlow`. Its voltage axis is the grid's full
node-phase layout ({meth}`~pgml.studies.EmissionFlow.rows` maps a node and phase to rows);
the frequency axis is `frequencies_hz`, ascending, without the fundamental.

- `v` is the coherent solution, every source at its stated phase. Dead-time and DC-link ripple
  components are locked to the unit's fundamental and add as phasors.
- `v_rms` is the expected magnitude when the components without a common phase reference
  (PWM residues on free-running carriers, MPPT lines on independent clocks) of different units
  have independent uniform phases: `E|V|² = |V_locked|² + Σ_d |V_unlocked,d|²`. Use it for
  planning numbers; `v` depends on the arbitrary carrier phases. `v_locked` is the locked
  part alone.
- `device_current[id]` is the coherent current leaving each inverter and
  `device_primary[id]` the part driven by its own sources with all others set to zero. The
  difference is the secondary emission, driven by the rest of the grid.
- `stability_margin[id]` is the smallest singular value of `I + Z_g Y` per frequency, with
  `Z_g` the network seen from the unit. A value near zero flags a frequency where the unit and
  the grid interact so strongly that the steady-state result may have no physical meaning. It
  is a margin indicator, not a Nyquist test; `stability=False` skips it.
- `norton[id]` is the {class}`~pgml.inverter.NortonEquivalent` evaluated at the solved
  operating point.

## Holding the result against compatibility levels

{meth}`~pgml.studies.EmissionFlow.compatibility` divides the solved voltage at a node by
the IEC 61000-2-2 (low voltage) or IEC 61000-2-12 (`network="mv"`) compatibility level,
per harmonic order and, for low voltage, per 200 Hz band from 2 kHz to 150 kHz:

```python
out = res.compatibility(2, "a")
print(float(out["max_ratio"]))    # 1.0 = at the compatibility level
```

The tables and what they leave out are described on {doc}`../modeling/limitations`.

## Separating the mechanisms

The `admittance` option selects which inverter admittance the network sees:

| `admittance` | the network sees |
|---|---|
| `"model"` (default) | the closed-loop admittance through the filter |
| `"passive"` | the filter with the bridge shorted, the control's contribution removed |
| `"none"` | a pure current source, the classic harmonic current-source model |

The Norton currents are identical in all three; they are short-circuit currents, so the
terminal current of a unit changes with the mode. Running the example with each option gives
a largest node-2 `v_rms` of 0.270 V, 0.212 V and 0.305 V respectively, so the control and
the filter measurably shape the spread.

## Batching an irradiance sweep

Every field of `InverterState` and of the `operating_point` entries accepts a scenario batch,
so a sweep is one call. The result gains the leading batch dimension.

```python
g = torch.tensor([50.0, 200.0, 500.0, 1000.0], dtype=torch.float64)   # [W/m2]
res = emission_flow(
    grid, {7: inverter},
    states={7: InverterState(u_dc_v=torch.full_like(g, 400.0))},
    operating_point={7: {"p_w": 3000.0 * g / 1000.0}},
    f_max_hz=40000.0,
)
print(res.v_rms.shape)                        # [batch, frequencies, node-phases]
```

```text
torch.Size([4, 161, 2])
```

For a single-stage unit the DC-link voltage follows the array:
{meth}`pgml.inverter.PVArray.mpp` returns the maximum-power point `(P, V)` at an irradiance and
cell temperature, which can feed `p_w` and `u_dc_v` directly. A night state is
`InverterState(u_dc_v=0.0, bridge_on=False, connected=False)`.

## Device and dtype, gradients

`dtype` (`"complex128"` by default, or `"complex64"`) and `device` select the working
precision and where the solves run. The study is differentiable in every tensor-valued grid
and inverter parameter and in the operating point; the frequency grid is chunked against the
`solver.harmonic.system_budget_mb` setting. The result has `to_dict()` and `from_dict()` like
the other studies. The other keywords (`operating_point`, `harmonic_injection`,
`node_sources`, `load_shunt`, `slack`, `method`) are those of
{func}`pgml.solve_harmonic_flow`.
