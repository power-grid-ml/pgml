# inverter — grid-following inverter emission models

Physical blocks of a grid-following inverter and their evaluation to a terminal Norton
equivalent `i = I_N - Y v` at any frequency. Consumed by `pgml.studies.emission_flow`.
User-facing description: `docs/pgml/modeling/inverter-emission.md`.

Signatures: docstrings / API_INDEX.md.

## Codemap

- `_common.py` — tensor helpers, `s = j 2 pi f`, symmetrical-component transforms
  (`sequence_to_element`, `element_to_sequence`, `sequence_to_vector`).
- `filters.py` — `LCLFilter` (plant `Yp`, `Yo`), EMC stages `SeriesStage` / `ShuntStage`,
  `propagate_norton` (Norton through stages, inner to outer), `inner_port_state` (terminal
  fundamental back to the inner port).
- `control.py` — `CurrentController` (PR or dq PI, delay, feedforward, capacitor-current
  damping), `PhaseLockedLoop`, `closed_loop` (exact 4x4 solve per frequency of circuit +
  control law), `ClosedLoop`.
- `pwm.py` — `PWMModulator` (naturally sampled double Fourier series, three topologies),
  `bessel_j` (differentiable trapezoidal Bessel integral).
- `dead_time.py` — `dead_time_voltage` (hard sign, or the local-ripple dead zone).
- `dc_link.py` — `DCLinkControl` (Sangwongwanich eq. 4 loop), `PerturbObserveMPPT`,
  `mppt_current_lines`, `ripple_third_harmonic`.
- `operating.py` — `InverterState` (DC-link voltage, bridge and relay state, carrier phase),
  `PVArray` (MPP power and voltage from irradiance and temperature).
- `model.py` — `GridFollowingInverter` (`emission_frequencies`, `norton`), `NortonEquivalent`,
  `_place_lines` (lines onto the frequency grid).

## Invariants an edit must not undo

- Conventions: RMS phasors in the grid's phase reference; current LEAVING the device; per
  connection element. Three-phase units in symmetrical components: positive sequence =
  space vector at `+f`, negative = `conj(G(-j w))`, zero = no inner path (three-wire). A
  single-phase unit averages the two space-vector evaluations.
- The solve is decoupled: only the same-frequency admittance enters `Y`; the PLL's
  mirror-frequency coefficient is a diagnostic (`y_mirror`).
- Above `f_s / 2` every control term is zero (masked by frequency, not by a branch).
- Resonators must have a positive bandwidth: an ideal resonator is infinite exactly at a
  harmonic grid point.
- Masks select with `torch.where` (a multiply keeps NaN from an unused branch); a bridge that
  does not switch gets a placeholder DC-link voltage and a zero modulation index.
- `torch.polar` only with a nonnegative magnitude: its backward has the wrong sign for a
  negative one (torch 2.13). Signed amplitudes multiply a unit phasor.
- Overmodulation (`M > 1` on a switching bridge) raises `ModelingError`; the check reads
  values off the tape.
- MPPT lower sidebands below 0 Hz are folded (conjugate phasor); lines sharing a frequency
  are summed by `_place_lines`, never merged earlier.
- No schema coupling: models are per-solve inputs keyed by device id.

## Tests

`tests/inverter/test_components.py` (closed forms per block, published values),
`tests/inverter/test_emission_flow.py` (study), shared builders `tests/inverter/grids.py`;
gates `tests/differentiability/test_inverter_emission_gradcheck.py`,
`tests/gpu/test_inverter_emission_parity.py`.
