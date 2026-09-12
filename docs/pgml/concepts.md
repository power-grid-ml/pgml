# Concepts

The model behind every `pgml` result, in the order it matters to a user. The modelling
decisions and their limits are recorded in {doc}`modeling/index`.

## One linear system per harmonic order

Steady-state harmonic power flow decouples by order. For each integer order `h` the engine
assembles a complex nodal admittance matrix and solves

$$
Y(h)\,V(h) = I(h)
$$

for the node voltages. Branch currents, terminal powers and spectra follow from `V(h)`.

Harmonic sources are current injections in the Norton sense. A distorting device is a current
source per order in parallel with a frequency-dependent shunt admittance derived from its
fundamental operating point, which is the main damping of a feeder-end parallel resonance.
`load_shunt` selects the shunt model, and a generation-sign device carries none by default.
At harmonic orders the voltage source behind the slack is a short circuit and contributes only
its own shunt. {doc}`modeling/references/opendss/harmonics` gives the admittance in full.

At the fundamental the load flow is nonlinear, because a constant-power load is not a fixed
admittance. `solve_power_flow` runs a current-injection fixed point or a Newton iteration,
both warm-started from a constant-impedance solve. The backward pass uses the implicit
function theorem in real coordinates, so the gradient does not unroll the iteration.

## Phase domain, never sequence domain

Every element is described by its own per-phase primitive admittance, an `n × n` complex
stamp per order. The core has no sequence-domain step. Sequence data such as `Z1` and `Z0`
is decomposed into phase quantities by the converters before assembly.

Rows are indexed compactly. `pgml.assembly.node_phase_index` gives one matrix row per
existing `(node, phase)` pair, not a padded A/B/C/N block, so a mixed one-phase and
three-phase network assembles without empty rows.

## SI units, and L and C rather than X and B

| Quantity | Unit | Field |
|---|---|---|
| Resistance | Ω | `r_ohm` |
| Inductance | H | `l_h` |
| Capacitance | F | `c_f` |
| Voltage | V | phasor as `(v_re, v_im)` |
| Current | A | phasor as `(i_re, i_im)` |
| Power | W, var | `p_w`, `q_var` |

Reactive elements store inductance and capacitance. Reactance and susceptance are derived at
the frequency of each order, `X(h) = 2π h f₀ L` and `B(h) = 2π h f₀ C`. Frequency scaling
stays physical that way, and a gradient lands on a physical parameter rather than on a
frequency-specific one.

`Node.u_rated_v` is the line-to-line nameplate for a node with three or more phases, and the
line-to-neutral value for a one-phase node. Every per-phase voltage inside the solver is
line-to-neutral. {doc}`modeling/conventions` pins the full convention set and compares it
with the reference tools.

## Branches are pi stamps

Every branch contributes a pi-form primitive admittance.

- Series admittance `y_s = 1 / (R + jωL)`, with an optional frequency-dependent resistance
  law.
- Shunt admittance `y_sh = G + jωC`, half at each end.
- A transformer adds a complex tap `t` and a winding incidence that carries the vector
  group, so a Dyn delta traps triplen harmonics the way the real unit does.

A line can instead carry `conductor_geometry`. Assembly then computes the impedance from
conductor coordinates with the Carson/Deri earth-return and skin-effect model, per order.
Lines defined by lumped R/L/C use one of the analytic harmonic line models instead, selected
by the typed field `Line.harmonic_line_model`. A converted grid arrives with the configured
default already resolved into that field, so a three-phase R/X line uses the sequence-aware
model rather than plain proportional scaling.
{doc}`modeling/harmonic-line-model` explains the difference and when it matters.

## Floats or tensors, same grid

Physical schema fields accept a plain Python float, a nested list, or any array-like object
such as a `torch.Tensor`. Array-likes pass through untouched, so the gradient chain

```text
grid parameters -> assemble_ybus -> solve -> outputs
```

closes without a second parameter container. The schema imports no compute framework and
detects an array-like by duck typing. {doc}`differentiability` shows what this is for.

## Symmetric and asymmetric power

The network solve is always per phase. Symmetry refers to how an appliance's power is
resolved across its phases.

| Mode | Behaviour |
|---|---|
| `"symmetric"` | Total P and Q split equally over the connected phases |
| `"asymmetric"` | Per-phase nameplate values and per-phase operating points are honoured |
| `"auto"` | Asymmetric when any appliance or operating point carries per-phase data |
| `None` | Take `calculation.symmetry` from the config, default `"auto"` |

The argument is accepted by `assemble_ybus`, `device_current_injections`,
`solve_power_flow` and `solve_harmonic_flow`. The resolved mode is logged once per top-level
call. power-grid-model's `symmetric=True/False` switch follows the same rule.

## Connections and the neutral

A `Load` or `Generator` can set `connection` to WYE, grounded WYE or DELTA. With
`connection` left unset, the config defaults apply, separately for multi-phase and
single-phase elements.

Terminal voltages come from an incidence matrix `M` that maps node-phase voltages to element
terminals.

- WYE without a neutral row reduces to the diagonal per-phase stamp, `M = I`.
- WYE on a four-wire node uses the phase rows plus the neutral row, `M = [I | -1]`. The
  neutral row receives the return current by construction.
- DELTA uses the circulant difference matrix, so the base voltage is line-to-line.

Each injection appliance may override the node-level neutral-or-ground choice through
`return_path`. {doc}`modeling/asymmetric` covers the per-phase model in full.

## Harmonic injection per phase

The same incidence matrix drives harmonic injection, so the normalisation always uses the
correct terminal voltage. The device Norton shunt is connection-aware in the same way: its
element admittance is mapped through the same incidence `M`, so a four-wire WYE device returns
its shunt current through the neutral row and a DELTA device puts each leg admittance on both
of its phase diagonals. A device carries either one `spectrum`, broadcast to every
connected phase or delta branch, or a `spectrum_per_phase` mapping. The two fields are
mutually exclusive. A phase absent from `spectrum_per_phase` injects nothing.

```python
from pgml.schemas.grid_schema import Load, Phase, SpectrumPoint, StaticSpectrum

load = Load(
    id=7,
    node=12,
    phases=(Phase.A, Phase.B, Phase.C),
    p_nom_w=7400.0,
    q_nom_var=0.0,
    spectrum_per_phase={
        Phase.A: StaticSpectrum(spectrum=SpectrumPoint(
            components=[{"order": 5, "magnitude_pu": 0.08, "phase_deg": 0.0},
                        {"order": 7, "magnitude_pu": 0.05, "phase_deg": 0.0}]
        )),
    },
)
```

`solve_harmonic_flow` also takes a `harmonic_injection` override of the form
`{appliance_id: {order: (magnitude_pu, phase_deg)}}`, which wins over any stored spectrum.
The shape rule removes the ambiguity that appears when a batch size happens to equal the
element count. A list or tuple is always per element and its length must equal the number of
connected phases or delta branches. A scalar, a 0-d tensor or a tensor with only batch
dimensions is broadcast to every element. Entries may be tensors, and gradients flow through
them.

### Sequence structure across a device's phases

A three-phase device's harmonic on phase B is its phase-A waveform delayed by a third of a
cycle, so order `h` is rotated by `−h·120°`. Triplen orders therefore come out as zero
sequence, which is what a Dyn transformer traps. Order 5 comes out as negative sequence,
order 7 as positive sequence, and so on. None of this is drawn per phase. It follows from how
the per-element angle is formed,

```text
|I_h^e|    = (mag_h / mag_1) * |I_1^e|
arg I_h^e  = ang_h + h * (arg I_1^e - ang_1)
```

per terminal element `e`, with `I_1^e` the fundamental current that element actually carries.
A device-level spectrum broadcasts to every element of the device, and the `h·arg I_1^e` term
supplies the rotation from the fundamental's own phase angles. An unbalanced fundamental
therefore propagates into the per-phase harmonic magnitudes of one device.

## Integer orders only

Orders are integer multiples of the fundamental. A non-integer order is rejected rather than
approximated, because the spectra and the per-order assembly are defined for integer
multiples. Interharmonics, flicker and transients need a time-domain model and are out of
scope.
