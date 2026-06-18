# Concepts

This page explains the core modelling conventions used throughout `pgml`.

## Phase-domain representation

`pgml` uses a **phase-domain, fully asymmetric** representation.  Every element
is described by its per-phase primitive admittance — an *n_phase × n_phase*
complex stamp per harmonic.  There is no implicit sequence-domain conversion
inside the core; sequence data (Z1/Z0, short-circuit input conventions) are
handled by converters that emit phase-domain objects.

Compact **node-phase indexing** (`pgml.assembly.node_phase_index`) assigns one
matrix row to each existing *(node, phase)* pair rather than a padded A/B/C/N
grid, so the assembled Y-bus has minimal size.

## SI base units and stored quantities

All physical fields are stored in SI units:

| Quantity | Unit | Storage convention |
|----------|------|--------------------|
| Resistance | Ω | `r_ohm` |
| Inductance | H | `l_h` (not reactance X) |
| Capacitance | F | `c_f` (not susceptance B) |
| Voltage | V | phasors as `(v_re, v_im)` |
| Current | A | phasors as `(i_re, i_im)` |
| Power | W / VAr | `p_w`, `q_var` |

Reactive elements store **L and C**, never X or B.  At assembly time:

$$
X(h) = 2\pi h f_0 L, \quad B(h) = 2\pi h f_0 C
$$

This keeps frequency scaling physical and gradients attached to physical
parameters.

## Residual-form equation registry

The physics equations live in `pgml.equations` as a residual-form registry:

$$
0 = a - b
$$

Equations are stored as SymPy expressions and compiled to both torch and numpy
evaluators.  This separation keeps the physics symbolic and the compute layer
replaceable.

## Pi-form branch model

Every branch (line or transformer) contributes a **pi-form primitive admittance
stamp** to the Y-bus:

- Series admittance: `y_s = 1 / (R + jωL)` with optional skin-effect resistance
- Shunt admittance: `y_sh = G + jωC` (half at each end for the pi model)
- Transformers add a **complex tap** `t = |t| e^{jφ}` so that the branch
  admittance becomes `t* y_s t` on the from-side.

## Harmonic power flow

Harmonic power flow **decouples per harmonic** into the linear solve:

$$
Y(h) \, V(h) = I(h)
$$

A linear solve has a clean, cheap adjoint so end-to-end gradients flow without
differentiating Newton iterations.  Sources inject complex harmonic currents
(Norton representation); loads are a current source in parallel with a
frequency-dependent shunt admittance.

For the fundamental frequency, a nonlinear const-P / ZIP fixed-point iteration
is available (`pgml.solver.solve_power_flow`), with backward pass via the
implicit function theorem (IFT adjoint in real coordinates).

## Float/tensor duality

Physical schema fields accept **either** plain Python floats/lists or any
array-like object (e.g. `torch.Tensor`).  Array-likes are passed through
untouched so autograd gradients flow:

```
grid parameters (tensors) → assemble_ybus → solve → outputs
```

The schema imports no compute framework; "array-like" is detected by duck typing
(`hasattr(v, "detach")` or `hasattr(v, "__array__")`).

## Symmetric vs asymmetric calculation

`pgml` is always phase-domain, so "symmetric" and "asymmetric" refer to how
**appliance power** is resolved across phases — not to the network solve itself.

The `symmetry` argument accepted by `assemble_ybus`, `device_current_injections`,
`solve_power_flow`, and `solve_harmonic_flow` controls this:

| Mode | Behaviour |
|------|-----------|
| `"symmetric"` | Total P/Q split equally over the connected phases (balanced). |
| `"asymmetric"` | Per-phase nameplate values or per-phase operating-point keys are honoured. |
| `"auto"` | Asymmetric iff any appliance or operating point carries per-phase data; otherwise symmetric. |
| `None` | Reads `calculation.symmetry` from the config (default `"auto"`). |

This mirrors power-grid-model's `symmetric=True/False` rule.  A single INFO
modeling summary is logged per top-level call showing the resolved mode.

## WYE/DELTA connection and neutral modeling

A `Load` or `Generator` can set `connection` explicitly
(`WindingConnection.WYE`, `WYE_GROUNDED`, or `DELTA`).  When `connection` is
`None`, the config defaults `appliance.load.default_connection` (multi-phase)
and `appliance.load.single_phase_connection` (1-phase) are used (both `"wye"`
by default).

`pgml` uses a **terminal incidence matrix** `M` to map node-phase voltages to
element (terminal) voltages:

- **WYE, no neutral row** (`Phase.N` absent from the node): `M = I_n`.
  Reduces exactly to the historical diagonal per-phase stamp.
- **WYE, with neutral** (`Phase.N` present, 4-wire network): used rows are the
  phase rows plus the neutral row; `M = [I_n | -1]`.  The neutral row receives
  the return current automatically (Kirchhoff).
- **DELTA-3**: used rows are the three phase rows; `M` is the circulant
  difference matrix.  The base voltage is line-to-line.

DELTA and 4-wire WYE connections are fully modelled in the harmonic solver
(connection-aware injection via the incidence matrix `M`); see the
"Per-phase / connection-aware harmonic injection" section below.
`NotImplementedError` is only raised for `include_load_shunt=True` (Norton
shunt from `HarmonicShuntModel`) and for DELTA with fewer than two phases,
neither of which is a topology restriction.

## Per-phase / connection-aware harmonic injection

Harmonic current sources are modelled per *element* (terminal) using the same
incidence matrix `M` that the fundamental power flow uses, so the correct
terminal voltage appears in the normalisation:

- **WYE, no neutral**: `M = I` — per-phase injection reduces exactly to the
  historical diagonal form.
- **WYE, 4-wire with neutral** (`Phase.N` present): `M = [I | -1]` — the
  terminal voltage is `V_phase - V_N`; the neutral row receives the return
  current automatically.
- **DELTA-3**: `M` is the circulant difference matrix — the terminal voltage
  is the relevant line-to-line voltage.

Both connections were guarded by `NotImplementedError` in Increment 1; that
guard is removed as of Increment 2.

### Symmetric vs per-phase spectra

| Field | Behaviour |
|-------|-----------|
| `spectrum` on `Load` / `Generator` | One `Spectrum` broadcast to every connected phase / delta branch (OpenDSS multi-phase semantics). |
| `spectrum_per_phase` on `Load` / `Generator` | `dict[Phase, Spectrum]` — each key maps a connected phase (for WYE) or the delta-branch starting phase (for DELTA) to its own `Spectrum`. Keys must be a subset of `phases`; a phase with no entry injects no harmonics. |

The two fields are **mutually exclusive**.  Set one or the other, never both.

Example — single-phase EV charger distorting only phase A on a three-phase load:

```python
from pgml.schemas.grid_schema import Load, Phase, StaticSpectrum, SpectrumPoint

load = Load(
    id="ev_load",
    node_id="bus_1",
    phases=(Phase.A, Phase.B, Phase.C),
    p_nom_w=7400.0,
    q_nom_var=0.0,
    spectrum_per_phase={
        Phase.A: StaticSpectrum(spectrum=SpectrumPoint(
            components=[{"order": 5, "magnitude_pu": 0.08, "phase_deg": 0.0},
                        {"order": 7, "magnitude_pu": 0.05, "phase_deg": 0.0}]
        )),
        # Phase.B and Phase.C are absent → no harmonic injection
    },
)
```

### Runtime override convention (`harmonic_injection`)

`solve_harmonic_flow` accepts a `harmonic_injection` dict that overrides
stored spectra for scenario-level or differentiable variation:

```
{appliance_id: {order: (magnitude_pu, phase_deg)}}
```

Each `magnitude_pu` / `phase_deg` value follows an **unambiguous** convention
that avoids silent misreads when the scenario batch size happens to equal the
element count:

- A Python **`list` or `tuple`** is always **per-element** — its length *must*
  equal `n_elem` (number of connected phases or delta branches).  Each entry
  may be a Python float or a `[*batch]` tensor (gradients are preserved).
- A **scalar, 0-d tensor, or `[*batch]` tensor** (no element axis) is
  **broadcast** identically to every element.  This is the backward-compatible
  path for symmetric overrides.

The override takes precedence over any stored `spectrum` or
`spectrum_per_phase`.

## Validation philosophy

`pgml` validates against two reference implementations:

- **OpenDSS** — harmonic ground truth.  We compare the assembled Y-bus (exported
  system Y) and per-node voltage phasors at each harmonic.
- **pandapower / power-grid-model** — fundamental-frequency load-flow oracles.

Tests run on small IEEE feeders (IEEE 33-bus) and the CIGRE LV network.
