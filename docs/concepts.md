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

**Current limitations:** DELTA with `n != 3` phases and connection-aware
harmonic injection (DELTA or 4-wire WYE) raise `NotImplementedError` and are
deferred to a later increment.

## Validation philosophy

`pgml` validates against two reference implementations:

- **OpenDSS** — harmonic ground truth.  We compare the assembled Y-bus (exported
  system Y) and per-node voltage phasors at each harmonic.
- **pandapower / power-grid-model** — fundamental-frequency load-flow oracles.

Tests run on small IEEE feeders (IEEE 33-bus) and the CIGRE LV network.
