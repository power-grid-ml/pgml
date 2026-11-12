# Loads and appliances: elements, connections and rated voltages

A {class}`~pgml.schemas.grid_schema.Load`, {class}`~pgml.schemas.grid_schema.Generator` or
{class}`~pgml.schemas.grid_schema.Storage` is a set of connection elements. Each element sits
between two terminals, draws a power and has a nominal voltage. This page states how the
elements follow from `phases` and `connection`, which voltage each element is referred to,
and how the fundamental and harmonic models act on an element. The wye/delta rules of the
reference tools are compared in {doc}`asymmetric`; inverter control laws are in
{doc}`der-pv-storage`.

## Connection elements

The elements are the rows of a constant incidence `M` from the node rows the appliance touches
to the element voltages, `V_elem = M·V_node`. A per-element admittance `y` is stamped as
`Mᵀ·diag(y)·M` and a per-element current `i` enters the nodes as `Mᵀ·i`.

| `connection` | `phases` | elements | `M` (columns: the phase rows, then `N` if used) |
|---|---|---|---|
| wye, node without `Phase.N` (or `return_path="ground"`) | 1, 2 or 3 | one per phase, phase to earth | `I` |
| wye, node with `Phase.N` | 1, 2 or 3 | one per phase, phase to neutral | `[I \| −1]` |
| delta | 3 | three, phase `k` to phase `k+1` | circulant difference |
| delta | 2 | **one**, `phases[0]` to `phases[1]` | `[1, −1]` |

A two-phase delta appliance is a phase-to-phase load: a 440 V appliance across X1-X2 of a
split-phase node, a 400 V welder across A-B of a three-phase node, or a single-phase MV load
across two phase conductors. It is OpenDSS's `Load phases=1 conn=delta` on two conductors,
which OpenDSS's neutral rules describe as a one-element line-to-line load. A delta appliance
on one phase is rejected by the schema, one with `Phase.N` among its phases or on four phases
by the assembly ({class}`~pgml.errors.ModelingError`).

An appliance's power fields are totals over its phases, and the optional per-phase fields
split them. Elements and phases coincide for wye and three-phase delta. The single element of
a two-phase delta appliance carries the sum of the per-phase values, whether they come from
the nameplate split, a per-phase operating point (`p_per_phase_w`) or the equal split of a
symmetric calculation, so all three give the same element. A differentiable
`param_overrides` entry `("load", id, "p_nom_per_phase_w")` replaces the per-ELEMENT vector,
which has one entry for such an appliance. Inverter control quantities are device totals
shared equally over the elements, so a two-phase delta inverter evaluates its curve at the
phase-to-phase voltage against the full rating.

## Rated voltage of an element

The load models refer to a nominal element voltage `V0`, taken from the host node
({func}`pgml.assembly.node_voltage_magnitude`; {func}`pgml.assembly.phase_voltage_magnitude`
computes it from the two rating fields):

- a wye element (and every per-phase base, see below) uses the node's phase-to-neutral
  rating: `Node.u_rated_ln_v` when it is set, otherwise `u_rated_v/√3` on a node with three or
  more phase rows and `u_rated_v` on a node with fewer;
- a delta element, two- or three-phase, uses `Node.u_rated_v`, the phase-to-phase rating.

On a node with three or more phase rows (a `Phase.N` row counts) `u_rated_v` is line-to-line,
so a delta element is rated correctly without further data. On a node with fewer, such as
`(A, B)`, `u_rated_v` is line-to-neutral unless the node sets `u_rated_ln_v`, and a delta
element would be rated at the line-to-neutral voltage. The assembly therefore refuses a delta
appliance (load, generator or storage, by its explicit or its configured connection) on such a
node without `u_rated_ln_v`, with an {class}`~pgml.errors.InputError` that names the node and
the appliance. Rate the node with both fields, e.g. `u_rated_v = 400`, `u_rated_ln_v = 230`,
or connect the appliance wye. A delta {class}`~pgml.schemas.grid_schema.ShuntAppliance` is
given by its admittances and needs no rating.

The same phase-to-neutral rating is the per-unit base of every row of the node
({func}`pgml.assembly.base_voltage_per_row`), the reference of the convergence test, of a
voltage-regulating generator's setpoint and of a node harmonic source.

`u_rated_ln_v` exists for the nodes where `u_rated_v/√3` is wrong. A split-phase node behind
a centre-tapped transformer has rows `(X1, X2, N)`, stored as `(A, B, N)`; X1-N and X2-N are
half of X1-X2, not `1/√3` of it. Rate such a node `u_rated_v = 440`, `u_rated_ln_v = 220`
(or 240 and 120): the two half loads are then referred to 220 V and the X1-X2 load to 440 V.
On a node with two phase conductors that gives `u_rated_ln_v`, `u_rated_v` is the
phase-to-phase voltage. When both are plain numbers the schema checks them against each other
within 2 %, by the number of phase conductors: `√3` for three, `√3` or `2` for two, `1` for
one. Nameplates such as 400/230 V pass as `√3`. Without the field every node keeps the rule it
had before, so existing grids assemble and solve to identical numbers (checked as exact
equality of the Y-bus, the per-unit bases and the fundamental and harmonic voltages on twelve
grids: IEEE 33-bus, CIGRE LV in both phase modes, a synthetic MV feeder, four-wire
hand-built grids in every load model, the centre-tap and Dyn earthing circuits).

## Load models on an element

With `S0 = sign·(P + jQ)` the element's operating power (`sign = +1` for a load, `−1` for
generation) and `r = |V_elem|/V0`, the fundamental model draws

```
S(V) = S0 · (z·r² + i·r + p)        I_elem = conj(S(V)) / conj(V_elem)
```

with `(z, i, p)` per power component: `(0, 0, 1)` constant power, `(1, 0, 0)` constant
impedance, `(0, 1, 0)` constant current, or the `ZipCoefficients`. The linear assembly folds
the constant-impedance part as `y = conj(S0)/V0²`, flat in conductance and scaled with the
order in susceptance (see {doc}`conventions`). For a two-phase delta element the nodal current
is `+I_elem` on `phases[0]` and `−I_elem` on `phases[1]`.

At a harmonic order `h` each element is a current source scaled from its own fundamental
current, `|I_h| = (mag_h/mag_1)·|I_elem|`, `arg I_h = ang_h + h·(arg I_elem − ang_1)`, and,
unless `load_shunt="none"`, carries the harmonic load shunt `conj(S)/V0²` in OpenDSS's
`%SeriesRL` form. Both read one element power, the power the element really draws at the
converged fundamental. A per-phase spectrum keys the element of a delta appliance by its first
phase: element `k` uses `spectrum_per_phase[phases[k]]`, so a two-phase delta appliance uses
the entry of `phases[0]`.

## Validation

Against live OpenDSS (opendssdirect 0.9.4, `Tolerance=1e-12`, complex128, CPU), in pu of each
row's phase-to-neutral voltage (`tests/reference/test_opendss_delta_appliances.py`):

- a 440 V load across X1-X2 of the 50 kVA centre-tapped unit, and an A-B load at 400 V on the
  four-wire Dyn1 feeder, each in all four load models: 3.4e-10 pu at the fundamental (the two
  tools' fundamental iteration, the same in every case), at most 9.2e-12 pu at order 3, and
  the transformer terminal and earth currents within 2.9e-8 A;
- the harmonic load shunt of the phase-to-phase element (`%SeriesRL=50` against
  `load_shunt="opendss"`): 3.1e-12 pu at order 3;
- constant-impedance half loads on the split-phase node with `u_rated_ln_v = 220`: the same
  agreement; without the field they deviate by 3.3e-3 pu at the fundamental.

The OpenDSS scenario exporter writes a two-phase delta appliance as that one-element `Load`
and refers wye loads to the node's `u_rated_ln_v`; on a four-wire feeder with a two-phase
delta constant-current load and a constant-impedance wye load on a 400/230 V node, pgml and
the exported circuit agree to 5.1e-10 (fundamental) and 5.0e-9 (orders 3 and 5) relative. The
float64 gradcheck covers the element power through the nonlinear solve, the harmonic
injection and shunt, and a tensor `u_rated_ln_v`
(`tests/differentiability/test_two_phase_delta_gradcheck.py`).

## Not modelled

- An open delta (two elements on three conductors, OpenDSS `Load phases=2 conn=delta`). Enter
  it as two two-phase delta appliances; the OpenDSS reader refuses it.
- The OpenDSS reader still converts a `phases=1 conn=delta` load as a wye load and an
  imported split-phase bus gets no `u_rated_ln_v`; see {doc}`limitations`.
- A voltage-regulating generator on a delta connection, and OpenDSS's internal-voltage
  harmonic reference (`spectrum_reference="opendss_voltage"`) on a two-phase delta device,
  raise.
