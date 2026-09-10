# OpenDSS

OpenDSS is the harmonic reference, and the reader for DSS files is built against it. It solves
in the phase domain, per phase, and is not differentiable. The Python bindings are
`opendssdirect.py`.

## Facts that matter

Voltage source. Sequence input, `Z1/Z2/Z0` or `R1/X1/R0/X0` or short-circuit powers with an
X/R ratio, becomes an internal 3×3 primitive admittance in siemens, and `Z2` may differ from
`Z1`. The phase-count semantics of `basekv` are worth stating, because the OpenDSS
documentation describes it as line-to-line unconditionally. For a source with three or more
phases it is the line-to-line nominal, so the solved line-to-neutral EMF is `basekv/√3`. For a
single-phase source OpenDSS uses `basekv` directly and unscaled as the solved
single-conductor-pair EMF. The reader therefore needs no phase-count branch, because
`u_ref_v = basekv·pu·1000` and `u_rated_v = kVBase()·√3·1000` reproduce whatever OpenDSS
itself solves for.

Transformer. Per-winding connection, kV, kVA and `%R`, plus the inter-winding `%XHL`,
`%loadloss`, `%noloadloss` and `%imag`. Both windings must share one kVA rating for the
per-unit leakage recovery to be well defined. There is no explicit clock parameter. The clock
comes from `LeadLag` combined with a cyclic rotation of a winding bus's phase-conductor order,
which reaches every clock of the pairing's parity except the polarity-flip clocks 2, 6 and 10,
since no bus wiring expresses a reversed winding. `XRConst=No`, the default, keeps R fixed
while X scales with the order. The reader converts two-winding units with solidly grounded wye
or delta windings. Three-winding units, regulators, tap-changer control and
frequency-correction curves are not read, and an explicit non-zero neutral node raises. A wye
winding's grounding follows OpenDSS's own shorthand-bus rule, where no extra conductor, or an
explicit `.0`, means solidly grounded.

Load. The harmonic model is a Norton equivalent, a current source from the spectrum in
parallel with a shunt admittance split between a series and a parallel R-L branch by
`%SeriesRL`, with a motor branch parameterised by `puXharm` and `XRharm`. `NeglectLoadY=yes`
reduces it to a pure current source. The reader maps load models 1, 2, 5 and 8 onto the
fundamental load model and its ZIP coefficients, and falls back to constant power with a
warning for models 3, 4, 6 and 7. Each element's own resolved return conductor carries over
per element, so two elements on one four-wire bus can return differently, as they do in
OpenDSS.

Line. Symmetric components, explicit matrices, or a conductor geometry. Matrices and geometry
take precedence over sequence data. The reader takes the native matrices directly in
three-phase mode and reduces a coupled multi-phase line to `Z1 = Z_self − Z_mutual` in
single-phase-equivalent mode. A phase-permuted terminal carries its own phase tuple.

Other elements. `Capacitor` and `Reactor` become shunt appliances, grounded wye or delta, with
per-leg values read from OpenDSS's own resolved numbers. A reactor's series R and X convert to
the equivalent shunt admittance, which is exact at the fundamental only. `Generator`,
`PVSystem` and `Storage` become injections, and the latter two are read at their present
solved power, already derated. Every other element class is enumerated and raises one warning
per class naming the kind and the count, so nothing disappears quietly.

## Extracting ground truth

```python
import opendssdirect as dss

dss.Text.Command("Redirect feeder.dss")
dss.Text.Command("Solve")
dss.Text.Command("Export Y")            # CSV of the system admittance
node_order = dss.Circuit.YNodeOrder()   # the row and column order of Y
Y = dss.Circuit.SystemY()               # dense Y, flat [G, B, ...]

dss.Text.Command("Solve mode=harmonics")
V = dss.Circuit.AllBusVolts()           # complex node voltages, interleaved
```

Align the assembled fundamental admittance against `Export Y` by `YNodeOrder`, minding the
node and phase ordering and the units. For harmonics compare `AllBusVolts` per order.
