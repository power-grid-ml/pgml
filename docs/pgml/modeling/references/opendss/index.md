# Reference brief: OpenDSS (EPRI) — the HARMONIC ground truth

Distilled context; consult OpenDSSDirect.py / dss-python (`pixi add --pypi opendssdirect.py`). OpenDSS is the established harmonic engine and our Y-matrix +
harmonic-flow oracle. Phase-domain, full per-phase. Not differentiable.

## Key modelling facts (harmonic-relevant)
- Vsource: sequence input (Z1/Z2/Z0, or R1/X1/R0/X0, or MVAsc3/MVAsc1 + x1r1/x0r0)
  -> internal 3x3 Yprim in siemens. Z2 may differ from Z1.
  **`basekv` phase-count semantics (verified empirically, not fully documented by
  OpenDSS's own docs, which describe `basekv` as line-to-line unconditionally):** for a
  `phases>=3` Vsource `basekv` genuinely is the L-L nominal (the solved line-to-neutral
  EMF is `basekv/sqrt(3)`); for a `phases=1` Vsource OpenDSS uses `basekv` DIRECTLY,
  unscaled, as the solved single-conductor-pair EMF magnitude -- no `sqrt(3)` anywhere.
  `pgml.convert.opendss.to_grid`'s `u_ref_v = basekv*pu*1000` and (for the node voltage
  base) `u_rated_v = Bus.kVBase()*sqrt(3)*1000` need no phase-count branch because they
  simply reproduce whatever OpenDSS itself solves for at that phase count; see
  `tests/convert/test_opendss_vsource_basekv.py` and
  `docs/pgml/modeling/conventions.md` sec. 1/6.
- Transformer: per-winding (conn, kV, kVA, %R), inter-winding %XHL (documented "on the
  kVA base of winding 1" -- both windings must share one kVA rating for `to_grid`'s
  per-unit leakage recovery, see conventions.md sec. 2); %loadloss (Cu),
  %noloadloss (Fe), %imag (magnetizing). OpenDSS has no explicit clock parameter; the
  clock is derived from two mechanisms combined: `LeadLag` (`Lag` -> clock 1 baseline
  /shift 30, `Lead` -> clock 11 baseline/shift 330, for a Dy/Yd pairing; a matching
  Yy/Dd pairing baselines at clock 0) plus a CYCLIC rotation of the winding bus's
  phase-conductor order (`bus=lv.2.3.1.0` etc.), which reaches every clock of the
  pairing's correct parity EXCEPT the polarity-flip clocks `{2, 6, 10}` (no bus
  wiring can express a reversed winding). Verified against live solves; see
  `src/pgml/convert/opendss/CONTEXT.md`, "Vector group / clock".
  XRConst {Yes|No}=No default: R fixed, X scales with h (X/R grows). We mirror via
  Transformer.harmonic_xr_constant. `pgml.convert.opendss.to_grid` converts two-winding
  units only (solidly grounded wye or delta windings; 3-winding, `RegControl`
  regulators, tap-changer control and `XfmrCode`/frequency-correction curves raise/are
  not read) -- see `src/pgml/convert/opendss/CONTEXT.md` and
  `tests/reference/test_opendss_transformer.py`. A wye winding's grounding follows
  OpenDSS's own shorthand-bus rule: no explicit `(n_phases+1)`-th conductor, or an
  explicit `.0`, solidly grounds it; an explicit non-zero neutral node
  (floating/impedance-grounded) is out of scope and raises.
- Load harmonic model = Norton: current source from SPECTRUM in parallel with a
  shunt admittance split series/parallel R-L by %SeriesRL (default 50/50);
  NeglectLoadY=yes -> pure current source; motor branch via puXharm + XRharm(=6).
  `to_grid` also maps `Loads.Model()` (1/2/5/8) to the fundamental-flow
  `LoadModel`/`ZipCoefficients` (1=const-power kept as `(None, None)` for a
  byte-identical default; 2=const-Z; 5=const-current; 8=ZIPV custom coefficients);
  models 3/4/6/7 (asymmetric P-vs-Q voltage dependence) fall back to `CONST_POWER`
  with a warning naming the model. A WYE load/generator/PVSystem/Storage's
  return conductor -- OpenDSS's own resolved choice, read from
  `CktElement.NodeOrder()` -- carries over per-element into the schema's
  `InjectionAppliance.return_path` (`"ground"` when solidly grounded despite the
  bus also carrying an explicit neutral tie elsewhere, `"neutral"` for an explicit
  `.4`-style tie, `"auto"` otherwise), so two elements on the SAME four-wire bus can
  return differently, matching OpenDSS exactly instead of one shared node-level rule.
- Line: symmetric components (R1/X1/R0/X0/C1/C0) OR Rmatrix/Xmatrix/Cmatrix OR
  geometry (Carson). Matrices take precedence. `to_grid` reads the native n×n
  matrices directly (`THREE_PHASE`) and reduces a coupled multi-phase line to
  `Z1 = Z_self - Z_mutual` under `SINGLE_PHASE_EQUIV` (not the bare self entry); a
  phase-permuted terminal (`bus1=a.1.2.3 bus2=b.3.2.1`) carries an independent
  `to_phases`.
- Other converted elements: `Capacitor`/`Reactor` -> `ShuntAppliance`, WYE (solidly
  grounded, uncoupled -- a Reactor's series R+X converts to the equivalent shunt
  admittance `Y=1/(R+jX)`, exact at the fundamental only) OR DELTA (`conn=delta`
  -> `ShuntAppliance.connection=DELTA`, per-leg G/C read from OpenDSS's own resolved
  per-leg `Cuf`/`R`/`X`; an unbalanced per-leg value has no representation and is
  refused when exported back by the scenario oracle); `Generator`/
  `PVSystem`/`Storage` -> generation-positive / signed discharge-positive appliances
  (`PVSystem`/`Storage` read OpenDSS's PRESENT solved kW/kvar, already derated).
  Every other DSS element class (`Isource`, `Monitor`, `EnergyMeter`, `RegControl`,
  `CapControl`, `InvControl`, `StorageController`, `Relay`, `Recloser`, `Fuse`,
  `Sensor`, ...) is enumerated from `Circuit.AllElementNames()` and triggers one
  warning per class naming the kind and count -- nothing vanishes silently. See
  `src/pgml/convert/opendss/CONTEXT.md` for the full field-mapping tables.

## Ground-truth extraction (for tests)
```python
import opendssdirect as dss
dss.Text.Command("Redirect feeder.dss")
dss.Text.Command("Solve")
# System admittance matrix:
dss.Text.Command("Export Y")                 # writes CSV of system Y (G,B per entry)
node_order = dss.Circuit.YNodeOrder()        # node names defining Y row/col order
Y = dss.Circuit.SystemY()                     # dense Y as flat [G,B,...] -> reshape
# Harmonic flow:
dss.Text.Command("Solve mode=harmonics")     # solves at spectrum harmonics
V = dss.Circuit.AllBusVolts()                 # complex node voltages (interleaved re,im)
```
Compare our assembled Y(h=1) to OpenDSS `Export Y` (align by YNodeOrder, mind
node/phase ordering and SI units). For harmonics, compare AllBusVolts per order.
