# Reference brief: OpenDSS (EPRI) — the HARMONIC ground truth

Distilled context; consult OpenDSSDirect.py / dss-python (`pip install
opendssdirect.py`). OpenDSS is the established harmonic engine and our Y-matrix +
harmonic-flow oracle. Phase-domain, full per-phase. Not differentiable.

## Key modelling facts (harmonic-relevant)
- Vsource: sequence input (Z1/Z2/Z0, or R1/X1/R0/X0, or MVAsc3/MVAsc1 + x1r1/x0r0)
  -> internal 3x3 Yprim in siemens. Z2 may differ from Z1.
- Transformer: per-winding (conn, kV, kVA, %R), inter-winding %XHL; %loadloss (Cu),
  %noloadloss (Fe), %imag (magnetizing). LeadLag sets the clock/vector group.
  XRConst {Yes|No}=No default: R fixed, X scales with h (X/R grows). We mirror via
  Transformer.harmonic_xr_constant.
- Load harmonic model = Norton: current source from SPECTRUM in parallel with a
  shunt admittance split series/parallel R-L by %SeriesRL (default 50/50);
  NeglectLoadY=yes -> pure current source; motor branch via puXharm + XRharm(=6).
- Line: symmetric components (R1/X1/R0/X0/C1/C0) OR Rmatrix/Xmatrix/Cmatrix OR
  geometry (Carson). Matrices take precedence.

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
