# pandapower

The primary load-flow reference, and an admittance-matrix reference as well, since it exposes
its internal bus admittance. Sequence domain at heart, and not differentiable.

## Data model
- One pandas DataFrame per element type: net.bus, net.line, net.trafo, net.load,
  net.gen/sgen, net.ext_grid, net.shunt. Result tables net.res_*.
- Engineering units (kV, MW, MVA, ohm/km, nF/km). The reader converts them to SI.
- Transformer: positive-seq vk_percent, vkr_percent, pfe_kw, i0_percent; zero-seq
  vk0_percent, vkr0_percent, mag0_percent, mag0_rx, si0_hv_partial; vector_group +
  shift_degree; tap_* ; trafo_model in {"t","pi"} (default "t").
- ext_grid (source): s_sc_max_mva, rx_max, r0x0_max, x0x_max, vm_pu, va_degree.
- Line: r_ohm_per_km, x_ohm_per_km, c_nf_per_km, g_us_per_km (+ r0/x0/c0), length_km.

## Output
- res_bus: vm_pu, va_degree, p_mw, q_mvar.
- res_line: p_from_mw, q_from_mvar, p_to_mw, q_to_mvar, i_from_ka, i_to_ka, i_ka,
  loading_percent, pl_mw, ql_mvar.
- res_trafo: loading_percent, p_hv_mw/p_lv_mw, etc.

## What the reader takes
- `gen` is pandapower's PV bus, and it converts to a voltage-regulating terminal by default.
- `shunt` converts to a fixed WYE shunt appliance. An inductive row becomes a negative
  capacitance, exact at the fundamental only.
- `ext_grid`'s `x0x_max` and `r0x0_max` are read into the source's zero-sequence impedance.
- `trafo`'s `vk0_percent` and `vkr0_percent` are read into the transformer's zero-sequence
  leakage. `mag0_percent`, `mag0_rx` and `si0_hv_partial` are not modelled and are named in a
  warning.

## Extracting ground truth
```python
import pandapower as pp
pp.runpp(net)                                   # AC power flow
res_bus = net.res_bus                            # voltages (vm_pu, va_degree)
res_line = net.res_line                          # flows + loading
Ybus = net._ppc["internal"]["Ybus"]              # scipy sparse, PER-UNIT on ppc base
bus_lookup = net._pd2ppc_lookups["bus"]          # net.bus index -> ppc bus index
```
The exported admittance is per unit on pandapower's own power and voltage base. Convert it to
SI, or convert the pgml matrix to the same base, before comparing. `pp.violated_buses` and
`pp.overloaded_lines` are useful for checking a converted network stays feasible.
