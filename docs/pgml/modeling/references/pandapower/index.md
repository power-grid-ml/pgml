# Reference brief: pandapower

Distilled context; consult installed package (`pip install pandapower`). This is
our PRIMARY load-flow result oracle AND a Y-bus oracle (it exposes the internal
bus admittance matrix). Not differentiable. Sequence-domain at heart.

## Data model
- One pandas DataFrame per element type: net.bus, net.line, net.trafo, net.load,
  net.gen/sgen, net.ext_grid, net.shunt. Result tables net.res_*.
- Engineering units (kV, MW, MVA, ohm/km, nF/km). Convert to our SI on import.
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

## Ground-truth extraction (for tests)
```python
import pandapower as pp
pp.runpp(net)                                   # AC power flow
res_bus = net.res_bus                            # voltages (vm_pu, va_degree)
res_line = net.res_line                          # flows + loading
Ybus = net._ppc["internal"]["Ybus"]              # scipy sparse, PER-UNIT on ppc base
bus_lookup = net._pd2ppc_lookups["bus"]          # net.bus index -> ppc bus index
```
CAUTION: Ybus is per-unit on the ppc S/V base — convert to SI (or convert ours to
the same pu base) before comparing to our SI Y. Feasibility helpers:
`pp.violated_buses(net, min_vm_pu, max_vm_pu)`, `pp.overloaded_lines(net, max_load)`.
