# Reference brief: asymmetric / per-phase load modeling across oracles

How power-grid-model (PGM), pandapower, and OpenDSS model (a) symmetric vs
asymmetric calculation, (b) wye/delta load connection, (c) single-phase loads,
(d) the neutral / earth return, and (e) per-phase harmonic injection — with
primary-source citations. This is the modeling basis for pgml's fully-asymmetric, per-phase representation. Companion
converter briefs: the [reference-library notes](references/index.md) for OpenDSS,
pandapower, and power-grid-model.

Verified against installed packages (`power_grid_model 1.13.94`, pandapower source
under the cpu pixi env) and the EPRI OpenDSS source/manual; citations inline.

## 1. Symmetric vs asymmetric calculation — trigger + resolution rule

**power-grid-model** — one model, a per-call flag:
`PowerGridModel.calculate_power_flow(symmetric: bool = True, ...)`. `symmetric=True`
(default) solves a positive-sequence single-phase equivalent; `symmetric=False`
solves the full abc system and every per-component output gains a trailing `(3,)`
phase axis (node `u` becomes line-to-neutral per phase). Symmetry is per-component
AND per-calculation:
- a `sym_load` in an asymmetric calc is **split equally** (P/3 per phase) — verified
  live: `p_specified=900 → p=[300,300,300]`;
- an `asym_load` in a symmetric calc is **aggregated** to the three-phase total
  (sum, = 3× the single-phase equivalent) — verified live: `[300,600,900] → 1800`.
Docs: <https://power-grid-model.readthedocs.io/en/stable/user_manual/calculations.html>,
`.../components.html#node`, `.../components.html#asymmetric-load-generator`.

**pandapower** — a separate entry point `pandapower.pf.runpp_3ph.runpp_3ph(net, ...)`
solving in the **sequence frame** (positive seq by Newton-Raphson; zero/neg seq by
current injection), with earth return. A symmetric `net.load`/`net.sgen` is split
`P/3` per phase unconditionally inside `runpp_3ph` (`_get_elements`/`_load_mapping`,
`pandapower/pf/runpp_3ph.py`); per-phase results in `net.res_bus_3ph`
(`vm_a_pu…vm_c_pu`, `va_a_degree…`) and `net.res_line_3ph`. Requires
`pp.add_zero_impedance_parameters(net)` first.
Docs: <https://pandapower.readthedocs.io/en/latest/powerflow/ac_3ph.html>.

**OpenDSS** — always native multi-phase/unbalanced phase-domain (our harmonic
oracle). There is no symmetric/asymmetric switch; you model the actual phases.
`Load.kW`/`kvar` are the **total**, divided equally by `phases` internally; genuine
per-phase imbalance is expressed by separate 1-phase Load objects.
Manual: <https://opendss.epri.com/Properties7.html>, `Load.pas` source.

**pgml adopted rule.** pgml is always phase-domain (like OpenDSS), so our
"symmetric vs asymmetric calculation" is purely an **operating-point resolution**
mode over each appliance's phases — it does not change the network solve:
- `SYMMETRIC`  → each appliance's total P/Q is split equally over its phases
  (today's behavior; matches a `sym_load` in PGM and `net.load` in `runpp_3ph`).
- `ASYMMETRIC` → honor per-phase `*_per_phase_*` / per-phase `operating_point`,
  falling back to equal split when a given appliance has none.
- `AUTO` (default) → ASYMMETRIC iff any appliance or operating-point carries
  per-phase asymmetric data, else SYMMETRIC. This realizes the requested config
  override: an asymmetric load distribution in the simulation config promotes the
  whole calculation to asymmetric even on a symmetrically-defined grid.
(Going asymmetric only manifests imbalance on genuinely multi-phase nodes; the
positive-sequence single-phase-equivalent grids from the pandapower/pgm converters
must first be expanded to abc nodes.)

## 2. Load connection: wye vs delta

**power-grid-model** — NO connection field on `(a)sym_load`/`(a)sym_gen`; all loads
are modeled wye, injecting at the node. The `WindingType` enum
(`wye=0, wye_n=1, delta=2, zigzag=3, zigzag_n=4`) applies only to transformer
`winding_from`/`winding_to`. (Verified: no `winding`/`connection` field in the
`asym_load` dtype.)

**pandapower** — `net.asymmetric_load.type ∈ {"wye","delta"}`. Wye uses
phase-to-earth voltage directly; delta converts node voltages to line-to-line via
`v_del_xfmn = [[1,-1,0],[0,1,-1],[-1,0,1]]`, computes the delta-loop currents, then
maps back to line currents with `i_del_xfmn = [[1,0,-1],[-1,1,0],[0,-1,1]]`
(`pandapower/pf/runpp_3ph.py`, `_load_mapping`). "PH-E load type is called wye since
Neutral and Earth are considered same" (docstring).

**OpenDSS** — `Load.conn ∈ {wye|LN, delta|LL}`, default `wye`. From `Load.pas`
`StickCurrInTerminalArray`: wye has `NConds=Nphases+1` and injects current between
phase conductor `i` and the neutral conductor (`NConds`); delta has
`NConds=Nphases` and injects from conductor `i` into `i+1` (wrapping `n→1`). VBase
differs: wye (≥2-phase) uses `kVLL/√3`, delta and 1-phase-wye use the supplied kV
directly. Manual: <https://opendss.epri.com/Properties7.html>, NeutralRules
<https://opendss.epri.com/NeutralRules.html>.

**pgml adopted model.** A connection-aware terminal **incidence** `M`
(n_terminals × n_phase_rows) per appliance, exactly the union of the OpenDSS
stick-current rule and pandapower's delta transform:
- WYE: each phase terminal → its phase row, return to the node's `Phase.N` row if
  present else to ground (the matrix reduces to identity ⇒ today's diagonal stamp).
- DELTA: terminal k → (phase_k, phase_{k+1 mod n}); the delta loop ordering is the
  appliance's `phases` tuple order.
Const-Z shunt stamped as `Mᵀ·diag(y)·M`; ZIP current as `I_node = Mᵀ·i_term` with
`V_term = M·V_node`. ZIGZAG is transformer-only (rejected on loads). DELTA requires
`len(phases) ≥ 2`.

## 3. Single-phase loads

**power-grid-model** — no node-phase concept; a node is always a 3-phase busbar.
A phase-A-only load is `p_specified=[P,0,0]`, `q_specified=[Q,0,0]` (verified live:
current flows on phase A only, voltages unbalance correctly).

**pandapower** — no dedicated 1-phase element; zero the unused phases of an
`asymmetric_load` (`p_b_mw=p_c_mw=0`, etc.).

**OpenDSS** — `phases=1`, `bus1=Bus.k.0` (phase k to ground/neutral). The European
LV residential default is single-phase **wye L-N** with `kV` = the line-to-neutral
voltage directly (no √3). Use `phases=1` two-conductor, NOT `phases=2`, to route the
return through the neutral wire rather than ground (R. Dugan, OpenDSS forum).
Refs: <https://sourceforge.net/p/electricdss/discussion/861977/thread/ec99a52a13/>,
NEVTestCase/Loads.DSS.

**pgml adopted model.** A single-phase load is `phases=(X,)` with a `connection`;
the default connection for an unspecified single-phase load is set in the pgml
config (default = WYE / L-N, per the European-LV convention above). Its return
follows §2 (to `Phase.N` row if the node has one, else ground).

## 4. Neutral / earth return

None of the three oracles solve an explicit neutral node in the load flow:
- **PGM**: 3-wire; for `asym_line` the 4×4 (incl. neutral) series matrix is
  **Kron-reduced** to 3×3 before solving; otherwise zero-sequence params
  (`r0,x0,c0`) carry the earth return.
  (<https://power-grid-model.readthedocs.io/en/stable/user_manual/components.html#line>)
- **pandapower**: pure sequence domain; neutral current is the derived residual
  `i_n = i_a+i_b+i_c = 3·i_0`, not a solved conductor.
- **OpenDSS**: can keep an explicit neutral as a 4th conductor node, but the
  shorthand `bus1=Bus` grounds it to node 0 (excluded from the system Y), which
  shorts `Rneut/Xneut`. A floating/impedance neutral needs explicit nodes
  (`bus1=Bus.1.2.3.4`). (<https://opendss.epri.com/NeutralRules.html>)

**pgml adopted model.** Phase-domain like OpenDSS. The neutral is represented only
when a node explicitly carries `Phase.N` in its `phases` (then it is a solved row
and a WYE load returns into it — the genuine 4-wire case, Kron-reducible later if a
node-elimination is wanted). With no `Phase.N` row, WYE returns to ground and the
solve matches PGM/pandapower/OpenDSS-default exactly. No separate neutral solver.

**Per-appliance return-path override.** The node-level rule above is a *default*, not
a hard constraint: a 4-wire bus in the field routinely mixes a solidly-grounded element
with a neutral-returning one on the SAME `Phase.N`-carrying node (e.g. OpenDSS's
`bus1=b1.1.2.3` — three phase conductors, no fourth — next to `bus1=b1.1.2.3.4` — an
explicit neutral tie — on one shared bus). Every `InjectionAppliance` (`Load`,
`Generator`, `Storage`) carries `return_path` (`"auto"` / `"neutral"` / `"ground"`,
default `"auto"` = the node-level rule) so this per-element choice is representable
directly, rather than forcing every WYE appliance on a neutral-carrying node into the
same return. `"ground"` pins the return to true ground even though the node has a
`Phase.N` row; `"neutral"` requires the node to carry one (raises otherwise). A
grounded and a neutral-returning appliance on the same node therefore stamp with
*different* incidence matrices (`M = I_n` vs `M = [I_n | -1]`) even though they share a
node and a connection — assembly groups appliances by `(connection, phase count,
effective neutral return)`, with `return_path` folded into that key. The OpenDSS
converter (`pgml.convert.opendss.to_grid`) reproduces this per-element choice exactly
from each element's own resolved return conductor (`CktElement.NodeOrder()`), rather
than approximating every WYE element on a bus by one shared node-level rule. Meaningless
for DELTA (no neutral to return through); a non-`"auto"` value on a DELTA appliance
raises.

## 5. Per-phase harmonic injection (OpenDSS, our harmonic oracle)

A `Load` carries exactly one `Spectrum`; one complex multiplier `Mult =
Spectrum.GetMult(h)` is applied to **every phase** of a multi-phase load. Per-phase
asymmetry enters only through the per-phase fundamental phasors captured in
`InitHarmonics` (`HarmMag[i]`, `HarmAng[i]`): for phase `i`, `I_h^i = Mult·HarmMag[i]`
then rotated by `h·HarmAng[i]` — i.e. `|I_h^i| = (mag_h/mag_1)|I_1^i|`,
`arg = ang_h + h·(a1_i − ang_1)`, **per phase** (`Load.pas` `DoHarmonicMode`). A
multi-phase Load is therefore inherently balanced in its spectrum; genuinely
different per-phase spectra require **three separate 1-phase loads, each with its
own `Spectrum`** (R. Dugan, OpenDSS forum). Refs:
<https://opendss.epri.com/HarmonicsLoadModeling.html>,
<https://sourceforge.net/p/electricdss/discussion/beginners/thread/81f0a80d/>.
This confirms and extends the convention in [OpenDSS harmonics](references/opendss/harmonics.md).

**pgml adopted model.** A per-phase spectrum option in the schema
and a phase axis to the `harmonic_injection` override; build `I(h)` per phase using
each phase's own fundamental phasor (already per-phase in `harmonic_flow.py`). A
device-level spectrum stays the "same on all phases" shorthand (OpenDSS multi-phase
Load semantics).

## Primary sources
- power-grid-model docs: calculations / components / data-model pages under
  <https://power-grid-model.readthedocs.io/en/stable/>; enums + dtypes verified via
  `initialize_array`, `LoadGenType`, `WindingType` on `power_grid_model 1.13.94`.
- pandapower: `runpp_3ph` + `asymmetric_load` docs
  <https://pandapower.readthedocs.io/en/latest/powerflow/ac_3ph.html>,
  `.../elements/asymmetric_load.html`; source `pandapower/pf/runpp_3ph.py`,
  `results_bus.py`, `results_branch.py`.
- OpenDSS (EPRI): Properties7, Load1, Model1, ZIPV, NeutralRules,
  HarmonicsLoadModeling, HarmonicFlowAnalysis pages under
  <https://opendss.epri.com/>; `Source/PCElements/Load.pas`; EPRI forum threads
  (single-phase neutral, unbalanced harmonic spectrum, ZIP model) cited inline.
