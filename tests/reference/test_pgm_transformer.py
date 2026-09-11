"""Oracle test: power-grid-model two-winding ``transformer`` -> pgml ``Transformer``.

Test strategy
-------------
1. **Convention pins** (``TestPgmConventionPins``) lock power-grid-model's OWN
   clock and tap sign conventions against a LIVE ``PowerGridModel.calculate_power_flow``
   solve on a 2-bus circuit -- a canary that fails loudly if a future
   power-grid-model release changes either convention, independent of our
   converter. Verified (see the module docstring of
   ``pgml.convert.pgm.converter`` for the source-level derivation from pgm's own
   ``transformer.hpp``/``branch.hpp``):

   - ``clock``: HV angle − LV angle == ``clock*30`` deg (positive clock -> LV
     LAGS HV) -- IDENTICAL sign to pgml's own ``tap.shift_deg`` convention, no
     flip needed.
   - ``tap_side=0`` (from/HV): effective from-voltage ``u1 + Δu`` (light load,
     near open-circuit) makes the LV bus pu voltage drop to ``u2/u1_eff``.
   - ``tap_side=1`` (to/LV, also the default when absent): effective
     to-voltage ``u2 + Δu`` makes the LV bus pu voltage rise to ``u2_eff/u2``.

2. **Sym oracle** (``TestSymOracle``): a 2-bus MV source -> transformer -> LV
   const-impedance load circuit (mirrors ``test_ieee33_pgm.py``'s pattern --
   const-Z load, linear ``solve_harmonic``, ideal-slack fixed at the pgm
   source's own phasor) swept over every requested winding pairing/clock, plus
   an off-nominal tap on each side. Both a REALISTIC magnetizing branch
   (i0=0.5%, non-zero p0) and a magnetizing-free (i0=p0=0) variant are run: the
   magnetizing-free variant isolates the leakage + vector-group + tap path
   (machine precision); the with-magnetizing variant carries a small, EXPECTED
   residual from a genuine topology difference (documented below).

3. **Asym oracle** (``TestAsymOracle``): the same circuit with an unbalanced
   ``asym_load``, converted with ``phase_mode=THREE_PHASE`` and compared
   per-phase against pgm's ``symmetric=False`` output. A balanced-load sanity
   check is machine-precision for every pairing (proves the THREE_PHASE
   conversion, including the mode-dependent delta coil factor below, is
   correct); Dyn5 and YNyn0 stay machine-precision even under IMBALANCE (delta
   zero-sequence blocking and grounded-wye zero-sequence passing are both
   exact, unambiguous topological facts in both tools); a grounded-zigzag
   pairing (YNzn5) shows a real, quantified, DOCUMENTED divergence under
   imbalance (see "Zigzag zero-sequence" below) -- reported, not hidden behind
   a loose tolerance.

Two genuine modelling-topology differences (not converter bugs)
-----------------------------------------------------------------
**Magnetizing-branch placement.** pgm's own ``calc_param_y_sym`` (``branch.hpp``)
splits the (to-side-referred) magnetizing admittance HALF onto its ``Y_tt``
and HALF (reflected through the tap) onto ``Y_ff``; pgml's SHIPPED DEFAULT
(``transformer.magnetizing_placement = from_terminal``) stamps it as one
undivided shunt at the FROM/HV terminal. The converter refers pgm's ``i0``/``p0``
to the HV terminal by the square nameplate ratio, so the residual below is the
placement alone: it vanishes either with ``i0=p0=0``
(``TestSymOracle::test_no_magnetizing_branch_is_machine_precision``) or by
selecting the matching placement
(``TestSymOracle::test_split_magnetizing_placement_removes_the_residual``).

**Zigzag zero-sequence VALUE (resolved).** power-grid-model hardcodes the
zero-sequence self-impedance of a grounded zigzag (``zigzag_n``) winding as
``0.1 * Z1`` (``transformer.hpp``: ``z0_series = (1/y_series)*0.1 + ...``, an
empirical stand-in for the half-coil zero-sequence leakage of a grounding
transformer). The converter now carries that factor into
``Transformer.zero_sequence``, which the stamp consumes as the zero-sequence
leakage VALUE on the topology-derived path, so the zigzag-side bus agrees with
power-grid-model under imbalance instead of presenting ``Z0 = Z1`` (measured ratio
of zero-sequence voltages 1.0 to 1.3e-11, against 10 before). Both tools always
agreed that a zigzag winding BLOCKS zero-sequence TRANSFER; only the winding's own
zero-sequence self-admittance differed.

Tolerance targets
------------------
- Sym, no magnetizing branch (i0=p0=0): atol = 1e-8 pu / 1e-6 deg (leakage +
  vector-group + tap path only). Achieved ~1e-11 pu / 1e-9 deg.
- Sym, realistic magnetizing branch (i0=0.5%): atol = 5e-4 pu / 5e-3 deg
  (documented magnetizing-topology residual). Achieved ~1.5e-4 pu / 1.2e-3 deg
  -- comparable to the OpenDSS transformer oracle's own documented ~1e-3 pu
  residual for a similar magnetizing current (``test_opendss_transformer.py``).
- Asym, balanced load (any pairing): atol = 1e-6 pu / 1e-4 deg. Achieved
  ~2e-11 pu / 3e-9 deg.
- Asym, unbalanced load, Dyn5 / YNyn0 (no zigzag): atol = 1e-6 pu / 1e-4 deg.
  Achieved ~3e-11 pu / 4e-9 deg.
- Asym, unbalanced load, YNzn5 (grounded zigzag): atol = 1e-6 pu / 1e-4 deg on BOTH
  buses now that the converter carries power-grid-model's ``0.1*Z1`` zigzag
  zero-sequence value (previously a documented ~6.8e-3 pu gap on the zigzag-side
  bus for a ~25 %-unbalanced load).
"""

from __future__ import annotations

import cmath
import math

import power_grid_model as pgm
import pytest
import torch
from power_grid_model import CalculationMethod, LoadGenType, PowerGridModel, WindingType

from pgml.assembly import assemble_ybus, build_injections, node_phase_index
from pgml.convert.pgm import PhaseMode, to_grid
from pgml.errors import ConversionError, ModelingError
from pgml.schemas.grid_schema import LoadModel, Phase, WindingConnection
from pgml.solver import solve_harmonic

_ABC = (Phase.A, Phase.B, Phase.C)
_U_HV = 20_000.0  # V, line-to-line
_U_LV = 400.0  # V, line-to-line
_SN = 630.0e3  # VA
_UK = 0.06
_PK = 6_300.0  # W
_F0 = 50.0

_NODE_HV, _NODE_LV, _SOURCE_ID, _LOAD_ID, _XFMR_ID = 1, 2, 10, 20, 30


# ---------------------------------------------------------------------------
# pgm input builders
# ---------------------------------------------------------------------------
def _nodes():
    node = pgm.initialize_array("input", "node", 2)
    node["id"] = [_NODE_HV, _NODE_LV]
    node["u_rated"] = [_U_HV, _U_LV]
    return node


def _ideal_source():
    src = pgm.initialize_array("input", "source", 1)
    src["id"] = [_SOURCE_ID]
    src["node"] = [_NODE_HV]
    src["status"] = [1]
    src["u_ref"] = [1.0]
    src["u_ref_angle"] = [0.0]
    src["sk"] = [1.0e16]  # near-ideal slack
    src["rx_ratio"] = [0.0]
    src["z01_ratio"] = [1.0]
    return src


def _transformer_row(
    *,
    winding_from: int,
    winding_to: int,
    clock: int,
    i0: float = 0.0,
    p0: float = 0.0,
    tap_side: int = 0,
    tap_pos: int = 0,
    tap_nom: int = 0,
    tap_min: int = -10,
    tap_max: int = 10,
    tap_size: float = 0.0,
    u1: float = _U_HV,
    u2: float = _U_LV,
):
    xfmr = pgm.initialize_array("input", "transformer", 1)
    xfmr["id"] = [_XFMR_ID]
    xfmr["from_node"] = [_NODE_HV]
    xfmr["to_node"] = [_NODE_LV]
    xfmr["from_status"] = [1]
    xfmr["to_status"] = [1]
    xfmr["u1"] = [u1]
    xfmr["u2"] = [u2]
    xfmr["sn"] = [_SN]
    xfmr["uk"] = [_UK]
    xfmr["pk"] = [_PK]
    xfmr["i0"] = [i0]
    xfmr["p0"] = [p0]
    xfmr["winding_from"] = [winding_from]
    xfmr["winding_to"] = [winding_to]
    xfmr["clock"] = [clock]
    xfmr["tap_side"] = [tap_side]
    xfmr["tap_pos"] = [tap_pos]
    xfmr["tap_nom"] = [tap_nom]
    xfmr["tap_min"] = [tap_min]
    xfmr["tap_max"] = [tap_max]
    xfmr["tap_size"] = [tap_size]
    return xfmr


def _sym_input(*, p_w: float = 200.0e3, q_var: float = 60.0e3, **xfmr_kwargs) -> dict:
    load = pgm.initialize_array("input", "sym_load", 1)
    load["id"] = [_LOAD_ID]
    load["node"] = [_NODE_LV]
    load["status"] = [1]
    load["type"] = [LoadGenType.const_impedance]
    load["p_specified"] = [p_w]
    load["q_specified"] = [q_var]
    return {
        "node": _nodes(),
        "source": _ideal_source(),
        "sym_load": load,
        "transformer": _transformer_row(**xfmr_kwargs),
    }


def _asym_input(
    *,
    p_phase=(200.0e3, 150.0e3, 250.0e3),
    q_phase=(60.0e3, 40.0e3, 80.0e3),
    **xfmr_kwargs,
) -> dict:
    load = pgm.initialize_array("input", "asym_load", 1)
    load["id"] = [_LOAD_ID]
    load["node"] = [_NODE_LV]
    load["status"] = [1]
    load["type"] = [LoadGenType.const_impedance]
    load["p_specified"] = [list(p_phase)]
    load["q_specified"] = [list(q_phase)]
    return {
        "node": _nodes(),
        "source": _ideal_source(),
        "asym_load": load,
        "transformer": _transformer_row(**xfmr_kwargs),
    }


def _angle_diff_deg(a: float, b: float) -> float:
    """Signed angle difference a - b in degrees, wrapped to (-180, 180]."""
    diff = (a - b) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


# ---------------------------------------------------------------------------
# 1. Convention pins (locks power-grid-model's OWN behaviour, no pgml)
# ---------------------------------------------------------------------------
class TestPgmConventionPins:
    """Live ``PowerGridModel`` solves pinning pgm's clock/tap sign conventions."""

    @pytest.mark.parametrize("clock", [1, 5, 11])
    def test_clock_sign_lv_lags_hv(self, clock: int) -> None:
        """HV angle - LV angle == clock*30 deg (Dyn family, light load)."""
        input_data = _sym_input(
            winding_from=WindingType.delta,
            winding_to=WindingType.wye_n,
            clock=clock,
            p_w=1.0e3,
            q_var=100.0,
        )
        model = PowerGridModel(input_data)
        result = model.calculate_power_flow(
            symmetric=True, calculation_method=CalculationMethod.newton_raphson
        )
        by_id = {int(r["id"]): r for r in result["node"]}
        hv_angle = math.degrees(float(by_id[_NODE_HV]["u_angle"]))
        lv_angle = math.degrees(float(by_id[_NODE_LV]["u_angle"]))
        lag = (hv_angle - lv_angle) % 360.0
        assert lag == pytest.approx(float(clock * 30), abs=0.02)

    def test_tap_from_side_lowers_lv_voltage(self) -> None:
        """tap_side=0 (from/HV), tap_pos>tap_nom: LV pu drops to ~u2/u1_eff."""
        tap_size = 200.0
        delta_u = 5 * tap_size
        input_data = _sym_input(
            winding_from=WindingType.wye_n,
            winding_to=WindingType.wye_n,
            clock=0,
            p_w=1.0,
            q_var=0.0,
            tap_side=0,
            tap_pos=5,
            tap_size=tap_size,
        )
        model = PowerGridModel(input_data)
        result = model.calculate_power_flow(
            symmetric=True, calculation_method=CalculationMethod.newton_raphson
        )
        by_id = {int(r["id"]): r for r in result["node"]}
        vm_pu_lv = float(by_id[_NODE_LV]["u_pu"])
        expected = _U_HV / (_U_HV + delta_u)
        assert vm_pu_lv == pytest.approx(expected, rel=1e-6)

    def test_tap_to_side_raises_lv_voltage(self) -> None:
        """tap_side=1 (to/LV), tap_pos<tap_nom: LV pu moves to ~u2_eff/u2."""
        tap_size = 10.0
        delta_u = -3 * tap_size
        input_data = _sym_input(
            winding_from=WindingType.wye_n,
            winding_to=WindingType.wye_n,
            clock=0,
            p_w=1.0,
            q_var=0.0,
            tap_side=1,
            tap_pos=-3,
            tap_size=tap_size,
        )
        model = PowerGridModel(input_data)
        result = model.calculate_power_flow(
            symmetric=True, calculation_method=CalculationMethod.newton_raphson
        )
        by_id = {int(r["id"]): r for r in result["node"]}
        vm_pu_lv = float(by_id[_NODE_LV]["u_pu"])
        expected = (_U_LV + delta_u) / _U_LV
        assert vm_pu_lv == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. Sym oracle: our solver vs pgm symmetric power flow
# ---------------------------------------------------------------------------
_SYM_CASES = [
    pytest.param(WindingType.delta, WindingType.wye_n, 1, id="Dyn1"),
    pytest.param(WindingType.delta, WindingType.wye_n, 5, id="Dyn5"),
    pytest.param(WindingType.wye_n, WindingType.delta, 5, id="YNd5"),
    pytest.param(WindingType.wye_n, WindingType.wye_n, 0, id="YNyn0"),
    pytest.param(WindingType.wye, WindingType.wye, 6, id="Yy6"),
    pytest.param(WindingType.wye_n, WindingType.zigzag_n, 5, id="YNzn5"),
]


def _run_sym_case(input_data: dict) -> tuple[float, float]:
    """Return (max |V| pu error, max angle error deg) vs a live pgm sym solve."""
    model = PowerGridModel(input_data)
    result = model.calculate_power_flow(
        symmetric=True, calculation_method=CalculationMethod.newton_raphson
    )
    pgm_by_id = {int(r["id"]): r for r in result["node"]}

    grid, id_map = to_grid(
        input_data, base_frequency_hz=_F0, load_model=LoadModel.CONST_IMPEDANCE
    )
    index = node_phase_index(grid)
    ybus = assemble_ybus(grid, [_F0], dtype=torch.complex128)
    i_inj = build_injections(grid, [_F0], index, dtype=torch.complex128)

    slack_node = id_map["node"][_NODE_HV]
    fixed_rows = torch.tensor([index.row(slack_node, Phase.A)], dtype=torch.int64)
    v_fixed = torch.tensor([id_map["slack_v_complex"]], dtype=torch.complex128)
    v_all = solve_harmonic(ybus.Y, i_inj, fixed_rows=fixed_rows, v_fixed=v_fixed)

    max_vm_err = 0.0
    max_va_err = 0.0
    for pgm_id, our_id in id_map["node"].items():
        row = index.row(our_id, Phase.A)
        v_c = complex(v_all[0, row].item())
        node_obj = next(n for n in grid.nodes if n.id == our_id)
        vm_pu_ours = abs(v_c) / node_obj.u_rated_v
        va_deg_ours = math.degrees(cmath.phase(v_c))
        pgm_row = pgm_by_id[pgm_id]
        vm_pu_pgm = float(pgm_row["u_pu"])
        va_deg_pgm = math.degrees(float(pgm_row["u_angle"]))
        max_vm_err = max(max_vm_err, abs(vm_pu_ours - vm_pu_pgm))
        max_va_err = max(max_va_err, abs(_angle_diff_deg(va_deg_ours, va_deg_pgm)))
    return max_vm_err, max_va_err


class TestSymOracle:
    ATOL_VM_PU_WITH_MAGNETIZING = 5.0e-4
    ATOL_VA_DEG_WITH_MAGNETIZING = 5.0e-3
    ATOL_VM_PU_NO_MAGNETIZING = 1.0e-8
    ATOL_VA_DEG_NO_MAGNETIZING = 1.0e-6

    @pytest.mark.parametrize("winding_from,winding_to,clock", _SYM_CASES)
    def test_vector_group_with_realistic_magnetizing_branch(
        self, winding_from: int, winding_to: int, clock: int
    ) -> None:
        input_data = _sym_input(
            winding_from=winding_from,
            winding_to=winding_to,
            clock=clock,
            i0=0.005,
            p0=1_000.0,
        )
        vm_err, va_err = _run_sym_case(input_data)
        assert vm_err < self.ATOL_VM_PU_WITH_MAGNETIZING
        assert va_err < self.ATOL_VA_DEG_WITH_MAGNETIZING

    @pytest.mark.parametrize("winding_from,winding_to,clock", _SYM_CASES)
    def test_no_magnetizing_branch_is_machine_precision(
        self, winding_from: int, winding_to: int, clock: int
    ) -> None:
        """i0=p0=0 isolates leakage + vector-group + tap: machine precision."""
        input_data = _sym_input(
            winding_from=winding_from, winding_to=winding_to, clock=clock
        )
        vm_err, va_err = _run_sym_case(input_data)
        assert vm_err < self.ATOL_VM_PU_NO_MAGNETIZING
        assert va_err < self.ATOL_VA_DEG_NO_MAGNETIZING

    def test_split_magnetizing_placement_removes_the_residual(
        self, tmp_path, monkeypatch
    ) -> None:
        """``transformer.magnetizing_placement="split"`` IS power-grid-model's topology.

        Switching the documented placement from the shipped ``from_terminal`` to
        ``split`` (half the shunt on each terminal, each referred to its own side) drops
        the magnetizing residual below from ~1.5e-4 pu to the magnetizing-free tolerance
        (1e-8 pu), which identifies the residual as the placement and nothing else.
        """
        import yaml

        from pgml import defaults

        input_data = _sym_input(
            winding_from=WindingType.wye_n,
            winding_to=WindingType.wye_n,
            clock=0,
            i0=0.005,
            p0=1_000.0,
        )
        vm_from, va_from = _run_sym_case(input_data)
        assert vm_from > 1.0e-5, "the from_terminal residual must be present"

        data = yaml.safe_load(yaml.safe_dump(defaults.defaults()))
        data["transformer"]["magnetizing_placement"]["value"] = "split"
        path = tmp_path / "split.yaml"
        path.write_text(yaml.safe_dump(data))
        monkeypatch.setenv("PGML_DEFAULTS", str(path))
        try:
            defaults.reload(str(path))
            vm_split, va_split = _run_sym_case(input_data)
        finally:
            monkeypatch.delenv("PGML_DEFAULTS", raising=False)
            defaults.reload()
        assert vm_split < self.ATOL_VM_PU_NO_MAGNETIZING
        assert va_split < self.ATOL_VA_DEG_NO_MAGNETIZING

    def test_off_nominal_tap_from_side(self) -> None:
        input_data = _sym_input(
            winding_from=WindingType.wye_n,
            winding_to=WindingType.wye_n,
            clock=0,
            tap_side=0,
            tap_pos=5,
            tap_size=200.0,
        )
        vm_err, va_err = _run_sym_case(input_data)
        assert vm_err < self.ATOL_VM_PU_NO_MAGNETIZING
        assert va_err < self.ATOL_VA_DEG_NO_MAGNETIZING

    def test_off_nominal_tap_to_side(self) -> None:
        input_data = _sym_input(
            winding_from=WindingType.wye_n,
            winding_to=WindingType.wye_n,
            clock=0,
            tap_side=1,
            tap_pos=-3,
            tap_size=10.0,
        )
        vm_err, va_err = _run_sym_case(input_data)
        assert vm_err < self.ATOL_VM_PU_NO_MAGNETIZING
        assert va_err < self.ATOL_VA_DEG_NO_MAGNETIZING


# ---------------------------------------------------------------------------
# 3. Converter unit test: the stored leakage is phase-mode INDEPENDENT
# ---------------------------------------------------------------------------
class TestCoilFactorIsModeIndependent:
    """The stored leakage is TO-coil-referred regardless of phase mode.

    ``series_resistance_ohm``/``series_inductance_h`` carry the coil value
    (``3·z_LL`` for a delta TO winding, ``z_LL`` otherwise) in both phase modes:
    the 3-phase winding-incidence stamp divides the delta factor back through
    ``Mᵀ M`` and the single-phase scalar pi applies its own ``y_LL = 3·y_coil``
    referral, so the same schema values solve identically either way. A
    mode-dependent stored value would break the schema contract (the field is
    defined as coil-referred) -- regression guarded here and by
    ``TestSymOracle::test_no_magnetizing_branch_is_machine_precision[YNd5]``.
    """

    @pytest.mark.parametrize(
        "winding_from,winding_to,clock",
        [
            (WindingType.wye_n, WindingType.delta, 5),
            (WindingType.delta, WindingType.wye_n, 5),
        ],
    )
    def test_stored_leakage_identical_across_modes(
        self, winding_from, winding_to, clock
    ) -> None:
        input_data = _sym_input(
            winding_from=winding_from, winding_to=winding_to, clock=clock
        )
        grid_1ph, _ = to_grid(
            input_data, base_frequency_hz=_F0, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV
        )
        grid_3ph, _ = to_grid(
            input_data, base_frequency_hz=_F0, phase_mode=PhaseMode.THREE_PHASE
        )
        assert grid_3ph.branches[0].series_resistance_ohm == pytest.approx(
            grid_1ph.branches[0].series_resistance_ohm, rel=1e-12
        )
        assert grid_3ph.branches[0].series_inductance_h == pytest.approx(
            grid_1ph.branches[0].series_inductance_h, rel=1e-12
        )


# ---------------------------------------------------------------------------
# 4. Asym oracle: our 3-phase solver vs pgm asymmetric power flow
# ---------------------------------------------------------------------------
def _run_asym_case(input_data: dict) -> dict[tuple[int, Phase], tuple[float, float]]:
    """Return ``{(pgm_node_id, phase): (vm_pu_err, va_deg_err)}`` vs a live pgm asym solve."""
    model = PowerGridModel(input_data)
    result = model.calculate_power_flow(
        symmetric=False, calculation_method=CalculationMethod.newton_raphson
    )
    pgm_by_id = {int(r["id"]): r for r in result["node"]}

    grid, id_map = to_grid(
        input_data,
        base_frequency_hz=_F0,
        load_model=LoadModel.CONST_IMPEDANCE,
        phase_mode=PhaseMode.THREE_PHASE,
    )
    index = node_phase_index(grid)
    ybus = assemble_ybus(grid, [_F0], dtype=torch.complex128)
    i_inj = build_injections(grid, [_F0], index, dtype=torch.complex128)

    slack_node = id_map["node"][_NODE_HV]
    fixed_rows = torch.tensor(
        [index.row(slack_node, p) for p in _ABC], dtype=torch.int64
    )
    u_ln = _U_HV / math.sqrt(3.0)
    v_fixed = torch.tensor(
        [u_ln * cmath.exp(1j * math.radians(ang)) for ang in (0.0, -120.0, 120.0)],
        dtype=torch.complex128,
    )
    v_all = solve_harmonic(ybus.Y, i_inj, fixed_rows=fixed_rows, v_fixed=v_fixed)

    errors: dict[tuple[int, Phase], tuple[float, float]] = {}
    for pgm_id, our_id in id_map["node"].items():
        node_obj = next(n for n in grid.nodes if n.id == our_id)
        u_ln_base = node_obj.u_rated_v / math.sqrt(3.0)
        pgm_row = pgm_by_id[pgm_id]
        for i, p in enumerate(_ABC):
            v_c = complex(v_all[0, index.row(our_id, p)].item())
            vm_pu_ours = abs(v_c) / u_ln_base
            va_deg_ours = math.degrees(cmath.phase(v_c))
            vm_pu_pgm = float(pgm_row["u_pu"][i])
            va_deg_pgm = math.degrees(float(pgm_row["u_angle"][i]))
            errors[(pgm_id, p)] = (
                abs(vm_pu_ours - vm_pu_pgm),
                abs(_angle_diff_deg(va_deg_ours, va_deg_pgm)),
            )
    return errors


class TestAsymOracle:
    ATOL_VM_PU = 1.0e-6
    ATOL_VA_DEG = 1.0e-4

    @pytest.mark.parametrize(
        "winding_from,winding_to,clock",
        [
            pytest.param(WindingType.delta, WindingType.wye_n, 5, id="Dyn5"),
            pytest.param(WindingType.wye_n, WindingType.delta, 5, id="YNd5"),
            pytest.param(WindingType.wye_n, WindingType.zigzag_n, 5, id="YNzn5"),
        ],
    )
    def test_balanced_load_matches_sym(
        self, winding_from: int, winding_to: int, clock: int
    ) -> None:
        """A balanced asym_load carries no zero sequence: tight for every pairing."""
        input_data = _asym_input(
            p_phase=(200.0e3,) * 3,
            q_phase=(60.0e3,) * 3,
            winding_from=winding_from,
            winding_to=winding_to,
            clock=clock,
        )
        errors = _run_asym_case(input_data)
        assert max(e[0] for e in errors.values()) < self.ATOL_VM_PU
        assert max(e[1] for e in errors.values()) < self.ATOL_VA_DEG

    def test_dyn5_unbalanced_tight(self) -> None:
        """Delta zero-sequence blocking is an exact topological fact in both tools."""
        input_data = _asym_input(
            winding_from=WindingType.delta, winding_to=WindingType.wye_n, clock=5
        )
        errors = _run_asym_case(input_data)
        assert max(e[0] for e in errors.values()) < self.ATOL_VM_PU
        assert max(e[1] for e in errors.values()) < self.ATOL_VA_DEG

    def test_ynyn0_unbalanced_tight(self) -> None:
        """Grounded-wye zero-sequence self path Z0=Z1 agrees exactly in both tools."""
        input_data = _asym_input(
            winding_from=WindingType.wye_n, winding_to=WindingType.wye_n, clock=0
        )
        errors = _run_asym_case(input_data)
        assert max(e[0] for e in errors.values()) < self.ATOL_VM_PU
        assert max(e[1] for e in errors.values()) < self.ATOL_VA_DEG

    def test_ynzn5_unbalanced_zigzag_zero_sequence(self) -> None:
        """Grounded zigzag under imbalance: tight on BOTH buses.

        No zero sequence crosses the zigzag winding in either model, and the zigzag
        winding's own zero-sequence self-impedance now carries power-grid-model's
        hardcoded ``0.1*Z1`` (``transformer.hpp``: ``z0_series = (1/y_series)*0.1``),
        converted into ``Transformer.zero_sequence`` and consumed by the stamp — so the
        zigzag-side bus is no longer a documented 10x gap.
        """
        input_data = _asym_input(
            winding_from=WindingType.wye_n, winding_to=WindingType.zigzag_n, clock=5
        )
        errors = _run_asym_case(input_data)
        assert max(e[0] for e in errors.values()) < self.ATOL_VM_PU
        assert max(e[1] for e in errors.values()) < self.ATOL_VA_DEG

    def test_ynzn5_zero_sequence_matches_power_grid_model(self) -> None:
        """Quantifies the zigzag Z0 directly: |V0_pgml|/|V0_pgm| == 1 on the zigzag bus."""
        input_data = _asym_input(
            winding_from=WindingType.wye_n, winding_to=WindingType.zigzag_n, clock=5
        )
        model = PowerGridModel(input_data)
        result = model.calculate_power_flow(
            symmetric=False, calculation_method=CalculationMethod.newton_raphson
        )
        pgm_row = next(r for r in result["node"] if int(r["id"]) == _NODE_LV)
        v0_pgm = (
            sum(
                pgm_row["u"][i] * cmath.exp(1j * pgm_row["u_angle"][i])
                for i in range(3)
            )
            / 3.0
        )

        grid, id_map = to_grid(
            input_data,
            base_frequency_hz=_F0,
            load_model=LoadModel.CONST_IMPEDANCE,
            phase_mode=PhaseMode.THREE_PHASE,
        )
        index = node_phase_index(grid)
        ybus = assemble_ybus(grid, [_F0], dtype=torch.complex128)
        i_inj = build_injections(grid, [_F0], index, dtype=torch.complex128)
        slack_node = id_map["node"][_NODE_HV]
        fixed_rows = torch.tensor(
            [index.row(slack_node, p) for p in _ABC], dtype=torch.int64
        )
        u_ln = _U_HV / math.sqrt(3.0)
        v_fixed = torch.tensor(
            [u_ln * cmath.exp(1j * math.radians(ang)) for ang in (0.0, -120.0, 120.0)],
            dtype=torch.complex128,
        )
        v_all = solve_harmonic(ybus.Y, i_inj, fixed_rows=fixed_rows, v_fixed=v_fixed)
        lv_node = id_map["node"][_NODE_LV]
        v0_pgml = (
            sum(complex(v_all[0, index.row(lv_node, p)].item()) for p in _ABC) / 3.0
        )

        ratio = abs(v0_pgml) / abs(v0_pgm)
        assert ratio == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Unsupported models: fail loud, never silently wrong
# ---------------------------------------------------------------------------
class TestUnsupportedRejections:
    def test_zigzag_zigzag_pairing_raises_at_assembly(self) -> None:
        input_data = _sym_input(
            winding_from=WindingType.zigzag_n, winding_to=WindingType.zigzag_n, clock=0
        )
        grid, _ = to_grid(input_data, base_frequency_hz=_F0)
        with pytest.raises(ModelingError, match="zigzag-zigzag"):
            assemble_ybus(grid, [_F0], dtype=torch.complex128)

    def test_finite_grounding_impedance_raises_at_assembly(self) -> None:
        input_data = _sym_input(
            winding_from=WindingType.delta, winding_to=WindingType.wye_n, clock=5
        )
        input_data["transformer"]["r_grounding_to"] = [2.0]
        grid, _ = to_grid(input_data, base_frequency_hz=_F0)
        assert grid.branches[0].to_grounding is not None
        with pytest.raises(ModelingError, match="grounding"):
            assemble_ybus(grid, [_F0], dtype=torch.complex128)

    def test_solid_grounding_does_not_raise(self) -> None:
        input_data = _sym_input(
            winding_from=WindingType.delta, winding_to=WindingType.wye_n, clock=5
        )
        grid, _ = to_grid(input_data, base_frequency_hz=_F0)
        assert grid.branches[0].to_grounding is None
        assemble_ybus(grid, [_F0], dtype=torch.complex128)  # must not raise

    def test_unknown_winding_type_raises_at_conversion(self) -> None:
        input_data = _sym_input(
            winding_from=WindingType.delta, winding_to=WindingType.wye_n, clock=5
        )
        input_data["transformer"]["winding_to"] = [99]
        with pytest.raises(ConversionError, match="unknown pgm winding_to"):
            to_grid(input_data, base_frequency_hz=_F0)

    def test_negative_clock_wraps_cyclically(self) -> None:
        """pgm allows clock in [-12, 12]; the converter wraps it like pgm's own
        `map_to_cyclic_range` (clock -7 == clock 5, matching Dyn5)."""
        input_data = _sym_input(
            winding_from=WindingType.delta, winding_to=WindingType.wye_n, clock=-7
        )
        grid, _ = to_grid(input_data, base_frequency_hz=_F0)
        assert grid.branches[0].tap.shift_deg == pytest.approx(150.0)

    def test_dropped_elements_no_longer_include_transformer(self) -> None:
        input_data = _sym_input(
            winding_from=WindingType.delta, winding_to=WindingType.wye_n, clock=5
        )
        grid, id_map = to_grid(input_data, base_frequency_hz=_F0)
        assert len(grid.branches) == 1
        assert grid.branches[0].from_connection == WindingConnection.DELTA
        assert grid.branches[0].to_connection == WindingConnection.WYE_GROUNDED
        assert id_map["transformer"][_XFMR_ID] == grid.branches[0].id
