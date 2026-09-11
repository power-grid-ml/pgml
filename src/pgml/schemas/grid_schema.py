"""Phase-0 canonical data contract for the grid description (rev 3).

**SINGLE SOURCE OF TRUTH** for the grid description. Every other subsystem (Y-bus
assembly, solver, parquet/SQL persistence, JSON export, the PyTorch-Geometric
adapter) consumes these models and MUST NOT redefine them. Edits are
orchestrator-only; subagents import, they do not modify.

The data OUTPUT contract (per-harmonic, per-phase voltages/currents/powers and
derived THD) lives in a separate artifact, :mod:`pgml.schemas.result_schema`, not
here.

**Conventions (read before using any field)**

*Canonical form is phase-domain.* The internal representation of every element is
its per-phase primitive admittance contribution (an ``n_phase x n_phase`` complex
stamp per harmonic). Sequence data (Z1/Z2/Z0, vk/vk0) and nameplate data are
INPUT CONVENTIONS handled by converters that emit phase-domain objects. This
mirrors OpenDSS, the only established harmonic engine.

*No complex numbers in the canonical schema.* Every impedance-like quantity is
stored as the physical REAL pair: series (R [Ohm], L [H]); shunt (G [S], C [F]).
Complex admittances are formed only at assembly time. This keeps frequency
scaling physical (``X(h) = 2*pi*h*f0*L``, ``B(h) = 2*pi*h*f0*C``) and gradients
attached to physical parameters. The sole exception is ``ComplexTap`` (magnitude
+ angle), which is a dimensionless ratio, not an impedance.

*Units are structured metadata.* Every physical field carries machine-readable
unit info in its JSON Schema under ``unit = {short, long, [reference]}`` (via the
``si_field`` helper). Dashboards read ``Grid.model_json_schema()`` and never parse
the description string. The unit is also echoed in the description for humans.

*Per-phase / asymmetry.* Nodes carry an ordered phase set; multi-phase element
parameters are full ``n x n`` matrices (diagonal = self, off-diagonal = mutual).

*Component symmetry vs calculation symmetry* (adopted from power-grid-model) are
independent. A grid may define symmetric and asymmetric appliances together, and
either a symmetric or an asymmetric calculation may run on the same grid:

- ``p_nom_w`` / ``q_nom_var`` are ALWAYS the total over the connected phases.
- Optional ``p_nom_per_phase_w`` / ``q_nom_per_phase_var`` give an asymmetric
  nameplate split (length == ``len(phases)``; must sum to the totals).
- Connection sets terminal pairing: WYE = each listed phase to neutral/ground
  (European LV single-phase L-N is ``phases=(A,)``, ``connection=WYE``); DELTA =
  between listed phases (``phases=(A,B)``, ``connection=DELTA`` is an L-L load).
  ``connection=None`` resolves from config (``appliance.load.default_connection`` /
  ``single_phase_connection``, both WYE by default).
- Neutral: a WYE appliance returns into its node's ``Phase.N`` row when that node
  carries a neutral (the explicit 4-wire case — assembly logs that the neutral is
  modeled); with no ``Phase.N`` it returns to ground (3-wire / solidly grounded,
  matching pandapower/pgm and the OpenDSS grounded-neutral default).
- Resolution at assembly time (config ``calculation.symmetry``):

  - Symmetric calculation: asymmetric / 1-ph / 2-ph appliances are averaged
    into an equivalent balanced appliance.
  - Asymmetric calculation: a symmetric appliance is split equally across its
    phases UNLESS a per-phase profile or ``*_per_phase_*`` is given, which wins.
  - ``auto`` (default): asymmetric iff any appliance / operating point carries
    per-phase data, else symmetric (the power-grid-model rule).

- Whether a run is symmetric/asymmetric is SOLVER CONFIG, not grid data. See
  ``docs/pgml/modeling/asymmetric.md`` for the cross-tool basis and citations.

*Rated vs operating point.* Every ``*_rated`` / ``*_nom`` field is a NAMEPLATE
RATING, not an operating point. Actual loading/generation comes from profiles
(external service). For loads/generators the harmonic SHUNT ADMITTANCE is derived
at assembly time from the OPERATING-POINT P,Q, while the schema stores only model
parameters.

*Frequency scaling.* Reactive elements store L and C, not X and B. Series
resistance frequency dependence (skin effect) is an explicit optional law.

*Standard types are an authoring/storage layer.* Lines/transformers may reference
a catalog entry via ``type_ref`` (resolved from ``Grid.types``) instead of
repeating parameters. A resolver materialises parameters before assembly;
per-instance explicit values may override a referenced type (the resolver warns
on override). By the time assembly runs, every element carries materialised
parameters; the solver core never resolves a type. A canonical element is
"materialised" when it either has no ``type_ref`` or has had its ``type_ref``
expanded; ``type_ref`` is retained afterwards as provenance and as an ML
categorical feature.

*Geo.* Nodes and lines may carry GeoJSON-shaped geometry. The CRS is grid-wide
(``GridMetadata.crs``). ``Line.length_m`` is the ELECTRICAL length and is never
implicitly derived from geometry.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    field_validator,
    model_validator,
)

# =============================================================================
# Float / tensor duality (torch-free, duck-typed)
# =============================================================================
# Physical fields accept EITHER plain python numbers/lists (the serializable
# default) OR any array-like object (e.g. a torch.Tensor or numpy.ndarray), which
# is passed through UNTOUCHED so autograd gradients flow grid -> Y-bus -> solve ->
# outputs without a separate parameter container. The schema imports NO compute
# framework: "array-like" is detected by duck typing (`.detach` / `__array__`).
# Note: pydantic numeric constraints (gt/ge) are NOT applied to these Any-typed
# fields, so positivity is enforced inside the validators below (floats only).


def _is_arraylike(v: Any) -> bool:
    return hasattr(v, "detach") or hasattr(v, "__array__")


def _ser_numeric(v: Any) -> Any:
    """JSON serializer: detach array-like to nested python lists/scalars."""
    if hasattr(v, "detach"):
        v = v.detach()
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


def _scalar_validator(*, positive: bool = False, nonneg: bool = False):
    def _v(x: Any) -> Any:
        if _is_arraylike(x):
            return x
        x = float(x)
        if positive and not x > 0.0:
            raise ValueError("must be > 0")
        if nonneg and not x >= 0.0:
            raise ValueError("must be >= 0")
        return x

    return _v


def _matrix_validator(x: Any) -> Any:
    if _is_arraylike(x):
        return x
    return [[float(e) for e in row] for row in x]


def _vector_validator(x: Any) -> Any:
    if _is_arraylike(x):
        return x
    return tuple(float(e) for e in x)


_SER = PlainSerializer(_ser_numeric, when_used="json")

# Dual scalar/vector/matrix types: use in place of float / tuple[float, ...] /
# list[list[float]]. Each accepts plain python OR an array-like (tensor) untouched.
Num = Annotated[Any, PlainValidator(_scalar_validator()), _SER]
PosNum = Annotated[Any, PlainValidator(_scalar_validator(positive=True)), _SER]
NonNegNum = Annotated[Any, PlainValidator(_scalar_validator(nonneg=True)), _SER]
Vec = Annotated[Any, PlainValidator(_vector_validator), _SER]
# PerPhaseMatrix is the canonical row-major SI matrix type, now tensor-capable.
PerPhaseMatrix = Annotated[Any, PlainValidator(_matrix_validator), _SER]


def si_field(
    description: str,
    *,
    short: Optional[str] = None,
    long: Optional[str] = None,
    reference: Optional[str] = None,
    **kwargs,
):
    """Field with structured unit metadata in json_schema_extra['unit'].
    `short` e.g. 'V', `long` e.g. 'volt', `reference` e.g. 'referred to HV side'."""
    if short is not None:
        unit = {"short": short, "long": long if long is not None else short}
        if reference is not None:
            unit["reference"] = reference
        extra = dict(kwargs.pop("json_schema_extra", {}) or {})
        extra["unit"] = unit
        kwargs["json_schema_extra"] = extra
    return Field(description=description, **kwargs)


class GridModel(BaseModel):
    """Base: forbid unknown fields so a subagent inventing a field fails loudly."""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
    )


# =============================================================================
# 1. Enums
# =============================================================================
class Phase(str, Enum):
    A = "a"
    B = "b"
    C = "c"
    N = "n"


class WindingConnection(str, Enum):
    """Winding / load connection topology. Determines the zero-sequence PATH."""

    WYE = "wye"
    WYE_GROUNDED = "wye_grounded"
    DELTA = "delta"
    ZIGZAG = "zigzag"
    ZIGZAG_GROUNDED = "zigzag_grounded"


class LoadModel(str, Enum):
    """Fundamental-frequency voltage dependence of a load/generator injection."""

    CONST_POWER = "const_power"
    CONST_IMPEDANCE = "const_impedance"
    CONST_CURRENT = "const_current"
    ZIP = "zip"


class NodeZone(str, Enum):
    """Land-use / consumer character of a node (closed taxonomy; ML stratifier)."""

    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    AGRICULTURAL = "agricultural"
    MIXED = "mixed"
    NONE = "none"


class GridEnvironment(str, Enum):
    """Settlement character of the whole grid/feeder (closed taxonomy)."""

    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"


class InterpolationMethod(str, Enum):
    LINEAR = "linear"
    LOG_LOG = "log_log"
    CUBIC = "cubic"
    NEAREST = "nearest"


class ExtrapolationMethod(str, Enum):
    CONSTANT = "constant"
    EXTEND = "extend"
    ERROR = "error"


class SourceConvention(str, Enum):
    SHORT_CIRCUIT = "short_circuit"
    IMPEDANCE = "impedance"
    SEQUENCE = "sequence"
    GEOMETRY = "geometry"
    DIRECT = "direct"
    MEASURED = "measured"


class ConsumerType(str, Enum):
    """Closed device taxonomy for loads / generators / storage (an ML categorical).

    This classifies the asset's character; it does NOT drive the physics by itself
    (the electrical behaviour comes from ``load_model``, ``control`` and the harmonic
    model). ``OTHER`` is the escape hatch for an unlisted character; ``None`` on the
    appliance means unspecified. Being a string enum, a value compares equal to its
    string (``ConsumerType.PV == "pv"``), so existing string-valued grids and
    selectors keep working.
    """

    HOUSEHOLD = "household"
    EV_CHARGING = "ev_charging"
    HEAT_PUMP = "heat_pump"
    RESTAURANT = "restaurant"
    OFFICE = "office"
    WORKSHOP = "workshop"
    INDUSTRIAL_DRIVE = "industrial_drive"
    PV = "pv"
    WIND = "wind"
    CHP = "chp"
    DIESEL_GENSET = "diesel_genset"
    BATTERY = "battery"
    OTHER = "other"


class MeasuredQuantity(str, Enum):
    """Electrical quantity a :class:`MeasurementDevice` records.

    ``VOLTAGE`` is the node voltage at the device's bus; ``CURRENT`` is a branch
    current through a :class:`CurrentChannel`; ``POWER`` covers active/reactive
    power (derived from voltage and current by the instrument). Being a string
    enum, a value compares equal to its string (``MeasuredQuantity.VOLTAGE ==
    "voltage"``).
    """

    VOLTAGE = "voltage"
    CURRENT = "current"
    POWER = "power"


# =============================================================================
# 2. Geometry (GeoJSON-shaped fragments; CRS is grid-wide)
# =============================================================================
class GeoPoint(GridModel):
    type: Literal["Point"] = "Point"
    coordinates: tuple[float, float] = Field(
        description="(x/longitude, y/latitude) in the grid CRS (GridMetadata.crs)."
    )


class GeoLineString(GridModel):
    type: Literal["LineString"] = "LineString"
    coordinates: list[tuple[float, float]] = Field(
        description="Ordered (x, y) vertices in the grid CRS. Routing only; not "
        "used to derive electrical length.",
        min_length=2,
    )


# =============================================================================
# 3. Frequency-dependent parameters
# =============================================================================
class ConstantParam(GridModel):
    kind: Literal["constant"] = "constant"
    value: float = Field(
        description="Frequency-independent value, SI unit of host field."
    )


class AnalyticParam(GridModel):
    kind: Literal["analytic"] = "analytic"
    law: str = Field(description="Identifier of the analytic frequency law f -> value.")
    base_value: float = Field(
        description="Reference-frequency (f0) value the law scales."
    )
    params: dict[str, float] = Field(
        default_factory=dict, description="Extra coefficients (SI)."
    )


class CurveParam(GridModel):
    kind: Literal["curve"] = "curve"
    frequencies_hz: list[float] = si_field(
        "Strictly increasing sample frequencies.",
        short="Hz",
        long="hertz",
        min_length=2,
    )
    values: list[float] = Field(
        description="Sampled values [SI of host], one per frequency."
    )
    interpolation: InterpolationMethod = Field(default=InterpolationMethod.LINEAR)
    extrapolation: ExtrapolationMethod = Field(default=ExtrapolationMethod.CONSTANT)

    @model_validator(mode="after")
    def _check(self) -> "CurveParam":
        if len(self.frequencies_hz) != len(self.values):
            raise ValueError("`frequencies_hz` and `values` must have equal length.")
        if any(b <= a for a, b in zip(self.frequencies_hz, self.frequencies_hz[1:])):
            raise ValueError("`frequencies_hz` must be strictly increasing.")
        return self


class EquationParam(GridModel):
    kind: Literal["equation"] = "equation"
    equation_id: str = Field(description="Identifier of the frequency equation.")
    bindings: dict[str, float] = Field(
        default_factory=dict, description="Symbol->value (SI)."
    )


FrequencyParam = Annotated[
    Union[ConstantParam, AnalyticParam, CurveParam, EquationParam],
    Field(discriminator="kind"),
]


class ResistanceFrequencyModel(GridModel):
    multiplier: FrequencyParam = Field(
        default_factory=lambda: ConstantParam(value=1.0),
        description="Per-unit resistance MULTIPLIER vs frequency, applied to the "
        "reference-frequency resistance (1.0 = no skin effect).",
    )


# =============================================================================
# 4. Harmonic spectra (operating-point based)
# =============================================================================
class HarmonicComponent(GridModel):
    order: int = Field(description="Harmonic order h (1 = fundamental).", ge=1)
    magnitude_pu: float = si_field(
        "Magnitude as a fraction of the fundamental injection at the same operating point.",
        short="pu",
        long="per unit of fundamental",
        ge=0.0,
    )
    phase_deg: float = si_field(
        "Phase relative to the fundamental reference.", short="deg", long="degree"
    )


class SpectrumPoint(GridModel):
    components: list[HarmonicComponent] = Field(
        description="Harmonic content at one operating point; absent orders = 0 injection."
    )


class StaticSpectrum(GridModel):
    kind: Literal["static"] = "static"
    spectrum: SpectrumPoint


class LoadVaryingSpectrum(GridModel):
    kind: Literal["load_varying"] = "load_varying"
    loading_levels_pu: list[float] = si_field(
        "Measured power levels as a fraction of rated power. Increasing.",
        short="pu",
        long="per unit of rated",
        min_length=2,
    )
    spectra: list[SpectrumPoint] = Field(description="Spectrum at each loading level.")
    interpolation: InterpolationMethod = Field(default=InterpolationMethod.LINEAR)

    @model_validator(mode="after")
    def _check(self) -> "LoadVaryingSpectrum":
        if len(self.loading_levels_pu) != len(self.spectra):
            raise ValueError(
                "`loading_levels_pu` and `spectra` must have equal length."
            )
        if any(
            b <= a for a, b in zip(self.loading_levels_pu, self.loading_levels_pu[1:])
        ):
            raise ValueError("`loading_levels_pu` must be strictly increasing.")
        return self


class TimeVaryingSpectrum(GridModel):
    kind: Literal["time_varying"] = "time_varying"
    schedule: dict[int, SpectrumPoint] = Field(
        description="Time/profile step index -> spectrum; gaps inherit the previous step."
    )
    profile_ref: Optional[str] = Field(
        default=None, description="External time/profile id."
    )


class RandomSpectrum(GridModel):
    kind: Literal["random"] = "random"
    order_magnitude_mean_pu: dict[int, float] = Field(
        description="Per-order mean magnitude (pu)."
    )
    order_magnitude_std_pu: dict[int, float] = Field(
        description="Per-order magnitude std (pu)."
    )
    order_phase_mean_deg: dict[int, float] = Field(default_factory=dict)
    order_phase_std_deg: dict[int, float] = Field(default_factory=dict)
    distribution: Literal["normal", "lognormal", "uniform"] = Field(default="normal")


class DistributionSpectrum(GridModel):
    kind: Literal["distribution"] = "distribution"
    distribution_ref: str = Field(
        description="Id of the fitted spectral distribution (service)."
    )
    params: dict[str, float] = Field(default_factory=dict)


Spectrum = Annotated[
    Union[
        StaticSpectrum,
        LoadVaryingSpectrum,
        TimeVaryingSpectrum,
        RandomSpectrum,
        DistributionSpectrum,
    ],
    Field(discriminator="kind"),
]


# =============================================================================
# 5. Provenance
# =============================================================================
class Provenance(GridModel):
    source_convention: SourceConvention = Field(
        description="Input convention converted FROM."
    )
    notes: Optional[str] = Field(default=None)
    extra: dict[str, str] = Field(
        default_factory=dict,
        description="Non-recoverable input data kept for round-tripping "
        "(e.g. transformer vector-group label).",
    )


# =============================================================================
# 5b. Conductor geometry (Carson/Deri line constants — geometry -> Z(h), Yc(h))
# =============================================================================
class ConductorPlacement(GridModel):
    """One physical conductor of a line: its position and electrical wire data.

    Positions are in metres in the line cross-section plane: ``x_m`` horizontal,
    ``y_m`` height above ground (> 0). Phase conductors carry the :class:`Phase`
    they belong to (aligned to the line's ``from_phases``); neutral / shield wires
    set ``is_neutral=True`` and are Kron-reduced out during assembly. All physical
    fields are tensor-capable (autograd flows ``geometry -> impedance``).
    """

    phase: Phase = Field(
        description="Phase this conductor serves (ignored if is_neutral)."
    )
    x_m: Num = si_field(
        "Horizontal position in the cross-section.", short="m", long="metre"
    )
    y_m: PosNum = si_field("Height above ground.", short="m", long="metre")
    gmr_m: PosNum = si_field(
        "Geometric mean radius (carries internal inductance).", short="m", long="metre"
    )
    radius_m: PosNum = si_field(
        "Outer radius (for shunt capacitance).", short="m", long="metre"
    )
    r_dc_ohm_per_m: PosNum = si_field(
        "DC resistance (skin-effect reference).", short="Ohm/m", long="ohm per metre"
    )
    is_neutral: bool = Field(
        default=False, description="True = neutral/shield, Kron-reduced out."
    )


class LineGeometry(GridModel):
    """Conductor geometry of a line for the Carson/Deri impedance path.

    When a :class:`Line` carries a ``conductor_geometry``, assembly computes its
    per-frequency ``Z(h)`` (Deri earth return + skin effect) and shunt ``Yc(h)``
    from this geometry instead of the explicit R/L/C matrices (see
    ``docs/pgml/modeling/references/opendss/carson.md``). Phase conductors must cover the line's
    ``from_phases``; extra ``is_neutral`` conductors are reduced out.
    """

    conductors: list[ConductorPlacement] = Field(
        description="Phase + neutral conductors."
    )
    earth_resistivity_ohm_m: PosNum = si_field(
        "Earth resistivity (Deri earth return).",
        short="Ohm*m",
        long="ohm metre",
        default=100.0,
    )
    provenance: Optional[Provenance] = Field(
        default=None, description="Origin of the geometry (e.g. synthesised from R/X)."
    )


# =============================================================================
# 6. Nodes
# =============================================================================
class Node(GridModel):
    id: int = Field(description="Unique node id within the grid.")
    name: Optional[str] = Field(default=None)
    u_rated_v: PosNum = si_field(
        "Rated voltage.",
        short="V",
        long="volt",
        reference="line-to-line (3-phase) / line-to-neutral (1-phase)",
    )
    phases: tuple[Phase, ...] = Field(
        description="Ordered phases present; fixes matrix row/column order for this node."
    )
    zone: NodeZone = Field(
        default=NodeZone.NONE, description="Land-use character (ML stratifier)."
    )
    geometry: Optional[GeoPoint] = Field(
        default=None, description="GeoJSON point in grid CRS."
    )
    tags: dict[str, str] = Field(
        default_factory=dict, description="Arbitrary key/value metadata."
    )

    @field_validator("phases")
    @classmethod
    def _unique_nonempty(cls, v: tuple[Phase, ...]) -> tuple[Phase, ...]:
        if not v:
            raise ValueError("A node must have at least one phase.")
        if len(set(v)) != len(v):
            raise ValueError("Node phases must be unique.")
        return v


# =============================================================================
# 7. Branches (canonical phase-domain form)
# =============================================================================
class BranchBase(GridModel):
    id: int = Field(description="Unique branch id within the grid.")
    name: Optional[str] = Field(default=None)
    from_node: int = Field(description="Id of the 'from' node.")
    to_node: int = Field(description="Id of the 'to' node.")
    from_phases: tuple[Phase, ...] = Field(
        description="Phases of `from_node`, in matrix order."
    )
    to_phases: tuple[Phase, ...] = Field(
        description="Phases of `to_node`, in matrix order."
    )
    in_service: bool = Field(default=True)
    provenance: Optional[Provenance] = Field(default=None)
    tags: dict[str, str] = Field(default_factory=dict)


class Line(BranchBase):
    """Multi-phase line/cable, canonical SI per-length phase-domain form.

    Series impedance and shunt admittance per harmonic ``h``::

        Z_series(h) = (R0 .* r_mult(h) + j*2*pi*h*f0 * L) * length_m
        Y_shunt(h)  = (G + j*2*pi*h*f0 * C) * length_m   (split half to each end)

    Electrical matrices may come from ``type_ref`` (``Grid.types.lines``) instead
    of being given explicitly; a resolver materialises them before assembly.
    """

    component: Literal["line"] = "line"
    length_m: PosNum = si_field("Electrical line length.", short="m", long="metre")
    type_ref: Optional[str] = Field(
        default=None,
        description="Catalog LineType id in Grid.types.lines; resolved/materialised.",
    )
    series_resistance_ohm_per_m: Optional[PerPhaseMatrix] = si_field(
        "Reference-frequency series resistance matrix R0.",
        short="Ohm/m",
        long="ohm per metre",
        reference="per-phase matrix entry",
        default=None,
    )
    series_inductance_h_per_m: Optional[PerPhaseMatrix] = si_field(
        "Series inductance matrix L. X(h)=2*pi*h*f0*L.",
        short="H/m",
        long="henry per metre",
        default=None,
    )
    shunt_capacitance_f_per_m: Optional[PerPhaseMatrix] = si_field(
        "Shunt capacitance matrix C. B(h)=2*pi*h*f0*C.",
        short="F/m",
        long="farad per metre",
        default=None,
    )
    shunt_conductance_s_per_m: Optional[PerPhaseMatrix] = si_field(
        "Shunt conductance matrix G. None = zeros.",
        short="S/m",
        long="siemens per metre",
        default=None,
    )
    resistance_frequency: ResistanceFrequencyModel = Field(
        default_factory=ResistanceFrequencyModel
    )
    geometry: Optional[GeoLineString] = Field(
        default=None, description="GeoJSON routing line."
    )
    conductor_geometry: Optional[LineGeometry] = Field(
        default=None,
        description="Carson conductor geometry; when set, Z(h)/Yc(h) are computed via "
        "the Carson/Deri model (earth return + skin effect) instead of explicit R/L/C.",
    )

    @model_validator(mode="after")
    def _check(self) -> "Line":
        n = len(self.from_phases)
        if len(self.to_phases) != n:
            raise ValueError("Line `from_phases`/`to_phases` must have equal length.")
        core = (
            self.series_resistance_ohm_per_m,
            self.series_inductance_h_per_m,
            self.shunt_capacitance_f_per_m,
        )
        if (
            self.type_ref is None
            and self.conductor_geometry is None
            and any(x is None for x in core)
        ):
            raise ValueError(
                "Line requires `type_ref`, `conductor_geometry`, or explicit series R/L and shunt C."
            )
        for f in (
            "series_resistance_ohm_per_m",
            "series_inductance_h_per_m",
            "shunt_capacitance_f_per_m",
            "shunt_conductance_s_per_m",
        ):
            m = getattr(self, f)
            if m is not None and (len(m) != n or any(len(r) != n for r in m)):
                raise ValueError(f"`{f}` must be {n}x{n} to match phase count.")
        return self


class ComplexTap(GridModel):
    """Off-nominal tap as a complex ratio: magnitude + phase shift (clock/vector group).
    Dimensionless; the one place a complex value is parameterised, as mag+angle.

    The transformer's NOMINAL turns ratio is derived from the rated voltages
    (``u_rated_from_v``/``u_rated_to_v``) and the winding connections, so
    ``ratio_magnitude`` is the OFF-NOMINAL tap deviation (1.0 = nominal, on-tap) and
    ``shift_deg`` carries the vector-group clock angle (``clock·30°``; e.g. 30 for
    Dyn1, 330 for Dyn11)."""

    ratio_magnitude: PosNum = Field(
        description="Off-nominal tap ratio magnitude (1.0 = nominal / on-tap)."
    )
    shift_deg: float = si_field(
        "Phase shift from the vector-group clock (clock·30°). A DISCRETE selector of "
        "the constant winding-incidence topology — a plain float, deliberately not "
        "tensor-capable (the continuous, differentiable tap is `ratio_magnitude`).",
        short="deg",
        long="degree",
        default=0.0,
    )


class GroundingImpedance(GridModel):
    """Neutral-to-ground impedance of a grounded-wye / zigzag winding (r=x=0 = solid)."""

    r_ohm: Num = si_field(
        "Neutral grounding resistance.", short="Ohm", long="ohm", default=0.0
    )
    x_ohm: Num = si_field(
        "Neutral grounding reactance at f0.", short="Ohm", long="ohm", default=0.0
    )


class TransformerZeroSeq(GridModel):
    """Explicit zero-sequence leakage impedance VALUE override, referred to HV. The
    zero-sequence PATH is always derived from winding connections + clock; this
    overrides only the value. None => Z0 = Z1 connected per topology."""

    r0_ohm: Num = si_field(
        "Zero-sequence series resistance.",
        short="Ohm",
        long="ohm",
        reference="referred to HV side",
    )
    x0_ohm: Num = si_field(
        "Zero-sequence series reactance at f0.",
        short="Ohm",
        long="ohm",
        reference="referred to HV side",
    )


class Transformer(BranchBase):
    """Two-winding transformer, canonical phase-domain form. Assembly builds the
    per-phase primitive from connections, clock (via tap.shift_deg), grounding and
    the series/magnetizing branches. Ratings/electrical/connections may come from
    `type_ref` (Grid.types.transformers); a resolver materialises before assembly."""

    component: Literal["transformer"] = "transformer"
    type_ref: Optional[str] = Field(
        default=None, description="Catalog TransformerType id; resolved/materialised."
    )
    s_rated_va: Optional[PosNum] = si_field(
        "Rated apparent power.", short="VA", long="volt-ampere", default=None
    )
    u_rated_from_v: Optional[PosNum] = si_field(
        "Rated voltage, from/HV side.", short="V", long="volt", default=None
    )
    u_rated_to_v: Optional[PosNum] = si_field(
        "Rated voltage, to/LV side.", short="V", long="volt", default=None
    )
    from_connection: Optional[WindingConnection] = Field(
        default=None, description="HV connection."
    )
    to_connection: Optional[WindingConnection] = Field(
        default=None, description="LV connection."
    )
    series_resistance_ohm: Optional[Num] = si_field(
        "Positive-sequence series (leakage) resistance, per-phase scalar.",
        short="Ohm",
        long="ohm",
        reference="referred to the to-side (LV) winding coil",
        default=None,
    )
    series_inductance_h: Optional[Num] = si_field(
        "Positive-sequence series (leakage) inductance. X(h)=2*pi*h*f0*L.",
        short="H",
        long="henry",
        reference="referred to the to-side (LV) winding coil",
        default=None,
    )
    magnetizing_conductance_s: Num = si_field(
        "Core-loss (no-load) conductance G_m.",
        short="S",
        long="siemens",
        reference="referred to HV side",
        default=0.0,
    )
    magnetizing_inductance_h: Optional[Num] = si_field(
        "Magnetizing inductance L_m; B_m(h)=1/(2*pi*h*f0*L_m). None = no branch.",
        short="H",
        long="henry",
        reference="referred to HV side",
        default=None,
    )
    tap: ComplexTap = Field(default_factory=lambda: ComplexTap(ratio_magnitude=1.0))
    from_grounding: Optional[GroundingImpedance] = Field(default=None)
    to_grounding: Optional[GroundingImpedance] = Field(default=None)
    zero_sequence: Optional[TransformerZeroSeq] = Field(default=None)
    harmonic_xr_constant: bool = Field(
        default=False,
        description="False (default): R fixed, X scales with h (X/R grows). True: R scales "
        "with frequency to hold X/R constant (OpenDSS XRConst).",
    )
    resistance_frequency: ResistanceFrequencyModel = Field(
        default_factory=ResistanceFrequencyModel
    )

    @model_validator(mode="after")
    def _check(self) -> "Transformer":
        core = (
            self.s_rated_va,
            self.u_rated_from_v,
            self.u_rated_to_v,
            self.from_connection,
            self.to_connection,
            self.series_resistance_ohm,
            self.series_inductance_h,
        )
        if self.type_ref is None and any(x is None for x in core):
            raise ValueError(
                "Transformer requires either `type_ref` or explicit ratings, "
                "connections and series R/L."
            )
        return self


class Switch(BranchBase):
    """Switch/breaker as a degenerate pi-branch (consistent R/L/G/C convention,
    replacing a single complex z). Closed + all-zero R/L = ideal (nodes fused)."""

    component: Literal["switch"] = "switch"
    closed: bool = Field(description="True = conducting, False = open.")
    resistance_ohm: NonNegNum = si_field(
        "Longitudinal resistance when closed (0 = ideal).",
        short="Ohm",
        long="ohm",
        default=0.0,
    )
    inductance_h: NonNegNum = si_field(
        "Longitudinal inductance when closed.", short="H", long="henry", default=0.0
    )
    shunt_conductance_s: NonNegNum = si_field(
        "Shunt conductance per end.", short="S", long="siemens", default=0.0
    )
    shunt_capacitance_f: NonNegNum = si_field(
        "Shunt capacitance per end.", short="F", long="farad", default=0.0
    )


class ShuntReactor(BranchBase):
    component: Literal["shunt_reactor"] = "shunt_reactor"
    conductance_s: PerPhaseMatrix = si_field(
        "Shunt conductance matrix G.", short="S", long="siemens"
    )
    capacitance_f: PerPhaseMatrix = si_field(
        "Shunt capacitance matrix C (B(h)=2*pi*h*f0*C).", short="F", long="farad"
    )

    @model_validator(mode="after")
    def _check(self) -> "ShuntReactor":
        # Single-terminal shunt: matrices align to `from_phases` (the stamp reads
        # only the from side; `to_node`/`to_phases` conventionally mirror it).
        n = len(self.from_phases)
        for f in ("conductance_s", "capacitance_f"):
            m = getattr(self, f)
            if len(m) != n or any(len(r) != n for r in m):
                raise ValueError(f"`{f}` must be {n}x{n} to match phase count.")
        return self


class GenericBranch(BranchBase):
    component: Literal["generic_branch"] = "generic_branch"
    series_resistance_ohm: PerPhaseMatrix = si_field(
        "Series resistance matrix.", short="Ohm", long="ohm"
    )
    series_inductance_h: PerPhaseMatrix = si_field(
        "Series inductance matrix.", short="H", long="henry"
    )
    shunt_capacitance_from_f: Optional[PerPhaseMatrix] = si_field(
        "Shunt C at from end.", short="F", long="farad", default=None
    )
    shunt_capacitance_to_f: Optional[PerPhaseMatrix] = si_field(
        "Shunt C at to end.", short="F", long="farad", default=None
    )

    @model_validator(mode="after")
    def _check(self) -> "GenericBranch":
        n = len(self.from_phases)
        if len(self.to_phases) != n:
            raise ValueError(
                "GenericBranch `from_phases`/`to_phases` must have equal length."
            )
        for f in (
            "series_resistance_ohm",
            "series_inductance_h",
            "shunt_capacitance_from_f",
            "shunt_capacitance_to_f",
        ):
            m = getattr(self, f)
            if m is not None and (len(m) != n or any(len(r) != n for r in m)):
                raise ValueError(f"`{f}` must be {n}x{n} to match phase count.")
        return self


Branch = Annotated[
    Union[Line, Transformer, Switch, ShuntReactor, GenericBranch],
    Field(discriminator="component"),
]


# =============================================================================
# 8. Appliances (single-terminal)
# =============================================================================
class ApplianceBase(GridModel):
    id: int = Field(description="Unique appliance id within the grid.")
    name: Optional[str] = Field(default=None)
    node: int = Field(description="Id of the node this appliance attaches to.")
    phases: tuple[Phase, ...] = Field(
        description="Connected phases at `node`, in vector order."
    )
    in_service: bool = Field(default=True)
    tags: dict[str, str] = Field(default_factory=dict)


class InjectionAppliance(ApplianceBase):
    """Marker base for single-terminal power-injecting appliances.

    :class:`Load`, :class:`Generator` and :class:`Storage` all resolve to a per-phase
    (P, Q) injection that the assembler folds into the const-Z shunt and the solver
    absorbs through the voltage-dependent ``I_device(V)`` term. Consumers test
    ``isinstance(a, InjectionAppliance)`` to treat the three uniformly; the
    consume/inject sign is positive for a :class:`Load` and negative (injecting) for a
    :class:`Generator` / :class:`Storage`.
    """

    return_path: Literal["auto", "neutral", "ground"] = Field(
        default="auto",
        description="Return-conductor choice of a WYE-connected appliance on a node "
        "that carries an explicit neutral (Phase.N): 'auto' (default) returns through "
        "the neutral whenever the node has one (the historical node-level rule), "
        "'neutral' requires it (assembly raises when the node has no Phase.N), "
        "'ground' pins the return to ground even on a neutral-carrying node (an "
        "OpenDSS `.1.2.3` load on a four-wire bus). Meaningful for WYE only — "
        "assembly rejects a non-'auto' value on a DELTA-connected appliance. On a "
        "node without Phase.N every value behaves as ground.",
    )


class Source(ApplianceBase):
    """Slack / external network equivalent: per-phase Thevenin voltage behind a
    per-phase impedance stored as R and L MATRICES (asymmetric, frequency-correct:
    Z(h)=R + j*2*pi*h*f0*L). Sequence / short-circuit-power inputs convert in."""

    component: Literal["source"] = "source"
    u_ref_v: Vec = si_field(
        "Per-phase reference voltage magnitude.", short="V", long="volt"
    )
    u_angle_deg: Vec = si_field(
        "Per-phase reference voltage angle.", short="deg", long="degree"
    )
    resistance_ohm: PerPhaseMatrix = si_field(
        "Per-phase Thevenin resistance matrix.", short="Ohm", long="ohm"
    )
    inductance_h: PerPhaseMatrix = si_field(
        "Per-phase Thevenin inductance matrix.", short="H", long="henry"
    )
    spectrum: Optional[Spectrum] = Field(
        default=None, description="Optional source distortion."
    )

    @model_validator(mode="after")
    def _check(self) -> "Source":
        n = len(self.phases)
        if len(self.u_ref_v) != n or len(self.u_angle_deg) != n:
            raise ValueError("`u_ref_v`/`u_angle_deg` length must match phase count.")
        for f in ("resistance_ohm", "inductance_h"):
            m = getattr(self, f)
            if len(m) != n or any(len(r) != n for r in m):
                raise ValueError(f"`{f}` must be {n}x{n} to match phase count.")
        return self


class HarmonicShuntModel(GridModel):
    """Norton-equivalent harmonic representation (OpenDSS): the attached Spectrum is
    a current source in parallel with a shunt admittance that is a mix of a SERIES
    R-L and a PARALLEL R-L branch, derived at assembly time from OPERATING-POINT P,Q."""

    series_rl_fraction: float = si_field(
        "Fraction modelled as the SERIES R-L branch vs PARALLEL R-L (OpenDSS %SeriesRL/100). "
        "0.5 = 50/50; 1.0 = all series (max distortion); 0.0 = all parallel (max damping).",
        short="pu",
        long="fraction",
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    neglect_shunt: bool = Field(
        default=False, description="True = pure current source, no shunt."
    )
    motor_x_harm_pu: Optional[float] = si_field(
        "Motor blocked-rotor / sub-transient reactance for the series branch. None = derive "
        "from P,Q. Typical ~0.20.",
        short="pu",
        long="per unit of rated kVA",
        default=None,
    )
    motor_xr_harm: float = Field(
        default=6.0, description="X/R ratio of motor_x_harm_pu at f0."
    )


class ZipCoefficients(GridModel):
    z_p: float = Field(description="Constant-impedance fraction of P.")
    i_p: float = Field(description="Constant-current fraction of P.")
    p_p: float = Field(description="Constant-power fraction of P.")
    z_q: float = Field(description="Constant-impedance fraction of Q.")
    i_q: float = Field(description="Constant-current fraction of Q.")
    p_q: float = Field(description="Constant-power fraction of Q.")

    @model_validator(mode="after")
    def _sums(self) -> "ZipCoefficients":
        if abs(self.z_p + self.i_p + self.p_p - 1.0) > 1e-6:
            raise ValueError("ZIP active coefficients must sum to 1.")
        if abs(self.z_q + self.i_q + self.p_q - 1.0) > 1e-6:
            raise ValueError("ZIP reactive coefficients must sum to 1.")
        return self


# =============================================================================
# 8b. Inverter / DER control laws (operating-point characteristics)
# =============================================================================
class Characteristic(GridModel):
    """Generic monotone-``x`` piecewise lookup ``y = f(x)`` for a control law.

    Used as the curve of an inverter control mode: ``Q(V)`` (Volt-VAr), ``P(V)``
    (Volt-Watt), or ``cosphi(P)``. ``x_values`` is strictly increasing; the value is
    interpolated between samples and (by default) held constant outside the range.
    ``x_values``/``y_values`` are tensor-capable (the float/tensor duality), so a
    curve breakpoint or level is a differentiable leaf — gradients flow to the curve
    shape through the solve. ``linear`` interpolation matches the OpenDSS XYcurve /
    pandapower ``Characteristic``; ``cubic`` gives a smooth (C\\ :sup:`1`) curve for a
    well-behaved gradient at the breakpoints (see ``smoothing`` on the control mode).
    """

    x_values: Vec = Field(description="Strictly increasing breakpoints (x axis).")
    y_values: Vec = Field(description="Curve value at each breakpoint (y axis).")
    interpolation: InterpolationMethod = Field(default=InterpolationMethod.LINEAR)
    extrapolation: ExtrapolationMethod = Field(default=ExtrapolationMethod.CONSTANT)

    @model_validator(mode="after")
    def _check(self) -> "Characteristic":
        x, y = self.x_values, self.y_values
        if _is_arraylike(x) or _is_arraylike(y):
            return self  # tensor curve: caller owns length / monotonicity
        if len(x) != len(y):
            raise ValueError("`x_values` and `y_values` must have equal length.")
        if len(x) < 2:
            raise ValueError("A characteristic needs at least 2 points.")
        if any(b <= a for a, b in zip(x, x[1:])):
            raise ValueError("`x_values` must be strictly increasing.")
        return self


class QReference(str, Enum):
    """Reactive-power base a Volt-VAr ``y`` (pu) scales (OpenDSS ``RefReactivePower``)."""

    RATED = "rated"  # fraction of the inverter apparent-power rating (VARMAX)
    AVAILABLE = "available"  # fraction of the vars available at the present P (VARAVAL)


class InverterControlBase(GridModel):
    """Shared fields of every inverter control mode.

    ``s_rated_va`` is the inverter apparent-power rating that bounds the (P, Q)
    operating point to the capability circle ``P**2 + Q**2 <= s_rated_va**2``; ``None``
    disables the clamp. ``smoothing`` (>= 0) sets the half-width, in the same units as
    the clamped/curve quantity, of a soft saturation / soft breakpoint used so the
    control stays C\\ :sup:`1` for gradient-based use (0 = the exact hard clamp /
    piecewise curve, which matches the reference tools at the operating point but has
    sub-gradients at the kinks). See ``docs/pgml/modeling/der-pv-storage.md`` section 4.3.
    """

    s_rated_va: Optional[PosNum] = si_field(
        "Inverter apparent-power rating bounding the (P, Q) capability circle.",
        short="VA",
        long="volt-ampere",
        default=None,
    )
    smoothing: float = si_field(
        "Soft-saturation / soft-breakpoint half-width for differentiable gradients "
        "(0 = exact hard clamp / piecewise curve).",
        short="pu",
        long="fraction",
        default=0.0,
        ge=0.0,
    )


class ConstantPowerFactorControl(InverterControlBase):
    """Fixed power factor: ``Q = +/- |P| * tan(acos(power_factor))``."""

    kind: Literal["constant_power_factor"] = "constant_power_factor"
    power_factor: float = Field(
        description="Displacement power factor magnitude |cos(phi)| in (0, 1].",
        gt=0.0,
        le=1.0,
    )
    overexcited: bool = Field(
        default=True,
        description="True = inject reactive power (capacitive / overexcited); "
        "False = absorb (inductive / underexcited).",
    )


class ConstantReactivePowerControl(InverterControlBase):
    """Fixed reactive-power setpoint, independent of the active power and voltage."""

    kind: Literal["constant_reactive_power"] = "constant_reactive_power"
    q_var: Num = si_field(
        "Reactive-power setpoint (sign per the appliance injection convention).",
        short="var",
        long="var",
    )


class PowerFactorWattControl(InverterControlBase):
    """Power-factor-vs-active-power characteristic ``cosphi(P)`` (VDE-AR-N 4105)."""

    kind: Literal["power_factor_watt"] = "power_factor_watt"
    characteristic: Characteristic = Field(
        description="x = P / p_ref (pu of available active power), y = SIGNED power "
        "factor (y > 0 = overexcited/inject Q, y < 0 = underexcited/absorb)."
    )
    p_ref_w: Optional[PosNum] = si_field(
        "Active-power reference normalising the curve x axis; None = |p_nom_w|.",
        short="W",
        long="watt",
        default=None,
    )


class VoltVarControl(InverterControlBase):
    """Volt-VAr ``Q(V)``: reactive power as a function of the terminal voltage."""

    kind: Literal["volt_var"] = "volt_var"
    characteristic: Characteristic = Field(
        description="x = |V| (pu of the element nominal voltage), y = Q / q_reference "
        "(pu; y > 0 = inject, y < 0 = absorb)."
    )
    q_reference: QReference = Field(default=QReference.RATED)


class VoltWattControl(InverterControlBase):
    """Volt-Watt ``P(V)``: active-power limit as a function of the terminal voltage."""

    kind: Literal["volt_watt"] = "volt_watt"
    characteristic: Characteristic = Field(
        description="x = |V| (pu of the element nominal voltage), y = active-power "
        "limit as a fraction of the available active power (in [0, 1])."
    )


class VoltVarVoltWattControl(InverterControlBase):
    """Combined Volt-VAr + Volt-Watt (OpenDSS ``InvControl CombiMode=VV_VW``)."""

    kind: Literal["volt_var_volt_watt"] = "volt_var_volt_watt"
    volt_var: Characteristic = Field(description="Q(V) curve (see VoltVarControl).")
    volt_watt: Characteristic = Field(description="P(V) curve (see VoltWattControl).")
    q_reference: QReference = Field(default=QReference.RATED)


InverterControl = Annotated[
    Union[
        ConstantPowerFactorControl,
        ConstantReactivePowerControl,
        PowerFactorWattControl,
        VoltVarControl,
        VoltWattControl,
        VoltVarVoltWattControl,
    ],
    Field(discriminator="kind"),
]


class RegulatedQuantity(str, Enum):
    """Which voltage magnitude a :class:`VoltageRegulation` holds at its setpoint."""

    #: The positive-sequence magnitude ``|V1|`` of the terminal (balanced regulation,
    #: the standard for a machine or a three-phase inverter). For a single-phase
    #: terminal this is the phase magnitude itself.
    POSITIVE_SEQUENCE = "positive_sequence"
    #: Each connected phase holds its own magnitude at the setpoint (independent
    #: single-phase regulators sharing one reactive capability).
    PER_PHASE = "per_phase"


class VoltageRegulation(GridModel):
    """Voltage setpoint of a regulating generator: the PV-terminal model.

    A generator carrying this block is a PV terminal: its ACTIVE power is the
    nameplate / operating-point value, its terminal voltage MAGNITUDE is held at
    ``v_set_pu``, and its REACTIVE power is whatever that takes, bounded by
    ``q_min_var`` / ``q_max_var``. The nonlinear power flow replaces the terminal's
    reactive power-balance row with ``|V|**2 - V_set**2`` and recovers the reactive
    injection from the converged solution; see ``docs/pgml/modeling/der-pv-storage.md``
    section 4.5. It is the model behind pandapower ``net.gen``, power-grid-model's
    ``source``-like voltage control and OpenDSS ``Generator model=3``.

    ``v_set_pu`` is per unit of the HOST NODE's rated voltage, i.e. the regulated
    magnitude in volts is ``v_set_pu * phase_voltage_magnitude(node.u_rated_v,
    len(node.phases))`` (line-to-neutral for a three-phase node, the rated value
    itself below three phases) — the same per-unit base the Volt-VAr characteristic
    and the convergence diagnostics use, and numerically equal to pandapower's
    ``vm_pu``.

    ``q_min_var`` / ``q_max_var`` are TOTAL over the connected phases and follow the
    generator injection convention (positive = injected into the grid), exactly like
    ``q_nom_var``. ``None`` means unbounded on that side. Limits are enforced by
    PV-to-PQ switching in the solver (``solve_power_flow(enforce_q_limits=...)``).

    Mutually exclusive with ``control``: an inverter control law states Q (or a P
    curtailment) as an explicit function of the terminal voltage, while voltage
    regulation states the voltage and leaves Q implicit.
    """

    v_set_pu: PosNum = si_field(
        "Regulated voltage magnitude, per unit of the host node's rated voltage.",
        short="pu",
        long="per unit of the node rated voltage",
        default=1.0,
    )
    q_min_var: Optional[Num] = si_field(
        "Lower reactive limit (TOTAL, injection-positive). None = unbounded.",
        short="var",
        long="var",
        default=None,
    )
    q_max_var: Optional[Num] = si_field(
        "Upper reactive limit (TOTAL, injection-positive). None = unbounded.",
        short="var",
        long="var",
        default=None,
    )
    regulated: RegulatedQuantity = Field(
        default=RegulatedQuantity.POSITIVE_SEQUENCE,
        description="Regulated quantity: the positive-sequence magnitude (balanced, "
        "the standard) or each phase magnitude independently.",
    )

    @model_validator(mode="after")
    def _check(self) -> "VoltageRegulation":
        lo, hi = self.q_min_var, self.q_max_var
        if lo is None or hi is None:
            return self
        if _is_arraylike(lo) or _is_arraylike(hi):
            return self  # tensor limits: caller owns the ordering
        if lo > hi:
            raise ValueError("`q_min_var` must not exceed `q_max_var`.")
        return self


def _check_voltage_regulation(obj) -> None:
    """Validate an appliance's optional ``voltage_regulation`` block.

    Voltage regulation and an inverter ``control`` law are two ways to state the same
    reactive degree of freedom, so they are mutually exclusive.
    """
    reg = getattr(obj, "voltage_regulation", None)
    if reg is None:
        return
    if getattr(obj, "control", None) is not None:
        raise ValueError(
            "Set either `voltage_regulation` (a PV terminal: the voltage magnitude is "
            "held and Q is free) or `control` (an inverter law that states Q as a "
            "function of the voltage), not both."
        )


def _check_control(obj) -> None:
    """Validate an appliance's optional inverter ``control`` block.

    A rated-reference Volt-VAr curve scales the inverter VAr rating, so it needs
    ``s_rated_va``; the available-reference variant derives the base from the present
    active power and does not.
    """
    c = getattr(obj, "control", None)
    if c is None:
        return
    needs_rating = (
        isinstance(c, (VoltVarControl, VoltVarVoltWattControl))
        and c.q_reference == QReference.RATED
    )
    if needs_rating and c.s_rated_va is None:
        raise ValueError(
            "A Volt-VAr control with q_reference='rated' requires `s_rated_va` "
            "(the VAr base it scales)."
        )


def _check_load_connection(obj) -> None:
    """Validate a Load/Generator ``connection`` (``None`` = resolve from config).

    DELTA (line-to-line) needs at least two phases; ZIGZAG is a transformer-only
    winding and is rejected on appliances. See ``docs/pgml/modeling/asymmetric.md``.
    """
    c = obj.connection
    if c is None:
        return
    if c in (WindingConnection.ZIGZAG, WindingConnection.ZIGZAG_GROUNDED):
        raise ValueError(
            "Load/Generator `connection` cannot be zigzag (transformer winding only)."
        )
    if c == WindingConnection.DELTA and len(obj.phases) < 2:
        raise ValueError(
            "`connection`=DELTA requires at least 2 phases (it is line-to-line); "
            "a single-phase load is line-to-neutral (WYE)."
        )


def _check_spectrum_per_phase(obj) -> None:
    """Validate per-phase harmonic spectra on a Load/Generator.

    ``spectrum_per_phase`` (asymmetric distortion) is mutually exclusive with the
    all-phases ``spectrum`` shorthand; its keys must be a subset of the appliance's
    ``phases`` (a phase with no entry injects no harmonics). See
    ``docs/pgml/modeling/asymmetric.md`` §5.
    """
    spp = obj.spectrum_per_phase
    if spp is None:
        return
    if obj.spectrum is not None:
        raise ValueError(
            "Set either `spectrum` (same on all phases) or `spectrum_per_phase` "
            "(asymmetric per phase), not both."
        )
    extra = set(spp) - set(obj.phases)
    if extra:
        raise ValueError(
            f"`spectrum_per_phase` keys {sorted(p.value for p in extra)} are not in "
            f"`phases` {tuple(p.value for p in obj.phases)}."
        )


def _check_per_phase_power(obj) -> None:
    n = len(obj.phases)
    for tot, per, label in (
        (obj.p_nom_w, obj.p_nom_per_phase_w, "p"),
        (obj.q_nom_var, obj.q_nom_per_phase_var, "q"),
    ):
        if per is not None:
            if len(per) != n:
                raise ValueError(
                    f"`{label}_nom_per_phase_*` length must match phase count."
                )
            if _is_arraylike(per) or _is_arraylike(tot):
                continue  # tensor params: caller owns per-phase/total consistency
            if abs(sum(per) - tot) > 1e-6 * max(1.0, abs(tot)):
                raise ValueError(
                    f"`{label}_nom_per_phase_*` must sum to the total `{label}_nom`."
                )


class Load(InjectionAppliance):
    """Consumer. Fundamental behaviour set by ``load_model``; harmonic behaviour by the
    Norton ``harmonic_model``.

    *Power.* ``p_nom_w``/``q_nom_var`` are the TOTAL over the connected phases; the
    optional ``p_nom_per_phase_w``/``q_nom_per_phase_var`` tuples give an ASYMMETRIC
    per-phase nameplate split (length == ``len(phases)``, summing to the totals). A
    per-phase ``operating_point`` overrides either at assembly time. Whether the totals
    are split equally (balanced) or the per-phase values are honored is the CALCULATION
    SYMMETRY decision (config ``calculation.symmetry``; see
    ``docs/pgml/modeling/asymmetric.md`` §1) — it is solver config, not grid data.

    *Connection.* ``connection`` is WYE (each phase to neutral/ground) or DELTA
    (phase-to-phase, line-to-line). ``None`` (default) means "resolve from config" at
    assembly: ``appliance.load.single_phase_connection`` for a 1-phase load,
    ``appliance.load.default_connection`` otherwise (both default to WYE — the LV norm).
    For a WYE load the return path is the node's ``Phase.N`` row when that node carries a
    neutral (the 4-wire case), else ground (3-wire / solidly grounded). DELTA needs
    ``len(phases) >= 2``; ZIGZAG is transformer-only. See
    ``docs/pgml/modeling/asymmetric.md`` §2-4.

    *Harmonics.* ``spectrum`` is one harmonic current source applied to every phase
    (OpenDSS multi-phase Load semantics). ``spectrum_per_phase`` instead gives an
    ASYMMETRIC spectrum per phase (e.g. a single-phase EV charger distorting only phase
    A); the two are mutually exclusive. For a DELTA load a per-phase spectrum key
    identifies the delta branch starting at that phase. See
    ``docs/pgml/modeling/asymmetric.md`` §5.
    """

    component: Literal["load"] = "load"
    connection: Optional[WindingConnection] = Field(
        default=None,
        description="WYE (phase-to-neutral/ground) or DELTA (phase-to-phase). None "
        "resolves from config (appliance.load.{single_phase_,}default_connection).",
    )
    load_model: LoadModel = Field(default=LoadModel.CONST_POWER)
    p_nom_w: Num = si_field(
        "Rated TOTAL active power (nameplate).", short="W", long="watt"
    )
    q_nom_var: Num = si_field(
        "Rated TOTAL reactive power (nameplate).", short="var", long="var", default=0.0
    )
    p_nom_per_phase_w: Optional[Vec] = si_field(
        "Optional asymmetric per-phase active power; must sum to p_nom_w.",
        short="W",
        long="watt",
        default=None,
    )
    q_nom_per_phase_var: Optional[Vec] = si_field(
        "Optional asymmetric per-phase reactive power; must sum to q_nom_var.",
        short="var",
        long="var",
        default=None,
    )
    zip_coefficients: Optional[ZipCoefficients] = Field(
        default=None, description="Required iff load_model==ZIP."
    )
    consumer_type: Optional[ConsumerType] = Field(
        default=None,
        description="Closed device taxonomy (ML categorical; does not drive physics). "
        "Typical loads: household, ev_charging, heat_pump, restaurant, office, "
        "workshop, industrial_drive. None = unspecified.",
    )
    profile_ref: Optional[str] = Field(
        default=None, description="External operating-point profile."
    )
    spectrum: Optional[Spectrum] = Field(
        default=None,
        description="Harmonic current source applied to ALL phases. None = linear. "
        "Mutually exclusive with spectrum_per_phase.",
    )
    spectrum_per_phase: Optional[dict[Phase, Spectrum]] = Field(
        default=None,
        description="Asymmetric per-phase harmonic current sources: maps a connected "
        "Phase to its Spectrum (keys subset of phases; missing phase = no harmonics). "
        "Mutually exclusive with spectrum. DELTA key = the delta branch at that phase.",
    )
    harmonic_model: HarmonicShuntModel = Field(default_factory=HarmonicShuntModel)

    @model_validator(mode="after")
    def _check(self) -> "Load":
        if self.load_model == LoadModel.ZIP and self.zip_coefficients is None:
            raise ValueError("load_model=ZIP requires zip_coefficients.")
        _check_per_phase_power(self)
        _check_load_connection(self)
        _check_spectrum_per_phase(self)
        return self


class Generator(InjectionAppliance):
    """Generation unit (synchronous machine, wind, CHP, or — most commonly on a
    distribution feeder — a grid-following PV/DER inverter). Same rated-vs-operating-point,
    per-phase asymmetry and connection semantics as :class:`Load` (see its docstring and
    ``docs/pgml/modeling/asymmetric.md``); injected-power sign handled at assembly.

    *Inverter control.* The optional ``control`` block makes the operating point a
    function of the local voltage and the available power — constant power factor,
    ``cosphi(P)``, Volt-VAr ``Q(V)``, Volt-Watt ``P(V)``, or a combination, bounded by the
    inverter capability circle. ``p_nom_w`` / the operating point is then the AVAILABLE
    active power (e.g. the PV MPP set by irradiance); the control derives the reactive
    power and any active-power curtailment. The voltage-dependent injection enters the
    nonlinear power-flow residual ``I_device(V)`` and is differentiated by the same IFT
    backward as the const-P/ZIP load (``docs/pgml/modeling/der-pv-storage.md`` section 4)."""

    component: Literal["generator"] = "generator"
    connection: Optional[WindingConnection] = Field(
        default=None,
        description="WYE (phase-to-neutral/ground) or DELTA (phase-to-phase). None "
        "resolves from config (appliance.load.{single_phase_,}default_connection).",
    )
    load_model: LoadModel = Field(default=LoadModel.CONST_POWER)
    p_nom_w: Num = si_field(
        "Rated TOTAL active power (nameplate).", short="W", long="watt"
    )
    q_nom_var: Num = si_field(
        "Rated TOTAL reactive power (nameplate).", short="var", long="var", default=0.0
    )
    p_nom_per_phase_w: Optional[Vec] = si_field(
        "Optional asymmetric per-phase active power; must sum to p_nom_w.",
        short="W",
        long="watt",
        default=None,
    )
    q_nom_per_phase_var: Optional[Vec] = si_field(
        "Optional asymmetric per-phase reactive power; must sum to q_nom_var.",
        short="var",
        long="var",
        default=None,
    )
    zip_coefficients: Optional[ZipCoefficients] = Field(default=None)
    control: Optional[InverterControl] = Field(
        default=None,
        description="Optional inverter control law (constant power factor, cosphi(P), "
        "Volt-VAr Q(V), Volt-Watt P(V), or combined). None = a plain const-P/ZIP "
        "injection. Honored by the nonlinear power flow (`solve_power_flow` / "
        "`solve_harmonic_flow`); the linear const-Z assembler uses the base P/Q.",
    )
    voltage_regulation: Optional[VoltageRegulation] = Field(
        default=None,
        description="Optional voltage setpoint making this generator a PV terminal: "
        "the terminal voltage magnitude is held at `v_set_pu` and the reactive power "
        "is free within `q_min_var`/`q_max_var`. None = a plain PQ injection. "
        "Mutually exclusive with `control`. Honored by the nonlinear power flow "
        "(`solve_power_flow`, which solves a grid with a PV terminal by Newton); the "
        "linear const-Z assembler uses the base P/Q.",
    )
    consumer_type: Optional[ConsumerType] = Field(
        default=None,
        description="Closed device taxonomy (ML categorical; does not drive physics). "
        "Typical generators: pv, wind, chp, diesel_genset. None = unspecified.",
    )
    profile_ref: Optional[str] = Field(default=None)
    spectrum: Optional[Spectrum] = Field(
        default=None,
        description="e.g. inverter spectrum, applied to ALL phases. Mutually "
        "exclusive with spectrum_per_phase.",
    )
    spectrum_per_phase: Optional[dict[Phase, Spectrum]] = Field(
        default=None,
        description="Asymmetric per-phase harmonic current sources (keys subset of "
        "phases; missing phase = no harmonics). Mutually exclusive with spectrum.",
    )
    harmonic_model: HarmonicShuntModel = Field(default_factory=HarmonicShuntModel)

    @model_validator(mode="after")
    def _check(self) -> "Generator":
        if self.load_model == LoadModel.ZIP and self.zip_coefficients is None:
            raise ValueError("load_model=ZIP requires zip_coefficients.")
        _check_per_phase_power(self)
        _check_load_connection(self)
        _check_spectrum_per_phase(self)
        _check_control(self)
        _check_voltage_regulation(self)
        return self


class Storage(InjectionAppliance):
    """Battery / energy storage as a bidirectional inverter injection.

    *Sign convention* (generator-consistent, so storage shares the injection path with
    :class:`Generator`): ``p_nom_w`` is the SIGNED active-power setpoint — ``> 0`` =
    DISCHARGING (injecting into the grid), ``< 0`` = CHARGING (drawing from it). The
    reactive setpoint / inverter ``control`` follow the same convention as a generator.

    *Snapshot vs. state.* At a power-flow snapshot the storage is a signed (P, Q)
    injection identical to a :class:`Generator` (same ``load_model`` / connection /
    per-phase / control / harmonic semantics). The energy-state fields
    (``energy_capacity_wh``, ``soc``, ``soc_min``/``soc_max``, the charge/discharge
    efficiencies, ``p_rated_w``) are INERT in the solve — they are not read by the
    assembler or the solver, matching pandapower ``storage.soc_percent`` and the OpenDSS
    ``Storage`` element. State-of-charge integration and the dispatch rule live in the
    time-series / scenario layer (``pgml.scenarios``), which resolves them into the
    per-step ``p_nom_w`` / operating point the solver consumes. See
    ``docs/pgml/modeling/der-pv-storage.md`` section 4.4.
    """

    component: Literal["storage"] = "storage"
    connection: Optional[WindingConnection] = Field(
        default=None,
        description="WYE (phase-to-neutral/ground) or DELTA (phase-to-phase). None "
        "resolves from config (appliance.load.{single_phase_,}default_connection).",
    )
    load_model: LoadModel = Field(default=LoadModel.CONST_POWER)
    p_nom_w: Num = si_field(
        "Signed active-power setpoint: > 0 discharging (inject), < 0 charging (draw).",
        short="W",
        long="watt",
    )
    q_nom_var: Num = si_field(
        "Reactive-power setpoint (injection convention).",
        short="var",
        long="var",
        default=0.0,
    )
    p_nom_per_phase_w: Optional[Vec] = si_field(
        "Optional asymmetric per-phase active setpoint; must sum to p_nom_w.",
        short="W",
        long="watt",
        default=None,
    )
    q_nom_per_phase_var: Optional[Vec] = si_field(
        "Optional asymmetric per-phase reactive setpoint; must sum to q_nom_var.",
        short="var",
        long="var",
        default=None,
    )
    zip_coefficients: Optional[ZipCoefficients] = Field(default=None)
    control: Optional[InverterControl] = Field(
        default=None,
        description="Optional inverter control law (same union as Generator.control).",
    )
    energy_capacity_wh: Optional[PosNum] = si_field(
        "Usable energy capacity (nameplate). Inert in the solve; used by dispatch.",
        short="Wh",
        long="watt-hour",
        default=None,
    )
    soc: Optional[float] = si_field(
        "State of charge as a fraction in [0, 1]. Inert in the solve.",
        short="pu",
        long="fraction",
        default=None,
        ge=0.0,
        le=1.0,
    )
    soc_min: float = si_field(
        "Minimum allowed state of charge (dispatch reserve).",
        short="pu",
        long="fraction",
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    soc_max: float = si_field(
        "Maximum allowed state of charge.",
        short="pu",
        long="fraction",
        default=1.0,
        ge=0.0,
        le=1.0,
    )
    efficiency_charge: float = si_field(
        "One-way charging efficiency.",
        short="pu",
        long="fraction",
        default=1.0,
        gt=0.0,
        le=1.0,
    )
    efficiency_discharge: float = si_field(
        "One-way discharging efficiency.",
        short="pu",
        long="fraction",
        default=1.0,
        gt=0.0,
        le=1.0,
    )
    p_rated_w: Optional[PosNum] = si_field(
        "Inverter active-power rating bounding |p_nom_w|. Inert in the solve.",
        short="W",
        long="watt",
        default=None,
    )
    consumer_type: Optional[ConsumerType] = Field(
        default=None,
        description="Closed device taxonomy (ML categorical). Typically battery.",
    )
    profile_ref: Optional[str] = Field(
        default=None, description="External operating-point / setpoint profile."
    )
    dispatch_ref: Optional[str] = Field(
        default=None,
        description="External dispatch rule/policy id resolved by the time-series layer.",
    )
    spectrum: Optional[Spectrum] = Field(
        default=None,
        description="Inverter harmonic spectrum, applied to ALL phases. Mutually "
        "exclusive with spectrum_per_phase.",
    )
    spectrum_per_phase: Optional[dict[Phase, Spectrum]] = Field(
        default=None,
        description="Asymmetric per-phase harmonic current sources. Mutually "
        "exclusive with spectrum.",
    )
    harmonic_model: HarmonicShuntModel = Field(default_factory=HarmonicShuntModel)

    @model_validator(mode="after")
    def _check(self) -> "Storage":
        if self.load_model == LoadModel.ZIP and self.zip_coefficients is None:
            raise ValueError("load_model=ZIP requires zip_coefficients.")
        if self.soc_min > self.soc_max:
            raise ValueError("`soc_min` must not exceed `soc_max`.")
        _check_per_phase_power(self)
        _check_load_connection(self)
        _check_spectrum_per_phase(self)
        _check_control(self)
        return self


class ShuntAppliance(ApplianceBase):
    component: Literal["shunt"] = "shunt"
    conductance_s: Vec = si_field(
        "Per-phase shunt conductance G.", short="S", long="siemens"
    )
    capacitance_f: Vec = si_field(
        "Per-phase shunt capacitance C (B(h)=2*pi*h*f0*C).", short="F", long="farad"
    )
    connection: WindingConnection = Field(
        default=WindingConnection.WYE,
        description="WYE (default): each element G/C connects its phase to ground "
        "(the historical behavior). DELTA: element k connects phase k to phase k+1 "
        "(cyclic over the appliance's phases; requires >= 2 phases) — a delta "
        "capacitor bank. Zigzag is rejected.",
    )

    @model_validator(mode="after")
    def _check_connection(self) -> "ShuntAppliance":
        if self.connection in (
            WindingConnection.ZIGZAG,
            WindingConnection.ZIGZAG_GROUNDED,
        ):
            raise ValueError("ShuntAppliance does not support zigzag connections.")
        if self.connection is WindingConnection.DELTA and len(self.phases) < 2:
            raise ValueError(
                "connection=DELTA requires at least 2 phases (a delta branch is a "
                "phase-to-phase element)."
            )
        return self


Appliance = Annotated[
    Union[Source, Load, Generator, Storage, ShuntAppliance],
    Field(discriminator="component"),
]


# =============================================================================
# 9. Standard-type catalog (authoring layer; materialised before assembly)
# =============================================================================
class LineType(GridModel):
    """Catalog line type: per-length electrical parameters, no connectivity/length."""

    id: str = Field(description="Unique catalog id, referenced by Line.type_ref.")
    n_phases: int = Field(
        description="Phase count this type's matrices are defined for.", ge=1
    )
    series_resistance_ohm_per_m: PerPhaseMatrix = si_field(
        "Series resistance matrix R0.", short="Ohm/m", long="ohm per metre"
    )
    series_inductance_h_per_m: PerPhaseMatrix = si_field(
        "Series inductance matrix L.", short="H/m", long="henry per metre"
    )
    shunt_capacitance_f_per_m: PerPhaseMatrix = si_field(
        "Shunt capacitance matrix C.", short="F/m", long="farad per metre"
    )
    shunt_conductance_s_per_m: Optional[PerPhaseMatrix] = si_field(
        "Shunt conductance matrix G.",
        short="S/m",
        long="siemens per metre",
        default=None,
    )
    resistance_frequency: ResistanceFrequencyModel = Field(
        default_factory=ResistanceFrequencyModel
    )
    max_i_a: Optional[float] = si_field(
        "Rated ampacity.", short="A", long="ampere", default=None
    )


class TransformerType(GridModel):
    """Catalog transformer type: ratings, connections, electrical; no connectivity."""

    id: str = Field(
        description="Unique catalog id, referenced by Transformer.type_ref."
    )
    s_rated_va: float = si_field(
        "Rated apparent power.", short="VA", long="volt-ampere", gt=0.0
    )
    u_rated_from_v: float = si_field(
        "Rated voltage, HV side.", short="V", long="volt", gt=0.0
    )
    u_rated_to_v: float = si_field(
        "Rated voltage, LV side.", short="V", long="volt", gt=0.0
    )
    from_connection: WindingConnection = Field(description="HV connection.")
    to_connection: WindingConnection = Field(description="LV connection.")
    series_resistance_ohm: float = si_field(
        "Series leakage resistance.",
        short="Ohm",
        long="ohm",
        reference="referred to HV side",
    )
    series_inductance_h: float = si_field(
        "Series leakage inductance.",
        short="H",
        long="henry",
        reference="referred to HV side",
    )
    magnetizing_conductance_s: float = si_field(
        "Core-loss conductance.", short="S", long="siemens", default=0.0
    )
    magnetizing_inductance_h: Optional[float] = si_field(
        "Magnetizing inductance.", short="H", long="henry", default=None
    )
    zero_sequence: Optional[TransformerZeroSeq] = Field(default=None)
    harmonic_xr_constant: bool = Field(default=False)
    resistance_frequency: ResistanceFrequencyModel = Field(
        default_factory=ResistanceFrequencyModel
    )
    nominal_tap: ComplexTap = Field(
        default_factory=lambda: ComplexTap(ratio_magnitude=1.0),
        description="Nominal ratio/clock phase shift of the vector group.",
    )


class TypeLibrary(GridModel):
    lines: dict[str, LineType] = Field(default_factory=dict)
    transformers: dict[str, TransformerType] = Field(default_factory=dict)


# =============================================================================
# 10. Input-convention DTOs (consumed by converters -> canonical objects)
# =============================================================================
class LineSequenceInput(GridModel):
    """Symmetrical-component line input. self=(Z0+2*Z1)/3, mutual=(Z0-Z1)/3."""

    from_node: int
    to_node: int
    phases: tuple[Phase, ...]
    length_m: float = Field(gt=0.0)
    r1_ohm_per_m: float
    x1_ohm_per_m: float
    c1_f_per_m: float
    r0_ohm_per_m: float
    x0_ohm_per_m: float
    c0_f_per_m: float


class SourceSequenceInput(GridModel):
    """Source from sequence impedances (Z2 may differ from Z1 for machines).
    Zs=(Z0+Z1+Z2)/3 (self), Zm=(Z0-Z1)/3 (mutual)."""

    node: int
    phases: tuple[Phase, ...]
    u_ref_v: tuple[float, ...]
    u_angle_deg: tuple[float, ...]
    r1_ohm: float
    x1_ohm: float
    r0_ohm: float
    x0_ohm: float
    r2_ohm: Optional[float] = Field(default=None, description="Defaults to R1.")
    x2_ohm: Optional[float] = Field(default=None, description="Defaults to X1.")


class SourceShortCircuitInput(GridModel):
    node: int
    phases: tuple[Phase, ...]
    u_nom_v: float = Field(gt=0.0)
    u_ref_pu: float = Field(default=1.0)
    angle_deg: float = Field(default=0.0)
    sk_va: float = Field(description="Three-phase short-circuit power.", gt=0.0)
    rx_ratio: float = Field(description="R/X of the positive-sequence impedance.")
    z0_z1_ratio: float = Field(default=1.0)


class TransformerShortCircuitInput(GridModel):
    from_node: int
    to_node: int
    s_rated_va: float = Field(gt=0.0)
    u_rated_from_v: float = Field(gt=0.0)
    u_rated_to_v: float = Field(gt=0.0)
    uk_percent: float = Field(description="Short-circuit voltage u_k [%].", gt=0.0)
    pk_w: float = Field(description="Short-circuit (copper) loss [W].", ge=0.0)
    i0_percent: float = Field(default=0.0, ge=0.0)
    p0_w: float = Field(default=0.0, ge=0.0)
    uk0_percent: Optional[float] = Field(
        default=None, description="Zero-seq u_k override [%]."
    )
    pk0_w: Optional[float] = Field(
        default=None, description="Zero-seq copper loss override [W]."
    )
    vector_group: Optional[str] = Field(
        default=None, description="e.g. 'Dyn11'. Kept in Provenance."
    )
    tap_position: int = Field(default=0)
    tap_step_percent: float = Field(default=0.0)
    tap_step_degree: float = Field(default=0.0)


class TransformerImpedanceInput(GridModel):
    from_node: int
    to_node: int
    s_rated_va: float = Field(gt=0.0)
    u_rated_from_v: float = Field(gt=0.0)
    u_rated_to_v: float = Field(gt=0.0)
    series_resistance_ohm: float
    series_reactance_ohm: float
    r0_ohm: Optional[float] = Field(default=None)
    x0_ohm: Optional[float] = Field(default=None)
    magnetizing_conductance_s: float = Field(default=0.0)
    magnetizing_susceptance_s: float = Field(default=0.0)
    from_connection: WindingConnection = WindingConnection.DELTA
    to_connection: WindingConnection = WindingConnection.WYE_GROUNDED
    phase_shift_deg: float = Field(default=0.0)
    vector_group: Optional[str] = Field(default=None)


# =============================================================================
# 11. Measurement devices (instrumentation metadata; never on the autograd tape)
# =============================================================================
class CurrentChannel(GridModel):
    """One current channel (CT) of a :class:`MeasurementDevice`: a metered branch.

    The measured current is the branch current at the terminal of ``branch`` that
    connects to the device's node — the branch must be incident to that node
    (validated at :class:`Grid` level). ``terminal`` is normally inferred from the
    device's node; an explicit value is only needed to disambiguate a self-loop
    branch (``from_node == to_node``).
    """

    branch: int = Field(description="Id of the metered branch.")
    phases: Optional[tuple[Phase, ...]] = Field(
        default=None,
        description=(
            "Measured phases of the branch terminal at the device's node; "
            "``None`` = every phase of that terminal."
        ),
    )
    terminal: Optional[Literal["from", "to"]] = Field(
        default=None,
        description=(
            "Which branch terminal is metered. ``None`` (default) infers the "
            "terminal touching the device's node; set explicitly only for a "
            "self-loop branch."
        ),
    )

    @field_validator("phases")
    @classmethod
    def _unique_phases(
        cls, v: Optional[tuple[Phase, ...]]
    ) -> Optional[tuple[Phase, ...]]:
        if v is not None and len(set(v)) != len(v):
            raise ValueError("CurrentChannel phases must be unique.")
        return v


class MeasurementDevice(GridModel):
    """A physical measurement instrument installed at a node.

    Inert instrumentation metadata: which electrical quantities are recorded
    where (and, for currents, on which incident branches), plus the device
    identity, acquisition settings, accuracy class and connectivity information a
    measurement-acquisition service needs to reach the instrument. Nothing here
    enters the admittance assembly or the solver — consumers are the ML
    measurement models (sensor placement, noise modelling keyed by
    ``accuracy_class``) and external acquisition services.

    Voltage is measured at the device's ``node`` (on ``phases``); currents are
    measured per :class:`CurrentChannel` on branches incident to that node —
    the physical picture is a meter cabinet at a bus with current transformers
    on its feeders. Devices typically ATTACH to an existing grid description
    (e.g. a converted DSO network plan) via
    :meth:`Grid.attach_measurement_devices`, which re-runs the grid integrity
    validation atomically.

    All fields are plain python values (no tensor duality) — instrumentation
    metadata is never a gradient leaf.
    """

    id: int = Field(description="Unique measurement-device id within the grid.")
    name: Optional[str] = Field(
        default=None, description="Instance label, e.g. 'PQ meter substation A'."
    )
    node: int = Field(
        description="Id of the node (bus) the device is installed at; voltage is "
        "measured here."
    )
    phases: Optional[tuple[Phase, ...]] = Field(
        default=None,
        description="Measured phases at `node`; ``None`` = every phase of the node.",
    )
    in_service: bool = Field(default=True)
    measured_quantities: tuple[MeasuredQuantity, ...] = Field(
        default=(MeasuredQuantity.VOLTAGE,),
        description="Quantities the device records.",
    )
    max_harmonic_order: Optional[int] = Field(
        default=None,
        ge=1,
        description="Highest harmonic order the device resolves (e.g. 50 per "
        "IEC 61000-4-7); ``None`` = unspecified.",
    )
    max_current_channels: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of current channels (CTs) the hardware supports.",
    )
    current_channels: list[CurrentChannel] = Field(
        default_factory=list,
        description="The metered incident branches (at most `max_current_channels`).",
    )
    manufacturer: Optional[str] = Field(default=None)
    model: Optional[str] = Field(
        default=None,
        description="Product / device name (`name` is the instance label).",
    )
    supported_averaging_intervals_s: Optional[list[float]] = si_field(
        "Averaging intervals the device supports (e.g. [1, 10, 600] = 1 s / 10 s "
        "/ 10 min aggregation).",
        short="s",
        long="second",
        default=None,
    )
    averaging_interval_s: Optional[float] = si_field(
        "Configured averaging interval; must be one of "
        "`supported_averaging_intervals_s` when both are given.",
        short="s",
        long="second",
        default=None,
        gt=0.0,
    )
    accuracy_class: Optional[str] = Field(
        default=None,
        description="Accuracy / performance class, e.g. '0.2S', '0.5S' (IEC 62053) "
        "or 'A', 'S' (IEC 61000-4-30). Categorical: keys the measurement-noise "
        "model of the ML layer; pgml attaches no numeric interpretation.",
    )
    connection: Optional[dict[str, Any]] = Field(
        default=None,
        description="Structured connectivity information for the acquisition "
        "service, free-form JSON by design (transports vary). Convention: a "
        "'kind' discriminator, e.g. {'kind': 'modbus_tcp', 'host': '10.0.0.5', "
        "'port': 502, 'unit_id': 1}. Interpreted by the external acquisition "
        "service, never by pgml.",
    )
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("phases")
    @classmethod
    def _unique_phases(
        cls, v: Optional[tuple[Phase, ...]]
    ) -> Optional[tuple[Phase, ...]]:
        if v is not None:
            if not v:
                raise ValueError("Device phases must be non-empty when given.")
            if len(set(v)) != len(v):
                raise ValueError("Device phases must be unique.")
        return v

    @field_validator("measured_quantities")
    @classmethod
    def _quantities(
        cls, v: tuple[MeasuredQuantity, ...]
    ) -> tuple[MeasuredQuantity, ...]:
        if not v:
            raise ValueError("A device must measure at least one quantity.")
        if len(set(v)) != len(v):
            raise ValueError("Measured quantities must be unique.")
        return v

    @field_validator("supported_averaging_intervals_s")
    @classmethod
    def _supported_intervals(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is not None:
            if not v:
                raise ValueError(
                    "Supported averaging intervals must be non-empty when given."
                )
            if any(t <= 0.0 for t in v):
                raise ValueError("Averaging intervals must be positive.")
            if len(set(v)) != len(v):
                raise ValueError("Supported averaging intervals must be unique.")
        return v

    @model_validator(mode="after")
    def _check(self) -> "MeasurementDevice":
        if (
            self.averaging_interval_s is not None
            and self.supported_averaging_intervals_s is not None
            and self.averaging_interval_s not in self.supported_averaging_intervals_s
        ):
            raise ValueError(
                f"averaging_interval_s={self.averaging_interval_s} is not one of "
                f"the supported intervals {self.supported_averaging_intervals_s}."
            )
        if (
            self.max_current_channels is not None
            and len(self.current_channels) > self.max_current_channels
        ):
            raise ValueError(
                f"{len(self.current_channels)} current channels exceed the "
                f"device's {self.max_current_channels} supported channels."
            )
        if len({c.branch for c in self.current_channels}) != len(self.current_channels):
            raise ValueError("Current channels must meter distinct branches.")
        if self.current_channels and (
            MeasuredQuantity.CURRENT not in self.measured_quantities
        ):
            raise ValueError(
                "Current channels require 'current' in measured_quantities."
            )
        return self


# =============================================================================
# 12. Grid container
# =============================================================================
class GridMetadata(GridModel):
    name: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    crs: Optional[str] = Field(
        default=None, description="Grid-wide CRS id for all geometry."
    )
    environment: Optional[GridEnvironment] = Field(
        default=None, description="Settlement character (ML feature)."
    )
    tags: dict[str, str] = Field(default_factory=dict)


class Grid(GridModel):
    """Complete grid: authored/stored single-grid object form. `base_frequency_hz`
    sets f0; harmonic ORDERS to solve are a simulation parameter, not stored here.
    Elements referencing `type_ref` must be materialised against `types` before
    assembly."""

    base_frequency_hz: float = si_field(
        "System fundamental f0.", short="Hz", long="hertz", default=50.0, gt=0.0
    )
    nodes: list[Node] = Field(description="All buses.")
    branches: list[Branch] = Field(default_factory=list)
    appliances: list[Appliance] = Field(default_factory=list)
    measurement_devices: list[MeasurementDevice] = Field(
        default_factory=list,
        description="Installed measurement instrumentation (inert metadata; "
        "typically attached to a converted grid by assignment).",
    )
    types: TypeLibrary = Field(
        default_factory=TypeLibrary, description="Standard-type catalog."
    )
    metadata: GridMetadata = Field(default_factory=GridMetadata)

    @model_validator(mode="after")
    def _integrity(self) -> "Grid":
        node_ids = {n.id for n in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("Node ids must be unique.")
        for b in self.branches:
            for ref in (b.from_node, b.to_node):
                if ref not in node_ids:
                    raise ValueError(f"Branch {b.id} references missing node {ref}.")
            if (
                isinstance(b, Line)
                and b.type_ref is not None
                and b.type_ref not in self.types.lines
            ):
                raise ValueError(
                    f"Line {b.id} references unknown line type '{b.type_ref}'."
                )
            if (
                isinstance(b, Transformer)
                and b.type_ref is not None
                and b.type_ref not in self.types.transformers
            ):
                raise ValueError(
                    f"Transformer {b.id} references unknown type '{b.type_ref}'."
                )
        for a in self.appliances:
            if a.node not in node_ids:
                raise ValueError(f"Appliance {a.id} references missing node {a.node}.")
        self._check_measurement_devices(node_ids)
        return self

    def attach_measurement_devices(self, devices: "list[MeasurementDevice]") -> "Grid":
        """Attach measurement devices to this grid, re-validating atomically.

        The intended authoring flow for instrumentation: a converted grid
        description (e.g. a DSO-provided network plan) usually arrives without
        measurement information, and the devices are mapped to their locations
        and attached afterwards. Appends ``devices`` to
        ``measurement_devices`` and re-runs the grid integrity validation; on a
        validation failure the previous device list is restored, so a rejected
        attach never leaves the grid inconsistent (a plain field assignment
        would — pydantic keeps the assigned value even when a model validator
        raises). Returns ``self`` for chaining.
        """
        previous = list(self.measurement_devices)
        try:
            self.measurement_devices = [*previous, *devices]
        except Exception:
            self.measurement_devices = previous
            raise
        return self

    def _check_measurement_devices(self, node_ids: set) -> None:
        """Cross-reference integrity of the installed measurement devices."""
        device_ids = {d.id for d in self.measurement_devices}
        if len(device_ids) != len(self.measurement_devices):
            raise ValueError("Measurement-device ids must be unique.")
        nodes_by_id = {n.id: n for n in self.nodes}
        branches_by_id = {b.id: b for b in self.branches}
        for d in self.measurement_devices:
            node = nodes_by_id.get(d.node)
            if node is None:
                raise ValueError(
                    f"Measurement device {d.id} references missing node {d.node}."
                )
            if d.phases is not None and not set(d.phases) <= set(node.phases):
                raise ValueError(
                    f"Measurement device {d.id} measures phases {tuple(d.phases)} "
                    f"not present at node {d.node} (phases {tuple(node.phases)})."
                )
            for ch in d.current_channels:
                b = branches_by_id.get(ch.branch)
                if b is None:
                    raise ValueError(
                        f"Measurement device {d.id} meters missing branch {ch.branch}."
                    )
                if d.node not in (b.from_node, b.to_node):
                    raise ValueError(
                        f"Measurement device {d.id} meters branch {ch.branch}, "
                        f"which is not incident to its node {d.node}."
                    )
                if ch.terminal is None and b.from_node == b.to_node:
                    raise ValueError(
                        f"Measurement device {d.id}: branch {ch.branch} is a "
                        "self-loop; the metered terminal must be set explicitly."
                    )
                if ch.terminal == "from" and b.from_node != d.node:
                    raise ValueError(
                        f"Measurement device {d.id}: branch {ch.branch} 'from' "
                        f"terminal is at node {b.from_node}, not the device node "
                        f"{d.node}."
                    )
                if ch.terminal == "to" and b.to_node != d.node:
                    raise ValueError(
                        f"Measurement device {d.id}: branch {ch.branch} 'to' "
                        f"terminal is at node {b.to_node}, not the device node "
                        f"{d.node}."
                    )
                terminal = ch.terminal or ("from" if b.from_node == d.node else "to")
                terminal_phases = b.from_phases if terminal == "from" else b.to_phases
                if ch.phases is not None and not set(ch.phases) <= set(terminal_phases):
                    raise ValueError(
                        f"Measurement device {d.id}: channel on branch "
                        f"{ch.branch} measures phases {tuple(ch.phases)} not "
                        f"present at its {terminal} terminal "
                        f"(phases {tuple(terminal_phases)})."
                    )


__all__ = [
    "Phase",
    "WindingConnection",
    "LoadModel",
    "NodeZone",
    "GridEnvironment",
    "InterpolationMethod",
    "ExtrapolationMethod",
    "SourceConvention",
    "ConsumerType",
    "GeoPoint",
    "GeoLineString",
    "ConstantParam",
    "AnalyticParam",
    "CurveParam",
    "EquationParam",
    "FrequencyParam",
    "ResistanceFrequencyModel",
    "HarmonicComponent",
    "SpectrumPoint",
    "StaticSpectrum",
    "LoadVaryingSpectrum",
    "TimeVaryingSpectrum",
    "RandomSpectrum",
    "DistributionSpectrum",
    "Spectrum",
    "Provenance",
    "Node",
    "BranchBase",
    "Line",
    "ComplexTap",
    "GroundingImpedance",
    "TransformerZeroSeq",
    "Transformer",
    "Switch",
    "ShuntReactor",
    "GenericBranch",
    "Branch",
    "ApplianceBase",
    "Source",
    "HarmonicShuntModel",
    "ZipCoefficients",
    "Characteristic",
    "QReference",
    "InverterControlBase",
    "ConstantPowerFactorControl",
    "ConstantReactivePowerControl",
    "PowerFactorWattControl",
    "VoltVarControl",
    "VoltWattControl",
    "VoltVarVoltWattControl",
    "InverterControl",
    "RegulatedQuantity",
    "VoltageRegulation",
    "InjectionAppliance",
    "Load",
    "Generator",
    "Storage",
    "ShuntAppliance",
    "Appliance",
    "MeasuredQuantity",
    "CurrentChannel",
    "MeasurementDevice",
    "LineType",
    "TransformerType",
    "TypeLibrary",
    "LineSequenceInput",
    "SourceSequenceInput",
    "SourceShortCircuitInput",
    "TransformerShortCircuitInput",
    "TransformerImpedanceInput",
    "GridMetadata",
    "Grid",
    "si_field",
]
