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
  ``references/asymmetric_modeling.md`` for the cross-tool basis and citations.

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
    law: str = Field(description="Equation-registry id of the law f -> value.")
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
    equation_id: str = Field(description="Equation-registry id.")
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
    ``references/opendss/carson.md``). Phase conductors must cover the line's
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
    Dimensionless; the one place a complex value is parameterised, as mag+angle."""

    ratio_magnitude: PosNum = Field(description="Tap ratio magnitude (1.0 = nominal).")
    shift_deg: Num = si_field(
        "Phase shift from tap/vector group.", short="deg", long="degree", default=0.0
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
        reference="referred to HV side",
        default=None,
    )
    series_inductance_h: Optional[Num] = si_field(
        "Positive-sequence series (leakage) inductance. X(h)=2*pi*h*f0*L.",
        short="H",
        long="henry",
        reference="referred to HV side",
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


def _check_load_connection(obj) -> None:
    """Validate a Load/Generator ``connection`` (``None`` = resolve from config).

    DELTA (line-to-line) needs at least two phases; ZIGZAG is a transformer-only
    winding and is rejected on appliances. See ``references/asymmetric_modeling.md``.
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
    ``references/asymmetric_modeling.md`` §5.
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


class Load(ApplianceBase):
    """Consumer. Fundamental behaviour set by ``load_model``; harmonic behaviour by the
    Norton ``harmonic_model``.

    *Power.* ``p_nom_w``/``q_nom_var`` are the TOTAL over the connected phases; the
    optional ``p_nom_per_phase_w``/``q_nom_per_phase_var`` tuples give an ASYMMETRIC
    per-phase nameplate split (length == ``len(phases)``, summing to the totals). A
    per-phase ``operating_point`` overrides either at assembly time. Whether the totals
    are split equally (balanced) or the per-phase values are honored is the CALCULATION
    SYMMETRY decision (config ``calculation.symmetry``; see
    ``references/asymmetric_modeling.md`` §1) — it is solver config, not grid data.

    *Connection.* ``connection`` is WYE (each phase to neutral/ground) or DELTA
    (phase-to-phase, line-to-line). ``None`` (default) means "resolve from config" at
    assembly: ``appliance.load.single_phase_connection`` for a 1-phase load,
    ``appliance.load.default_connection`` otherwise (both default to WYE — the LV norm).
    For a WYE load the return path is the node's ``Phase.N`` row when that node carries a
    neutral (the 4-wire case), else ground (3-wire / solidly grounded). DELTA needs
    ``len(phases) >= 2``; ZIGZAG is transformer-only. See
    ``references/asymmetric_modeling.md`` §2-4.

    *Harmonics.* ``spectrum`` is one harmonic current source applied to every phase
    (OpenDSS multi-phase Load semantics). ``spectrum_per_phase`` instead gives an
    ASYMMETRIC spectrum per phase (e.g. a single-phase EV charger distorting only phase
    A); the two are mutually exclusive. For a DELTA load a per-phase spectrum key
    identifies the delta branch starting at that phase. See
    ``references/asymmetric_modeling.md`` §5.
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
    consumer_type: Optional[str] = Field(
        default=None,
        description="Open vocabulary (snake_case). Recommended: household, ev_charging, heat_pump, "
        "restaurant, office, workshop, pv, battery, industrial_drive. ML categorical.",
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


class Generator(ApplianceBase):
    """Generation unit. Same rated-vs-operating-point, per-phase asymmetry and
    connection semantics as :class:`Load` (see its docstring and
    ``references/asymmetric_modeling.md``); injected-power sign handled at assembly."""

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
    consumer_type: Optional[str] = Field(
        default=None,
        description="Open vocabulary (snake_case). Recommended: pv, wind, chp, battery, "
        "diesel_genset. ML categorical.",
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
        return self


class ShuntAppliance(ApplianceBase):
    component: Literal["shunt"] = "shunt"
    conductance_s: Vec = si_field(
        "Per-phase shunt conductance G.", short="S", long="siemens"
    )
    capacitance_f: Vec = si_field(
        "Per-phase shunt capacitance C (B(h)=2*pi*h*f0*C).", short="F", long="farad"
    )


Appliance = Annotated[
    Union[Source, Load, Generator, ShuntAppliance],
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
# 11. Grid container
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
        return self


__all__ = [
    "Phase",
    "WindingConnection",
    "LoadModel",
    "NodeZone",
    "GridEnvironment",
    "InterpolationMethod",
    "ExtrapolationMethod",
    "SourceConvention",
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
    "Load",
    "Generator",
    "ShuntAppliance",
    "Appliance",
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
