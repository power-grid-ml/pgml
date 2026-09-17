"""Structured conversion reports shared by every importer and exporter.

A conversion between pgml and a reference tool can lose or change a model in three
ways, and a caller comparing results across tools needs to know each of them:

``dropped``
    A source element has no counterpart in the target and is omitted.
``approximated``
    A source element is represented, but by something that is not the same model
    (a solved P/Q snapshot in place of a controlled inverter, a small resistance in
    place of an ideal switch, several generators merged into one terminal).
``model_difference``
    Every element converts, yet the two tools solve different equations for it by
    default (where the transformer magnetizing branch sits, how a line impedance
    grows with frequency, whether reactive limits are enforced). Each such entry
    names the pgml option or reference preset that reproduces the other tool's
    model, or states that none exists.

Every ``to_grid`` returns the report with ``return_report=True`` and every
``from_grid`` result carries it as ``.report``. The report is plain data: it
serialises to JSON, prints as text through :meth:`ConversionReport.summary`, and
:meth:`ConversionReport.log` sends it to the ``pgml`` logger.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

#: Version of the serialised report layout (``ConversionReport.to_dict``).
REPORT_FORMAT_VERSION = 1

#: Element ids listed per entry before the list is truncated (the count stays exact).
MAX_LISTED_IDS = 50

#: Result classes an entry can change. ``fundamental`` is the balanced power flow,
#: ``unbalanced`` the per-phase power flow, ``harmonic`` every order above one.
AFFECTS = ("fundamental", "unbalanced", "harmonic")


class ReportCategory(str, Enum):
    """The three ways a conversion can change the model."""

    DROPPED = "dropped"
    APPROXIMATED = "approximated"
    MODEL_DIFFERENCE = "model_difference"


def _plain(value: Any) -> Any:
    """Return ``value`` as JSON-serialisable builtins."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, complex):
        return {"re": value.real, "im": value.imag}
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _plain(item())
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _plain(tolist())
    return str(value)


@dataclass(frozen=True)
class ModelMatch:
    """How to make pgml and the other tool solve the same model for one entry.

    Attributes
    ----------
    preset:
        Name of the reference preset (:func:`pgml.defaults.use_preset`) that carries
        the settings below, when one does.
    settings:
        Modeling-default keys and the values that reproduce the other tool's model.
    arguments:
        Keyword arguments of a pgml call that do the same, keyed by
        ``"function.argument"`` (for example ``"solve_power_flow.enforce_q_limits"``).
    reference:
        What to set on the other tool's side instead, as free text, when the match is
        reached by configuring that tool rather than pgml.
    """

    preset: Optional[str] = None
    settings: Mapping[str, Any] = field(default_factory=dict)
    arguments: Mapping[str, Any] = field(default_factory=dict)
    reference: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "settings": _plain(self.settings),
            "arguments": _plain(self.arguments),
            "reference": self.reference,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelMatch":
        return cls(
            preset=data.get("preset"),
            settings=dict(data.get("settings") or {}),
            arguments=dict(data.get("arguments") or {}),
            reference=data.get("reference"),
        )

    def describe(self) -> str:
        parts = []
        if self.preset:
            parts.append(f'use_preset("{self.preset}")')
        if self.settings:
            body = ", ".join(f"{k}={v}" for k, v in self.settings.items())
            parts.append(f"defaults {body}")
        if self.arguments:
            parts.append(", ".join(f"{k}={v}" for k, v in self.arguments.items()))
        if self.reference:
            parts.append(f"on the other side: {self.reference}")
        return "; ".join(parts) if parts else "none"


@dataclass(frozen=True)
class ReportEntry:
    """One dropped element kind, approximation or model difference.

    Attributes
    ----------
    key:
        Stable identifier, ``dropped.<kind>``, ``approx.<topic>`` or
        ``model.<topic>``. The same topic uses the same key for every tool and in
        both directions, so a caller can filter on it.
    category:
        :class:`ReportCategory`.
    message:
        One sentence for a person.
    element_type:
        The element kind in the vocabulary of the side that was read (a pandapower
        table, a DSS class, a pgm component, or a pgml schema class on export).
    count:
        Number of affected elements. Exact even when ``ids`` is truncated.
    ids:
        Identifiers of the affected elements on the side that was read, at most
        :data:`MAX_LISTED_IDS` of them.
    affects:
        Which result classes the entry can change, a subset of :data:`AFFECTS`.
    source_model, pgml_model:
        For a model difference, the two models in a few words.
    match:
        :class:`ModelMatch` that reproduces the other tool's model, or ``None`` when
        pgml cannot match it.
    matched:
        ``True`` when the modeling defaults active during the conversion already
        equal ``match.settings``, so the difference is closed for this conversion.
    values:
        Numbers that size the entry (parameter values read but not represented,
        ratios applied, magnitudes).
    """

    key: str
    category: ReportCategory
    message: str
    element_type: Optional[str] = None
    count: int = 0
    ids: tuple = ()
    affects: tuple = ()
    source_model: Optional[str] = None
    pgml_model: Optional[str] = None
    match: Optional[ModelMatch] = None
    matched: bool = False
    values: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "category": self.category.value,
            "message": self.message,
            "element_type": self.element_type,
            "count": int(self.count),
            "ids": _plain(self.ids),
            "affects": list(self.affects),
            "source_model": self.source_model,
            "pgml_model": self.pgml_model,
            "match": None if self.match is None else self.match.to_dict(),
            "matched": bool(self.matched),
            "values": _plain(self.values),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReportEntry":
        match = data.get("match")
        return cls(
            key=data["key"],
            category=ReportCategory(data["category"]),
            message=data["message"],
            element_type=data.get("element_type"),
            count=int(data.get("count", 0)),
            ids=tuple(data.get("ids") or ()),
            affects=tuple(data.get("affects") or ()),
            source_model=data.get("source_model"),
            pgml_model=data.get("pgml_model"),
            match=None if match is None else ModelMatch.from_dict(match),
            matched=bool(data.get("matched", False)),
            values=dict(data.get("values") or {}),
        )

    def describe(self) -> str:
        """The entry as one line of text."""
        head = f"[{self.key}]"
        if self.count:
            head += f" {self.count} x {self.element_type or 'element'}"
        line = f"{head}: {self.message}"
        if self.category is ReportCategory.MODEL_DIFFERENCE:
            how = "none" if self.match is None else self.match.describe()
            state = " (active now)" if self.matched else ""
            line += f" Match: {how}{state}."
        return line


def _settings_active(settings: Mapping[str, Any]) -> bool:
    """Whether the active modeling defaults already carry every value in ``settings``."""
    if not settings:
        return False
    from pgml import defaults

    for name, wanted in settings.items():
        try:
            if defaults.get(name) != wanted:
                return False
        except Exception:  # noqa: BLE001 -- an unknown key is simply not active
            return False
    return True


@dataclass
class ConversionReport:
    """Everything a conversion dropped, approximated or models differently.

    Attributes
    ----------
    tool:
        ``"pandapower"``, ``"opendss"`` or ``"power-grid-model"``.
    direction:
        ``"import"`` (tool to pgml ``Grid``) or ``"export"`` (``Grid`` to tool).
    options:
        The conversion arguments that shape the report (phase mode, generator mode,
        export mode).
    comparable:
        The result classes the tool computes, a subset of :data:`AFFECTS`. An entry
        that touches none of them (a harmonic model next to a fundamental-only
        tool) is recorded but does not raise the log level.
    entries:
        The :class:`ReportEntry` records, in the order they were found.
    """

    tool: str
    direction: str
    options: dict[str, Any] = field(default_factory=dict)
    comparable: tuple = AFFECTS
    entries: list[ReportEntry] = field(default_factory=list)
    _announced: set = field(default_factory=set, init=False, repr=False, compare=False)

    # -- building ---------------------------------------------------------- #
    def add(
        self,
        key: str,
        category: ReportCategory,
        message: str,
        *,
        element_type: Optional[str] = None,
        ids: Optional[Iterable[Any]] = None,
        count: Optional[int] = None,
        affects: Sequence[str] = AFFECTS,
        source_model: Optional[str] = None,
        pgml_model: Optional[str] = None,
        match: Optional[ModelMatch] = None,
        values: Optional[Mapping[str, Any]] = None,
        matched: Optional[bool] = None,
        announced: bool = False,
    ) -> ReportEntry:
        """Append one entry and return it.

        ``matched`` defaults to whether the active modeling defaults already equal
        ``match.settings``; pass it for a match that rests on call arguments.
        ``announced=True`` marks an entry whose detail the converter has already
        logged itself, so :meth:`log` does not repeat it.
        """
        unknown = set(affects) - set(AFFECTS)
        if unknown:
            raise ValueError(f"unknown result class(es) {sorted(unknown)}")
        listed = [] if ids is None else list(ids)
        entry = ReportEntry(
            key=key,
            category=ReportCategory(category),
            message=message,
            element_type=element_type,
            count=len(listed) if count is None else int(count),
            ids=tuple(_plain(listed[:MAX_LISTED_IDS])),
            affects=tuple(a for a in AFFECTS if a in affects),
            source_model=source_model,
            pgml_model=pgml_model,
            match=match,
            matched=(
                match is not None and _settings_active(match.settings)
                if matched is None
                else bool(matched)
            ),
            values=dict(_plain(values or {})),
        )
        self.entries.append(entry)
        if announced:
            self._announced.add(len(self.entries) - 1)
        return entry

    def dropped(self, kind: str, message: str, **kwargs: Any) -> ReportEntry:
        """Record source elements of ``kind`` that are omitted."""
        kwargs.setdefault("element_type", kind)
        return self.add(f"dropped.{kind}", ReportCategory.DROPPED, message, **kwargs)

    def approximated(self, topic: str, message: str, **kwargs: Any) -> ReportEntry:
        """Record elements represented by a different model."""
        return self.add(
            f"approx.{topic}", ReportCategory.APPROXIMATED, message, **kwargs
        )

    def model_difference(self, topic: str, message: str, **kwargs: Any) -> ReportEntry:
        """Record a default-model difference between the tool and pgml."""
        return self.add(
            f"model.{topic}", ReportCategory.MODEL_DIFFERENCE, message, **kwargs
        )

    def extend(self, entries: Iterable[ReportEntry]) -> None:
        self.entries.extend(entries)

    # -- reading ----------------------------------------------------------- #
    def __iter__(self) -> Iterator[ReportEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def of(self, category: ReportCategory) -> list[ReportEntry]:
        category = ReportCategory(category)
        return [e for e in self.entries if e.category is category]

    @property
    def dropped_entries(self) -> list[ReportEntry]:
        return self.of(ReportCategory.DROPPED)

    @property
    def approximated_entries(self) -> list[ReportEntry]:
        return self.of(ReportCategory.APPROXIMATED)

    @property
    def model_differences(self) -> list[ReportEntry]:
        return self.of(ReportCategory.MODEL_DIFFERENCE)

    def keys(self) -> list[str]:
        """Entry keys in order, without duplicates."""
        return list(dict.fromkeys(e.key for e in self.entries))

    def get(self, key: str) -> list[ReportEntry]:
        """Every entry carrying ``key`` (several when the ids were reported apart)."""
        return [e for e in self.entries if e.key == key]

    def __contains__(self, key: object) -> bool:
        return any(e.key == key for e in self.entries)

    def open_entries(self, affects: Optional[str] = None) -> list[ReportEntry]:
        """Entries that still separate the two models.

        A model difference whose match is already active is closed; everything else
        is open. ``affects`` keeps only entries that can change that result class.
        """
        if affects is not None and affects not in AFFECTS:
            raise ValueError(f"affects must be one of {AFFECTS}, got {affects!r}")
        return [
            e
            for e in self.entries
            if not e.matched and (affects is None or affects in e.affects)
        ]

    def is_exact(self, affects: Optional[str] = None) -> bool:
        """Whether both sides describe the same model for the named result class."""
        return not self.open_entries(affects)

    def match_settings(self) -> dict[str, Any]:
        """Union of the modeling-default settings that close the model differences."""
        merged: dict[str, Any] = {}
        for entry in self.model_differences:
            if entry.match is not None:
                merged.update(entry.match.settings)
        return merged

    # -- output ------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": REPORT_FORMAT_VERSION,
            "tool": self.tool,
            "direction": self.direction,
            "options": _plain(self.options),
            "comparable": list(self.comparable),
            "counts": {c.value: len(self.of(c)) for c in ReportCategory},
            "entries": [e.to_dict() for e in self.entries],
        }

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConversionReport":
        return cls(
            tool=data["tool"],
            direction=data["direction"],
            options=dict(data.get("options") or {}),
            comparable=tuple(data.get("comparable") or AFFECTS),
            entries=[ReportEntry.from_dict(e) for e in data.get("entries", [])],
        )

    def headline(self) -> str:
        arrow = (
            f"{self.tool} -> Grid"
            if self.direction == "import"
            else f"Grid -> {self.tool}"
        )
        open_models = [e for e in self.model_differences if not e.matched]
        return (
            f"{arrow}: {len(self.dropped_entries)} dropped, "
            f"{len(self.approximated_entries)} approximated, "
            f"{len(open_models)} open model difference(s)"
            f" ({len(self.model_differences) - len(open_models)} matched by the "
            "active defaults)"
        )

    def summary(self) -> str:
        """The whole report as text, one entry per line under its category."""
        lines = [self.headline()]
        titles = {
            ReportCategory.DROPPED: "Dropped",
            ReportCategory.APPROXIMATED: "Approximated",
            ReportCategory.MODEL_DIFFERENCE: "Model differences",
        }
        for category, title in titles.items():
            entries = self.of(category)
            if not entries:
                continue
            lines.append(f"{title}:")
            lines.extend(f"  {e.describe()}" for e in entries)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def log(self, logger: Optional[logging.Logger] = None) -> None:
        """Send the report to ``logger`` (default: the ``pgml`` logger).

        Dropped and approximated entries are logged at WARNING, model differences at
        INFO. One closing line at WARNING names the counts whenever an open entry
        touches a result class the tool computes, so a session with default logging
        still learns that the converted model is not the source model; otherwise the
        closing line is INFO.
        """
        logger = logger or logging.getLogger("pgml")
        for position, entry in enumerate(self.entries):
            if position in self._announced:
                continue
            level = (
                logging.INFO
                if entry.category is ReportCategory.MODEL_DIFFERENCE
                else logging.WARNING
            )
            logger.log(level, "%s", entry.describe())
        if not self.entries:
            return
        relevant = [
            e for e in self.open_entries() if set(e.affects) & set(self.comparable)
        ]
        level = logging.WARNING if relevant else logging.INFO
        logger.log(
            level,
            "%s. Pass return_report=True (import) or read .report (export) for the "
            "details, or set the pgml logger to INFO.",
            self.headline(),
        )


__all__ = [
    "AFFECTS",
    "MAX_LISTED_IDS",
    "REPORT_FORMAT_VERSION",
    "ConversionReport",
    "ModelMatch",
    "ReportCategory",
    "ReportEntry",
]
