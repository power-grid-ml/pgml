"""The modeling-defaults config (`pgml.config`) — loading, precedence, and values.

The config is the single source of truth for default VALUES and default MODEL choices;
these tests pin the documented constants, the precedence contract (explicit > config >
converter), and the dispatcher that turns the config into per-line models.
"""

from __future__ import annotations

import math

import pytest

from pgml import config


# --- loading + documentation -----------------------------------------------
def test_every_leaf_has_value_units_description():
    """Each config leaf is self-documenting: value + units + description."""

    def walk(node, path=""):
        if isinstance(node, dict) and "value" in node:
            for field in ("value", "units", "description"):
                assert field in node, f"{path} missing {field!r}"
            assert str(node["description"]).strip(), f"{path} has empty description"
            return
        assert isinstance(node, dict), f"{path} is neither a leaf nor a mapping"
        for k, v in node.items():
            walk(v, f"{path}.{k}" if path else k)

    walk(config.defaults())


def test_documented_constants_match_legacy_values():
    """The config reproduces the previously hard-coded constants exactly."""
    assert config.get("line.conductor.gmr_over_radius") == 0.7788  # e^{-1/4}
    assert config.get("line.conductor.radius_m") == 0.0102
    assert config.get("line.conductor.height_overhead_m") == 10.0
    assert config.get("line.conductor.height_cable_m") == 1.0
    assert config.get("line.earth_return.resistivity_ohm_m") == 100.0
    assert (
        config.get("line.earth_return.resistance_coeff_ohm_per_m_per_hz")
        == math.pi**2 * 1e-7
    )
    assert config.get("line.harmonic_model.three_phase") == "sequence_aware"
    assert config.get("line.harmonic_model.single_phase") == "positive_sequence"
    assert config.get("line.harmonic_model.skin_effect") is True


def test_module_constants_are_sourced_from_config():
    """The geometry modules read their constants from the config (no drift)."""
    from pgml.geometry import sequence as sq
    from pgml.geometry import synthesis as syn

    assert sq._DEFAULT_GMR_OVER_RADIUS == config.get("line.conductor.gmr_over_radius")
    assert sq.CARSON_EARTH_R_PER_HZ == config.get(
        "line.earth_return.resistance_coeff_ohm_per_m_per_hz"
    )
    assert syn._DEFAULT_RADIUS == config.get("line.conductor.radius_m")
    assert syn._DEFAULT_EARTH_RHO == config.get("line.earth_return.resistivity_ohm_m")


def test_describe_and_units():
    assert "GMR" in config.describe("line.conductor.gmr_over_radius")
    assert config.units("line.earth_return.resistance_coeff_ohm_per_m_per_hz")


def test_missing_key_raises_or_defaults():
    with pytest.raises(KeyError):
        config.get("line.does_not_exist")
    assert config.get("line.does_not_exist", default=7) == 7
    with pytest.raises(KeyError):
        config.get("line.conductor")  # a branch, not a leaf with a value


# --- precedence: explicit > config > converter -----------------------------
def test_resolve_precedence():
    key = "line.conductor.gmr_over_radius"
    # 1. explicit wins.
    assert config.resolve(key, explicit=0.5) == 0.5
    # 2. config when no explicit.
    assert config.resolve(key) == 0.7788
    # 3. converter only when the key is absent from config.
    assert config.resolve("line.absent", converted=42) == 42
    # explicit beats converter too.
    assert config.resolve("line.absent", explicit=1, converted=42) == 1


def test_reload_with_override(tmp_path, monkeypatch):
    """A PGML_CONFIG override is honored and can be reverted."""
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "line:\n"
        "  conductor:\n"
        "    gmr_over_radius:\n"
        "      value: 0.5\n"
        "      units: ratio\n"
        "      description: custom override\n"
    )
    monkeypatch.setenv("PGML_CONFIG", str(custom))
    try:
        config.reload(str(custom))
        assert config.get("line.conductor.gmr_over_radius") == 0.5
    finally:
        monkeypatch.delenv("PGML_CONFIG", raising=False)
        config.reload()  # back to the packaged defaults
    assert config.get("line.conductor.gmr_over_radius") == 0.7788


# --- the config-default harmonic-model dispatcher --------------------------
def _three_phase_grid():
    from pgml.schemas.grid_schema import Grid, Line, Node, Phase, Source

    f0 = 50.0
    z1, z0 = complex(0.3e-3, 0.3e-3), complex(0.6e-3, 1.2e-3)
    zs, zm = (z0 + 2 * z1) / 3, (z0 - z1) / 3
    w = 2 * math.pi * f0
    ph = (Phase.A, Phase.B, Phase.C)
    return Grid(
        base_frequency_hz=f0,
        nodes=[
            Node(id=1, u_rated_v=400.0, phases=ph),
            Node(id=2, u_rated_v=400.0, phases=ph),
        ],
        branches=[
            Line(
                id=10,
                from_node=1,
                to_node=2,
                from_phases=ph,
                to_phases=ph,
                length_m=100.0,
                series_resistance_ohm_per_m=[
                    [zs.real if i == j else zm.real for j in range(3)] for i in range(3)
                ],
                series_inductance_h_per_m=[
                    [(zs.imag if i == j else zm.imag) / w for j in range(3)]
                    for i in range(3)
                ],
                shunt_capacitance_f_per_m=[[0.0] * 3 for _ in range(3)],
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ph,
                u_ref_v=(230.0, 230.0, 230.0),
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=[
                    [1e-3 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                inductance_h=[
                    [1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
            )
        ],
    )


def test_default_model_three_phase_is_sequence_aware():
    """A 3-phase R/X line gets the config default (sequence_aware) for 4-wire studies."""
    from pgml.geometry import apply_default_harmonic_model

    grid = _three_phase_grid()
    apply_default_harmonic_model(grid)
    assert grid.branches[0].tags["harmonic_line_model"] == "sequence_aware"


def test_default_model_single_phase_is_positive_sequence():
    """A 1-phase R/X line gets positive_sequence (skin law), not the 3-phase model."""
    from pgml.geometry import apply_default_harmonic_model
    from pgml.schemas.grid_schema import Grid, Line, Node, Phase, Source

    ph = (Phase.A,)
    grid = Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=230.0, phases=ph),
            Node(id=2, u_rated_v=230.0, phases=ph),
        ],
        branches=[
            Line(
                id=10,
                from_node=1,
                to_node=2,
                from_phases=ph,
                to_phases=ph,
                length_m=100.0,
                series_resistance_ohm_per_m=[[0.3e-3]],
                series_inductance_h_per_m=[[1e-6]],
                shunt_capacitance_f_per_m=[[0.0]],
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ph,
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[1e-3]],
                inductance_h=[[1e-6]],
            )
        ],
    )
    apply_default_harmonic_model(grid)
    ln = grid.branches[0]
    assert "harmonic_line_model" not in (ln.tags or {})
    assert ln.resistance_frequency.multiplier.law == "carson_skin_multiplier"


def test_default_model_respects_explicit_precedence():
    """An explicit model (precedence 1) is never overwritten by the default."""
    from pgml.geometry import apply_default_harmonic_model

    grid = _three_phase_grid()
    grid.branches[0].tags = {"harmonic_line_model": "positive_sequence"}
    apply_default_harmonic_model(grid)
    assert grid.branches[0].tags["harmonic_line_model"] == "positive_sequence"
