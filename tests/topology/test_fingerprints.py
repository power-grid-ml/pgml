"""Grid fingerprints: row-layout identity vs operating-point-independent network."""

from __future__ import annotations

from pgml.topology import layout_fingerprint, network_fingerprint
from tests.fixtures.tiny_grids import single_phase_chain


def test_layout_fingerprint_ignores_parameter_values():
    a = single_phase_chain()
    b = single_phase_chain()
    line = next(br for br in b.branches if br.id == 20)
    line.series_resistance_ohm_per_m = [[9.9e-3]]
    assert layout_fingerprint(a) == layout_fingerprint(b)


def test_layout_fingerprint_tracks_node_layout():
    a = single_phase_chain()
    b = single_phase_chain()
    b.nodes[0], b.nodes[1] = b.nodes[1], b.nodes[0]  # row order IS the contract
    assert layout_fingerprint(a) != layout_fingerprint(b)


def test_network_fingerprint_tracks_branch_parameters():
    a = single_phase_chain()
    b = single_phase_chain()
    assert network_fingerprint(a) == network_fingerprint(b)
    line = next(br for br in b.branches if br.id == 20)
    line.series_resistance_ohm_per_m = [[9.9e-3]]
    assert network_fingerprint(a) != network_fingerprint(b)


def test_network_fingerprint_ignores_injection_nameplates():
    """Load/generator P/Q is per-call operating-point data, not network identity."""
    a = single_phase_chain()
    b = single_phase_chain()
    for ap in b.appliances:
        if hasattr(ap, "p_nom_w"):
            ap.p_nom_w = float(ap.p_nom_w) * 2.0
    assert network_fingerprint(a) == network_fingerprint(b)


def test_network_fingerprint_tracks_injection_identity():
    from pgml.schemas.grid_schema import InjectionAppliance

    a = single_phase_chain()
    b = single_phase_chain()
    for ap in b.appliances:
        if isinstance(ap, InjectionAppliance):
            ap.in_service = False
    assert network_fingerprint(a) != network_fingerprint(b)
