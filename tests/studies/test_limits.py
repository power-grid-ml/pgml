"""Branch-rating resolver and the hosting-limit container."""

from __future__ import annotations

import json

import pytest
import torch

from pgml.errors import InputError
from pgml.schemas.grid_schema import Transformer
from pgml.studies import BranchRating, HostingLimits, branch_ratings

from .grids import lv_feeder_with_switch, mv_lv_transformer, two_bus


def test_line_ampacity_comes_from_the_catalog_type():
    ratings = branch_ratings(lv_feeder_with_switch(i_max_a=180.0))
    assert ratings[21] == BranchRating(i_max_a=180.0, s_rated_va=None)
    assert ratings[22].i_max_a == 180.0
    # a switch carries no rating, and is still listed
    assert ratings[20] == (None, None)


def test_line_without_type_or_ampacity_reports_none():
    grid = two_bus(i_max_a=None)
    assert branch_ratings(grid)[10] == (None, None)
    explicit = mv_lv_transformer()
    assert branch_ratings(explicit)[31] == (None, None)  # explicit line, no type_ref


def test_transformer_rating_from_component_then_type():
    grid = mv_lv_transformer(s_rated_va=160.0e3)
    assert branch_ratings(grid)[30] == (None, 160.0e3)

    t = grid.branches[0]
    typed = Transformer(
        id=t.id,
        from_node=t.from_node,
        to_node=t.to_node,
        from_phases=t.from_phases,
        to_phases=t.to_phases,
        type_ref="t100",
    )
    grid_typed = grid.model_copy(update={"branches": [typed, grid.branches[1]]})
    assert branch_ratings(grid_typed)[30].s_rated_va == 160.0e3


def test_tensor_valued_rating_reads_as_float_and_input_is_untouched():
    grid = mv_lv_transformer(s_rated_va=100.0e3)
    rated = torch.tensor(250.0e3, dtype=torch.float64, requires_grad=True)
    t = grid.branches[0].model_copy(update={"s_rated_va": rated})
    grid_t = grid.model_copy(update={"branches": [t, grid.branches[1]]})
    out = branch_ratings(grid_t)[30]
    assert isinstance(out.s_rated_va, float) and out.s_rated_va == 250.0e3
    assert grid_t.branches[0].s_rated_va is rated


def test_out_of_service_branches_are_listed():
    grid = lv_feeder_with_switch()
    off = grid.branches[2].model_copy(update={"in_service": False})
    grid_off = grid.model_copy(update={"branches": [*grid.branches[:2], off]})
    assert set(branch_ratings(grid_off)) == {20, 21, 22}


def test_hosting_limits_round_trip_and_validation():
    lim = HostingLimits(v_max_pu=1.03, vuf_max=0.02, line_loading=None)
    data = json.loads(json.dumps(lim.to_dict()))
    assert HostingLimits.from_dict(data) == lim
    assert HostingLimits.from_dict({}) == HostingLimits()
    with pytest.raises(InputError):
        HostingLimits(v_min_pu=1.0, v_max_pu=0.95)
    with pytest.raises(InputError):
        HostingLimits(line_loading=0.0)
    with pytest.raises(InputError):
        HostingLimits(thd_max=float("nan"))
