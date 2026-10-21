from __future__ import annotations

import importlib


MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction."
    "drivers.build_lia_191_flowline_population93_prospective_route_v1"
)


def route_module():
    return importlib.import_module(MODULE)


def test_population_and_matrix_invariants() -> None:
    route = route_module()
    population = route.load_population()
    spinup = route.build_spinup_matrix(population)
    forward = route.build_forward_matrix(population)
    route.validate_matrices(population, spinup, forward)

    assert len(population) == 93
    assert len(spinup) == 930
    assert len(forward) == 93
    assert {row["curve_id"] for row in spinup} == {row["curve_id"] for row in forward}


def test_adaptive_blocks_are_conditional_resumes() -> None:
    route = route_module()
    population = route.load_population()[:1]
    rows = route.build_spinup_matrix(population)

    assert [row["block_end_year"] for row in rows] == list(range(300, 3001, 300))
    assert rows[0]["operation"] == "initialize"
    assert all(row["operation"] == "resume" for row in rows[1:])
    assert all(rows[index]["predecessor_task_id"] == rows[index - 1]["task_id"] for index in range(1, 10))
    assert all(row["launch_authorized"] is False for row in rows)


def test_length_is_output_coordinate_and_manual_tuning_is_forbidden() -> None:
    route = route_module()
    population = route.load_population()
    spinup = route.build_spinup_matrix(population)
    forward = route.build_forward_matrix(population)

    assert all(row["endpoint_length_target_used"] is False for row in spinup)
    assert all(row["equilibrium_length_role"] == "output-state-coordinate-not-input-target" for row in spinup)
    assert all(row["equilibrium_length_used_as_forcing_target"] is False for row in forward)
    assert all(row["extra_endpoint_ramp_or_fit"] == "none" for row in forward)
    assert all(row["glacier_specific_manual_tuning"] is False for row in spinup + forward)


def test_spinup_and_forward_task_spaces_are_disjoint() -> None:
    route = route_module()
    population = route.load_population()
    spinup = route.build_spinup_matrix(population)
    forward = route.build_forward_matrix(population)

    assert not ({row["task_id"] for row in spinup} & {row["task_id"] for row in forward})
    assert all(row["forward_variants_per_system"] == 1 for row in forward)
