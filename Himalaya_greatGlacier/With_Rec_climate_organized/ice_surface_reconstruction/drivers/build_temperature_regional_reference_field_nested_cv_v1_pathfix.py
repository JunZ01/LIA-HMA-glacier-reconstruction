"""Path-only launcher for the frozen regional reference-field v1 experiment."""

import build_temperature_regional_reference_field_nested_cv_v1 as experiment


experiment.ASSIGNMENT = (
    experiment.RESULTS
    / "climate_reconstruction_hma_gtng_subregion_inventory_v1/hma_grid_gtng15_assignment.csv"
)


if __name__ == "__main__":
    experiment.main()
