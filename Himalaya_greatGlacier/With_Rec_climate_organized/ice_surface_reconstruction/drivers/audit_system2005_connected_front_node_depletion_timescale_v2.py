"""OGGM 1.6 date-helper compatibility wrapper for the v1 read-only audit."""

from oggm import utils

from . import audit_system2005_connected_front_node_depletion_timescale_v1 as v1


if __name__ == "__main__":
    if not hasattr(utils, "date_to_float"):
        utils.date_to_float = utils.date_to_floatyear
    v1.OUTPUT = (
        v1.PROJECT
        / "validation_data/results/paleo_surface_validation_v0/"
        "lia_191_system2005_connected_front_node_depletion_timescale_v2"
    )
    v1.main()
