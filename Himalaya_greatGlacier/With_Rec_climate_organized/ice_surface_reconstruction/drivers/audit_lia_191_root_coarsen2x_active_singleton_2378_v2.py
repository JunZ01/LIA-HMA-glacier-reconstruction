"""Independent v2 after v1 stopped on a Python boolean spelling defect."""

from pathlib import Path

from . import audit_lia_191_root_coarsen2x_active_singleton_2378_v1 as implementation


PROJECT = Path(__file__).resolve().parents[4]
implementation.CONFIG = (
    PROJECT
    / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction"
    / "config/lia_191_root_coarsen2x_active_singleton_2378_v2.json"
)
# V1 referenced the JSON spelling ``false`` while creating its final payload.
# Supplying that one missing global is the only executable correction in v2.
implementation.false = False


if __name__ == "__main__":
    implementation.main()
