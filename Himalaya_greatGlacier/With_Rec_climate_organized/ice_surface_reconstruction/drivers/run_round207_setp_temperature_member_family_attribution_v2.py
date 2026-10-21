from __future__ import annotations

import importlib.util
from pathlib import Path

# Windows can resolve GDAL incorrectly if HDF5 is loaded first in this
# environment. Preload rasterio before importing the frozen v1 driver.
import rasterio  # noqa: F401


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVERS = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized"
    / "ice_surface_reconstruction/drivers"
)
V1_DRIVER = DRIVERS / "run_round207_setp_temperature_member_family_attribution_v1.py"
V2_OUT = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "round_207_setp_temperature_member_family_attribution_v2"
)


def main() -> None:
    spec = importlib.util.spec_from_file_location("round207_v1_for_v2", V1_DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load Round 207 v1")
    round207 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(round207)
    round207.OUT = V2_OUT
    round207.CONTRACT = V2_OUT / "ROUND_CONTRACT.md"
    round207.DRIVER = Path(__file__).resolve()
    round207.main()


if __name__ == "__main__":
    main()
