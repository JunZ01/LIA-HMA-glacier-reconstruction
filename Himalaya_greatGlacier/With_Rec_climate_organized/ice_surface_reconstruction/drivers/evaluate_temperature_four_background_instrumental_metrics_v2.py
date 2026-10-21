from pathlib import Path

import evaluate_temperature_four_background_instrumental_metrics_v1 as core


core.OUT = (
    Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "temperature_four_background_instrumental_metrics_v2"
)


if __name__ == "__main__":
    core.main()
