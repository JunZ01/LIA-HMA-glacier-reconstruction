from pathlib import Path

from matplotlib.axes import Axes

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import analyze_lia_year900_parameter_diagnostics_v1 as v1


_original_boxplot = Axes.boxplot


def _compatible_boxplot(self, *args, **kwargs):
    if "tick_labels" in kwargs:
        kwargs["labels"] = kwargs.pop("tick_labels")
    return _original_boxplot(self, *args, **kwargs)


Axes.boxplot = _compatible_boxplot
v1.OUT = v1.ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_year900_parameter_diagnostics_v2"


if __name__ == "__main__":
    v1.main()
