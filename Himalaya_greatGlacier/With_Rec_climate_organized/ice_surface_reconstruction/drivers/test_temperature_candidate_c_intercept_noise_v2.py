from matplotlib.axes import Axes

import test_temperature_candidate_c_intercept_noise_v1 as impl


_original_boxplot = Axes.boxplot


def _compatible_boxplot(self, *args, **kwargs):
    if "tick_labels" in kwargs and "labels" not in kwargs:
        kwargs["labels"] = kwargs.pop("tick_labels")
    return _original_boxplot(self, *args, **kwargs)


Axes.boxplot = _compatible_boxplot
impl.OUTPUT = impl.LOOP / "temp_candidate_c_intercept_noise_v2"


if __name__ == "__main__":
    impl.main()
