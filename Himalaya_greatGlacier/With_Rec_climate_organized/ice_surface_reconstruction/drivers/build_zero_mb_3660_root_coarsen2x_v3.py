"""Independent v3 of the 3660 2x-root preflight.

V2 stopped before scientific adjudication because Centerline cached the old
``flows_to_indice`` under ``_lazy_flows_to_indice``.  This wrapper invalidates
that cache whenever ``set_flows_to`` is called and otherwise runs the unchanged
v2 construction in a new output directory.
"""

from pathlib import Path

from oggm.core.centerlines import Centerline

from . import build_zero_mb_3660_root_coarsen2x_v2 as implementation


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_zero_mb_3660_root_coarsen2x_preflight_v3"
)


def main() -> None:
    original_set_flows_to = Centerline.set_flows_to

    def cache_safe_set_flows_to(self, *args, **kwargs):
        self.__dict__.pop("_lazy_flows_to_indice", None)
        return original_set_flows_to(self, *args, **kwargs)

    Centerline.set_flows_to = cache_safe_set_flows_to
    implementation.OUTPUT = OUTPUT
    implementation.main()


if __name__ == "__main__":
    main()
