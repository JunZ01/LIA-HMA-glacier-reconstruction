"""Extend batch-1 formal scoring to the approved 2815/2902 spatial rerun."""

from __future__ import annotations

from . import summarize_plot_lia_191_formal_delta_bisection_batch1_v1 as _base


def main() -> None:
    _base.GROUPS = ("regular", "spatial_2928", "spatial_2815_2902")
    _base.main()


if __name__ == "__main__":
    main()
