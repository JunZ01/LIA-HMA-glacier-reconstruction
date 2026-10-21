"""Independent v2: gzip-aware execution of the frozen v1 classifier."""

from __future__ import annotations

import gzip

from . import classify_current_route_v8_zero_mb_structure_v1 as v1


def _gzip_pickle_open(path, mode):
    return gzip.open(path, mode)


def main() -> None:
    v1.OUT = v1.VALIDATION / "lia_191_current_route_v8_zero_mb_structure_classification_v2"
    v1.Path.open = v1.Path.open
    original_open = v1.Path.open

    def open_dispatch(path, mode="r", *args, **kwargs):
        if str(path).endswith("model_flowlines.pkl") and "b" in mode:
            return _gzip_pickle_open(path, mode)
        return original_open(path, mode, *args, **kwargs)

    v1.Path.open = open_dispatch
    try:
        v1.main()
    finally:
        v1.Path.open = original_open


if __name__ == "__main__":
    main()
