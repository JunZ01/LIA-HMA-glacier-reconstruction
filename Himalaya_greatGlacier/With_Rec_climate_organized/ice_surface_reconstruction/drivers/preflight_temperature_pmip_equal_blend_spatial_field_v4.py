"""Fresh versioned retry of the frozen spatial preflight after OPeNDAP recovery."""

from pathlib import Path

import preflight_temperature_pmip_equal_blend_spatial_field_v3 as runner


def main() -> int:
    runner.OUTPUT_NAME = "climate_reconstruction_temperature_pmip_equal_blend_spatial_preflight_v4"
    runner.__file__ = str(Path(__file__).resolve())
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
