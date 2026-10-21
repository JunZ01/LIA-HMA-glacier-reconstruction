"""Run the seasonal PSM audit with the corrected prior-December index."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def main() -> int:
    base_path = Path(__file__).with_name("audit_positive_b2_event_seasonal_psm_v1.py")
    spec = importlib.util.spec_from_file_location("positive_b2_event_seasonal_psm_v1", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("base-driver-import-failed")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)

    def corrected_time_slice(start_year: int, first_year: int, last_year: int) -> slice:
        start = (first_year - 1 - start_year) * 12 + 11
        stop = (last_year - start_year + 1) * 12
        return slice(start, stop)

    base.time_slice = corrected_time_slice
    status = int(base.main())
    root = base.project_root()
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_event_seasonal_psm_v1"
    manifest_path = output / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["launcher_driver"] = Path(__file__).name
    manifest["launcher_driver_sha256"] = base.digest(Path(__file__))
    manifest["index_correction"] = "Prior December begins at (first_year - 1 - source_start_year) * 12 + 11."
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
