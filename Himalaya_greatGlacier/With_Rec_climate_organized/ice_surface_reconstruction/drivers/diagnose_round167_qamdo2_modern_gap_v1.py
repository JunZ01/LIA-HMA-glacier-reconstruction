from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_driver(path: Path):
    spec = importlib.util.spec_from_file_location("round167_driver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    driver_path = Path(__file__).with_name(
        "run_round167_qamdo2_std_chronology_modern_oof_v2.py"
    )
    driver = load_driver(driver_path)
    root = driver.project_root()
    output = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/"
        "round_167_chin074_qamdo2_identity_recovery_v1"
    )
    sys.path.insert(0, str(output / "vendor"))
    import dplpy

    archive = output / "raw_unopened/mainz_qamdo_archive/Qamdo"
    raw, metadata = driver.parse_rwl_without_locked_values(archive / "Qamdo.txt")
    rwi = dplpy.detrend(raw, fit="ModNegEx", method="residual", plot=False)
    chronology = dplpy.chron(rwi, biweight=True, prewhiten=False, plot=False)
    std = chronology["Mean RWI"].astype(float)
    depth = chronology["Sample depth"].astype(float)
    published = driver.parse_two_column_without_locked_values(
        archive / "Qamdo_Recon.txt"
    )

    years = driver.MATRIX_YEARS
    raw_depth = raw.notna().sum(axis=1).reindex(years).fillna(0).astype(int)
    report = {
        "locked_numeric_values_decoded": False,
        "raw_metadata": metadata,
        "raw_missing_modern_years": [
            int(year) for year in years if int(raw_depth.loc[year]) == 0
        ],
        "raw_depth_by_modern_year": {
            str(int(year)): int(raw_depth.loc[year]) for year in years
        },
        "std_missing_modern_years": [
            int(year) for year in years if not np.isfinite(std.get(year, np.nan))
        ],
        "published_missing_modern_years": [
            int(year)
            for year in years
            if not np.isfinite(published.get(year, np.nan))
        ],
        "chronology_allowed_min": int(std.index.min()),
        "chronology_allowed_max": int(std.index.max()),
    }
    destination = output / "modern_candidate_gap_diagnostic_v1.json"
    destination.write_text(json.dumps(report, indent=2), encoding="ascii")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
