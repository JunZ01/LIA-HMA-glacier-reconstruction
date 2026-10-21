"""V2 full-20-year regional evidence entrypoint; Delingha is separate."""

from __future__ import annotations

from pathlib import Path

import compare_precipitation_annual_oof_candidate_regional_hydroclimate_v1 as implementation


if __name__ == "__main__":
    original_read_csv = implementation.pd.read_csv

    def read_csv_without_short_delingha(path, *args, **kwargs):
        frame = original_read_csv(path, *args, **kwargs)
        if Path(path).name == "evidence_catalog_and_overlap.csv":
            frame = frame[frame["evidence"] != "delingha_precip"].copy()
        return frame

    implementation.EXPERIMENT_ID = "climate_reconstruction_precipitation_annual_oof_candidate_regional_hydroclimate_v2"
    implementation.pd.read_csv = read_csv_without_short_delingha
    raise SystemExit(implementation.main())
