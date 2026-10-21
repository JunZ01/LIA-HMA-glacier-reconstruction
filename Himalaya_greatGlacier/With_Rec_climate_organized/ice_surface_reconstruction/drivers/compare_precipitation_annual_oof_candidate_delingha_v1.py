"""Six-year Delingha phase check for the annual-OOF early candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def main() -> int:
    project = root()
    results = project / "validation_data/results/paleo_surface_validation_v0"
    output = results / "climate_reconstruction_precipitation_annual_oof_candidate_delingha_v1"
    if output.exists():
        raise FileExistsError(output)
    candidate_path = results / "climate_reconstruction_precipitation_annual_oof_early_candidate_v2/annual_oof_candidate_native_2deg_1901_1920.npz"
    curves_path = results / "climate_reconstruction_improvement_loop_v2/round_102_tree_ring_hydroclimate_evidence_v4/allowed_period_standardized_curves.csv"
    with np.load(candidate_path) as handle:
        years = np.asarray(handle["years"], int)
        lat = np.asarray(handle["lat"], float)
        lon = np.asarray(handle["lon"], float)
        field = np.asarray(handle["increment_mm_day"], float)
    local = pd.read_csv(curves_path)
    local = local[(local.period == "early_1901_1920") & (local.evidence == "delingha_precip") & (local.region == "delingha_precip")]
    evidence = local[local.series == "evidence"].sort_values("year")
    production = local[local.series == "project_mean"].sort_values("year")
    selected_years = evidence.year.to_numpy(int)
    if len(selected_years) != 6 or not np.array_equal(selected_years, production.year.to_numpy(int)):
        raise RuntimeError("delingha-year-contract-failed")
    iy = int(np.argmin(np.abs(lat - 37.465)))
    ix = int(np.argmin(np.abs(lon - 97.725)))
    candidate = field[[int(np.flatnonzero(years == year)[0]) for year in selected_years], iy, ix]
    ev = evidence.raw_value.to_numpy(float)
    pr = production.raw_value.to_numpy(float)
    z = lambda values: (values - np.mean(values)) / np.std(values)
    evz, caz, prz = z(ev), z(candidate), z(pr)
    result = {
        "experiment_id": "climate_reconstruction_precipitation_annual_oof_candidate_delingha_v1",
        "status": "complete",
        "years": selected_years.tolist(),
        "n": 6,
        "candidate_grid_cell": {"latitude": float(lat[iy]), "longitude": float(lon[ix])},
        "candidate_pearson_r": float(pearsonr(evz, caz).statistic),
        "production_pearson_r": float(pearsonr(evz, prz).statistic),
        "candidate_spearman_r": float(spearmanr(evz, caz).statistic),
        "production_spearman_r": float(spearmanr(evz, prz).statistic),
        "locked_1921_1939_accessed": False,
        "interpretation_boundary": "Only six early years are available. This is likely-independent but low-sample phase evidence, not an amplitude or acceptance gate.",
    }
    output.mkdir(parents=True)
    pd.DataFrame({"year": selected_years, "evidence_z": evz, "candidate_z": caz, "production_z": prz}).to_csv(output / "delingha_six_year_curves.csv", index=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
