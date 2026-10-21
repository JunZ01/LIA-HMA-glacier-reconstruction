"""Prepare the exact Round 36 annual moisture records for Round 37."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


EXPECTED_IDS = ("iso2k_1867", "iso2k_272", "iso2k_745")
SEASONS = {"Jun-Aug": (6, 8), "Jun-Sep": (6, 9)}


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def main() -> int:
    project = root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_37_dod2k_annual_moisture_nested_cross_product_v1"
    paths = {
        "round36_candidates": loop / "round_36_dod2k_hma_identity_chronology_admissibility_v1/annual_moisture_psm_candidates.csv",
        "round35_values": loop / "round_35_dod2k_hma_hydroclimate_coverage_audit_v1/dod2k_hma_hydroclimate_allowed_values.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    target = output / "round37_dod2k_annual_moisture_input.mat"
    if target.exists():
        raise FileExistsError(target)
    before = {name: state(path) for name, path in paths.items()}

    candidates = pd.read_csv(paths["round36_candidates"]).sort_values("datasetId").reset_index(drop=True)
    if tuple(candidates.datasetId) != EXPECTED_IDS or not candidates.annual_moisture_psm_admissible.all():
        raise RuntimeError("round36-candidate-contract-failed")
    values = pd.read_csv(paths["round35_values"])
    years = np.arange(1940, 2001, dtype=np.int32)
    observations = np.full((len(years), len(candidates)), np.nan)
    starts: list[int] = []
    ends: list[int] = []
    for column, item in enumerate(candidates.itertuples(index=False)):
        if item.interpretation_direction != "negative" or item.interpretation_seasonality not in SEASONS:
            raise RuntimeError(f"metadata-gate-failed:{item.datasetId}")
        starts.append(SEASONS[item.interpretation_seasonality][0])
        ends.append(SEASONS[item.interpretation_seasonality][1])
        series = values[(values.datasetId == item.datasetId) & (values.year_ce >= 1940) & (values.year_ce <= 2000)].copy()
        if ((series.year_ce >= 1921) & (series.year_ce <= 1939)).any():
            raise RuntimeError("locked-period-value-present")
        rounded = np.round(series.year_ce).astype(int)
        if not np.allclose(series.year_ce, rounded) or len(np.unique(rounded)) != 61 or not np.array_equal(np.sort(rounded), years):
            raise RuntimeError(f"modern-annual-axis-failed:{item.datasetId}")
        lookup = dict(zip(rounded, series.paleo_value.astype(float)))
        observations[:, column] = [lookup[int(year)] for year in years]
    if not np.isfinite(observations).all():
        raise RuntimeError("nonfinite-modern-observation")

    manifest = candidates[[
        "datasetId", "dataSetName", "geo_siteName", "latitude", "longitude",
        "interpretation_direction", "interpretation_seasonality", "originalDataURL",
    ]].copy()
    manifest["season_start_month"] = starts
    manifest["season_end_month"] = ends
    manifest.to_csv(output / "candidate_manifest.csv", index=False)
    savemat(target, {
        "years": years[:, None],
        "observations": observations,
        "locations": candidates[["latitude", "longitude"]].to_numpy(float),
        "identity_ids": np.asarray(EXPECTED_IDS, dtype=object)[:, None],
        "directions": np.asarray(["negative"] * 3, dtype=object)[:, None],
        "season_start_month": np.asarray(starts, dtype=np.int32)[:, None],
        "season_end_month": np.asarray(ends, dtype=np.int32)[:, None],
    }, do_compression=True)

    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    summary = {
        "status": "complete",
        "record_ids": list(EXPECTED_IDS),
        "year_start": 1940,
        "year_end": 2000,
        "locked_1921_1939_accessed": False,
        "source_states_unchanged": unchanged,
        "output_sha256": state(target)["sha256"],
    }
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "preparation_source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
