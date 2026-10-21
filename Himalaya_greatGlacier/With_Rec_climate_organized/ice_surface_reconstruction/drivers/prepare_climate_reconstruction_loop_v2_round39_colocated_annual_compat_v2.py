"""Round 39 preparation compatible with contiguous endpoint-missing records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


EXPECTED_IDS = ("iso2k_1086", "iso2k_1344", "iso2k_503", "iso2k_556")
SEASONS = {"Summer": (6, 8, "calendar_year"), "Jun-Sep": (6, 9, "calendar_year"), "Oct-Sep": (10, 9, "end_year")}


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
    output = loop / "round_39_dod2k_colocated_annual_nested_cross_product_v1"
    paths = {
        "round38_candidates": loop / "round_38_dod2k_colocated_measurement_identity_v2/colocated_distinct_annual_candidates.csv",
        "round35_values": loop / "round_35_dod2k_hma_hydroclimate_coverage_audit_v1/dod2k_hma_hydroclimate_allowed_values.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "failed_v1_driver": Path(__file__).with_name("prepare_climate_reconstruction_loop_v2_round39_colocated_annual_v1.py"),
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    target = output / "round39_colocated_annual_input.mat"
    if target.exists():
        raise FileExistsError(target)
    before = {name: state(path) for name, path in paths.items()}
    candidates = pd.read_csv(paths["round38_candidates"]).sort_values("datasetId").reset_index(drop=True)
    values = pd.read_csv(paths["round35_values"])
    if tuple(candidates.datasetId) != EXPECTED_IDS:
        raise RuntimeError("round38-candidate-contract-failed")
    years = np.arange(1940, 2001, dtype=np.int32)
    observations = np.full((61, 4), np.nan)
    starts: list[int] = []
    ends: list[int] = []
    anchors: list[str] = []
    counts: dict[str, int] = {}
    for column, item in enumerate(candidates.itertuples(index=False)):
        start, end, anchor = SEASONS[item.interpretation_seasonality]
        starts.append(start); ends.append(end); anchors.append(anchor)
        series = values[(values.datasetId == item.datasetId) & (values.year_ce >= 1940) & (values.year_ce <= 2000)].copy()
        rounded = np.round(series.year_ce).astype(int)
        if ((series.year_ce >= 1921) & (series.year_ce <= 1939)).any() or not np.allclose(series.year_ce, rounded):
            raise RuntimeError("locked-or-nonannual-year")
        unique = np.sort(np.unique(rounded))
        if len(unique) < 50 or not np.all(np.diff(unique) == 1):
            raise RuntimeError(f"modern-contiguous-axis-failed:{item.datasetId}")
        lookup = dict(zip(rounded, series.paleo_value.astype(float)))
        observations[:, column] = [lookup.get(int(year), np.nan) for year in years]
        counts[item.datasetId] = int(np.isfinite(observations[:, column]).sum())
    manifest = candidates[["datasetId", "dataSetName", "geo_siteName", "latitude", "longitude", "interpretation_direction", "interpretation_seasonality", "originalDataURL"]].copy()
    manifest["season_start_month"] = starts; manifest["season_end_month"] = ends; manifest["season_year_anchor"] = anchors
    manifest.to_csv(output / "candidate_manifest.csv", index=False)
    savemat(target, {
        "years": years[:, None], "observations": observations,
        "locations": candidates[["latitude", "longitude"]].to_numpy(float),
        "identity_ids": np.asarray(EXPECTED_IDS, dtype=object)[:, None],
        "directions": np.asarray(candidates.interpretation_direction, dtype=object)[:, None],
        "season_start_month": np.asarray(starts, dtype=np.int32)[:, None],
        "season_end_month": np.asarray(ends, dtype=np.int32)[:, None],
        "season_end_year_anchor": np.asarray([anchor == "end_year" for anchor in anchors], dtype=np.uint8)[:, None],
    }, do_compression=True)
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    summary = {"status": "complete", "compatibility_correction": "allow contiguous modern records with missing endpoints", "modern_finite_counts": counts, "locked_1921_1939_accessed": False, "source_states_unchanged": unchanged, "output_sha256": state(target)["sha256"]}
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "preparation_source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
