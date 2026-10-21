"""Prepare modern annual means for five independent low-frequency records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


EXPECTED_IDS = ("iso2k_1397", "sisal_111.0_64", "sisal_539.0_336", "sisal_543.0_340", "sisal_900.0_540")


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists(): return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def main() -> int:
    project = root(); loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_40_dod2k_low_frequency_interpretability_v1"
    paths = {
        "candidates": loop / "round_36_dod2k_hma_identity_chronology_admissibility_v1/independent_low_frequency_candidates.csv",
        "values": loop / "round_35_dod2k_hma_hydroclimate_coverage_audit_v1/dod2k_hma_hydroclimate_allowed_values.csv",
        "contract": output / "ROUND_CONTRACT.md", "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists(): raise FileNotFoundError(path)
    target = output / "round40_low_frequency_input.mat"
    if target.exists(): raise FileExistsError(target)
    before = {name: state(path) for name, path in paths.items()}
    candidates = pd.read_csv(paths["candidates"]).sort_values("datasetId").reset_index(drop=True)
    if tuple(candidates.datasetId) != EXPECTED_IDS: raise RuntimeError("candidate-contract-failed")
    values = pd.read_csv(paths["values"]); years = np.arange(1940, 2001, dtype=np.int32)
    observations = np.full((61, 5), np.nan); sample_counts = np.zeros((61, 5), dtype=np.int32)
    starts: list[int] = []; ends: list[int] = []
    for column, item in enumerate(candidates.itertuples(index=False)):
        start, end = (5, 9) if item.datasetId == "iso2k_1397" else (1, 12)
        starts.append(start); ends.append(end)
        series = values[(values.datasetId == item.datasetId) & (values.year_ce >= 1940) & (values.year_ce <= 2000)].copy()
        if ((series.year_ce >= 1921) & (series.year_ce <= 1939)).any(): raise RuntimeError("locked-period-present")
        series["calendar_year"] = np.floor(series.year_ce).astype(int)
        grouped = series.groupby("calendar_year").paleo_value.agg(["mean", "count"])
        for row, year in enumerate(years):
            if int(year) in grouped.index:
                observations[row, column] = float(grouped.loc[int(year), "mean"])
                sample_counts[row, column] = int(grouped.loc[int(year), "count"])
        if np.isfinite(observations[:, column]).sum() < 15: raise RuntimeError(f"insufficient-modern-years:{item.datasetId}")
    manifest = candidates[["datasetId", "dataSetName", "geo_siteName", "latitude", "longitude", "archiveType", "paleoData_proxy", "interpretation_variable", "interpretation_seasonality", "originalDataURL"]].copy()
    manifest["season_start_month"] = starts; manifest["season_end_month"] = ends
    manifest.to_csv(output / "candidate_manifest.csv", index=False)
    savemat(target, {"years": years[:, None], "observations": observations, "sample_counts": sample_counts, "locations": candidates[["latitude", "longitude"]].to_numpy(float), "identity_ids": np.asarray(EXPECTED_IDS, dtype=object)[:, None], "season_start_month": np.asarray(starts, dtype=np.int32)[:, None], "season_end_month": np.asarray(ends, dtype=np.int32)[:, None]}, do_compression=True)
    after = {name: state(path) for name, path in paths.items()}; unchanged = before == after
    summary = {"status": "complete", "record_ids": list(EXPECTED_IDS), "modern_unique_year_counts": {EXPECTED_IDS[i]: int(np.isfinite(observations[:, i]).sum()) for i in range(5)}, "locked_1921_1939_accessed": False, "source_states_unchanged": unchanged, "output_sha256": state(target)["sha256"]}
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "preparation_source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2), encoding="utf-8")
    if not unchanged: raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
