"""Prepare Malan while preserving its missing 2000 proxy value as NaN."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


YEARS = np.arange(1940, 2001)


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
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def main() -> int:
    project = root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_54_malan_icecore_temperature_nested_validation_v1"
    paths = {
        "candidate": loop
        / "round_53_dod2k_temperature_identity_chronology_v1/annual_temperature_psm_candidates.csv",
        "values": loop
        / "round_52_dod2k_temperature_allowed_period_coverage_v1/dod2k_hma_temperature_allowed_values.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "preflight_conflict": output / "PREFLIGHT_CONFLICT.md",
        "original_driver": Path(__file__).with_name(
            "prepare_climate_reconstruction_loop_v2_round54_malan_temperature_v1.py"
        ),
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    target = output / "malan_temperature_network_v1.mat"
    if target.exists():
        raise FileExistsError(target)
    before = {name: state(path) for name, path in paths.items()}
    candidate = pd.read_csv(paths["candidate"])
    if len(candidate) != 1 or candidate.iloc[0].datasetId != "iso2k_1151":
        raise RuntimeError("malan-candidate-contract-failed")
    row = candidate.iloc[0]
    values = pd.read_csv(paths["values"])
    values = values[
        (values.datasetId.astype(str) == "iso2k_1151")
        & values.year_ce.between(1940, 2000)
    ].copy()
    if ((values.year_ce >= 1921) & (values.year_ce <= 1939)).any():
        raise RuntimeError("locked-period-value-present")
    values["year"] = np.round(values.year_ce).astype(int)
    if not np.allclose(values.year_ce, values.year, atol=1e-8) or values.year.duplicated().any():
        raise RuntimeError("modern-year-contract-failed")
    observations = values.set_index("year").paleo_value.reindex(YEARS).to_numpy(float).reshape(-1, 1)
    finite = int(np.isfinite(observations).sum())
    missing = YEARS[~np.isfinite(observations[:, 0])].tolist()
    if finite != 60 or missing != [2000]:
        raise RuntimeError(f"unexpected-malan-modern-support:{finite}:{missing}")
    locations = np.asarray([[float(row.latitude), float(row.longitude)]], dtype=float)
    savemat(
        target,
        {
            "years": YEARS.reshape(-1, 1),
            "observations": observations,
            "locations": locations,
            "identity_ids": np.asarray(["iso2k_1151"], dtype=object).reshape(-1, 1),
        },
        do_compression=True,
    )
    manifest = {
        "network_index_1based": 1,
        "datasetId": "iso2k_1151",
        "site_name": row.geo_siteName,
        "latitude": float(row.latitude),
        "longitude": float(row.longitude),
        "archive_type": row.archiveType,
        "measurement": row.paleoData_proxy,
        "season_code": "JJA",
        "documented_direction": "positive",
        "modern_finite_count": finite,
        "missing_years": "2000",
    }
    with (output / "network_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest))
        writer.writeheader()
        writer.writerow(manifest)
    summary = {
        "status": "complete",
        "datasetId": "iso2k_1151",
        "finite_count": finite,
        "missing_years": missing,
        "missing_values_preserved_without_interpolation": True,
        "locked_1921_1939_accessed": False,
    }
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "preparation_source_manifest.json").write_text(
        json.dumps(
            {"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged},
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
