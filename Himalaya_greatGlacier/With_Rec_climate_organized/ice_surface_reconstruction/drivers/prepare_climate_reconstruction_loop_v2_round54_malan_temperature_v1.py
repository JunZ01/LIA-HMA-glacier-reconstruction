"""Prepare the allowed-period Malan ice-core record for temperature validation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


YEARS = np.arange(1940, 2001)


def project_root() -> Path:
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
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_54_malan_icecore_temperature_nested_validation_v1"
    paths = {
        "candidate": loop
        / "round_53_dod2k_temperature_identity_chronology_v1/annual_temperature_psm_candidates.csv",
        "values": loop
        / "round_52_dod2k_temperature_allowed_period_coverage_v1/dod2k_hma_temperature_allowed_values.csv",
        "contract": output / "ROUND_CONTRACT.md",
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
    if (
        str(row.interpretation_direction).lower() != "positive"
        or str(row.interpretation_seasonality).lower() != "summer"
    ):
        raise RuntimeError("malan-direction-season-contract-failed")
    values = pd.read_csv(paths["values"])
    values = values[values.datasetId.astype(str) == "iso2k_1151"].copy()
    if ((values.year_ce >= 1921) & (values.year_ce <= 1939)).any():
        raise RuntimeError("locked-period-value-present")
    modern = values[(values.year_ce >= 1940) & (values.year_ce <= 2000)].copy()
    modern["rounded_year"] = np.round(modern.year_ce).astype(int)
    if not np.allclose(modern.year_ce, modern.rounded_year, atol=1e-8):
        raise RuntimeError("noninteger-modern-year")
    if modern.rounded_year.duplicated().any():
        raise RuntimeError("duplicate-modern-year")
    modern = modern.set_index("rounded_year").reindex(YEARS)
    if not np.array_equal(modern.index.to_numpy(int), YEARS):
        raise RuntimeError("modern-year-axis-failed")
    observations = modern.paleo_value.to_numpy(float).reshape(-1, 1)
    if not np.all(np.isfinite(observations)):
        raise RuntimeError("malan-modern-coverage-incomplete")
    locations = np.asarray([[float(row.latitude), float(row.longitude)]], dtype=float)
    identity_ids = np.asarray(["iso2k_1151"], dtype=object).reshape(-1, 1)
    savemat(
        target,
        {
            "years": YEARS.reshape(-1, 1),
            "observations": observations,
            "locations": locations,
            "identity_ids": identity_ids,
        },
        do_compression=True,
    )
    manifest_row = {
        "network_index_1based": 1,
        "datasetId": "iso2k_1151",
        "site_name": row.geo_siteName,
        "latitude": float(row.latitude),
        "longitude": float(row.longitude),
        "archive_type": row.archiveType,
        "measurement": row.paleoData_proxy,
        "season_code": "JJA",
        "documented_direction": "positive",
        "modern_finite_count": int(np.isfinite(observations).sum()),
    }
    with (output / "network_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_row))
        writer.writeheader()
        writer.writerow(manifest_row)
    summary = {
        "status": "complete",
        "datasetId": "iso2k_1151",
        "year_start": int(YEARS[0]),
        "year_end": int(YEARS[-1]),
        "finite_count": int(np.isfinite(observations).sum()),
        "locked_1921_1939_accessed": False,
    }
    (output / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "preparation_source_manifest.json").write_text(
        json.dumps(
            {
                "sources_before": before,
                "sources_after": after,
                "source_states_unchanged": unchanged,
            },
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
