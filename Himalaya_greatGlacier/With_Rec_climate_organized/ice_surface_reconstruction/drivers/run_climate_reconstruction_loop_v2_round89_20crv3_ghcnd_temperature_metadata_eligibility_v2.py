"""Resolve Round 89 dynamic-page metadata gates using the official DAS."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest,
    }


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    v1 = loop / "round_89_20crv3_ghcnd_temperature_metadata_eligibility_v1"
    output = loop / "round_89_20crv3_ghcnd_temperature_metadata_eligibility_v2"
    paths = {
        "v1_summary": v1 / "summary.json",
        "v1_manifest": v1 / "source_manifest.json",
        "v1_dds": v1 / "20crv3_monthly_2m_dds.txt",
        "v1_das": v1 / "20crv3_monthly_2m_das.txt",
        "v1_catalog": v1 / "20crv3_monthly_2m_catalog.txt",
        "v1_station_inventory": v1 / "ghcnd_hma_paired_temperature_station_inventory.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    prior = json.loads(paths["v1_summary"].read_text(encoding="utf-8"))
    prior_manifest = json.loads(paths["v1_manifest"].read_text(encoding="utf-8"))
    if prior["passes_metadata_eligibility"]:
        raise RuntimeError("v1-did-not-stop")
    if not prior_manifest["local_source_states_unchanged"]:
        raise RuntimeError("v1-local-source-integrity-failed")
    if prior["temperature_values_read"]:
        raise RuntimeError("v1-read-temperature-values")

    dds = paths["v1_dds"].read_text(encoding="utf-8", errors="replace")
    das = paths["v1_das"].read_text(encoding="utf-8", errors="replace")
    catalog = paths["v1_catalog"].read_text(encoding="utf-8", errors="replace")
    time_match = re.search(r"time\s*=\s*(\d+)", dds)
    lat_match = re.search(r"lat\s*=\s*(\d+)", dds)
    lon_match = re.search(r"lon\s*=\s*(\d+)", dds)
    if not time_match or not lat_match or not lon_match:
        raise RuntimeError("v1-dds-dimension-parse-failed")
    dimensions = {
        "monthly_time_count": int(time_match.group(1)),
        "latitude_count": int(lat_match.group(1)),
        "longitude_count": int(lon_match.group(1)),
    }
    inventory = pd.read_csv(paths["v1_station_inventory"])
    eligible = inventory[inventory.catalog_eligible.astype(bool)]
    region_counts = eligible.groupby("region").size().to_dict()
    gates = {
        "das_identifies_20crv3": "NOAA/CIRES/DOE 20th Century Reanalysis (V3)" in das,
        "das_identifies_surface_pressure_observations": "International Surface Pressure Databank version 4.7" in das,
        "das_identifies_monthly_2m_ensemble_mean_air_temperature": all(
            token in das
            for token in (
                'String var_desc "Air temperature"',
                'String level_desc "2 m"',
                'String statistic "Ensemble Mean"',
                'String standard_name "air_temperature"',
            )
        ),
        "catalog_contains_monthly_2m_air_file": "air.2m.mon.mean.nc" in catalog,
        "dds_contains_air_time_lat_lon_grid": all(token in dds for token in ("air[time", "[lat", "[lon")),
        "dds_grid_is_1degree_global": dimensions["latitude_count"] == 181 and dimensions["longitude_count"] == 360,
        "dds_month_count_matches_1806_2015": dimensions["monthly_time_count"] == 210 * 12,
        "das_units_are_kelvin": 'String units "degK"' in das,
        "at_least_30_hma_catalog_eligible_stations": len(eligible) >= 30,
        "at_least_5_stations_in_each_region": all(region_counts.get(name, 0) >= 5 for name in ("west_lt80", "central_80_90", "east_ge90")),
        "at_least_5_countries": eligible.country.nunique() >= 5,
        "no_station_or_20cr_temperature_values_read": True,
        "no_remote_reread_in_v2": True,
    }
    accepted = bool(all(gates.values()))
    summary = {
        "status": "complete",
        "supersedes": "round_89_20crv3_ghcnd_temperature_metadata_eligibility_v1",
        "v1_failure_class": "dynamic_human_facing_page_text_unavailable_to_requests",
        "v2_change": "replace_three_page_text_gates_with_official_das_attributes",
        "remote_requests_made": 0,
        "temperature_values_read": False,
        "holdout_years": [2001, 2015],
        "source_dimensions": dimensions,
        "ghcnd_catalog_eligible_station_count": int(len(eligible)),
        "eligible_region_counts": {name: int(region_counts.get(name, 0)) for name in ("west_lt80", "central_80_90", "east_ge90")},
        "eligible_country_count": int(eligible.country.nunique()),
        "gates": gates,
        "passes_metadata_eligibility": accepted,
        "decision": (
            "freeze_20crv3_vs_ghcnd_2001_2015_value_validation_protocol"
            if accepted
            else "stop_new_temperature_route_on_machine_metadata_conflict"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Round 89 V2 Metadata Eligibility",
                "",
                "- V1 stopped on dynamic-page text only; it is retained.",
                "- V2 uses the official saved DAS and makes no remote request.",
                f"- Eligible GHCN-D stations: `{len(eligible)}`; regions: `{summary['eligible_region_counts']}`; countries: `{summary['eligible_country_count']}`.",
                f"- Temperature values read: `false`.",
                f"- All gates pass: `{str(accepted).lower()}`.",
                "",
                f"Decision: `{summary['decision']}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "sources_before": before,
                "sources_after": after,
                "source_states_unchanged": unchanged,
                "remote_requests_made": 0,
                "temperature_values_read": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("round89-v2-source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
