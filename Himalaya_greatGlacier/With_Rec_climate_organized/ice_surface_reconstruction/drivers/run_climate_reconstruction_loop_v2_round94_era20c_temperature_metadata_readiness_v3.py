"""Audit ERA-20C metadata and freeze its untouched modern validation."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests


EXPECTED_FACTS_SHA256 = "9a74f57d1aa43044ecdbba0db0e14b7e74370be3e31f02189b9eee3df8266c87"
ECMWF_URL = "https://confluence.ecmwf.int/spaces/CKB/pages/73011217/What+is+ERA-20C"
JAMSTEC_CATALOG_URL = (
    "https://www.jamstec.go.jp/RID/thredds/catalog/testAll/surface/"
    "common-grid/monthly/t2m/ERA-20C/catalog.xml"
)
WDCC_PARENT_URL = "https://www.wdc-climate.de/ui/entry?acronym=EKF400_v2.0"
WDCC_MEAN_URL = "https://www.wdc-climate.de/ui/entry?acronym=EKF400_ens_mem_Mean_v2.0"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(session: requests.Session, url: str) -> tuple[requests.Response, dict[str, Any]]:
    response = session.get(url, timeout=120)
    response.raise_for_status()
    return response, {
        "url": response.url,
        "http_status": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "byte_count": len(response.content),
        "sha256": sha256_bytes(response.content),
    }


def project_root() -> Path:
    root = Path(__file__).resolve()
    while not (root / "LIA_PROJECT_FACTS.md").exists():
        if root.parent == root:
            raise RuntimeError("project-root-not-found")
        root = root.parent
    return root


def catalog_months(content: bytes) -> tuple[list[tuple[int, int]], list[str]]:
    root = ET.fromstring(content)
    pattern = re.compile(r"ERA-20C_surface_common-grid_monthly_t2m_(\d{4})-(\d{2})\.nc$")
    rows: list[tuple[int, int, str]] = []
    for element in root.iter():
        url_path = element.attrib.get("urlPath", "")
        match = pattern.search(url_path)
        if match:
            rows.append((int(match.group(1)), int(match.group(2)), url_path))
    rows = sorted(set(rows))
    return [(year, month) for year, month, _ in rows], [url for _, _, url in rows]


def main() -> int:
    root = project_root()
    output = (
        root
        / "validation_data/results/paleo_surface_validation_v0"
        / "climate_reconstruction_improvement_loop_v2"
        / "round_94_era20c_temperature_metadata_readiness_v3"
    )
    output.mkdir(parents=True, exist_ok=True)
    contract = output / "ROUND_CONTRACT.md"
    if not contract.exists():
        raise FileNotFoundError(contract)

    facts = root / "LIA_PROJECT_FACTS.md"
    facts_before = sha256_path(facts)
    if facts_before != EXPECTED_FACTS_SHA256:
        raise RuntimeError(f"facts-hash-conflict:{facts_before}")

    round93 = (
        root
        / "validation_data/results/paleo_surface_validation_v0"
        / "climate_reconstruction_improvement_loop_v2"
        / "round_93_20crv3_ghcnm_v4_temperature_value_validation_v4"
        / "summary.json"
    )
    round93_summary = json.loads(round93.read_text(encoding="utf-8"))
    if round93_summary["passes_20crv3_temperature_value_validation"]:
        raise RuntimeError("round93-state-conflict")
    if round93_summary["validation_years"] != [2001, 2015]:
        raise RuntimeError("round93-target-period-conflict")

    local_before = {
        "facts": {"path": str(facts), "sha256": facts_before},
        "round93_summary": {"path": str(round93), "sha256": sha256_path(round93)},
        "contract": {"path": str(contract), "sha256": sha256_path(contract)},
    }
    session = requests.Session()
    session.headers.update({"User-Agent": "LIA-HMA-ERA20C-metadata-readiness/1.0"})
    ecmwf, ecmwf_request = fetch(session, ECMWF_URL)
    catalog, catalog_request = fetch(session, JAMSTEC_CATALOG_URL)
    wdcc_parent, wdcc_parent_request = fetch(session, WDCC_PARENT_URL)
    wdcc_mean, wdcc_mean_request = fetch(session, WDCC_MEAN_URL)

    months, url_paths = catalog_months(catalog.content)
    expected_months = [(year, month) for year in range(1900, 2011) for month in range(1, 13)]
    sample_path = next(
        path for path in url_paths if path.endswith("ERA-20C_surface_common-grid_monthly_t2m_1981-01.nc")
    )
    das_url = f"https://www.jamstec.go.jp/RID/thredds/ncss/grid/{sample_path}/dataset.html"
    das, das_request = fetch(session, das_url)

    ecmwf_text = ecmwf.text.lower()
    das_text = das.text.lower()
    wdcc_parent_text = wdcc_parent.text.lower()
    wdcc_mean_text = wdcc_mean.text.lower()
    gates = {
        "era20c_1900_2010": "1900-2010" in ecmwf_text,
        "era20c_surface_pressure_only": "surface pressure" in ecmwf_text,
        "era20c_surface_marine_winds_only": "surface marine winds only" in ecmwf_text,
        "jamstec_exact_1332_month_axis": months == expected_months,
        "jamstec_t2m_metadata": "t2m" in das_text and "era-20c" in das_text,
        "jamstec_public_metadata_access": das.status_code == 200,
        "ekf400_1602_2003": "1602-01-15" in wdcc_parent_text and "2003-12-15" in wdcc_parent_text,
        "ekf400_two_degree": "2x2 degree" in wdcc_parent_text,
        "ekf400_mean_monthly_air_temperature": "air_temperature" in wdcc_mean_text and "monthly" in wdcc_mean_text,
        "ekf400_registered_access": "registered users" in wdcc_mean_text,
        "ekf400_mean_size_3_31_gib": "3.31 gib" in wdcc_mean_text,
    }
    era_ready = all(gates[key] for key in (
        "era20c_1900_2010",
        "era20c_surface_pressure_only",
        "era20c_surface_marine_winds_only",
        "jamstec_exact_1332_month_axis",
        "jamstec_t2m_metadata",
        "jamstec_public_metadata_access",
    ))
    summary = {
        "status": "complete",
        "climate_values_read": False,
        "locked_1921_1939_accessed": False,
        "era20c_catalog_month_count": len(months),
        "era20c_catalog_first_month": list(months[0]),
        "era20c_catalog_last_month": list(months[-1]),
        "era20c_sample_das_url": das_url,
        "gates": gates,
        "era20c_metadata_ready": era_ready,
        "round95_frozen_target": {
            "product": "GHCN-M v4 QCU monthly TAVG",
            "years": [1981, 2000],
            "previously_parsed_by_project": False,
            "minimum_complete_years": 16,
            "station_support": {"total": 30, "each_region": 5, "countries": 5},
            "performance_gates_unchanged_from_round92": True,
        },
        "conditional_early_period": [1900, 1920],
        "ekf400_route": {
            "coverage": [1602, 2003],
            "ensemble_mean_size_gib": 3.31,
            "access": "registered_users",
            "priority": "defer_until_era20c_early_twentieth_century_route_is_resolved",
        },
        "decision": (
            "freeze_era20c_1981_2000_ghcnm_qcu_validation"
            if era_ready
            else "reject_era20c_metadata_route"
        ),
    }
    (output / "ecmwf_era20c_metadata.html").write_bytes(ecmwf.content)
    (output / "jamstec_era20c_t2m_catalog.xml").write_bytes(catalog.content)
    (output / "jamstec_era20c_t2m_1981_01_ncss_metadata.html").write_bytes(das.content)
    (output / "wdcc_ekf400_parent_metadata.html").write_bytes(wdcc_parent.content)
    (output / "wdcc_ekf400_mean_metadata.html").write_bytes(wdcc_mean.content)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    final_lines = [
        "# ERA-20C Metadata Readiness",
        "",
        f"- ERA-20C metadata ready: `{str(era_ready).lower()}`.",
        f"- Public monthly t2m catalog: `{len(months)}` months, `{months[0]}` to `{months[-1]}`.",
        "- Frozen validation target: GHCN-M v4 QCU 1981-2000; no target value was read.",
        "- ERA-20C 1900-1920 remains inaccessible until every modern validation gate passes.",
        "- EKF400v2 is deferred because its ensemble mean requires registered access and is 3.31 GiB.",
        "",
        f"Decision: `{summary['decision']}`.",
        "",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(final_lines), encoding="utf-8")
    local_after = {
        "facts": {"path": str(facts), "sha256": sha256_path(facts)},
        "round93_summary": {"path": str(round93), "sha256": sha256_path(round93)},
        "contract": {"path": str(contract), "sha256": sha256_path(contract)},
    }
    manifest = {
        "local_sources_before": local_before,
        "local_sources_after": local_after,
        "local_source_states_unchanged": local_before == local_after,
        "requests": {
            "ecmwf": ecmwf_request,
            "jamstec_catalog": catalog_request,
            "jamstec_ncss_metadata": das_request,
            "wdcc_parent": wdcc_parent_request,
            "wdcc_mean": wdcc_mean_request,
        },
        "climate_values_read": False,
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["local_source_states_unchanged"]:
        raise RuntimeError("local-source-integrity-failed")
    print(json.dumps(summary, indent=2))
    return 0 if era_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
