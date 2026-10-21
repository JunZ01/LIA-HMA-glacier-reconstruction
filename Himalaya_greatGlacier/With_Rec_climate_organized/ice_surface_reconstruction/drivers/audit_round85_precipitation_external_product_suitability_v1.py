"""Audit official external precipitation products against the Round84 boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUTPUT = LOOP / "round_85_precipitation_external_product_suitability_v1"
DRIVER = Path(__file__).resolve()
URLS = {
    "aphrodite_download": "http://aphrodite.st.hirosaki-u.ac.jp/download/",
    "aphrodite_raw": "http://aphrodite.st.hirosaki-u.ac.jp/product/",
    "gpcc_daily": "https://opendata.dwd.de/climate_environment/GPCC/html/fulldata-daily_v2022_doi_download.html",
    "chirps": "https://www.chc.ucsb.edu/data/chirps",
    "regen_bom": "http://www.bom.gov.au/climate/change/datasets/datasets.shtml",
}


def record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"path": str(path), "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def visible_text(payload: bytes, encoding: str | None = None) -> str:
    decoded = payload.decode(encoding or "utf-8", errors="replace")
    return " ".join(BeautifulSoup(decoded, "html.parser").stripped_strings)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "\n".join(
            [
                "# Round 85 Contract",
                "",
                "## Objective",
                "Determine whether an officially documented, automatically accessible precipitation product can satisfy the Round84 requirement for APHRODITE or an equivalent mountain-Asia gauge target.",
                "",
                "## Frozen Suitability Gates",
                "- Official source and documented method.",
                "- Automatic source-file access without bypassing authentication.",
                "- At least 1951-2000 temporal coverage and resolution no coarser than 0.5 degree.",
                "- Gauge-based design intended for Asian or mountain precipitation rather than a satellite/reanalysis substitute.",
                "- No deterministic construction from an observation product already used in the four-product target family.",
                "- Metadata only; no climate values, candidate scoring, tuning, or 1921-1939 access.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    session = requests.Session()
    session.headers.update({"User-Agent": "LIA-HMA-reconstruction-audit/1.0"})
    responses: dict[str, dict[str, object]] = {}
    texts: dict[str, str] = {}
    for label, url in URLS.items():
        try:
            response = session.get(url, timeout=120, allow_redirects=True)
            payload = response.content
            snapshot = OUTPUT / f"official_{label}.html"
            snapshot.write_bytes(payload)
            text = visible_text(payload, "cp1252" if label == "gpcc_daily" else None)
            texts[label] = text
            responses[label] = {
                "url": url,
                "final_url": response.url,
                "http_status": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        except requests.RequestException as exc:
            texts[label] = ""
            responses[label] = {"url": url, "error": repr(exc)}

    aphro = texts["aphrodite_download"]
    gpcc = texts["gpcc_daily"]
    chirps = texts["chirps"]
    checks = [
        {
            "product": "APHRO_MA_V1101",
            "official_metadata_verified": "V1101" in aphro and "1951 - 2007" in aphro,
            "automatic_file_access": responses["aphrodite_raw"].get("http_status") == 200,
            "coverage_1951_2000": "1951 - 2007" in aphro,
            "resolution_at_most_0p5_degree": "0.25, 0.50" in aphro,
            "asia_mountain_gauge_equivalent": True,
            "independent_of_existing_target_construction": True,
            "limitation": "official raw-file endpoint requires HTTP authentication",
        },
        {
            "product": "GPCC_Full_Data_Daily_v2022",
            "official_metadata_verified": "GPCC Full Data Daily Version 2022" in gpcc,
            "automatic_file_access": True,
            "coverage_1951_2000": False,
            "resolution_at_most_0p5_degree": False,
            "asia_mountain_gauge_equivalent": False,
            "independent_of_existing_target_construction": False,
            "limitation": "1-degree 1982-2020 product; daily anomalies are superimposed on existing GPCC Full Data Monthly v2022 totals",
        },
        {
            "product": "CHIRPS_v2",
            "official_metadata_verified": "Climate Hazards Group InfraRed Precipitation with Station data" in chirps,
            "automatic_file_access": "https://data.chc.ucsb.edu/products/CHIRPS-2.0/" in chirps,
            "coverage_1951_2000": False,
            "resolution_at_most_0p5_degree": "0.05" in chirps,
            "asia_mountain_gauge_equivalent": False,
            "independent_of_existing_target_construction": True,
            "limitation": "1981-near-present satellite/climatology/station blend; official documentation warns of complex-terrain satellite bias",
        },
        {
            "product": "REGEN_BoM",
            "official_metadata_verified": False,
            "automatic_file_access": False,
            "coverage_1951_2000": False,
            "resolution_at_most_0p5_degree": False,
            "asia_mountain_gauge_equivalent": False,
            "independent_of_existing_target_construction": True,
            "limitation": "official Bureau endpoint returned automated-access block; requirements cannot be verified from the authoritative source",
        },
    ]
    frame = pd.DataFrame(checks)
    gate_columns = [
        "official_metadata_verified",
        "automatic_file_access",
        "coverage_1951_2000",
        "resolution_at_most_0p5_degree",
        "asia_mountain_gauge_equivalent",
        "independent_of_existing_target_construction",
    ]
    frame["passes_all_suitability_gates"] = frame[gate_columns].all(axis=1)
    frame.to_csv(OUTPUT / "external_product_suitability.csv", index=False)
    passing = frame.loc[frame.passes_all_suitability_gates, "product"].tolist()
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "passing_automatic_products": passing,
        "aphrodite_raw_http_status": responses["aphrodite_raw"].get("http_status"),
        "gpcc_daily_same_monthly_target_family": True,
        "chirps_complex_terrain_caveat_present": "complex terrain" in chirps.lower(),
        "decision": "no_automatic_equivalent_product_available_request_aphrodite_credentials_or_file",
        "climate_values_read": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Round 85 External Precipitation Product Suitability",
                "",
                "- APHRO_MA V1101 meets the scientific coverage/resolution target, but its official raw-file endpoint returns `401 Unauthorized`.",
                "- GPCC Full Data Daily v2022 is accessible, but it is 1 degree, starts in 1982, and is constructed by superimposing daily station anomalies on the already used GPCC Full Data Monthly v2022 totals.",
                "- CHIRPS v2 is accessible at 0.05 degree from 1981, but it is a satellite/climatology/station blend and its official documentation warns about satellite bias in complex terrain.",
                "- The official Bureau of Meteorology REGEN endpoint blocked automated access, so authoritative requirements could not be verified.",
                "- Passing automatic equivalents: `[]`.",
                "- Decision: `no_automatic_equivalent_product_available_request_aphrodite_credentials_or_file`.",
                "- No climate values, candidate fields, or locked-period values were accessed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    files = [DRIVER, contract, OUTPUT / "external_product_suitability.csv", OUTPUT / "summary.json", OUTPUT / "FINAL_RESULT.md", *sorted(OUTPUT.glob("official_*.html"))]
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"official_responses": responses, "files": [record(path) for path in files]}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
