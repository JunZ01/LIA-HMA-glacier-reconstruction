from __future__ import annotations

import csv
import hashlib
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
AUDIT = LOOP / "temperature_historical_imd_direct_station_table_leaf_audit_v4"
ROUND277 = LOOP / "round_277_direct_temperature_station_panel_1850_1950_v2"
OUT = LOOP / "temperature_historical_imd_ghcn_overlap_page_panel_v1"
DRIVER = Path(__file__).resolve()

TARGET_STATIONS = {
    "Darjeeling": "DARJEELING",
    "Simla": "SHIMLA",
    "Leh": "LEH_KASHMIR",
    "Murree": "MURREE",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, attempts: int = 8) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 LIA historical climate audit"},
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                destination.write_bytes(response.read())
            return
        except Exception as error:
            last_error = error
            if attempt == attempts - 1:
                break
            time.sleep(min(45, 3 * (attempt + 1)))
    raise RuntimeError(f"failed to download {url}") from last_error


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    images = OUT / "official_full_resolution_station_pages"
    images.mkdir()

    candidates = read_csv(AUDIT / "station_table_leaf_candidates.csv")
    selected = [
        row
        for row in candidates
        if row["station"] in TARGET_STATIONS
        and row["leaf_relation"] == "same_leaf"
        and int(row["report_year"]) <= 1890
    ]
    selected.sort(key=lambda row: (int(row["report_year"]), row["station"]))

    page_rows: list[dict[str, object]] = []
    page_lookup: dict[tuple[str, int], Path] = {}
    for row in selected:
        item_id = row["item_id"]
        leaf = int(row["candidate_leaf"])
        key = (item_id, leaf)
        if key in page_lookup:
            continue
        item_dir = images / item_id
        item_dir.mkdir(exist_ok=True)
        path = item_dir / f"leaf_{leaf:04d}.jpg"
        url = f"https://archive.org/download/{item_id}/page/leaf{leaf}.jpg"
        download(url, path)
        page_lookup[key] = path
        page_rows.append(
            {
                "item_id": item_id,
                "leaf": leaf,
                "url": url,
                "image_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    station_rows: list[dict[str, object]] = []
    for row in selected:
        key = (row["item_id"], int(row["candidate_leaf"]))
        station_rows.append(
            {
                "report_year": int(row["report_year"]),
                "imd_station": row["station"],
                "ghcn_station_name": TARGET_STATIONS[row["station"]],
                "item_id": row["item_id"],
                "leaf": int(row["candidate_leaf"]),
                "leaf_relation": row["leaf_relation"],
                "image_path": str(page_lookup[key]),
                "numeric_status": "pending_matlab_coordinate_ocr",
            }
        )

    panel = read_csv(ROUND277 / "monthly_temperature_panel.csv")
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in panel:
        if row["station_name"] not in TARGET_STATIONS.values():
            continue
        year = int(row["year"])
        if year < 1875 or year > 1890:
            continue
        if row["series"] not in {"qcu", "qcf"}:
            continue
        if row["quality_accepted"].lower() != "true":
            continue
        value = row["analysis_temperature_c"].strip()
        if not value:
            continue
        grouped[(row["station_name"], row["series"], year)].append(float(value))

    ghcn_rows: list[dict[str, object]] = []
    selected_years = {
        (str(row["ghcn_station_name"]), int(row["report_year"]))
        for row in station_rows
    }
    for (station, series, year), values in sorted(grouped.items()):
        if (station, year) not in selected_years:
            continue
        ghcn_rows.append(
            {
                "station_name": station,
                "series": series,
                "year": year,
                "accepted_months": len(values),
                "annual_mean_c": sum(values) / len(values),
                "complete_12_month_year": len(values) == 12,
            }
        )

    write_csv(
        OUT / "station_year_page_panel.csv",
        station_rows,
        [
            "report_year", "imd_station", "ghcn_station_name", "item_id",
            "leaf", "leaf_relation", "image_path", "numeric_status",
        ],
    )
    write_csv(
        OUT / "unique_page_manifest.csv",
        page_rows,
        ["item_id", "leaf", "url", "image_path", "bytes", "sha256"],
    )
    write_csv(
        OUT / "ghcn_annual_overlap.csv",
        ghcn_rows,
        [
            "station_name", "series", "year", "accepted_months",
            "annual_mean_c", "complete_12_month_year",
        ],
    )

    summary = {
        "experiment": "temperature_historical_imd_ghcn_overlap_page_panel_v1",
        "same_leaf_station_years": len(station_rows),
        "unique_full_resolution_pages": len(page_rows),
        "downloaded_bytes": sum(int(row["bytes"]) for row in page_rows),
        "ghcn_annual_rows": len(ghcn_rows),
        "target_stations": TARGET_STATIONS,
        "annual_imd_values_promoted": False,
        "locked_period_values_read": False,
        "decision": "advance_matlab_coordinate_ocr",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                "inputs": {
                    "leaf_audit": str(AUDIT / "station_table_leaf_candidates.csv"),
                    "round277_panel": str(ROUND277 / "monthly_temperature_panel.csv"),
                },
                "pages": page_rows,
            },
            indent=2,
            ensure_ascii=True,
        ) + "\n",
        encoding="utf-8",
    )
    (OUT / "EXPERIMENT_CONTRACT.md").write_text(
        """# IMD-GHCN Overlap Page Panel Contract

- Admit only same-leaf station/table-title identities from the v4 audit.
- Restrict to Darjeeling, Shimla, Leh, and Murree, which overlap Round 277.
- Download full-resolution 1875-1890 primary publication pages to E only.
- Compute GHCN annual means only for matching allowed years and accepted months.
- Do not promote IMD OCR values in this preparation step.
- Do not access 1921-1939, production data, EnKF, or OGGM.
""",
        encoding="utf-8",
    )
    (OUT / "FINAL_RESULT.md").write_text(
        f"""# IMD-GHCN Overlap Page Panel

Prepared {len(station_rows)} same-leaf station-year identities from four
stations shared with Round 277. Downloaded {len(page_rows)} unique official
full-resolution pages ({summary['downloaded_bytes']} bytes) and calculated
{len(ghcn_rows)} matching qcu/qcf annual rows.

No IMD OCR number is accepted yet. The panel is ready for MATLAB coordinate
OCR and engine-to-engine agreement checks.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
