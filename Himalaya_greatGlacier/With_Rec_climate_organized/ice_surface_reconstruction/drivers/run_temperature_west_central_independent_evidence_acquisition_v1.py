from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


SEALED_START = 1921
SEALED_END = 1939
ALLOWED_WINDOWS = ((1880, 1920), (1940, 1950))

SOURCES = {
    "reaches_readme": "https://www.ncei.noaa.gov/pub/data/paleo/historical/asia/china/reaches/readme-reaches.txt",
    "reaches_annual": "https://www.ncei.noaa.gov/pub/data/paleo/historical/asia/china/reaches/wang2023-reaches_annual_temp.txt",
    "reaches_summer": "https://www.ncei.noaa.gov/pub/data/paleo/historical/asia/china/reaches/wang2023-reaches_summer_temp.txt",
    "reaches_winter": "https://www.ncei.noaa.gov/pub/data/paleo/historical/asia/china/reaches/wang2023-reaches_winter_temp.txt",
    "reaches_record_count": "https://www.ncei.noaa.gov/pub/data/paleo/historical/asia/china/reaches/wang2023-reaches_num_temp_records.txt",
    "reaches_polygons": "https://www.ncei.noaa.gov/pub/data/paleo/historical/asia/china/reaches/wang2023-reaches_spatialpolygon.zip",
    "upper_indus_crossref": "https://api.crossref.org/works/10.1016/j.scitotenv.2024.177393",
    "pakistan_2026_crossref": "https://api.crossref.org/works/10.1088/2515-7620/ae718e",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "LIA-HMA-evidence-audit/1.0 (research use)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with target.open("wb") as stream:
            shutil.copyfileobj(response, stream)


def parse_allowed_table(path: Path, label: str) -> tuple[list[dict[str, str]], list[str]]:
    """Read values only after the row year is known to be outside the seal."""
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    with path.open("r", encoding="utf-8-sig", errors="strict") as stream:
        for raw_line in stream:
            if raw_line.startswith("#") or not raw_line.strip():
                continue
            year_token = raw_line.split("\t", 1)[0].strip()
            if header is None and not re.fullmatch(r"-?\d+", year_token):
                header = raw_line.rstrip("\r\n").split("\t")
                continue
            if not re.fullmatch(r"-?\d+", year_token):
                continue
            year = int(year_token)
            if SEALED_START <= year <= SEALED_END:
                continue
            if not any(start <= year <= end for start, end in ALLOWED_WINDOWS):
                continue
            if header is None:
                raise RuntimeError(f"No header found before data in {path}")
            values = raw_line.rstrip("\r\n").split("\t")
            rows.append({"source": label, **dict(zip(header, values))})
    if header is None:
        raise RuntimeError(f"No tabular header found in {path}")
    return rows, header


def inventory_polygon_zip(path: Path, output_dir: Path) -> list[dict[str, object]]:
    extracted = output_dir / "reaches_polygons_unmodified"
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(extracted)
    inventory = []
    for item in sorted(extracted.rglob("*")):
        if item.is_file():
            inventory.append(
                {
                    "relative_path": item.relative_to(extracted).as_posix(),
                    "size_bytes": item.stat().st_size,
                    "sha256": sha256(item),
                }
            )
    return inventory


def main(output_dir: str) -> None:
    out = Path(output_dir).resolve()
    raw = out / "official_sources_unmodified"
    raw.mkdir(parents=True, exist_ok=False)

    manifest = []
    local_paths: dict[str, Path] = {}
    for key, url in SOURCES.items():
        suffix = Path(urllib.request.urlparse(url).path).suffix or ".json"
        target = raw / f"{key}{suffix}"
        download(url, target)
        local_paths[key] = target
        manifest.append(
            {
                "source_id": key,
                "url": url,
                "local_path": str(target),
                "size_bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )

    allowed_rows: list[dict[str, str]] = []
    headers: dict[str, list[str]] = {}
    for key in ("reaches_annual", "reaches_summer", "reaches_winter", "reaches_record_count"):
        rows, header = parse_allowed_table(local_paths[key], key)
        allowed_rows.extend(rows)
        headers[key] = header

    all_fields = []
    for row in allowed_rows:
        for field in row:
            if field not in all_fields:
                all_fields.append(field)
    with (out / "reaches_allowed_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(allowed_rows)

    polygon_inventory = inventory_polygon_zip(local_paths["reaches_polygons"], out)
    with (out / "reaches_polygon_file_inventory.json").open("w", encoding="utf-8") as stream:
        json.dump(polygon_inventory, stream, indent=2)

    summary = {
        "experiment": "temperature_west_central_independent_evidence_acquisition_v1",
        "role": "metadata_and_allowed_year_acquisition_only",
        "sealed_period": [SEALED_START, SEALED_END],
        "allowed_windows": [list(window) for window in ALLOWED_WINDOWS],
        "sealed_values_converted_to_numeric": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "download_count": len(manifest),
        "allowed_reaches_row_count": len(allowed_rows),
        "reaches_headers": headers,
        "polygon_file_count": len(polygon_inventory),
        "next_gate": "spatial_overlap_record_density_and_source_identity_audit",
    }
    with (out / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    with (out / "source_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_temperature_west_central_independent_evidence_acquisition_v1.py OUTPUT_DIR")
    main(sys.argv[1])
