from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUT = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_167_chin074_qamdo2_identity_recovery_v1"
RAW = OUT / "raw_unopened"

URLS = {
    "chin074_rwl": "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/asia/chin074.rwl",
    "chin074_noaa": "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/asia/chin074-rwl-noaa.txt",
    "chin074_crn": "https://www.ncei.noaa.gov/pub/data/paleo/treering/chronologies/asia/chin074.crn",
    "chin046_rwl": "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/asia/chin046.rwl",
    "chin046_noaa": "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/asia/chin046-rwl-noaa.txt",
    "chin046_crn": "https://www.ncei.noaa.gov/pub/data/paleo/treering/chronologies/asia/chin046.crn",
    "chin047_rwl": "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/asia/chin047.rwl",
    "chin047_noaa": "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/asia/chin047-rwl-noaa.txt",
    "chin047_crn": "https://www.ncei.noaa.gov/pub/data/paleo/treering/chronologies/asia/chin047.crn",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, path: Path) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "LIA-HMA-source-audit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
            status = getattr(response, "status", 200)
        path.write_bytes(payload)
        return {"status": int(status), "bytes": len(payload), "sha256": sha256(path), "error": None}
    except Exception as exc:
        return {"status": None, "bytes": 0, "sha256": None, "error": f"{type(exc).__name__}: {exc}"}


def rwl_sample_ids(path: Path) -> list[str]:
    # Tucson RWL sample identifiers occupy columns 1-8. Ring-width values are never parsed.
    ids: set[str] = set()
    for raw_line in path.read_bytes().splitlines():
        line = raw_line.decode("ascii", errors="ignore")
        if len(line) < 12:
            continue
        sample = line[:8].strip()
        year_field = line[8:12].strip()
        if sample and re.fullmatch(r"-?\d{1,4}", year_field):
            ids.add(sample)
    return sorted(ids)


def normalized_id(sample: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", sample.upper())


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key, url in URLS.items():
        suffix = url.rsplit("/", 1)[-1]
        path = RAW / suffix
        item = download(url, path)
        item.update({"url": url, "path": str(path)})
        manifest[key] = item

    sample_sets = {}
    for code in ("chin074", "chin046", "chin047"):
        item = manifest[f"{code}_rwl"]
        if item["error"] is None:
            sample_sets[code] = rwl_sample_ids(Path(item["path"]))
        else:
            sample_sets[code] = []

    normalized = {key: {normalized_id(v) for v in values} for key, values in sample_sets.items()}
    overlaps = {}
    for older in ("chin046", "chin047"):
        exact = sorted(set(sample_sets["chin074"]) & set(sample_sets[older]))
        norm = sorted(normalized["chin074"] & normalized[older])
        overlaps[older] = {
            "exact_sample_id_count": len(exact),
            "normalized_sample_id_count": len(norm),
            "exact_sample_ids": exact,
            "normalized_sample_ids": norm,
        }

    old_union = normalized["chin046"] | normalized["chin047"]
    new_set = normalized["chin074"]
    summary = {
        "experiment": "round_167_chin074_qamdo2_identity_recovery_v1",
        "locked_period_numeric_values_decoded": False,
        "paleo_trajectory_inspected": False,
        "sample_counts": {key: len(value) for key, value in sample_sets.items()},
        "overlaps": overlaps,
        "chin074_ids_reused_fraction": (len(new_set & old_union) / len(new_set)) if new_set else None,
        "chin074_unique_normalized_id_count": len(new_set - old_union),
        "all_required_rwl_downloads_succeeded": all(manifest[f"{code}_rwl"]["error"] is None for code in ("chin074", "chin046", "chin047")),
        "chin074_official_chronology_available": manifest["chin074_crn"]["error"] is None,
        "decision": "pending_metadata_and_project_alias_audit",
    }
    (OUT / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "sample_identity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
