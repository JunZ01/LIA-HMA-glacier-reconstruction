from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


URL = "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/asia/chin041.rwl"


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_id(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def prefix_identity(path: Path) -> dict[str, Any]:
    sample_ids: set[str] = set()
    segment_count = 0
    locked_intersecting_segment_count = 0
    with path.open("rb") as stream:
        for raw_line in stream:
            prefix = raw_line[:12].decode("ascii", errors="ignore")
            if len(prefix) < 12:
                continue
            sample = prefix[:8].strip()
            year_token = prefix[8:12].strip()
            if not sample or not re.fullmatch(r"-?[0-9]+", year_token):
                continue
            start = int(year_token)
            sample_ids.add(normalize_id(sample))
            segment_count += 1
            if start <= 1939 and start + 9 >= 1921:
                locked_intersecting_segment_count += 1
    return {
        "path": str(path),
        "sha256": sha256(path),
        "sample_count": len(sample_ids),
        "segment_count": segment_count,
        "locked_intersecting_segment_count_skipped_after_prefix": locked_intersecting_segment_count,
        "sample_ids": sorted(sample_ids),
        "ring_width_tokens_read_or_decoded": False,
    }


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_191_chin041_hengduan_sample_identity_v1"
    raw = output / "raw_unopened"
    raw.mkdir(parents=True, exist_ok=False)
    destination = raw / "chin041.rwl"
    request = urllib.request.Request(URL, headers={"User-Agent": "LIA-HMA-source-audit/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    if len(payload) < 1000:
        raise RuntimeError("unexpectedly-small-chin041-download")
    destination.write_bytes(payload)

    hengduan_root = loop / (
        "round_178_mendeley_east_hma_temperature_candidates_v1/raw_unopened/889n6ccx4z_v1"
    )
    hengduan_files = [hengduan_root / name for name in ["DLL", "FGH", "KQH", "XYS", "YPL"]]
    source_hashes_before = {str(path): sha256(path) for path in hengduan_files}
    chin = prefix_identity(destination)
    hengduan = [prefix_identity(path) for path in hengduan_files]
    source_hashes_after = {str(path): sha256(path) for path in hengduan_files}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("hengduan-source-hash-changed")

    candidate_union = {sample for item in hengduan for sample in item["sample_ids"]}
    exact_overlap = sorted(set(chin["sample_ids"]) & candidate_union)
    comparison = {
        "official_chin041": chin,
        "hengduan_files": hengduan,
        "exact_normalized_sample_id_overlap_count": len(exact_overlap),
        "exact_normalized_sample_id_overlaps": exact_overlap,
    }
    (output / "sample_identity.json").write_text(
        json.dumps(comparison, indent=2), encoding="ascii"
    )
    decision = (
        "direct_sample_reuse_confirmed"
        if exact_overlap
        else "no_direct_sample_reuse_but_near_site_climate_independence_not_established"
    )
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "official_url": URL,
        "download_sha256": chin["sha256"],
        "chin041_sample_count": chin["sample_count"],
        "hengduan_sample_count": len(candidate_union),
        "exact_normalized_sample_id_overlap_count": len(exact_overlap),
        "ring_width_tokens_read_or_decoded": False,
        "locked_1921_1939_ring_width_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": True,
        "decision": decision,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "official_download": {"url": URL, "path": str(destination), "sha256": chin["sha256"]},
                "hengduan_sources": source_hashes_after,
            },
            indent=2,
        ),
        encoding="ascii",
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 191 CHIN041--Hengduan Sample Identity\n\n"
        f"- Official CHIN041 sample IDs: `{chin['sample_count']}`.\n"
        f"- Hengduan candidate sample IDs: `{len(candidate_union)}`.\n"
        f"- Exact normalized sample-ID overlaps: `{len(exact_overlap)}`.\n"
        "- Ring-width tokens decoded: `false`.\n"
        "- Locked-period ring-width values accessed: `false`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

