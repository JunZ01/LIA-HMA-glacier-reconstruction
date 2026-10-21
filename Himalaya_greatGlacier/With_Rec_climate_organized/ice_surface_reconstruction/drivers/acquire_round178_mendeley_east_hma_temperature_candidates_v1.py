from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASETS = (
    ("889n6ccx4z", 1),
    ("s7t4zc34sr", 2),
)
USER_AGENT = "Mozilla/5.0 HMA-climate-public-data-recovery/1.0"


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


def fetch(url: str, accept: str = "*/*") -> tuple[bytes, str, str]:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": accept}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read(), response.geturl(), response.headers.get("Content-Type", "")


def safe_filename(value: str) -> str:
    value = value.replace("/", "_").replace("\\", "_")
    value = re.sub(r"[\x00-\x1f<>:\"|?*]", "_", value).strip(" .")
    if not value:
        raise RuntimeError("empty-filename")
    return value


def main() -> int:
    root = project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/"
        "round_178_mendeley_east_hma_temperature_candidates_v1"
    )
    output.mkdir(parents=True, exist_ok=False)
    metadata_dir = output / "metadata_unmodified"
    raw_dir = output / "raw_unopened"
    metadata_dir.mkdir()
    raw_dir.mkdir()

    manifest: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []
    for dataset_id, version in DATASETS:
        snapshot_url = (
            f"https://data.mendeley.com/public-api/datasets/{dataset_id}/"
            f"snapshot/{version}"
        )
        files_url = (
            f"https://data.mendeley.com/public-api/datasets/{dataset_id}/files?"
            f"folder_id=root&version={version}"
        )
        snapshot_bytes, snapshot_final, snapshot_type = fetch(
            snapshot_url, "application/json"
        )
        files_bytes, files_final, files_type = fetch(
            files_url, "application/vnd.mendeley-public-dataset.1+json"
        )
        snapshot_path = metadata_dir / f"{dataset_id}_v{version}_snapshot.json"
        files_path = metadata_dir / f"{dataset_id}_v{version}_files.json"
        snapshot_path.write_bytes(snapshot_bytes)
        files_path.write_bytes(files_bytes)
        snapshot = json.loads(snapshot_bytes.decode("utf-8"))
        files = json.loads(files_bytes.decode("utf-8"))
        dataset_target = raw_dir / f"{dataset_id}_v{version}"
        dataset_target.mkdir()
        for item in files:
            filename = safe_filename(str(item["filename"]))
            content = item["content_details"]
            payload, final_url, content_type = fetch(str(content["download_url"]))
            path = dataset_target / filename
            path.write_bytes(payload)
            actual_hash = sha256(path)
            expected_hash = str(content["sha256_hash"])
            if actual_hash.lower() != expected_hash.lower():
                raise RuntimeError(f"publisher-sha256-mismatch:{dataset_id}:{filename}")
            manifest.append(
                {
                    "dataset_id": dataset_id,
                    "version": version,
                    "dataset_doi": snapshot.get("doi", ""),
                    "file_id": item["id"],
                    "filename": filename,
                    "description": item.get("description", ""),
                    "publisher_size_bytes": content["size"],
                    "downloaded_size_bytes": len(payload),
                    "publisher_sha256": expected_hash,
                    "downloaded_sha256": actual_hash,
                    "content_type": content_type,
                    "download_url": content["download_url"],
                    "final_url": final_url,
                    "opened_or_parsed": False,
                }
            )
        dataset_summaries.append(
            {
                "dataset_id": dataset_id,
                "version": version,
                "doi": snapshot.get("doi", ""),
                "name": snapshot.get("name", ""),
                "licence": snapshot.get("licence", {}).get("short_name", ""),
                "file_count": len(files),
                "snapshot_url": snapshot_url,
                "snapshot_final_url": snapshot_final,
                "snapshot_content_type": snapshot_type,
                "snapshot_sha256": sha256(snapshot_path),
                "files_url": files_url,
                "files_final_url": files_final,
                "files_content_type": files_type,
                "files_sha256": sha256(files_path),
            }
        )

    (output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="ascii"
    )
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "datasets": dataset_summaries,
        "downloaded_file_count": len(manifest),
        "all_publisher_sha256_checks_pass": all(
            row["publisher_sha256"] == row["downloaded_sha256"] for row in manifest
        ),
        "all_files_left_unopened": all(not row["opened_or_parsed"] for row in manifest),
        "spreadsheet_or_tree_ring_value_accessed": False,
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "decision": "advance_to_independent_structure_and_source_identity_audit",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 178 Mendeley East-HMA Temperature Candidate Acquisition\n\n"
        f"- Datasets preserved: `{len(dataset_summaries)}`.\n"
        f"- Files preserved unchanged: `{len(manifest)}`.\n"
        "- Publisher SHA256 checks passed: `true`.\n"
        "- File contents opened or parsed: `false`.\n"
        "- Locked-period values accessed: `false`.\n\n"
        "Decision: `advance_to_independent_structure_and_source_identity_audit`.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
