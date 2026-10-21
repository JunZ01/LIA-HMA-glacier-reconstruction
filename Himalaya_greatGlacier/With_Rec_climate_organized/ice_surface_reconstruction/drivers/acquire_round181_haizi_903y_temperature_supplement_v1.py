from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC11554784"
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


def fetch(url: str) -> tuple[bytes, str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=240) as response:
        return response.read(), response.geturl(), response.headers.get("Content-Type", "")


def main() -> int:
    root = project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/"
        "round_181_haizi_903y_temperature_supplement_v1"
    )
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw_unopened"
    raw.mkdir()

    oa_payload, oa_final, oa_type = fetch(OA_URL)
    oa_path = raw / "PMC11554784_oa_response.xml"
    oa_path.write_bytes(oa_payload)
    oa = ET.fromstring(oa_payload)
    links = [item.attrib for item in oa.findall(".//link")]
    tgz_links = [item["href"] for item in links if item.get("format") == "tgz"]
    if len(tgz_links) != 1:
        raise RuntimeError(f"unexpected-tgz-link-count:{len(tgz_links)}")
    tgz_url = tgz_links[0].replace("ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/")
    package, package_final, package_type = fetch(tgz_url)
    package_path = raw / "PMC11554784.tar.gz"
    package_path.write_bytes(package)

    members: list[dict[str, object]] = []
    extracted: list[dict[str, object]] = []
    with tarfile.open(package_path, "r:gz") as archive:
        for member in archive.getmembers():
            members.append(
                {
                    "name": member.name,
                    "size_bytes": member.size,
                    "is_file": member.isfile(),
                }
            )
            if not member.isfile() or not member.name.lower().endswith((".xlsx", ".xls")):
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"supplement-extract-failed:{member.name}")
            payload = source.read()
            destination = raw / Path(member.name).name
            destination.write_bytes(payload)
            extracted.append(
                {
                    "archive_member": member.name,
                    "filename": destination.name,
                    "size_bytes": len(payload),
                    "sha256": sha256(destination),
                    "opened_or_parsed": False,
                }
            )
    (output / "archive_member_manifest.json").write_text(
        json.dumps(members, indent=2, ensure_ascii=True), encoding="ascii"
    )
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "oa_url": OA_URL,
        "oa_final_url": oa_final,
        "oa_content_type": oa_type,
        "oa_sha256": sha256(oa_path),
        "package_url": tgz_url,
        "package_final_url": package_final,
        "package_content_type": package_type,
        "package_size_bytes": len(package),
        "package_sha256": sha256(package_path),
        "archive_member_count": len(members),
        "extracted_spreadsheet_count": len(extracted),
        "extracted_spreadsheets": extracted,
        "all_spreadsheets_left_unopened": all(
            not item["opened_or_parsed"] for item in extracted
        ),
        "spreadsheet_value_accessed": False,
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "decision": (
            "advance_to_boundary_safe_structure_audit"
            if extracted
            else "close_no_spreadsheet_in_official_supplement"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 181 Haizi 903-Year Temperature Supplement Acquisition\n\n"
        f"- OA package members: `{len(members)}`.\n"
        f"- Extracted spreadsheet files: `{len(extracted)}`.\n"
        "- Spreadsheet values accessed: `false`.\n"
        "- Locked-period values accessed: `false`.\n\n"
        f"Decision: `{summary['decision']}`.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
