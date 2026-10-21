from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC11554784"
SUPPLEMENT_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41598-024-79096-6/MediaObjects/"
    "41598_2024_79096_MOESM1_ESM.xlsx"
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


def fetch(url: str) -> tuple[bytes, str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=240) as response:
        return response.read(), response.geturl(), response.headers.get("Content-Type", "")


def main() -> int:
    root = project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/"
        "round_181_haizi_903y_temperature_supplement_v2"
    )
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw_unopened"
    raw.mkdir()

    oa_payload, oa_final, oa_type = fetch(OA_URL)
    oa_path = raw / "PMC11554784_oa_response.xml"
    oa_path.write_bytes(oa_payload)
    supplement, supplement_final, supplement_type = fetch(SUPPLEMENT_URL)
    if not supplement.startswith(b"PK"):
        raise RuntimeError("official-supplement-not-zip-container")
    supplement_path = raw / "41598_2024_79096_MOESM1_ESM.xlsx"
    supplement_path.write_bytes(supplement)

    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "doi": "10.1038/s41598-024-79096-6",
        "pmcid": "PMC11554784",
        "oa_url": OA_URL,
        "oa_final_url": oa_final,
        "oa_content_type": oa_type,
        "oa_sha256": sha256(oa_path),
        "supplement_url": SUPPLEMENT_URL,
        "supplement_final_url": supplement_final,
        "supplement_content_type": supplement_type,
        "supplement_size_bytes": len(supplement),
        "supplement_sha256": sha256(supplement_path),
        "supplement_opened_or_parsed": False,
        "spreadsheet_value_accessed": False,
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "decision": "advance_to_boundary_safe_structure_audit",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 181 v2 Haizi 903-Year Supplement Acquisition\n\n"
        f"- Official XLSX bytes: `{len(supplement)}`.\n"
        f"- SHA256: `{summary['supplement_sha256']}`.\n"
        "- Spreadsheet opened or parsed: `false`.\n"
        "- Locked-period values accessed: `false`.\n\n"
        "Decision: `advance_to_boundary_safe_structure_audit`.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
