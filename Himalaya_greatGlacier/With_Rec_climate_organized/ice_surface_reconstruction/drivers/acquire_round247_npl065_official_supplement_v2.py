from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


PII = "S1125786520301387"
BASE = "https://ars.els-cdn.com/content/image/"
EXTENSIONS = ("docx", "xlsx", "pdf", "zip", "csv", "txt")
INDICES = range(1, 6)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_247_npl065_official_supplement_acquisition_v2"
    raw = output / "raw_unopened"
    raw.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 climate-method-audit/1.0"})
    records = []
    successes = []
    for index in INDICES:
        for extension in EXTENSIONS:
            name = f"1-s2.0-{PII}-mmc{index}.{extension}"
            url = BASE + name
            try:
                response = session.get(url, timeout=45, allow_redirects=True)
                payload = response.content
                record = {
                    "url": url,
                    "final_url": response.url,
                    "status": response.status_code,
                    "content_type": response.headers.get("Content-Type", ""),
                    "content_disposition": response.headers.get("Content-Disposition", ""),
                    "byte_count": len(payload),
                }
                if response.status_code == 200 and len(payload) > 0:
                    path = raw / name
                    path.write_bytes(payload)
                    record["saved_path"] = str(path)
                    record["sha256"] = sha256_bytes(payload)
                    successes.append(record)
                records.append(record)
            except requests.RequestException as error:
                records.append({"url": url, "error": type(error).__name__, "detail": str(error)})
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "pii": PII,
        "candidate_count": len(records),
        "successful_attachment_count": len(successes),
        "successful_attachments": successes,
        "all_attempts": records,
        "payloads_parsed": False,
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
    }
    (output / "acquisition_manifest.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 247 NPL065 official supplement acquisition\n\n"
        f"- Candidate URLs tested: `{len(records)}`.\n"
        f"- Official attachments acquired: `{len(successes)}`.\n"
        "- Payloads parsed: `false`.\n",
        encoding="ascii",
    )
    print(json.dumps({"successful_attachment_count": len(successes), "successful_attachments": successes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
