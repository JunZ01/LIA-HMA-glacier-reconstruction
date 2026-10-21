from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import requests


URL = "https://cran.r-project.org/src/contrib/dplR_1.7.9.tar.gz"
ALLOWED_PREFIXES = ("dplR/R/", "dplR/src/", "dplR/man/")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_251_dplr179_method_source_acquisition_v1"
    raw = output / "raw_unopened"
    vendor = output / "vendor_source"
    raw.mkdir(parents=True, exist_ok=False)
    vendor.mkdir(parents=True, exist_ok=False)
    response = requests.get(URL, timeout=120)
    response.raise_for_status()
    payload = response.content
    archive_path = raw / "dplR_1.7.9.tar.gz"
    archive_path.write_bytes(payload)
    extracted = []
    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            name = member.name.replace("\\", "/")
            if not member.isfile() or not name.startswith(ALLOWED_PREFIXES):
                continue
            relative = Path(name).relative_to("dplR")
            target = (vendor / relative).resolve()
            if vendor.resolve() not in target.parents:
                raise RuntimeError(f"unsafe-archive-member:{name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"archive-read-failed:{name}")
            data = source.read()
            target.write_bytes(data)
            extracted.append({"archive_member": name, "path": str(target), "bytes": len(data), "sha256": sha256(data)})
    required = ["R/ads.R", "R/powt.R"]
    missing = [name for name in required if not (vendor / name).exists()]
    if missing:
        raise RuntimeError("required-source-missing:" + ";".join(missing))
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "official_url": URL,
        "archive_path": str(archive_path),
        "archive_bytes": len(payload),
        "archive_sha256": sha256(payload),
        "extracted_file_count": len(extracted),
        "extracted_files": extracted,
        "climate_or_proxy_values_accessed": False,
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "binary_executed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
    }
    (output / "source_manifest.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 251 dplR 1.7.9 method-source acquisition\n\n"
        f"- Official source archive SHA256: `{sha256(payload)}`.\n"
        f"- Extracted R/src/man files: `{len(extracted)}`.\n",
        encoding="ascii",
    )
    print(json.dumps({"archive_sha256": sha256(payload), "extracted_file_count": len(extracted)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
