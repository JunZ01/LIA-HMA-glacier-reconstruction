from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUT = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2/"
    "round_202_zhu2011_researchgate_exact_download_v1"
)
PUBLICATION = "https://www.researchgate.net/publication/241075556_August_temperature_variability_in_the_southeastern_Tibetan_Plateau_since_AD_1385_inferred_from_tree_rings"
URL = "https://www.researchgate.net/publication/profile/Haifeng-Zhu-4/publication/355473374_Zhuetal2011P3xls/data/61721622c10b387664d2e84b/Zhuetal2011P3.xls?origin=publication_list"
OLE_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")
ZIP_MAGIC = b"PK\x03\x04"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite independent result: {OUT}")
    OUT.mkdir(parents=True)
    request = urllib.request.Request(
        URL,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
            "Referer": PUBLICATION,
            "Accept": "application/vnd.ms-excel,application/octet-stream;q=0.9,*/*;q=0.1",
        },
    )
    status = None
    headers: dict[str, str] = {}
    payload = b""
    error = None
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            status = getattr(response, "status", None)
            headers = {key.lower(): value for key, value in response.headers.items()}
            payload = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        headers = {key.lower(): value for key, value in exc.headers.items()}
        payload = exc.read()
        error = f"HTTPError {exc.code}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    is_ole = payload.startswith(OLE_MAGIC)
    is_zip = payload.startswith(ZIP_MAGIC)
    is_spreadsheet = is_ole or is_zip
    suffix = ".xls" if is_ole else ".xlsx" if is_zip else ".response"
    response_path = OUT / f"Zhuetal2011P3{suffix}"
    response_path.write_bytes(payload)
    sha256 = hashlib.sha256(payload).hexdigest()

    summary = {
        "round": 202,
        "source": "ResearchGate indexed supplementary resource Zhuetal2011P3.xls",
        "request_url": URL,
        "http_status": status,
        "content_type": headers.get("content-type"),
        "content_length_header": headers.get("content-length"),
        "response_size": len(payload),
        "response_sha256": sha256,
        "ole_xls_signature": is_ole,
        "zip_xlsx_signature": is_zip,
        "spreadsheet_recovered": is_spreadsheet,
        "error": error,
        "spreadsheet_cells_opened": False,
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
        "decision": "advance_unopened_file_structure_preflight" if is_spreadsheet else "exact_download_access_failed",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "publication_url": PUBLICATION,
                "download_url": URL,
                "response_path": str(response_path),
                "response_sha256": sha256,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = [
        "# Round 202 Zhu 2011 exact ResearchGate download",
        "",
        f"- HTTP status: `{status}`; error: `{error}`.",
        f"- Response bytes / SHA256: `{len(payload)}` / `{sha256}`.",
        f"- OLE XLS / ZIP XLSX signature: `{str(is_ole).lower()}` / `{str(is_zip).lower()}`.",
        f"- Decision: `{summary['decision']}`.",
        "- No spreadsheet cell or climate/proxy value was opened.",
    ]
    (OUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
