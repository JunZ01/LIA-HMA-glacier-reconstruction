from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVERS = Path(__file__).resolve().parent
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "round_224_central_himalaya_mxd2026_public_recovery_v4"
PDFINFO = Path(r"C:\Users\JUNZ\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdfinfo.exe")


def pdf_container(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    if not raw.startswith(b"%PDF-"):
        raise RuntimeError(f"Not a PDF container: {path}")
    process = subprocess.run(
        [str(PDFINFO), str(path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    info = {}
    for line in process.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    markers = len(re.findall(rb"/EmbeddedFiles\b", raw))
    return {
        "page_count": int(info["Pages"]),
        "embedded_files_catalog_marker_count": markers,
        "embedded_file_count": markers,
        "pdfinfo": info,
        "page_text_extracted": False,
    }


def main() -> None:
    source = DRIVERS / "run_round224_central_himalaya_mxd2026_public_recovery_v2.py"
    spec = importlib.util.spec_from_file_location("round224_v2_for_v4", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.RAW = OUT / "official_sources_unmodified"
    module.CONTRACT = OUT / "ROUND_CONTRACT.md"
    module.DRIVER = Path(__file__).resolve()
    module.URLS = dict(module.URLS)
    module.URLS["zenodo"] = "https://zenodo.org/api/records?q=Central%20Himalayan%20tree-ring%20density%20network&size=25"
    module.pdf_container = pdf_container
    module.main()

    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["version"] = 4
    summary["v1_stop"] = "missing_pypdf_before_network_access"
    summary["v2_stop"] = "zenodo_anonymous_page_size_100_rejected"
    summary["v3_stop"] = "pdfinfo_cmd_wrapper_lost_long_path"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    result_path = OUT / "FINAL_RESULT.md"
    result_path.write_text(result_path.read_text(encoding="utf-8").replace("# Round 224 V2 Result", "# Round 224 V4 Result", 1), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
