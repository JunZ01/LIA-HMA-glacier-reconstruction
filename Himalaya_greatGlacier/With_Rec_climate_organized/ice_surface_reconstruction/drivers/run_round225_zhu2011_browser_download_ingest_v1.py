from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUT = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2/"
    "round_225_zhu2011_browser_download_ingest_v1"
)
DEFAULT_DOWNLOADS = Path.home() / "Downloads"
ALLOWED_NAMES = {"zhuetal2011p3.xls", "zhuetal2011p3.xlsx"}
OLE_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")
ZIP_MAGIC = b"PK\x03\x04"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_candidate(explicit: Path | None) -> Path:
    if explicit is not None:
        candidates = [explicit]
    else:
        candidates = [
            path
            for path in DEFAULT_DOWNLOADS.iterdir()
            if path.is_file() and path.name.lower() in ALLOWED_NAMES
        ]
    candidates = [path.resolve() for path in candidates if path.exists() and path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            "No completed Zhuetal2011P3.xls/xlsx download found. "
            f"Checked {DEFAULT_DOWNLOADS}."
        )
    invalid_names = [path for path in candidates if path.name.lower() not in ALLOWED_NAMES]
    if invalid_names:
        raise ValueError(f"Refusing unexpected filename: {invalid_names[0].name}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", type=Path)
    args = parser.parse_args()

    source = locate_candidate(args.source)
    header = source.read_bytes()[:8]
    is_ole = header.startswith(OLE_MAGIC)
    is_zip = header.startswith(ZIP_MAGIC)
    if not (is_ole or is_zip):
        raise ValueError(
            f"Refusing non-spreadsheet response: {source} "
            f"(first bytes {header.hex()})"
        )
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite independent result: {OUT}")

    source_hash_before = sha256(source)
    source_size = source.stat().st_size
    OUT.mkdir(parents=True)
    admitted = OUT / ("Zhuetal2011P3.xls" if is_ole else "Zhuetal2011P3.xlsx")
    shutil.copyfile(source, admitted)
    source_hash_after = sha256(source)
    admitted_hash = sha256(admitted)
    if source_hash_before != source_hash_after or source_hash_before != admitted_hash:
        raise RuntimeError("Source changed or byte-exact copy verification failed")

    summary = {
        "round": 225,
        "source": str(source),
        "admitted_copy": str(admitted),
        "size_bytes": source_size,
        "sha256": admitted_hash,
        "ole_xls_signature": is_ole,
        "zip_xlsx_signature": is_zip,
        "byte_exact_copy": True,
        "spreadsheet_cells_opened": False,
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
        "decision": "advance_separate_locked_safe_structure_preflight",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "browser_download": str(source),
                "admitted_copy": str(admitted),
                "size_bytes": source_size,
                "sha256": admitted_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Round 225 Zhu 2011 browser-download ingest",
                "",
                f"- Admitted file: `{admitted.name}` (`{source_size}` bytes).",
                f"- SHA256: `{admitted_hash}`.",
                f"- OLE XLS / ZIP XLSX signature: `{str(is_ole).lower()}` / `{str(is_zip).lower()}`.",
                "- The copy is byte-exact; no workbook cell or climate/proxy value was opened.",
                "- Decision: `advance_separate_locked_safe_structure_preflight`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
