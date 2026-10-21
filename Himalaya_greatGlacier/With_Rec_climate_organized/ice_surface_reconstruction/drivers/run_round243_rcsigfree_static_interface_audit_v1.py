from __future__ import annotations

import hashlib
import json
import re
import struct
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


EXPECTED_EXE_SHA256 = "e8a380642ae94a72c136abfc344392f74215acdcefb3368b90d957dc0960485e"
EXPECTED_PDF_SHA256 = "533499f293f64f4952d76593725c474efb3999a72fee9d805e06f6150a8eba38"
KEYWORDS = (
    "age", "alpha", "batch", "chron", "command", "control", "curve",
    "data", "detrend", "file", "input", "iteration", "menu", "output",
    "power", "prompt", "raw", "rcs", "rwl", "signal", "spline", "tucson",
    "variance", ".crn", ".raw", ".rwl", ".txt",
)


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


def ascii_strings(data: bytes) -> list[str]:
    return [match.decode("ascii") for match in re.findall(rb"[\x20-\x7e]{4,}", data)]


def utf16le_strings(data: bytes) -> list[str]:
    pattern = re.compile(rb"(?:[\x20-\x7e]\x00){4,}")
    return [match.decode("utf-16le") for match in pattern.findall(data)]


def pe_metadata(data: bytes) -> dict[str, int | str]:
    if data[:2] != b"MZ":
        raise RuntimeError("not-pe-mz")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_offset:pe_offset + 4] != b"PE\x00\x00":
        raise RuntimeError("not-pe-signature")
    machine, sections, timestamp, _, _, optional_size, characteristics = struct.unpack_from(
        "<HHIIIHH", data, pe_offset + 4
    )
    optional_offset = pe_offset + 24
    optional_magic = struct.unpack_from("<H", data, optional_offset)[0]
    subsystem = struct.unpack_from("<H", data, optional_offset + 68)[0]
    return {
        "pe_offset": pe_offset,
        "machine_hex": f"0x{machine:04x}",
        "section_count": sections,
        "coff_timestamp": timestamp,
        "optional_header_size": optional_size,
        "optional_magic_hex": f"0x{optional_magic:04x}",
        "subsystem": subsystem,
        "characteristics_hex": f"0x{characteristics:04x}",
    }


def main() -> int:
    root = project_root()
    drivers = Path(__file__).resolve().parent
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    upstream = loop / "round_195_sigfree_toolchain_preflight_v3/sigfree_tool_unexecuted"
    output = loop / "round_243_rcsigfree_static_interface_audit_v1"
    output.mkdir(parents=True, exist_ok=False)
    text_dir = output / "method_paper_text"
    text_dir.mkdir()
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": drivers / "round243_rcsigfree_static_interface_audit_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "executable": upstream / "001_RCSigFree_2019_MRWE.exe",
        "method_pdf": upstream / "002_Melvin2008.pdf",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: sha256(path) for name, path in paths.items()}
    if before["executable"] != EXPECTED_EXE_SHA256 or before["method_pdf"] != EXPECTED_PDF_SHA256:
        raise RuntimeError("round195-source-hash-changed")

    executable_bytes = paths["executable"].read_bytes()
    ascii_inventory = ascii_strings(executable_bytes)
    utf16_inventory = utf16le_strings(executable_bytes)
    inventory = [f"ASCII\t{value}" for value in ascii_inventory]
    inventory.extend(f"UTF16LE\t{value}" for value in utf16_inventory)
    (output / "all_printable_strings.txt").write_text("\n".join(inventory) + "\n", encoding="utf-8")
    filtered = [
        item for item in inventory
        if any(keyword in item.lower() for keyword in KEYWORDS)
    ]
    (output / "interface_keyword_strings.txt").write_text("\n".join(filtered) + "\n", encoding="utf-8")

    reader = PdfReader(paths["method_pdf"])
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_path = text_dir / f"page_{page_number:03d}.txt"
        page_path.write_text(text, encoding="utf-8")
        pages.append({"page": page_number, "characters": len(text), "path": str(page_path)})

    after = {name: sha256(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("source-hash-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "pe_metadata": pe_metadata(executable_bytes),
        "ascii_string_count": len(ascii_inventory),
        "utf16le_string_count": len(utf16_inventory),
        "keyword_filtered_string_count": len(filtered),
        "method_pdf_page_count": len(reader.pages),
        "method_pdf_pages": pages,
        "binary_executed_or_loaded": False,
        "tree_ring_or_climate_values_accessed": False,
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "source_hashes_unchanged": True,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 243 RCSigFree static interface audit\n\n"
        f"- ASCII strings: `{len(ascii_inventory)}`.\n"
        f"- UTF-16LE strings: `{len(utf16_inventory)}`.\n"
        f"- Interface-keyword strings: `{len(filtered)}`.\n"
        f"- Method-paper pages extracted: `{len(reader.pages)}`.\n"
        "- Binary executed or loaded: `false`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
