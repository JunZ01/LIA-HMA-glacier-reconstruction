from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


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


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    archive_path = loop / "round_195_sigfree_toolchain_preflight_v2/raw_unopened/Windows7_Prgms-1.zip"
    output = loop / "round_195_sigfree_toolchain_preflight_v3"
    extracted = output / "sigfree_tool_unexecuted"
    extracted.mkdir(parents=True, exist_ok=False)
    archive_hash_before = sha256(archive_path)
    extracted_rows: list[dict[str, str | int]] = []
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"zip-crc-failure:{bad_member}")
        names = archive.namelist()
        selected = [
            name for name in names
            if "__MACOSX" not in name
            and ("sigfree" in name.lower() or "signal" in name.lower())
            and not name.endswith("/")
        ]
        for index, name in enumerate(selected, start=1):
            suffix = Path(PurePosixPath(name).name).suffix
            target = extracted / f"{index:03d}_{PurePosixPath(name).name}"
            with archive.open(name) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted_rows.append(
                {
                    "archive_member": name,
                    "local_path": str(target),
                    "size_bytes": target.stat().st_size,
                    "suffix": suffix,
                    "sha256": sha256(target),
                }
            )
    archive_hash_after = sha256(archive_path)
    if archive_hash_before != archive_hash_after:
        raise RuntimeError("archive-hash-changed")
    executable_count = sum(Path(str(row["local_path"])).suffix.lower() in {".exe", ".bat", ".cmd"} for row in extracted_rows)
    documentation_count = sum(Path(str(row["local_path"])).suffix.lower() in {".txt", ".md", ".pdf", ".doc", ".docx", ".rtf"} for row in extracted_rows)
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "upstream_archive_path": str(archive_path),
        "archive_sha256": archive_hash_after,
        "archive_size_bytes": archive_path.stat().st_size,
        "zip_crc_passed": True,
        "selected_sigfree_member_count": len(extracted_rows),
        "selected_executable_count": executable_count,
        "selected_documentation_count": documentation_count,
        "selected_members": extracted_rows,
        "binary_executed": False,
        "tree_ring_or_chronology_values_read": False,
        "locked_1921_1939_numeric_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "decision": (
            "inspect_sigfree_help_and_batch_interface"
            if executable_count and documentation_count
            else "close_sigfree_reproduction_due_missing_executable_or_documentation"
        ),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 195 v3 SigFree Toolchain Preflight\n\n"
        f"- Complete ZIP CRC passed: `true`.\n"
        f"- SigFree files extracted: `{len(extracted_rows)}`.\n"
        f"- Executables: `{executable_count}`; documentation: `{documentation_count}`.\n"
        "- Binary executed: `false`.\n\n"
        f"Decision: `{result['decision']}`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

