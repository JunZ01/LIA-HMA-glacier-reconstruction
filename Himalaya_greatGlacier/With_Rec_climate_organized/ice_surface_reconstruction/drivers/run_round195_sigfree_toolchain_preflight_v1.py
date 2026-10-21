from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


URL = "https://www.geog.cam.ac.uk/wp-content/uploads/Windows7_Prgms-1.zip"


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
    output = loop / "round_195_sigfree_toolchain_preflight_v1"
    raw = output / "raw_unopened"
    extracted = output / "tool_snapshot_unexecuted"
    raw.mkdir(parents=True, exist_ok=False)
    extracted.mkdir()
    archive_path = raw / "Windows7_Prgms-1.zip"
    request = urllib.request.Request(URL, headers={"User-Agent": "LIA-HMA-toolchain-audit/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if len(payload) < 100_000:
        raise RuntimeError("unexpectedly-small-tool-suite")
    archive_path.write_bytes(payload)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        for member in members:
            target = (extracted / member.filename).resolve()
            if extracted.resolve() not in target.parents and target != extracted.resolve():
                raise RuntimeError(f"zip-path-traversal:{member.filename}")
        archive.extractall(extracted)
    member_names = [member.filename for member in members]
    sigfree_names = [name for name in member_names if "sigfree" in name.lower() or "signal" in name.lower()]
    documentation_names = [
        name for name in member_names
        if Path(name).suffix.lower() in {".txt", ".md", ".pdf", ".doc", ".docx", ".rtf"}
    ]
    example_names = [
        name for name in member_names
        if "example" in name.lower() or "sample" in name.lower() or Path(name).suffix.lower() in {".inp", ".ctl"}
    ]
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "official_url": URL,
        "archive_path": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_member_count": len(member_names),
        "sigfree_member_names": sigfree_names,
        "documentation_member_names": documentation_names,
        "example_or_control_member_names": example_names,
        "binary_executed": False,
        "tree_ring_or_chronology_values_read": False,
        "locked_1921_1939_numeric_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "decision": (
            "inspect_sigfree_documentation_and_batch_controls"
            if sigfree_names
            else "close_official_sigfree_toolchain_not_present"
        ),
    }
    (output / "archive_members.json").write_text(json.dumps({"members": member_names}, indent=2), encoding="ascii")
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 195 SigFree Toolchain Preflight\n\n"
        f"- Official archive SHA256: `{result['archive_sha256']}`.\n"
        f"- Archive members: `{len(member_names)}`.\n"
        f"- SigFree-related members: `{len(sigfree_names)}`.\n"
        f"- Documentation members: `{len(documentation_names)}`.\n"
        "- Binary executed: `false`.\n\n"
        f"Decision: `{result['decision']}`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

