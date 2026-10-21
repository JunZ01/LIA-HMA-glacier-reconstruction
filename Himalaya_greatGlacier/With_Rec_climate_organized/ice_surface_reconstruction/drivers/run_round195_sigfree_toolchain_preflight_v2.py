from __future__ import annotations

import hashlib
import json
import re
import time
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


def download_resumable(url: str, destination: Path) -> dict[str, int]:
    part = destination.with_suffix(destination.suffix + ".part")
    expected_total: int | None = None
    request_count = 0
    for attempt in range(1, 9):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "LIA-HMA-toolchain-audit/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        request_count += 1
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = int(getattr(response, "status", 200))
                content_range = response.headers.get("Content-Range", "")
                if status == 206:
                    match = re.search(r"/(\d+)$", content_range)
                    if not match:
                        raise RuntimeError(f"missing-content-range-total:{content_range}")
                    expected_total = int(match.group(1))
                    mode = "ab"
                else:
                    expected_total = int(response.headers.get("Content-Length", "0") or 0) or None
                    mode = "wb"
                with part.open(mode) as stream:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
        except Exception:
            if attempt == 8:
                raise
            time.sleep(min(2 * attempt, 10))
            continue
        size = part.stat().st_size
        if expected_total is not None and size >= expected_total:
            part.replace(destination)
            return {"request_count": request_count, "expected_size_bytes": expected_total}
        time.sleep(min(2 * attempt, 10))
    raise RuntimeError("download-retries-exhausted")


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_195_sigfree_toolchain_preflight_v2"
    raw = output / "raw_unopened"
    extracted = output / "tool_snapshot_unexecuted"
    raw.mkdir(parents=True, exist_ok=False)
    extracted.mkdir()
    archive_path = raw / "Windows7_Prgms-1.zip"
    transfer = download_resumable(URL, archive_path)
    if archive_path.stat().st_size < 100_000:
        raise RuntimeError("unexpectedly-small-tool-suite")
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"zip-crc-failure:{bad_member}")
        members = archive.infolist()
        for member in members:
            target = (extracted / member.filename).resolve()
            if extracted.resolve() not in target.parents and target != extracted.resolve():
                raise RuntimeError(f"zip-path-traversal:{member.filename}")
        archive.extractall(extracted)
    names = [member.filename for member in members]
    sigfree = [name for name in names if "sigfree" in name.lower() or "signal" in name.lower()]
    docs = [name for name in names if Path(name).suffix.lower() in {".txt", ".md", ".pdf", ".doc", ".docx", ".rtf"}]
    examples = [name for name in names if "example" in name.lower() or "sample" in name.lower() or Path(name).suffix.lower() in {".inp", ".ctl"}]
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "official_url": URL,
        "archive_sha256": sha256(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "download_request_count": transfer["request_count"],
        "expected_size_bytes": transfer["expected_size_bytes"],
        "zip_crc_passed": True,
        "archive_member_count": len(names),
        "sigfree_member_names": sigfree,
        "documentation_member_names": docs,
        "example_or_control_member_names": examples,
        "binary_executed": False,
        "tree_ring_or_chronology_values_read": False,
        "locked_1921_1939_numeric_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "decision": "inspect_sigfree_documentation_and_batch_controls" if sigfree else "close_official_sigfree_toolchain_not_present",
    }
    (output / "archive_members.json").write_text(json.dumps({"members": names}, indent=2), encoding="ascii")
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 195 v2 SigFree Toolchain Preflight\n\n"
        f"- Archive SHA256: `{result['archive_sha256']}`.\n"
        f"- Complete ZIP CRC passed: `true`; members: `{len(names)}`.\n"
        f"- SigFree-related members: `{len(sigfree)}`.\n"
        f"- Documentation members: `{len(docs)}`.\n"
        "- Binary executed: `false`.\n\n"
        f"Decision: `{result['decision']}`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

