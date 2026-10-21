"""Apply the verified 2026-08-13 pending amendment to LIA_PROJECT_FACTS.md.

This one-purpose repair exists because the Windows apply_patch sandbox helper
cannot open the existing root facts file. It is idempotent, requires the exact
old header, saves a byte-for-byte backup, and copies only the two verified
sections from the pending amendment record.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


OLD_HEADER = "Last updated: 2026-08-11 (2266 delta-boundary preflight completed)"
NEW_HEADER = "Last updated: 2026-08-13 (three-system A-window branch mechanism completed)"
SECTION_MARKER = "## 14. RGI-only Layer-1 mechanism diagnosis V2 (2026-08-13)"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    project_root = Path(__file__).resolve().parents[4]
    facts = project_root / "LIA_PROJECT_FACTS.md"
    pending = (
        project_root
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "lia_191_a_window_branch_mechanism_v1"
        / "FACTS_AMENDMENT_PENDING.md"
    )
    backup = facts.with_name("LIA_PROJECT_FACTS.pre_20260813_amendment.md")

    original_bytes = facts.read_bytes()
    original = original_bytes.decode("utf-8")
    if SECTION_MARKER in original:
        print(f"already-applied sha256={sha256(original_bytes)}")
        return 0
    if OLD_HEADER not in original:
        raise RuntimeError("expected-old-header-missing")
    if backup.exists() and backup.read_bytes() != original_bytes:
        raise RuntimeError("existing-backup-does-not-match-current-facts")

    pending_text = pending.read_text(encoding="utf-8")
    source_marker = "## RGI-only Layer-1 mechanism diagnosis V2"
    if source_marker not in pending_text:
        raise RuntimeError("pending-amendment-section-missing")
    amendment = pending_text[pending_text.index(source_marker) :].strip()
    amendment = amendment.replace(
        "## RGI-only Layer-1 mechanism diagnosis V2",
        SECTION_MARKER,
        1,
    ).replace(
        "## Three-system A-window branch mechanism",
        "## 15. Three-system A-window branch mechanism experiment (2026-08-13)",
        1,
    )

    updated = original.replace(OLD_HEADER, NEW_HEADER, 1).rstrip() + "\n\n" + amendment + "\n"
    if updated.count(SECTION_MARKER) != 1 or NEW_HEADER not in updated:
        raise RuntimeError("updated-facts-validation-failed")

    if not backup.exists():
        backup.write_bytes(original_bytes)
    temporary = facts.with_suffix(".md.updated.tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(facts)
    print(
        f"updated old_sha256={sha256(original_bytes)} "
        f"new_sha256={sha256(facts.read_bytes())} backup={backup}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
