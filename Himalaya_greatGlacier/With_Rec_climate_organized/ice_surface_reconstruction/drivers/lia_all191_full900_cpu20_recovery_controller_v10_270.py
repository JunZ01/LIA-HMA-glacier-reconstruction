"""Recovery wrapper for the exact unfinished v10.267 CPU20 queue.

The numerical controller remains the hash-bound v10.267 controller.  This
wrapper validates the v10.270 recovery staging area and permits that controller
to start with exactly the 20 pre-staged, hash-validated member checkpoint
directories.  All other controller and chain behaviour is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

from . import lia_all191_full900_cpu20_controller_v10_267 as base


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def validate_recovery_staging(transaction_dir: Path, output_root: Path) -> None:
    manifest_path = transaction_dir / "transaction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("launch_authorized") or not manifest.get("production_authorized"):
        raise ValueError("recovery-transaction-not-authorized")
    runtime = manifest["runtime"]
    if runtime.get("recovery_wrapper_sha256") != sha256(Path(__file__)):
        raise ValueError("recovery-wrapper-hash-mismatch")
    if runtime.get("controller_sha256") != sha256(Path(base.__file__)):
        raise ValueError("base-controller-hash-mismatch")
    if canonical(Path(runtime["unified_short_root"])) != canonical(output_root):
        raise ValueError("recovery-output-root-mismatch")
    if (output_root / "r").exists():
        raise FileExistsError("recovery-runtime-root-collision")

    audit_path = transaction_dir / "recovery_staging_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True or int(audit.get("staged_members", -1)) != 20:
        raise ValueError("recovery-staging-audit-failed")
    if audit.get("file_inventory_sha256") != sha256(
        transaction_dir / "recovery_staging_files.csv"
    ):
        raise ValueError("recovery-staging-inventory-hash-mismatch")

    with (transaction_dir / "recovery_staging_files.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        files = list(csv.DictReader(handle))
    for row in files:
        destination = Path(row["destination_path"])
        try:
            destination.resolve().relative_to(output_root.resolve())
        except ValueError as exc:
            raise ValueError("staged-file-outside-output-root") from exc
        if not destination.is_file() or sha256(destination) != row["sha256"]:
            raise ValueError(f"staged-file-validation-failed:{destination}")

    expected_members = set(audit["member_keys"])
    actual_members = {
        path.name for path in (output_root / "m").iterdir() if path.is_dir()
    }
    if actual_members != expected_members:
        raise ValueError("staged-member-set-mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--transaction-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    known, _ = parser.parse_known_args()
    output_root = known.output_root.resolve()
    validate_recovery_staging(known.transaction_dir.resolve(), output_root)

    # The unchanged base controller rejects any pre-existing member root.  The
    # recovery contract intentionally requires exactly 20 pre-staged member
    # roots, already validated above.  Bypass only that single collision probe;
    # every subsequent Path.exists call retains its normal behaviour.
    real_exists = Path.exists
    staged_member_root = canonical(output_root / "m")
    bypassed = False

    def recovery_aware_exists(path: Path) -> bool:
        nonlocal bypassed
        if not bypassed and canonical(path) == staged_member_root:
            bypassed = True
            return False
        return real_exists(path)

    Path.exists = recovery_aware_exists
    try:
        result = base.main()
    finally:
        Path.exists = real_exists
    if not bypassed:
        raise RuntimeError("expected-staged-member-root-probe-not-observed")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
