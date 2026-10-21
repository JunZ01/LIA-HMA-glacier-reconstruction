"""Recount and freeze terminal evidence for the v10.270 CPU recovery."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
TRANSACTION = BASE / "lia_191_all191_fullgrid_recovery_v10_270_transaction_v1"
OUTPUT = ROOT / "w270"
EVIDENCE = BASE / "lia_191_all191_fullgrid_recovery_v10_270_terminal_v1"
EXPECTED_YEARS = list(range(0, 901, 30))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> int:
    required = (
        TRANSACTION / "transaction_manifest.json",
        TRANSACTION / "production_matrix.csv",
        OUTPUT / "r/p.json",
        OUTPUT / "r/status.json",
        OUTPUT / "r/summary.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if EVIDENCE.exists():
        raise FileExistsError(EVIDENCE)

    manifest = json.loads((TRANSACTION / "transaction_manifest.json").read_text(encoding="utf-8"))
    controller = json.loads((OUTPUT / "r/summary.json").read_text(encoding="utf-8"))
    status = json.loads((OUTPUT / "r/status.json").read_text(encoding="utf-8"))
    parent = json.loads((OUTPUT / "r/p.json").read_text(encoding="utf-8"))
    if not (
        controller.get("status") == "completed"
        and controller.get("batch_completed") is True
        and int(controller.get("requested_cells", -1)) == 884
        and int(controller.get("dispatched_cells", -1)) == 884
        and int(controller.get("completed_cells", -1)) == 884
        and int(status.get("completed", -1)) == 884
        and int(status.get("pending", -1)) == 0
    ):
        raise ValueError("controller-not-terminal-complete")
    if psutil.pid_exists(int(parent["pid"])):
        raise ValueError("parent-still-alive")

    valid = []
    failures = []
    result_files = sorted(OUTPUT.glob("m/*/c/result.json"))
    if len(result_files) != 884:
        raise ValueError(f"result-file-count:{len(result_files)}")
    for path in result_files:
        result = json.loads(path.read_text(encoding="utf-8"))
        record = {
            "member_key": path.parents[1].name,
            "fingerprint": result.get("fingerprint"),
            "status": result.get("status"),
            "result_path": str(path.resolve()),
            "result_sha256": sha256(path),
        }
        if (
            result.get("terminal") is True
            and result.get("all900_completed") is True
            and result.get("checkpoint_contract_pass") is True
            and result.get("current_year") == 900
            and result.get("checkpoint_years_actual") == EXPECTED_YEARS
            and not bool(result.get("technical_failure"))
        ):
            valid.append((record, result))
        elif bool(result.get("technical_failure")):
            failures.append(
                {
                    **record,
                    "error": result.get("error"),
                    "traceback": result.get("traceback"),
                }
            )
        else:
            raise ValueError(f"unclassified-result:{path}")
    if len(valid) != 883 or len(failures) != 1:
        raise ValueError(f"terminal-classification:{len(valid)}:{len(failures)}")
    failure = failures[0]
    if failure["member_key"] != "s2679" or "KeyError: 450" not in str(failure["error"]):
        raise ValueError(f"unexpected-failure:{failure}")

    stable = sum(not bool(result.get("scientific_nonconvergence")) for _, result in valid)
    nonconverged = sum(bool(result.get("scientific_nonconvergence")) for _, result in valid)
    target_pass = sum(bool(result.get("spinup_target_pass")) for _, result in valid)
    if stable + nonconverged != 883:
        raise ValueError("scientific-classification-count-failure")

    EVIDENCE.mkdir(parents=True, exist_ok=False)
    atomic_csv(EVIDENCE / "technical_failures.csv", failures)
    summary = {
        "schema": "lia-191-all191-fullgrid-recovery-v10.270-terminal-v1",
        "status": "terminal-controller-complete-one-isolated-technical-failure",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "transaction": {
            "path": str((TRANSACTION / "transaction_manifest.json").relative_to(ROOT)),
            "sha256": sha256(TRANSACTION / "transaction_manifest.json"),
        },
        "output_root": str(OUTPUT.relative_to(ROOT)),
        "controller": {
            "parent_pid": int(parent["pid"]),
            "parent_alive": False,
            "started_at": controller["started_at_utc"],
            "finished_at": controller["finished_at_utc"],
            "requested_cells": 884,
            "dispatched_cells": 884,
            "terminal_records": 884,
            "batch_completed": True,
            "pause_reason": controller.get("pause_reason"),
            "stop_reason": controller.get("stop_reason"),
            "summary_path": str((OUTPUT / "r/summary.json").relative_to(ROOT)),
            "summary_sha256": sha256(OUTPUT / "r/summary.json"),
        },
        "recovery_results": {
            "valid_year900_checkpoint_contract_cells": 883,
            "stable_cells": stable,
            "scientific_nonconvergence_cells": nonconverged,
            "spinup_target_pass_cells": target_pass,
            "technical_failure_cells": 1,
        },
        "combined_full_grid": {
            "target_cells": 4011,
            "valid_cells": 3127 + len(valid),
            "stable_cells": 1144 + stable,
            "scientific_nonconvergence_cells": 1983 + nonconverged,
            "spinup_target_pass_cells": 166 + target_pass,
            "unresolved_technical_cells": 1,
        },
        "technical_failure": {
            "member_key": failure["member_key"],
            "fingerprint": failure["fingerprint"],
            "lia_unique_id": 3850,
            "rgi_id": "RGI60-14.15895",
            "delta_t_eq_degC": 0.8,
            "source_checkpoint_year": 450,
            "failure": "diagnostic history missing year 450 after a checkpoint was committed before the interrupted history write",
            "classification": "technical-not-scientific",
            "automatic_retry_authorized": False,
        },
        "accepted_count_effect": 0,
        "independent_gpu_branches_untouched": True,
        "next_legal_action": "explicit user decision whether to authorize one exact-cell technical retry using the validated year-450 checkpoint state and its checkpoint-embedded diagnostic state",
    }
    atomic_json(EVIDENCE / "summary.json", summary)
    atomic_json(
        EVIDENCE / "artifact_hashes.json",
        {path.name: sha256(path) for path in sorted(EVIDENCE.iterdir()) if path.is_file()},
    )
    print(json.dumps({"evidence": str(EVIDENCE), **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
