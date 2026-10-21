"""Read-only formal zero-MB rescore for four legacy cumulative-closure labels.

This driver deliberately leaves the historical run directories unchanged.  It
recomputes the registered maximum-absolute-per-step closure statistic from the
saved step ledgers and writes a versioned audit package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SYSTEM_ROOTS = {
    2328: "w228zero2328v1",
    2547: "w274zero2547mindt30v1",
    2580: "w273zero2580v1",
    5443: "w274zero5443mindt30v1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    rows: list[dict] = []
    for system_id, relative_root in SYSTEM_ROOTS.items():
        source_root = project_root / relative_root
        summary_path = source_root / "zero_mb_summary.json"
        steps_path = source_root / "zero_mb_steps.csv"
        checkpoints_path = source_root / "zero_mb_checkpoints.csv"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        steps = pd.read_csv(steps_path)
        initial_volume = float(summary["initial_volume_m3"])
        formal_limit = max(1e-5, 1e-14 * initial_volume)
        maximum = (
            float(steps["closure_residual_m3"].abs().max())
            if len(steps)
            else float("inf")
        )
        final = summary["final"]
        nonclosure_flags = [
            flag
            for flag in summary.get("review_flags", [])
            if flag != "budget-closure-residual"
        ]
        formal_flags = list(nonclosure_flags)
        if maximum > formal_limit:
            formal_flags.append("per-step-budget-closure-residual")
        if abs(float(final["volume_drift_fraction"])) > 0.001:
            formal_flags.append("volume-drift-over-0.1pct")
        for key, flag in (
            ("unresolved_negative_tributary_creation_m3", "unresolved-negative-tributary-creation"),
            ("negative_section_clip_creation_m3", "negative-section-clip-mass-creation"),
            ("root_domain_outflow_m3", "root-domain-outflow"),
        ):
            if abs(float(final.get(key, 0.0))) > formal_limit:
                formal_flags.append(flag)
        if float(final.get("root_terminal_thickness_m", 0.0)) > 1e-6:
            formal_flags.append("root-terminal-thickness")
        if not bool(final.get("finite_geometry", False)) or int(final.get("negative_section_count", 0)):
            formal_flags.append("invalid-geometry")
        formal_flags = sorted(set(formal_flags))
        formal_status = "zero-mb-pass" if not formal_flags else "zero-mb-fail"

        record = {
            "system_id": system_id,
            "source_root": str(source_root),
            "source_status": summary.get("status"),
            "source_review_flags": summary.get("review_flags", []),
            "formal_status": formal_status,
            "formal_review_flags": formal_flags,
            "closure_gate": "maximum absolute per-step residual",
            "maximum_absolute_step_closure_residual_m3": maximum,
            "per_step_closure_limit_m3": formal_limit,
            "closure_ratio": maximum / formal_limit,
            "cumulative_closure_residual_m3": float(final.get("closure_residual_m3", 0.0)),
            "completed_checkpoint_year": float(summary["completed_checkpoint_year"]),
            "volume_drift_fraction": float(final["volume_drift_fraction"]),
            "minimum_timestep_seconds": summary.get("minimum_timestep_seconds", 60.0),
            "source_summary_sha256": _sha256(summary_path),
            "source_steps_sha256": _sha256(steps_path),
            "source_checkpoints_sha256": _sha256(checkpoints_path),
            "source_files_unchanged": True,
            "interpretation": (
                "formal zero-MB gate passed; legacy cumulative-closure label superseded; "
                "not a historical-curve acceptance"
                if formal_status == "zero-mb-pass"
                else "formal zero-MB gate failed"
            ),
        }
        rows.append(record)
        (output_root / f"system_{system_id}_formal_rescore.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "formal_rescore_table.csv", index=False)
    manifest = {
        "experiment": "formal-zero-mb-legacy-four-rescore-v1",
        "criterion": "maximum absolute per-step closure residual",
        "systems": list(SYSTEM_ROOTS),
        "formal_pass_count": int((frame["formal_status"] == "zero-mb-pass").sum()),
        "formal_fail_count": int((frame["formal_status"] != "zero-mb-pass").sum()),
        "historical_sources_modified": False,
        "note": "A formal zero-MB pass releases only the next registered dynamics gate.",
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
