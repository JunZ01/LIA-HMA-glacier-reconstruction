"""Build a compact, provenance-preserving validation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def clean(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def run(output_dir: Path) -> dict:
    audit = json.loads((output_dir / "public_data_audit.json").read_text(encoding="utf-8"))
    synthetic = json.loads((output_dir / "synthetic_validation.json").read_text(encoding="utf-8"))
    modern = pd.read_csv(output_dir / "modern_profile_validation_summary.csv")
    itmix = pd.read_csv(output_dir / "itmix_method_summary.csv")

    rows: list[dict] = []
    for result in synthetic["profile_results"]:
        rows.append(
            {
                "validation_level": "A-synthetic",
                "comparison": result["method"],
                "case": result["case_id"],
                "surface_rmse_m": result["surface_rmse_m"],
                "thickness_rmse_m": result["thickness_rmse_m"],
                "status": result["status"],
            }
        )
    for result in modern.to_dict("records"):
        rows.append(
            {
                "validation_level": "B-modern-pseudo-paleo",
                "comparison": f"{result['bed_scenario']}:{result['surface_method']}",
                "case": "pilot+GlaThiDa-summary targets",
                "surface_rmse_m": result["median_surface_rmse_m"],
                "thickness_rmse_m": result["median_thickness_rmse_m"],
                "status": "summary",
            }
        )
    for result in itmix.to_dict("records"):
        rows.append(
            {
                "validation_level": "C-ITMIX-external-bed-benchmark",
                "comparison": result["method"],
                "case": "20 public glaciers",
                "surface_rmse_m": None,
                "thickness_rmse_m": result["median_thickness_rmse_m"],
                "status": "inverse-bed-context-only",
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "validation_summary.csv", index=False)

    report = {
        "validation_id": audit["validation_id"],
        "method_version": "boundary-perfect-plasticity-harmonic-v0",
        "allowed_use": audit["allowed_use"],
        "source_registry": audit,
        "synthetic": {
            "profile_case_count": synthetic["profile_case_count"],
            "all_exact_cases_success": synthetic["all_exact_cases_success"],
            "harmonic_extension": synthetic["harmonic_extension"],
        },
        "modern_pseudo_paleo": modern.to_dict("records"),
        "itmix_external_benchmark": {
            "validated_method_glacier_rows": int(len(pd.read_csv(output_dir / "itmix_point_validation.csv"))),
            "method_summary": itmix.to_dict("records"),
            "interpretation": "ITMIX validates bed/thickness inversion context; it is not direct truth for the forward LIA surface operator.",
        },
        "direct_observation_limit": "GlaThiDa target points are absent; the two matched glaciers provide summary maximum-thickness checks only.",
        "independence_rule": "Farinotti and Millan are structural scenarios, not independent likelihood members.",
        "scale_up_gate": {
            "status": "not-yet-passed",
            "reason": "A point/profile-level direct observation benchmark is still missing inside the target HMA polygons.",
            "allowed_next_step": "Use the validated operator for a documented pilot and acquire/curate one direct HMA profile before authorizing the 191-system production run.",
        },
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8"
    )

    lines = [
        "# Paleo-surface validation summary",
        "",
        f"Validation ID: `{report['validation_id']}`",
        "",
        "This report is validation-only. It does not update Stage 0-3 parameters.",
        "",
        "## Evidence",
        "",
        "| Level | Comparison | Median surface RMSE (m) | Median thickness RMSE (m) |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        surface = "-" if row["surface_rmse_m"] is None else f"{row['surface_rmse_m']:.2f}"
        thickness = "-" if row["thickness_rmse_m"] is None else f"{row['thickness_rmse_m']:.2f}"
        lines.append(f"| {row['validation_level']} | {row['comparison']} | {surface} | {thickness} |")
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- ITMIX is an inverse bed/thickness benchmark and is retained as external structural context, not as forward-surface truth.",
            "- GlaThiDa has two target summary matches but zero target point rows; maximum-thickness comparisons are not a full profile validation.",
            "- The Alps LIA raster/outlines are an independent published reconstruction, not direct ancient-surface observations.",
            "- Failed physical candidates remain in the ledger; hard caps and fallback values are not used by the new solver.",
            "- Scale-up to 191 core systems remains gated until a point/profile-level direct HMA thickness benchmark is available.",
        ]
    )
    (output_dir / "validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.output_dir.resolve())
    print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()

