from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    output_default = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_localization_pilot_v1"
    )
    parser.add_argument("--output-dir", type=Path, default=output_default)
    args = parser.parse_args()
    output = args.output_dir.resolve()

    regional_decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    target = pd.read_csv(output / "target_3584_summary.csv").set_index("branch")
    baseline = target.loc["no_localization"]
    target_checks = []
    for branch in ["localization_3000km", "localization_4500km", "localization_6000km"]:
        row = target.loc[branch]
        checks = {
            "rolling30_correlation_improved": bool(
                row["median_rolling30_correlation"]
                > baseline["median_rolling30_correlation"]
            ),
            "rolling30_rmse_improved": bool(
                row["median_rolling30_rmse_c"] < baseline["median_rolling30_rmse_c"]
            ),
            "trend_error_improved": bool(
                row["median_trend_absolute_error_c_per_century"]
                < baseline["median_trend_absolute_error_c_per_century"]
            ),
            "early_late_error_improved": bool(
                row["median_early_late_absolute_error_c"]
                < baseline["median_early_late_absolute_error_c"]
            ),
        }
        target_checks.append(
            {
                "branch": branch,
                "checks": checks,
                "all_3584_low_frequency_metrics_improved": bool(all(checks.values())),
            }
        )

    final = {
        "experiment_id": regional_decision["experiment_id"],
        "baseline_replay_preflight": regional_decision["preflight"],
        "regional_proxy_screen_passes": regional_decision["accepted_branches"],
        "target_3584_checks": target_checks,
        "promoted_branch": None,
        "final_classification": "regional-local tradeoff; localization not promoted",
        "reason": (
            "Finite localization improved the regional aggregate and did not degrade the "
            "single-repeat held-out proxy metrics, but every tested radius worsened the "
            "3584 rolling-30 RMSE, trend error, and early-to-late change error relative "
            "to the unlocalized baseline. Correlation alone became less negative."
        ),
        "scope_limit": (
            "This is one proxy holdout repeat and one fixed prior experiment. It is a "
            "mechanism screen, not a final reconstruction comparison."
        ),
        "next_direction": (
            "Do not expand the localization radius sweep. Test a low-frequency/operator "
            "or instrumental-constraint design that directly addresses sign and timing, "
            "with the same proxy-plus-instrument evaluation split."
        ),
    }
    decision_path = output / "decision_v2.json"
    decision_path.write_text(json.dumps(final, indent=2), encoding="utf-8")

    report = f"""# Localization pilot final interpretation

## Result

The full-proxy unlocalized baseline reproduced the archived reconstruction
within `{regional_decision['preflight']['full_proxy_baseline_max_abs_difference_c']:.7f} degC`,
inside the declared `0.024 degC` replay bound.

Radii `4500 km` and `6000 km` passed the predeclared regional-plus-proxy screen.
They are **not promoted**. At the named 3584 diagnostic pixel, every finite
radius worsened rolling-30 RMSE, twentieth-century trend error, and the
1901-1930 to 1971-2000 change error relative to no localization. The rolling
correlation became less negative, but that isolated improvement is
insufficient.

Final classification: **regional-local tradeoff; localization not promoted**.

## Scope

This pilot used one archived holdout repeat and one fixed 500-member prior
experiment. It is a deterministic mechanism screen, not a final climate
reconstruction selection. No glacier simulation or glacier curve entered any
metric.

## Next direction

Do not broaden the localization-radius sweep. The next small experiment must
target low-frequency sign and timing through a declared operator or
instrumental-constraint design while retaining the same separate proxy and
five-product instrumental diagnostics.
"""
    report_path = output / "FINAL_RESULT.md"
    report_path.write_text(report, encoding="utf-8")

    manifest = {
        "decision_v2_sha256": _sha256(decision_path),
        "final_result_sha256": _sha256(report_path),
    }
    (output / "finalization_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
