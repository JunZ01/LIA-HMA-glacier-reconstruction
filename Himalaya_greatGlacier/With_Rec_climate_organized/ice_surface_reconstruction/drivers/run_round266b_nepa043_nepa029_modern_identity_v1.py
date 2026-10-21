from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from safe_rwl_year_first_v1 import parse_rwl_allowed_years


MIN_YEAR = 1940
MAX_YEAR = 2000
LOCKED = ((1921, 1939),)
EXACT_MIN_OVERLAP = 20
EXACT_ABS_CORRELATION = 0.999999
EXACT_AFFINE_RMSE_FRACTION = 1e-6
NEAR_MIN_OVERLAP = 30
NEAR_ABS_CORRELATION = 0.995
NEAR_STANDARDIZED_RMSE = 0.15


def fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def pair_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float | int | bool]:
    valid = np.isfinite(left) & np.isfinite(right)
    count = int(valid.sum())
    if count < 3:
        return {
            "overlap_count": count,
            "correlation": np.nan,
            "absolute_correlation": np.nan,
            "affine_rmse_fraction": np.nan,
            "standardized_rmse_best_sign": np.nan,
            "exact_affine_alias": False,
            "near_exact_standardized_sequence": False,
        }

    x = left[valid].astype(float)
    y = right[valid].astype(float)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        correlation = np.nan
        affine_fraction = np.nan
        standardized_rmse = np.nan
    else:
        correlation = float(np.corrcoef(x, y)[0, 1])
        design = np.column_stack((np.ones(count), y))
        fitted = design @ np.linalg.lstsq(design, x, rcond=None)[0]
        affine_fraction = float(np.sqrt(np.mean((x - fitted) ** 2)) / sx)
        zx = (x - np.mean(x)) / sx
        zy = (y - np.mean(y)) / sy
        standardized_rmse = float(
            min(np.sqrt(np.mean((zx - zy) ** 2)), np.sqrt(np.mean((zx + zy) ** 2)))
        )

    absolute = abs(correlation) if np.isfinite(correlation) else np.nan
    exact = bool(
        count >= EXACT_MIN_OVERLAP
        and absolute >= EXACT_ABS_CORRELATION
        and affine_fraction <= EXACT_AFFINE_RMSE_FRACTION
    )
    near = bool(
        count >= NEAR_MIN_OVERLAP
        and absolute >= NEAR_ABS_CORRELATION
        and standardized_rmse <= NEAR_STANDARDIZED_RMSE
    )
    return {
        "overlap_count": count,
        "correlation": correlation,
        "absolute_correlation": absolute,
        "affine_rmse_fraction": affine_fraction,
        "standardized_rmse_best_sign": standardized_rmse,
        "exact_affine_alias": exact,
        "near_exact_standardized_sequence": near,
    }


def site_chronology(frame: pd.DataFrame) -> pd.Series:
    standardized = frame.copy()
    for column in standardized.columns:
        values = standardized[column].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if int(valid.sum()) < 3 or float(np.std(values[valid])) <= 1e-12:
            standardized[column] = np.nan
        else:
            standardized.loc[valid, column] = (
                values[valid] - float(np.mean(values[valid]))
            ) / float(np.std(values[valid]))
    return standardized.mean(axis=1, skipna=True).rename("site_standardized_mean")


def run_audit(
    nepa043_path: Path,
    nepa029_path: Path,
    output: Path,
    expected_nepa043_sha256: str,
    expected_nepa029_sha256: str,
) -> dict[str, object]:
    nepa043_path = Path(nepa043_path)
    nepa029_path = Path(nepa029_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)

    before = {
        "nepa043": fingerprint(nepa043_path),
        "nepa029": fingerprint(nepa029_path),
    }
    if before["nepa043"]["sha256"].lower() != expected_nepa043_sha256.lower():
        raise RuntimeError("nepa043-hash-conflict")
    if before["nepa029"]["sha256"].lower() != expected_nepa029_sha256.lower():
        raise RuntimeError("nepa029-hash-conflict")

    nepa043, metadata043 = parse_rwl_allowed_years(
        nepa043_path,
        minimum_year=MIN_YEAR,
        maximum_year=MAX_YEAR,
        forbidden_intervals=LOCKED,
    )
    nepa029, metadata029 = parse_rwl_allowed_years(
        nepa029_path,
        minimum_year=MIN_YEAR,
        maximum_year=MAX_YEAR,
        forbidden_intervals=LOCKED,
    )
    if nepa043.empty or nepa029.empty:
        raise RuntimeError("empty-modern-rwl-matrix")

    comparisons: list[dict[str, object]] = []
    for left_name in nepa043.columns:
        for right_name in nepa029.columns:
            comparisons.append(
                {
                    "nepa043_sample": str(left_name),
                    "nepa029_sample": str(right_name),
                    **pair_metrics(
                        nepa043[left_name].to_numpy(dtype=float),
                        nepa029[right_name].to_numpy(dtype=float),
                    ),
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    exact_count = int(comparison_frame.exact_affine_alias.sum())
    near_count = int(comparison_frame.near_exact_standardized_sequence.sum())

    chronology043 = site_chronology(nepa043)
    chronology029 = site_chronology(nepa029)
    chronology_metrics = pair_metrics(
        chronology043.to_numpy(dtype=float), chronology029.to_numpy(dtype=float)
    )
    independent = exact_count == 0 and near_count == 0
    decision = (
        "advance_nepa043_to_modern_temperature_oof"
        if independent
        else "quarantine_nepa043_as_nepa029_sequence_alias"
    )

    after = {
        "nepa043": fingerprint(nepa043_path),
        "nepa029": fingerprint(nepa029_path),
    }
    if before != after:
        raise RuntimeError("source-fingerprint-changed")

    output.mkdir(parents=True)
    comparison_frame.to_csv(output / "sample_pair_identity_metrics.csv", index=False)
    pd.DataFrame(
        {
            "year": np.arange(MIN_YEAR, MAX_YEAR + 1),
            "nepa043_standardized_mean": chronology043.to_numpy(dtype=float),
            "nepa029_standardized_mean": chronology029.to_numpy(dtype=float),
        }
    ).to_csv(output / "modern_site_chronologies.csv", index=False)
    summary: dict[str, object] = {
        "experiment_id": "round_266b_nepa043_nepa029_modern_identity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "years_decoded": [MIN_YEAR, MAX_YEAR],
        "nepa043_sample_count": int(nepa043.shape[1]),
        "nepa029_sample_count": int(nepa029.shape[1]),
        "sample_pair_count": int(len(comparison_frame)),
        "exact_affine_alias_count": exact_count,
        "near_exact_standardized_sequence_count": near_count,
        "site_chronology_metrics": chronology_metrics,
        "nepa043_parser_metadata": metadata043,
        "nepa029_parser_metadata": metadata029,
        "decision": decision,
        "locked_1921_1939_values_accessed": False,
        "nepa043_may_be_used_for_locked_final_validation": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 266b NEPA043 versus NEPA029 modern identity audit\n\n"
        f"- Exact sample aliases: `{exact_count}`.\n"
        f"- Near-exact standardized sample sequences: `{near_count}`.\n"
        f"- Decision: `{decision}`.\n"
        "- Only 1940-2000 values were decoded.\n"
        "- NEPA043 remains permanently excluded from locked-period validation.\n",
        encoding="ascii",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-modern-only", action="store_true")
    parser.add_argument("--nepa043", type=Path, required=True)
    parser.add_argument("--nepa029", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-nepa043-sha256", required=True)
    parser.add_argument("--expected-nepa029-sha256", required=True)
    args = parser.parse_args()
    if not args.execute_modern_only:
        raise RuntimeError("explicit-modern-only-authorization-flag-required")
    summary = run_audit(
        args.nepa043,
        args.nepa029,
        args.output,
        args.expected_nepa043_sha256,
        args.expected_nepa029_sha256,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

