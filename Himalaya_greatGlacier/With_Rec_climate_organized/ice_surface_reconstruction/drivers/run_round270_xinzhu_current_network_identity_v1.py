from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from safe_two_column_year_first_v1 import parse_two_column_allowed_years


YEARS = np.arange(1940, 2001, dtype=int)


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if stat.st_size <= 100_000_000 else None
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest,
    }


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    valid = np.isfinite(candidate) & np.isfinite(reference)
    count = int(valid.sum())
    if count < 3:
        return {
            "overlap_count": count,
            "correlation": np.nan,
            "absolute_correlation": np.nan,
            "affine_rmse_fraction": np.nan,
            "exact_affine_alias": False,
            "near_alias": False,
        }
    left = candidate[valid].astype(float)
    right = reference[valid].astype(float)
    left_sd = float(np.std(left))
    right_sd = float(np.std(right))
    if left_sd <= 1e-12 or right_sd <= 1e-12:
        correlation = np.nan
        fraction = np.nan
    else:
        correlation = float(np.corrcoef(left, right)[0, 1])
        design = np.column_stack((np.ones(count), right))
        fitted = design @ np.linalg.lstsq(design, left, rcond=None)[0]
        fraction = float(np.sqrt(np.mean((left - fitted) ** 2)) / left_sd)
    absolute = abs(correlation) if np.isfinite(correlation) else np.nan
    exact = bool(count >= 50 and absolute >= 0.999999 and fraction <= 1e-6)
    near = bool(count >= 50 and absolute >= 0.995)
    return {
        "overlap_count": count,
        "correlation": correlation,
        "absolute_correlation": absolute,
        "affine_rmse_fraction": fraction,
        "exact_affine_alias": exact,
        "near_alias": near,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-modern-only", action="store_true")
    parser.add_argument("--xinzhu", type=Path, required=True)
    parser.add_argument("--expected-xinzhu-sha256", required=True)
    parser.add_argument("--proxy6", type=Path, required=True)
    parser.add_argument("--proxy9", type=Path, required=True)
    parser.add_argument("--proxy6-metadata", type=Path, required=True)
    parser.add_argument("--proxy9-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute_modern_only:
        raise RuntimeError("explicit-modern-only-authorization-flag-required")
    if args.output.exists():
        raise FileExistsError(args.output)

    paths = {
        "xinzhu": args.xinzhu,
        "proxy6": args.proxy6,
        "proxy9": args.proxy9,
        "proxy6_metadata": args.proxy6_metadata,
        "proxy9_metadata": args.proxy9_metadata,
        "driver": Path(__file__).resolve(),
        "contract": Path(__file__).resolve().parent / "round270_xinzhu_current_network_identity_v1_contract.md",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: fingerprint(path) for name, path in paths.items()}
    if before["xinzhu"]["sha256"].lower() != args.expected_xinzhu_sha256.lower():
        raise RuntimeError("xinzhu-hash-conflict")

    series, parse_metadata = parse_two_column_allowed_years(
        args.xinzhu,
        minimum_year=1940,
        maximum_year=2000,
        forbidden_intervals=((1921, 1939),),
    )
    candidate = series.reindex(YEARS).to_numpy(dtype=float)
    if not np.isfinite(candidate).all():
        raise RuntimeError("xinzhu-modern-gap")

    metadata = {
        "proxy6": pd.read_csv(args.proxy6_metadata),
        "proxy9": pd.read_csv(args.proxy9_metadata),
    }
    rows: list[dict[str, object]] = []
    for network, path in (("proxy6", args.proxy6), ("proxy9", args.proxy9)):
        observations = np.asarray(loadmat(path, variable_names=["observations"])["observations"], dtype=float)
        if observations.shape[0] != len(YEARS):
            raise RuntimeError(f"unexpected-{network}-shape:{observations.shape}")
        if observations.shape[1] != len(metadata[network]):
            raise RuntimeError(f"{network}-metadata-count-mismatch")
        for column in range(observations.shape[1]):
            meta = metadata[network].iloc[column]
            item = {
                "network": network,
                "record_1based": column + 1,
                "latitude": float(meta.latitude),
                "longitude": float(meta.longitude),
                **metrics(candidate, observations[:, column]),
            }
            if network == "proxy9":
                item["record_name"] = str(meta.get("combined_name", ""))
                item["source_flag"] = str(meta.get("combined_source_flag", ""))
            else:
                item["record_name"] = ""
                item["source_flag"] = ""
            rows.append(item)
    comparison = pd.DataFrame(rows)
    exact_count = int(comparison.exact_affine_alias.sum())
    near_count = int(comparison.near_alias.sum())
    decision = (
        "advance_xinzhu_to_modern_ngs_precipitation_oof"
        if exact_count == 0 and near_count == 0
        else "quarantine_xinzhu_as_current_network_sequence_alias"
    )
    after = {name: fingerprint(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("source-fingerprint-changed")

    args.output.mkdir(parents=True)
    comparison.to_csv(args.output / "all_current_network_sequence_comparisons.csv", index=False)
    comparison.sort_values("absolute_correlation", ascending=False).head(30).to_csv(
        args.output / "top_current_network_sequence_matches.csv", index=False
    )
    summary = {
        "experiment_id": "round_270_xinzhu_current_network_identity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "years_decoded": [1940, 2000],
        "parse_metadata": parse_metadata,
        "proxy6_record_count": int((comparison.network == "proxy6").sum()),
        "proxy9_record_count": int((comparison.network == "proxy9").sum()),
        "exact_affine_alias_count": exact_count,
        "near_alias_count": near_count,
        "maximum_absolute_correlation": float(comparison.absolute_correlation.max()),
        "decision": decision,
        "locked_1921_1939_values_accessed": False,
        "paleo_trajectory_read": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": True,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    (args.output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="ascii"
    )
    (args.output / "FINAL_RESULT.md").write_text(
        "# Round 270 Xinzhu current-network identity\n\n"
        f"- Exact aliases: `{exact_count}`.\n"
        f"- Near aliases: `{near_count}`.\n"
        f"- Maximum absolute correlation: `{comparison.absolute_correlation.max():.4f}`.\n"
        f"- Decision: `{decision}`.\n"
        "- Only 1940-2000 Xinzhu values were decoded.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
