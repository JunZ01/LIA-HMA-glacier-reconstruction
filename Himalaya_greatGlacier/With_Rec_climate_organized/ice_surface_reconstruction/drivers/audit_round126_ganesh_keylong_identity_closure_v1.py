"""Close Ganesh and Keylong identities against all 297 current proxy records."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-spec-failed:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    driver = Path(__file__).resolve().parent
    r125 = load_module("round126_r125", driver / "qualify_round125_ganesh_keylong_d18o_v1.py")
    r123 = load_module("round126_r123", driver / "qualify_round123_southeast_tibet_d18o_v1.py")
    root = r123.project_root()
    round125 = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_ganesh_keylong_d18o_proxy_qualification_20260818_v1"
    )
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_ganesh_keylong_identity_closure_20260818_v1"
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "proxy9": root / (
            "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/"
            "round_03_nested_ensemble_selection/proxy9_allowed_1940_2000_official_baseline_v2.mat"
        ),
        "ledger": root / (
            "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/"
            "round_110_unresolved_proxy_metadata_recovery_v4/proxy_evidence_master_ledger_metadata_recovered.csv"
        ),
        "round125_summary": round125 / "summary.json",
        "ganesh": round125 / "raw_unopened/ganesh2018d18o-noaa.txt",
        "keylong": round125 / "raw_unopened/keylong2020d18o-noaa.txt",
        "driver": Path(__file__).resolve(),
        "round125_driver": driver / "qualify_round125_ganesh_keylong_d18o_v1.py",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: r123.file_state(path) for name, path in paths.items()}
    prior = json.loads(paths["round125_summary"].read_text(encoding="utf-8"))
    if prior["status"] != "complete" or prior["locked_1921_1939_candidate_values_decoded"] != 0:
        raise RuntimeError("round125-boundary-contract-failed")

    mat = scipy.io.loadmat(paths["proxy9"], variable_names=["years", "observations"])
    years = np.asarray(mat["years"], dtype=int).reshape(-1)
    observations = np.asarray(mat["observations"], dtype=float)
    if not np.array_equal(years, r123.YEARS) or observations.shape != (61, 297):
        raise RuntimeError("proxy9-contract-failed")
    ledger = pd.read_csv(paths["ledger"], low_memory=False)
    ledger = ledger[ledger.network.eq("proxy9")].copy()
    ledger["network_record_1based"] = ledger.network_record_1based.astype(int)
    ledger = ledger.sort_values("network_record_1based").reset_index(drop=True)
    if len(ledger) != 297:
        raise RuntimeError("ledger-contract-failed")

    rows = []
    decisions = {}
    for site in ("ganesh", "keylong"):
        candidate, audit = r125.parse_noaa_candidate(paths[site], years)
        if audit["locked_1921_1939_values_decoded"]:
            raise RuntimeError("locked-candidate-value-decoded")
        current_rows = []
        for index in range(observations.shape[1]):
            current = observations[:, index]
            valid = np.isfinite(candidate) & np.isfinite(current)
            if int(valid.sum()) < 20 or np.std(current[valid]) <= 1e-12:
                continue
            correlation = float(np.corrcoef(candidate[valid], current[valid])[0, 1])
            design = np.column_stack([np.ones(int(valid.sum())), current[valid]])
            coefficients, _, _, _ = np.linalg.lstsq(design, candidate[valid], rcond=None)
            residual = candidate[valid] - design @ coefficients
            normalized_rmse = float(np.sqrt(np.mean(residual**2)) / np.std(candidate[valid], ddof=1))
            item = ledger.iloc[index]
            current_rows.append(
                {
                    "site": site,
                    "proxy9_record_1based": index + 1,
                    "correlation": correlation,
                    "absolute_correlation": abs(correlation),
                    "affine_intercept": float(coefficients[0]),
                    "affine_slope": float(coefficients[1]),
                    "affine_normalized_rmse": normalized_rmse,
                    "latitude": float(item.latitude),
                    "longitude": float(item.longitude),
                    "site_name": str(item.get("site_name", "")),
                    "measurement": str(item.get("measurement", "")),
                    "data_name": str(item.get("data_name", "")),
                    "formal_metadata_identity": str(item.get("formal_metadata_identity", "")),
                }
            )
        ranked = sorted(current_rows, key=lambda row: (-row["absolute_correlation"], row["affine_normalized_rmse"]))
        for rank, row in enumerate(ranked[:10], start=1):
            rows.append({"rank": rank, **row})
        best = ranked[0]
        exact_sequence_duplicate = bool(
            best["absolute_correlation"] >= 0.999999 and best["affine_normalized_rmse"] <= 1e-6
        )
        decisions[site] = {
            "identity_status": "existing_sequence_duplicate" if exact_sequence_duplicate else "new_sequence",
            "best_proxy9_record_1based": best["proxy9_record_1based"],
            "best_absolute_correlation": best["absolute_correlation"],
            "best_affine_normalized_rmse": best["affine_normalized_rmse"],
            "best_site_name": best["site_name"],
            "best_latitude": best["latitude"],
            "best_longitude": best["longitude"],
        }
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "candidate_to_proxy9_top_matches.csv", index=False)
    after = {name: r123.file_state(path) for name, path in paths.items()}
    unchanged = bool(before == after)
    if not unchanged:
        raise RuntimeError("source-changed-during-round126")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2), encoding="utf-8"
    )
    summary = {
        "experiment_id": "round126_ganesh_keylong_identity_closure_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "locked_1921_1939_candidate_values_decoded": 0,
        "external_validation_reserve_opened": False,
        "source_hashes_unchanged": unchanged,
        "decisions": decisions,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Round 126 Ganesh and Keylong Identity Closure",
        "",
        "- Locked 1921-1939 candidate values decoded: `0`.",
        f"- Source hashes unchanged: `{str(unchanged).lower()}`.",
        "",
    ]
    for site, result in decisions.items():
        lines.extend(
            [
                f"## {site}",
                "",
                f"- Identity: `{result['identity_status']}`.",
                f"- Best proxy9 record: `{result['best_proxy9_record_1based']}`.",
                f"- Absolute correlation: `{result['best_absolute_correlation']:.6f}`.",
                f"- Affine normalized RMSE: `{result['best_affine_normalized_rmse']:.6g}`.",
                f"- Existing site label: `{result['best_site_name']}`.",
                "",
            ]
        )
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
