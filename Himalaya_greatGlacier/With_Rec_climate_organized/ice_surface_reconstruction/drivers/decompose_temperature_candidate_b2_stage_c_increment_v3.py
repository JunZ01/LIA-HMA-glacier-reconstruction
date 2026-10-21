from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import decompose_temperature_candidate_b2_stage_c_increment_v1 as impl


ROOT = Path(__file__).resolve().parents[4]
impl.ROOT = ROOT
impl.LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
impl.STATIC = impl.LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2/candidate_b2_stage_c_static_input_v2.npz"
impl.REAL = impl.LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_real_proxy_v3/candidate_b2_stage_c_real_proxy_output_v1.npz"
impl.OUTPUT = impl.LOOP / "temp_b2_stage_c_increment_decomp_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    impl.main()
    static = np.load(impl.STATIC)
    real = np.load(impl.REAL)
    years = np.asarray(real["years"], dtype=int)
    weights = np.asarray(static["hma_weights"], dtype=float)
    weights /= weights.sum()
    background = np.asarray(real["no_proxy_balanced_background"], dtype=float) @ weights

    increments = {}
    for name, field in (
        ("proxy6", "proxy6_posterior_temperature_anomaly"),
        ("proxy9", "proxy9_posterior_temperature_anomaly"),
        ("joint", "candidate_b2_joint_temperature_anomaly"),
    ):
        increments[name] = np.asarray(real[field], dtype=float) @ weights - background

    baseline = (years >= 1961) & (years <= 1990)
    periods = ((1500, 1849), (1850, 1920), (1940, 2000), (1961, 1990))
    rows = []
    for name, values in increments.items():
        baseline_mean = float(values[baseline].mean())
        for start, end in periods:
            period = (years >= start) & (years <= end)
            raw_mean = float(values[period].mean())
            rows.append(
                {
                    "network": name,
                    "period": f"{start}-{end}",
                    "raw_mean_increment_degC": raw_mean,
                    "mean_increment_1961_1990_degC": baseline_mean,
                    "baseline_normalized_displacement_degC": raw_mean - baseline_mean,
                }
            )
    normalized = pd.DataFrame(rows)
    normalized.to_csv(impl.OUTPUT / "baseline_normalized_increment_summary.csv", index=False)

    table = normalized.set_index(["network", "period"])
    joint_early_raw = float(table.loc[("joint", "1500-1849"), "raw_mean_increment_degC"])
    joint_modern_raw = float(table.loc[("joint", "1961-1990"), "raw_mean_increment_degC"])
    joint_early_relative = float(table.loc[("joint", "1500-1849"), "baseline_normalized_displacement_degC"])
    joint_bridge_relative = float(table.loc[("joint", "1850-1920"), "baseline_normalized_displacement_degC"])

    summary_path = impl.OUTPUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "experiment": impl.OUTPUT.name,
            "mechanism_correction": "the EnKF cools all periods, but cools the 1961-1990 anchor much more strongly than the early period; anomaly re-anchoring therefore raises the early reconstruction relative to modern",
            "joint_raw_mean_increment_1500_1849_degC": joint_early_raw,
            "joint_raw_mean_increment_1961_1990_degC": joint_modern_raw,
            "joint_baseline_normalized_displacement_1500_1849_degC": joint_early_relative,
            "joint_baseline_normalized_displacement_1850_1920_degC": joint_bridge_relative,
            "finding": "both proxy networks produce a stronger negative update in the modern anchor than in the early period; the apparent early warm return is a differential-update and re-anchoring effect, not a positive early EnKF increment",
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = f"""# Candidate B2 Stage C corrected increment decomposition

The HMA decomposition closes to the saved posterior within {max(summary['closure'].values()):.3e} degC.

- The joint raw EnKF increment is {joint_early_raw:+.3f} degC in 1500-1849 and {joint_modern_raw:+.3f} degC in the 1961-1990 anchor.
- Because the modern anchor is cooled more strongly, re-expression as a 1961-1990 anomaly raises 1500-1849 by {joint_early_relative:+.3f} degC and 1850-1920 by {joint_bridge_relative:+.3f} degC relative to the balanced background.
- Proxy6 and proxy9 show the same differential-update mechanism. The apparent restoration of early warmth is not a positive early EnKF increment and is not unique to one network.

This identifies the failed low-frequency bridge: the real-proxy update has a time-dependent mean strength, and modern-baseline normalization converts that difference into an early warm displacement. The next controlled sensitivity must preserve the successful covariance response while preventing the proxy update from redefining the 1961-1990 zero point.
"""
    (impl.OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")

    manifest = {
        "inputs": {str(impl.STATIC): sha256(impl.STATIC), str(impl.REAL): sha256(impl.REAL)},
        "outputs": {
            str(path): sha256(path)
            for path in sorted(impl.OUTPUT.iterdir())
            if path.name != "source_manifest.json"
        },
    }
    (impl.OUTPUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
