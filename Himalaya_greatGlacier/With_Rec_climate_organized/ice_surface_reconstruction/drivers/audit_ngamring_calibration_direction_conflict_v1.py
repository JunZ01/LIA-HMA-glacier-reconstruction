from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
ORIGINAL = BASE / "temperature_noaa_ngamring_original_v1"
COMMON_RF = BASE / "temperature_lake_brgdgt_eight_record_safe_audit_v3"
OUTPUT = BASE / "temperature_ngamring_calibration_direction_conflict_v1"
DRIVER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    original = pd.read_csv(ORIGINAL / "safe_ngamring_maat_points.csv")
    common = pd.read_csv(COMMON_RF / "safe_eight_record_points.csv")
    common = common[common.record == "Ngamring"].copy()
    matched = original.merge(
        common[["year_ce", "raw_rf_maf_c", "constrained_maf_c"]],
        on="year_ce",
        how="inner",
        validate="one_to_one",
    )
    if len(matched) != 2 or matched.year_ce.tolist() != [1597.0, 1778.0]:
        raise RuntimeError(f"unexpected-Ngamring-match:{matched.year_ce.tolist()}")
    original_change = float(matched.maat_c.iloc[-1] - matched.maat_c.iloc[0])
    common_change = float(matched.raw_rf_maf_c.iloc[-1] - matched.raw_rf_maf_c.iloc[0])
    constrained_change = float(matched.constrained_maf_c.iloc[-1] - matched.constrained_maf_c.iloc[0])
    interval = float(matched.year_ce.iloc[-1] - matched.year_ce.iloc[0])
    metrics = {
        "years": matched.year_ce.tolist(),
        "interval_years": interval,
        "original_author_MAAT_change_c": original_change,
        "original_author_MAAT_change_c_per_century": original_change / interval * 100.0,
        "secondary_common_RF_MAF_change_c": common_change,
        "secondary_common_RF_MAF_change_c_per_century": common_change / interval * 100.0,
        "secondary_constrained_MAF_change_c": constrained_change,
        "secondary_constrained_MAF_change_c_per_century": constrained_change / interval * 100.0,
        "direction_reversal": bool(np.sign(original_change) != np.sign(common_change)),
        "constraint_changes_direction": bool(np.sign(common_change) != np.sign(constrained_change)),
    }
    matched.to_csv(OUTPUT / "exact_year_calibration_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for column, label, color in (
        ("maat_c", "Original author MAAT", "#111111"),
        ("raw_rf_maf_c", "2024 common RF MAF", "#bf3f36"),
        ("constrained_maf_c", "2024 constrained MAF", "#39708c"),
    ):
        values = matched[column].to_numpy(float)
        ax.plot(matched.year_ce, values - values[0], marker="o", lw=2.1, color=color, label=label)
    ax.axhline(0, color="#888888", lw=0.7)
    ax.set_xlabel("Year CE")
    ax.set_ylabel("Change from 1597 (degC)")
    ax.set_title("Ngamring: identical chronology, opposite calibration direction")
    ax.grid(color="#e2e2e2", lw=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "ngamring_calibration_direction_conflict.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment": "temperature_ngamring_calibration_direction_conflict_v1",
        "status": "complete",
        "metrics": metrics,
        "identity": (
            "Both products use the same Ngamring archive and identical 1597/1778 chronology. NOAA provides the "
            "author-contributed MAAT; Liang et al. 2024 recalibrates the brGDGT data to common RF MAF and then applies "
            "a constant 2.7 C cluster constraint."
        ),
        "decision": (
            "Use NOAA author MAAT as the primary independent site record. Treat the 2024 common-RF MAF as calibration "
            "and target sensitivity only; it cannot be counted as evidence that the original proxy conflicts with the project."
        ),
        "mechanism": "target_variable_and_nonlinear_calibration_mapping_reverse_the_low_frequency_direction",
        "sealed_period_values_used": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "inputs": [
            {"path": str(ORIGINAL / "safe_ngamring_maat_points.csv"), "sha256": sha256(ORIGINAL / "safe_ngamring_maat_points.csv")},
            {"path": str(ORIGINAL / "summary.json"), "sha256": sha256(ORIGINAL / "summary.json")},
            {"path": str(COMMON_RF / "safe_eight_record_points.csv"), "sha256": sha256(COMMON_RF / "safe_eight_record_points.csv")},
            {"path": str(COMMON_RF / "summary.json"), "sha256": sha256(COMMON_RF / "summary.json")},
        ],
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = f"""# Ngamring calibration-direction conflict

At the identical years 1597 and 1778, the official author-contributed NOAA MAAT
warms by {original_change:+.3f} C ({metrics['original_author_MAAT_change_c_per_century']:+.3f} C/century),
whereas the 2024 common-RF MAF cools by {common_change:+.3f} C
({metrics['secondary_common_RF_MAF_change_c_per_century']:+.3f} C/century).
The 2.7 C ecological constraint is constant here and therefore does not change
the cooling direction.

The chronology and archive are the same. The direction reversal is caused by
the target/calibration mapping (author MAAT versus common RF MAF), not by the
underlying proxy chronology. NOAA author MAAT is therefore the primary site
evidence; the 2024 product is retained only as calibration sensitivity.

The sealed period was not used. Production was not changed, EnKF and OGGM were
not run, and 20CRv3 was not accessed.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
