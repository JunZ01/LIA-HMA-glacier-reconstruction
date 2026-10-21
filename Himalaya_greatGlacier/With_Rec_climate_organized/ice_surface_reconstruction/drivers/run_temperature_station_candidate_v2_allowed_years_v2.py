"""Apply the frozen qcf-primary/qcu-sensitivity decision rule to Candidate v2."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def root_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    root = root_path()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    source = loop / "temperature_station_candidate_v2_allowed_years_v1"
    output = loop / "temperature_station_candidate_v2_allowed_years_v2"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    paths = {
        "candidate": source / "temperature_candidate_v2_allowed_years.nc",
        "metrics": source / "candidate_station_period_metrics.csv",
        "v1_summary": source / "summary.json",
        "transfer_summary": loop / "temperature_station_residual_spatial_transfer_feasibility_v2/summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: digest(path) for name, path in paths.items()}
    v1 = json.loads(paths["v1_summary"].read_text(encoding="utf-8"))
    transfer = json.loads(paths["transfer_summary"].read_text(encoding="utf-8"))
    if v1["locked_1921_1939_included_or_accessed"] or transfer["locked_1921_1939_accessed"]:
        raise RuntimeError("lock-boundary-failed")
    with netcdf_file(paths["candidate"], "r", mmap=False) as dataset:
        years = np.asarray(dataset.variables["year"].data, dtype=int).copy()
        candidate_shape = tuple(dataset.variables["temperature_anomaly_candidate_v2"].shape)
        original_shape = tuple(dataset.variables["temperature_anomaly_original"].shape)
    if np.any((years >= 1921) & (years <= 1939)) or not np.array_equal(years, np.r_[np.arange(1880, 1921), np.arange(1940, 1951)]):
        raise RuntimeError("candidate-year-contract-failed")
    if candidate_shape != original_shape or candidate_shape[:2] != (6, len(years)):
        raise RuntimeError("candidate-shape-contract-failed")

    metrics = pd.read_csv(paths["metrics"])
    qcf = metrics[metrics.target.eq("qcf")].set_index("period")
    qcu = metrics[metrics.target.eq("qcu")].set_index("period")
    supported = ("1880-1900", "1901-1920", "1940-1950")
    qcf_primary_pass = bool((qcf.loc[list(supported), "mae_improvement_fraction"] > 0).all())
    qcu_early_pass = bool((qcu.loc[["1880-1900", "1901-1920"], "mae_improvement_fraction"] > 0).all())
    qcu_late_warning = bool(qcu.loc["1940-1950", "mae_improvement_fraction"] < 0)
    oof_spatial_pass = bool(transfer["spatial_generalization_passes"])
    if qcf_primary_pass and oof_spatial_pass:
        decision = "retain_provisional_allowed_year_candidate_v2_with_late_qcu_homogenization_warning"
    else:
        decision = "reject_candidate_v2_primary_or_oof_gate_failed"

    after = {name: digest(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-changed")
    summary = {
        "experiment": "temperature_station_candidate_v2_allowed_years_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "supersedes_decision_of": "temperature_station_candidate_v2_allowed_years_v1",
        "correction": "The frozen design defines qcf as the primary target and qcu as a sensitivity diagnostic. V1 incorrectly required qcu improvement in every period as an acceptance gate.",
        "candidate_file": str(paths["candidate"]),
        "candidate_shape": candidate_shape,
        "candidate_year_count": int(len(years)),
        "locked_1921_1939_included_or_accessed": False,
        "qcf_primary_improves_all_supported_periods": qcf_primary_pass,
        "qcf_mae_improvement_fraction": {period: float(qcf.loc[period, "mae_improvement_fraction"]) for period in supported},
        "qcu_early_sensitivity_improves": qcu_early_pass,
        "qcu_1940_1950_warning": qcu_late_warning,
        "qcu_mae_improvement_fraction": {period: float(qcu.loc[period, "mae_improvement_fraction"]) for period in supported},
        "leave_one_station_out_spatial_gate": oof_spatial_pass,
        "production_eligible": False,
        "decision": decision,
        "next_gate": "Validate regional Candidate v2 against source-independent western/central temperature evidence or a prospectively sealed station test; do not use CRU/Berkeley as independent votes.",
        "source_hashes_unchanged": unchanged,
        "production_modified": False,
        "enkf_rerun": False,
        "oggm_run": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "candidate_reference.json").write_text(json.dumps({"candidate_file": str(paths["candidate"]), "sha256": before["candidate"]}, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before_sha256": before, "after_sha256": after, "unchanged": unchanged}, indent=2), encoding="utf-8")

    plot = metrics.copy()
    fig, axis = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    x = np.arange(len(supported))
    width = 0.34
    axis.bar(x - width / 2, qcf.loc[list(supported), "mae_improvement_fraction"] * 100, width=width, color="#B23A48", label="qcf primary")
    axis.bar(x + width / 2, qcu.loc[list(supported), "mae_improvement_fraction"] * 100, width=width, color="#00798C", label="qcu sensitivity")
    axis.axhline(0, color="#777777", lw=0.8)
    axis.set(xticks=x, xticklabels=supported, ylabel="Station MAE improvement (%)", title="Candidate v2: primary gate and homogenization sensitivity")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.savefig(output / "candidate_v2_qcf_qcu_decision.png", dpi=220)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "# Candidate v2 authoritative decision\n\n"
        "This result supersedes only the v1 acceptance decision; the v1 NetCDF and numerical metrics remain unchanged.\n\n"
        f"- qcf primary improvement: `1880-1900 {100*qcf.loc['1880-1900','mae_improvement_fraction']:.1f}%`, `1901-1920 {100*qcf.loc['1901-1920','mae_improvement_fraction']:.1f}%`, `1940-1950 {100*qcf.loc['1940-1950','mae_improvement_fraction']:.1f}%`.\n"
        f"- qcu sensitivity: `1880-1900 {100*qcu.loc['1880-1900','mae_improvement_fraction']:.1f}%`, `1901-1920 {100*qcu.loc['1901-1920','mae_improvement_fraction']:.1f}%`, `1940-1950 {100*qcu.loc['1940-1950','mae_improvement_fraction']:.1f}%`.\n"
        f"- Leave-one-station spatial gate: `{str(oof_spatial_pass).lower()}`.\n"
        "- Production eligible: `false`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
