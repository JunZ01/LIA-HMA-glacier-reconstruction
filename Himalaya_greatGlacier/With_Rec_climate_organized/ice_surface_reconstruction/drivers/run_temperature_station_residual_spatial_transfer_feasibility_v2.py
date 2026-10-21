"""V2 station residual transfer test with corrected upstream lock-field preflight."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DRIVER_DIR = Path(__file__).resolve().parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))
import run_temperature_station_residual_spatial_transfer_feasibility_v1 as base


def main() -> int:
    root = base.root_path()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "temperature_station_residual_spatial_transfer_feasibility_v2"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    paths = {
        "station_curves": loop / "temperature_station_annual_anomaly_benchmark_v4/station_annual_anomalies_and_products.csv",
        "station_benchmark_summary": loop / "temperature_station_annual_anomaly_benchmark_v4/summary.json",
        "local_proxy_summary": loop / "temperature_six_station_local_proxy_innovation_adjudication_v1/summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "failed_v1_driver": DRIVER_DIR / "run_temperature_station_residual_spatial_transfer_feasibility_v1.py",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: base.digest(path) for name, path in paths.items()}
    benchmark = json.loads(paths["station_benchmark_summary"].read_text(encoding="utf-8"))
    local = json.loads(paths["local_proxy_summary"].read_text(encoding="utf-8"))
    if benchmark["locked_monthly_cells_accessed"] != 0 or benchmark["unit_preflight"]["locked_monthly_cells_accessed"] != 0:
        raise RuntimeError("station-benchmark-lock-boundary-failed")
    if local["locked_1921_1939_accessed"]:
        raise RuntimeError("local-proxy-lock-boundary-failed")
    annual = pd.read_csv(paths["station_curves"])
    annual = annual[annual.year.between(1880, 1920) | annual.year.between(1940, 1950)].copy()
    if annual.year.between(1921, 1939).any():
        raise RuntimeError("locked-year-present")

    rows = []
    for station_id, group in annual.groupby("station_id"):
        identity = group.iloc[0]
        for period in base.PERIODS:
            start, end = map(int, period.split("-"))
            section = group[group.year.between(start, end)]
            for column, target in (("qcf_station_degC", "qcf"), ("qcu_station_degC", "qcu")):
                selected = section[np.isfinite(section[column]) & np.isfinite(section.project_six_member_degC)]
                if len(selected) >= 5:
                    rows.append({
                        "station_id": station_id,
                        "station_name": identity.station_name,
                        "latitude": float(identity.latitude),
                        "longitude": float(identity.longitude),
                        "period": period,
                        "target": target,
                        "matched_years": int(len(selected)),
                        "residual_degC": float(np.mean(selected.project_six_member_degC - selected[column])),
                    })
    station_period = pd.DataFrame(rows)
    station_period.to_csv(output / "station_period_residuals.csv", index=False)

    loo_rows = []
    for (period, target), group in station_period.groupby(["period", "target"]):
        if len(group) < 3:
            continue
        for method in base.METHODS:
            for _, held in group.iterrows():
                correction = base.predict(group[group.station_id != held.station_id], held, method)
                loo_rows.append({
                    "period": period, "target": target, "method": method,
                    "station_id": held.station_id, "station_name": held.station_name,
                    "residual_degC": held.residual_degC,
                    "predicted_correction_degC": correction,
                    "corrected_residual_degC": held.residual_degC - correction,
                })
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(output / "leave_one_station_out_predictions.csv", index=False)
    loo_summary = pd.DataFrame([
        {"period": period, "target": target, "method": method, **base.score(group)}
        for (period, target, method), group in loo.groupby(["period", "target", "method"])
    ])
    loo_summary.to_csv(output / "leave_one_station_out_summary.csv", index=False)

    transfer_rows = []
    for train_period, test_period in (("1880-1900", "1901-1920"), ("1901-1920", "1940-1950")):
        for target in ("qcf", "qcu"):
            train = station_period[(station_period.period == train_period) & (station_period.target == target)]
            test = station_period[(station_period.period == test_period) & (station_period.target == target)]
            if len(train) < 3 or len(test) < 3:
                continue
            for method in base.METHODS:
                for _, held in test.iterrows():
                    correction = base.predict(train, held, method)
                    transfer_rows.append({
                        "train_period": train_period, "test_period": test_period, "target": target, "method": method,
                        "station_id": held.station_id, "station_name": held.station_name,
                        "residual_degC": held.residual_degC,
                        "predicted_correction_degC": correction,
                        "corrected_residual_degC": held.residual_degC - correction,
                    })
    transfer = pd.DataFrame(transfer_rows)
    transfer.to_csv(output / "cross_period_transfer_predictions.csv", index=False)
    transfer_summary = pd.DataFrame([
        {"train_period": train_period, "test_period": test_period, "target": target, "method": method, **base.score(group)}
        for (train_period, test_period, target, method), group in transfer.groupby(["train_period", "test_period", "target", "method"])
    ])
    transfer_summary.to_csv(output / "cross_period_transfer_summary.csv", index=False)

    primary = loo_summary[(loo_summary.period == "1901-1920") & (loo_summary.target == "qcf")]
    best = primary.sort_values("mae_improvement_fraction", ascending=False).iloc[0]
    qcu = loo_summary[(loo_summary.period == "1901-1920") & (loo_summary.target == "qcu") & (loo_summary.method == best.method)].iloc[0]
    late = transfer_summary[(transfer_summary.train_period == "1901-1920") & (transfer_summary.test_period == "1940-1950") & (transfer_summary.target == "qcf") & (transfer_summary.method == best.method)].iloc[0]
    spatial = bool(best.mae_improvement_fraction >= 0.30 and qcu.mae_improvement_fraction >= 0.20)
    persists = bool(late.mae_improvement_fraction > 0)
    decision = (
        "regional_correction_spatially_supported_but_early_amplitude_must_decay_before_1940_1950"
        if spatial and not persists else
        "regional_correction_spatially_and_cross_period_supported" if spatial else
        "station_residual_correction_not_spatially_generalizable"
    )
    after = {name: base.digest(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("derived-input-changed")
    summary = {
        "experiment": "temperature_station_residual_spatial_transfer_feasibility_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "supersedes_failed_execution_of": "temperature_station_residual_spatial_transfer_feasibility_v1",
        "v1_failure": "upstream station benchmark exposes locked_monthly_cells_accessed rather than a generic boolean; no v1 scientific output was accepted",
        "periods": list(base.PERIODS),
        "locked_1921_1939_accessed": False,
        "primary_1901_1920_qcf_best_method": best.to_dict(),
        "same_method_qcu_sensitivity": qcu.to_dict(),
        "same_method_1901_1920_to_1940_1950_qcf_transfer": late.to_dict(),
        "spatial_generalization_passes": spatial,
        "early_amplitude_persists_to_1940_1950": persists,
        "decision": decision,
        "interpretation_boundary": "LOO tests transfer among available GHCN-M stations, not source independence. Cross-period transfer applies the early amplitude unchanged and reads no sealed-period value.",
        "derived_inputs_unchanged": unchanged,
        "production_modified": False,
        "enkf_rerun": False,
        "oggm_run": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before_sha256": before, "after_sha256": after, "unchanged": unchanged}, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    qcf_loo = loo_summary[loo_summary.target.eq("qcf")]
    for method, color in zip(base.METHODS, ("#00798C", "#7A5195", "#B23A48")):
        values = qcf_loo[qcf_loo.method.eq(method)].set_index("period").reindex(base.PERIODS)
        axes[0].plot(base.PERIODS, values.mae_improvement_fraction * 100, marker="o", color=color, label=method)
    axes[0].axhline(0, color="#777777", lw=0.8)
    axes[0].set(ylabel="LOO MAE improvement (%)", title="Within-period spatial transfer")
    axes[0].legend(frameon=False)
    late_all = transfer_summary[(transfer_summary.train_period == "1901-1920") & (transfer_summary.test_period == "1940-1950") & (transfer_summary.target == "qcf")]
    x = np.arange(len(late_all))
    axes[1].bar(x - 0.18, late_all.before_mae_degC, width=0.36, color="#777777", label="before")
    axes[1].bar(x + 0.18, late_all.after_mae_degC, width=0.36, color="#B23A48", label="after early correction")
    axes[1].set(xticks=x, xticklabels=late_all.method, ylabel="1940-1950 MAE (degC)", title="Transfer early amplitude into 1940-1950")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "station_residual_spatial_temporal_transfer.png", dpi=220)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "# Station residual spatial/temporal transfer feasibility v2\n\n"
        f"- Best 1901-1920 qcf LOO method: `{best.method}`, MAE improvement `{100 * best.mae_improvement_fraction:.1f}%`.\n"
        f"- Same-method qcu improvement: `{100 * qcu.mae_improvement_fraction:.1f}%`.\n"
        f"- Applying unchanged early amplitude to 1940-1950 changes MAE by `{100 * late.mae_improvement_fraction:+.1f}%`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
