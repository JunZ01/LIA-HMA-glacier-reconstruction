"""Test spatial and temporal transferability of direct-station residual correction."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PERIODS = ("1880-1900", "1901-1920", "1940-1950")
METHODS = ("equal_mean", "gaussian_500km", "nearest")


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


def haversine(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    radius = 6371.0088
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = phi1 - phi2
    dlambda = np.deg2rad(lon1 - lon2)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def predict(train: pd.DataFrame, target: pd.Series, method: str) -> float:
    distances = haversine(train.latitude.to_numpy(float), train.longitude.to_numpy(float), float(target.latitude), float(target.longitude))
    values = train.residual_degC.to_numpy(float)
    if method == "equal_mean":
        weights = np.ones(len(train))
    elif method == "gaussian_500km":
        weights = np.exp(-(distances / 500.0) ** 2)
    elif method == "nearest":
        weights = np.zeros(len(train))
        weights[int(np.argmin(distances))] = 1.0
    else:
        raise ValueError(method)
    return float(np.sum(weights * values) / np.sum(weights))


def score(frame: pd.DataFrame) -> dict[str, float]:
    before = frame.residual_degC.to_numpy(float)
    after = frame.corrected_residual_degC.to_numpy(float)
    before_mae = float(np.mean(np.abs(before)))
    after_mae = float(np.mean(np.abs(after)))
    return {
        "station_count": int(len(frame)),
        "before_mean_bias_degC": float(np.mean(before)),
        "after_mean_bias_degC": float(np.mean(after)),
        "before_mae_degC": before_mae,
        "after_mae_degC": after_mae,
        "mae_improvement_fraction": 1.0 - after_mae / before_mae if before_mae > 0 else np.nan,
        "before_rmse_degC": float(np.sqrt(np.mean(before**2))),
        "after_rmse_degC": float(np.sqrt(np.mean(after**2))),
    }


def main() -> int:
    root = root_path()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "temperature_station_residual_spatial_transfer_feasibility_v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    paths = {
        "station_curves": loop / "temperature_station_annual_anomaly_benchmark_v4/station_annual_anomalies_and_products.csv",
        "station_benchmark_summary": loop / "temperature_station_annual_anomaly_benchmark_v4/summary.json",
        "local_proxy_summary": loop / "temperature_six_station_local_proxy_innovation_adjudication_v1/summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before_hash = {name: digest(path) for name, path in paths.items()}
    upstream = json.loads(paths["station_benchmark_summary"].read_text(encoding="utf-8"))
    local = json.loads(paths["local_proxy_summary"].read_text(encoding="utf-8"))
    if upstream["locked_1921_1939_accessed"] or local["locked_1921_1939_accessed"]:
        raise RuntimeError("upstream-lock-boundary-failed")
    annual = pd.read_csv(paths["station_curves"])
    allowed = annual.year.between(1880, 1920) | annual.year.between(1940, 1950)
    annual = annual[allowed].copy()
    if annual.year.between(1921, 1939).any():
        raise RuntimeError("locked-year-present")

    station_period_rows = []
    for station_id, group in annual.groupby("station_id"):
        identity = group.iloc[0]
        for period in PERIODS:
            start, end = map(int, period.split("-"))
            section = group[group.year.between(start, end)]
            for target_column, target in (("qcf_station_degC", "qcf"), ("qcu_station_degC", "qcu")):
                selected = section[np.isfinite(section[target_column]) & np.isfinite(section.project_six_member_degC)]
                if len(selected) < 5:
                    continue
                station_period_rows.append({
                    "station_id": station_id,
                    "station_name": identity.station_name,
                    "latitude": float(identity.latitude),
                    "longitude": float(identity.longitude),
                    "region_group": identity.region_group,
                    "period": period,
                    "target": target,
                    "matched_years": int(len(selected)),
                    "residual_degC": float(np.mean(selected.project_six_member_degC - selected[target_column])),
                })
    station_period = pd.DataFrame(station_period_rows)
    station_period.to_csv(output / "station_period_residuals.csv", index=False)

    loo_rows = []
    for (period, target), group in station_period.groupby(["period", "target"]):
        if len(group) < 3:
            continue
        for method in METHODS:
            for _, target_station in group.iterrows():
                train = group[group.station_id != target_station.station_id]
                correction = predict(train, target_station, method)
                loo_rows.append({
                    "period": period,
                    "target": target,
                    "method": method,
                    "station_id": target_station.station_id,
                    "station_name": target_station.station_name,
                    "residual_degC": target_station.residual_degC,
                    "predicted_correction_degC": correction,
                    "corrected_residual_degC": target_station.residual_degC - correction,
                })
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(output / "leave_one_station_out_predictions.csv", index=False)
    loo_summary_rows = []
    for keys, group in loo.groupby(["period", "target", "method"]):
        period, target, method = keys
        loo_summary_rows.append({"period": period, "target": target, "method": method, **score(group)})
    loo_summary = pd.DataFrame(loo_summary_rows)
    loo_summary.to_csv(output / "leave_one_station_out_summary.csv", index=False)

    transfer_rows = []
    for train_period, test_period in (("1880-1900", "1901-1920"), ("1901-1920", "1940-1950")):
        for target in ("qcf", "qcu"):
            train = station_period[(station_period.period == train_period) & (station_period.target == target)]
            test = station_period[(station_period.period == test_period) & (station_period.target == target)]
            if len(train) < 3 or len(test) < 3:
                continue
            for method in METHODS:
                predictions = []
                for _, target_station in test.iterrows():
                    correction = predict(train, target_station, method)
                    predictions.append({
                        "train_period": train_period,
                        "test_period": test_period,
                        "target": target,
                        "method": method,
                        "station_id": target_station.station_id,
                        "station_name": target_station.station_name,
                        "residual_degC": target_station.residual_degC,
                        "predicted_correction_degC": correction,
                        "corrected_residual_degC": target_station.residual_degC - correction,
                    })
                frame = pd.DataFrame(predictions)
                transfer_rows.extend(predictions)
    transfer = pd.DataFrame(transfer_rows)
    transfer.to_csv(output / "cross_period_transfer_predictions.csv", index=False)
    transfer_summary_rows = []
    for keys, group in transfer.groupby(["train_period", "test_period", "target", "method"]):
        train_period, test_period, target, method = keys
        transfer_summary_rows.append({"train_period": train_period, "test_period": test_period, "target": target, "method": method, **score(group)})
    transfer_summary = pd.DataFrame(transfer_summary_rows)
    transfer_summary.to_csv(output / "cross_period_transfer_summary.csv", index=False)

    primary = loo_summary[(loo_summary.period == "1901-1920") & (loo_summary.target == "qcf")]
    best_primary = primary.sort_values("mae_improvement_fraction", ascending=False).iloc[0]
    qcu_match = loo_summary[(loo_summary.period == "1901-1920") & (loo_summary.target == "qcu") & (loo_summary.method == best_primary.method)].iloc[0]
    late_transfer = transfer_summary[(transfer_summary.train_period == "1901-1920") & (transfer_summary.test_period == "1940-1950") & (transfer_summary.target == "qcf") & (transfer_summary.method == best_primary.method)].iloc[0]
    spatial_generalizes = bool(best_primary.mae_improvement_fraction >= 0.30 and qcu_match.mae_improvement_fraction >= 0.20)
    amplitude_persists = bool(late_transfer.mae_improvement_fraction > 0)
    if spatial_generalizes and not amplitude_persists:
        decision = "regional_correction_spatially_supported_but_early_amplitude_must_decay_before_1940_1950"
    elif spatial_generalizes and amplitude_persists:
        decision = "regional_correction_spatially_and_cross_period_supported"
    else:
        decision = "station_residual_correction_not_spatially_generalizable"

    after_hash = {name: digest(path) for name, path in paths.items()}
    unchanged = before_hash == after_hash
    if not unchanged:
        raise RuntimeError("derived-input-changed")
    summary = {
        "experiment": "temperature_station_residual_spatial_transfer_feasibility_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "periods": list(PERIODS),
        "locked_1921_1939_accessed": False,
        "primary_1901_1920_qcf_best_method": best_primary.to_dict(),
        "same_method_qcu_sensitivity": qcu_match.to_dict(),
        "same_method_1901_1920_to_1940_1950_qcf_transfer": late_transfer.to_dict(),
        "spatial_generalization_passes": spatial_generalizes,
        "early_amplitude_persists_to_1940_1950": amplitude_persists,
        "decision": decision,
        "interpretation_boundary": "LOO tests transfer among available stations, not independence from GHCN-M. Cross-period transfer applies the early correction amplitude unchanged and therefore diagnoses whether temporal decay is required; it does not estimate values in sealed 1921-1939.",
        "derived_inputs_unchanged": unchanged,
        "production_modified": False,
        "enkf_rerun": False,
        "oggm_run": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before_sha256": before_hash, "after_sha256": after_hash, "unchanged": unchanged}, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    qcf_loo = loo_summary[loo_summary.target.eq("qcf")]
    for method, color in zip(METHODS, ("#00798C", "#7A5195", "#B23A48")):
        values = qcf_loo[qcf_loo.method.eq(method)].set_index("period").reindex(PERIODS)
        axes[0].plot(PERIODS, values.mae_improvement_fraction * 100, marker="o", color=color, label=method)
    axes[0].axhline(0, color="#777777", lw=0.8)
    axes[0].set(ylabel="LOO MAE improvement (%)", title="Within-period spatial transfer")
    axes[0].legend(frameon=False)
    late = transfer_summary[(transfer_summary.train_period == "1901-1920") & (transfer_summary.test_period == "1940-1950") & (transfer_summary.target == "qcf")]
    x = np.arange(len(late))
    axes[1].bar(x - 0.18, late.before_mae_degC, width=0.36, color="#777777", label="before")
    axes[1].bar(x + 0.18, late.after_mae_degC, width=0.36, color="#B23A48", label="after early correction")
    axes[1].set(xticks=x, xticklabels=late.method, ylabel="1940-1950 MAE (degC)", title="Transfer early amplitude into 1940-1950")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "station_residual_spatial_temporal_transfer.png", dpi=220)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "# Station residual spatial/temporal transfer feasibility\n\n"
        f"- Best 1901-1920 qcf LOO method: `{best_primary.method}`, MAE improvement `{100 * best_primary.mae_improvement_fraction:.1f}%`.\n"
        f"- Same-method qcu improvement: `{100 * qcu_match.mae_improvement_fraction:.1f}%`.\n"
        f"- Applying the unchanged 1901-1920 amplitude to 1940-1950 changes MAE by `{100 * late_transfer.mae_improvement_fraction:+.1f}%`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
