"""Separate raw proxy-record common mode from frozen PSM operator amplification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io


PERIOD = (1901, 1920)


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def trend(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100.0) if int(valid.sum()) >= 3 else float("nan")


def zscore_column(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    output = np.full(values.shape, np.nan)
    if int(valid.sum()) >= 3:
        std = np.std(values[valid])
        if std > 0:
            output[valid] = (values[valid] - np.mean(values[valid])) / std
    return output


def archive_proxy_path() -> Path:
    archive = "".join(chr(value) for value in (25991, 20214, 25972, 29702))
    paper = "1 " + "".join(chr(value) for value in (35770, 25991))
    return Path(r"D:\Users\JUNZ\Desktop") / archive / paper / "work1" / "compared_result_all" / "python_data" / "proxy_data_6.mat"


def main() -> int:
    project = root()
    mechanism = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_proxy_mechanism_audit_v1/positive_b2_proxy_mechanism_table.csv"
    replay = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_enkf_exp1_replay_v1/replay_input.mat"
    proxy6 = archive_proxy_path()
    result = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_observation_operator_source_audit_v1"
    if any(not path.exists() for path in (mechanism, replay, proxy6)):
        raise FileNotFoundError("missing-input")
    if result.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{result}")
    before = {"mechanism": digest(mechanism), "replay": digest(replay), "proxy6": digest(proxy6)}
    table = pd.read_csv(mechanism).set_index("proxy_index")
    selected = table.index.to_numpy(dtype=int)
    if len(selected) != 60:
        raise RuntimeError("positive-b2-count-contract-failed")
    S = scipy.io.loadmat(replay)
    sample_indices = np.asarray(S["sample_indices"]).reshape(-1).astype(int)
    years = np.arange(PERIOD[0], PERIOD[1] + 1)
    y_indices = years - 1000  # MATLAB t = year - 999; Python zero-based t - 1.
    observed = np.asarray(S["y0"], dtype=float)[selected - 1][:, y_indices].T
    replay_hxb = np.asarray(S["HXb0"], dtype=float)[selected - 1]
    raw = np.full_like(observed, np.nan)
    raw_hxb = np.full_like(replay_hxb, np.nan)
    p_component_std = np.full(len(selected), np.nan)
    t_component_std = np.full(len(selected), np.nan)
    rows = []
    with h5py.File(proxy6, "r") as handle:
        refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for column, proxy_index in enumerate(selected):
            proxy = handle[refs[proxy_index - 1]]
            data = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            ages = np.asarray(proxy["age_yng"], dtype=float).reshape(-1, order="F")
            b = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
            mp = np.asarray(proxy["M_p"], dtype=float).reshape(-1, order="F")[sample_indices - 1]
            mt = np.asarray(proxy["M_t"], dtype=float).reshape(-1, order="F")[sample_indices - 1]
            hxb = np.asarray(proxy["HXb"], dtype=float).reshape(-1, order="F")[sample_indices - 1]
            for row, year in enumerate(years):
                match = np.flatnonzero(ages == year)
                if len(match) == 1:
                    raw[row, column] = data[match[0]]
                elif len(match) > 1:
                    values = data[match]
                    if np.nanmax(values) - np.nanmin(values) > 1e-12:
                        raise RuntimeError(f"ambiguous-raw-record:{proxy_index}:{year}")
                    raw[row, column] = values[0]
            raw_hxb[column] = hxb
            p_component_std[column] = float(np.std(b[2] * (mp - np.mean(mp))))
            t_component_std[column] = float(np.std(b[1] * (mt - np.mean(mt))))
    raw_replay_error = float(np.nanmax(np.abs(raw - observed)))
    hxb_replay_error = float(np.nanmax(np.abs(raw_hxb - replay_hxb)))
    if raw_replay_error > 1e-12 or hxb_replay_error > 1e-12:
        raise RuntimeError(f"replay-closure-failed:raw={raw_replay_error}:hxb={hxb_replay_error}")
    operator_mean = replay_hxb.mean(axis=1)
    innovation = observed - operator_mean[None, :]
    observation_trends = np.array([trend(years, observed[:, column]) for column in range(len(selected))])
    innovation_trends = np.array([trend(years, innovation[:, column]) for column in range(len(selected))])
    innovation_observation_trend_error = float(np.nanmax(np.abs(observation_trends - innovation_trends)))
    if innovation_observation_trend_error > 1e-12:
        raise RuntimeError("innovation-observation-trend-contract-failed")
    raw_network = np.nanmean(np.column_stack([zscore_column(raw[:, column]) for column in range(len(selected))]), axis=1)
    innovation_network = np.nanmean(np.column_stack([zscore_column(innovation[:, column]) for column in range(len(selected))]), axis=1)
    network_closure = float(np.nanmax(np.abs(raw_network - innovation_network)))
    if network_closure > 1e-12:
        raise RuntimeError("network-common-mode-closure-failed")
    output = table.copy()
    output["raw_observation_trend_proxy_units_per_century"] = observation_trends
    output["innovation_trend_proxy_units_per_century"] = innovation_trends
    output["frozen_operator_mean"] = operator_mean
    output["precipitation_component_ensemble_std"] = p_component_std
    output["temperature_component_ensemble_std"] = t_component_std
    output["precipitation_to_temperature_ensemble_std_ratio"] = p_component_std / t_component_std
    output["raw_replay_observation_max_abs_error"] = raw_replay_error
    output["raw_hxb_replay_max_abs_error"] = hxb_replay_error
    output["direct_effect_vs_observation_trend_correlation"] = np.corrcoef(output["precipitation_pathway_effect_trend_mm_day_per_century"], observation_trends)[0, 1]
    output["direct_effect_vs_precipitation_component_std_correlation"] = np.corrcoef(output["precipitation_pathway_effect_trend_mm_day_per_century"], p_component_std)[0, 1]
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_positive_b2_observation_operator_source_audit_v1",
        "period": list(PERIOD), "locked_final_period_accessed": False,
        "positive_b2_proxy_count": int(len(selected)),
        "closure": {"raw_proxy_data_to_replay_observation_max_abs_error": raw_replay_error, "raw_HXb_to_replay_HXb_max_abs_error": hxb_replay_error, "innovation_minus_observation_trend_max_abs_error": innovation_observation_trend_error, "raw_vs_innovation_network_common_mode_max_abs_error": network_closure},
        "mechanism": {"observed_record_common_mode": "The frozen operator mean is time-independent over the replay; therefore each proxy innovation trend equals its replayed raw-observation trend exactly.", "operator_role": "M_p and b2 determine ensemble covariance/gain and spatial projection, but do not originate the annual innovation trend in this frozen replay."},
        "correlations": {"direct_effect_vs_observation_trend": float(np.corrcoef(output["precipitation_pathway_effect_trend_mm_day_per_century"], observation_trends)[0, 1]), "direct_effect_vs_precipitation_component_ensemble_std": float(np.corrcoef(output["precipitation_pathway_effect_trend_mm_day_per_century"], p_component_std)[0, 1])},
        "interpretation_limit": "This identifies the source of the replayed temporal innovation common mode, not whether the raw proxy common mode is climatically correct or which preprocessing decision created it.",
    }
    result.mkdir(parents=True)
    output.reset_index().to_csv(result / "positive_b2_observation_operator_source_table.csv", index=False)
    pd.DataFrame({"year": years, "raw_proxy_network_common_mode": raw_network, "innovation_network_common_mode": innovation_network}).to_csv(result / "raw_observation_innovation_common_mode_1901_1920.csv", index=False)
    (result / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(2, 1, figsize=(9.2, 7.4), constrained_layout=True)
    axes[0].plot(years, raw_network, color="#1b1b1b", linewidth=2.2, label="raw Proxy6 data")
    axes[0].plot(years, innovation_network, color="#D55E00", linewidth=1.4, linestyle="--", label="replay innovations")
    axes[0].axhline(0, color="#B5B5B5", linewidth=0.8); axes[0].set_ylabel("Standardized network mean"); axes[0].set_title("Raw record and replay-innovation common modes"); axes[0].legend(frameon=False)
    axes[1].scatter(observation_trends, output["precipitation_pathway_effect_trend_mm_day_per_century"], s=30 + 220 * np.clip(p_component_std / np.nanmax(p_component_std), 0, 1), c="#0072B2", alpha=0.85)
    axes[1].axhline(0, color="#B5B5B5", linewidth=0.8); axes[1].axvline(0, color="#B5B5B5", linewidth=0.8)
    axes[1].set_xlabel("Raw observation trend (proxy units century$^{-1}$)"); axes[1].set_ylabel("Direct P-pathway trend (mm day$^{-1}$ century$^{-1}$)"); axes[1].set_title("Observation trend versus P-pathway effect (marker size: P ensemble spread)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(result / "positive_b2_observation_operator_source_audit.png", dpi=180); plt.close(figure)
    report = ["# Positive-b2 Observation-versus-Operator Source Audit", "", "Period: 1901-1920 only. The locked 1921-1939 final test period was not accessed.", "", f"- Raw Proxy6 data to replay observation closure: `{raw_replay_error:.3e}`.", f"- Raw HXb to replay HXb closure: `{hxb_replay_error:.3e}`.", f"- Innovation versus observation trend closure: `{innovation_observation_trend_error:.3e}`.", "", "The common temporal innovation signal originates in the replayed proxy-record observations. The PSM precipitation term determines how that signal is projected through the EnKF gain, not the annual signal origin in this frozen replay."]
    (result / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    after = {"mechanism": digest(mechanism), "replay": digest(replay), "proxy6": digest(proxy6)}
    if before != after:
        raise RuntimeError("source-inputs-changed-during-audit")
    (result / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "analysis_driver_sha256": digest(Path(__file__))}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
