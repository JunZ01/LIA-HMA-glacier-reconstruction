"""Decompose the HMA effect of the controlled 50-record JJA repair."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io


ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_jja_temperature_network_repair_v1"
BASELINE_DRIVER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_controlled_enkf_early20_v1/run_controlled_enkf.py"
PRIOR = ROOT / "validation_data/results/paleo_surface_validation_v0/enkf_reconstruction_source_recovery_audit_v2/replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy"
REGEM = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_regem_replay_v1/regem_output.mat"
ARCHIVE = Path(r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all")
INDICES = ARCHIVE / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat"
PROXY9 = ARCHIVE / "python_data/proxy_data_9.mat"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("controlled_enkf", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module-load-failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def anomaly(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    baseline = (years >= 1961) & (years <= 1990)
    return values - np.mean(values[..., baseline], axis=-1, keepdims=True)


def period_summary(values: np.ndarray, years: np.ndarray, start: int, end: int) -> dict[str, float]:
    selected = (years >= start) & (years <= end)
    y = values[selected]
    x = years[selected].astype(float)
    return {
        "mean_c": float(np.mean(y)),
        "trend_c_per_century": float(100 * np.polyfit(x, y, 1)[0]),
        "std_c": float(np.std(y)),
    }


def main() -> None:
    base = load_module(BASELINE_DRIVER)
    years = np.arange(1000, 2001)
    prior = np.load(PRIOR)
    with h5py.File(INDICES, "r") as handle:
        indices = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(PROXY9)
    residuals = np.asarray(scipy.io.loadmat(REGEM)["R0v_filled"], dtype=float)
    psm = scipy.io.loadmat(EXPERIMENT / "network_jja_psm_delta_anchor.mat", squeeze_me=True)
    selected_records = np.atleast_1d(psm["record_ids"]).astype(int) - 1
    repaired_proxy = {key: np.asarray(value).copy() for key, value in proxy.items()}
    repaired_proxy["hxb"][selected_records] = np.asarray(psm["anchored_jja_hxb"], dtype=float)
    grid_lat, grid_lon, state_lat, _ = base._state_coordinates()
    target_weights = base._target_weights(grid_lat, grid_lon, 33.75, 75.75)
    regional_weights = np.cos(np.deg2rad(state_lat))
    regional_weights /= regional_weights.sum()
    baseline = base._run_branch(prior, indices, proxy, residuals, target_weights, regional_weights)
    repaired = base._run_branch(prior, indices, repaired_proxy, residuals, target_weights, regional_weights)

    selected_mask = np.zeros(proxy["hxb"].shape[0], dtype=bool)
    selected_mask[selected_records] = True
    rows = []
    curves: dict[str, np.ndarray] = {"year": years}
    for branch_name, result in (("baseline_annual", baseline), ("repair_50_temperature_JJA", repaired)):
        contributions = np.asarray(result["regional_contributions"])
        groups = {
            "selected_50": contributions[selected_mask].sum(axis=0),
            "other_247": contributions[~selected_mask].sum(axis=0),
            "all_297": contributions.sum(axis=0),
        }
        analysis = np.asarray(result["regional_analysis"])
        groups["analysis"] = analysis
        for group, values in groups.items():
            centred = anomaly(values, years)
            curves[f"{branch_name}_{group}"] = centred
            for start, end in ((1901, 1950), (1951, 2000)):
                rows.append({"branch": branch_name, "component": group, "period": f"{start}-{end}", **period_summary(centred, years, start, end)})

    difference = curves["repair_50_temperature_JJA_analysis"] - curves["baseline_annual_analysis"]
    curves["repair_minus_baseline_analysis"] = difference
    difference_selected = curves["repair_50_temperature_JJA_selected_50"] - curves["baseline_annual_selected_50"]
    difference_other = curves["repair_50_temperature_JJA_other_247"] - curves["baseline_annual_other_247"]
    curves["repair_minus_baseline_selected_50"] = difference_selected
    curves["repair_minus_baseline_other_247"] = difference_other
    for component, values in (("repair_minus_baseline_analysis", difference), ("repair_minus_baseline_selected_50", difference_selected), ("repair_minus_baseline_other_247", difference_other)):
        for start, end in ((1901, 1950), (1951, 2000)):
            rows.append({"branch": "difference", "component": component, "period": f"{start}-{end}", **period_summary(values, years, start, end)})
    table = pd.DataFrame(rows)
    table.to_csv(EXPERIMENT / "network_contribution_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(EXPERIMENT / "network_contribution_curves.csv", index=False)
    summary = {
        "selected_record_count": int(selected_mask.sum()),
        "other_record_count": int((~selected_mask).sum()),
        "early_analysis_change_c": float(table.loc[(table.branch == "difference") & (table.component == "repair_minus_baseline_analysis") & (table.period == "1901-1950"), "mean_c"].iloc[0]),
        "early_selected_50_change_c": float(table.loc[(table.branch == "difference") & (table.component == "repair_minus_baseline_selected_50") & (table.period == "1901-1950"), "mean_c"].iloc[0]),
        "early_other_247_change_c": float(table.loc[(table.branch == "difference") & (table.component == "repair_minus_baseline_other_247") & (table.period == "1901-1950"), "mean_c"].iloc[0]),
        "interpretation": "Simultaneous EnKF contributions are algebraic covariance-weighted terms, not independent physical causation. The selected-50 and other-247 changes sum to the analysis change after common anomaly centering.",
    }
    (EXPERIMENT / "network_contribution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
