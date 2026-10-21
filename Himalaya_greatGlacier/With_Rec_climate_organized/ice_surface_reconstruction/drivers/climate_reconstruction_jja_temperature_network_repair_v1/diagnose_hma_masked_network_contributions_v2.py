"""HMA-mask-corrected decomposition of the 50-record JJA repair effect."""

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
    reference = (years >= 1961) & (years <= 1990)
    return values - np.mean(values[reference])


def summary(values: np.ndarray, years: np.ndarray, start: int, end: int) -> dict[str, float]:
    selected = (years >= start) & (years <= end)
    return {
        "mean_c": float(np.mean(values[selected])),
        "trend_c_per_century": float(100 * np.polyfit(years[selected], values[selected], 1)[0]),
        "std_c": float(np.std(values[selected])),
    }


def main() -> None:
    base = load_module(BASELINE_DRIVER)
    years = np.arange(1000, 2001)
    saved_series = pd.read_csv(EXPERIMENT / "hma_regional_series_2deg.csv")
    with np.load(EXPERIMENT / "controlled_repair_arrays.npz") as saved:
        mask2 = np.asarray(saved["mask2"], dtype=bool)

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
    if mask2.shape != (len(grid_lat), len(grid_lon)):
        raise RuntimeError(f"HMA mask shape {mask2.shape} does not match state grid")
    state_mask = mask2.T.reshape(-1)
    regional_weights = np.cos(np.deg2rad(state_lat)) * state_mask
    regional_weights /= regional_weights.sum()
    target_weights = base._target_weights(grid_lat, grid_lon, 33.75, 75.75)

    branches = {
        "baseline_annual": base._run_branch(prior, indices, proxy, residuals, target_weights, regional_weights),
        "repair_50_temperature_JJA": base._run_branch(prior, indices, repaired_proxy, residuals, target_weights, regional_weights),
    }
    selected_mask = np.zeros(proxy["hxb"].shape[0], dtype=bool)
    selected_mask[selected_records] = True
    curves: dict[str, np.ndarray] = {"year": years}
    rows: list[dict[str, object]] = []
    closure = {}

    for branch, result in branches.items():
        contributions = np.asarray(result["regional_contributions"])
        components = {
            "selected_50_increment": contributions[selected_mask].sum(axis=0),
            "other_247_increment": contributions[~selected_mask].sum(axis=0),
            "all_297_increment": contributions.sum(axis=0),
            "analysis": np.asarray(result["regional_analysis"]),
        }
        for component, values in components.items():
            centred = anomaly(values, years)
            curves[f"{branch}_{component}"] = centred
            for start, end in ((1901, 1950), (1951, 2000)):
                rows.append({"branch": branch, "component": component, "period": f"{start}-{end}", **summary(centred, years, start, end)})
        expected = saved_series[branch].to_numpy(dtype=float)
        closure[branch] = float(np.max(np.abs(curves[f"{branch}_analysis"] - expected)))

    if max(closure.values()) > 1e-12:
        raise RuntimeError(f"Masked HMA analysis does not close to saved series: {closure}")

    for component in ("selected_50_increment", "other_247_increment", "all_297_increment", "analysis"):
        difference = curves[f"repair_50_temperature_JJA_{component}"] - curves[f"baseline_annual_{component}"]
        name = f"repair_minus_baseline_{component}"
        curves[name] = difference
        for start, end in ((1901, 1950), (1951, 2000)):
            rows.append({"branch": "difference", "component": name, "period": f"{start}-{end}", **summary(difference, years, start, end)})

    table = pd.DataFrame(rows)
    table.to_csv(EXPERIMENT / "hma_masked_network_contribution_summary_v2.csv", index=False)
    pd.DataFrame(curves).to_csv(EXPERIMENT / "hma_masked_network_contribution_curves_v2.csv", index=False)
    def value(component: str, metric: str = "mean_c") -> float:
        return float(table.loc[(table.branch == "difference") & (table.component == component) & (table.period == "1901-1950"), metric].iloc[0])
    result = {
        "status": "pass",
        "hma_series_closure_max_abs_c": closure,
        "selected_record_count": int(selected_mask.sum()),
        "other_record_count": int((~selected_mask).sum()),
        "early_analysis_change_c": value("repair_minus_baseline_analysis"),
        "early_selected_50_increment_change_c": value("repair_minus_baseline_selected_50_increment"),
        "early_other_247_increment_change_c": value("repair_minus_baseline_other_247_increment"),
        "early_all_297_increment_change_c": value("repair_minus_baseline_all_297_increment"),
        "interpretation_boundary": "These are exact simultaneous-update algebraic contributions under the HMA-masked regional weights. They are not independent causal effects because all proxy weights respond to the changed covariance system.",
    }
    (EXPERIMENT / "hma_masked_network_contribution_summary_v2.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
