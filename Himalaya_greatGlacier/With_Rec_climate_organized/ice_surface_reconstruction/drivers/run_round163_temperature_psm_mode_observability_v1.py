"""Map prior temperature modes to residual-standardized archived-PSM observability."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.linalg

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_round161_temperature_perfect_model_twin_v2 as round161
import run_round162_temperature_twin_information_geometry_v1 as round162
from temperature_fold_native_oenkf_increment_cv_v1 import run_experiment_v1 as common


EXPERIMENT_ID = "round_163_temperature_psm_mode_observability_v1"


def fingerprint(path: Path, digest: str | None = None) -> dict[str, object]:
    stat = path.stat()
    if digest is None and stat.st_size <= 100_000_000:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }


def standardized_proxy_anomalies(operator: dict[str, np.ndarray], sample: np.ndarray) -> tuple[np.ndarray, int]:
    hxb = operator["hxb"]
    variance = operator["variance"]
    active = np.isfinite(variance) & (variance > 0) & np.all(np.isfinite(hxb[:, sample]), axis=1)
    modeled = hxb[active][:, sample]
    anomaly = modeled - np.mean(modeled, axis=1, keepdims=True)
    return anomaly / np.sqrt(variance[active])[:, None], int(active.sum())


def participation_rank(energy: np.ndarray) -> float:
    normalized = energy / np.sum(energy)
    return float(1.0 / np.sum(normalized**2))


def rank_at(energy: np.ndarray, level: float) -> int:
    cumulative = np.cumsum(energy / np.sum(energy))
    return int(np.searchsorted(cumulative, level, side="left") + 1)


def main() -> int:
    root = common.project_root()
    driver = Path(__file__).resolve().parent
    validation = root / "validation_data/results/paleo_surface_validation_v0"
    output = validation / "climate_reconstruction_improvement_loop_v2" / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round163_temperature_psm_mode_observability_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")

    fold_root = validation / "temperature_fold_native_oenkf_increment_cv_v5_regem_R"
    loop = validation / "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    manifest = json.loads((fold_root / "source_hash_manifest.json").read_text(encoding="utf-8"))
    recovered = manifest["sources_before"]
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "round161_driver": driver / "run_round161_temperature_perfect_model_twin_v2.py",
        "round162_driver": driver / "run_round162_temperature_twin_information_geometry_v1.py",
        "fold_manifest": fold_root / "source_hash_manifest.json",
        "operator6": fold_root / "proxy6_fold_native_operators_500.mat",
        "operator9": fold_root / "proxy9_fold_native_operators_500.mat",
        "predictor6": fold_root / "proxy6_archived_model_predictors_848.mat",
        "predictor9": fold_root / "proxy9_archived_model_predictors_848.mat",
        "allowed6": loop / "proxy6_allowed_1940_2000_official_baseline_v2.mat",
        "allowed9": loop / "proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "prior": Path(recovered["prior"]["path"]),
        "indices": Path(recovered["indices"]["path"]),
        "native_grid": fold_root / "heldout_oof_fields.npz",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {
        name: fingerprint(path, recovered["prior"].get("sha256") if name == "prior" else None)
        for name, path in paths.items()
    }

    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        pools = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    sample = pools[0]
    complement = np.setdiff1d(np.arange(prior.shape[1]), sample)
    truth_members = complement[np.linspace(0, len(complement) - 1, round161.TRUTH_COUNT, dtype=int)]
    operators = {
        network: round161.load_full_operator(
            paths[f"operator{network}"], paths[f"predictor{network}"], paths[f"allowed{network}"]
        )
        for network in round161.NETWORKS
    }
    if any(not np.array_equal(operator["sample_indices"], sample) for operator in operators.values()):
        raise RuntimeError("operator-sample-index-closure")

    with np.load(paths["native_grid"]) as native:
        grid_lat = np.asarray(native["grid_lat"], dtype=float)
        grid_lon = np.asarray(native["grid_lon"], dtype=float)
        mask = np.asarray(native["hma_mask"], dtype=bool)
    weights = common.region_weights(grid_lat, grid_lon, mask)["hma"]
    background = np.mean(prior[:, sample], axis=1)
    anomaly = prior[:, sample] - background[:, None]
    state_basis = round162.weighted_basis(anomaly, weights)
    selected = state_basis["selected"]
    root_weight = state_basis["root_weight"]
    weighted_anomaly = root_weight[:, None] * anomaly[selected]
    u, singular, vt = np.linalg.svd(weighted_anomaly, full_matrices=False)
    rank = int(state_basis["numerical_rank"])
    u = u[:, :rank]
    singular = singular[:rank]
    vt = vt[:rank]
    prior_energy = singular**2
    prior_fraction = prior_energy / np.sum(prior_energy)

    truth_energy = []
    for truth_member in truth_members:
        error = root_weight * (prior[selected, truth_member] - background[selected])
        coefficients = u.T @ error
        truth_energy.append(coefficients**2 / np.dot(error, error))
    truth_energy = np.asarray(truth_energy)
    median_truth_fraction = np.median(truth_energy, axis=0)

    proxy_matrices = {}
    proxy_counts = {}
    for network in round161.NETWORKS:
        matrix, count = standardized_proxy_anomalies(operators[network], sample)
        proxy_matrices[str(network)] = matrix
        proxy_counts[str(network)] = count
    proxy_matrices["stacked_6_9"] = np.concatenate([proxy_matrices["6"], proxy_matrices["9"]], axis=0)
    proxy_counts["stacked_6_9"] = proxy_counts["6"] + proxy_counts["9"]

    mode_rows = []
    spectrum_rows = []
    member_count = len(sample)
    for network, proxy_anomaly in proxy_matrices.items():
        proxy_singular = np.linalg.svd(proxy_anomaly, compute_uv=False)
        proxy_energy = proxy_singular**2
        system = (member_count - 1.0) * np.eye(member_count) + proxy_anomaly.T @ proxy_anomaly
        factor = scipy.linalg.cho_factor(system, lower=True, check_finite=False)
        solved = scipy.linalg.cho_solve(factor, vt.T, check_finite=False)
        posterior_ratio = (member_count - 1.0) * np.sum(vt.T * solved, axis=0)
        variance_reduction = np.clip(1.0 - posterior_ratio, 0.0, 1.0)
        projected_proxy = proxy_anomaly @ vt.T
        marginal_information = np.sum(projected_proxy**2, axis=0) / (member_count - 1.0)
        coupling_energy = prior_energy * np.sum(projected_proxy**2, axis=0) / (member_count - 1.0) ** 2
        coupling_fraction = coupling_energy / np.sum(coupling_energy)
        for index in range(rank):
            mode_rows.append({
                "network": network,
                "mode": index + 1,
                "prior_variance_fraction": prior_fraction[index],
                "prior_cumulative_variance": np.sum(prior_fraction[: index + 1]),
                "median_heldout_truth_error_fraction": median_truth_fraction[index],
                "marginal_proxy_information_ratio": marginal_information[index],
                "exact_posterior_variance_reduction": variance_reduction[index],
                "cross_covariance_coupling_fraction": coupling_fraction[index],
                "cross_covariance_cumulative_fraction": np.sum(coupling_fraction[: index + 1]),
            })
        spectrum_rows.append({
            "network": network,
            "active_proxy_count": proxy_counts[network],
            "proxy_anomaly_effective_rank": participation_rank(proxy_energy),
            "proxy_anomaly_rank_at_90pct": rank_at(proxy_energy, 0.90),
            "proxy_anomaly_rank_at_99pct": rank_at(proxy_energy, 0.99),
            "cross_covariance_coupling_effective_rank": participation_rank(coupling_energy),
            "cross_covariance_coupling_rank_at_90pct": rank_at(coupling_energy, 0.90),
            "cross_covariance_coupling_rank_at_99pct": rank_at(coupling_energy, 0.99),
            "prior_modes_for_80pct_variance": rank_at(prior_energy, 0.80),
            "median_variance_reduction_first_prior_80pct_modes": float(
                np.median(variance_reduction[: rank_at(prior_energy, 0.80)])
            ),
        })

    mode_frame = pd.DataFrame(mode_rows)
    spectrum_frame = pd.DataFrame(spectrum_rows)
    mode_frame.to_csv(output / "mode_observability.csv", index=False)
    spectrum_frame.to_csv(output / "network_spectrum_summary.csv", index=False)
    primary = mode_frame[mode_frame.network == "stacked_6_9"].copy()
    first_mode_coupling = float(primary.iloc[0].cross_covariance_coupling_fraction)
    first_mode_prior = float(primary.iloc[0].prior_variance_fraction)
    modes_80 = rank_at(prior_energy, 0.80)
    weak_secondary = primary[(primary["mode"] >= 2) & (primary["mode"] <= modes_80)]
    secondary_reduction = float(weak_secondary.exact_posterior_variance_reduction.median())
    common_mode_collapse = bool(first_mode_coupling >= 0.75 and first_mode_coupling > first_mode_prior + 0.20)
    secondary_modes_weak = bool(secondary_reduction < 0.50)
    if common_mode_collapse and secondary_modes_weak:
        decision = "operator_geometry_overconcentrated_on_leading_common_mode"
    elif secondary_modes_weak:
        decision = "secondary_prior_modes_underobserved"
    else:
        decision = "mode_observability_not_primary_limit"

    fig = plt.figure(figsize=(14.5, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    full_selected = np.flatnonzero(selected)
    amplitude = np.nanmax(np.abs(u[:, :4] / root_weight[:, None]))
    for index, axis in enumerate(axes):
        state = np.full(prior.shape[0], np.nan)
        state[full_selected] = u[:, index] / root_weight
        cube = common.state_to_cube(state[:, None], len(grid_lat), len(grid_lon))[0]
        image = axis.imshow(
            cube,
            origin="lower",
            extent=[grid_lon.min(), grid_lon.max(), grid_lat.min(), grid_lat.max()],
            cmap="RdBu_r",
            vmin=-amplitude,
            vmax=amplitude,
            aspect="auto",
        )
        axis.set_title(f"Prior mode {index + 1} ({prior_fraction[index] * 100:.1f}%)")
        axis.set_xlabel("longitude")
        if index == 0:
            axis.set_ylabel("latitude")
    fig.colorbar(image, ax=axes, shrink=0.72, label="weighted-SVD spatial loading")
    line_axis = fig.add_subplot(grid[1, :2])
    show = primary[primary["mode"] <= 20]
    line_axis.plot(show["mode"], show.prior_variance_fraction, marker="o", label="prior variance")
    line_axis.plot(show["mode"], show.median_heldout_truth_error_fraction, marker="s", label="held-out truth error")
    line_axis.plot(show["mode"], show.cross_covariance_coupling_fraction, marker="^", label="PSM coupling")
    line_axis.set(title="Energy by prior mode", xlabel="prior mode", ylabel="fraction")
    line_axis.legend(frameon=False)
    reduction_axis = fig.add_subplot(grid[1, 2:])
    for network, color in (("6", "#0072B2"), ("9", "#D55E00"), ("stacked_6_9", "#009E73")):
        current = mode_frame[(mode_frame.network == network) & (mode_frame["mode"] <= 20)]
        reduction_axis.plot(current["mode"], current.exact_posterior_variance_reduction, marker="o", label=network, color=color)
    reduction_axis.set(title="Exact posterior variance reduction", xlabel="prior mode", ylabel="fraction reduced", ylim=(0, 1.02))
    reduction_axis.legend(frameon=False, title="network")
    for axis in (line_axis, reduction_axis):
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "mode_observability.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path, before[name].get("sha256")) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "prior_hma_numerical_rank": rank,
        "prior_hma_effective_rank": float(state_basis["effective_rank"]),
        "prior_modes_for_80pct_variance": modes_80,
        "stacked_first_mode_prior_variance_fraction": first_mode_prior,
        "stacked_first_mode_cross_covariance_coupling_fraction": first_mode_coupling,
        "stacked_secondary_modes_median_exact_variance_reduction": secondary_reduction,
        "classification": {
            "common_mode_collapse": common_mode_collapse,
            "secondary_modes_weak": secondary_modes_weak,
            "decision": decision,
        },
        "network_summaries": spectrum_frame.to_dict(orient="records"),
        "actual_proxy_observations_used": False,
        "instrumental_values_used": False,
        "locked_1921_1939_accessed": False,
        "jionglacuo_opened": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8"
    )
    lines = [
        "# Round 163 Temperature PSM Mode Observability",
        "",
        f"- Prior mode 1 holds `{first_mode_prior * 100:.2f}%` of HMA prior variance but `{first_mode_coupling * 100:.2f}%` of stacked state/PSM coupling energy.",
        f"- `{modes_80}` prior modes are needed for 80% of state variance; modes 2-{modes_80} have median exact posterior variance reduction `{secondary_reduction:.3f}`.",
        f"- Decision: `{decision}`.",
        "- This mode audit uses no real observations and does not authorize production reconstruction or bias correction.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
