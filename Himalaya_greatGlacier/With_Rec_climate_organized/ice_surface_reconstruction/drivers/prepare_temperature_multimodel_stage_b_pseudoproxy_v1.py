from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io
from scipy.ndimage import uniform_filter1d

import run_temperature_four_prior_strategy_c_replay_1901_1920_v1 as replay


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_preregistration_v1"
STATE = RESULTS / (
    "climate_reconstruction_temperature_four_prior_strategy_c_replay_1901_1920_v1/"
    "four_prior_state_pools_1500_1848_v1.npz"
)
STAGE_A = LOOP / "temperature_multimodel_evolving_prior_stage_a_v2"
ASSIGNMENT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/"
    "hma_grid_gtng15_assignment.csv"
)
R_INPUTS = {
    6: RESULTS / (
        "enkf_reconstruction_source_recovery_audit_v2/"
        "replay_proxy6_enkf_exp1_matlab_v1/replay_input.mat"
    ),
    9: RESULTS / "climate_reconstruction_proxy9_enkf_exp1_replay_v1/replay_input.mat",
}

MODELS = ("cesm", "mode", "mpi", "miroc")
MODEL_LABELS = ("CESM", "ModE-Sim", "MPI", "MIROC")
POOL_YEARS = np.arange(1500, 1849, dtype=int)
EVALUATION_YEARS = np.arange(1515, 1834, dtype=int)
LOWPASS_YEARS = 31


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detrend(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    x = np.arange(len(data), dtype=float)
    x -= x.mean()
    centered = data - data.mean(axis=0, keepdims=True)
    slope = (x[:, None] * centered).sum(axis=0) / np.sum(x * x)
    return centered - x[:, None] * slope[None, :]


def lowpass(values: np.ndarray) -> np.ndarray:
    return uniform_filter1d(np.asarray(values, dtype=float), size=LOWPASS_YEARS, axis=0, mode="reflect")


def state_point_coordinates(native_lat: np.ndarray, native_lon: np.ndarray) -> np.ndarray:
    lat2, lon2 = np.meshgrid(native_lat, native_lon, indexing="ij")
    return np.column_stack([lat2.reshape(-1, order="F"), lon2.reshape(-1, order="F")])


def regional_weights(
    assignment: pd.DataFrame,
    state_points: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    regions = np.asarray(sorted(assignment["o2region"].unique()), dtype="U5")
    if len(regions) != 15:
        raise RuntimeError(f"gtng-region-count:{len(regions)}")
    weights = np.zeros((len(regions), int(valid_mask.sum())), dtype=float)
    retained_points = state_points[valid_mask]
    for cell_index, (latitude, longitude) in enumerate(retained_points):
        inside = assignment[
            (assignment["latitude"] >= latitude - 1.0)
            & (assignment["latitude"] < latitude + 1.0)
            & (assignment["longitude"] >= longitude - 1.0)
            & (assignment["longitude"] < longitude + 1.0)
        ]
        grouped = inside.groupby("o2region", sort=False)["area_weight_coslat"].sum()
        for region_index, region in enumerate(regions):
            weights[region_index, cell_index] = float(grouped.get(region, 0.0))
    if np.any(weights.sum(axis=1) <= 0):
        raise RuntimeError(f"empty-gtng-region:{regions[weights.sum(axis=1) <= 0].tolist()}")
    hma = weights.sum(axis=0)
    if hma.sum() <= 0:
        raise RuntimeError("empty-hma-weight")
    return regions, weights, hma


def load_complete_operator_components(network: int, state: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    path = replay.prior_tools.ARCHIVE / f"python_data/proxy_data_{network}.mat"
    pool_indices = np.arange(499, 848, dtype=int)
    coefficients = []
    locations = []
    archived_hxb = []
    record_indices = []
    with h5py.File(path, "r") as handle:
        references = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for record_index, reference in enumerate(references):
            proxy = handle[reference]
            b = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
            if b.size != 3 or "M_t" not in proxy or "M_p" not in proxy:
                continue
            mt = np.asarray(proxy["M_t"], dtype=float).reshape(-1, order="F")[pool_indices]
            mp = np.asarray(proxy["M_p"], dtype=float).reshape(-1, order="F")[pool_indices]
            hxb = np.asarray(proxy["HXb"], dtype=float).reshape(-1, order="F")[pool_indices]
            rebuilt = b[0] + b[1] * mt + b[2] * mp
            if float(np.max(np.abs(rebuilt - hxb))) > 1e-12:
                raise RuntimeError(f"cesm-operator-closure:proxy{network}:record{record_index + 1}")
            coefficients.append(b)
            locations.append(np.asarray(proxy["location"], dtype=float).reshape(-1, order="F"))
            archived_hxb.append(hxb)
            record_indices.append(record_index)

    b = np.asarray(coefficients, dtype=float)
    location = np.asarray(locations, dtype=float)
    cesm = np.asarray(archived_hxb, dtype=float).T
    unique_indices = np.asarray(state[f"proxy{network}_unique_point_indices"], dtype=int)[record_indices]
    operators = [cesm]
    for model in MODELS[1:]:
        tas = np.asarray(state[f"{model}_tas_proxy"], dtype=float)[:, unique_indices]
        pr = np.asarray(state[f"{model}_pr_proxy"], dtype=float)[:, unique_indices]
        operator = b[:, 0][None, :] + b[:, 1][None, :] * tas + b[:, 2][None, :] * pr
        operators.append(operator)
    operator = np.asarray(operators, dtype=float)
    operator -= operator.mean(axis=1, keepdims=True)

    replay_input = scipy.io.loadmat(R_INPUTS[network], variable_names=["R0v_filled"])
    residual = np.asarray(replay_input["R0v_filled"], dtype=float)[record_indices]
    residual_variance = np.nanvar(residual, axis=1, ddof=1)
    if not np.all(np.isfinite(residual_variance)) or np.any(residual_variance <= 0):
        raise RuntimeError(f"invalid-residual-variance:proxy{network}")
    return {
        "record_indices": np.asarray(record_indices, dtype=int),
        "coefficients": b,
        "locations": location,
        "operator_anomaly": operator,
        "residual_variance": residual_variance,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "state_pool": STATE,
        "stage_a_summary": STAGE_A / "summary.json",
        "stage_a_mask": STAGE_A / "common_valid_state_mask_336.npy",
        "gtng_assignment": ASSIGNMENT,
        "proxy6": replay.prior_tools.ARCHIVE / "python_data/proxy_data_6.mat",
        "proxy9": replay.prior_tools.ARCHIVE / "python_data/proxy_data_9.mat",
        "proxy6_R": R_INPUTS[6],
        "proxy9_R": R_INPUTS[9],
        "driver": Path(__file__),
    }
    before = {name: sha256(path) for name, path in sources.items()}

    stage_a = json.loads((STAGE_A / "summary.json").read_text(encoding="utf-8"))
    if not stage_a.get("stage_a_pass") or not stage_a.get("paired_external_prior_ready"):
        raise RuntimeError("stage-a-not-ready")

    state = np.load(STATE, allow_pickle=False)
    years = np.asarray(state["years"], dtype=int)
    if not np.array_equal(years, POOL_YEARS):
        raise RuntimeError("pool-year-axis")
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("locked-period-present")
    evaluation_indices = np.flatnonzero(np.isin(years, EVALUATION_YEARS))
    if not np.array_equal(years[evaluation_indices], EVALUATION_YEARS):
        raise RuntimeError("evaluation-year-axis")

    valid_mask = np.load(STAGE_A / "common_valid_state_mask_336.npy").astype(bool)
    if valid_mask.shape != (336,) or int(valid_mask.sum()) != 333:
        raise RuntimeError("stage-a-common-mask")
    native_lat = np.asarray(state["native_lat"], dtype=float)
    native_lon = np.asarray(state["native_lon"], dtype=float)
    points = state_point_coordinates(native_lat, native_lon)
    assignment = pd.read_csv(ASSIGNMENT)
    regions, region_weight, hma_weight = regional_weights(assignment, points, valid_mask)

    state_anomaly = []
    state_lowpass = []
    state_residual = []
    state_rows = []
    for model in MODELS:
        tas = np.asarray(state[f"{model}_tas_state"], dtype=float)
        if model != "cesm":
            tas -= 273.15
        tas = tas[:, valid_mask]
        anomaly = tas - tas.mean(axis=0, keepdims=True)
        forced_mean = lowpass(anomaly)
        residual = detrend(anomaly - forced_mean)
        state_anomaly.append(anomaly[evaluation_indices])
        state_lowpass.append(forced_mean[evaluation_indices])
        state_residual.append(residual)
        state_rows.append(
            {
                "model": model,
                "state_members": len(years),
                "evaluation_years": len(evaluation_indices),
                "retained_cells": int(valid_mask.sum()),
                "residual_max_abs_mean": float(np.max(np.abs(residual.mean(axis=0)))),
                "residual_min_std": float(np.min(residual.std(axis=0))),
                "full_anomaly_min": float(anomaly.min()),
                "full_anomaly_max": float(anomaly.max()),
            }
        )

    arrays: dict[str, np.ndarray] = {
        "model_keys": np.asarray(MODELS, dtype="U5"),
        "model_labels": np.asarray(MODEL_LABELS, dtype="U10"),
        "pool_years": years,
        "evaluation_years": EVALUATION_YEARS,
        "retained_state_indices": np.flatnonzero(valid_mask),
        "retained_state_latitude": points[valid_mask, 0],
        "retained_state_longitude": points[valid_mask, 1],
        "truth_temperature_anomaly": np.asarray(state_anomaly),
        "prior_temperature_forced_mean": np.asarray(state_lowpass),
        "prior_temperature_residual_library": np.asarray(state_residual),
        "gtng_region_codes": regions,
        "gtng_region_weights": region_weight,
        "hma_weights": hma_weight,
    }

    operator_rows = []
    for network in (6, 9):
        data = load_complete_operator_components(network, state)
        anomaly = data["operator_anomaly"]
        forced_mean = np.asarray([lowpass(model) for model in anomaly])
        residual = np.asarray([detrend(anomaly[index] - forced_mean[index]) for index in range(len(MODELS))])
        arrays[f"proxy{network}_record_indices_0based"] = data["record_indices"]
        arrays[f"proxy{network}_locations"] = data["locations"]
        arrays[f"proxy{network}_residual_variance"] = data["residual_variance"]
        arrays[f"proxy{network}_truth_operator_anomaly"] = anomaly[:, evaluation_indices]
        arrays[f"proxy{network}_prior_operator_forced_mean"] = forced_mean[:, evaluation_indices]
        arrays[f"proxy{network}_prior_operator_residual_library"] = residual
        operator_rows.append(
            {
                "proxy_network": network,
                "complete_operator_count": int(anomaly.shape[2]),
                "excluded_component_free_count": int((202 if network == 6 else 297) - anomaly.shape[2]),
                "R_min": float(data["residual_variance"].min()),
                "R_median": float(np.median(data["residual_variance"])),
                "R_max": float(data["residual_variance"].max()),
                "operator_all_finite": bool(np.all(np.isfinite(anomaly)) and np.all(np.isfinite(residual))),
                "residual_min_row_std": float(np.min(residual.std(axis=1))),
            }
        )

    np.savez_compressed(OUT / "stage_b_pseudoproxy_input_v1.npz", **arrays)
    pd.DataFrame(state_rows).to_csv(OUT / "state_preflight.csv", index=False)
    pd.DataFrame(operator_rows).to_csv(OUT / "operator_R_preflight.csv", index=False)
    pd.DataFrame(
        {
            "o2region": regions,
            "coarse_state_positive_cell_count": (region_weight > 0).sum(axis=1),
            "coarse_weight_sum": region_weight.sum(axis=1),
        }
    ).to_csv(OUT / "gtng15_coarse_weight_support.csv", index=False)

    contract = """# Stage B biased-model pseudoproxy contract v1

## Purpose

Test whether the proposed evolving multimodel prior recovers a withheld model's
known HMA temperature field more reliably than a single prior model. This is a
method validation, not a reconstruction and not evidence that any climate model
is observational truth.

## Leakage-free folds

- Truth models: CESM, ModE-Sim, MPI, and MIROC, withheld one at a time.
- The withheld truth model is absent from every candidate prior and comparator.
- Evaluation years are every year from 1515 through 1833. The 15-year margins
  remove the edge of the frozen 31-year centered low-pass.
- All 349 years from 1500 through 1848 form each nontruth residual library.
- Three CESM constant-zero southwest boundary placeholders identified in Stage A
  are removed from every model with the same 333-cell mask.

## Prior and operator

- Each model field is centered on its own 1500-1848 mean.
- The model-specific evolving mean is a 31-year centered low-pass.
- The member library is the centered and detrended residual about that low-pass.
- Temperature and precipitation predictors remain paired by model and year.
- Only archived bivariate PSM rows with recoverable temperature and precipitation
  components are used: 201 proxy6 rows and 290 proxy9 rows.
- Pseudoproxy observations are the withheld model's operator anomalies. The
  archived RegEM diagonal R is retained. No real proxy observation is used.
- Ordinary EnKF, inflation 1, and no localization are frozen.

## Estimates and comparators

- For each proxy network, assimilate the pseudoproxies separately with each of
  the three nontruth priors.
- Candidate: equal mean of the three nontruth posterior means, then equal mean
  of proxy6 and proxy9. No model or network weight is fitted.
- Single-model comparators: the corresponding equal proxy6/proxy9 posterior for
  each nontruth prior.
- Primary comparator: median single-model RMSE. This is defined for all four
  truth folds without allowing the truth model into its own prior.
- CESM comparator: reported only for the three non-CESM truth folds. No CESM
  control is fabricated for the CESM-truth fold.

## Frozen primary gates

1. Candidate HMA RMSE improves over the median nontruth single-model RMSE in at
   least three of four withheld-truth folds.
2. No withheld-truth HMA RMSE degrades by more than 5 percent relative to that
   fold's median single-model RMSE.
3. Across withheld truths, at least 10 of 15 GTN-G regions have lower median
   candidate RMSE than median single-model RMSE.
4. For the three non-CESM truth folds, Candidate improves over CESM in at least
   two and degrades none by more than 5 percent.

Network-specific RMSE, correlation, amplitude ratio, low-pass RMSE, residual
RMSE, leave-one-prior-out estimates, and fixed-seed RegEM-noise sensitivity are
reported but cannot change the primary gates. Noise sensitivity uses seed
20260827 and 50 realizations.

The previous requirement to compare all four truth folds with CESM is
superseded because it is incompatible with withholding CESM when CESM is truth.
Passing Stage B authorizes only Stage C on allowed real data. It does not
authorize production replacement, OGGM, or access to 1921-1939.
"""
    (OUT / "STAGE_B_CONTRACT.md").write_text(contract, encoding="utf-8")

    gates = {
        "stage_a_passed": True,
        "four_truth_models_present": np.asarray(state_anomaly).shape == (4, 319, 333),
        "all_319_internal_evaluation_years_present": np.array_equal(EVALUATION_YEARS, years[evaluation_indices]),
        "common_333_cell_mask_used": int(valid_mask.sum()) == 333,
        "all_15_gtng_regions_have_support": bool(np.all(region_weight.sum(axis=1) > 0)),
        "all_state_residual_cells_have_variance": bool(
            all(row["residual_min_std"] > 1e-8 for row in state_rows)
        ),
        "proxy6_complete_operator_count_201": operator_rows[0]["complete_operator_count"] == 201,
        "proxy9_complete_operator_count_290": operator_rows[1]["complete_operator_count"] == 290,
        "all_operator_values_finite_and_variable": bool(
            all(row["operator_all_finite"] and row["residual_min_row_std"] > 1e-8 for row in operator_rows)
        ),
        "all_R_positive_and_finite": bool(
            all(np.isfinite(row["R_min"]) and row["R_min"] > 0 for row in operator_rows)
        ),
        "locked_period_absent": True,
        "real_proxy_observations_used": False,
    }
    preparation_pass = all(value for key, value in gates.items() if key != "real_proxy_observations_used") and not gates["real_proxy_observations_used"]
    summary = {
        "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "preparation_pass": bool(preparation_pass),
        "method_conflict_resolved": "CESM cannot be both withheld truth and comparator; use leakage-free median-single-model primary comparator",
        "evaluation_years": [int(EVALUATION_YEARS[0]), int(EVALUATION_YEARS[-1])],
        "evaluation_year_count": len(EVALUATION_YEARS),
        "prior_member_count_per_model": len(POOL_YEARS),
        "retained_state_cell_count": int(valid_mask.sum()),
        "truth_models": list(MODEL_LABELS),
        "primary_candidate": "equal mean of three nontruth posterior means, equal mean of proxy6 and proxy9",
        "primary_comparator": "within-fold median RMSE of three nontruth single-model posterior estimates",
        "cesm_comparator_fold_count": 3,
        "gates": gates,
        "enkf_run": False,
        "production_modified": False,
        "oggm_run": False,
        "sealed_1921_1939_values_used": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
        "next_action": "await_explicit_authorization_then_execute_frozen_stage_b_ordinary_enkf",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {name: {"path": str(path), "sha256": sha256(path)} for name, path in sources.items()}
    manifest["prohibited_source_accessed"] = False
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    after = {name: sha256(path) for name, path in sources.items()}
    if before != after:
        raise RuntimeError("source-changed")
    if not preparation_pass:
        raise RuntimeError("stage-b-preparation-gate-failed")


if __name__ == "__main__":
    main()
