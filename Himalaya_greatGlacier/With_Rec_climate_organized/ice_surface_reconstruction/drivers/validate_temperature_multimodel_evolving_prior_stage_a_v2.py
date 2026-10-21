from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_temperature_four_prior_strategy_c_replay_1901_1920_v1 as replay


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "temperature_multimodel_evolving_prior_stage_a_v2"
STATE = RESULTS / (
    "climate_reconstruction_temperature_four_prior_strategy_c_replay_1901_1920_v1/"
    "four_prior_state_pools_1500_1848_v1.npz"
)
STATE_SUMMARY = STATE.parent / "state_pool_acquisition_summary.json"
METHOD = LOOP / "temperature_multimodel_evolving_prior_method_preflight_v3/METHOD_CONTRACT.md"
VALIDATION = LOOP / "temperature_multimodel_evolving_prior_method_preflight_v3/VALIDATION_CONTRACT.md"
V1_SUMMARY = LOOP / "temperature_multimodel_evolving_prior_stage_a_v1/summary.json"

MODELS = ("cesm", "mode", "mpi", "miroc")


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


def max_abs_slope(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=float)
    x = np.arange(len(data), dtype=float)
    x -= x.mean()
    centered = data - data.mean(axis=0, keepdims=True)
    slope = (x[:, None] * centered).sum(axis=0) / np.sum(x * x)
    return float(np.max(np.abs(slope)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    sources = [
        STATE,
        STATE_SUMMARY,
        METHOD,
        VALIDATION,
        V1_SUMMARY,
        ROOT / "LIA_PROJECT_FACTS.md",
        Path(__file__),
    ]
    before = {str(path): sha256(path) for path in sources}

    states = np.load(STATE, allow_pickle=False)
    years = np.asarray(states["years"], dtype=int)
    native_lat = np.asarray(states["native_lat"], dtype=float)
    native_lon = np.asarray(states["native_lon"], dtype=float)
    if not np.array_equal(years, np.arange(1500, 1849)):
        raise RuntimeError("state-year-axis")
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("locked-years-in-state")

    cesm_tas = np.asarray(states["cesm_tas_state"], dtype=float)
    cesm_pr = np.asarray(states["cesm_pr_state"], dtype=float)
    tas_std = cesm_tas.std(axis=0)
    pr_std = cesm_pr.std(axis=0)
    common_valid = (tas_std > 1e-8) & (pr_std > 1e-8)
    excluded = np.flatnonzero(~common_valid)
    expected_excluded = np.asarray([0, 14, 28], dtype=int)
    if not np.array_equal(excluded, expected_excluded):
        raise RuntimeError(f"unexpected-cesm-constant-columns:{excluded.tolist()}")

    excluded_rows = []
    for index in excluded:
        lat_index = int(index % len(native_lat))
        lon_index = int(index // len(native_lat))
        excluded_rows.append(
            {
                "state_index": int(index),
                "latitude": float(native_lat[lat_index]),
                "longitude": float(native_lon[lon_index]),
                "cesm_tas_min": float(cesm_tas[:, index].min()),
                "cesm_tas_max": float(cesm_tas[:, index].max()),
                "cesm_pr_min": float(cesm_pr[:, index].min()),
                "cesm_pr_max": float(cesm_pr[:, index].max()),
                "reason": "constant_zero_boundary_placeholder_in_both_cesm_fields",
            }
        )
    excluded_table = pd.DataFrame(excluded_rows)

    proxy_longitude_min = min(
        float(np.asarray(states["proxy6_locations"])[:, 1].min()),
        float(np.asarray(states["proxy9_locations"])[:, 1].min()),
    )
    excluded_outside_proxy_longitude = bool((excluded_table["longitude"] < proxy_longitude_min).all())

    state_rows = []
    for model in MODELS:
        tas = np.asarray(states[f"{model}_tas_state"], dtype=float)
        pr = np.asarray(states[f"{model}_pr_state"], dtype=float)
        if model != "cesm":
            tas = tas - 273.15
        if tas.shape != (349, 336) or pr.shape != (349, 336):
            raise RuntimeError(f"state-shape:{model}:{tas.shape}:{pr.shape}")
        if not np.all(np.isfinite(tas)) or not np.all(np.isfinite(pr)):
            raise RuntimeError(f"nonfinite-state:{model}")
        if float(tas.min()) < -80 or float(tas.max()) > 60 or float(pr.min()) < 0 or float(pr.max()) > 30:
            raise RuntimeError(f"physical-range:{model}")

        tas_retained = tas[:, common_valid]
        pr_retained = pr[:, common_valid]
        tas_z = detrend(tas_retained)
        pr_z = detrend(pr_retained)
        state_rows.append(
            {
                "model": model,
                "state_count": int(len(years)),
                "source_state_cell_count": int(tas.shape[1]),
                "retained_common_state_cell_count": int(tas_retained.shape[1]),
                "excluded_common_state_cell_count": int((~common_valid).sum()),
                "tas_conversion": "none_already_degC" if model == "cesm" else "kelvin_minus_273p15",
                "tas_retained_min_degC": float(tas_retained.min()),
                "tas_retained_max_degC": float(tas_retained.max()),
                "pr_retained_min_mm_day": float(pr_retained.min()),
                "pr_retained_max_mm_day": float(pr_retained.max()),
                "tas_detrended_max_abs_mean": float(np.max(np.abs(tas_z.mean(axis=0)))),
                "pr_detrended_max_abs_mean": float(np.max(np.abs(pr_z.mean(axis=0)))),
                "tas_detrended_max_abs_slope_per_year": max_abs_slope(tas_z),
                "pr_detrended_max_abs_slope_per_year": max_abs_slope(pr_z),
                "tas_nonzero_variance_retained_fraction": float(np.mean(tas_z.std(axis=0) > 1e-8)),
                "pr_nonzero_variance_retained_fraction": float(np.mean(pr_z.std(axis=0) > 1e-8)),
            }
        )
    state_table = pd.DataFrame(state_rows)

    operator_rows = []
    maximum_cesm_closure = 0.0
    for network in (6, 9):
        proxy = replay.load_proxy(network, years - 1001)
        closures = [float(record.get("closure", 0.0)) for record in proxy["records"] if record["has_components"]]
        maximum_cesm_closure = max(maximum_cesm_closure, max(closures) if closures else 0.0)
        for model in MODELS:
            hxb = replay.rebuild_hxb(model, network, proxy, states)
            if hxb.shape != (len(proxy["records"]), len(years)) or not np.all(np.isfinite(hxb)):
                raise RuntimeError(f"hxb-shape-finite:{model}:{network}:{hxb.shape}")
            operator_rows.append(
                {
                    "model": model,
                    "proxy_network": network,
                    "proxy_count": int(hxb.shape[0]),
                    "state_count": int(hxb.shape[1]),
                    "component_free_proxy_count": int(np.sum(~proxy["complete"])),
                    "finite_fraction": float(np.mean(np.isfinite(hxb))),
                    "nonzero_operator_variance_fraction": float(np.mean(np.var(hxb, axis=1) > 1e-12)),
                    "operator_min": float(hxb.min()),
                    "operator_max": float(hxb.max()),
                    "archived_component_closure_max": max(closures) if model == "cesm" and closures else np.nan,
                }
            )
    operator_table = pd.DataFrame(operator_rows)

    gates = {
        "four_models_present": set(state_table["model"]) == set(MODELS),
        "equal_349_paired_states": bool((state_table["state_count"] == 349).all()),
        "common_mask_retains_exactly_333_cells": bool((state_table["retained_common_state_cell_count"] == 333).all()),
        "excluded_exactly_three_cesm_zero_boundary_placeholders": len(excluded_table) == 3,
        "excluded_cells_west_of_all_proxy_sites": excluded_outside_proxy_longitude,
        "tas_units_explicitly_degC": True,
        "precipitation_units_mm_day": True,
        "all_retained_state_values_finite_and_physical": True,
        "retained_detrended_state_means_at_most_1e_12": bool(
            state_table[["tas_detrended_max_abs_mean", "pr_detrended_max_abs_mean"]].to_numpy().max() <= 1e-12
        ),
        "retained_detrended_state_slopes_at_most_1e_12_per_year": bool(
            state_table[["tas_detrended_max_abs_slope_per_year", "pr_detrended_max_abs_slope_per_year"]].to_numpy().max() <= 1e-12
        ),
        "all_retained_state_cells_have_nonzero_variance": bool(
            (state_table["tas_nonzero_variance_retained_fraction"] == 1).all()
            and (state_table["pr_nonzero_variance_retained_fraction"] == 1).all()
        ),
        "archived_cesm_hxb_closure_at_most_1e_12": maximum_cesm_closure <= 1e-12,
        "all_rebuilt_operators_finite": bool((operator_table["finite_fraction"] == 1).all()),
        "all_rebuilt_operator_rows_have_variance": bool((operator_table["nonzero_operator_variance_fraction"] == 1).all()),
        "locked_period_absent": True,
    }
    stage_a_pass = all(gates.values())

    state_table.to_csv(OUT / "paired_state_common_mask_audit.csv", index=False)
    excluded_table.to_csv(OUT / "excluded_cesm_boundary_state_cells.csv", index=False)
    operator_table.to_csv(OUT / "model_specific_psm_operator_audit.csv", index=False)
    np.save(OUT / "common_valid_state_mask_336.npy", common_valid)

    summary = {
        "experiment": OUT.name,
        "status": "complete",
        "supersedes": "temperature_multimodel_evolving_prior_stage_a_v1",
        "v1_failure_resolution": "remove_exactly_three_constant-zero CESM boundary placeholders with one common four-model state mask",
        "paired_external_prior_ready": bool(stage_a_pass),
        "stage_a_pass": bool(stage_a_pass),
        "state_years": [int(years[0]), int(years[-1])],
        "equal_state_count_per_model": int(len(years)),
        "source_state_cell_count": 336,
        "retained_common_state_cell_count": int(common_valid.sum()),
        "excluded_state_indices": excluded.tolist(),
        "minimum_proxy_longitude": proxy_longitude_min,
        "maximum_archived_cesm_hxb_closure": maximum_cesm_closure,
        "gates": gates,
        "next_action": (
            "request_explicit_user_authorization_for_stage_b_biased_model_pseudoproxy_enkf_only"
            if stage_a_pass
            else "repair_stage_a_before_any_enkf"
        ),
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "sealed_1921_1939_values_used": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {str(path): sha256(path) for path in sources}
    manifest["prohibited_source_accessed"] = False
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Multimodel evolving-prior Stage A v2",
        "",
        f"Stage A pass: `{stage_a_pass}`.",
        f"Paired external prior ready: `{stage_a_pass}`.",
        f"Common retained state cells: `{int(common_valid.sum())}/336`.",
        "Excluded CESM constant-zero boundary cells: `25N, 63E`; `25N, 65E`; `25N, 67E`.",
        f"The westernmost proxy site is `{proxy_longitude_min:.2f}E`.",
        f"Maximum archived CESM PSM closure: `{maximum_cesm_closure:.3e}`.",
        "",
    ]
    lines.extend([f"- `{name}`: `{value}`" for name, value in gates.items()])
    lines.extend(
        [
            "",
            "No EnKF was run. Passing Stage A authorizes only requesting Stage B pseudoproxy-EnKF permission.",
            "No production or OGGM file was changed; 1921-1939 was absent; 20CRv3 was not accessed.",
        ]
    )
    (OUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {str(path): sha256(path) for path in sources}
    if before != after:
        raise RuntimeError("source-changed")


if __name__ == "__main__":
    main()
