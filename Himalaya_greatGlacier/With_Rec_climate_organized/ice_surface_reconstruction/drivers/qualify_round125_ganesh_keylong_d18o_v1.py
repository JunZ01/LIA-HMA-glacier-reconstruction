"""Qualify unopened Ganesh and Keylong tree-ring d18O records with frozen OOF tests."""

from __future__ import annotations

import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SITES = {
    "ganesh": {"file": "ganesh2018d18o-noaa.txt", "latitude": 28.183, "longitude": 85.183},
    "keylong": {"file": "keylong2020d18o-noaa.txt", "latitude": 32.75, "longitude": 76.62},
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-spec-failed:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_noaa_candidate(path: Path, years: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    wanted = set(int(year) for year in years)
    values: dict[int, float] = {}
    all_years: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                year = int(round(float(parts[0])))
            except ValueError:
                continue
            all_years.append(year)
            if 1921 <= year <= 1939:
                continue
            if year in wanted:
                values[year] = float(parts[1])
    missing = sorted(wanted.difference(values))
    if missing:
        raise RuntimeError(f"missing-modern-years:{path.name}:{missing}")
    result = np.asarray([values[int(year)] for year in years], dtype=float)
    if not np.all(np.isfinite(result)) or np.std(result, ddof=1) <= 1e-12:
        raise RuntimeError(f"invalid-modern-candidate:{path.name}")
    return result, {
        "first_year": int(min(all_years)),
        "last_year": int(max(all_years)),
        "year_count": int(len(all_years)),
        "modern_value_count": int(len(result)),
        "locked_1921_1939_values_decoded": 0,
    }


def main() -> int:
    driver = Path(__file__).resolve().parent
    r123 = load_module("round125_r123", driver / "qualify_round123_southeast_tibet_d18o_v1.py")
    root = r123.project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_ganesh_keylong_d18o_proxy_qualification_20260818_v1"
    )
    raw = output / "raw_unopened"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "ledger": root / (
            "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/"
            "round_108_proxy_metadata_reconciliation_v2/proxy_evidence_master_ledger_reconciled.csv"
        ),
        "temperature_fields": root / (
            "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/"
            "round_115_temperature_low_rank_oof_v2/temperature_low_rank_oof_fields.npz"
        ),
        "precipitation_fields": root / (
            "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/"
            "round_117_precipitation_documented_moisture_residual_oof_v1/"
            "precipitation_documented_moisture_residual_oof_fields.npz"
        ),
        "contract": driver / "round125_ganesh_keylong_d18o_proxy_qualification_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "round123_driver": driver / "qualify_round123_southeast_tibet_d18o_v1.py",
        "download_manifest": output / "download_manifest.json",
        **{name: raw / meta["file"] for name, meta in SITES.items()},
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    shutil.copy2(paths["contract"], output / "ROUND_CONTRACT.md")
    before = {name: r123.file_state(path) for name, path in paths.items()}

    temperature = np.load(paths["temperature_fields"])
    precipitation = np.load(paths["precipitation_fields"])
    if not np.array_equal(temperature["years"], r123.YEARS) or not np.array_equal(
        precipitation["years"], r123.YEARS
    ):
        raise RuntimeError("modern-year-contract-failed")
    grid_lat = np.asarray(temperature["grid_lat"], dtype=float)
    grid_lon = np.asarray(temperature["grid_lon"], dtype=float)
    support = np.asarray(temperature["observed_support"], dtype=bool) & np.asarray(
        precipitation["observed_support"], dtype=bool
    )
    ledger = pd.read_csv(paths["ledger"], low_memory=False)
    coordinates = ledger[["latitude", "longitude"]].dropna().drop_duplicates()
    ledger_lat = coordinates.latitude.to_numpy(float)
    ledger_lon = coordinates.longitude.to_numpy(float)

    identity_rows = []
    fold_rows = []
    metric_rows = []
    curve_rows = []
    decisions = {}
    plot_payload = {}
    for site, metadata in SITES.items():
        proxy, audit = parse_noaa_candidate(paths[site], r123.YEARS)
        distances = r123.haversine_km(metadata["latitude"], metadata["longitude"], ledger_lat, ledger_lon)
        nearest = int(np.argmin(distances))
        state_index, cell_lat, cell_lon, cell_distance = r123.nearest_supported_index(
            metadata["latitude"], metadata["longitude"], grid_lat, grid_lon, support
        )
        temp, temp_products = r123.product_consensus(temperature, "observation_", state_index)
        precip, precip_products = r123.product_consensus(precipitation, "observation_", state_index)
        identity_rows.append(
            {
                "site": site,
                "latitude": metadata["latitude"],
                "longitude": metadata["longitude"],
                **audit,
                "nearest_existing_proxy_latitude": float(ledger_lat[nearest]),
                "nearest_existing_proxy_longitude": float(ledger_lon[nearest]),
                "nearest_existing_proxy_distance_km": float(distances[nearest]),
                "exact_spatial_duplicate_within_1km": bool(distances[nearest] < 1.0),
                "nearest_supported_grid_latitude": cell_lat,
                "nearest_supported_grid_longitude": cell_lon,
                "nearest_supported_grid_distance_km": cell_distance,
                "temperature_product_count": len(temp_products),
                "precipitation_product_count": len(precip_products),
            }
        )
        branch_predictors = {
            "temperature_only": temp[:, None],
            "precipitation_only": precip[:, None],
            "bivariate": np.column_stack([temp, precip]),
        }
        payload = {"proxy": proxy}
        passes = []
        for branch, predictors in branch_predictors.items():
            prediction, rows = r123.run_oof_branch(proxy, predictors, branch)
            frame = pd.DataFrame(rows)
            frame.insert(0, "site", site)
            fold_rows.extend(frame.to_dict("records"))
            metrics = r123.pooled_metrics(proxy, prediction, frame)
            passed = bool(
                metrics["pooled_oof_correlation"] > 0
                and metrics["pooled_oof_skill_vs_training_mean"] > 0
            )
            metric_rows.append({"site": site, "branch": branch, **metrics, "passes_preliminary_gate": passed})
            passes.append(passed)
            payload[branch] = prediction
            for year, observed, predicted in zip(r123.YEARS, proxy, prediction):
                curve_rows.append(
                    {"site": site, "year": int(year), "branch": branch, "proxy_d18o": observed, "oof_prediction": predicted}
                )
        decisions[site] = (
            "advance_to_incremental_assimilation_experiment"
            if distances[nearest] >= 1.0 and any(passes)
            else "retain_for_seasonal_psm_reassessment_only"
        )
        plot_payload[site] = payload

    identity = pd.DataFrame(identity_rows)
    folds = pd.DataFrame(fold_rows)
    metrics = pd.DataFrame(metric_rows)
    curves = pd.DataFrame(curve_rows)
    identity.to_csv(output / "candidate_identity_and_extraction.csv", index=False)
    folds.to_csv(output / "blocked_oof_fold_metrics.csv", index=False)
    metrics.to_csv(output / "blocked_oof_pooled_metrics.csv", index=False)
    curves.to_csv(output / "blocked_oof_curves.csv", index=False)

    colors = {"temperature_only": "#0072B2", "precipitation_only": "#009E73", "bivariate": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for column, site in enumerate(SITES):
        payload = plot_payload[site]
        axes[0, column].plot(r123.YEARS, payload["proxy"], color="#111111", linewidth=1.8, label="observed d18O")
        for branch, color in colors.items():
            axes[0, column].plot(r123.YEARS, payload[branch], color=color, linewidth=1.2, label=branch)
        axes[0, column].set_title(f"{site}: blocked OOF predictions")
        axes[0, column].set_ylabel("tree-ring d18O")
        axes[0, column].legend(frameon=False, fontsize=8)
        subset = metrics[metrics.site.eq(site)].set_index("branch")
        x = np.arange(3)
        axes[1, column].bar(x - 0.18, [subset.loc[b, "pooled_oof_correlation"] for b in colors], 0.36, label="OOF r")
        axes[1, column].bar(x + 0.18, [subset.loc[b, "pooled_oof_skill_vs_training_mean"] for b in colors], 0.36, label="OOF skill")
        axes[1, column].axhline(0, color="#555555", linewidth=0.8)
        axes[1, column].set_xticks(x, ["T", "P", "T+P"])
        axes[1, column].legend(frameon=False, fontsize=8)
    fig.savefig(output / "round125_ganesh_keylong_oof.png", dpi=220)
    plt.close(fig)

    after = {name: r123.file_state(path) for name, path in paths.items()}
    unchanged = bool(before == after)
    if not unchanged:
        raise RuntimeError("source-changed-during-round125")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2), encoding="utf-8"
    )
    summary = {
        "experiment_id": "round125_ganesh_keylong_d18o_proxy_qualification_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "locked_1921_1939_candidate_values_decoded": 0,
        "wache_excluded_due_to_boundary_incident": True,
        "external_validation_reserve_opened": False,
        "source_hashes_unchanged": unchanged,
        "decisions": decisions,
        "identity_and_extraction": identity.to_dict("records"),
        "pooled_metrics": metrics.to_dict("records"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Round 125 Ganesh and Keylong d18O Qualification",
        "",
        "- Locked 1921-1939 candidate values decoded: `0`.",
        "- Wache excluded: `true`.",
        f"- Source hashes unchanged: `{str(unchanged).lower()}`.",
        "",
    ]
    for site, decision in decisions.items():
        item = identity[identity.site.eq(site)].iloc[0]
        lines.extend([f"## {site}", "", f"- Decision: `{decision}`.", f"- Nearest existing proxy: `{item.nearest_existing_proxy_distance_km:.1f} km`."])
        for row in metrics[metrics.site.eq(site)].itertuples(index=False):
            lines.append(
                f"- `{row.branch}`: OOF r=`{row.pooled_oof_correlation:.3f}`, skill=`{row.pooled_oof_skill_vs_training_mean:.3f}`, gate=`{str(bool(row.passes_preliminary_gate)).lower()}`."
            )
        lines.append("")
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
