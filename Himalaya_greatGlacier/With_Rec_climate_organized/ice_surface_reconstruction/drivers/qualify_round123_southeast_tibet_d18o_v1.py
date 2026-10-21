"""Qualify two new southeast-Tibet tree-ring d18O proxies with frozen modern OOF tests."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1940, 2001, dtype=int)
BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
LOCKED_START = 1921
LOCKED_END = 1939
SITES = {
    "mimei": {"file": "mimei_tree_ring_d18o.txt", "latitude": 29.45, "longitude": 96.43},
    "xinpu": {"file": "xinpu_tree_ring_d18o.txt", "latitude": 30.42, "longitude": 95.07},
}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
    }


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius = 6371.0088
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(a))


def parse_candidate(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    values: dict[int, float] = {}
    all_years: list[int] = []
    locked_values_decoded = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                year = int(round(float(parts[0].strip())))
            except ValueError:
                continue
            all_years.append(year)
            if LOCKED_START <= year <= LOCKED_END:
                # The candidate value field is deliberately not converted or retained.
                continue
            if 1940 <= year <= 2000:
                values[year] = float(parts[2].strip())
    missing = sorted(set(YEARS).difference(values))
    if missing:
        raise RuntimeError(f"candidate-modern-years-missing:{path.name}:{missing}")
    result = np.asarray([values[int(year)] for year in YEARS], dtype=float)
    if not np.all(np.isfinite(result)) or np.std(result, ddof=1) <= 1e-12:
        raise RuntimeError(f"candidate-modern-values-invalid:{path.name}")
    return result, {
        "first_year": int(min(all_years)),
        "last_year": int(max(all_years)),
        "year_count": int(len(all_years)),
        "modern_value_count": int(len(result)),
        "locked_1921_1939_values_decoded": int(locked_values_decoded),
    }


def nearest_supported_index(
    latitude: float,
    longitude: float,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    support: np.ndarray,
) -> tuple[int, float, float, float]:
    state_lat = np.tile(grid_lat, len(grid_lon))
    state_lon = np.repeat(grid_lon, len(grid_lat))
    distances = haversine_km(latitude, longitude, state_lat, state_lon)
    distances[~support] = np.inf
    index = int(np.argmin(distances))
    if not np.isfinite(distances[index]):
        raise RuntimeError("no-supported-grid-cell")
    return index, float(state_lat[index]), float(state_lon[index]), float(distances[index])


def product_consensus(fields: Any, prefix: str, index: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    products = {
        key[len(prefix) :]: np.asarray(fields[key], dtype=float)[index]
        for key in fields.files
        if key.startswith(prefix)
    }
    if not products:
        raise RuntimeError(f"no-products:{prefix}")
    matrix = np.stack(list(products.values()))
    consensus = np.nanmean(matrix, axis=0)
    if not np.all(np.isfinite(consensus)):
        raise RuntimeError(f"nonfinite-consensus:{prefix}:{index}")
    return consensus, products


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def run_oof_branch(target: np.ndarray, predictors: np.ndarray, branch: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    prediction = np.full_like(target, np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    for fold, (start, end) in enumerate(BLOCKS, start=1):
        test = (YEARS >= start) & (YEARS <= end)
        train = ~test
        x_mean = np.mean(predictors[train], axis=0)
        x_scale = np.std(predictors[train], axis=0, ddof=1)
        if np.any(x_scale <= 1e-12):
            raise RuntimeError(f"constant-predictor:{branch}:{fold}")
        x_train = (predictors[train] - x_mean) / x_scale
        x_test = (predictors[test] - x_mean) / x_scale
        design_train = np.column_stack([np.ones(int(train.sum())), x_train])
        coefficients, _, _, _ = np.linalg.lstsq(design_train, target[train], rcond=None)
        prediction[test] = np.column_stack([np.ones(int(test.sum())), x_test]) @ coefficients
        baseline = np.full(int(test.sum()), np.mean(target[train]), dtype=float)
        sse = float(np.sum((prediction[test] - target[test]) ** 2))
        baseline_sse = float(np.sum((baseline - target[test]) ** 2))
        rows.append(
            {
                "branch": branch,
                "fold": fold,
                "test_period": f"{start}-{end}",
                "test_count": int(test.sum()),
                "test_correlation": correlation(prediction[test], target[test]),
                "test_skill_vs_training_mean": float(1.0 - sse / baseline_sse),
                "intercept": float(coefficients[0]),
                "temperature_coefficient_standardized": float(coefficients[1])
                if branch in {"temperature_only", "bivariate"}
                else float("nan"),
                "precipitation_coefficient_standardized": float(coefficients[-1])
                if branch in {"precipitation_only", "bivariate"}
                else float("nan"),
            }
        )
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"oof-stitch-failed:{branch}")
    return prediction, rows


def pooled_metrics(target: np.ndarray, prediction: np.ndarray, folds: pd.DataFrame) -> dict[str, float]:
    baseline_prediction = np.empty_like(target)
    for start, end in BLOCKS:
        test = (YEARS >= start) & (YEARS <= end)
        baseline_prediction[test] = np.mean(target[~test])
    return {
        "pooled_oof_correlation": correlation(prediction, target),
        "pooled_oof_skill_vs_training_mean": float(
            1.0 - np.sum((prediction - target) ** 2) / np.sum((baseline_prediction - target) ** 2)
        ),
        "median_fold_correlation": float(np.nanmedian(folds.test_correlation)),
        "median_fold_skill": float(np.nanmedian(folds.test_skill_vs_training_mean)),
        "positive_fold_skill_fraction": float(np.mean(folds.test_skill_vs_training_mean > 0)),
    }


def main() -> int:
    root = project_root()
    driver = Path(__file__).resolve().parent
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_southeast_tibet_d18o_proxy_qualification_20260818_v1"
    )
    raw = output / "raw_unopened"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "ledger": root / (
            "validation_data/results/paleo_surface_validation_v0/"
            "climate_reconstruction_improvement_loop_v2/round_108_proxy_metadata_reconciliation_v2/"
            "proxy_evidence_master_ledger_reconciled.csv"
        ),
        "temperature_fields": root / (
            "validation_data/results/paleo_surface_validation_v0/"
            "climate_reconstruction_improvement_loop_v2/round_115_temperature_low_rank_oof_v2/"
            "temperature_low_rank_oof_fields.npz"
        ),
        "precipitation_fields": root / (
            "validation_data/results/paleo_surface_validation_v0/"
            "climate_reconstruction_improvement_loop_v2/"
            "round_117_precipitation_documented_moisture_residual_oof_v1/"
            "precipitation_documented_moisture_residual_oof_fields.npz"
        ),
        "contract": driver / "round123_southeast_tibet_d18o_proxy_qualification_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "download_manifest": output / "download_manifest.json",
        **{name: raw / metadata["file"] for name, metadata in SITES.items()},
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    shutil.copy2(paths["contract"], output / "ROUND_CONTRACT.md")
    before = {name: file_state(path) for name, path in paths.items()}

    temperature = np.load(paths["temperature_fields"])
    precipitation = np.load(paths["precipitation_fields"])
    if not np.array_equal(temperature["years"], YEARS) or not np.array_equal(precipitation["years"], YEARS):
        raise RuntimeError("modern-year-contract-failed")
    grid_lat = np.asarray(temperature["grid_lat"], dtype=float)
    grid_lon = np.asarray(temperature["grid_lon"], dtype=float)
    if not np.array_equal(grid_lat, precipitation["grid_lat"]) or not np.array_equal(
        grid_lon, precipitation["grid_lon"]
    ):
        raise RuntimeError("temperature-precipitation-grid-mismatch")
    support = np.asarray(temperature["observed_support"], dtype=bool) & np.asarray(
        precipitation["observed_support"], dtype=bool
    )

    ledger = pd.read_csv(paths["ledger"], low_memory=False)
    unique_coordinates = ledger[["latitude", "longitude"]].dropna().drop_duplicates()
    ledger_lat = unique_coordinates.latitude.to_numpy(float)
    ledger_lon = unique_coordinates.longitude.to_numpy(float)

    extraction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    decisions: dict[str, str] = {}
    plot_payload: dict[str, dict[str, np.ndarray]] = {}

    for site, metadata in SITES.items():
        proxy, extraction = parse_candidate(paths[site])
        distances = haversine_km(
            metadata["latitude"], metadata["longitude"], ledger_lat, ledger_lon
        )
        nearest_ledger = int(np.argmin(distances))
        state_index, cell_lat, cell_lon, cell_distance = nearest_supported_index(
            metadata["latitude"], metadata["longitude"], grid_lat, grid_lon, support
        )
        temp_consensus, temp_products = product_consensus(temperature, "observation_", state_index)
        precip_consensus, precip_products = product_consensus(precipitation, "observation_", state_index)
        extraction_rows.append(
            {
                "site": site,
                "latitude": metadata["latitude"],
                "longitude": metadata["longitude"],
                **extraction,
                "nearest_existing_proxy_latitude": float(ledger_lat[nearest_ledger]),
                "nearest_existing_proxy_longitude": float(ledger_lon[nearest_ledger]),
                "nearest_existing_proxy_distance_km": float(distances[nearest_ledger]),
                "exact_spatial_duplicate_within_1km": bool(distances[nearest_ledger] < 1.0),
                "nearest_supported_grid_latitude": cell_lat,
                "nearest_supported_grid_longitude": cell_lon,
                "nearest_supported_grid_distance_km": cell_distance,
                "temperature_product_count": len(temp_products),
                "precipitation_product_count": len(precip_products),
            }
        )
        branches = {
            "temperature_only": temp_consensus[:, None],
            "precipitation_only": precip_consensus[:, None],
            "bivariate": np.column_stack([temp_consensus, precip_consensus]),
        }
        site_payload = {"proxy": proxy, "temperature": temp_consensus, "precipitation": precip_consensus}
        branch_passes = []
        for branch, predictors in branches.items():
            prediction, current_folds = run_oof_branch(proxy, predictors, branch)
            current_frame = pd.DataFrame(current_folds)
            current_frame.insert(0, "site", site)
            fold_rows.extend(current_frame.to_dict("records"))
            metrics = pooled_metrics(proxy, prediction, current_frame)
            advances = bool(
                metrics["pooled_oof_correlation"] > 0
                and metrics["pooled_oof_skill_vs_training_mean"] > 0
            )
            metric_rows.append({"site": site, "branch": branch, **metrics, "passes_preliminary_gate": advances})
            branch_passes.append(advances)
            site_payload[f"prediction_{branch}"] = prediction
            for year, observed, predicted in zip(YEARS, proxy, prediction):
                curve_rows.append(
                    {
                        "site": site,
                        "year": int(year),
                        "branch": branch,
                        "proxy_d18o": float(observed),
                        "oof_prediction": float(predicted),
                    }
                )
        spatially_new = bool(distances[nearest_ledger] >= 1.0)
        decisions[site] = (
            "advance_to_incremental_assimilation_experiment"
            if spatially_new and any(branch_passes)
            else "retain_for_seasonal_psm_reassessment_only"
        )
        plot_payload[site] = site_payload

    extraction_frame = pd.DataFrame(extraction_rows)
    folds_frame = pd.DataFrame(fold_rows)
    metrics_frame = pd.DataFrame(metric_rows)
    curves_frame = pd.DataFrame(curve_rows)
    extraction_frame.to_csv(output / "candidate_identity_and_extraction.csv", index=False)
    folds_frame.to_csv(output / "blocked_oof_fold_metrics.csv", index=False)
    metrics_frame.to_csv(output / "blocked_oof_pooled_metrics.csv", index=False)
    curves_frame.to_csv(output / "blocked_oof_curves.csv", index=False)

    colors = {
        "temperature_only": "#0072B2",
        "precipitation_only": "#009E73",
        "bivariate": "#D55E00",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for column, site in enumerate(SITES):
        payload = plot_payload[site]
        axes[0, column].plot(YEARS, payload["proxy"], color="#111111", linewidth=1.8, label="observed d18O")
        for branch in colors:
            axes[0, column].plot(
                YEARS,
                payload[f"prediction_{branch}"],
                color=colors[branch],
                linewidth=1.25,
                label=branch,
            )
        for boundary in [1952, 1964, 1976, 1988]:
            axes[0, column].axvline(boundary - 0.5, color="#aaaaaa", linewidth=0.6)
        axes[0, column].set_title(f"{site}: blocked OOF predictions")
        axes[0, column].set_ylabel("tree-ring d18O")
        axes[0, column].legend(frameon=False, fontsize=8)
        subset = metrics_frame[metrics_frame.site.eq(site)].set_index("branch")
        x = np.arange(3)
        axes[1, column].bar(
            x - 0.18,
            [subset.loc[name, "pooled_oof_correlation"] for name in colors],
            width=0.36,
            color="#4C78A8",
            label="OOF correlation",
        )
        axes[1, column].bar(
            x + 0.18,
            [subset.loc[name, "pooled_oof_skill_vs_training_mean"] for name in colors],
            width=0.36,
            color="#F58518",
            label="OOF skill",
        )
        axes[1, column].axhline(0, color="#555555", linewidth=0.8)
        axes[1, column].set_xticks(x, ["T", "P", "T+P"])
        axes[1, column].set_ylabel("correlation / skill")
        axes[1, column].legend(frameon=False, fontsize=8)
    fig.savefig(output / "round123_southeast_tibet_d18o_oof.png", dpi=220)
    plt.close(fig)

    after = {name: file_state(path) for name, path in paths.items()}
    unchanged = bool(before == after)
    if not unchanged:
        raise RuntimeError("source-changed-during-round123")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    summary = {
        "experiment_id": "round123_southeast_tibet_d18o_proxy_qualification_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "locked_1921_1939_candidate_values_decoded": 0,
        "external_validation_reserve_opened": False,
        "source_hashes_unchanged": unchanged,
        "candidate_count": len(SITES),
        "decisions": decisions,
        "identity_and_extraction": extraction_frame.to_dict("records"),
        "pooled_metrics": metrics_frame.to_dict("records"),
        "interpretation_boundary": (
            "This annual-consensus test qualifies candidates for the current annual reconstruction. "
            "A failure does not reject the proxy's documented summer-season relationship."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Round 123 Southeast Tibet d18O Proxy Qualification",
        "",
        "- Status: `complete`.",
        "- Locked 1921-1939 candidate values decoded: `0`.",
        "- External final-test reserve opened: `false`.",
        f"- Source hashes unchanged: `{str(unchanged).lower()}`.",
        "",
        "## Decisions",
        "",
    ]
    for site, decision in decisions.items():
        identity = extraction_frame[extraction_frame.site.eq(site)].iloc[0]
        lines.extend(
            [
                f"### {site}",
                "",
                f"- Decision: `{decision}`.",
                f"- Nearest existing proxy: `{identity.nearest_existing_proxy_distance_km:.1f} km`.",
                f"- Exact spatial duplicate within 1 km: `{str(bool(identity.exact_spatial_duplicate_within_1km)).lower()}`.",
            ]
        )
        for row in metrics_frame[metrics_frame.site.eq(site)].itertuples(index=False):
            lines.append(
                f"- `{row.branch}`: pooled OOF r=`{row.pooled_oof_correlation:.3f}`, "
                f"skill=`{row.pooled_oof_skill_vs_training_mean:.3f}`, "
                f"gate=`{str(bool(row.passes_preliminary_gate)).lower()}`."
            )
        lines.append("")
    lines.extend(
        [
            "## Boundary",
            "",
            "This is an annual temperature/precipitation-consensus qualification. A failed branch does not "
            "invalidate the published summer-season proxy relationship; it redirects the record to a seasonal PSM test.",
        ]
    )
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
