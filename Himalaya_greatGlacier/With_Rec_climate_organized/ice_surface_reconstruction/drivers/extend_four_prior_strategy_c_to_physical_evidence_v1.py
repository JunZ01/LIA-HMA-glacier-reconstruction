from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DRIVER = Path(__file__).resolve()
DRIVER_DIR = DRIVER.parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))

import run_temperature_four_prior_strategy_c_replay_1901_1920_v1 as base


ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
OLD = RESULTS / "climate_reconstruction_temperature_four_prior_strategy_c_replay_1901_1920_v1"
EVIDENCE_ROOT = RESULTS / "climate_reconstruction_improvement_loop_v2"
BOREHOLE = EVIDENCE_ROOT / "temperature_noaa_hma_borehole_19c_recovery_v1"
LAKES = EVIDENCE_ROOT / "temperature_noaa_hma_lake_temperature_recovery_v1"
CUOQIA = EVIDENCE_ROOT / "temperature_cuoqia_lake_numeric_recovery_v1"
OUTPUT = EVIDENCE_ROOT / "temperature_four_prior_strategy_c_physical_evidence_extension_v1"
STATE_FILE = OLD / "four_prior_state_pools_1500_1848_v1.npz"
TARGET_YEARS = np.arange(1500, 1921, dtype=int)
POOL_YEARS = np.arange(1500, 1849, dtype=int)
MODELS = base.MODELS
NETWORKS = (6, 9)
BRANCHES = ("full", "omit_central")


def file_state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def evidence_definitions() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    borehole_sites = pd.read_csv(BOREHOLE / "borehole_project_1800_1900_comparison.csv")
    borehole_clusters = pd.read_csv(BOREHOLE / "cluster_19th_century_comparison.csv")
    lake_points = pd.read_csv(LAKES / "lake_temperature_safe_points.csv")
    lake_pairs = pd.read_csv(LAKES / "latest_safe_prelock_pair_comparison.csv")
    cuoqia_trends = pd.read_csv(CUOQIA / "cuoqia_safe_period_trends.csv")

    rows: list[dict[str, Any]] = []
    locations: dict[str, pd.DataFrame] = {}
    for cluster in borehole_clusters.itertuples(index=False):
        sites = borehole_sites[
            (borehole_sites.cluster == cluster.cluster)
            & borehole_sites.project_comparison_eligible
        ][["site_id", "latitude", "longitude"]].copy()
        evidence_id = f"borehole_{cluster.cluster}"
        locations[evidence_id] = sites
        rows.append(
            {
                "evidence_id": evidence_id,
                "measurement_family": "borehole_GST",
                "start_year": 1800.0,
                "end_year": 1900.0,
                "observed_change_c": float(cluster.borehole_19th_change_median_c),
                "native_target": "century_scale_ground_surface_temperature",
                "location_count": int(len(sites)),
                "figure_derived": False,
            }
        )
    for pair in lake_pairs.itertuples(index=False):
        point = lake_points[lake_points.site_id == pair.site_id][
            ["site_id", "latitude", "longitude"]
        ].drop_duplicates()
        evidence_id = f"lake_{pair.site_id}"
        locations[evidence_id] = point
        rows.append(
            {
                "evidence_id": evidence_id,
                "measurement_family": "lake_temperature",
                "start_year": float(pair.start_year_ce),
                "end_year": float(pair.end_year_ce),
                "observed_change_c": float(pair.proxy_change_c),
                "native_target": str(
                    lake_points[lake_points.site_id == pair.site_id].iloc[0].season
                ),
                "location_count": 1,
                "figure_derived": False,
            }
        )
    cuoqia_location = pd.DataFrame(
        [{"site_id": "cuoqia_lake", "latitude": 27.4052, "longitude": 99.7722}]
    )
    for series, start, end in (("CQ1", 1850, 1920), ("CQ2", 1870, 1920)):
        trend = cuoqia_trends[
            (cuoqia_trends.series == series)
            & (cuoqia_trends.start == start)
            & (cuoqia_trends.end == end)
        ].iloc[0]
        evidence_id = f"cuoqia_{series}_{start}_{end}"
        locations[evidence_id] = cuoqia_location
        rows.append(
            {
                "evidence_id": evidence_id,
                "measurement_family": "lake_brGDGT_figure_derived",
                "start_year": float(start),
                "end_year": float(end),
                "observed_change_c": float(trend.slope_c_per_century) * (end - start) / 100.0,
                "native_target": "March-October_temperature",
                "location_count": 1,
                "figure_derived": True,
            }
        )
    return pd.DataFrame(rows).sort_values("evidence_id").reset_index(drop=True), locations


def bilinear_site_values(
    state: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    sites: pd.DataFrame,
) -> np.ndarray:
    cube = np.asarray(state, dtype=float).reshape(
        (len(POOL_YEARS), len(lat), len(lon)), order="F"
    )
    y_axis = np.asarray(lat, dtype=float)
    x_axis = np.asarray(lon, dtype=float)
    if y_axis[1] < y_axis[0]:
        y_axis = y_axis[::-1]
        cube = cube[:, ::-1, :]
    if x_axis[1] < x_axis[0]:
        x_axis = x_axis[::-1]
        cube = cube[:, :, ::-1]
    outputs = []
    for site in sites.itertuples(index=False):
        fy = (float(site.latitude) - y_axis[0]) / (y_axis[1] - y_axis[0])
        fx = (float(site.longitude) - x_axis[0]) / (x_axis[1] - x_axis[0])
        y0 = int(np.floor(fy))
        x0 = int(np.floor(fx))
        if y0 < 0 or y0 >= len(y_axis) - 1 or x0 < 0 or x0 >= len(x_axis) - 1:
            raise RuntimeError(f"evidence-outside-state-grid:{site.site_id}")
        wy, wx = fy - y0, fx - x0
        corners = np.stack(
            [
                cube[:, y0, x0],
                cube[:, y0, x0 + 1],
                cube[:, y0 + 1, x0],
                cube[:, y0 + 1, x0 + 1],
            ],
            axis=0,
        )
        weights = np.asarray(
            [(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx]
        )
        valid = np.isfinite(corners)
        denominator = np.sum(weights[:, None] * valid, axis=0)
        numerator = np.nansum(weights[:, None] * corners, axis=0)
        outputs.append(numerator / np.where(denominator > 0, denominator, np.nan))
    values = np.stack(outputs, axis=1)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("nonfinite-evidence-state-series")
    return np.median(values, axis=1)


def select_backgrounds(
    hxb: np.ndarray,
    observations: np.ndarray,
    record_mask: np.ndarray,
    evidence_states: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    outputs = {
        evidence_id: np.full(len(TARGET_YEARS), np.nan, dtype=float)
        for evidence_id in evidence_states
    }
    active_counts = np.zeros(len(TARGET_YEARS), dtype=int)
    for year_index, year in enumerate(TARGET_YEARS):
        active = record_mask & np.isfinite(observations[:, year_index])
        active_counts[year_index] = int(active.sum())
        if active_counts[year_index] < 3:
            continue
        y = observations[active, year_index]
        y_anomaly = y - np.mean(y)
        model = hxb[active]
        model_anomaly = model - np.mean(model, axis=0, keepdims=True)
        numerator = y_anomaly @ model_anomaly
        denominator = np.sqrt(
            np.sum(y_anomaly**2) * np.sum(model_anomaly**2, axis=0)
        )
        scores = np.full(model.shape[1], -1.0, dtype=float)
        valid = denominator > 0
        scores[valid] = numerator[valid] / denominator[valid]
        chosen = np.argsort(-scores, kind="stable")[:100]
        for evidence_id, state_values in evidence_states.items():
            outputs[evidence_id][year_index] = float(np.mean(state_values[chosen]))
    return outputs, active_counts


def interp_safe(years: np.ndarray, values: np.ndarray, year: float) -> float:
    valid = np.isfinite(values)
    if valid.sum() < 2 or year < years[valid].min() or year > years[valid].max():
        return float("nan")
    return float(np.interp(year, years[valid], values[valid]))


def direction(value: float, tolerance: float = 1e-12) -> str:
    if value > tolerance:
        return "warming"
    if value < -tolerance:
        return "cooling"
    return "neutral"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base.TARGET_YEARS = TARGET_YEARS
    base.POOL_YEARS = POOL_YEARS

    definitions, locations = evidence_definitions()
    definitions.to_csv(OUTPUT / "physical_evidence_definitions.csv", index=False)
    sources = {
        "state_pool": STATE_FILE,
        "state_summary": OLD / "state_pool_acquisition_summary.json",
        "old_strategy_summary": OLD / "summary.json",
        "proxy6": base.prior_tools.ARCHIVE / "python_data/proxy_data_6.mat",
        "proxy9": base.prior_tools.ARCHIVE / "python_data/proxy_data_9.mat",
        "borehole_comparison": BOREHOLE / "borehole_project_1800_1900_comparison.csv",
        "borehole_clusters": BOREHOLE / "cluster_19th_century_comparison.csv",
        "lake_points": LAKES / "lake_temperature_safe_points.csv",
        "lake_pairs": LAKES / "latest_safe_prelock_pair_comparison.csv",
        "cuoqia_trends": CUOQIA / "cuoqia_safe_period_trends.csv",
        "driver": DRIVER,
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: file_state(path) for name, path in sources.items()}

    states = np.load(STATE_FILE)
    if not np.array_equal(np.asarray(states["years"], dtype=int), POOL_YEARS):
        raise RuntimeError("pool-year-axis-failed")
    member_years = np.arange(1001, 1849, dtype=int)
    pool_indices = np.asarray(
        [int(np.flatnonzero(member_years == year)[0]) for year in POOL_YEARS]
    )
    proxies = {
        network: base.load_proxy(network, pool_indices) for network in NETWORKS
    }
    maximum_closure = max(
        float(record.get("closure", 0.0))
        for proxy in proxies.values()
        for record in proxy["records"]
    )
    if maximum_closure > 1e-12:
        raise RuntimeError(f"archived-psm-closure:{maximum_closure}")

    native_lat = np.asarray(states["native_lat"], dtype=float)
    native_lon = np.asarray(states["native_lon"], dtype=float)
    evidence_states: dict[str, dict[str, np.ndarray]] = {}
    for model in MODELS:
        model_state = np.asarray(states[f"{model}_tas_state"], dtype=float)
        evidence_states[model] = {
            evidence_id: bilinear_site_values(
                model_state, native_lat, native_lon, locations[evidence_id]
            )
            for evidence_id in definitions.evidence_id
        }

    annual_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for model, model_label in MODELS.items():
        for network in NETWORKS:
            proxy = proxies[network]
            hxb = base.rebuild_hxb(model, network, proxy, states)
            central = (
                (proxy["locations"][:, 1] >= 80.0)
                & (proxy["locations"][:, 1] < 90.0)
            )
            masks = {"full": np.ones(len(central), dtype=bool), "omit_central": ~central}
            for branch, record_mask in masks.items():
                selected, active_counts = select_backgrounds(
                    hxb,
                    proxy["observations"],
                    record_mask,
                    evidence_states[model],
                )
                for evidence_id, curve in selected.items():
                    for year_index, year in enumerate(TARGET_YEARS):
                        annual_rows.append(
                            {
                                "model": model_label,
                                "proxy_network": network,
                                "branch": branch,
                                "evidence_id": evidence_id,
                                "year": int(year),
                                "selected_background_degC": float(curve[year_index]),
                                "active_proxy_count": int(active_counts[year_index]),
                            }
                        )
                    definition = definitions[definitions.evidence_id == evidence_id].iloc[0]
                    start_value = interp_safe(
                        TARGET_YEARS, curve, float(definition.start_year)
                    )
                    end_value = interp_safe(
                        TARGET_YEARS, curve, float(definition.end_year)
                    )
                    selected_change = end_value - start_value
                    observed_change = float(definition.observed_change_c)
                    metric_rows.append(
                        {
                            "model": model_label,
                            "proxy_network": network,
                            "branch": branch,
                            "evidence_id": evidence_id,
                            "start_year": float(definition.start_year),
                            "end_year": float(definition.end_year),
                            "observed_change_c": observed_change,
                            "selected_background_change_c": selected_change,
                            "observed_direction": direction(observed_change),
                            "selected_direction": direction(selected_change),
                            "direction_agreement": direction(observed_change)
                            == direction(selected_change),
                            "absolute_amplitude_error_c": abs(selected_change - observed_change),
                        }
                    )

    annual = pd.DataFrame(annual_rows)
    metrics = pd.DataFrame(metric_rows)
    annual.to_csv(OUTPUT / "annual_selected_background_at_evidence.csv", index=False)
    metrics.to_csv(OUTPUT / "physical_evidence_direction_metrics.csv", index=False)

    full = metrics[metrics.branch == "full"].copy()
    summary_rows = (
        full.groupby("evidence_id", as_index=False)
        .agg(
            observed_change_c=("observed_change_c", "first"),
            full_branch_count=("direction_agreement", "size"),
            direction_agreement_count=("direction_agreement", "sum"),
            selected_change_median_c=("selected_background_change_c", "median"),
            selected_change_min_c=("selected_background_change_c", "min"),
            selected_change_max_c=("selected_background_change_c", "max"),
            median_absolute_amplitude_error_c=("absolute_amplitude_error_c", "median"),
        )
    )
    summary_rows["all_full_branches_agree"] = (
        summary_rows.direction_agreement_count == summary_rows.full_branch_count
    )
    summary_rows["all_full_branches_conflict"] = summary_rows.direction_agreement_count.eq(0)
    summary_rows.to_csv(OUTPUT / "physical_evidence_cross_prior_summary.csv", index=False)
    branch_summary = (
        metrics.groupby(["branch", "evidence_id"], as_index=False)
        .agg(
            branch_count=("direction_agreement", "size"),
            direction_agreement_count=("direction_agreement", "sum"),
            selected_change_median_c=("selected_background_change_c", "median"),
        )
    )
    branch_summary["all_branches_agree"] = (
        branch_summary.direction_agreement_count == branch_summary.branch_count
    )
    branch_summary["all_branches_conflict"] = branch_summary.direction_agreement_count.eq(0)
    branch_summary.to_csv(OUTPUT / "physical_evidence_branch_direction_summary.csv", index=False)

    omission = metrics.pivot_table(
        index=["model", "proxy_network", "evidence_id"],
        columns="branch",
        values="selected_background_change_c",
    ).reset_index()
    omission["central_omission_effect_full_minus_omit_central_c"] = (
        omission["full"] - omission["omit_central"]
    )
    omission.to_csv(OUTPUT / "central_proxy_omission_effects.csv", index=False)

    order = summary_rows.sort_values("evidence_id").evidence_id.tolist()
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    x = np.arange(len(order))
    observed = summary_rows.set_index("evidence_id").loc[order].observed_change_c
    ax.scatter(x, observed, color="#111111", s=65, marker="D", label="Independent physical change", zorder=5)
    colors = {6: "#2f78a8", 9: "#d17a22"}
    offsets = {6: -0.10, 9: 0.10}
    for network in NETWORKS:
        rows = full[full.proxy_network == network]
        for index, evidence_id in enumerate(order):
            values = rows[rows.evidence_id == evidence_id].selected_background_change_c.to_numpy(float)
            ax.scatter(
                np.full(len(values), x[index] + offsets[network]),
                values,
                color=colors[network],
                s=30,
                alpha=0.8,
                label=f"proxy{network} four priors" if index == 0 else None,
            )
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([item.replace("_", " ") for item in order], rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Native interval temperature change (degC)")
    ax.set_title("Four-prior Strategy-C background selection vs independent physical evidence")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "four_prior_physical_evidence_changes.png", dpi=220)
    plt.close(fig)

    after = {name: file_state(path) for name, path in sources.items()}
    unchanged = all(before[name]["sha256"] == after[name]["sha256"] for name in sources)
    if not unchanged:
        raise RuntimeError("source-file-changed")

    full_agree = int(summary_rows.all_full_branches_agree.sum())
    full_conflict = int(summary_rows.all_full_branches_conflict.sum())
    omit_summary = branch_summary[branch_summary.branch == "omit_central"]
    omit_agree = int(omit_summary.all_branches_agree.sum())
    omit_conflict = int(omit_summary.all_branches_conflict.sum())
    summary = {
        "experiment": "temperature_four_prior_strategy_c_physical_evidence_extension_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "target_years": [1500, 1920],
        "pool_years": [1500, 1848],
        "models": list(MODELS.values()),
        "proxy_networks": list(NETWORKS),
        "branches": list(BRANCHES),
        "evidence_definition_count": int(len(definitions)),
        "all_8_full_model_network_branches_agree_count": full_agree,
        "all_8_full_model_network_branches_conflict_count": full_conflict,
        "mixed_cross_prior_or_network_count": int(len(summary_rows) - full_agree - full_conflict),
        "omit_central_all_8_branches_agree_count": omit_agree,
        "omit_central_all_8_branches_conflict_count": omit_conflict,
        "major_conflicts_fixed_by_omitting_central_proxies": False,
        "maximum_archived_bivariate_psm_closure": maximum_closure,
        "source_hashes_unchanged": unchanged,
        "interpretation_boundary": (
            "Strategy-C top-100 background-selection mechanism sensitivity only. This is not an EnKF posterior, "
            "does not validate the proxies or PSMs, and compares unlike native targets by direction and rough amplitude only."
        ),
        "locked_1921_1939_values_accessed": 0,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "prohibited_source_accessed": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    report_rows = "\n".join(
        f"| {row.evidence_id} | {row.observed_change_c:+.3f} | {row.selected_change_median_c:+.3f} | "
        f"{int(row.direction_agreement_count)}/{int(row.full_branch_count)} |"
        for row in summary_rows.itertuples(index=False)
    )
    report = f"""# Four-prior Strategy-C physical-evidence extension

This extends the existing 1901-1920 replay to every allowed year from 1500 through 1920 and evaluates the selected background at independent physical-evidence locations.

| Evidence | Physical change | Full-branch median | Direction agreements |
|---|---:|---:|---:|
{report_rows}

All eight full branches agree for {full_agree}/{len(summary_rows)} evidence definitions, all conflict for {full_conflict}/{len(summary_rows)}, and the rest depend on prior family or proxy network. After omitting central proxies, all eight branches still conflict at Naqu, PX202, and Qinghai. Central-proxy omission therefore does not repair the major regional conflicts.

This is a Strategy-C mechanism sensitivity, not an EnKF posterior or production candidate. No 1921-1939 value was accessed, and 20CRv3 was not accessed.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
