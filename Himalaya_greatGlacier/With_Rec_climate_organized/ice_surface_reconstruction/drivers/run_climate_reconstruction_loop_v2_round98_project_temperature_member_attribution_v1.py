"""Attribute the Round 97 western temperature failure across project members."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DRIVER_DIR = Path(__file__).resolve().parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))
import run_climate_reconstruction_loop_v2_round97_western_ghcnm_temperature_phase_adjudication_v1 as round97


EXPECTED_FACTS_SHA256 = "d523009867d5481aed48e5c3f622eb2727848da5cfa9ca2be194d63cc78efce3"
METHODS = {"AO": ("AO_6", "AO_9"), "HGAO": ("HGAO_6", "HGAO_9"), "O": ("O_6", "O_9")}
NETWORKS = {"network_6": ("AO_6", "HGAO_6", "O_6"), "network_9": ("AO_9", "HGAO_9", "O_9")}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest(path)}


def network_curve(sample: pd.DataFrame, value_column: str) -> pd.Series:
    values = sample.copy()
    values[value_column] -= values.groupby("station_id")[value_column].transform("mean")
    result = values.groupby("year")[value_column].mean().reindex(round97.YEARS)
    return pd.Series(round97.centered(result.to_numpy(float)), index=round97.YEARS)


def evaluate(years: np.ndarray, observed: np.ndarray, values: np.ndarray, label: str, kind: str) -> dict[str, Any]:
    observed = round97.centered(observed)
    values = round97.centered(values)
    rmse = float(np.sqrt(np.mean((values - observed) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(observed ** 2)))
    observed_trend = round97.slope_century(years, observed)
    trend = round97.slope_century(years, values)
    return {
        "kind": kind,
        "label": label,
        "annual_correlation": round97.correlation(values, observed),
        "centered_rmse_degC": rmse,
        "rmse_skill_vs_zero": 1.0 - rmse / zero_rmse,
        "trend_degC_per_century": trend,
        "observed_trend_degC_per_century": observed_trend,
        "trend_sign_agreement": bool(np.sign(trend) == np.sign(observed_trend)),
        "opposite_sign_year_count": int(np.sum(values * observed < 0)),
    }


def main() -> int:
    root = round97.project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_98_project_temperature_member_attribution_v1"
    facts = root / "LIA_PROJECT_FACTS.md"
    contract = output / "ROUND_CONTRACT.md"
    station_path = loop / "round_97_western_ghcnm_temperature_phase_adjudication_v1/station_annual_anomalies_and_products.csv"
    round97_summary = loop / "round_97_western_ghcnm_temperature_phase_adjudication_v1/summary.json"
    reference_path = loop / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc"
    lineage_path = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_00_production_reproduction/member_lineage.csv"
    lineage = pd.read_csv(lineage_path)
    tas = lineage[(lineage.variable == "tas") & lineage.member.isin(round97.PROJECT_MEMBERS)].copy()
    if len(tas) != 6:
        raise RuntimeError("member-lineage-contract-failed")
    member_paths = {row.member: Path(row.source_tif) for row in tas.itertuples(index=False)}
    paths = {"facts": facts, "contract": contract, "station_annual": station_path, "round97_summary": round97_summary, "reference_grid": reference_path, "lineage": lineage_path, "driver": Path(__file__).resolve(), **{f"member_{name}": path for name, path in member_paths.items()}}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if digest(facts) != EXPECTED_FACTS_SHA256:
        raise RuntimeError(f"facts-hash-conflict:{digest(facts)}")
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    for row in tas.itertuples(index=False):
        if before[f"member_{row.member}"]["sha256"] != row.source_tif_sha256:
            raise RuntimeError(f"source-tif-hash-mismatch:{row.member}")

    annual = pd.read_csv(station_path)
    if annual.station_id.nunique() != 7 or annual.year.min() != 1901 or annual.year.max() != 1920:
        raise RuntimeError("round97-station-table-contract-failed")
    lat, lon, _, _ = round97.read_reference(reference_path)
    fields = {name: round97.read_project_member(path, lat, lon) for name, path in member_paths.items()}
    curves = pd.DataFrame({"year": round97.YEARS})
    observed = annual.groupby("year").station_anomaly_degC.mean().reindex(round97.YEARS).to_numpy(float)
    curves["station_network_degC"] = round97.centered(observed)

    member_samples: dict[str, pd.DataFrame] = {}
    for name, field in fields.items():
        sampled = round97.sample_station_cells(field, lat, lon, annual, f"{name}_degC")
        member_samples[name] = sampled
        curves[f"member_{name}_degC"] = network_curve(sampled, f"{name}_degC").to_numpy(float)

    group_definitions: dict[str, tuple[str, ...]] = {
        **{f"method_{name}": members for name, members in METHODS.items()},
        **NETWORKS,
        "ensemble_six": tuple(round97.PROJECT_MEMBERS),
    }
    for omitted in round97.PROJECT_MEMBERS:
        group_definitions[f"leave_out_{omitted}"] = tuple(name for name in round97.PROJECT_MEMBERS if name != omitted)
    for label, names in group_definitions.items():
        curves[label] = curves[[f"member_{name}_degC" for name in names]].mean(axis=1)
        curves[label] = round97.centered(curves[label].to_numpy(float))
    curves.to_csv(output / "member_and_composite_curves.csv", index=False)

    rows = []
    years = curves.year.to_numpy(float)
    obs = curves.station_network_degC.to_numpy(float)
    for name in round97.PROJECT_MEMBERS:
        rows.append(evaluate(years, obs, curves[f"member_{name}_degC"].to_numpy(float), name, "member"))
    for label in group_definitions:
        kind = "ensemble" if label == "ensemble_six" else ("method" if label.startswith("method_") else ("network" if label.startswith("network_") else "leave_one_member_out"))
        rows.append(evaluate(years, obs, curves[label].to_numpy(float), label, kind))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "member_and_composite_metrics.csv", index=False)

    ensemble = metrics[metrics.label.eq("ensemble_six")].iloc[0]
    alternatives = metrics[metrics.kind.isin(["method", "network", "leave_one_member_out"])].copy()
    alternatives["correlation_gain_over_ensemble"] = alternatives.annual_correlation - ensemble.annual_correlation
    alternatives["rmse_ratio_to_ensemble"] = alternatives.centered_rmse_degC / ensemble.centered_rmse_degC
    alternatives["passes_weighting_route"] = (alternatives.correlation_gain_over_ensemble >= 0.15) & (alternatives.rmse_ratio_to_ensemble <= 0.90) & alternatives.trend_sign_agreement
    alternatives.to_csv(output / "weighting_route_candidates.csv", index=False)

    member_metrics = metrics[metrics.kind.eq("member")].copy()
    all_members_wrong_trend = bool((~member_metrics.trend_sign_agreement).all())
    weighting_candidates = alternatives[alternatives.passes_weighting_route]
    if all_members_wrong_trend:
        decision = "common_all_member_trend_failure_close_nonnegative_weighting_route"
    elif len(weighting_candidates):
        decision = "advance_predeclared_member_weighting_experiment"
    else:
        decision = "distributed_member_failure_without_simple_weighting_solution"

    event_year = 1917
    event = curves[curves.year.eq(event_year)].iloc[0]
    summary = {
        "status": "complete",
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "observed_trend_degC_per_century": float(ensemble.observed_trend_degC_per_century),
        "member_trend_sign_agreement_count": int(member_metrics.trend_sign_agreement.sum()),
        "all_six_members_have_wrong_trend_sign": all_members_wrong_trend,
        "member_trend_range_degC_per_century": [float(member_metrics.trend_degC_per_century.min()), float(member_metrics.trend_degC_per_century.max())],
        "best_member_by_correlation": member_metrics.sort_values("annual_correlation", ascending=False).iloc[0].to_dict(),
        "best_member_by_rmse": member_metrics.sort_values("centered_rmse_degC").iloc[0].to_dict(),
        "weighting_route_candidate_count": int(len(weighting_candidates)),
        "weighting_route_candidates": weighting_candidates.label.tolist(),
        "year_1917_station_anomaly_degC": float(event.station_network_degC),
        "year_1917_member_range_degC": [float(event[[f"member_{name}_degC" for name in round97.PROJECT_MEMBERS]].min()), float(event[[f"member_{name}_degC" for name in round97.PROJECT_MEMBERS]].max())],
        "decision": decision,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    axes[0].plot(curves.year, curves.station_network_degC, color="#111111", marker="o", linewidth=2.4, label="GHCN-M stations")
    colors = {"AO_6": "#E69F00", "AO_9": "#B26A00", "HGAO_6": "#009E73", "HGAO_9": "#006B4F", "O_6": "#56B4E9", "O_9": "#0072B2"}
    for name in round97.PROJECT_MEMBERS:
        axes[0].plot(curves.year, curves[f"member_{name}_degC"], color=colors[name], linewidth=1.0, alpha=0.75, label=name)
    axes[0].plot(curves.year, curves.ensemble_six, color="#CC3311", linewidth=2.5, label="Six-member mean")
    axes[0].set_ylabel("Centered anomaly (degC)")
    axes[0].set_title("Individual project members")
    axes[0].legend(frameon=False, ncol=4)
    axes[1].plot(curves.year, curves.station_network_degC, color="#111111", linewidth=2.4, marker="o", label="Stations")
    for label, color in [("method_AO", "#E69F00"), ("method_HGAO", "#009E73"), ("method_O", "#0072B2")]:
        axes[1].plot(curves.year, curves[label], linewidth=1.8, marker="s", label=label.replace("method_", ""))
    axes[1].set_ylabel("Centered anomaly (degC)")
    axes[1].set_title("Method composites")
    axes[1].legend(frameon=False, ncol=4)
    axes[2].plot(curves.year, curves.station_network_degC, color="#111111", linewidth=2.4, marker="o", label="Stations")
    axes[2].plot(curves.year, curves.network_6, color="#AA4499", linewidth=1.8, marker="^", label="Proxy network 6")
    axes[2].plot(curves.year, curves.network_9, color="#4477AA", linewidth=1.8, marker="v", label="Proxy network 9")
    axes[2].axhline(0, color="#777777", linewidth=0.8)
    axes[2].set(ylabel="Centered anomaly (degC)", xlabel="Year", title="Proxy-network composites", xticks=round97.YEARS[::2])
    axes[2].legend(frameon=False, ncol=3)
    fig.savefig(output / "project_member_attribution_curves.png", dpi=220)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "\n".join([
            "# Project Temperature Member Attribution",
            "",
            f"- Observed trend: {summary['observed_trend_degC_per_century']:+.3f} degC/century.",
            f"- Member trend range: {summary['member_trend_range_degC_per_century'][0]:+.3f} to {summary['member_trend_range_degC_per_century'][1]:+.3f} degC/century; correct-sign members: {summary['member_trend_sign_agreement_count']}/6.",
            f"- Best member correlation: {summary['best_member_by_correlation']['label']} at {summary['best_member_by_correlation']['annual_correlation']:.3f}.",
            f"- Valid weighting-route candidates: {summary['weighting_route_candidate_count']}.",
            f"- In 1917 the station network is {summary['year_1917_station_anomaly_degC']:+.3f} degC; all-member range is {summary['year_1917_member_range_degC'][0]:+.3f} to {summary['year_1917_member_range_degC'][1]:+.3f} degC.",
            "",
            f"Decision: {decision}.",
        ]) + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    manifest = {"local_sources_before": before, "local_sources_after": after, "local_source_states_unchanged": before == after, "parsed_years": [1901, 1920], "locked_1921_1939_accessed": False}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["local_source_states_unchanged"]:
        raise RuntimeError("local-source-integrity-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
