"""V2 of the six-station O-EnKF attribution; fixes v1 proxy/year axis indexing."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DRIVER_DIR = Path(__file__).resolve().parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))

import run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 as round25
import run_climate_reconstruction_loop_v2_round28_temperature_proxy_innovation_concentration_v1 as round28
import run_temperature_six_station_oenkf_innovation_gain_attribution_v1 as base


def main() -> int:
    root = round25.project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "temperature_six_station_oenkf_innovation_gain_attribution_v2"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    manifest27 = json.loads((loop / "round_27_temperature_innovation_gain_sign_decomposition_v1/source_manifest.json").read_text(encoding="utf-8"))
    old = manifest27["sources_before"]
    recovered = {key: Path(old[key]["path"]) for key in ("prior", "indices", "proxy6", "proxy9", "regem6", "regem9")}
    paths = {
        **recovered,
        "station_selection": loop / "temperature_early_station_homogeneity_audit_v2/early_station_metrics.csv",
        "member_annual": loop / "temperature_six_station_prior_increment_level_attribution_v1/station_member_annual_components.csv",
        "method_v2_summary": loop / "temperature_six_station_method_mechanism_attribution_v2/summary.json",
        "proxy9_identity": loop / "round_07_proxy_network_suitability_inventory_v2/proxy9_suitability_inventory.csv",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": DRIVER_DIR / "temperature_six_station_oenkf_innovation_gain_attribution_v1_contract.md",
        "failed_v1_driver": DRIVER_DIR / "run_temperature_six_station_oenkf_innovation_gain_attribution_v1.py",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: {"size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for name, path in paths.items()}
    if json.loads(paths["method_v2_summary"].read_text(encoding="utf-8"))["locked_1921_1939_accessed"]:
        raise RuntimeError("upstream-lock-boundary-failed")

    stations = pd.read_csv(paths["station_selection"])[["station_id", "station_name", "latitude", "longitude"]]
    member_annual = pd.read_csv(paths["member_annual"])
    member_annual = member_annual[member_annual.year.isin(base.YEARS)]
    if len(stations) != 6 or member_annual.year.between(1921, 1939).any():
        raise RuntimeError("station-or-lock-contract-failed")
    station_qcf = member_annual.groupby(["station_id", "year"], as_index=False).qcf_station_degC.first()

    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        pools = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    target_prior = base.sample_prior_points(prior, stations)
    network_summaries = []
    ranking_frames = []
    group_frames = []
    annual_frames = []
    maximum_closure = 0.0
    for network in (6, 9):
        hxb, observations, locked_reads = round25.read_proxy_allowed(paths[f"proxy{network}"], base.YEARS)
        if locked_reads:
            raise RuntimeError("locked-proxy-read")
        residuals = np.asarray(scipy.io.loadmat(paths[f"regem{network}"])["R0v_filled"], dtype=float)
        result = base.run_all_pools(target_prior, hxb, observations, np.nanvar(residuals, axis=1, ddof=1), pools)
        maximum_closure = max(maximum_closure, float(result["closure"]))
        contributions = np.asarray(result["contributions"])
        recovered_increment = np.asarray(result["increment"])
        gains = np.asarray(result["gains"])
        innovations = np.asarray(result["innovations"])
        exact = member_annual[member_annual.member.eq(f"O_{network}")][["station_id", "year", "increment_degC"]]
        comparison_rows = []
        proxy_station_means = np.full((len(stations), contributions.shape[1]), np.nan)
        for station_index, station_row in enumerate(stations.itertuples(index=False)):
            qcf = station_qcf[station_qcf.station_id.eq(station_row.station_id)].set_index("year").qcf_station_degC
            matched = np.asarray([bool(year <= 1920 and np.isfinite(qcf.get(int(year), np.nan))) for year in base.YEARS])
            if matched.sum() < 5:
                raise RuntimeError(f"station-qcf-support-failed:{station_row.station_id}")
            # V1 indexed the boolean year mask before the proxy axis and transposed this slice.
            proxy_station_means[station_index] = np.mean(contributions[station_index][:, matched], axis=1)
            exact_station = exact[exact.station_id.eq(station_row.station_id)].set_index("year").increment_degC
            for year_index, year in enumerate(base.YEARS):
                comparison_rows.append({
                    "network": network,
                    "station_id": station_row.station_id,
                    "station_name": station_row.station_name,
                    "year": int(year),
                    "qcf_matched_historical_year": bool(matched[year_index]),
                    "recovered_increment_degC": recovered_increment[station_index, year_index],
                    "archived_O_increment_degC": exact_station.get(int(year), np.nan),
                })
        comparison = pd.DataFrame(comparison_rows)
        comparison["difference_degC"] = comparison.recovered_increment_degC - comparison.archived_O_increment_degC
        annual_frames.append(comparison)
        historical = comparison[comparison.qcf_matched_historical_year].dropna(subset=["recovered_increment_degC", "archived_O_increment_degC"])
        correlation = float(np.corrcoef(historical.recovered_increment_degC, historical.archived_O_increment_degC)[0, 1])
        rmse = float(np.sqrt(np.mean(historical.difference_degC**2)))
        recovered_level = float(np.mean(np.sum(proxy_station_means, axis=1)))
        archived_level = float(historical.groupby("station_id").archived_O_increment_degC.mean().mean())
        level_error = recovered_level - archived_level

        proxy_values = np.mean(proxy_station_means, axis=0)
        ranking = round28.metadata(paths[f"proxy{network}"])
        ranking["network"] = network
        ranking["equal_station_mean_contribution_degC"] = proxy_values
        ranking["absolute_contribution_degC"] = np.abs(proxy_values)
        ranking["proxy_location_group"] = np.select(
            [ranking.longitude < 80.0, ranking.longitude < 90.0],
            ["west_lt80", "central_80_90"],
            default="east_ge90",
        )
        ranking["innovation_1901_1920_minus_1961_1990_proxy_units"] = np.nanmean(innovations[:, base.HISTORICAL], axis=1) - np.nanmean(innovations[:, base.BASELINE], axis=1)
        ranking["mean_six_station_gain"] = np.nanmean(gains[:, :, base.HISTORICAL], axis=(0, 2))
        ranking["mean_sign_product_matches_contribution"] = (
            np.sign(ranking.innovation_1901_1920_minus_1961_1990_proxy_units)
            * np.sign(ranking.mean_six_station_gain)
            == np.sign(ranking.equal_station_mean_contribution_degC)
        )
        if network == 9:
            identity = pd.read_csv(paths["proxy9_identity"])
            columns = [column for column in ("record_index_1based", "metadata_sources", "archive_type", "measurement", "documented_climate_variable", "documented_seasonality", "network_role") if column in identity.columns]
            ranking = ranking.merge(identity[columns], on="record_index_1based", how="left", validate="one_to_one")
        ranking = ranking.sort_values("absolute_contribution_degC", ascending=False)
        ranking.to_csv(output / f"proxy{network}_six_station_contribution_ranking.csv", index=False)
        ranking_frames.append(ranking)

        group_rows = []
        dimensions = ["proxy_location_group", "proxy_type", "psm_method"]
        if network == 9:
            dimensions += [column for column in ("documented_climate_variable", "documented_seasonality", "network_role") if column in ranking.columns]
        for dimension in dimensions:
            for group_name, values in ranking.groupby(dimension, dropna=False):
                group_rows.append({
                    "network": network,
                    "dimension": dimension,
                    "group": str(group_name),
                    "record_count": int(len(values)),
                    "net_contribution_degC": float(values.equal_station_mean_contribution_degC.sum()),
                    "positive_contribution_degC": float(values.loc[values.equal_station_mean_contribution_degC > 0, "equal_station_mean_contribution_degC"].sum()),
                    "negative_contribution_degC": float(values.loc[values.equal_station_mean_contribution_degC < 0, "equal_station_mean_contribution_degC"].sum()),
                })
        group_frames.append(pd.DataFrame(group_rows))
        concentration = base.concentration(proxy_values)
        reproduction_pass = bool(correlation >= 0.90 and rmse <= 0.15 and abs(level_error) <= 0.10)
        network_summaries.append({
            "network": network,
            "proxy_count": int(len(proxy_values)),
            "contribution_closure_max_abs_degC": float(result["closure"]),
            "archived_reproduction_correlation": correlation,
            "archived_reproduction_rmse_degC": rmse,
            "recovered_equal_station_increment_degC": recovered_level,
            "archived_equal_station_increment_degC": archived_level,
            "level_error_degC": level_error,
            "reproduction_passes": reproduction_pass,
            "mean_gain_innovation_product_sign_match_fraction": float(ranking.mean_sign_product_matches_contribution.mean()),
            **concentration,
        })

    pd.concat(annual_frames, ignore_index=True).to_csv(output / "recovered_archived_increment_closure.csv", index=False)
    pd.concat(ranking_frames, ignore_index=True).to_csv(output / "all_proxy_six_station_contributions.csv", index=False)
    pd.concat(group_frames, ignore_index=True).to_csv(output / "proxy_group_contributions.csv", index=False)
    network_table = pd.DataFrame(network_summaries)
    network_table.to_csv(output / "network_attribution_summary.csv", index=False)
    reproduction_passes = bool(network_table.reproduction_passes.all() and maximum_closure <= 1e-10)
    concentrated = bool(((network_table.top10_positive_share >= 0.50) | (network_table.effective_absolute_contributor_count <= 20)).all())
    decision = (
        "halt_proxy_ranking_archived_increment_reproduction_failed" if not reproduction_passes
        else "positive_update_concentrated_route_to_top_proxy_psm_audit" if concentrated
        else "positive_update_broad_route_to_network_wide_psm_covariance_audit"
    )
    after = {name: {"size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-state-changed")
    summary = {
        "experiment": "temperature_six_station_oenkf_innovation_gain_attribution_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if reproduction_passes else "failed_reproduction_gate",
        "supersedes_failed_execution_of": "temperature_six_station_oenkf_innovation_gain_attribution_v1",
        "v1_failure": "proxy/year boolean indexing transposed the contribution slice before aggregation; no v1 scientific result was accepted",
        "period": [1901, 1920],
        "baseline": [1961, 1990],
        "pool_count": 50,
        "locked_1921_1939_accessed": False,
        "maximum_algebraic_closure_degC": maximum_closure,
        "network_metrics": network_summaries,
        "both_networks_concentrated": concentrated,
        "decision": decision,
        "interpretation_boundary": "Proxy rankings are authorized only if all-pool sums reproduce archived O-member increments. Mean gain times mean innovation is a sign diagnostic; exact contribution also contains pool/year covariance.",
        "source_states_unchanged": unchanged,
        "enkf_rerun": False,
        "production_modified": False,
        "oggm_run": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "unchanged": unchanged, "inherited_sha256": {key: old[key]["sha256"] for key in recovered}}, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 8.2), constrained_layout=True)
    for axis, network, color in zip(axes, (6, 9), ("#00798C", "#B23A48")):
        ranking = next(frame for frame in ranking_frames if int(frame.network.iloc[0]) == network).head(20).sort_values("equal_station_mean_contribution_degC")
        y = np.arange(len(ranking))
        axis.barh(y, ranking.equal_station_mean_contribution_degC, color=np.where(ranking.equal_station_mean_contribution_degC >= 0, color, "#8A8A8A"))
        axis.set(yticks=y, yticklabels=[str(int(value)) for value in ranking.record_index_1based], xlabel="Equal-station contribution (degC)", ylabel="Proxy record index", title=f"proxy{network}: top 20 absolute O-EnKF contributions")
        axis.axvline(0, color="#555555", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "top_proxy_contributions_1901_1920.png", dpi=220)
    plt.close(fig)

    lines = ["# Six-station O-EnKF innovation/gain attribution v2", "", f"- Archived reproduction gate: `{str(reproduction_passes).lower()}`.", f"- Maximum algebraic closure: `{maximum_closure:.3e} degC`."]
    for row in network_table.itertuples(index=False):
        lines.append(f"- proxy{row.network}: recovered/archive increment `{row.recovered_equal_station_increment_degC:+.3f}`/`{row.archived_equal_station_increment_degC:+.3f} degC`; correlation `{row.archived_reproduction_correlation:.3f}`, RMSE `{row.archived_reproduction_rmse_degC:.3f} degC`, top-10 positive share `{row.top10_positive_share:.1%}`.")
    lines += ["", f"Decision: `{decision}`.", ""]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
