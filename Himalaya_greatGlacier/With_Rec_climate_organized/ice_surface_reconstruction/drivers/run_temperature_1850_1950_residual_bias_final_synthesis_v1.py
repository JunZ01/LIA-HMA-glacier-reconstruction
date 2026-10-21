"""Synthesize the 1850-1950 HMA temperature residual-bias investigation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PERIODS = ("1880-1900", "1901-1920", "1940-1950")


def root_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    root = root_path()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "temperature_1850_1950_residual_bias_final_synthesis_v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    paths = {
        "station_curves": loop / "temperature_station_annual_anomaly_benchmark_v4/station_annual_anomalies_and_products.csv",
        "station_summary": loop / "temperature_station_annual_anomaly_benchmark_v4/summary.json",
        "homogeneity_summary": loop / "temperature_early_station_homogeneity_audit_v2/summary.json",
        "station_period": loop / "temperature_station_residual_spatial_transfer_feasibility_v2/station_period_residuals.csv",
        "transfer_summary": loop / "temperature_station_residual_spatial_transfer_feasibility_v2/summary.json",
        "level_attribution": loop / "temperature_six_station_prior_increment_level_attribution_v2/summary.json",
        "method_attribution": loop / "temperature_six_station_method_mechanism_attribution_v2/summary.json",
        "oenkf_attribution": loop / "temperature_six_station_oenkf_innovation_gain_attribution_v2/summary.json",
        "orientation_summary": loop / "temperature_six_station_psm_covariance_orientation_audit_v1/summary.json",
        "local_proxy_summary": loop / "temperature_six_station_local_proxy_innovation_adjudication_v1/summary.json",
        "candidate_level": loop / "temperature_station_candidate_v2_allowed_years_v2/summary.json",
        "candidate_trend": loop / "temperature_candidate_v2_trend_direction_validation_v1/summary.json",
        "iitm_summary": loop / "temperature_1850_1950_iitm_homogeneous_india_qualification_v2/summary.json",
        "iitm_matrix": loop / "temperature_1850_1950_iitm_homogeneous_india_qualification_v2/derived/iitm_wh_ne_qualification_matrix.csv",
        "round217": loop / "round_217_temperature_bias_phase_three_record_synthesis_v1/summary.json",
        "round224": loop / "round_224_central_himalaya_mxd2026_public_recovery_v6/summary.json",
        "round230": loop / "round_230_yang2010_project_reference_locked_safe_comparison_v2/summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: digest(path) for name, path in paths.items()}
    summaries = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items() if path.suffix == ".json"}
    lock_flags = [
        summaries["homogeneity_summary"]["locked_1921_1939_accessed"],
        summaries["transfer_summary"]["locked_1921_1939_accessed"],
        summaries["level_attribution"]["locked_1921_1939_accessed"],
        summaries["method_attribution"]["locked_1921_1939_accessed"],
        summaries["oenkf_attribution"]["locked_1921_1939_accessed"],
        summaries["orientation_summary"]["locked_1921_1939_accessed"],
        summaries["local_proxy_summary"]["locked_1921_1939_accessed"],
        summaries["candidate_level"]["locked_1921_1939_included_or_accessed"],
        summaries["candidate_trend"]["locked_1921_1939_accessed"],
    ]
    if any(lock_flags):
        raise RuntimeError("upstream-lock-boundary-failed")

    station_period = pd.read_csv(paths["station_period"])
    station_identity = pd.read_csv(paths["station_curves"])[["station_id", "station_name", "latitude", "longitude", "region_group"]].drop_duplicates()
    station_period = station_period.merge(station_identity, on=["station_id", "station_name", "latitude", "longitude"], how="left", validate="many_to_one")
    regional_rows = []
    for keys, group in station_period.groupby(["region_group", "period", "target"], dropna=False):
        region, period, target = keys
        regional_rows.append({
            "region_group": region, "period": period, "target": target,
            "station_count": int(len(group)),
            "mean_project_minus_station_degC": float(group.residual_degC.mean()),
            "median_project_minus_station_degC": float(group.residual_degC.median()),
            "positive_residual_station_count": int((group.residual_degC > 0).sum()),
        })
    regional = pd.DataFrame(regional_rows)
    regional.to_csv(output / "regional_station_residual_summary.csv", index=False)

    evidence_rows = [
        {"evidence_id": "GHCN-M_qcf", "region": "western_and_central_western_HMA", "period": "1880-1920;1940-1950", "target": "calendar annual Tmean anomaly", "independence": "direct_station_primary", "role": "primary_level_and_trend_gate", "finding": "Warm residual robust in 1880-1920 and smaller by 1940-1950."},
        {"evidence_id": "GHCN-M_qcu", "region": "same_stations", "period": "1880-1920;1940-1950", "target": "unadjusted calendar annual Tmean anomaly", "independence": "same_source_not_independent", "role": "homogenization_sensitivity", "finding": "Supports early warm residual; late 1940-1950 level is near zero and warns against overcorrection."},
        {"evidence_id": "IITM_WHIND_NEIND", "region": "Indian_western_Himalaya_and_northeast_India", "period": "1901-1920;1940-1950", "target": "regional Tmin/Tmax and derived Tmean", "independence": "IMD_source_correlated_processing_route", "role": "regional_direction_processing_sensitivity", "finding": "Early direction differs by region; rejects one HMA-wide phase correction."},
        {"evidence_id": "Baiyu_Haizi_Huang", "region": "eastern_HMA", "period": "1500-1899;1901-1920;1940-2000", "target": "seasonal_and_annual_temperature_reconstructions", "independence": "source_independent_chronologies", "role": "external_level_and_seasonality_boundary", "finding": "Reject uniform historical cooling; early phase conflict is season-dependent."},
        {"evidence_id": "Yang2010", "region": "southeastern_Tibet_94.27E", "period": "1377-1899;1940-1998", "target": "January-June temperature digitized low-pass", "independence": "external_but_digitized", "role": "eastern_shape_boundary", "finding": "Does not support a frozen local warm-bias correction; project closer than reference on 50-year RMSE."},
        {"evidence_id": "Rara_Cholangpati_MXD2026", "region": "central_Himalaya", "period": "post_1775", "target": "temperature_reconstruction_lead", "independence": "source_independent_lead", "role": "coverage_gap", "finding": "Annual numerical series is not publicly machine-readable; cannot validate Candidate v2."},
        {"evidence_id": "CRU_Berkeley", "region": "HMA_grid", "period": "allowed_instrumental_years", "target": "gridded annual temperature", "independence": "station_source_overlap", "role": "background_product_only", "finding": "Not counted as independent votes; early CRU coverage is sparse."},
        {"evidence_id": "Local_proxy_innovations", "region": "100-300km_around_six_stations", "period": "1901-1920", "target": "temperature-oriented proxy innovation", "independence": "assimilated_proxy_not_validation", "role": "mechanism_diagnostic", "finding": "Local proxy-station direction is mixed/insufficient; does not authorize proxy deletion."},
    ]
    evidence = pd.DataFrame(evidence_rows)
    evidence.to_csv(output / "evidence_matrix.csv", index=False)

    mechanism_rows = [
        {"step": "station_residual", "result": "+0.846 C equal-station qcf residual in 1901-1920", "implication": "A western direct-station level discrepancy is real on available support."},
        {"step": "background_vs_increment", "result": "71.7% background-minus-station; 28.3% assimilation increment", "implication": "Background is primary, but the update reinforces warming."},
        {"step": "six_member_consistency", "result": "All six members warm; all six increments positive", "implication": "Not a single method or proxy-network outlier."},
        {"step": "innovation_gain_replay", "result": "Archived O increments reproduced at r>0.99998", "implication": "Contribution attribution is numerically trustworthy."},
        {"step": "psm_covariance_orientation", "result": "Only 1.6%-4.4% of positive contribution from b1/gain-opposed records", "implication": "Covariance sign reversal is not the cause; western proxy innovations orient warm."},
        {"step": "local_proxy_adjudication", "result": "Mixed/insufficient within 100-300 km", "implication": "Cannot decide proxy truth versus station conflict or delete records."},
        {"step": "spatial_transfer", "result": "1901-1920 qcf/qcu LOO MAE improves 55.3%/29.9%", "implication": "The level residual transfers spatially among available stations."},
        {"step": "temporal_transfer", "result": "Unchanged early amplitude worsens 1940-1950 qcf MAE by 80.9%", "implication": "A constant correction is invalid; amplitude is not stationary."},
        {"step": "candidate_v2", "result": "Level improves, qcf trend error worsens 170% and 135% in early blocks", "implication": "The inferred continuous temporal correction is rejected."},
    ]
    pd.DataFrame(mechanism_rows).to_csv(output / "mechanism_chain.csv", index=False)

    candidate_trend = summaries["candidate_trend"]
    final_decision = "retain_original_production_temperature_reject_candidate_v2_temporal_shape"
    unresolved = [
        "No strict direct-station inference for 1850-1879.",
        "Central/eastern HMA lacks comparable early direct-station density.",
        "The western station level residual is validated, but its continuous temporal evolution is not.",
        "Assimilated proxies and direct stations conflict locally in a mixed way; current evidence cannot label either side erroneous.",
        "The sealed 1921-1939 interval remains untouched and was not used to select a correction.",
    ]
    final = {
        "objective": "diagnose_and_if_supported_correct_HMA_temperature_residual_bias_1850_1950",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_with_original_retained",
        "locked_1921_1939_accessed": False,
        "validated_bias_statement": "The six-member reconstruction is too warm relative to available direct qcf stations in western/central-western HMA during 1880-1920. The discrepancy is smaller by 1940-1950. Evidence does not support an HMA-wide or pre-1880 correction.",
        "mechanism_statement": "The fixed/adaptive background does not contain enough of the observed cold anomaly, and all six assimilation updates add warming. The update is driven by a broad western proxy-innovation common mode with covariance-consistent gains, not by a few records or gain-sign reversal.",
        "candidate_v2_statement": "A regional allowed-year Candidate v2 improved period-mean station level but failed direct-station temporal-shape gates and worsened IITM WHIND trend sensitivity. It is a rejected diagnostic artifact, not a valid climate replacement.",
        "final_decision": final_decision,
        "production_temperature_modified": False,
        "production_eligible_candidate": None,
        "unresolved_boundaries": unresolved,
        "next_scientific_route": "Acquire source-independent numeric western/central temperature evidence or redesign the reconstruction with station-aware low-frequency/seasonal PSM validation under a new authorized EnKF experiment; do not apply a post-hoc deterministic field correction.",
    }
    (output / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

    after = {name: digest(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-changed")
    (output / "source_manifest.json").write_text(json.dumps({"before_sha256": before, "after_sha256": after, "unchanged": unchanged}, indent=2), encoding="utf-8")

    station_map = station_identity.copy()
    station_map["has_early_qcf"] = station_map.station_id.isin(station_period[(station_period.period == "1901-1920") & (station_period.target == "qcf")].station_id)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)
    axis = axes[0, 0]
    axis.axvspan(62, 80, color="#E8F2F1", alpha=0.8)
    axis.axvspan(80, 90, color="#F3EEE2", alpha=0.8)
    axis.axvspan(90, 110, color="#F2E8EA", alpha=0.8)
    axis.scatter(station_map.longitude, station_map.latitude, c=np.where(station_map.has_early_qcf, "#B23A48", "#777777"), s=np.where(station_map.has_early_qcf, 42, 24), label="GHCN-M stations")
    external = pd.DataFrame({"name": ["Yang", "Baiyu", "Haizi"], "lat": [29.3667, 31.15, 29.58], "lon": [94.2667, 99.81, 100.65]})
    axis.scatter(external.lon, external.lat, marker="^", color="#00798C", s=52, label="independent reconstructions")
    for row in external.itertuples(index=False):
        axis.text(row.lon + 0.4, row.lat + 0.15, row.name, fontsize=8)
    axis.set(xlim=(62, 110), ylim=(24, 43), xlabel="Longitude", ylabel="Latitude", title="Evidence coverage: west / central / east")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[0, 1]
    qcf = station_period[station_period.target.eq("qcf")].groupby("period").residual_degC.mean().reindex(PERIODS)
    qcu = station_period[station_period.target.eq("qcu")].groupby("period").residual_degC.mean().reindex(PERIODS)
    x = np.arange(len(PERIODS))
    axis.bar(x - 0.18, qcf, width=0.36, color="#B23A48", label="qcf primary")
    axis.bar(x + 0.18, qcu, width=0.36, color="#00798C", label="qcu sensitivity")
    axis.axhline(0, color="#777777", lw=0.8)
    axis.set(xticks=x, xticklabels=PERIODS, ylabel="Project - station (degC)", title="Direct-station residual level")
    axis.legend(frameon=False)

    axis = axes[1, 0]
    network = pd.DataFrame(summaries["orientation_summary"]["network_metrics"])
    axis.bar(np.arange(2) - 0.18, network.west_fraction_of_net_increment * 100, width=0.36, color="#B23A48", label="west share of net update")
    axis.bar(np.arange(2) + 0.18, network.opposed_fraction_of_positive_contribution * 100, width=0.36, color="#777777", label="b1/gain-opposed positive share")
    axis.set(xticks=np.arange(2), xticklabels=["proxy6", "proxy9"], ylabel="Share (%)", title="Assimilation mechanism")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 1]
    trend = pd.DataFrame(candidate_trend["direct_station_trend_summary"])
    trend = trend[trend.target.eq("qcf")].set_index("period").reindex(PERIODS)
    axis.bar(x - 0.18, trend.original_median_abs_trend_error, width=0.36, color="#777777", label="original")
    axis.bar(x + 0.18, trend.candidate_median_abs_trend_error, width=0.36, color="#B23A48", label="Candidate v2")
    axis.set(xticks=x, xticklabels=PERIODS, ylabel="Median abs trend error", title="Why Candidate v2 is rejected")
    axis.legend(frameon=False)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "temperature_1850_1950_final_synthesis.png", dpi=230)
    plt.close(fig)

    report = [
        "# HMA temperature residual-bias investigation, 1850/1860-1950",
        "",
        "## Final decision",
        "",
        f"`{final_decision}`",
        "",
        "The original production temperature remains unchanged. Candidate v2 is rejected as a temporal field because it improves mean level while materially worsening direct-station trend error.",
        "",
        "## What is established",
        "",
        "- Strict direct-station evidence supports a western/central-western warm residual in 1880-1920, not an HMA-wide bias and not a claim about 1850-1879.",
        "- Mean qcf residuals are +1.418 degC (1880-1900, 3 stations), +0.847 degC (1901-1920, 6 stations), and +0.250 degC (1940-1950, 8 stations).",
        "- The 1901-1920 six-station decomposition is 71.7% background-minus-station and 28.3% assimilation increment. All six archived members have positive increments.",
        "- The O-EnKF replay reproduces archived station increments at r > 0.99998. Broad western proxy innovations, not gain-sign reversal or a few records, drive the positive update.",
        "",
        "## Why no correction is accepted",
        "",
        "- Station residuals transfer spatially within 1901-1920, but the amplitude is not temporally stationary; carrying it into 1940-1950 worsens qcf MAE by 80.9%.",
        "- Candidate v2 improved period-mean qcf level but worsened median absolute qcf trend error by 169.8% in 1880-1900 and 135.2% in 1901-1920.",
        "- Eastern independent reconstructions reject a uniform historical cooling, while central-Himalaya numerical evidence remains unavailable. A deterministic HMA-wide or continuous western correction is therefore unsupported.",
        "",
        "## Boundaries",
        "",
        *[f"- {item}" for item in unresolved],
        "",
        "## Next route",
        "",
        "Obtain source-independent numeric western/central temperature evidence or, under a new explicit authorization, redesign the reconstruction with station-aware low-frequency and seasonal PSM validation. Do not apply the rejected post-hoc Candidate v2 field.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
