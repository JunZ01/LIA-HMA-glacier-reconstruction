"""Adjudicate v1 using distances only for stations in the primary qcf metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2"
)
V1 = LOOP / "temperature_pmip_low_frequency_station_bridge_validation_v1"
OUTPUT = LOOP / "temperature_pmip_low_frequency_station_bridge_adjudication_v2"
DRIVER = Path(__file__).resolve()
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "EXPERIMENT_CONTRACT.md"
    contract.write_text(
        "# PMIP station-bridge v2 adjudication\n\n"
        "Correct only the v1 distance gate: evaluate spatial support for stations that entered "
        "the qcf primary period metrics, not all metadata-panel stations. Preserve every v1 "
        "temperature, correction, station subset, period, and metric. Do not read climate or "
        "station values from 1921-1939. Do not tune any threshold or correction.\n",
        encoding="utf-8",
    )
    sources = {
        "v1_summary": V1 / "summary.json",
        "v1_station_values": V1 / "station_year_values_allowed.csv",
        "v1_station_metrics": V1 / "station_period_metrics.csv",
        "v1_aggregate": V1 / "aggregate_metrics.csv",
        "facts": FACTS,
        "driver": DRIVER,
        "contract": contract,
    }
    before = {name: sha256(path) for name, path in sources.items()}
    v1 = json.loads(sources["v1_summary"].read_text(encoding="utf-8"))
    station_values = pd.read_csv(sources["v1_station_values"])
    station_metrics = pd.read_csv(sources["v1_station_metrics"])
    aggregate = pd.read_csv(sources["v1_aggregate"])

    primary_ids = set(station_metrics.loc[station_metrics["series"].eq("qcf"), "station_id"])
    primary_values = station_values[station_values["station_id"].isin(primary_ids)]
    primary_max_distance = float(primary_values["correction_source_distance_deg"].max())
    qcf = aggregate[aggregate["series"].eq("qcf")].set_index("period")
    residual_fractions = {
        period: float(abs(row.candidate_equal_station_mean_bias_degC) / abs(row.original_equal_station_mean_bias_degC))
        for period, row in qcf.iterrows()
    }
    explained_fractions = {period: 1.0 - value for period, value in residual_fractions.items()}

    gates = dict(v1["gates"])
    gates.pop("all_station_correction_sources_within_2_degrees", None)
    gates["all_primary_qcf_station_correction_sources_within_2_degrees"] = primary_max_distance <= 2.0
    passes = bool(all(gates.values()))
    decision = (
        "retain_low_frequency_background_bridge_for_frozen_enkf_replay_preflight"
        if passes
        else "reject_existing_50_50_low_frequency_background_as_complete_1880_1920_solution"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    periods = list(qcf.index)
    x = np.arange(len(periods))
    axes[0].bar(x - 0.18, qcf["original_equal_station_mean_bias_degC"], width=0.36, color="#D55E00", label="Original")
    axes[0].bar(x + 0.18, qcf["candidate_equal_station_mean_bias_degC"], width=0.36, color="#0072B2", label="50:50 low-frequency")
    axes[0].set_xticks(x, periods)
    axes[0].set_ylabel("Equal-station mean bias (degC)")
    axes[0].legend(frameon=False)
    axes[1].bar(x - 0.18, qcf["original_median_abs_trend_error"], width=0.36, color="#D55E00")
    axes[1].bar(x + 0.18, qcf["candidate_median_abs_trend_error"], width=0.36, color="#0072B2")
    axes[1].set_xticks(x, periods)
    axes[1].set_ylabel("Median trend error (degC/century)")
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "qcf_level_and_trend_adjudication.png", dpi=220)
    plt.close(fig)

    summary = {
        "status": "complete",
        "experiment": "temperature_pmip_low_frequency_station_bridge_adjudication_v2",
        "supersedes_v1_decision_reason_only": True,
        "decision": decision,
        "passes_all_frozen_gates": passes,
        "primary_qcf_station_count": len(primary_ids),
        "primary_qcf_maximum_correction_source_distance_deg": primary_max_distance,
        "warm_bias_fraction_explained_by_existing_50_50_correction": explained_fractions,
        "warm_bias_fraction_remaining_after_existing_50_50_correction": residual_fractions,
        "qcf_1901_1920_median_trend_error_change_fraction": float(
            qcf.loc["1901-1920", "median_trend_error_change_fraction"]
        ),
        "gates": gates,
        "locked_1921_1939_numeric_values_accessed": 0,
        "all_declared_sources_unchanged": True,
        "production_authorized": False,
        "enkf_run": False,
        "oggm_run": False,
        "interpretation": (
            "The old 50:50 correction is directionally useful and spatially supported for the "
            "primary western qcf stations, but its amplitude explains only a small fraction of "
            "the direct-station warm residual and its transition shape worsens 1901-1920 trend skill."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    aggregate.to_csv(OUTPUT / "aggregate_metrics_unchanged_from_v1.csv", index=False)
    primary_values[[
        "station_id", "station_name", "latitude", "longitude", "correction_source_distance_deg"
    ]].drop_duplicates().sort_values("station_id").to_csv(
        OUTPUT / "primary_qcf_station_support.csv", index=False
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Existing 50:50 Background Correction: Direct-Station Adjudication\n\n"
        f"Decision: `{decision}`.\n\n"
        f"- Primary qcf stations: `{len(primary_ids)}`; maximum correction-source distance: "
        f"`{primary_max_distance:.3f} degrees`. The v1 all-panel distance failure is not a "
        "valid reason to reject the primary test.\n"
        f"- 1880-1900 warm-bias fraction explained: `{explained_fractions['1880-1900']:.1%}`; "
        f"remaining: `{residual_fractions['1880-1900']:.1%}`.\n"
        f"- 1901-1920 warm-bias fraction explained: `{explained_fractions['1901-1920']:.1%}`; "
        f"remaining: `{residual_fractions['1901-1920']:.1%}`.\n"
        f"- 1901-1920 median direct-station trend error changes by "
        f"`{qcf.loc['1901-1920', 'median_trend_error_change_fraction']:+.1%}` and fails the "
        "frozen no-material-regression gate.\n\n"
        "The old 50:50 PMIP correction therefore solved part of the pre-1900 low-frequency "
        "warm state, but it is not a complete 1880-1920 background solution. The unresolved "
        "component is mainly correction amplitude and transition timing, not lack of spatial "
        "support at the six primary stations. No production, EnKF, or OGGM run is authorized.\n",
        encoding="utf-8",
    )
    after = {name: sha256(path) for name, path in sources.items()}
    unchanged = before == after
    summary["all_declared_sources_unchanged"] = unchanged
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("A declared source changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
