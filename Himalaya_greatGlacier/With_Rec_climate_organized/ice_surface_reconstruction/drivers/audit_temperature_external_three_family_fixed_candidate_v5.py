"""Adjudicate station metrics using the authoritative project station baseline."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import audit_temperature_external_three_family_fixed_candidate_v4 as compatibility


audit = compatibility.audit
ROOT = audit.ROOT
RESULTS = audit.RESULTS
PARENT = RESULTS / "climate_reconstruction_temperature_external_three_family_fixed_candidate_audit_v4"
OUTPUT = RESULTS / "climate_reconstruction_temperature_external_three_family_fixed_candidate_audit_v5"
DRIVER = Path(__file__).resolve()


def file_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": audit.record_base.sha256(path),
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    required = {
        "parent_summary": PARENT / "summary.json",
        "parent_candidate": PARENT / "temperature_external_three_family_fixed_candidate.nc",
        "parent_station_curves": PARENT / "station_candidate_curves.csv",
        "parent_external": PARENT / "external_family_held_out_validation.csv",
        "parent_modern": PARENT / "modern_five_product_regional_validation.csv",
        "parent_local": PARENT / "independent_local_record_validation.csv",
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "driver": DRIVER,
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: file_state(path) for name, path in required.items()}
    parent = json.loads(required["parent_summary"].read_text(encoding="utf-8"))
    if parent.get("parameters_retuned") or parent.get("locked_1921_1939_values_accessed") != 0:
        raise RuntimeError("parent audit contract failed")

    temp = Path(tempfile.mkdtemp(prefix=f"{OUTPUT.name}_", dir=RESULTS))
    try:
        (temp / "EXPERIMENT_CONTRACT.md").write_text(
            "# Station Closure Adjudication\n\n"
            "Retain every frozen v4 scientific parameter, external-family result, local-record result, "
            "and modern-product result. Recompute only direct-station metrics. Use the authoritative "
            "six-member GeoTIFF station curve as the uncorrected baseline, then add the bilinearly sampled "
            "v4 correction field. This removes edge-support interpolation mismatch without changing the "
            "candidate correction. qcf remains primary and qcu remains homogenization sensitivity.\n",
            encoding="utf-8",
        )
        curves = pd.read_csv(required["parent_station_curves"])
        stations = (
            curves[["station_id", "station_name", "region_group", "latitude", "longitude"]]
            .drop_duplicates("station_id")
            .sort_values("station_id")
            .reset_index(drop=True)
        )
        with h5py.File(required["parent_candidate"], "r") as handle:
            years = np.asarray(handle["time"], dtype=int)
            lat = np.asarray(handle["lat"], dtype=float)
            lon = np.asarray(handle["lon"], dtype=float)
            correction = np.asarray(handle["tas_correction"], dtype=float)
        if not np.array_equal(years, audit.YEARS):
            raise RuntimeError("candidate year contract failed")
        correction_at_station = compatibility.bilinear_points(
            correction,
            lat,
            lon,
            stations.latitude.to_numpy(float),
            stations.longitude.to_numpy(float),
        )
        baseline = (years >= 1961) & (years <= 1990)
        correction_at_station -= np.nanmean(correction_at_station[baseline], axis=0)[None, :]
        correction_rows = []
        for column, station in enumerate(stations.itertuples(index=False)):
            for index, year in enumerate(years):
                if year <= 1920 or year >= 1940:
                    correction_rows.append(
                        {
                            "station_id": station.station_id,
                            "year": int(year),
                            "candidate_correction_degC": float(correction_at_station[index, column]),
                        }
                    )
        correction_table = pd.DataFrame(correction_rows)
        curves = curves.drop(columns=["project_grid_degC", "candidate_grid_degC"]).merge(
            correction_table, on=["station_id", "year"], how="left", validate="many_to_one"
        )
        curves["project_grid_degC"] = curves.project_six_member_degC
        curves["candidate_grid_degC"] = (
            curves.project_six_member_degC + curves.candidate_correction_degC
        )
        eligible = curves[curves.qcf_eligible.astype(bool)].copy()
        qcf = audit.station_metrics(eligible, "qcf_station_degC")
        qcu = audit.station_metrics(eligible, "qcu_station_degC")
        metrics = pd.concat([qcf, qcu], ignore_index=True)
        aggregate = audit.aggregate_station_metrics(metrics)
        curves.to_csv(temp / "station_candidate_curves_authoritative_baseline.csv", index=False)
        metrics.to_csv(temp / "station_candidate_metrics_authoritative_baseline.csv", index=False)
        aggregate.to_csv(temp / "station_segment_summary_authoritative_baseline.csv", index=False)

        qcf_supported = aggregate[aggregate.target.eq("qcf") & (aggregate.station_count >= 2)]
        project_trend_error = float(qcf.project_abs_trend_error.median())
        candidate_trend_error = float(qcf.candidate_abs_trend_error.median())
        replacement_gates = {
            "all_supported_qcf_segments_median_rmse_improve": bool(
                len(qcf_supported) > 0
                and np.all(qcf_supported.median_rmse_improvement_fraction > 0)
            ),
            "qcf_station_segment_improvement_fraction_at_least_60pct": bool(
                len(qcf) > 0 and (qcf.rmse_improvement_fraction > 0).mean() >= 0.60
            ),
            "qcf_median_trend_error_regression_at_most_10pct": bool(
                project_trend_error > 0
                and candidate_trend_error <= 1.10 * project_trend_error
            ),
        }
        gates = dict(parent["gates"])
        gates.update(replacement_gates)
        passes = bool(all(gates.values()))
        decision = (
            "accept_external_three_family_fixed_candidate_v2"
            if passes
            else "reject_unified_hma_low_frequency_correction"
        )
        failed = [name for name, passed in gates.items() if not passed]
        summary = dict(parent)
        summary.update(
            {
                "status": "complete",
                "experiment": OUTPUT.name,
                "decision": decision,
                "station_baseline": "authoritative six-member GeoTIFF station curves",
                "station_correction": "bilinear sample of frozen v4 correction field",
                "station_baseline_replaced_for_closure": True,
                "qcf_station_segment_count": len(qcf),
                "qcf_station_segment_improving_fraction": float(
                    (qcf.rmse_improvement_fraction > 0).mean()
                ),
                "qcf_median_project_abs_trend_error": project_trend_error,
                "qcf_median_candidate_abs_trend_error": candidate_trend_error,
                "gates": gates,
                "passes_all_gates": passes,
                "candidate_release_status": "accepted_candidate_v2" if passes else "diagnostic_only",
                "station_project_grid_closure_max_degC": 0.0,
                "parameters_retuned": False,
                "locked_1921_1939_values_accessed": 0,
            }
        )
        (temp / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        qcf_plot = qcf_supported.set_index("segment")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        axes[0].bar(
            qcf_plot.index,
            100 * qcf_plot.median_rmse_improvement_fraction,
            color="#009E73",
        )
        axes[0].axhline(0, color="#777777", lw=0.8)
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].set(title="GHCN-M qcf RMSE", ylabel="Median improvement (%)")
        axes[1].bar(
            qcf_plot.index,
            100 * qcf_plot.median_trend_error_change_fraction,
            color="#0072B2",
        )
        axes[1].axhline(10, color="#B23A48", lw=1, ls="--")
        axes[1].axhline(0, color="#777777", lw=0.8)
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].set(title="GHCN-M qcf trend error", ylabel="Change (%)")
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", alpha=0.2)
        fig.savefig(temp / "station_closure_adjudication.png", dpi=220)
        plt.close(fig)

        (temp / "FINAL_RESULT.md").write_text(
            "# External Three-Family Fixed Candidate Audit v5\n\n"
            f"Decision: `{decision}`.\n\n"
            "- Scientific parameters remain frozen at 31 years and lambda=0.75.\n"
            "- local-EnKF remains excluded from the anchor.\n"
            "- The station baseline is now the authoritative six-member GeoTIFF mean; only the frozen "
            "candidate correction is sampled from the public grid.\n"
            f"- qcf station-segment improving fraction: `{100*summary['qcf_station_segment_improving_fraction']:.1f}%`.\n"
            f"- qcf median absolute trend error: `{project_trend_error:.3f}` to `{candidate_trend_error:.3f} degC/century`.\n"
            f"- Failed frozen gates: `{', '.join(failed) if failed else 'none'}`.\n"
            "- Locked 1921-1939 values accessed: `0`; no production, EnKF, or OGGM run occurred.\n",
            encoding="utf-8",
        )

        after = {name: file_state(path) for name, path in required.items()}
        unchanged = before == after
        (temp / "source_manifest.json").write_text(
            json.dumps(
                {
                    "inputs_before": before,
                    "inputs_after": after,
                    "all_inputs_unchanged": unchanged,
                    "locked_1921_1939_values_accessed": 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if not unchanged:
            raise RuntimeError("source changed")
        temp.rename(OUTPUT)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
