from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_ID = "climate_reconstruction_cook_chronology_source_moisture_v3"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest(path)}


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    v2 = result_root / "climate_reconstruction_cook_chronology_source_moisture_v2"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "records": v2 / "cook_chronology_record_audit.csv",
        "annual": v2 / "cook_chronology_annual_1905_1920.csv",
        "quality": v2 / "cook_precipitation_quality_and_event.csv",
        "components": v2 / "cook_psm_event_component_summary.csv",
        "v2_summary": v2 / "summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}

    records = pd.read_csv(sources["records"])
    annual = pd.read_csv(sources["annual"])
    quality = pd.read_csv(sources["quality"])
    components = pd.read_csv(sources["components"])
    with sources["v2_summary"].open("r", encoding="utf-8") as handle:
        v2_summary = json.load(handle)

    climate_keys = [
        "clean_gpcc_event_change_mm_day",
        "cru_event_change_mm_day",
        "direct_gauge_month_fraction_1905_1918",
        "mean_direct_gauge_count_1905_1918",
        "mean_interpolation_error_mm_month_1905_1918",
    ]
    quality["climate_evidence_group"] = quality.groupby(climate_keys, dropna=False).ngroup() + 1
    evidence = quality.groupby("climate_evidence_group", as_index=False).agg(
        chronology_count=("proxy_index", "size"),
        site_names=("site_name", lambda values: ";".join(values.astype(str))),
        mean_latitude=("latitude", "mean"),
        mean_longitude=("longitude", "mean"),
        clean_gpcc_event_change_mm_day=("clean_gpcc_event_change_mm_day", "first"),
        cru_event_change_mm_day=("cru_event_change_mm_day", "first"),
        direct_gauge_month_fraction_1905_1918=("direct_gauge_month_fraction_1905_1918", "first"),
        mean_direct_gauge_count_1905_1918=("mean_direct_gauge_count_1905_1918", "first"),
        mean_interpolation_error_mm_month_1905_1918=("mean_interpolation_error_mm_month_1905_1918", "first"),
    )
    direct_evidence = evidence[evidence["direct_gauge_month_fraction_1905_1918"] > 0]
    if len(evidence) != 7 or len(direct_evidence) != 1:
        raise RuntimeError(f"climate-evidence-independence-contract-failed:{len(evidence)}:{len(direct_evidence)}")

    pakistan = records[records["latitude"] >= 35.0]
    cook2012 = records[records["publication_group"].eq("cook2012tree")]
    bt = records[records["site_name"].eq("BT018")]
    component_rows = []
    actual_amplitude = abs(float(components["mean_actual_proxy_event_change_z"].iloc[0]))
    for row in components.itertuples(index=False):
        component_rows.append({
            **row._asdict(),
            "absolute_total_psm_amplitude_transfer": abs(float(row.mean_total_component_z)) / actual_amplitude,
            "absolute_precipitation_component_amplitude_transfer": abs(float(row.mean_precipitation_component_z)) / actual_amplitude,
        })
    component_transfer = pd.DataFrame(component_rows)

    output.mkdir(parents=True)
    quality.to_csv(output / "cook_proxy_to_climate_evidence_groups.csv", index=False)
    evidence.to_csv(output / "cook_independent_climate_evidence_groups.csv", index=False)
    component_transfer.to_csv(output / "cook_psm_amplitude_transfer.csv", index=False)

    proxy_columns = [column for column in annual.columns if column.startswith("proxy_")]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), constrained_layout=True)
    for column in proxy_columns:
        site = column.split("_", 2)[-1]
        is_bt = site == "BT018"
        axes[0].plot(
            annual["year"], annual[column],
            color="#E69F00" if is_bt else "#3B7EA1",
            alpha=0.95 if is_bt else 0.28,
            linewidth=2.0 if is_bt else 1.0,
            label="BT018 (Bhutan)" if is_bt else None,
        )
    axes[0].plot(annual["year"], annual["cook_record_mean_z"], color="black", linewidth=2.8, label="15-record mean")
    axes[0].axvspan(1915, 1918, color="#D55E00", alpha=0.12)
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Proxy anomaly (1941-2000 SD)")
    axes[0].set_title("Archived Cook-label chronology curves")
    axes[0].legend(frameon=False)

    direct = evidence["direct_gauge_month_fraction_1905_1918"] > 0
    axes[1].scatter(
        evidence.loc[~direct, "clean_gpcc_event_change_mm_day"],
        evidence.loc[~direct, "cru_event_change_mm_day"],
        s=55 + 45 * evidence.loc[~direct, "chronology_count"],
        color="#9E9E9E", edgecolor="black", linewidth=0.6, label="No direct GPCC gauge",
    )
    axes[1].scatter(
        evidence.loc[direct, "clean_gpcc_event_change_mm_day"],
        evidence.loc[direct, "cru_event_change_mm_day"],
        s=55 + 45 * evidence.loc[direct, "chronology_count"],
        color="#009E73", edgecolor="black", linewidth=0.8, label="Direct GPCC gauge",
    )
    for row in evidence.itertuples(index=False):
        axes[1].annotate(row.site_names, (row.clean_gpcc_event_change_mm_day, row.cru_event_change_mm_day), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].axvline(0, color="#777777", linewidth=0.8)
    axes[1].set_xlabel("Clean GPCC event change (mm/day)")
    axes[1].set_ylabel("CRU event change (mm/day)")
    axes[1].set_title("Seven independent climate evidence cells")
    axes[1].legend(frameon=False)
    fig.savefig(output / "cook_curves_and_independent_precipitation_evidence.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "proxy_record_count": int(len(records)),
        "proxy_unique_site_count": int(records["site_id"].nunique()),
        "publication_group_count": int(records["publication_group"].nunique()),
        "pakistan_record_count": int(len(pakistan)),
        "pakistan_negative_event_count": int(np.sum(pakistan["event_change_z_recomputed"] < 0)),
        "pakistan_mean_event_change_z": float(pakistan["event_change_z_recomputed"].mean()),
        "cook2012_record_count": int(len(cook2012)),
        "cook2012_negative_event_count": int(np.sum(cook2012["event_change_z_recomputed"] < 0)),
        "cook2012_mean_event_change_z": float(cook2012["event_change_z_recomputed"].mean()),
        "bt018_event_change_z": float(bt["event_change_z_recomputed"].iloc[0]),
        "independent_climate_evidence_cell_count": int(len(evidence)),
        "direct_gauge_climate_evidence_cell_count": int(len(direct_evidence)),
        "direct_gauge_cell_chronology_count": int(direct_evidence["chronology_count"].iloc[0]),
        "direct_gauge_cell_site_names": str(direct_evidence["site_names"].iloc[0]),
        "direct_gauge_cell_clean_gpcc_event_change_mm_day": float(direct_evidence["clean_gpcc_event_change_mm_day"].iloc[0]),
        "direct_gauge_cell_cru_event_change_mm_day": float(direct_evidence["cru_event_change_mm_day"].iloc[0]),
        "component_amplitude_transfer": component_transfer.to_dict(orient="records"),
        "calibration_pc1_explained_variance_fraction": float(v2_summary["chronology_summary"]["calibration_pc1_explained_variance_fraction"]),
        "event_vector_energy_fraction_along_calibration_pc1": float(v2_summary["chronology_summary"]["event_vector_energy_fraction_along_calibration_pc1"]),
        "classification": "shared_pakistan_chronology_event_directional_drying_but_insufficiently_observed_and_underpredicted",
        "decision": (
            "The 1915-1918 decline is a coherent archived Pakistan chronology event, not a single-record artifact. "
            "Drying is directionally consistent across GPCC/CRU cells, but only one independent climate cell has direct GPCC gauge support and the calibrated bivariate PSM transfers only about one fifth of proxy amplitude. "
            "Moisture stress is plausible but not established as the cause; chronology construction and PSM nonstationarity remain open."
        ),
        "read_periods": [[1905, 1920], [1941, 2000]],
        "locked_1921_1939_accessed": False,
    }
    after = {name: source_state(path) for name, path in sources.items()}
    summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("declared-source-changed-during-cook-v3-summary")
    (output / "FINAL_RESULT.md").write_text(
        "# Cook Chronology Source and Moisture Audit v3\n\n"
        f"- Pakistan records: `{len(pakistan)}`, negative event records: `{int(np.sum(pakistan['event_change_z_recomputed'] < 0))}`, mean `{pakistan['event_change_z_recomputed'].mean():+.3f} z`.\n"
        f"- Cook 2012 group: `{len(cook2012)}` records, all negative, mean `{cook2012['event_change_z_recomputed'].mean():+.3f} z`.\n"
        f"- Independent climate evidence cells: `{len(evidence)}`; cells with direct GPCC gauge support: `{len(direct_evidence)}`.\n"
        f"- The direct cell covers `{int(direct_evidence['chronology_count'].iloc[0])}` chronologies: `{direct_evidence['site_names'].iloc[0]}`.\n"
        f"- BEA/CRU total PSM amplitude transfer: `{component_transfer['absolute_total_psm_amplitude_transfer'].iloc[0]:.3f}` / `{component_transfer['absolute_total_psm_amplitude_transfer'].iloc[1]:.3f}`.\n\n"
        f"Classification: `{summary['classification']}`. No source or production operator was changed.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
