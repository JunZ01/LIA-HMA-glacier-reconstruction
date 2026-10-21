from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


EXPERIMENT_ID = "climate_reconstruction_northern_pages_annual_temperature_products_v2"
OBSERVATIONS = ("Berkeley Earth", "CRU TS", "GISTEMP", "HadCRUT5", "NOAA")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest(path),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    v1 = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_northern_pages_annual_temperature_products_v1"
    output = root / f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_ID}"
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "regional_changes_v1": v1 / "regional_temperature_product_event_changes.csv",
        "cell_changes_v1": v1 / "grid_cell_temperature_event_changes.csv",
        "annual_curves_v1": v1 / "regional_annual_temperature_curves_1905_1920.csv",
        "grid_mapping_v1": v1 / "north_proxy_to_unique_temperature_grid_mapping.csv",
        "figure_v1": v1 / "northern_temperature_products_1905_1920.png",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}

    changes = pd.read_csv(sources["regional_changes_v1"])
    observations = changes[changes["product"].isin(OBSERVATIONS)].copy()
    rows = []
    for region, group in observations.groupby("region"):
        full = group[group["complete_grid_cell_fraction"].eq(1.0)]
        partial = group[group["complete_grid_cell_fraction"].lt(1.0)]
        rows.append({
            "region": region,
            "observation_product_count": int(len(group)),
            "full_spatial_coverage_product_count": int(len(full)),
            "partial_spatial_coverage_product_count": int(len(partial)),
            "full_coverage_product_names": ";".join(full["product"]),
            "partial_coverage_product_names": ";".join(partial["product"]),
            "negative_full_coverage_product_count": int((full["unique_cell_mean_event_change_c"] < 0).sum()),
            "positive_full_coverage_product_count": int((full["unique_cell_mean_event_change_c"] > 0).sum()),
            "full_coverage_product_mean_event_change_c": float(full["unique_cell_mean_event_change_c"].mean()),
            "full_coverage_product_min_event_change_c": float(full["unique_cell_mean_event_change_c"].min()),
            "full_coverage_product_max_event_change_c": float(full["unique_cell_mean_event_change_c"].max()),
            "all_available_product_mean_event_change_c": float(group["unique_cell_mean_event_change_c"].mean()),
        })
    consensus = pd.DataFrame(rows).sort_values("region")
    west = consensus[consensus["region"] == "north_west_ge35_lt80E"].iloc[0]
    east = consensus[consensus["region"] == "north_east_ge35_90to101E"].iloc[0]
    support_warming = bool(
        int(west["full_spatial_coverage_product_count"]) >= 4
        and int(east["full_spatial_coverage_product_count"]) >= 4
        and int(west["negative_full_coverage_product_count"]) == 0
        and int(east["negative_full_coverage_product_count"]) == 0
    )

    output.mkdir(parents=True)
    consensus.to_csv(output / "corrected_regional_observation_product_consensus.csv", index=False)
    shutil.copy2(sources["regional_changes_v1"], output / "regional_temperature_product_event_changes.csv")
    shutil.copy2(sources["cell_changes_v1"], output / "grid_cell_temperature_event_changes.csv")
    shutil.copy2(sources["annual_curves_v1"], output / "regional_annual_temperature_curves_1905_1920.csv")
    shutil.copy2(sources["grid_mapping_v1"], output / "north_proxy_to_unique_temperature_grid_mapping.csv")
    shutil.copy2(sources["figure_v1"], output / "northern_temperature_products_1905_1920.png")

    summary = {
        "supersedes": "v1 summary coverage semantics; v1 numerical CSV values and figure remain valid",
        "event_definition": "mean(1915-1918) minus mean(1905-1914)",
        "aggregation": "nearest 0.5-degree grid cell for each proxy; duplicate cells removed; unique cells equally weighted",
        "annual_only_limitation": "The mostly Summer PAGES interpretation is not seasonally resolved by this annual product comparison.",
        "west_full_coverage_consensus": west.to_dict(),
        "east_full_coverage_consensus": east.to_dict(),
        "gistemp_east_coverage": "1 of 8 unique grid cells; positive at the one complete cell but excluded from full-coverage consensus",
        "classification": (
            "all_full_coverage_annual_observation_products_support_warming_in_both_northern_blocks"
            if support_warming
            else "full_coverage_annual_observation_products_do_not_have_a_common_event_sign"
        ),
        "interpretation": (
            "The negative proxy event is opposite to the annual instrumental-product sign. This does not decide whether summer climate, "
            "early station representativeness, chronology behavior, or PSM nonstationarity is responsible."
        ),
    }
    after = {name: source_state(path) for name, path in sources.items()}
    unchanged = before == after
    summary["all_declared_sources_unchanged"] = unchanged
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("declared-source-changed-during-v2-summary")
    (output / "FINAL_RESULT.md").write_text(
        "# Northern PAGES Annual Temperature-product Comparison v2\n\n"
        f"West: {int(west['positive_full_coverage_product_count'])}/{int(west['full_spatial_coverage_product_count'])} "
        f"full-coverage products show warming, mean {float(west['full_coverage_product_mean_event_change_c']):+.3f} degC. "
        f"East: {int(east['positive_full_coverage_product_count'])}/{int(east['full_spatial_coverage_product_count'])}, "
        f"mean {float(east['full_coverage_product_mean_event_change_c']):+.3f} degC.\n\n"
        "GISTEMP has complete 1905-1918 data at only 1/8 eastern grid cells and is not counted as a full-coverage eastern product. "
        "The plotted and tabulated v1 numerical values are unchanged.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
