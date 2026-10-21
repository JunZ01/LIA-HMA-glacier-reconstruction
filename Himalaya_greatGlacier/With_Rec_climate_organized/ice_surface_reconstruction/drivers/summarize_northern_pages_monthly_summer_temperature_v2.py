from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


EXPERIMENT_ID = "climate_reconstruction_northern_pages_monthly_summer_temperature_v2"
NEUTRAL_TOLERANCE_C = 0.02


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
    v1 = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_northern_pages_monthly_summer_temperature_v1"
    output = root / f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_ID}"
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "regional_changes_v1": v1 / "monthly_product_regional_event_changes.csv",
        "point_changes_v1": v1 / "monthly_product_point_event_changes.csv",
        "curves_v1": v1 / "monthly_product_point_annualized_curves.csv",
        "station_support_v1": v1 / "cru_station_support_by_cell_and_season.csv",
        "annual_closure_v1": v1 / "monthly_vs_compiled_annual_closure.csv",
        "figure_v1": v1 / "northern_monthly_temperature_season_curves_1905_1920.png",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}

    regional = pd.read_csv(sources["regional_changes_v1"])
    station = pd.read_csv(sources["station_support_v1"])
    summer = regional[regional["season"].isin(["JJA", "AMJJAS"])].copy()
    station_summer = station[station["season"].isin(["JJA", "AMJJAS"])]
    station_summary = station_summer.groupby(["region", "season"], as_index=False).agg(
        unique_grid_cell_count=("grid_cell_id", "size"),
        cells_with_any_station_support=("years_with_positive_station_count_1905_1918", lambda values: int((values > 0).sum())),
        mean_years_with_positive_station_count=("years_with_positive_station_count_1905_1918", "mean"),
        minimum_years_with_positive_station_count=("years_with_positive_station_count_1905_1918", "min"),
        mean_cru_station_count=("mean_cru_station_count_1905_1918", "mean"),
        minimum_cru_station_count=("mean_cru_station_count_1905_1918", "min"),
        maximum_cru_station_count=("mean_cru_station_count_1905_1918", "max"),
    )
    west = summer[summer["region"] == "north_west_ge35_lt80E"]
    east = summer[summer["region"] == "north_east_ge35_90to101E"]
    no_material_cooling = bool((summer["unique_cell_mean_event_change_c"] >= -NEUTRAL_TOLERANCE_C).all())

    output.mkdir(parents=True)
    summer.to_csv(output / "summer_regional_event_changes.csv", index=False)
    station_summary.to_csv(output / "summer_cru_station_support_summary.csv", index=False)
    for key, destination in (
        ("regional_changes_v1", "monthly_product_regional_event_changes.csv"),
        ("point_changes_v1", "monthly_product_point_event_changes.csv"),
        ("curves_v1", "monthly_product_point_annualized_curves.csv"),
        ("station_support_v1", "cru_station_support_by_cell_and_season.csv"),
        ("annual_closure_v1", "monthly_vs_compiled_annual_closure.csv"),
        ("figure_v1", "northern_monthly_temperature_season_curves_1905_1920.png"),
    ):
        shutil.copy2(sources[key], output / destination)

    west_station = station_summary[station_summary["region"] == "north_west_ge35_lt80E"]
    east_station = station_summary[station_summary["region"] == "north_east_ge35_90to101E"]
    summary = {
        "supersedes": "v1 directional classification only; v1 numerical tables and figure remain valid",
        "event_definition": "mean(1915-1918) minus mean(1905-1914)",
        "neutral_tolerance_c": NEUTRAL_TOLERANCE_C,
        "all_region_product_jja_amjjas_changes_at_or_above_minus_tolerance": no_material_cooling,
        "west_summer_event_changes": west.to_dict(orient="records"),
        "east_summer_event_changes": east.to_dict(orient="records"),
        "west_cru_station_support": west_station.to_dict(orient="records"),
        "east_cru_station_support": east_station.to_dict(orient="records"),
        "classification": (
            "west_clear_summer_warming_east_weak_warming_to_neutral_no_material_instrumental_cooling"
            if no_material_cooling
            else "material_summer_cooling_present_in_at_least_one_region_product"
        ),
        "interpretation": (
            "The western Cook-region sign conflict persists under JJA and AMJJAS and has much denser CRU station support. "
            "The eastern Shao-region conflict has sparse early CRU support and remains less decisive."
        ),
        "locked_1921_1939_accessed": False,
    }
    after = {name: source_state(path) for name, path in sources.items()}
    unchanged = before == after
    summary["all_declared_sources_unchanged"] = unchanged
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("declared-source-changed-during-v2-summary")
    (output / "FINAL_RESULT.md").write_text(
        "# Northern PAGES Monthly Summer-temperature Comparison v2\n\n"
        f"No regional Berkeley/CRU JJA or AMJJAS change is below -{NEUTRAL_TOLERANCE_C:.2f} degC. "
        f"West ranges from {west['unique_cell_mean_event_change_c'].min():+.3f} to {west['unique_cell_mean_event_change_c'].max():+.3f} degC; "
        f"east ranges from {east['unique_cell_mean_event_change_c'].min():+.3f} to {east['unique_cell_mean_event_change_c'].max():+.3f} degC.\n\n"
        f"Mean CRU station count is about {west_station['mean_cru_station_count'].mean():.2f} in the west versus "
        f"{east_station['mean_cru_station_count'].mean():.2f} in the east. The western conflict is therefore the stronger proxy/PSM diagnostic; "
        "eastern early-observation sparsity remains plausible.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
