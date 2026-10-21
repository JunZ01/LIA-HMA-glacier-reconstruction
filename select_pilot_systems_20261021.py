"""Freeze the stratified pilot list for the volume-tau route (fact 1499).

Rule, fixed before any pilot result exists:
  * population: the legacy 191 Himalayan systems with one modern member
    (the full-chain exporter reads one modern gdir per system; multi-member
    systems are counted and reported, not silently dropped);
  * strata: region x modern-area tercile within region;
  * picks: three per stratum at the 1/6, 1/2 and 5/6 positions of the
    area-sorted stratum (lia_unique_id breaks ties); the three existing
    controls 3632/3131/2906 replace the middle pick of their stratum so
    their exports and results are reused.
Output: pilot_systems.csv plus a JSON summary of population coverage.

Usage (repo root, any Python with pandas):
    python select_pilot_systems_20261021.py OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path("H:/ToYangxin/result_yx")
AUD = BASE / "hma_harmonized_analysis_20261012_v1/legacy_volume_comparability"
PAIRING = AUD / "legacy191_source_pairing_diagnostic.csv"
REGIONS = BASE / "mean_thickness829_20261020_v1/mean_thickness_per_system829.csv"
MEMBERS = AUD / "modern_member_array_audit.csv"
EXISTING_CONTROLS = (3632, 3131, 2906)
PICK_POSITIONS = (1 / 6, 1 / 2, 5 / 6)


def select(pairing: pd.DataFrame, regions: pd.DataFrame, members: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    population = pairing.merge(regions[["lia_unique_id", "region", "region_zh"]], on="lia_unique_id")
    population["member_type"] = population.modern_member_count.map(lambda n: "single" if n == 1 else "multi")
    coverage = {
        "population_systems": int(len(population)),
        "by_region_member_type": population.groupby(["region", "member_type"]).size().to_dict(),
    }
    single = population[population.member_type == "single"].copy()
    single["area_km2"] = single.modern_outline_union_area_manifest_km2.astype(float)
    picks = []
    for region, g in single.groupby("region"):
        g = g.sort_values(["area_km2", "lia_unique_id"]).reset_index(drop=True)
        g["area_tercile"] = pd.qcut(g.index, 3, labels=["small", "medium", "large"]).astype(str)
        for tercile, t in g.groupby("area_tercile", sort=False):
            t = t.reset_index(drop=True)
            positions = sorted({min(len(t) - 1, int(p * len(t))) for p in PICK_POSITIONS})
            chosen = t.iloc[positions].copy()
            chosen["selection_reason"] = "stratum-position"
            control = t[t.lia_unique_id.isin(EXISTING_CONTROLS)]
            if len(control):
                middle = chosen.index[len(chosen) // 2]
                chosen = chosen.drop(index=middle)
                c = control.iloc[[0]].copy()
                c["selection_reason"] = "existing-control-reused"
                chosen = pd.concat([chosen, c])
            chosen["region"] = region
            chosen["area_tercile"] = tercile
            picks.append(chosen)
    pilot = pd.concat(picks).drop_duplicates("lia_unique_id")
    member = members.drop_duplicates("lia_unique_id").set_index("lia_unique_id")
    pilot["rgi_id"] = pilot.lia_unique_id.map(member.rgi_id)
    pilot["gdir"] = pilot.lia_unique_id.map(member.source_path).map(lambda p: str(Path(p).parent))
    pilot = pilot.rename(columns={"lia_netcdf_path": "lia_netcdf",
                                  "modern_reference_native_sum_km3": "native_modern_reference_volume_km3"})
    columns = ["lia_unique_id", "region", "region_zh", "area_tercile", "area_km2", "modern_member_count",
               "rgi_id", "gdir", "lia_netcdf", "native_modern_reference_volume_km3", "selection_reason"]
    pilot = pilot[columns].sort_values(["region", "area_km2"]).reset_index(drop=True)
    multi = population[population.member_type == "multi"]
    coverage.update({
        "pilot_systems": int(len(pilot)),
        "existing_controls_included": sorted(int(u) for u in pilot.lia_unique_id if u in EXISTING_CONTROLS),
        "multi_member_systems_not_covered": int(len(multi)),
        "multi_member_share_of_population_percent": round(100 * len(multi) / max(len(population), 1), 1),
        "multi_member_note": "exporter reads one modern gdir per system; multi-member systems need an exporter extension before they can enter a pilot",
    })
    return pilot, coverage


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    pilot, coverage = select(pd.read_csv(PAIRING), pd.read_csv(REGIONS), pd.read_csv(MEMBERS))
    pilot.to_csv(out / "pilot_systems.csv", index=False, encoding="utf-8-sig")
    coverage["by_region_member_type"] = {f"{k[0]}|{k[1]}": v for k, v in coverage["by_region_member_type"].items()}
    (out / "pilot_selection_summary.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pilot[["lia_unique_id", "region", "area_tercile", "area_km2", "selection_reason"]].to_string(index=False))
    print(json.dumps(coverage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
