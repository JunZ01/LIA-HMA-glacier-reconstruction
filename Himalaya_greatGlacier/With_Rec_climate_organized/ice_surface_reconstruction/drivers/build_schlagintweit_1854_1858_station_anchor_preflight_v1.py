from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUT = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2/"
    "schlagintweit_1854_1858_station_anchor_preflight_v1"
)
FACTS = ROOT / "LIA_PROJECT_FACTS.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    facts_before = sha256(FACTS)

    rows = [
        {
            "station": "Darjeeling_Observatory_Hill",
            "region": "east",
            "printed_pages": "480-485",
            "iiif_canvases": "0504-0509",
            "latitude": "27.05",
            "longitude": "88.25",
            "elevation_ft": "7168",
            "direct_period": "1855_to_1856",
            "usable_component": "four_fixed_daily_times_plus_self_registering_1855_04_to_1856_05",
            "anchor_class": "A_fixed_station",
            "exclude": "Jillapahar_Hodgson_bungalow_multiyear_composites",
        },
        {
            "station": "Massuri_Mary_Villa_Mackinnon",
            "region": "central_west",
            "printed_pages": "499-500",
            "iiif_canvases": "0523-0524",
            "latitude": "30.46",
            "longitude": "78.0513",
            "elevation_ft": "6715",
            "direct_period": "1855_to_1856",
            "usable_component": "observer_and_site_separable_values_only",
            "anchor_class": "B_conditional_fixed_station",
            "exclude": "merged_observer_or_site_values",
        },
        {
            "station": "Kotghar",
            "region": "central_west",
            "printed_pages": "504",
            "iiif_canvases": "0528",
            "latitude": "31.3167",
            "longitude": "77.4667",
            "elevation_ft": "6412",
            "direct_period": "1855_to_1856",
            "usable_component": "raw_times_for_sensitivity_only",
            "anchor_class": "BC_time_correction_sensitive",
            "exclude": "Simla_time_corrected_absolute_anchor",
        },
        {
            "station": "Kyelang",
            "region": "west",
            "printed_pages": "513",
            "iiif_canvases": "0537",
            "latitude": "32.5633",
            "longitude": "77.01",
            "elevation_ft": "10242",
            "direct_period": "1855_05_to_1855_10",
            "usable_component": "May_to_October_direct_season",
            "anchor_class": "A_seasonal",
            "exclude": "winter_months_filled_from_other_equal_elevation_sites",
        },
        {
            "station": "Leh_daily_period",
            "region": "west",
            "printed_pages": "530-531",
            "iiif_canvases": "0554-0555",
            "latitude": "34.1383",
            "longitude": "77.2433",
            "elevation_ft": "11532",
            "direct_period": "1856_07_to_1856_09",
            "usable_component": "subtable_A_fixed_station_hourly_period",
            "anchor_class": "A_seasonal_diurnal",
            "exclude": "subtable_B_annual_months_mixed_or_inferred_from_Kanam",
        },
        {
            "station": "Tonglo_Peak",
            "region": "east",
            "printed_pages": "486",
            "iiif_canvases": "0510",
            "latitude": "27.03",
            "longitude": "88.0663",
            "elevation_ft": "10080",
            "direct_period": "1855_05_short",
            "usable_component": "diurnal_and_vertical_gradient_only",
            "anchor_class": "C_short_campaign",
            "exclude": "annual_or_decadal_level",
        },
        {
            "station": "Jhosimath_Badrinath_Niti_route",
            "region": "central_west",
            "printed_pages": "501",
            "iiif_canvases": "0525",
            "latitude": "route",
            "longitude": "route",
            "elevation_ft": "4724_10124_11464",
            "direct_period": "1855_summer",
            "usable_component": "route_vertical_gradient_only",
            "anchor_class": "C_route_profile",
            "exclude": "regional_annual_mean",
        },
        {
            "station": "Skardo",
            "region": "west",
            "printed_pages": "532",
            "iiif_canvases": "0556",
            "latitude": "not_frozen",
            "longitude": "not_frozen",
            "elevation_ft": "7255",
            "direct_period": "1856",
            "usable_component": "local_direction_context_only",
            "anchor_class": "C_moving_local_network",
            "exclude": "single_fixed_station_interpretation",
        },
    ]
    with (OUT / "station_table_map.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": OUT.name,
        "decision": "extract_Darjeeling_first_then_only_explicit_direct_seasonal_segments",
        "candidate_tables": len(rows),
        "A_class_candidates": 3,
        "primary_anchor": "Darjeeling_Observatory_Hill",
        "direct_values_transcribed": 0,
        "inferred_or_filled_values_authorized": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "sealed_period_values_used": False,
        "prohibited_source_accessed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = """# Schlagintweit 1854-1858 direct-observation preflight

## Decision

`extract_Darjeeling_first_then_only_explicit_direct_seasonal_segments`

The High Asia tables are organized by region, station and A/B subtable rather than formal table numbers. Citations must therefore use volume, section, station/subtable, printed page and IIIF canvas.

Darjeeling Observatory Hill is the primary fixed-station candidate. It has four fixed observation times from 1855 and self-registering coverage from April 1855 to May 1856. It must be kept separate from Jillapahar, the Hodgson bungalow and multiyear composite values.

Kyelang May-October 1855 and Leh subtable A July-September 1856 are admissible seasonal anchors. Their completed annual tables are not admissible because missing months were supplied from other places or inferred from another station's curve. Massuri is conditional on separating observer and exact site. Kotghar is suitable only for a time-correction sensitivity test. Tonglo, the Jhosimath-Badrinath-Niti route and Skardo are short or moving-site context, not annual level anchors.

## Extraction order

1. Darjeeling pp.480-485 / canvases 0504-0509.
2. Kyelang p.513 / canvas 0537 and Leh pp.530-531 / canvases 0554-0555, direct segments only.
3. Massuri pp.499-500 / canvases 0523-0524 if station and observer separation is explicit.
4. Retain the remaining tables for exposure, diurnal-cycle and elevation-gradient sensitivity only.

Every extracted value must carry instrument type, exposure, observation time, site identity, direct-versus-derived status and printed-page provenance. No reconstructed monthly fill or completed annual mean may enter the direct-station gate.

## Primary source

- Heidelberg volume: https://digi.ub.uni-heidelberg.de/diglit/schlagintweit1866bd4
- IIIF manifest: https://digi.ub.uni-heidelberg.de/diglit/iiif/schlagintweit1866bd4/manifest.json

## Safety

- This preflight transcribed no temperature value.
- No original or production file was modified.
- No EnKF or OGGM run occurred.
- No sealed-period value was accessed or used.
- The permanently prohibited source was not accessed, used or cited.
"""
    (OUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")

    facts_after = sha256(FACTS)
    manifest = {
        "facts_sha256_before": facts_before,
        "facts_sha256_after": facts_after,
        "facts_unchanged": facts_before == facts_after,
        "primary_source": "https://digi.ub.uni-heidelberg.de/diglit/schlagintweit1866bd4",
        "iiif_manifest": "https://digi.ub.uni-heidelberg.de/diglit/iiif/schlagintweit1866bd4/manifest.json",
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if facts_before != facts_after:
        raise RuntimeError("Project facts changed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
