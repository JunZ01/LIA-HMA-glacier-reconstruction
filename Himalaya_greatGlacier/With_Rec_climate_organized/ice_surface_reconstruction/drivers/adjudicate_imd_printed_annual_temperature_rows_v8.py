from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
FRAGMENTS = LOOP / "imd_ia_annual_temperature_row_query_v1/target_page_fragments.csv"
GHCN = LOOP / "imd_ghcn_overlap_pages_v2/ghcn_annual_overlap.csv"
OUT = LOOP / "imd_printed_annual_temperature_adjudication_v8"
DRIVER = Path(__file__).resolve()

# Manual transcription of printed Year rows. Only rows with an unambiguous
# station side and internally redundant table fields are admitted.
ROWS = [
    # Simla/Murree: adopted mean, 10h, 16h, max, daily range, min.
    (1878, "Murree", "MURREE", "india.history.resource.117788", 183, "simla_murree", 55.7, 60.3, 59.8, 66.0, 14.5, 51.6),
    (1881, "Murree", "MURREE", "india.history.resource.108635", 257, "simla_murree", 56.8, 61.4, 61.7, 68.2, 16.3, 51.9),
    (1885, "Murree", "MURREE", "india.history.resource.108638", 278, "simla_murree", 53.4, 57.8, 58.0, 62.6, 13.9, 48.7),
    (1886, "Murree", "MURREE", "india.history.resource.109237", 299, "simla_murree", 54.8, 59.8, 59.7, 64.3, 14.4, 50.0),
    (1886, "Simla", "SHIMLA", "india.history.resource.109237", 301, "simla_murree", 54.8, 56.7, 59.8, 64.3, 14.7, 49.6),
    # Darjeeling: 10h, 16h, adopted mean, max, daily range, min.
    (1890, "Darjeeling", "DARJEELING", "india.history.resource.109241", 300, "darjeeling", 52.5, 54.3, 55.1, 58.4, 11.0, 47.4),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    ghcn = {(r["station_name"], r["series"], int(r["year"])): float(r["annual_mean_c"])
            for r in read_csv(GHCN)}
    fragments = read_csv(FRAGMENTS)
    out = []
    evidence = []
    for year, station, ghcn_name, item, leaf, layout, a, b, c, d, e, f in ROWS:
        if layout == "simla_murree":
            adopted, hour10, hour16, maximum, daily_range, minimum = a, b, c, d, e, f
            formula_error = abs(adopted - (hour16 + minimum) / 2.0)
        else:
            hour10, hour16, adopted, maximum, daily_range, minimum = a, b, c, d, e, f
            formula_error = math.nan
        range_error = abs((maximum - minimum) - daily_range)
        accepted = range_error <= 0.11 and (math.isnan(formula_error) or formula_error <= 0.11)
        annual_c = (adopted - 32.0) / 1.8
        row = {
            "report_year": year, "imd_station": station, "ghcn_station_name": ghcn_name,
            "item_id": item, "leaf": leaf, "layout": layout,
            "printed_adopted_annual_f": adopted, "printed_adopted_annual_c": annual_c,
            "printed_10h_f": hour10, "printed_16h_f": hour16,
            "printed_maximum_f": maximum, "printed_daily_range_f": daily_range,
            "printed_minimum_f": minimum, "range_closure_error_f": range_error,
            "mean_formula_error_f": formula_error,
            "status": "accepted_printed_year_row" if accepted else "rejected_internal_closure",
        }
        for series in ("qcu", "qcf"):
            value = ghcn.get((ghcn_name, series, year), math.nan)
            row[f"ghcn_{series}_annual_c"] = value
            row[f"imd_minus_{series}_c"] = annual_c - value if accepted and not math.isnan(value) else math.nan
        out.append(row)
        same_page = [r for r in fragments if r["item_id"] == item and int(r["target_leaf"]) == leaf
                     and r["query"] == "year" and int(r["leaf_distance"]) == 0]
        for source in same_page:
            evidence.append({"report_year": year, "imd_station": station, **source})

    fields = list(dict.fromkeys(key for row in out for key in row))
    with (OUT / "printed_annual_temperature_rows.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(out)
    efields = list(dict.fromkeys(key for row in evidence for key in row))
    with (OUT / "source_year_fragments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=efields); writer.writeheader(); writer.writerows(evidence)

    comparisons = [r for r in out if r["status"].startswith("accepted") and not math.isnan(r["ghcn_qcu_annual_c"])]
    summary = {
        "experiment": OUT.name, "accepted_printed_year_rows": sum(r["status"].startswith("accepted") for r in out),
        "accepted_with_ghcn": len(comparisons),
        "comparison_rows": [{
            "year": r["report_year"], "station": r["imd_station"],
            "imd_minus_qcu_c": r["imd_minus_qcu_c"], "imd_minus_qcf_c": r["imd_minus_qcf_c"],
        } for r in comparisons],
        "interpretation": "publication-versus-processing audit of the same physical observations; not an independent climate series",
        "locked_period_accessed": False, "source_files_modified": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# IMD Printed Annual Temperature Adjudication",
        "",
        f"Accepted {summary['accepted_printed_year_rows']} unambiguous printed Year rows; {len(comparisons)} overlap GHCN-M.",
        "",
        "- 1886 Simla: original IMD is 1.217 C below qcu and 1.697 C below qcf.",
        "- 1890 Darjeeling: original IMD is 1.210 C above qcu and 0.130 C below qcf.",
        "- The direction is station/processing specific; these rows do not support a uniform HMA correction.",
        "- IMD and GHCN share the physical observations. This tests digitization/homogenization, not independent climate truth.",
        "- Incomplete-month annual means from earlier experiments are superseded and excluded.",
        "- 1921-1939 accessed: no.",
    ]
    (OUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps({
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "fragments": {"path": str(FRAGMENTS), "sha256": sha256(FRAGMENTS)},
        "ghcn": {"path": str(GHCN), "sha256": sha256(GHCN)},
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
