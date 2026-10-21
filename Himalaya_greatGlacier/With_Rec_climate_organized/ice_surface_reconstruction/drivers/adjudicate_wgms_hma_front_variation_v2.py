from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def as_int(value: str) -> int | None:
    value = value.strip()
    return int(value) if value else None


def as_float(value: str) -> float | None:
    value = value.strip()
    return float(value) if value else None


def main(input_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = read_rows(input_csv)

    adjudicated = []
    for row in rows:
        survey_year = as_int(row["survey_year"])
        reference_year = as_int(row["reference_year"])
        value = as_float(row["front_variation_m"])
        chronology_valid = (
            survey_year is not None
            and reference_year is not None
            and reference_year < survey_year
            and value is not None
        )
        reason = "accepted" if chronology_valid else "reject_nonchronological_or_incomplete_interval"
        adjudicated.append(
            {
                **row,
                "chronology_valid": chronology_valid,
                "adjudication": reason,
            }
        )

    fields = list(adjudicated[0]) if adjudicated else []
    with (output_dir / "adjudicated_front_variation_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(adjudicated)

    grouped: dict[str, list[dict[str, str | bool]]] = defaultdict(list)
    for row in adjudicated:
        if row["chronology_valid"]:
            grouped[str(row["wgms_id"])].append(row)

    summaries = []
    for glacier_id, group in grouped.items():
        group.sort(key=lambda row: int(str(row["survey_year"])))
        total = sum(float(str(row["front_variation_m"])) for row in group)
        summaries.append(
            {
                "wgms_id": glacier_id,
                "name": group[0]["name"],
                "region": group[0]["region"],
                "latitude": group[0]["latitude"],
                "longitude": group[0]["longitude"],
                "first_reference_year": min(int(str(row["reference_year"])) for row in group),
                "last_survey_year": max(int(str(row["survey_year"])) for row in group),
                "valid_interval_count": len(group),
                "net_front_variation_m": total,
                "direction": "advance" if total > 0 else "retreat" if total < 0 else "stationary",
            }
        )

    summary_fields = [
        "wgms_id",
        "name",
        "region",
        "latitude",
        "longitude",
        "first_reference_year",
        "last_survey_year",
        "valid_interval_count",
        "net_front_variation_m",
        "direction",
    ]
    with (output_dir / "valid_glacier_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(sorted(summaries, key=lambda row: row["name"]))

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for glacier_id, group in sorted(grouped.items(), key=lambda item: item[1][0]["name"]):
        group.sort(key=lambda row: int(str(row["survey_year"])))
        years = [min(int(str(row["reference_year"])) for row in group)]
        cumulative = [0.0]
        for row in group:
            years.append(int(str(row["survey_year"])))
            cumulative.append(cumulative[-1] + float(str(row["front_variation_m"])))
        ax.plot(years, cumulative, marker="o", linewidth=2, label=str(group[0]["name"]))
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative front variation (m)")
    ax.set_title("WGMS HMA records available before 1921")
    ax.grid(True, alpha=0.25)
    if grouped:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "wgms_pre1921_front_variation.png", dpi=180)
    plt.close(fig)

    result = {
        "experiment": "wgms_hma_front_variation_adjudication_v2",
        "input_row_count": len(rows),
        "valid_interval_count": sum(bool(row["chronology_valid"]) for row in adjudicated),
        "rejected_interval_count": sum(not bool(row["chronology_valid"]) for row in adjudicated),
        "valid_glacier_count": len(summaries),
        "regions_represented": sorted({str(row["region"]) for row in summaries}),
        "advancing_glacier_count": sum(row["direction"] == "advance" for row in summaries),
        "retreating_glacier_count": sum(row["direction"] == "retreat" for row in summaries),
        "decision": "west_karakoram_direction_only_insufficient_for_hma_uniform_temperature_correction",
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = f"""# WGMS HMA front-variation adjudication v2

- Valid intervals: `{result['valid_interval_count']}`; rejected metadata anomaly: `{result['rejected_interval_count']}`.
- Valid glaciers: `{result['valid_glacier_count']}`; represented regions: `{', '.join(result['regions_represented'])}`.
- Advancing/retreating glaciers: `{result['advancing_glacier_count']}` / `{result['retreating_glacier_count']}`.

The surviving observations are restricted to western Karakoram glaciers. They support a locally non-warming or positive-balance response before 1921, but they cannot diagnose an HMA-wide temperature bias or define a temperature offset.

Decision: `{result['decision']}`.
"""
    (output_dir / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: adjudicate_wgms_hma_front_variation_v2.py INPUT_CSV OUTPUT_DIR")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
