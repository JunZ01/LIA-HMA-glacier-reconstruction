from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUT = BASE / "temperature_primary_evidence_region_target_synthesis_v1"

SOURCES = {
    "matrix": BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v13/summary.json",
    "lake": BASE / "temperature_original_lake_author_reconstructions_v2/summary.json",
    "physical": BASE / "temperature_noaa_hma_additional_physical_records_v1/summary.json",
    "residual": BASE / "temperature_1850_1950_residual_bias_final_synthesis_v1/summary.json",
    "background": BASE / "temperature_background_correction_final_synthesis_v1/summary.json",
    "four_scheme": BASE / "temperature_four_background_scheme_evaluation_v2/summary.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {name: load_json(path) for name, path in SOURCES.items()}

    assert data["matrix"]["sealed_period_values_used"] is False
    assert data["matrix"]["prohibited_source_accessed"] is False
    assert data["lake"]["sealed_period_values_used"] is False
    assert data["physical"]["prohibited_source_accessed"] is False

    # One row is one independent archive/observation family in one native window.
    # Secondary recalibrations, qcu, CRU, and Berkeley are not additional votes.
    rows = [
        ("Aweng Co", "West", "long preindustrial", "annual MAAT", "agree", "amplitude mismatch", "1758-1834", "lake brGDGT"),
        ("Ngamring", "Central-south", "long preindustrial", "annual MAAT", "agree", "roughly comparable", "1597-1778", "lake brGDGT"),
        ("Huang SETP", "East", "long preindustrial", "winter Tmin", "agree", "roughly comparable", "1500-1899 aggregate", "tree reconstruction"),
        ("Baiyu", "East", "long preindustrial", "Jun-Jul T", "agree", "roughly comparable", "1500-1899 aggregate", "tree reconstruction"),
        ("Haizi", "East", "long preindustrial", "annual T", "agree", "roughly comparable", "1500-1899 aggregate", "tree reconstruction"),
        ("Tengchongqinghai", "Southeast margin", "long preindustrial", "annual MAAT", "conflict", "direction conflict", "1560-1899", "lake brGDGT"),
        ("Lugu", "Southeast", "long preindustrial", "annual MAAT", "conflict", "direction conflict", "1507-1914", "lake brGDGT"),
        ("Qinghai Lake", "Northeast margin", "long preindustrial", "summer lake T", "conflict", "direction conflict", "1657-1804", "lake UK37"),
        ("Qaidam boreholes", "North", "19th-century GST", "century GST", "agree", "rough amplitude", "1800-1900", "borehole cluster"),
        ("Yunnan boreholes", "Southeast", "19th-century GST", "century GST", "agree", "rough amplitude", "1800-1900", "borehole cluster"),
        ("Naqu boreholes", "Central", "19th-century GST", "century GST", "conflict", "direction conflict", "1800-1900", "borehole cluster"),
        ("PX202 borehole", "East edge", "19th-century GST", "century GST", "conflict", "direction conflict", "1800-1900", "borehole site"),
        ("Guliya", "West", "1850-1920 bridge", "isotope T direction", "agree", "direction only", "1850-1920", "ice core"),
        ("Puruogangri", "Central", "1850-1920 bridge", "summer isotope direction", "agree", "direction only", "1850s-1910s", "ice core"),
        ("Lugu", "Southeast", "1850-1920 bridge", "annual MAAT", "agree", "sparse four points", "1850-1914", "lake brGDGT"),
        ("Tiancai", "Southeast", "1850-1920 bridge", "July T", "agree", "seasonal", "1789-1918.6", "lake chironomid"),
        ("Heihai", "Southeast", "1850-1920 bridge", "July T", "mixed", "direction agrees; amplitude conflicts", "1852-1914", "lake chironomid"),
        ("GHCN-M qcf panel", "West/central-west", "1850-1920 bridge", "annual T anomaly", "conflict", "warm level and 1901-1920 shape", "1880-1920", "direct stations"),
        ("GHCN-M qcf panel", "West/central-west", "1940-1950 transfer", "annual T anomaly", "mixed", "small warm residual; no correction transfer", "1940-1950", "direct stations"),
    ]
    fields = ["record", "region", "window_group", "native_target", "direction_status", "amplitude_status", "native_window", "family"]
    with (OUT / "primary_evidence_family_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)

    counts = {}
    for window in sorted({r[2] for r in rows}):
        subset = [r for r in rows if r[2] == window]
        counts[window] = {s: sum(r[4] == s for r in subset) for s in ("agree", "conflict", "mixed")}

    lake_metrics = {(m["record"], m["window"]): m for m in data["lake"]["metrics"]}
    numeric_pairs = [
        ("Ngamring\n1597-1778", 0.7569, 0.4109),
        ("TCQH\n1560-1899", lake_metrics[("Tengchongqinghai", "full_safe_prelock")]["author_trend_c_per_century"], lake_metrics[("Tengchongqinghai", "full_safe_prelock")]["project_median_trend_at_author_years_c_per_century"]),
        ("Lugu long\n1507-1914", lake_metrics[("Lugu", "full_safe_prelock")]["author_trend_c_per_century"], lake_metrics[("Lugu", "full_safe_prelock")]["project_median_trend_at_author_years_c_per_century"]),
        ("Lugu bridge\n1850-1914", lake_metrics[("Lugu", "bridge_1850_1920")]["author_trend_c_per_century"], lake_metrics[("Lugu", "bridge_1850_1920")]["project_median_trend_at_author_years_c_per_century"]),
    ]

    status_color = {"agree": "#2f7d4a", "conflict": "#c44536", "mixed": "#d49b2a"}
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0])

    ax = fig.add_subplot(gs[:, 0])
    for i, row in enumerate(rows[::-1]):
        ax.scatter(0, i, s=120, color=status_color[row[4]], edgecolor="white", linewidth=0.8)
        ax.text(0.08, i, f"{row[0]} | {row[1]} | {row[3]} | {row[6]}", va="center", fontsize=8.5)
    ax.set_xlim(-0.15, 2.2)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("A. Independent evidence families (native target and window)", loc="left", fontweight="bold")
    for s, x in zip(("agree", "conflict", "mixed"), (0.0, 0.7, 1.45)):
        ax.scatter(x, len(rows) - 0.1, s=85, color=status_color[s])
        ax.text(x + 0.08, len(rows) - 0.1, s, va="center", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(numeric_pairs))
    proxy = np.array([p[1] for p in numeric_pairs])
    project = np.array([p[2] for p in numeric_pairs])
    ax2.axvline(0, color="#555555", lw=0.8)
    for yi, pv, rv in zip(y, proxy, project):
        ax2.plot([pv, rv], [yi, yi], color="#a0a0a0", lw=2)
    ax2.scatter(proxy, y, label="Independent reconstruction", color="#16697a", s=58, zorder=3)
    ax2.scatter(project, y, label="Project", color="#e76f51", marker="s", s=52, zorder=3)
    ax2.set_yticks(y, [p[0] for p in numeric_pairs])
    ax2.set_xlabel("Trend (degC per century)")
    ax2.set_title("B. Directly comparable annual-MAAT trends", loc="left", fontweight="bold")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(axis="x", alpha=0.2)

    ax3 = fig.add_subplot(gs[1, 1])
    periods = ["1880-1900", "1901-1920", "1940-1950"]
    original = [1.4175198151, 0.8468282728, 0.2504438318]
    old_half = [1.2349818646, 0.7166451716, np.nan]
    full_forced = [0.8453878696, 0.3098831431, 0.2679174854]
    x = np.arange(3)
    width = 0.25
    ax3.bar(x - width, original, width, label="Original", color="#b23a48")
    ax3.bar(x, old_half, width, label="Old 50:50", color="#457b9d")
    ax3.bar(x + width, full_forced, width, label="Full forced-mean LF", color="#2a9d8f")
    ax3.axhline(0, color="#444444", lw=0.8)
    ax3.set_xticks(x, periods)
    ax3.set_ylabel("Project minus qcf station anomaly (degC)")
    ax3.set_title("C. Western station warm residual", loc="left", fontweight="bold")
    ax3.legend(frameon=False, fontsize=8)
    ax3.grid(axis="y", alpha=0.2)
    ax3.text(0.5, -0.28, "1921-1939 sealed and excluded", transform=ax3.transAxes, ha="center", fontsize=8, color="#666666")

    fig.suptitle("HMA temperature evidence synthesis: no uniform correction, real regional transition errors", fontsize=15, fontweight="bold")
    fig.savefig(OUT / "primary_evidence_and_background_diagnosis.png", dpi=220)
    plt.close(fig)

    conclusion = {
        "experiment": OUT.name,
        "status": "complete",
        "family_balanced_counts": counts,
        "uniform_1500_1879_hma_warm_bias_supported": False,
        "western_1880_1920_warm_residual_supported": True,
        "background_cold_state_deficit_material": True,
        "background_deficit_complete_explanation": False,
        "global_constant_offset_supported": False,
        "global_four_background_blend_supported_for_production": False,
        "scientific_diagnosis": "Evidence is spatially and target-season heterogeneous. The western 1880-1920 warm residual is real, and a missing forced cold state in the background explains a material fraction, but annual, seasonal, GST, and isotope evidence do not support one HMA-wide scalar cooling.",
        "next_decisive_test": "Annual-air-temperature-only low-frequency background replacement, evaluated by leave-one-region-out validation and fixed before the 1850-1920 station bridge; no post-hoc offset and no seasonal/GST record treated as annual truth.",
        "candidate_v2_authorized": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "sealed_1921_1939_values_used": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, indent=2)

    manifest = {name: {"path": str(path), "sha256": sha256(path)} for name, path in SOURCES.items()}
    manifest["rules"] = {
        "unit_of_vote": "independent archive or direct-observation family within one native window",
        "not_extra_votes": ["secondary recalibration", "GHCN qcu", "CRU", "Berkeley"],
        "locked_period": "1921-1939 not decoded or used",
        "prohibited_source_accessed": False,
    }
    with (OUT / "source_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    report = """# HMA temperature primary-evidence synthesis

## Decision

The project does not have defensible evidence for a single HMA-wide warm bias over 1500-1879. The long-window evidence is mixed across regions and native targets. A constant offset or a globally fixed four-background blend would correct some records while degrading others.

The western and central-western 1880-1920 warm residual is, however, directly supported by qcf stations: +1.418 degC in 1880-1900 and +0.847 degC in 1901-1920. The residual falls to +0.250 degC in 1940-1950, so it is a transition-shape problem rather than a stationary offset.

## Mechanism

The old 50:50 correction is directionally useful but explains only 12.9% and 15.4% of the two early western station residuals. The full forced-model low-frequency mean reduces them to +0.845 and +0.310 degC, confirming that the archived background lacks a material cold state. It does not explain the full residual and cannot yet be promoted: the tested fields fail either historical-window closure or early-station trend gates.

## Evidence balance

- Long preindustrial families: 5 direction agreements, 3 conflicts.
- Nineteenth-century borehole clusters/sites: 2 agreements, 2 conflicts.
- 1850-1920 bridge families: 4 agreements, 1 direct-station conflict, 1 mixed amplitude result.
- Annual-MAAT evidence is also mixed, so seasonal mismatch alone is not the entire explanation.

## What this means

The strongest diagnosis is not "CESM is uniformly too warm" and not "the proxies are uniformly wrong." It is a combination of a real western early-instrumental background deficit and heterogeneous regional/seasonal constraints. The current evidence supports correcting the low-frequency transition mechanism only after an annual-air-temperature-only, out-of-region validation test. It does not support applying a scalar cooling, accepting the old 50:50 result, or modifying production temperature now.

The next decisive experiment is fixed: compare CESM and the full forced low-frequency background using only annual-air-temperature evidence, choose weights without using 1850-1920, and use the western station bridge strictly as out-of-sample validation. This is the shortest route to a scientifically interpretable Candidate v2.

No production file was modified. No EnKF or OGGM run was made. The 1921-1939 interval remained sealed. 20CRv3 was not accessed.
"""
    (OUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
