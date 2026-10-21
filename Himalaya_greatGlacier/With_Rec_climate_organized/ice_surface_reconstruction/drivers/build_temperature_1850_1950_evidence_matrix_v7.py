from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
BASE = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
V6 = BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v6"
BOREHOLE = BASE / "temperature_noaa_hma_borehole_19c_recovery_v1"
LAKES = BASE / "temperature_noaa_hma_lake_temperature_recovery_v1"
CUOQIA = BASE / "temperature_cuoqia_lake_numeric_recovery_v1"
OUTPUT = BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v7"
DRIVER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def set_row(matrix: pd.DataFrame, evidence_id: str, values: dict[str, Any]) -> None:
    index = matrix.index[matrix.evidence_id == evidence_id]
    if len(index) != 1:
        raise RuntimeError(f"expected-one-evidence-row:{evidence_id}:{len(index)}")
    for key, value in values.items():
        matrix.loc[index[0], key] = value


def append_row(matrix: pd.DataFrame, values: dict[str, Any]) -> pd.DataFrame:
    unknown = set(values) - set(matrix.columns)
    if unknown:
        raise RuntimeError(f"unknown-evidence-columns:{sorted(unknown)}")
    row = {column: values.get(column, "") for column in matrix.columns}
    return pd.concat([matrix, pd.DataFrame([row])], ignore_index=True)


def plot_evidence_map(
    borehole_sites: pd.DataFrame,
    borehole_clusters: pd.DataFrame,
    lake_points: pd.DataFrame,
    lake_pairs: pd.DataFrame,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    color = {True: "#2b7a4b", False: "#bb3e3e"}
    for cluster_name, group in borehole_sites.groupby("cluster"):
        cluster = borehole_clusters[borehole_clusters.cluster == cluster_name].iloc[0]
        ax.scatter(
            group.longitude.mean(),
            group.latitude.mean(),
            s=90 + 18 * len(group),
            marker="o",
            color=color[bool(cluster.direction_agreement)],
            edgecolor="white",
            linewidth=1.2,
            zorder=4,
        )
        ax.annotate(
            cluster_name.replace("_cluster", "").replace("_", " "),
            (group.longitude.mean(), group.latitude.mean()),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
        )
    locations = lake_points.groupby("site_id", as_index=False).agg(latitude=("latitude", "first"), longitude=("longitude", "first"))
    for row in lake_pairs.itertuples(index=False):
        point = locations[locations.site_id == row.site_id].iloc[0]
        ax.scatter(
            point.longitude,
            point.latitude,
            s=110,
            marker="s",
            color=color[bool(row.direction_agreement)],
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            row.site_id.replace("_", " "),
            (point.longitude, point.latitude),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=8,
        )
    ax.scatter(99.7722, 27.4052, s=120, marker="D", color="#7b6aa8", edgecolor="white", linewidth=1.2, zorder=6)
    ax.annotate("Cuoqia (mixed series)", (99.7722, 27.4052), xytext=(5, 7), textcoords="offset points", fontsize=8)
    ax.scatter([], [], s=95, marker="o", color="#777777", label="Borehole regional family")
    ax.scatter([], [], s=95, marker="s", color="#777777", label="Lake numeric pair")
    ax.scatter([], [], s=95, marker="D", color="#7b6aa8", label="Figure-derived mixed result")
    ax.scatter([], [], s=95, marker="o", color=color[True], label="Direction agrees")
    ax.scatter([], [], s=95, marker="o", color=color[False], label="Direction conflicts")
    ax.set_xlim(88.0, 104.0)
    ax.set_ylim(23.5, 41.0)
    ax.set_xlabel("Longitude (degE)")
    ax.set_ylabel("Latitude (degN)")
    ax.set_title("Independent low-frequency temperature evidence: spatially heterogeneous verdict")
    ax.grid(color="#dfdfdf", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(V6 / "evidence_matrix.csv")
    v6_summary = load_json(V6 / "summary.json")
    borehole_summary = load_json(BOREHOLE / "summary.json")
    lake_summary = load_json(LAKES / "summary.json")
    cuoqia_summary = load_json(CUOQIA / "summary.json")
    borehole_sites = pd.read_csv(BOREHOLE / "borehole_project_1800_1900_comparison.csv")
    borehole_clusters = pd.read_csv(BOREHOLE / "cluster_19th_century_comparison.csv")
    lake_points = pd.read_csv(LAKES / "lake_temperature_safe_points.csv")
    lake_pairs = pd.read_csv(LAKES / "latest_safe_prelock_pair_comparison.csv")

    set_row(
        matrix,
        "CN_PX202_BOREHOLE_GST",
        {
            "admitted_role": "part_of_13_site_NOAA_borehole_panel_not_an_extra_vote",
            "finding": (
                "Official nineteenth-century GST coefficient is +0.023 K/100a, while the matched project six-member "
                "1800-1900 median change is -0.919 C. The directions conflict, but the single east-edge site remains low resolution."
            ),
            "next_action": "interpret_inside_the_four_cluster_borehole_panel_only",
        },
    )
    set_row(
        matrix,
        "CUOQIA_LAKE_BRGDGT_300Y",
        {
            "machine_readable": "official_figure_recovered_to_approximate_numeric_CSV",
            "numeric_now": True,
            "admitted_role": "independent_southeast_warm_season_shape_gate_figure_derived",
            "finding": (
                "Official Figure 2 yielded 43 safe approximate points. CQ1 trends are -0.199 C/century over 1700-1920 "
                "and +0.240 over 1850-1920; CQ2 is -1.413 over 1870-1920. The series disagree in sign and cannot set one offset."
            ),
            "next_action": "retain_as_mixed_shape_envelope_pending_author_numeric_table",
        },
    )
    set_row(
        matrix,
        "WUDAOLIANG_BOREHOLE_GST_2021",
        {
            "machine_readable": "no_public_numeric_GST_sequence_found_after_repository_and_author_route_search",
            "numeric_now": False,
            "finding": (
                "Paper and public routes provide figures but no auditable numeric GST history; the published approximately "
                "83-year interpretable history conflicts with a separate 308-year wording and is not safe for an 1850-1920 metric."
            ),
            "next_action": "author_request_only;do_not_block_current_decision",
        },
    )
    set_row(
        matrix,
        "NGRAMRING_TIANCAI_AWENG_ALREADY_ASSESSED",
        {
            "admitted_role": "identity_context_only_Tiancai_explicit_numeric_row_below_is_same_record",
            "finding": (
                "Tiancai official NOAA points are now explicitly parsed below; this older grouped row is retained only for identity "
                "and must not be counted as another vote. Ngamring and Aweng remain previously adjudicated sparse records."
            ),
        },
    )

    common = {"sealed_period_values_used": False, "production_modified": False}
    matrix = append_row(
        matrix,
        {
            "evidence_id": "NOAA_HMA_BOREHOLE_13_SITE_19C_PANEL",
            "region": "central_eastern_southeastern_and_north_margin_HMA",
            "longitude": "90.65-102.22",
            "coverage": "16th-19th_century_GST_coefficients;matched_test_1800-1900",
            "target": "century_scale_ground_surface_temperature_change",
            "measurement_family": "official_borehole_physical_inversion",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "independent_non_tree_physical_family;clusters_not_holes_are_votes",
            "spatial_relevance": "four_regional_families;11_of_13_holes_match_valid_project_cells",
            "season_match": "GST_low_frequency_not_annual_air_Tmean",
            "admitted_role": "direction_and_rough_century_amplitude_spatial_gate",
            "finding": (
                "Direction agrees in Qaidam and Yunnan but conflicts at Naqu and PX202. Naqu borehole +1.413 vs project -0.718 C; "
                "PX202 +0.023 vs -0.919; Qaidam +0.396 vs +0.255; Yunnan -0.240 vs -0.788."
            ),
            "next_action": "reject_uniform_offset;use_as_regional_low_frequency_redesign_constraints",
            "source_url": "https://www.ncei.noaa.gov/products/paleoclimatology/borehole",
            **common,
        },
    )
    for site_id, source_id, finding, independence in [
        (
            "tiancai_lake",
            "NOAA_TIANCAI_LAKE_CHIRONOMID",
            "1789-1918.6 July proxy cools -0.914 C; project median cools -0.263 C. Direction agrees but project cooling is weaker.",
            "same_Tiancai_physical_record_as_prior_sparse_audit;explicit_numeric_recovery_not_new_vote",
        ),
        (
            "heihai_lake",
            "NOAA_HEIHAI_LAKE_CHIRONOMID",
            "1852-1914 July proxy is nearly stable (+0.014 C); project warms +0.419 C. Nominal direction agrees but amplitude conflicts.",
            "independent_lake_sediment;shared_chironomid_method_family_with_Tiancai",
        ),
        (
            "qinghai_lake",
            "NOAA_QINGHAI_LAKE_ALKENONE",
            "1657-1804 summer lake temperature cools -0.290 C; project warms +1.089 C. Direction conflicts before the target bridge.",
            "independent_lake_sediment_and_alkenone_measurement_family",
        ),
    ]:
        pair = lake_pairs[lake_pairs.site_id == site_id].iloc[0]
        source_file = Path(lake_points[lake_points.site_id == site_id].iloc[0].source_file)
        matrix = append_row(
            matrix,
            {
                "evidence_id": source_id,
                "region": "southeast_HMA" if site_id != "qinghai_lake" else "northeast_Tibetan_Plateau_margin",
                "longitude": float(lake_points[lake_points.site_id == site_id].iloc[0].longitude),
                "coverage": f"latest_safe_prelock_pair_{pair.start_year_ce:.1f}-{pair.end_year_ce:.1f}",
                "target": "July_temperature" if site_id != "qinghai_lake" else "summer_lake_surface_temperature",
                "measurement_family": "lake_sediment_chironomid" if site_id != "qinghai_lake" else "lake_sediment_alkenone_UK37",
                "machine_readable": True,
                "numeric_now": True,
                "source_independence": independence,
                "spatial_relevance": "site_specific_eastern_margin",
                "season_match": "summer_not_annual_Tmean",
                "admitted_role": "seasonal_low_frequency_direction_and_amplitude_gate",
                "finding": finding,
                "next_action": "use_only_in_season_aware_regional_redesign;do_not_translate_to_uniform_annual_offset",
                "source_url": str(source_file),
                **common,
            },
        )

    if matrix.evidence_id.duplicated().any():
        raise RuntimeError("duplicate-evidence-id")
    matrix.to_csv(OUTPUT / "evidence_matrix.csv", index=False)

    plot_evidence_map(
        borehole_sites,
        borehole_clusters,
        lake_points,
        lake_pairs,
        OUTPUT / "independent_low_frequency_evidence_map.png",
    )

    summary = {
        "experiment": "temperature_1850_1950_low_frequency_evidence_matrix_v7",
        "supersedes": v6_summary["experiment"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "localized_spatial_and_seasonal_low_frequency_errors_supported_no_uniform_HMA_correction",
        "evidence_rows": int(len(matrix)),
        "new_numeric_packages": [borehole_summary["experiment"], lake_summary["experiment"], cuoqia_summary["experiment"]],
        "bias_type_verdict": {
            "direction": "regionally_conflicting;central_east_borehole_and_northeast_lake_conflicts_exist",
            "amplitude": "inconsistent_by_site;both_overresponse_and_underresponse_occur",
            "spatial_distribution": "material_and_decisive;uniform_HMA_offset_rejected",
            "season_mapping": "material_uncertainty;July_and_warm-season_lake_targets_cannot_define_annual_offset",
            "early_reference_coverage": "improved_but_still_sparse_and_low_resolution_outside_western_station_network",
        },
        "original_reconstruction_verdict": (
            "Original production contains credible localized low-frequency bridge errors, but an HMA-wide 1500-1879 warm bias "
            "is not independently established. Retain production provisionally because no validated replacement exists, not because it is proven correct."
        ),
        "candidate_v2_authorized": False,
        "candidate_v2_reason": (
            "Previously failed direct-station temporal-shape gates; new physical evidence is spatially heterogeneous and cannot rescue a uniform or fixed-mixture correction."
        ),
        "next_scientific_route": (
            "Design, but do not yet run, a region- and season-aware low-frequency PSM/background reconstruction preflight constrained by the independent physical envelopes."
        ),
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "sealed_period_values_used": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")

    inputs = [
        V6 / "evidence_matrix.csv",
        V6 / "summary.json",
        BOREHOLE / "summary.json",
        BOREHOLE / "borehole_project_1800_1900_comparison.csv",
        BOREHOLE / "cluster_19th_century_comparison.csv",
        LAKES / "summary.json",
        LAKES / "lake_temperature_safe_points.csv",
        LAKES / "latest_safe_prelock_pair_comparison.csv",
        CUOQIA / "summary.json",
    ]
    manifest = {
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in inputs],
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "production_modified": False,
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="ascii")

    report = """# HMA 1850-1950 low-frequency temperature evidence matrix v7

## Decisive result

The new independent physical evidence does not support one HMA-wide cooling constant, one fixed four-background mixture, or the rejected Candidate v2. It does support real low-frequency bridge errors that vary by region and target season.

- NOAA boreholes: direction agrees in Qaidam and Yunnan, but conflicts at central Naqu and eastern PX202.
- NOAA lake records: Tiancai agrees in direction but project cooling is too weak; Heihai is almost stable while the project warms appreciably; Qinghai conflicts in direction over its latest safe pre-1805 pair.
- Cuoqia: the two published warm-season series give mixed pre-1921 trends, so even one southeast site does not define a single correction.
- Wudaoliang: no auditable public numeric GST sequence was found; it remains an author-request route and does not block this decision.

## Bias classification

1. **Direction:** wrong in some central/eastern locations, correct in others.
2. **Amplitude:** both overresponse and underresponse occur.
3. **Spatial distribution:** decisive; a uniform HMA correction is rejected.
4. **Season mapping:** material; summer and warm-season proxies cannot directly set annual-temperature offsets.
5. **Early reference coverage:** improved by physical records, but still sparse and low resolution outside the western station network.

## Reconstruction decision

The original production field is not declared correct. It is retained provisionally because no replacement passes the frozen gates. Candidate v2 remains rejected because it failed direct-station temporal-shape tests and the new evidence cannot rescue a uniform or fixed-mixture correction.

The scientifically defensible next route is a **region- and season-aware low-frequency reconstruction preflight**, constrained by these independent physical envelopes. This report does not authorize an EnKF run or production modification.

## Safety

- No original or production file was modified.
- No EnKF or OGGM run occurred.
- No 1921-1939 value was used.
- 20CRv3 was not accessed (`prohibited_source_accessed=false`).
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="ascii")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
