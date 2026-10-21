from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


EXPERIMENT_ID = "climate_reconstruction_pages2k_fixed_network_metadata_family_v1"
MIN_CORRELATION = 0.999999999
MAX_AFFINE_RESIDUAL = 1e-10
COORDINATE_TOLERANCE = 1e-6

COMBINED = Path(r"E:\2Data\2Rec_ref\a_new_two_proxy\process_past_20000_proxy\combined_proxydata6.mat")
PAGES_RECONSTRUCTION = Path(r"E:\2Data\2Rec_ref\a_new_two_proxy\process_past_20000_proxy\pages2k_dataset_p.mat")
PAGES_ROOT = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\proxy-data\PAGES2k_v2\proxy-data\pages2k_proxy")
PAGES_ANNUAL = PAGES_ROOT / "proxy_data_1_13_1.mat"
PAGES_METADATA = PAGES_ROOT / "metadata_1_13_1.txt"
PAGES_TS = PAGES_ROOT / "Pages2kTSv1_13_1.mat"

METADATA_FIELDS = {
    "archive_type": "archiveType",
    "archive_species": "paleoData_archiveSpecies",
    "archive_genus": "paleoData_archiveGenus",
    "proxy": "paleoData_proxy",
    "measurement_material": "paleoData_measurementMaterial",
    "measurement_method": "paleoData_measurementMethod",
    "data_type": "paleoData_dataType",
    "variable_name": "paleoData_variableName",
    "climate_variable": "climateInterpretation_variable",
    "climate_variable_detail": "climateInterpretation_variableDetail",
    "seasonality": "climateInterpretation_seasonality",
    "interpretation_direction": "climateInterpretation_interpDirection",
    "site_name": "geo_siteName",
    "elevation": "geo_elevation",
}


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


def text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("latin-1").strip()
    array = np.asarray(value)
    if array.size == 0:
        return ""
    if array.dtype.kind in "US":
        return " ".join(str(item).strip() for item in array.reshape(-1) if str(item).strip()).strip()
    if array.dtype == object:
        return " ".join(text_value(item) for item in array.reshape(-1) if text_value(item)).strip()
    if array.size == 1:
        return str(array.reshape(-1)[0]).strip()
    return " ".join(str(item) for item in array.reshape(-1)).strip()


def clean_group(value: object) -> str:
    text = text_value(value)
    return text if text and text.lower() not in {"nan", "none"} else "<missing>"


def affine_closure(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float, float, int]:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 20:
        return np.nan, np.nan, np.nan, np.nan, int(valid.sum())
    x = right[valid]
    y = left[valid]
    matrix = np.column_stack([x, np.ones(len(x))])
    slope, intercept = np.linalg.lstsq(matrix, y, rcond=None)[0]
    residual = y - matrix @ np.array([slope, intercept])
    return (
        float(np.corrcoef(x, y)[0, 1]),
        float(slope),
        float(intercept),
        float(np.max(np.abs(residual))),
        int(valid.sum()),
    )


def align_series(
    left_years: np.ndarray,
    left_values: np.ndarray,
    right_years: np.ndarray,
    right_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    common = np.intersect1d(left_years.astype(int), right_years.astype(int))
    left_lookup = {int(year): value for year, value in zip(left_years, left_values)}
    right_lookup = {int(year): value for year, value in zip(right_years, right_values)}
    return (
        np.asarray([left_lookup[int(year)] for year in common], dtype=float),
        np.asarray([right_lookup[int(year)] for year in common], dtype=float),
    )


def passing_matches(
    target_years: np.ndarray,
    target_values: np.ndarray,
    candidate_years: np.ndarray,
    candidate_matrix: np.ndarray,
    candidates: np.ndarray,
) -> list[dict[str, object]]:
    passed = []
    for column in candidates:
        left, right = align_series(
            target_years,
            target_values,
            candidate_years,
            candidate_matrix[:, int(column)],
        )
        corr, slope, intercept, residual, count = affine_closure(left, right)
        if np.isfinite(corr) and corr >= MIN_CORRELATION and residual <= MAX_AFFINE_RESIDUAL:
            passed.append({
                "column_0based": int(column),
                "correlation": corr,
                "slope": slope,
                "intercept": intercept,
                "maximum_abs_residual": residual,
                "finite_overlap": count,
            })
    return passed


def metadata_table(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, sep="\t", encoding="latin-1", low_memory=False)
    if raw.shape != (6, 693):
        raise RuntimeError(f"unexpected-pages-metadata-shape:{raw.shape}")
    table = raw.set_index(0).iloc[:, :].T.reset_index(drop=True)
    table.columns.name = None
    table["geo_latitude"] = pd.to_numeric(table["geo_latitude"], errors="raise")
    table["geo_longitude"] = pd.to_numeric(table["geo_longitude"], errors="raise")
    table["official_column_1based"] = np.arange(1, len(table) + 1, dtype=int)
    return table


def collapse_sites(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("site_id", as_index=False).agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        event_change_z=("event_change_z", "mean"),
        record_count=("proxy_index", "size"),
    )


def group_audit(frame: pd.DataFrame, fields: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    deletion_rows: list[dict[str, object]] = []
    for scope, scoped in (
        ("all_fixed_records", frame),
        ("north_ge_35n", frame[frame["latitude"] >= 35.0]),
    ):
        full_sites = collapse_sites(scoped)
        full_mean = float(full_sites["event_change_z"].mean())
        for field in fields:
            pages_values = scoped.loc[scoped["source_family"] == "PAGES2k2017", field]
            for group in sorted(pages_values.map(clean_group).unique()):
                member = (scoped["source_family"] == "PAGES2k2017") & (scoped[field].map(clean_group) == group)
                omitted = scoped[member]
                omitted_sites = collapse_sites(omitted) if len(omitted) else omitted
                summary_rows.append({
                    "scope": scope,
                    "metadata_field": field,
                    "metadata_group": group,
                    "record_count": int(len(omitted)),
                    "unique_site_count": int(omitted["site_id"].nunique()),
                    "mean_record_event_change_z": float(omitted["event_change_z"].mean()),
                    "mean_site_event_change_z": float(omitted_sites["event_change_z"].mean()),
                })
                retained = scoped[~member]
                retained_sites = collapse_sites(retained)
                retained_mean = float(retained_sites["event_change_z"].mean())
                deletion_rows.append({
                    "scope": scope,
                    "metadata_field": field,
                    "omitted_group": group,
                    "omitted_record_count": int(member.sum()),
                    "omitted_unique_site_count": int(omitted["site_id"].nunique()),
                    "retained_record_count": int(len(retained)),
                    "retained_site_count": int(len(retained_sites)),
                    "full_site_mean_event_change_z": full_mean,
                    "retained_site_mean_event_change_z": retained_mean,
                    "change_from_full_site_mean_z": retained_mean - full_mean,
                    "retained_sign_negative": bool(retained_mean < 0),
                    "eligible_interpretation": bool(int(member.sum()) >= 2 and len(retained_sites) >= 10),
                })
    return pd.DataFrame(summary_rows), pd.DataFrame(deletion_rows)


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    robustness_root = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_positive_b2_fixed_network_common_mode_robustness_v1"
    common_root = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_proxy_common_mode_source_audit_v2"
    output = root / f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_ID}"
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")

    sources = {
        "fixed_records": robustness_root / "fixed_record_event_and_correlation_metadata.csv",
        "source_mapping": common_root / "proxy_source_mapping.csv",
        "combined_proxydata6": COMBINED,
        "pages_reconstruction": PAGES_RECONSTRUCTION,
        "pages_annual": PAGES_ANNUAL,
        "pages_metadata": PAGES_METADATA,
        "pages_ts": PAGES_TS,
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}

    fixed = pd.read_csv(sources["fixed_records"])
    mapping = pd.read_csv(sources["source_mapping"])
    fixed = fixed.merge(
        mapping[["proxy_index", "archive_column_1based"]],
        on="proxy_index",
        validate="one_to_one",
    )
    pages_fixed = fixed[fixed["source_family"] == "PAGES2k2017"].copy()
    if len(fixed) != 50 or len(pages_fixed) != 47:
        raise RuntimeError(f"fixed-network-count-contract-failed:{len(fixed)}:{len(pages_fixed)}")

    combined = loadmat(COMBINED, squeeze_me=True, struct_as_record=False)
    combined_years = np.asarray(combined["year"], dtype=int).reshape(-1)
    combined_matrix = np.asarray(combined["combined6_data"], dtype=float)
    combined_lat = np.asarray(combined["p_lat"], dtype=float).reshape(-1)
    combined_lon = np.asarray(combined["p_lon"], dtype=float).reshape(-1)

    pages = loadmat(PAGES_RECONSTRUCTION, squeeze_me=True, struct_as_record=False)
    pages_years = np.asarray(pages["year"], dtype=int).reshape(-1)
    pages_matrix = np.asarray(pages["pages2k_data"], dtype=float)
    pages_lat = np.asarray(pages["p_lat"], dtype=float).reshape(-1)
    pages_lon = np.asarray(pages["p_lon"], dtype=float).reshape(-1)
    pages_names = np.asarray(pages["pages2k_dataName"], dtype=object).reshape(-1)

    official = loadmat(PAGES_ANNUAL, squeeze_me=True, struct_as_record=False)
    official_years = np.asarray(official["year"], dtype=int).reshape(-1)
    official_matrix = np.asarray(official["proxy_ann"], dtype=float)
    official_metadata = metadata_table(PAGES_METADATA)
    official_lat = official_metadata["geo_latitude"].to_numpy(dtype=float)
    official_lon = official_metadata["geo_longitude"].to_numpy(dtype=float)

    identity_rows: list[dict[str, object]] = []
    for record in pages_fixed.itertuples(index=False):
        combined_column = int(record.archive_column_1based) - 1
        if not np.isclose(combined_lat[combined_column], record.latitude, atol=COORDINATE_TOLERANCE):
            raise RuntimeError(f"combined-latitude-mismatch:{record.proxy_index}")
        if not np.isclose(combined_lon[combined_column], record.longitude, atol=COORDINATE_TOLERANCE):
            raise RuntimeError(f"combined-longitude-mismatch:{record.proxy_index}")
        page_candidates = np.flatnonzero(
            np.isclose(pages_lat, record.latitude, atol=COORDINATE_TOLERANCE)
            & np.isclose(pages_lon, record.longitude, atol=COORDINATE_TOLERANCE)
        )
        page_passed = passing_matches(
            combined_years,
            combined_matrix[:, combined_column],
            pages_years,
            pages_matrix,
            page_candidates,
        )
        if len(page_passed) != 1:
            raise RuntimeError(
                f"combined-to-pages-identity-not-unique:{record.proxy_index}:"
                f"candidates={len(page_candidates)}:passing={len(page_passed)}"
            )
        page_match = page_passed[0]
        page_column = int(page_match["column_0based"])

        official_candidates = np.flatnonzero(
            np.isclose(official_lat, pages_lat[page_column], atol=COORDINATE_TOLERANCE)
            & np.isclose(official_lon, pages_lon[page_column], atol=COORDINATE_TOLERANCE)
        )
        official_passed = passing_matches(
            pages_years,
            pages_matrix[:, page_column],
            official_years,
            official_matrix,
            official_candidates,
        )
        if len(official_passed) != 1:
            raise RuntimeError(
                f"pages-to-official-identity-not-unique:{record.proxy_index}:"
                f"candidates={len(official_candidates)}:passing={len(official_passed)}"
            )
        official_match = official_passed[0]
        official_column = int(official_match["column_0based"])
        meta = official_metadata.iloc[official_column]
        identity_rows.append({
            "proxy_index": int(record.proxy_index),
            "latitude": float(record.latitude),
            "longitude": float(record.longitude),
            "combined_column_1based": combined_column + 1,
            "combined_to_pages_coordinate_candidate_count": int(len(page_candidates)),
            "combined_to_pages_passing_count": int(len(page_passed)),
            "pages_reconstruction_column_1based": page_column + 1,
            "pages_reconstruction_name": text_value(pages_names[page_column]),
            "combined_to_pages_correlation": page_match["correlation"],
            "combined_to_pages_affine_slope": page_match["slope"],
            "combined_to_pages_affine_intercept": page_match["intercept"],
            "combined_to_pages_maximum_abs_residual": page_match["maximum_abs_residual"],
            "combined_to_pages_finite_overlap": page_match["finite_overlap"],
            "pages_to_official_coordinate_candidate_count": int(len(official_candidates)),
            "pages_to_official_passing_count": int(len(official_passed)),
            "official_column_1based": official_column + 1,
            "official_dataset_name": clean_group(meta["dataSetName"]),
            "official_tsid": clean_group(meta["paleoData_TSid"]),
            "official_archive_type_index": clean_group(meta["archiveType"]),
            "official_resolution_median": clean_group(meta["resMed"]),
            "pages_to_official_correlation": official_match["correlation"],
            "pages_to_official_affine_slope": official_match["slope"],
            "pages_to_official_affine_intercept": official_match["intercept"],
            "pages_to_official_maximum_abs_residual": official_match["maximum_abs_residual"],
            "pages_to_official_finite_overlap": official_match["finite_overlap"],
        })
    identity = pd.DataFrame(identity_rows).sort_values("proxy_index").reset_index(drop=True)

    ts_values = np.asarray(loadmat(PAGES_TS, squeeze_me=True, struct_as_record=False, variable_names=["TS"])["TS"]).reshape(-1)
    ts_index: dict[tuple[str, str], list[object]] = defaultdict(list)
    for item in ts_values:
        ts_index[(clean_group(getattr(item, "dataSetName", "")), clean_group(getattr(item, "paleoData_TSid", "")))].append(item)

    rich_rows: list[dict[str, object]] = []
    for row in identity.itertuples(index=False):
        key = (row.official_dataset_name, row.official_tsid)
        matches = ts_index.get(key, [])
        if len(matches) != 1:
            raise RuntimeError(f"official-to-ts-identity-not-unique:{row.proxy_index}:{key}:{len(matches)}")
        item = matches[0]
        rich = {"proxy_index": int(row.proxy_index), "ts_identity_match_count": 1}
        for output_name, field_name in METADATA_FIELDS.items():
            rich[output_name] = clean_group(getattr(item, field_name, ""))
        rich_rows.append(rich)
    rich_metadata = pd.DataFrame(rich_rows)
    records = fixed.merge(identity, on=["proxy_index", "latitude", "longitude"], how="left", validate="one_to_one")
    records = records.merge(rich_metadata, on="proxy_index", how="left", validate="one_to_one")
    for field in METADATA_FIELDS:
        records[field] = records[field].fillna("<non_pages2k_named_tree>")

    group_fields = [
        "archive_species", "archive_genus", "proxy", "measurement_material",
        "measurement_method", "data_type", "variable_name", "climate_variable",
        "seasonality", "interpretation_direction",
    ]
    group_summary, deletions = group_audit(records, group_fields)
    eligible = deletions[deletions["eligible_interpretation"]].copy()
    all_eligible = eligible[eligible["scope"] == "all_fixed_records"]
    north_eligible = eligible[eligible["scope"] == "north_ge_35n"]
    all_weakest = all_eligible.sort_values("retained_site_mean_event_change_z", ascending=False).head(1)
    north_weakest = north_eligible.sort_values("retained_site_mean_event_change_z", ascending=False).head(1)

    output.mkdir(parents=True)
    identity.to_csv(output / "pages2k_identity_chain.csv", index=False)
    records.to_csv(output / "fixed_network_rich_pages2k_metadata.csv", index=False)
    group_summary.to_csv(output / "metadata_group_summary.csv", index=False)
    deletions.to_csv(output / "metadata_group_deletion_audit.csv", index=False)

    full_site_mean = float(collapse_sites(records)["event_change_z"].mean())
    north_site_mean = float(collapse_sites(records[records["latitude"] >= 35.0])["event_change_z"].mean())
    summary = {
        "fixed_record_count": int(len(records)),
        "pages2k_record_count": int(len(identity)),
        "full_unique_site_count": int(records["site_id"].nunique()),
        "pages2k_unique_site_count": int(records.loc[records["source_family"] == "PAGES2k2017", "site_id"].nunique()),
        "north_ge_35n_pages2k_record_count": int(((records["source_family"] == "PAGES2k2017") & (records["latitude"] >= 35.0)).sum()),
        "minimum_combined_to_pages_correlation": float(identity["combined_to_pages_correlation"].min()),
        "maximum_combined_to_pages_affine_residual": float(identity["combined_to_pages_maximum_abs_residual"].max()),
        "minimum_pages_to_official_correlation": float(identity["pages_to_official_correlation"].min()),
        "maximum_pages_to_official_affine_residual": float(identity["pages_to_official_maximum_abs_residual"].max()),
        "all_identity_chains_unique": bool((identity["combined_to_pages_passing_count"] == 1).all() and (identity["pages_to_official_passing_count"] == 1).all()),
        "full_site_mean_event_change_z": full_site_mean,
        "north_ge_35n_site_mean_event_change_z": north_site_mean,
        "eligible_group_deletion_count_all": int(len(all_eligible)),
        "eligible_group_deletion_count_north": int(len(north_eligible)),
        "all_eligible_deletions_retain_negative_sign": bool(all_eligible["retained_sign_negative"].all()),
        "north_eligible_deletions_retain_negative_sign": bool(north_eligible["retained_sign_negative"].all()),
        "weakest_all_scope_deletion": None if all_weakest.empty else all_weakest.iloc[0].to_dict(),
        "weakest_north_scope_deletion": None if north_weakest.empty else north_weakest.iloc[0].to_dict(),
        "interpretation": "Metadata groups are sensitivity strata, not grounds for deleting proxies unless independent evidence establishes a metadata/operator error.",
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), constrained_layout=True)
    pages_records = records[records["source_family"] == "PAGES2k2017"].copy()
    proxy_counts = pages_records["proxy"].value_counts()
    top_proxy = set(proxy_counts.head(5).index)
    pages_records["proxy_plot_group"] = pages_records["proxy"].where(pages_records["proxy"].isin(top_proxy), "other")
    palette = plt.get_cmap("tab10")
    for color_index, (group, subset) in enumerate(sorted(pages_records.groupby("proxy_plot_group"))):
        axes[0].scatter(subset["latitude"], subset["event_change_z"], s=42, alpha=0.85, label=group, color=palette(color_index))
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].axvline(35, color="0.4", linewidth=0.9, linestyle="--")
    axes[0].set_xlabel("Latitude (degrees N)")
    axes[0].set_ylabel("1901-1920 event change (z)")
    axes[0].set_title("Fixed PAGES records")
    axes[0].legend(fontsize=7, loc="best", frameon=False)

    for axis, scope, title in (
        (axes[1], "all_fixed_records", "All-site group deletions"),
        (axes[2], "north_ge_35n", "North >=35N group deletions"),
    ):
        view = eligible[eligible["scope"] == scope].sort_values("retained_site_mean_event_change_z", ascending=False).head(15).copy()
        view = view.sort_values("retained_site_mean_event_change_z")
        labels = [f"{field}: {group}" for field, group in zip(view["metadata_field"], view["omitted_group"])]
        axis.scatter(view["retained_site_mean_event_change_z"], np.arange(len(view)), color="#C9483B", s=36)
        axis.axvline(0, color="black", linewidth=0.8)
        if len(view):
            axis.axvline(float(view["full_site_mean_event_change_z"].iloc[0]), color="#247BA0", linewidth=1.1, linestyle="--")
        axis.set_yticks(np.arange(len(view)), labels=labels, fontsize=7)
        axis.set_xlabel("Retained site mean event change (z)")
        axis.set_title(title)
    fig.savefig(output / "pages2k_metadata_family_robustness.png", dpi=220)
    plt.close(fig)

    after = {name: source_state(path) for name, path in sources.items()}
    unchanged = before == after
    summary["all_declared_sources_unchanged"] = unchanged
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("declared-source-changed-during-audit")

    weakest_all_text = "none" if all_weakest.empty else (
        f"{all_weakest.iloc[0]['metadata_field']}={all_weakest.iloc[0]['omitted_group']} -> "
        f"{all_weakest.iloc[0]['retained_site_mean_event_change_z']:.3f} z"
    )
    weakest_north_text = "none" if north_weakest.empty else (
        f"{north_weakest.iloc[0]['metadata_field']}={north_weakest.iloc[0]['omitted_group']} -> "
        f"{north_weakest.iloc[0]['retained_site_mean_event_change_z']:.3f} z"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# PAGES2k Fixed-network Metadata-family Audit v1\n\n"
        f"All {len(identity)} PAGES2k records were uniquely closed through both sequence-mapping stages and uniquely linked to full TS metadata.\n\n"
        f"The full fixed-network site mean is {full_site_mean:.3f} z; the >=35N site mean is {north_site_mean:.3f} z. "
        f"The weakest eligible all-site deletion is {weakest_all_text}; the weakest northern deletion is {weakest_north_text}.\n\n"
        "Metadata-family deletion is a robustness diagnostic only. It does not establish that an omitted family is erroneous.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
