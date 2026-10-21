"""Close identity and scientific roles for east-HMA external hydroclimate data."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


YEARS = np.concatenate((np.arange(1000, 1921), np.arange(1940, 2001))).astype(int)
PRECIP_SITES = {
    "biru_precipitation": (31.55, 93.50),
    "jiali_precipitation": (30.60, 93.66),
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    if np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def series_from_csv(path: Path, year_column: str, value_column: str) -> np.ndarray:
    frame = pd.read_csv(path)
    years = frame[year_column].to_numpy(dtype=int)
    if ((years >= 1921) & (years <= 1939)).any():
        raise RuntimeError(f"locked-year-in-allowed-export:{path}")
    values = frame[value_column].to_numpy(dtype=float)
    lookup = dict(zip(years, values))
    return np.asarray([lookup.get(int(year), np.nan) for year in YEARS], dtype=float)


def affine_metrics(source: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(source) & np.isfinite(candidate)
    count = int(valid.sum())
    if count < 20:
        return {
            "overlap_count": count,
            "correlation": float("nan"),
            "affine_slope": float("nan"),
            "affine_intercept": float("nan"),
            "maximum_abs_residual": float("nan"),
            "normalized_rmse": float("nan"),
        }
    design = np.column_stack((np.ones(count), candidate[valid]))
    coefficients = np.linalg.lstsq(design, source[valid], rcond=None)[0]
    residual = source[valid] - design @ coefficients
    scale = float(np.std(source[valid], ddof=1))
    return {
        "overlap_count": count,
        "correlation": correlation(source[valid], candidate[valid]),
        "affine_slope": float(coefficients[1]),
        "affine_intercept": float(coefficients[0]),
        "maximum_abs_residual": float(np.max(np.abs(residual))),
        "normalized_rmse": float(np.sqrt(np.mean(residual**2)) / scale)
        if scale > 1e-12
        else float("nan"),
    }


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius = 6371.0088
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = phi2 - phi1
    dlambda = np.deg2rad(lon2 - lon1)
    value = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(value))


def centered_rolling(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, center=True, min_periods=window).mean().to_numpy()


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    output = loop / "round_140_east_hma_external_identity_role_closure_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round140_east_hma_external_identity_role_closure_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    base = load_module(
        "round140_base",
        driver / "run_round134_netp_mjj_low_frequency_reconstruction_v2.py",
    )
    archive = load_module(
        "round140_archive",
        driver / "audit_positive_b2_event_seasonal_psm_v1.py",
    )

    r139 = loop / "round_139_east_hma_xlsx_preflight_v2"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "proxy6": archive.archive_proxy_path(),
        "metadata": root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/"
        "proxy6_metadata.csv",
        "source_mapping": root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_proxy_common_mode_source_audit_v2/"
        "proxy_source_mapping.csv",
        "r139_summary": r139 / "summary.json",
        "streamflow": r139 / "sb__streamflow__allowed.csv",
        "biru": r139 / "sb__biru_reconstruction__allowed.csv",
        "jiali": r139 / "sb__jiali_reconstruction__allowed.csv",
        "three_rivers": r139 / "tr__sheet1__allowed.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    r139_summary = json.loads(paths["r139_summary"].read_text(encoding="utf-8"))
    if r139_summary["locked_1921_1939_values_decoded"] != 0:
        raise RuntimeError("round139-boundary-failed")

    external = {
        "same_workbook_streamflow": series_from_csv(
            paths["streamflow"], "Year", "streamflow \uff08108m3\uff09"
        ),
        "biru_precipitation": series_from_csv(
            paths["biru"], "Year", "Precipitation\uff08mm\uff09"
        ),
        "jiali_precipitation": series_from_csv(
            paths["jiali"], "Year", "Precipitation\uff08mm\uff09"
        ),
        "three_rivers_streamflow": series_from_csv(
            paths["three_rivers"], "column_1", "column_2"
        ),
    }
    three_frame = pd.read_csv(paths["three_rivers"])
    helper_counts = {
        column: int(three_frame[column].notna().sum())
        for column in ("column_3", "column_4", "column_5", "column_6")
    }
    if helper_counts["column_3"] != 0 or max(helper_counts.values()) > 5:
        raise RuntimeError(f"unexpected-three-rivers-helper-columns:{helper_counts}")

    metadata = pd.read_csv(paths["metadata"])
    mapping = pd.read_csv(paths["source_mapping"])[
        ["proxy_index", "source_family", "data_name"]
    ]
    metadata = metadata.merge(mapping, on="proxy_index", how="left", validate="one_to_one")
    proxy_ids = tuple(metadata.proxy_index.astype(int).tolist())
    proxy_values = base.read_proxy_values(paths["proxy6"], proxy_ids, YEARS)
    if proxy_values.shape != (len(YEARS), len(proxy_ids)):
        raise RuntimeError("proxy-read-shape-failed")

    match_rows = []
    alias_counts = {}
    for external_name, external_values in external.items():
        aliases = 0
        for column, proxy_id in enumerate(proxy_ids):
            values = affine_metrics(external_values, proxy_values[:, column])
            exact_alias = bool(
                values["overlap_count"] >= 20
                and np.isfinite(values["correlation"])
                and abs(values["correlation"]) >= 0.999999
                and values["maximum_abs_residual"] <= 1e-8
            )
            aliases += int(exact_alias)
            row = metadata.iloc[column]
            match_rows.append(
                {
                    "external_series": external_name,
                    "proxy_index": proxy_id,
                    "proxy_latitude": float(row.latitude),
                    "proxy_longitude": float(row.longitude),
                    "source_family": row.source_family,
                    "data_name": row.data_name,
                    **values,
                    "exact_sequence_alias": exact_alias,
                }
            )
        alias_counts[external_name] = aliases
    matches = pd.DataFrame(match_rows)
    matches["absolute_correlation"] = matches.correlation.abs()
    top_matches = (
        matches.sort_values(
            ["external_series", "exact_sequence_alias", "absolute_correlation"],
            ascending=[True, False, False],
        )
        .groupby("external_series", as_index=False)
        .head(10)
        .reset_index(drop=True)
    )

    site_rows = []
    for name, (latitude, longitude) in PRECIP_SITES.items():
        distance = haversine_km(
            latitude,
            longitude,
            metadata.latitude.to_numpy(dtype=float),
            metadata.longitude.to_numpy(dtype=float),
        )
        degree_distance = np.sqrt(
            (metadata.latitude.to_numpy(dtype=float) - latitude) ** 2
            + (metadata.longitude.to_numpy(dtype=float) - longitude) ** 2
        )
        nearest = int(np.argmin(distance))
        site_rows.append(
            {
                "external_series": name,
                "source_latitude": latitude,
                "source_longitude": longitude,
                "nearest_proxy_index": int(metadata.iloc[nearest].proxy_index),
                "nearest_proxy_latitude": float(metadata.iloc[nearest].latitude),
                "nearest_proxy_longitude": float(metadata.iloc[nearest].longitude),
                "nearest_distance_km": float(distance[nearest]),
                "nearest_coordinate_distance_degrees": float(degree_distance[nearest]),
                "proxy_within_one_degree_count": int((degree_distance <= 1.0).sum()),
                "exact_sequence_alias_count": int(alias_counts[name]),
                "site_independence_pass": bool(
                    (degree_distance <= 1.0).sum() == 0 and alias_counts[name] == 0
                ),
            }
        )
    sites = pd.DataFrame(site_rows)

    paleo_mask = YEARS <= 1920
    pair_rows = []
    for left_name, right_name in (
        ("biru_precipitation", "jiali_precipitation"),
        ("biru_precipitation", "same_workbook_streamflow"),
        ("jiali_precipitation", "same_workbook_streamflow"),
        ("biru_precipitation", "three_rivers_streamflow"),
        ("jiali_precipitation", "three_rivers_streamflow"),
    ):
        left = centered_rolling(np.where(paleo_mask, external[left_name], np.nan), 31)
        right = centered_rolling(np.where(paleo_mask, external[right_name], np.nan), 31)
        use = paleo_mask & np.isfinite(left) & np.isfinite(right)
        pair_rows.append(
            {
                "left_series": left_name,
                "right_series": right_name,
                "first_common_year": int(YEARS[use][0]),
                "last_common_year": int(YEARS[use][-1]),
                "common_year_count": int(use.sum()),
                "paleo_31yr_correlation": correlation(left[use], right[use]),
            }
        )
    pairs = pd.DataFrame(pair_rows)

    direct_independence = bool(sites.site_independence_pass.all())
    roles = pd.DataFrame(
        [
            {
                "series": "biru_precipitation",
                "variable": "previous_Sep_current_Jun_precipitation",
                "role": "independent_validation_only" if direct_independence else "quarantined",
                "may_train": False,
                "may_validate_precipitation_phase": direct_independence,
                "may_validate_precipitation_amplitude": direct_independence,
            },
            {
                "series": "jiali_precipitation",
                "variable": "previous_Jul_current_Apr_precipitation",
                "role": "independent_validation_only" if direct_independence else "quarantined",
                "may_train": False,
                "may_validate_precipitation_phase": direct_independence,
                "may_validate_precipitation_amplitude": direct_independence,
            },
            {
                "series": "same_workbook_streamflow",
                "variable": "streamflow",
                "role": "secondary_hydroclimate_phase_benchmark",
                "may_train": False,
                "may_validate_precipitation_phase": True,
                "may_validate_precipitation_amplitude": False,
            },
            {
                "series": "three_rivers_streamflow",
                "variable": "multi_river_streamflow",
                "role": "secondary_hydroclimate_phase_benchmark_independence_unresolved",
                "may_train": False,
                "may_validate_precipitation_phase": True,
                "may_validate_precipitation_amplitude": False,
            },
        ]
    )

    matches.to_csv(output / "all_external_proxy_sequence_matches.csv", index=False)
    top_matches.to_csv(output / "top_external_proxy_sequence_matches.csv", index=False)
    sites.to_csv(output / "direct_precipitation_site_independence.csv", index=False)
    pairs.to_csv(output / "external_low_frequency_pair_correlations.csv", index=False)
    roles.to_csv(output / "external_series_scientific_roles.csv", index=False)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-file-changed")
    summary = {
        "experiment_id": "round_140_east_hma_external_identity_role_closure_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "external_series_count": len(external),
        "proxy6_record_count": len(proxy_ids),
        "exact_sequence_alias_counts": alias_counts,
        "direct_precipitation_site_independence_pass": direct_independence,
        "three_rivers_helper_nonempty_counts": helper_counts,
        "decision": (
            "advance_east_hma_reconstruction_with_biru_jiali_held_out"
            if direct_independence
            else "stop_direct_precipitation_validation_at_identity_failure"
        ),
        "locked_1921_1939_values_decoded": 0,
        "frozen_external_validation_reserve_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 140 Final Result\n\n"
        f"- Exact sequence aliases: `{alias_counts}`.\n"
        f"- Biru/Jiali site independence: `{direct_independence}`.\n"
        f"- Nearest-site diagnostics: `{sites.to_dict(orient='records')}`.\n"
        f"- Decision: `{summary['decision']}`.\n\n"
        "Biru and Jiali remain validation-only. Streamflow curves remain secondary "
        "phase benchmarks and cannot train or validate precipitation amplitude.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(sites.to_string(index=False))
    print(pairs.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
