from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from scipy.io import loadmat


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
LOCKED_START = 1921
LOCKED_END = 1939
MATRIX_YEARS = np.arange(1940, 2001, dtype=int)
TARGET_YEARS = np.arange(1940, 2000, dtype=int)
BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 1999))
SITE_LATITUDES = np.array([27 + 47 / 60, 27 + 11 / 60, 28 + 4 / 60, 26 + 26 / 60, 27 + 10 / 60])
SITE_LONGITUDES = np.array([98 + 28 / 60, 98 + 43 / 60, 98 + 45 / 60, 98 + 46 / 60, 98 + 43 / 60])


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if stat.st_size <= 100_000_000 else None
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }


def workbook_sheet_path(archive: zipfile.ZipFile, title: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in relationships.findall(f"{{{NS_PKG_REL}}}Relationship")
    }
    for sheet in workbook.findall(f".//{{{NS_MAIN}}}sheet"):
        if sheet.attrib["name"] != title:
            continue
        target = targets[sheet.attrib[f"{{{NS_REL}}}id"]].replace("\\", "/")
        if target.startswith("/"):
            return target.lstrip("/")
        return target if target.startswith("xl/") else f"xl/{target}"
    raise KeyError(title)


def cell_numeric(cell: ET.Element | None) -> float:
    if cell is None:
        raise ValueError("missing-cell")
    node = cell.find(f"{{{NS_MAIN}}}v")
    if node is None or node.text is None:
        raise ValueError("missing-numeric-value")
    return float(node.text)


def parse_modern_two_columns(
    path: Path, sheet_title: str, has_header: bool
) -> tuple[pd.Series, dict[str, Any]]:
    values: dict[int, float] = {}
    locked_rows_skipped = 0
    outside_modern_value_cells_skipped = 0
    with zipfile.ZipFile(path) as archive:
        sheet_path = workbook_sheet_path(archive, sheet_title)
        root = ET.fromstring(archive.read(sheet_path))
        for row in root.findall(f".//{{{NS_MAIN}}}row"):
            row_number = int(row.attrib.get("r", "0") or 0)
            if has_header and row_number == 1:
                continue
            cells = {cell.attrib.get("r", "")[0]: cell for cell in row.findall(f"{{{NS_MAIN}}}c")}
            if "A" not in cells:
                continue
            try:
                year = int(round(cell_numeric(cells["A"])))
            except ValueError:
                continue
            if LOCKED_START <= year <= LOCKED_END:
                locked_rows_skipped += 1
                continue
            if not 1940 <= year <= 2000:
                outside_modern_value_cells_skipped += int("B" in cells)
                continue
            values[year] = cell_numeric(cells.get("B"))
    series = pd.Series(values, dtype=float).sort_index()
    if any(LOCKED_START <= int(year) <= LOCKED_END for year in series.index):
        raise RuntimeError("locked-year-entered-modern-series")
    metadata = {
        "sheet_title": sheet_title,
        "modern_value_count": int(series.notna().sum()),
        "locked_rows_skipped_before_column_b_conversion": locked_rows_skipped,
        "outside_modern_column_b_values_skipped": outside_modern_value_cells_skipped,
        "locked_column_b_values_decoded": False,
        "paleo_column_b_values_decoded": False,
    }
    return series, metadata


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def affine_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(candidate) & np.isfinite(reference)
    count = int(valid.sum())
    if count < 3:
        return {"overlap_count": count, "correlation": np.nan, "maximum_abs_affine_residual": np.nan}
    left = candidate[valid]
    right = reference[valid]
    design = np.column_stack((right, np.ones(count)))
    coefficient = np.linalg.lstsq(design, left, rcond=None)[0]
    residual = float(np.max(np.abs(left - design @ coefficient)))
    return {
        "overlap_count": count,
        "correlation": correlation(left, right),
        "maximum_abs_affine_residual": residual,
    }


def main() -> int:
    root = project_root()
    drivers = Path(__file__).resolve().parent
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    output = loop / "round_180_hengduan_chronology_modern_oof_v1"
    output.mkdir(parents=True, exist_ok=False)
    source_xlsx = loop / (
        "round_178_mendeley_east_hma_temperature_candidates_v1/raw_unopened/"
        "889n6ccx4z_v1/Upload information.xlsx"
    )
    matrix_dir = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    )
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": drivers / "round180_hengduan_chronology_modern_oof_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "source_xlsx": source_xlsx,
        "round179": loop / "round_179_mendeley_temperature_structure_identity_v1/summary.json",
        "proxy6": matrix_dir / "proxy6_allowed_1940_2000_official_baseline_v2.mat",
        "proxy9": matrix_dir / "proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "proxy9_inventory": loop / "round_07_proxy_network_suitability_inventory_v2/proxy9_suitability_inventory.csv",
        "bea": phyda / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
        "cru": phyda / "input-recon/cruts.2004151855.v4.04/tmp/cru_ts4.04.1901.2019.tmp.dat.nc",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}

    chronology, chronology_parse = parse_modern_two_columns(
        source_xlsx, "Tree-ring width chronology", True
    )
    published, published_parse = parse_modern_two_columns(
        source_xlsx, "Temperature reconstruction", True
    )
    chronology_matrix = chronology.reindex(MATRIX_YEARS).to_numpy(float)
    published_matrix = published.reindex(MATRIX_YEARS).to_numpy(float)
    if not np.isfinite(chronology_matrix).all() or not np.isfinite(published_matrix).all():
        raise RuntimeError("candidate-modern-gap")
    lineage_correlation = correlation(chronology_matrix, published_matrix)

    proxy9_inventory = pd.read_csv(paths["proxy9_inventory"])
    proxy9_identity = {
        index: (
            str(rows.iloc[0].identity_key)
            if not (rows := proxy9_inventory[proxy9_inventory.record_index_1based == index]).empty
            else ""
        )
        for index in range(1, 298)
    }
    alias_rows: list[dict[str, Any]] = []
    for network in ("proxy6", "proxy9"):
        observations = np.asarray(
            loadmat(paths[network], variable_names=["observations"])["observations"],
            dtype=float,
        )
        if observations.shape[0] != len(MATRIX_YEARS):
            raise RuntimeError(f"{network}-year-shape:{observations.shape}")
        for column in range(observations.shape[1]):
            metrics = affine_metrics(chronology_matrix, observations[:, column])
            corr = float(metrics["correlation"])
            residual = float(metrics["maximum_abs_affine_residual"])
            alias_rows.append(
                {
                    "network": network,
                    "column_1based": column + 1,
                    "proxy9_identity_if_available": (
                        proxy9_identity.get(column + 1, "") if network == "proxy9" else ""
                    ),
                    **metrics,
                    "exact_affine_alias": bool(
                        metrics["overlap_count"] >= 50
                        and abs(corr) >= 0.999999
                        and residual <= 1e-8
                    ),
                    "near_alias": bool(metrics["overlap_count"] >= 50 and abs(corr) >= 0.995),
                }
            )
    alias = pd.DataFrame(alias_rows)
    alias.to_csv(output / "proxy6_proxy9_alias_metrics.csv", index=False)
    top_alias = alias.reindex(alias.correlation.abs().sort_values(ascending=False).index).groupby("network", as_index=False).head(5)
    top_alias.to_csv(output / "proxy6_proxy9_top_aliases.csv", index=False)
    alias_rejected = bool(alias.exact_affine_alias.any() or alias.near_alias.any())

    oof_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    gate = None
    decision = "close_hengduan_existing_proxy_alias" if alias_rejected else "pending_modern_oof"
    if not alias_rejected:
        r118 = load_module(
            "round180_r118",
            drivers / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
        )
        bea_monthly = r118.read_berkeley_points(paths["bea"], SITE_LATITUDES, SITE_LONGITUDES)
        cru_monthly = r118.read_netcdf3_points(paths["cru"], "tmp", 1901, SITE_LATITUDES, SITE_LONGITUDES)
        proxy = chronology.reindex(TARGET_YEARS).to_numpy(float)
        if not np.isfinite(proxy).all():
            raise RuntimeError("chronology-oof-gap")
        predictions: dict[str, np.ndarray] = {}
        targets: dict[str, np.ndarray] = {}
        for product, monthly in (("BEA", bea_monthly), ("CRU", cru_monthly)):
            annual = r118.aggregate_temperature(monthly, [tuple(range(5, 10))])
            if annual.shape != (61, 5) or not np.isfinite(annual).all():
                raise RuntimeError(f"temperature-target-failed:{product}:{annual.shape}")
            target = np.mean(annual[:60, :], axis=1)
            targets[product] = target
            prediction = np.full(60, np.nan)
            climatology = np.full(60, np.nan)
            for fold_index, (start, end) in enumerate(BLOCKS, start=1):
                test = (TARGET_YEARS >= start) & (TARGET_YEARS <= end)
                train = ~test
                design = np.column_stack((np.ones(int(train.sum())), proxy[train]))
                coefficients = np.linalg.lstsq(design, target[train], rcond=None)[0]
                prediction[test] = coefficients[0] + coefficients[1] * proxy[test]
                climatology[test] = np.mean(target[train])
                fold_rows.append(
                    {
                        "product": product,
                        "fold": fold_index,
                        "test_years": f"{start}-{end}",
                        "fit_intercept": float(coefficients[0]),
                        "fit_slope": float(coefficients[1]),
                        "test_correlation": correlation(target[test], prediction[test]),
                        "test_rmse_degC": float(np.sqrt(np.mean((target[test] - prediction[test]) ** 2))),
                    }
                )
            predictions[product] = prediction
            product_folds = pd.DataFrame([row for row in fold_rows if row["product"] == product])
            candidate_sse = float(np.sum((target - prediction) ** 2))
            baseline_sse = float(np.sum((target - climatology) ** 2))
            early = TARGET_YEARS <= 1969
            late = TARGET_YEARS >= 1970
            oof_rows.append(
                {
                    "product": product,
                    "oof_correlation": correlation(target, prediction),
                    "oof_skill_vs_training_fold_climatology": 1 - candidate_sse / baseline_sse,
                    "oof_rmse_degC": float(np.sqrt(candidate_sse / len(TARGET_YEARS))),
                    "positive_fold_correlation_count": int((product_folds.test_correlation > 0).sum()),
                    "positive_fit_slope_count": int((product_folds.fit_slope > 0).sum()),
                    "early_split_proxy_target_correlation": correlation(proxy[early], target[early]),
                    "late_split_proxy_target_correlation": correlation(proxy[late], target[late]),
                }
            )
        oof = pd.DataFrame(oof_rows)
        product_prediction_correlation = correlation(predictions["BEA"], predictions["CRU"])
        gate = {
            "both_oof_correlations_ge_0_30": bool((oof.oof_correlation >= 0.30).all()),
            "both_oof_skills_positive": bool((oof.oof_skill_vs_training_fold_climatology > 0).all()),
            "both_products_four_of_five_fold_correlations_positive": bool((oof.positive_fold_correlation_count >= 4).all()),
            "all_ten_fit_slopes_positive": bool((oof.positive_fit_slope_count == 5).all()),
            "both_products_early_late_split_correlations_positive": bool(
                ((oof.early_split_proxy_target_correlation > 0) & (oof.late_split_proxy_target_correlation > 0)).all()
            ),
            "bea_cru_oof_prediction_correlation_ge_0_80": bool(product_prediction_correlation >= 0.80),
        }
        decision = "advance_hengduan_to_source_lineage_and_psm_design" if all(gate.values()) else "close_hengduan_modern_temperature_oof"
        pd.DataFrame(fold_rows).to_csv(output / "blocked_oof_fold_metrics.csv", index=False)
        oof.to_csv(output / "blocked_oof_summary.csv", index=False)
        curves = pd.DataFrame({"year": TARGET_YEARS, "chronology": proxy})
        for product in ("BEA", "CRU"):
            curves[f"{product}_observed"] = targets[product]
            curves[f"{product}_oof"] = predictions[product]
        curves.to_csv(output / "modern_oof_curves.csv", index=False)

    pd.DataFrame(
        {
            "year": MATRIX_YEARS,
            "chronology": chronology_matrix,
            "published_reconstruction_lineage_only": published_matrix,
        }
    ).to_csv(output / "modern_candidate_series.csv", index=False)
    after = {name: fingerprint(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "candidate": "Mendeley 889n6ccx4z v1 Hengduan regional chronology",
        "documented_target": "May-September mean temperature",
        "site_count": 5,
        "site_mean_latitude": float(np.mean(SITE_LATITUDES)),
        "site_mean_longitude": float(np.mean(SITE_LONGITUDES)),
        "chronology_parse": chronology_parse,
        "published_reconstruction_parse": published_parse,
        "modern_chronology_vs_published_reconstruction_correlation": lineage_correlation,
        "exact_alias_count": int(alias.exact_affine_alias.sum()),
        "near_alias_count": int(alias.near_alias.sum()),
        "proxy_alias_rejected": alias_rejected,
        "top_aliases": top_alias.replace({np.nan: None}).to_dict(orient="records"),
        "oof": oof_rows,
        "modern_gate": gate,
        "decision": decision,
        "published_reconstruction_used_as_oof_predictor": False,
        "paleo_trajectory_exported_or_inspected": False,
        "locked_1921_1939_numeric_values_decoded": False,
        "jionglacuo_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": True,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="ascii"
    )
    lines = [
        "# Round 180 Hengduan Chronology Modern OOF",
        "",
        f"- Exact proxy6/proxy9 aliases: `{result['exact_alias_count']}`; near aliases: `{result['near_alias_count']}`.",
        f"- Chronology versus published reconstruction, 1940-2000: `{lineage_correlation:.3f}`.",
    ]
    for row in oof_rows:
        lines.append(
            f"- {row['product']}: OOF r=`{row['oof_correlation']:.3f}`, "
            f"skill=`{row['oof_skill_vs_training_fold_climatology']:.3f}`, "
            f"positive folds=`{row['positive_fold_correlation_count']}/5`."
        )
    lines.extend(["", f"Decision: `{decision}`."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
