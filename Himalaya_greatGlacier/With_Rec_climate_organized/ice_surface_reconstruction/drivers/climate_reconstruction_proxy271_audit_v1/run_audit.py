"""Audit proxy9 record 271 provenance, metadata direction, and target contribution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def align(years: np.ndarray, values: np.ndarray, target_years: np.ndarray) -> np.ndarray:
    by_year = {int(y): float(v) for y, v in zip(years, values) if np.isfinite(y) and np.isfinite(v)}
    return np.array([by_year.get(int(y), np.nan) for y in target_years], dtype=float)


def affine(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(left) & np.isfinite(right)
    x = right[valid]
    y = left[valid]
    A = np.column_stack([x, np.ones(len(x))])
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    residual = y - A @ np.array([slope, intercept])
    return {
        "n": int(valid.sum()),
        "correlation": float(np.corrcoef(y, x)[0, 1]),
        "slope": float(slope),
        "intercept": float(intercept),
        "maximum_abs_residual": float(np.max(np.abs(residual))),
    }


def decode_char(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1, order="F")
    return "".join(chr(int(v)) for v in values if int(v)).strip()


def trend(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    x = years[valid].astype(float)
    x -= x.mean()
    return float(100.0 * np.polyfit(x, values[valid], 1)[0])


def main() -> int:
    project = root()
    driver = Path(__file__).resolve().parent
    cfg = json.loads((driver / "experiment_config.json").read_text(encoding="utf-8"))
    src = {key: Path(value) for key, value in cfg["sources"].items()}
    paths = {key: (project / value if not value.is_absolute() else value) for key, value in src.items()}
    output = project / cfg["result_root"]
    output.mkdir(parents=True, exist_ok=True)
    hashes_before = {key: sha256(path) for key, path in paths.items()}

    record_id = int(cfg["record"]["proxy9_record_1based"])
    with h5py.File(paths["proxy9"], "r") as archive:
        refs = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        proxy = archive[refs[record_id - 1]]
        proxy_years_all = np.asarray(proxy["age_yng"], dtype=float).reshape(-1, order="F")
        proxy_values_all = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
        finite = np.isfinite(proxy_years_all) & np.isfinite(proxy_values_all)
        proxy_years = proxy_years_all[finite].astype(int)
        proxy_values = proxy_values_all[finite]
        location = np.asarray(proxy["location"], dtype=float).reshape(-1, order="F")[:2]
        ptype = decode_char(proxy["ptype"])
        psm_method = decode_char(proxy["psm_mthd"])
        coefficients = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
        pcr = float(np.asarray(proxy["pcr"]).squeeze())
        pcr_t = float(np.asarray(proxy["pcr_t"]).squeeze())
        pcr_p = float(np.asarray(proxy["pcr_p"]).squeeze())
        pval = float(np.asarray(proxy["pval"]).squeeze())
        mt = np.asarray(proxy["M_t"], dtype=float).reshape(-1, order="F")
        mp = np.asarray(proxy["M_p"], dtype=float).reshape(-1, order="F")
        hxb = np.asarray(proxy["HXb"], dtype=float).reshape(-1, order="F")

    combined = scipy.io.loadmat(paths["combined"], simplify_cells=True)
    combined_years = np.asarray(combined["year"], dtype=float).reshape(-1).astype(int)
    combined_data = np.asarray(combined["combined6_data"], dtype=float)
    lat = np.asarray(combined["p_lat"], dtype=float).reshape(-1)
    lon = np.asarray(combined["p_lon"], dtype=float).reshape(-1)
    names = np.asarray(combined["dataName"]).reshape(-1)
    archives = np.atleast_1d(combined["archive"])
    flags = np.asarray(combined["source_flag"], dtype=int).reshape(-1)
    candidates = np.flatnonzero(np.isclose(lat, location[0]) & np.isclose(lon, location[1]))
    if len(candidates) != 1:
        raise RuntimeError(f"combined-coordinate-match-count:{len(candidates)}")
    column = int(candidates[0])
    combined_aligned = align(combined_years, combined_data[:, column], proxy_years)
    combined_closure = affine(proxy_values, combined_aligned)

    lmr_id = cfg["record"]["lmr_id"]
    meta = pd.read_csv(paths["lmr_metadata"])
    meta_rows = meta.loc[meta["NCDC ID"].eq(lmr_id)]
    if len(meta_rows) != 1:
        raise RuntimeError(f"lmr-metadata-match-count:{len(meta_rows)}")
    meta_row = meta_rows.iloc[0]
    lmr = pd.read_csv(paths["lmr_series"], usecols=["Year C.E.", lmr_id])
    lmr_aligned = align(lmr["Year C.E."].to_numpy(), lmr[lmr_id].to_numpy(), proxy_years)
    lmr_closure = affine(proxy_values, lmr_aligned)

    arrays = np.load(paths["baseline_arrays"])
    years = np.asarray(arrays["years"], dtype=int)
    contribution = np.asarray(arrays["proxy9_target_contributions"], dtype=float)[record_id - 1]
    early = (years >= cfg["periods"]["early_20c"][0]) & (years <= cfg["periods"]["early_20c"][1])
    contribution_table = pd.DataFrame({
        "year": years,
        "proxy241_target_contribution": contribution,
        "proxy271_target_contribution": contribution,
    })
    contribution_table = contribution_table[["year", "proxy271_target_contribution"]]
    contribution_table.to_csv(output / "proxy271_target_contribution.csv", index=False)

    calibration_mask = (proxy_years >= cfg["periods"]["calibration"][0]) & (proxy_years <= cfg["periods"]["calibration"][1])
    early_proxy_mask = (proxy_years >= cfg["periods"]["early_20c"][0]) & (proxy_years <= cfg["periods"]["early_20c"][1])
    proxy_stats = {
        "calibration_proxy_mean": float(np.nanmean(proxy_values[calibration_mask])),
        "calibration_proxy_std": float(np.nanstd(proxy_values[calibration_mask], ddof=1)),
        "early_proxy_mean": float(np.nanmean(proxy_values[early_proxy_mask])),
        "early_proxy_std": float(np.nanstd(proxy_values[early_proxy_mask], ddof=1)),
        "full_proxy_mean": float(np.nanmean(proxy_values)),
        "full_proxy_std": float(np.nanstd(proxy_values, ddof=1)),
    }
    psm_table = pd.DataFrame([{
        "proxy9_record": record_id,
        "combined_column_1based": column + 1,
        "location_lat": float(location[0]),
        "location_lon": float(location[1]),
        "ptype": ptype,
        "psm_method": psm_method,
        "b0": float(coefficients[0]),
        "b_temperature": float(coefficients[1]),
        "b_precipitation": float(coefficients[2]),
        "pcr": pcr,
        "pcr_t": pcr_t,
        "pcr_p": pcr_p,
        "pval": pval,
        "model_predictor_temperature_min": float(np.nanmin(mt)),
        "model_predictor_temperature_max": float(np.nanmax(mt)),
        "model_predictor_precipitation_min": float(np.nanmin(mp)),
        "model_predictor_precipitation_max": float(np.nanmax(mp)),
        "hxb_min": float(np.nanmin(hxb)),
        "hxb_max": float(np.nanmax(hxb)),
        "metadata_relation": str(meta_row["Relation_to_climateVariable"]),
        "metadata_seasonality": str(meta_row["Seasonality"]),
        "metadata_climate_variable": str(meta_row["climateVariable"]),
        "metadata_measurement": str(meta_row["Proxy measurement"]),
    }])
    psm_table.to_csv(output / "proxy271_psm_diagnostics.csv", index=False)

    identity = {
        "proxy9_record_1based": record_id,
        "location": location.tolist(),
        "proxy_year_range": [int(proxy_years.min()), int(proxy_years.max())],
        "proxy_count": int(len(proxy_years)),
        "combined_column_1based": column + 1,
        "combined_name": str(names[column]).strip(),
        "combined_archive": str(archives[column]).strip(),
        "combined_source_flag": int(flags[column]),
        "lmr_id": lmr_id,
        "lmr_site_name": str(meta_row["Site name"]),
        "lmr_study_name": str(meta_row["Study name"]),
        "lmr_investigators": str(meta_row["Investigators"]),
        "lmr_measurement": str(meta_row["Proxy measurement"]),
        "lmr_climate_variable": str(meta_row["climateVariable"]),
        "lmr_relation": str(meta_row["Relation_to_climateVariable"]),
        "lmr_seasonality": str(meta_row["Seasonality"]),
        "lmr_year_range": [int(meta_row["Oldest (C.E.)"]), int(meta_row["Youngest (C.E.)"])],
        "proxy_stats": proxy_stats,
        "source_closure": {"combined": combined_closure, "lmr": lmr_closure},
        "archived_psm": {
            "coefficients": coefficients.tolist(),
            "pcr": pcr,
            "pcr_t": pcr_t,
            "pcr_p": pcr_p,
            "pval": pval,
        },
        "interpretation": "Exact source closure establishes identity, not PSM validity. The LMR metadata says positive JJA temperature relation, while the archived annual bivariate fit has a negative temperature coefficient and negative calibration temperature correlation; this is a model-direction mismatch requiring a controlled sign/seasonality test, not silent deletion.",
    }
    (output / "identity_and_psm_audit.json").write_text(json.dumps(identity, indent=2, ensure_ascii=True), encoding="utf-8")

    plt.figure(figsize=(12, 7))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(proxy_years, proxy_values, color="#333333", lw=1.2, label="proxy9 #271 standardized")
    ax1.axvspan(1901, 1950, color="#dddddd", alpha=0.35)
    ax1.axhline(0, color="#777777", lw=0.7)
    ax1.set_ylabel("Proxy index")
    ax1.set_title("Proxy9 #271: Kanasar / asia_indi010B:trsgi_breit")
    ax1.grid(alpha=0.2)
    ax1.legend(frameon=False)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(years, contribution, color="#c44e52", lw=1.4, label="#271 contribution to target 3584")
    ax2.axvspan(1901, 1950, color="#dddddd", alpha=0.35)
    ax2.axhline(0, color="#777777", lw=0.7)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("degC contribution")
    ax2.grid(alpha=0.2)
    ax2.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output / "proxy271_source_and_target_contribution.png", dpi=220)
    plt.close()

    hashes_after = {key: sha256(path) for key, path in paths.items()}
    pure = hashes_before == hashes_after
    summary = {
        "experiment_id": cfg["experiment_id"],
        "status": "pass" if pure and combined_closure["correlation"] >= cfg["gates"]["minimum_correlation"] and lmr_closure["correlation"] >= cfg["gates"]["minimum_correlation"] and combined_closure["maximum_abs_residual"] <= cfg["gates"]["maximum_affine_residual"] and lmr_closure["maximum_abs_residual"] <= cfg["gates"]["maximum_affine_residual"] else "fail",
        "source_hashes_unchanged": pure,
        "identity": identity,
        "contribution": {
            "early_20c_trend_c_per_century": trend(years[early], contribution[early]),
            "early_20c_mean": float(np.mean(contribution[early])),
            "early_20c_min": float(np.min(contribution[early])),
            "early_20c_max": float(np.max(contribution[early])),
        },
        "next_experiment_question": "Does correcting #271 to its documented positive JJA temperature relation reduce the target-3584 early-20th-century phase conflict, while freezing chronology, all other proxy operators, prior, localization, and error model?",
        "boundary": "This audit does not authorize sign correction, proxy deletion, or OGGM forcing.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    pd.DataFrame([{
        "proxy9_record": record_id,
        "combined_column_1based": column + 1,
        "combined_name": str(names[column]).strip(),
        "lmr_id": lmr_id,
        "site_name": str(meta_row["Site name"]),
        "latitude": float(location[0]),
        "longitude": float(location[1]),
        "elevation_m": float(meta_row["Elev"]),
        "measurement": str(meta_row["Proxy measurement"]),
        "metadata_relation": str(meta_row["Relation_to_climateVariable"]),
        "metadata_seasonality": str(meta_row["Seasonality"]),
        "psm_temperature_coefficient": float(coefficients[1]),
        "psm_temperature_correlation": pcr_t,
        "psm_precipitation_coefficient": float(coefficients[2]),
        "psm_precipitation_correlation": pcr_p,
        "source_correlation_combined": combined_closure["correlation"],
        "source_correlation_lmr": lmr_closure["correlation"],
    }]).to_csv(output / "record_public_identity.csv", index=False)
    (output / "input_manifest.json").write_text(json.dumps({"before": hashes_before, "after": hashes_after}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Proxy9 Record 271 Audit\n\n"
        f"Status: **{summary['status']}**. Source hashes unchanged: **{pure}**.\n\n"
        "## Identity and closure\n\n"
        f"- Exact merged column: `{column + 1}` (`{str(names[column]).strip()}`).\n"
        f"- Public record: `{lmr_id}`, {meta_row['Site name']}, {meta_row['Proxy measurement']}.\n"
        f"- Combined closure correlation: `{combined_closure['correlation']:.12f}`, maximum affine residual `{combined_closure['maximum_abs_residual']:.3e}`.\n"
        f"- LMR closure correlation: `{lmr_closure['correlation']:.12f}`, maximum affine residual `{lmr_closure['maximum_abs_residual']:.3e}`.\n\n"
        "## Decision-relevant mismatch\n\n"
        f"LMR metadata declares relation `{meta_row['Relation_to_climateVariable']}` and seasonality `{meta_row['Seasonality']}`, while the archived annual bivariate PSM has temperature coefficient `{coefficients[1]:.6f}` and calibration temperature correlation `{pcr_t:.6f}`. This is a documented operator-direction mismatch, not a source identity failure.\n\n"
        "The next controlled experiment may test a positive JJA operator for #271, but this audit alone does not authorize sign correction, deletion, or OGGM forcing.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if summary["status"] != "pass":
        raise RuntimeError("proxy271-audit-hard-gate-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())