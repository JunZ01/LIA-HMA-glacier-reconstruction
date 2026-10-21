"""Evaluate the Chitral PDSI reconstruction without reading locked-period values."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy.io
from lxml import etree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_precipitation_annual_oof_early_candidate_v1 as early


EXPERIMENT_ID = "climate_reconstruction_precipitation_chitral_pdsi_evidence_v1"
EARLY = np.arange(1901, 1921, dtype=int)
MODERN = np.arange(1941, 2001, dtype=int)
LOCKED = set(range(1921, 1940))


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_allowed(zip_path: Path) -> tuple[pd.DataFrame, int]:
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(zip_path) as outer:
        docx = outer.read("Supplement.docx")
    records: list[dict[str, float | int]] = []
    locked_skipped = 0
    with zipfile.ZipFile(io.BytesIO(docx)) as inner:
        with inner.open("word/document.xml") as xml:
            for _, table in etree.iterparse(xml, events=("end",), tag=namespace + "tbl"):
                for row in table.findall(namespace + "tr"):
                    cells = row.findall(namespace + "tc")
                    texts = ["".join(node.text or "" for node in cell.iter(namespace + "t")).strip() for cell in cells]
                    for index in range(0, len(texts) - 1, 2):
                        if not re.fullmatch(r"\d{4}", texts[index]):
                            continue
                        year = int(texts[index])
                        if year in LOCKED:
                            locked_skipped += 1
                            continue
                        if year not in set(EARLY) | set(MODERN):
                            continue
                        value = float(texts[index + 1])
                        records.append({"year": year, "chitral_mar_aug_pdsi": value})
                table.clear()
    result = pd.DataFrame(records).drop_duplicates("year").sort_values("year")
    expected = set(EARLY) | set(MODERN)
    if set(result["year"]) != expected or locked_skipped != 19:
        raise RuntimeError(f"allowed-year-contract-failed:{len(result)}:{locked_skipped}")
    return result, locked_skipped


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def affine_residual(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    design = np.column_stack((np.ones(valid.sum()), left[valid]))
    coef = np.linalg.lstsq(design, right[valid], rcond=None)[0]
    return float(np.max(np.abs(right[valid] - design @ coef)))


def trend(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100.0)


def circular_shift_p(left: np.ndarray, right: np.ndarray) -> float:
    observed = abs(correlation(left, right))
    shifted = np.asarray([abs(correlation(left, np.roll(right, shift))) for shift in range(1, len(right))])
    return float((1 + np.sum(shifted >= observed)) / (1 + len(shifted)))


def main() -> int:
    project = root()
    results = project / "validation_data/results/paleo_surface_validation_v0"
    output = results / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    source_dir = project / "validation_data/sources/noaa_chitral_hindukush2020pdsi_v1"
    loop = results / "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    archive = Path(r"D:\Users\JUNZ\Desktop") / "".join(chr(v) for v in (25991, 20214, 25972, 29702)) / ("1 " + "".join(chr(v) for v in (35770, 25991))) / "work1/compared_result_all"
    sources = {
        "supplement": source_dir / "cp-16-783-2020-supplement.zip",
        "source_note": source_dir / "SOURCE_NOTE.md",
        "allowed_proxy6": loop / "proxy6_allowed_1940_2000_official_baseline_v2.mat",
        "proxy6": archive / "python_data/proxy_data_6.mat",
        "candidate_curves": results / "climate_reconstruction_precipitation_annual_oof_early_candidate_v2/candidate_and_production_regional_curves_1901_1920.csv",
        "candidate_field": results / "climate_reconstruction_precipitation_annual_oof_early_candidate_v2/annual_oof_candidate_native_2deg_1901_1920.npz",
        "facts": project / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: {"path": str(path), "sha256": sha256(path)} for name, path in sources.items()}

    evidence, locked_skipped = extract_allowed(sources["supplement"])
    early_evidence = evidence[evidence["year"].isin(EARLY)].set_index("year")["chitral_mar_aug_pdsi"].reindex(EARLY).to_numpy(float)
    modern_evidence = evidence[evidence["year"].isin(MODERN)].set_index("year")["chitral_mar_aug_pdsi"].reindex(MODERN).to_numpy(float)

    allowed = scipy.io.loadmat(sources["allowed_proxy6"])
    years = np.asarray(allowed["years"]).reshape(-1).astype(int)
    observations = np.asarray(allowed["observations"], dtype=float)[:, :201]
    modern_rows = np.asarray([int(np.flatnonzero(years == year)[0]) for year in MODERN])
    modern_proxy = observations[modern_rows]
    early_proxy, proxy_locked_skipped = early.early_observations(sources["proxy6"])
    identity_rows = []
    for column in range(201):
        for period, evidence_values, proxy_values in (
            ("1901_1920", early_evidence, early_proxy[:, column]),
            ("1941_2000", modern_evidence, modern_proxy[:, column]),
        ):
            identity_rows.append({
                "period": period,
                "proxy_index": column + 1,
                "correlation": correlation(evidence_values, proxy_values),
                "affine_max_abs_residual": affine_residual(evidence_values, proxy_values),
            })
    identity = pd.DataFrame(identity_rows)

    curves = pd.read_csv(sources["candidate_curves"])
    comparison_rows = []
    for (series, region), group in curves.groupby(["series", "region"]):
        values = group.set_index("year")["value_mm_day"].reindex(EARLY).to_numpy(float)
        comparison_rows.append({
            "series": series,
            "region": region,
            "correlation_with_chitral_pdsi": correlation(values, early_evidence),
            "circular_shift_p": circular_shift_p(values, early_evidence),
            "trend_mm_day_per_century": trend(EARLY, values),
            "trend_sign_matches_chitral_pdsi": bool(np.sign(trend(EARLY, values)) == np.sign(trend(EARLY, early_evidence))),
        })
    with np.load(sources["candidate_field"]) as field:
        lat = np.asarray(field["lat"], dtype=float)
        lon = np.asarray(field["lon"], dtype=float)
        cube = np.asarray(field["increment_mm_day"], dtype=float)
    ilat = int(np.argmin(abs(lat - 35.36)))
    ilon = int(np.argmin(abs(((lon - 71.48 + 180) % 360) - 180)))
    local = cube[:, ilat, ilon]
    comparison_rows.append({
        "series": "annual_oof_candidate",
        "region": "nearest_2deg_cell",
        "correlation_with_chitral_pdsi": correlation(local, early_evidence),
        "circular_shift_p": circular_shift_p(local, early_evidence),
        "trend_mm_day_per_century": trend(EARLY, local),
        "trend_sign_matches_chitral_pdsi": bool(np.sign(trend(EARLY, local)) == np.sign(trend(EARLY, early_evidence))),
    })
    comparisons = pd.DataFrame(comparison_rows)

    output.mkdir(parents=True)
    evidence.to_csv(output / "chitral_pdsi_allowed_windows.csv", index=False)
    identity.to_csv(output / "proxy6_sequence_identity.csv", index=False)
    comparisons.to_csv(output / "candidate_production_comparison.csv", index=False)

    fig, axis = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    axis.plot(EARLY, (early_evidence - early_evidence.mean()) / early_evidence.std(), color="#0072B2", linewidth=2.3, label="Chitral Mar-Aug PDSI")
    for series, color in (("annual_oof_candidate", "#009E73"), ("production_equal_six", "#D55E00")):
        values = curves[(curves["series"] == series) & (curves["region"] == "west_lt80")].sort_values("year")["value_mm_day"].to_numpy(float)
        axis.plot(EARLY, (values - values.mean()) / values.std(), color=color, linewidth=2.0, label=series.replace("_", " "))
    axis.axhline(0, color="#BBBBBB", linewidth=0.7)
    axis.set(xlabel="Year", ylabel="Standardized anomaly", title="Chitral independent evidence and western HMA precipitation")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    axis.legend(frameon=False)
    fig.savefig(output / "chitral_pdsi_vs_western_hma_1901_1920.png", dpi=220)
    plt.close(fig)

    top_early = identity[identity["period"] == "1901_1920"].iloc[identity[identity["period"] == "1901_1920"]["correlation"].abs().argmax()]
    top_modern = identity[identity["period"] == "1941_2000"].iloc[identity[identity["period"] == "1941_2000"]["correlation"].abs().argmax()]
    candidate_west = comparisons[(comparisons["series"] == "annual_oof_candidate") & (comparisons["region"] == "west_lt80")].iloc[0]
    production_west = comparisons[(comparisons["series"] == "production_equal_six") & (comparisons["region"] == "west_lt80")].iloc[0]
    exact_duplicate = bool(identity["affine_max_abs_residual"].min() <= 1e-10)
    supports_candidate = bool(
        candidate_west["trend_sign_matches_chitral_pdsi"]
        and candidate_west["correlation_with_chitral_pdsi"] > production_west["correlation_with_chitral_pdsi"]
    )
    after = {name: {"path": str(path), "sha256": sha256(path)} for name, path in sources.items()}
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "allowed_windows": [[1901, 1920], [1941, 2000]],
        "locked_1921_1939_values_read_or_exported": False,
        "supplement_locked_value_cells_skipped": int(locked_skipped),
        "proxy6_locked_value_rows_skipped": int(proxy_locked_skipped),
        "chitral_pdsi_trend_units_per_century_1901_1920": trend(EARLY, early_evidence),
        "candidate_west": candidate_west.to_dict(),
        "production_west": production_west.to_dict(),
        "top_proxy6_match_early": top_early.to_dict(),
        "top_proxy6_match_modern": top_modern.to_dict(),
        "exact_affine_duplicate_of_proxy6": exact_duplicate,
        "supports_candidate_over_production_in_west": supports_candidate,
        "decision": "retain_as_new_source_independent_western_evidence" if supports_candidate and not exact_duplicate else "do_not_promote_as_independent_evidence",
        "interpretation_boundary": "PAKI041 is source-new but spatially close to the existing Chitral proxy network; it validates western-HMA hydroclimate phase, not the full HMA field.",
        "all_declared_sources_unchanged": before == after,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Chitral PDSI Evidence Test\n\n"
        f"- Chitral 1901-1920 trend: `{summary['chitral_pdsi_trend_units_per_century_1901_1920']:.6f}` PDSI/century.\n"
        f"- Candidate west correlation: `{candidate_west['correlation_with_chitral_pdsi']:.4f}`; production west: `{production_west['correlation_with_chitral_pdsi']:.4f}`.\n"
        f"- Exact affine duplicate of proxy6: `{exact_duplicate}`.\n"
        f"- Decision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("declared-source-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
