"""Audit reference-family dependence of the consumed 1921-1939 blind test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_blind_reference_family_dependency_audit_v1"
BLIND = RESULTS / (
    "climate_reconstruction_temperature_low_frequency_blind_test_1921_1939_v1/"
    "temperature_low_frequency_blind_test_1921_1939_v1.nc"
)
REFERENCE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\"
    r"climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
ASSIGNMENT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/"
    "hma_grid_gtng15_assignment.csv"
)
SUPPORT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/"
    "gtng15_region_support_inventory.csv"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
ADDENDUM = PROJECT / "LIA_PROJECT_FACTS.temperature_blind_test_1921_1939_v1.addendum.md"

EXPECTED_BLIND_SHA256 = "427bd22b28b23340073415d5cc9a2dae8217c2c9f43fe31f1dda372ed77005ed"
EXPECTED_REFERENCE_SHA256 = "450850a6654e9080e9586df06d227205b564b2e91e5ba1f95b1610b723664d90"
YEARS = np.arange(1921, 1940, dtype=int)
LOW_CENTERS = np.arange(1926, 1935, dtype=int)
MEMBERS = {
    "HGAO": "tas_rec_ref_HGAO_c05d",
    "LMR": "tas_rec_ref_LMR_c05d",
    "Nk_AM": "tas_rec_ref_Nk_AM_c05d",
    "Nk_CCA": "tas_rec_ref_Nk_CCA_c05d",
    "Nk_CPS": "tas_rec_ref_Nk_CPS_c05d",
    "Nk_DA": "tas_rec_ref_Nk_DA_c05d",
    "Nk_GraphEM": "tas_rec_ref_Nk_GraphEM_c05d",
    "Nk_PCR": "tas_rec_ref_Nk_PCR_c05d",
    "O": "tas_rec_ref_O_c05d",
    "PHYDA": "tas_rec_ref_PHYDA_c05d",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_mean(values: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(values)
    count = np.isfinite(stack).sum(axis=0)
    return np.nansum(stack, axis=0) / np.where(count == 0, np.nan, count)


def read_reference_families() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(REFERENCE, "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int).reshape(-1)
        lat = np.asarray(handle["lat"], dtype=float).reshape(-1)
        lon = np.asarray(handle["lon"], dtype=float).reshape(-1)
        test_indices = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        baseline_indices = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in range(1961, 1991)])
        members = {}
        for name, variable_name in MEMBERS.items():
            dataset = handle[variable_name]
            test = np.asarray(dataset[test_indices], dtype=float)
            baseline = np.asarray(dataset[baseline_indices], dtype=float)
            fill = float(np.asarray(dataset.attrs.get("_FillValue", -9999.0)).squeeze())
            for values in (test, baseline):
                values[np.isclose(values, fill) | (values < -9000)] = np.nan
            members[name] = test - np.nanmean(baseline, axis=0, keepdims=True)
    families = {
        "local_enkf": finite_mean([members["HGAO"], members["O"]]),
        "lmr": members["LMR"],
        "neukom": finite_mean([members[name] for name in MEMBERS if name.startswith("Nk_")]),
        "phyda": members["PHYDA"],
    }
    return lat, lon, families


def domain_mean(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(values)
    numerator = np.nansum(values * weights[None], axis=(1, 2))
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def smooth11(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rolling(11, center=True, min_periods=11).mean().to_numpy(float)


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    return float(np.sqrt(np.mean((left[valid] - right[valid]) ** 2)))


def slope(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.polyfit(YEARS[valid], values[valid], 1)[0] * 100.0)


def main() -> int:
    inputs = {
        "blind": BLIND,
        "reference": REFERENCE,
        "assignment": ASSIGNMENT,
        "support": SUPPORT,
        "facts": FACTS,
        "addendum": ADDENDUM,
        "driver": Path(__file__).resolve(),
    }
    for path in inputs.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256(BLIND) != EXPECTED_BLIND_SHA256:
        raise RuntimeError("blind-candidate-hash-mismatch")
    if sha256(REFERENCE) != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("reference-source-hash-mismatch")
    before = {
        name: {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
        }
        for name, path in inputs.items()
    }

    OUTPUT.mkdir(parents=True)
    (OUTPUT / "ROUND_CONTRACT.md").write_text(
        "# Consumed-Blind Reference-Family Dependency Audit\n\n"
        "The original and corrected 1921-1939 fields are frozen. No correction, blend weight, threshold, or candidate is changed. The four reference families are local-EnKF, LMR, Neukom, and PHYDA. Metrics use the same 1926-1934 centers of an 11-year running mean as the frozen blind test. Four leave-one-family central references are also scored. Results diagnose reference dependence only and cannot reverse the frozen rejection or restore blind status.\n",
        encoding="utf-8",
    )

    with Dataset(BLIND) as dataset:
        blind_years = np.asarray(dataset.variables["year"][:], dtype=int)
        blind_lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        blind_lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        hma_mask = np.asarray(dataset.variables["hma_mask"][:], dtype=bool)
        common = np.asarray(dataset.variables["common_support_mask"][:], dtype=bool)
        stored_center = np.asarray(dataset.variables["tas_reference_family_balanced"][:], dtype=float)
        original = np.asarray(dataset.variables["tas_project_six_member_original"][:], dtype=float)
        correction = np.asarray(dataset.variables["tas_low_frequency_correction"][:], dtype=float)
        candidate = np.asarray(dataset.variables["tas_project_six_member_corrected"][:], dtype=float)
    if not np.array_equal(blind_years, YEARS):
        raise RuntimeError("blind-year-contract-failed")

    lat, lon, families = read_reference_families()
    if not np.array_equal(lat, blind_lat) or not np.array_equal(lon, blind_lon):
        raise RuntimeError("reference-grid-mismatch")
    center = finite_mean(list(families.values()))
    center_closure = float(np.nanmax(np.abs(center - stored_center)))
    candidate_closure = float(np.nanmax(np.abs(candidate - (original + correction))))
    if center_closure > 2e-6 or candidate_closure > 2e-6:
        raise RuntimeError(f"field-closure-failed:{center_closure}:{candidate_closure}")

    references = dict(families)
    references["four_family_center"] = center
    for omitted in families:
        references[f"loo_without_{omitted}"] = finite_mean(
            [values for name, values in families.items() if name != omitted]
        )

    assignment = pd.read_csv(ASSIGNMENT)
    support = pd.read_csv(SUPPORT).sort_values("o2region")
    domains: dict[str, tuple[str, np.ndarray]] = {"hma": ("High Mountain Asia", hma_mask & common)}
    for region in support.itertuples(index=False):
        mask = np.zeros_like(hma_mask)
        cells = assignment[assignment.o2region.eq(region.o2region)]
        mask[cells.grid_row.to_numpy(int), cells.grid_col.to_numpy(int)] = True
        domains[str(region.o2region)] = (str(region.full_name), mask & common)

    centers = np.isin(YEARS, LOW_CENTERS)
    metric_rows = []
    direction_rows = []
    curve_rows = []
    for domain, (domain_name, mask) in domains.items():
        original_series = domain_mean(original, mask, lat)
        correction_series = domain_mean(correction, mask, lat)
        candidate_series = domain_mean(candidate, mask, lat)
        original_low = smooth11(original_series)
        correction_low = smooth11(correction_series)
        candidate_low = smooth11(candidate_series)
        family_series = {name: domain_mean(values, mask, lat) for name, values in families.items()}
        family_low = {name: smooth11(values) for name, values in family_series.items()}
        for reference_name, reference_field in references.items():
            reference_series = domain_mean(reference_field, mask, lat)
            reference_low = smooth11(reference_series)
            original_rmse = rmse(original_low[centers], reference_low[centers])
            candidate_rmse = rmse(candidate_low[centers], reference_low[centers])
            metric_rows.append(
                {
                    "domain": domain,
                    "domain_name": domain_name,
                    "reference": reference_name,
                    "reference_kind": (
                        "family" if reference_name in families else
                        "four_family_center" if reference_name == "four_family_center" else
                        "leave_one_family_out"
                    ),
                    "original_low11_rmse_degC": original_rmse,
                    "candidate_low11_rmse_degC": candidate_rmse,
                    "improvement_fraction": 1.0 - candidate_rmse / original_rmse,
                    "original_minus_reference_low11_mean_degC": float(np.mean(original_low[centers] - reference_low[centers])),
                    "candidate_minus_reference_low11_mean_degC": float(np.mean(candidate_low[centers] - reference_low[centers])),
                    "reference_trend_degC_century": slope(reference_series),
                    "original_trend_degC_century": slope(original_series),
                    "candidate_trend_degC_century": slope(candidate_series),
                }
            )
        for family, reference_low in family_low.items():
            desired = reference_low[centers] - original_low[centers]
            applied = correction_low[centers]
            direction_rows.append(
                {
                    "domain": domain,
                    "domain_name": domain_name,
                    "family": family,
                    "desired_correction_mean_degC": float(np.mean(desired)),
                    "applied_correction_mean_degC": float(np.mean(applied)),
                    "desired_applied_same_sign": bool(np.mean(desired) * np.mean(applied) > 0),
                    "applied_to_desired_magnitude_ratio": float(abs(np.mean(applied)) / abs(np.mean(desired))) if abs(np.mean(desired)) > 0 else np.nan,
                }
            )
        for index, year in enumerate(YEARS):
            row = {
                "domain": domain,
                "domain_name": domain_name,
                "year": int(year),
                "original_degC": float(original_series[index]),
                "candidate_degC": float(candidate_series[index]),
                "correction_degC": float(correction_series[index]),
                "original_low11_degC": float(original_low[index]),
                "candidate_low11_degC": float(candidate_low[index]),
            }
            for family, values in family_series.items():
                row[f"{family}_degC"] = float(values[index])
                row[f"{family}_low11_degC"] = float(family_low[family][index])
            curve_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    directions = pd.DataFrame(direction_rows)
    curves = pd.DataFrame(curve_rows)
    metrics.to_csv(OUTPUT / "reference_family_domain_metrics.csv", index=False)
    directions.to_csv(OUTPUT / "reference_family_correction_direction.csv", index=False)
    curves.to_csv(OUTPUT / "reference_family_curves_1921_1939.csv", index=False)

    hma_family = metrics[(metrics.domain == "hma") & (metrics.reference_kind == "family")]
    hma_loo = metrics[(metrics.domain == "hma") & (metrics.reference_kind == "leave_one_family_out")]
    focus = directions[directions.domain.isin(["13-09", "15-03"])]
    family_improving_count = int((hma_family.improvement_fraction > 0).sum())
    loo_improving_count = int((hma_loo.improvement_fraction > 0).sum())
    focus_desired_negative_count = int((focus.desired_correction_mean_degC < 0).sum())
    focus_applied_wrong_sign_count = int((~focus.desired_applied_same_sign).sum())
    family_sensitive = family_improving_count < 4 or loo_improving_count < 4
    if not family_sensitive and focus_desired_negative_count == len(focus):
        decision = "common_hma_under_correction_with_two_robust_regional_sign_errors"
    elif not family_sensitive:
        decision = "common_hma_under_correction_but_focus_region_reference_directions_differ"
    else:
        decision = "blind_result_is_reference_family_sensitive"

    summary = {
        "status": "complete",
        "decision": decision,
        "reference_center_field_closure_max_degC": center_closure,
        "candidate_formula_closure_max_degC": candidate_closure,
        "hma_family_improving_count": family_improving_count,
        "hma_family_improvement_range": [float(hma_family.improvement_fraction.min()), float(hma_family.improvement_fraction.max())],
        "hma_family_improvement_median": float(hma_family.improvement_fraction.median()),
        "hma_loo_improving_count": loo_improving_count,
        "hma_loo_improvement_range": [float(hma_loo.improvement_fraction.min()), float(hma_loo.improvement_fraction.max())],
        "focus_region_family_comparison_count": int(len(focus)),
        "focus_region_desired_negative_count": focus_desired_negative_count,
        "focus_region_applied_wrong_sign_count": focus_applied_wrong_sign_count,
        "frozen_four_family_center_improvement": float(metrics[(metrics.domain == "hma") & (metrics.reference == "four_family_center")].iloc[0].improvement_fraction),
        "frozen_30pct_gate_changed": False,
        "candidate_changed": False,
        "blind_status_restored": False,
        "production_authorized": False,
        "oggm_authorized": False,
        "interpretation_boundary": "Post-reveal attribution only; no tuning or acceptance decision can use these values.",
    }

    after = {
        name: {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
        }
        for name, path in inputs.items()
    }
    unchanged = all(before[name] == after[name] for name in before)
    summary["source_states_unchanged"] = unchanged
    if not unchanged:
        raise RuntimeError("source-state-changed")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_states_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )

    plot_domains = ["hma", "13-09", "15-03"]
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    colors = {"local_enkf": "#0072B2", "lmr": "#009E73", "neukom": "#CC79A7", "phyda": "#E69F00"}
    for axis, domain in zip(axes, plot_domains, strict=True):
        rows = curves[curves.domain.eq(domain)].sort_values("year")
        axis.plot(rows.year, rows.original_low11_degC, color="#666666", lw=1.8, ls="--", label="Original")
        axis.plot(rows.year, rows.candidate_low11_degC, color="#b2182b", lw=2.4, label="Mixed candidate")
        for family, color in colors.items():
            axis.plot(rows.year, rows[f"{family}_low11_degC"], color=color, lw=1.4, label=family)
        axis.axhline(0, color="#888888", lw=0.7)
        axis.grid(alpha=0.18)
        axis.set_title(f"{domain}: {rows.domain_name.iloc[0]}")
        axis.set_ylabel("11-year mean (degC)")
    axes[-1].set_xlabel("Year")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    figure.savefig(OUTPUT / "temperature_blind_reference_family_dependency.png", dpi=220)
    plt.close(figure)

    family_lines = "\n".join(
        f"- {row.reference}: `{row.improvement_fraction:.1%}` improvement."
        for row in hma_family.itertuples(index=False)
    )
    focus_lines = "\n".join(
        f"- {row.domain} / {row.family}: desired `{row.desired_correction_mean_degC:+.3f} degC`, applied `{row.applied_correction_mean_degC:+.3f} degC`."
        for row in focus.itertuples(index=False)
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Blind Reference-Family Dependency Audit\n\n"
        f"Decision: `{decision}`.\n\n"
        "## HMA family results\n\n" + family_lines + "\n\n"
        f"All-family positive count: `{family_improving_count}/4`; leave-one-family positive count: `{loo_improving_count}/4`.\n\n"
        "## Focus-region correction direction\n\n" + focus_lines + "\n\n"
        "This is post-reveal attribution only. The candidate, frozen 30% gate, rejection, and production status are unchanged.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
