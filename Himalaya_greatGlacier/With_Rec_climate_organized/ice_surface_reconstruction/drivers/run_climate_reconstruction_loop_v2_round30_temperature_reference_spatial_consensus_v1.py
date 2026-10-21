"""Audit LIA grid-cell consensus of the approved four-family temperature reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
FAMILIES = ("local_enkf", "lmr", "neukom", "phyda")
LIA_YEARS = np.arange(1400, 1900)
BASELINE_YEARS = np.arange(1961, 1991)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def clean(dataset: h5py.Dataset, selection: slice) -> np.ndarray:
    values = np.asarray(dataset[selection], dtype=float)
    fill = float(np.asarray(dataset.attrs.get("_FillValue", -9999.0)).squeeze())
    values[np.isclose(values, fill) | (values < -9000)] = np.nan
    return values


def member_anomaly(dataset: h5py.Dataset) -> np.ndarray:
    lia = clean(dataset, slice(400, 900))
    baseline = clean(dataset, slice(961, 991))
    with np.errstate(invalid="ignore"):
        center = np.nanmean(baseline, axis=0)
    return lia - center[None]


def finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays)
    count = np.isfinite(stack).sum(axis=0)
    return np.nansum(stack, axis=0) / np.where(count == 0, np.nan, count)


def weighted_fraction(condition: np.ndarray, valid: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.sum(weights[valid]))
    return float(np.sum(weights[valid & condition]) / denominator) if denominator else np.nan


def weighted_series(values: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(values)
    numerator = np.nansum(values * weights[None], axis=(1, 2))
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def pairwise_spatial_correlations(fields: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = []
    for left in range(fields.shape[0]):
        for right in range(left + 1, fields.shape[0]):
            selected = valid & np.isfinite(fields[left]) & np.isfinite(fields[right])
            if selected.sum() < 10:
                values.append(np.nan)
            else:
                values.append(float(np.corrcoef(fields[left][selected], fields[right][selected])[0, 1]))
    return np.asarray(values, dtype=float)


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_30_temperature_reference_spatial_consensus_v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    paths = {
        "public_reference_source": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "hma_mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "round22_windows": loop / "round_22_temperature_reference_preinstrumental_consensus_v1/temperature_50year_family_consensus.csv",
        "round21_reference": loop / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: state(path) for name, path in paths.items()}

    contract = """# Round 30 Contract: Temperature Reference Spatial Consensus

## Question

Does the Round 19-22 approved four-family reference support an interim HMA temperature field in the LIA, or only a regional 50-year curve?

## Frozen Data and Definition

- The original ten-member public-reference NetCDF remains read-only.
- Families are unchanged: local EnKF is the HGAO/O mean, LMR is one family, Neukom is the six-method mean, and PHYDA is one family.
- Each member is recentered independently to its 1961-1990 grid-cell mean.
- Only 1400-1899 and 1961-1990 are read. The locked 1921-1939 period is never accessed.
- The HMA mask and cosine-latitude area weights are unchanged from Rounds 19-22.

## Spatial Metrics and Frozen Gates

- For each ten nonoverlapping 50-year LIA windows, calculate HMA area fractions with at least three cold families, signal amplitude at least as large as family spread, and a cold mean after every leave-one-family-out omission.
- Also calculate the median of six pairwise family spatial-pattern correlations.
- Field promotion requires at least 8/10 windows to reach 70% majority-cold area, 50% amplitude-resolved area, 70% all-LOO-cold area, and median pairwise spatial correlation at least 0.30.
- All four gates must pass. Regional consensus cannot substitute for failed spatial consensus.

## Exclusions

- No proxy fitting, EnKF, localization, OGGM, production reconstruction, or 1921-1939 access.
"""
    (output / "ROUND_CONTRACT.md").write_text(contract, encoding="utf-8")

    mask = np.load(paths["hma_mask"]).astype(bool)
    with h5py.File(paths["public_reference_source"], "r") as handle:
        years = np.asarray(handle["time"], dtype=int).ravel()
        latitudes = np.asarray(handle["lat"], dtype=float).ravel()
        longitudes = np.asarray(handle["lon"], dtype=float).ravel()
        if not np.array_equal(years, np.arange(1000, 2001)):
            raise RuntimeError("source-year-contract-failed")
        if mask.shape != (len(latitudes), len(longitudes)):
            raise RuntimeError("hma-mask-shape-failed")
        local_enkf = finite_mean([
            member_anomaly(handle[MEMBERS["HGAO"]]),
            member_anomaly(handle[MEMBERS["O"]]),
        ])
        lmr = member_anomaly(handle[MEMBERS["LMR"]])
        neukom_sum = np.zeros_like(lmr)
        neukom_count = np.zeros_like(lmr, dtype=np.uint8)
        for name in MEMBERS:
            if not name.startswith("Nk_"):
                continue
            values = member_anomaly(handle[MEMBERS[name]])
            finite = np.isfinite(values)
            neukom_sum[finite] += values[finite]
            neukom_count[finite] += 1
        neukom = neukom_sum / np.where(neukom_count == 0, np.nan, neukom_count)
        phyda = member_anomaly(handle[MEMBERS["PHYDA"]])
    families = {
        "local_enkf": local_enkf,
        "lmr": lmr,
        "neukom": neukom,
        "phyda": phyda,
    }

    round22 = pd.read_csv(paths["round22_windows"])
    round22 = round22[round22.is_lia_window_1400_1899.astype(bool)].set_index("start_year")
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    window_rows: list[dict[str, object]] = []
    majority_count = np.zeros(mask.shape, dtype=np.uint8)
    resolved_count = np.zeros(mask.shape, dtype=np.uint8)
    loo_count = np.zeros(mask.shape, dtype=np.uint8)
    regional_closure = 0.0
    for start in range(1400, 1900, 50):
        selection = (LIA_YEARS >= start) & (LIA_YEARS <= start + 49)
        family_fields = np.stack([np.nanmean(families[name][selection], axis=0) for name in FAMILIES])
        valid = mask & np.all(np.isfinite(family_fields), axis=0)
        central = np.mean(family_fields, axis=0)
        spread = np.std(family_fields, axis=0, ddof=1)
        cold_count = np.sum(family_fields < 0, axis=0)
        majority_cold = cold_count >= 3
        resolved = np.abs(central) >= spread
        loo_fields = np.stack([
            np.mean(np.delete(family_fields, omitted, axis=0), axis=0)
            for omitted in range(len(FAMILIES))
        ])
        all_loo_cold = np.all(loo_fields < 0, axis=0)
        correlations = pairwise_spatial_correlations(family_fields, valid)
        majority_count[valid & majority_cold] += 1
        resolved_count[valid & resolved] += 1
        loo_count[valid & all_loo_cold] += 1
        annual_regional = {
            name: float(np.mean(weighted_series(families[name][selection], mask, latitudes)))
            for name in FAMILIES
        }
        for name in FAMILIES:
            regional_closure = max(
                regional_closure,
                abs(annual_regional[name] - float(round22.loc[start, f"{name}_mean_degC"])),
            )
        window_rows.append({
            "start_year": start,
            "end_year": start + 49,
            "valid_hma_cell_count": int(valid.sum()),
            "majority_cold_area_fraction": weighted_fraction(majority_cold, valid, weights),
            "all_four_cold_area_fraction": weighted_fraction(cold_count == 4, valid, weights),
            "central_cold_area_fraction": weighted_fraction(central < 0, valid, weights),
            "amplitude_resolved_area_fraction": weighted_fraction(resolved, valid, weights),
            "majority_cold_and_resolved_area_fraction": weighted_fraction(majority_cold & resolved, valid, weights),
            "all_leave_one_family_out_cold_area_fraction": weighted_fraction(all_loo_cold, valid, weights),
            "median_pairwise_family_spatial_correlation": float(np.nanmedian(correlations)),
            "min_pairwise_family_spatial_correlation": float(np.nanmin(correlations)),
            **{f"{name}_regional_mean_degC": value for name, value in annual_regional.items()},
        })
    windows = pd.DataFrame(window_rows)
    windows.to_csv(output / "temperature_lia_spatial_consensus_windows.csv", index=False)
    np.savez_compressed(
        output / "temperature_lia_spatial_consensus_frequency.npz",
        lat=latitudes,
        lon=longitudes,
        hma_mask=mask,
        majority_cold_window_fraction=majority_count.astype(float) / 10,
        amplitude_resolved_window_fraction=resolved_count.astype(float) / 10,
        all_loo_cold_window_fraction=loo_count.astype(float) / 10,
    )

    gate_counts = {
        "majority_cold_area_at_least_70pct_window_count": int(np.sum(windows.majority_cold_area_fraction >= 0.70)),
        "amplitude_resolved_area_at_least_50pct_window_count": int(np.sum(windows.amplitude_resolved_area_fraction >= 0.50)),
        "all_loo_cold_area_at_least_70pct_window_count": int(np.sum(windows.all_leave_one_family_out_cold_area_fraction >= 0.70)),
        "median_spatial_correlation_at_least_0_30_window_count": int(np.sum(windows.median_pairwise_family_spatial_correlation >= 0.30)),
    }
    gates = {name.replace("_window_count", "_at_least_8_of_10"): count >= 8 for name, count in gate_counts.items()}
    field_supported = bool(all(gates.values()))
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "lia_window_count": 10,
        "round22_regional_family_max_abs_closure_degC": regional_closure,
        "median_majority_cold_area_fraction": float(windows.majority_cold_area_fraction.median()),
        "median_amplitude_resolved_area_fraction": float(windows.amplitude_resolved_area_fraction.median()),
        "median_all_leave_one_family_out_cold_area_fraction": float(windows.all_leave_one_family_out_cold_area_fraction.median()),
        "median_pairwise_family_spatial_correlation": float(windows.median_pairwise_family_spatial_correlation.median()),
        "gate_counts": gate_counts,
        "gates": gates,
        "interim_hma_temperature_field_supported": field_supported,
        "regional_50year_curve_remains_supported": True,
        "decision": "permit_interim_family_balanced_hma_temperature_field" if field_supported else "restrict_family_balanced_reference_to_regional_curve_and_uncertainty",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    maps = [
        (majority_count / 10, "Majority-cold window fraction"),
        (resolved_count / 10, "Amplitude-resolved window fraction"),
        (loo_count / 10, "All-LOO-cold window fraction"),
    ]
    extent = [float(longitudes.min()), float(longitudes.max()), float(latitudes.min()), float(latitudes.max())]
    for axis, (values, title) in zip(axes.ravel()[:3], maps):
        shown = np.where(mask, values, np.nan)
        image = axis.imshow(shown, origin="lower", extent=extent, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set(title=title, xlabel="Longitude", ylabel="Latitude")
        fig.colorbar(image, ax=axis, shrink=0.82)
    centers = (windows.start_year + windows.end_year) / 2
    axes[1, 1].plot(centers, windows.majority_cold_area_fraction, marker="o", label="3/4 cold area")
    axes[1, 1].plot(centers, windows.amplitude_resolved_area_fraction, marker="s", label="resolved area")
    axes[1, 1].plot(centers, windows.all_leave_one_family_out_cold_area_fraction, marker="^", label="all LOO cold area")
    axes[1, 1].axhline(0.7, color="#9b2226", linestyle="--", linewidth=1)
    axes[1, 1].axhline(0.5, color="#555555", linestyle=":", linewidth=1)
    axes[1, 1].set(xlabel="50-year window center", ylabel="HMA area fraction", ylim=(0, 1), title="Window-level spatial support")
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Four-family LIA temperature spatial consensus", fontsize=14)
    fig.savefig(output / "temperature_reference_spatial_consensus.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Temperature Reference Spatial Consensus",
        "",
        f"- Round 22 regional closure: `{regional_closure:.3e} degC`.",
        f"- Median majority-cold HMA area: `{summary['median_majority_cold_area_fraction']:.1%}`.",
        f"- Median amplitude-resolved HMA area: `{summary['median_amplitude_resolved_area_fraction']:.1%}`.",
        f"- Median all-LOO-cold HMA area: `{summary['median_all_leave_one_family_out_cold_area_fraction']:.1%}`.",
        f"- Median pairwise family spatial correlation: `{summary['median_pairwise_family_spatial_correlation']:.3f}`.",
        "",
        f"Interim HMA field supported: `{str(field_supported).lower()}`.",
        "",
        f"Decision: `{summary['decision']}`.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}
    source_unchanged = before == after
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": source_unchanged}, indent=2), encoding="utf-8")
    if not source_unchanged:
        raise RuntimeError("source-state-changed")
    if regional_closure > 1e-12:
        raise RuntimeError("round22-regional-closure-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
