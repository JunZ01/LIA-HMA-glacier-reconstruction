from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import run_temperature_candidate_b2_stage_c_real_proxy_v1 as b2_reader
import temperature_candidate_b2_stage_c_core_v1 as core
from preflight_temperature_candidate_c_intercept_synthetic_v1 import estimate_intercepts
from preflight_temperature_candidate_d2_resolution_v1 import make_blocks
from preflight_temperature_candidate_d_decadal_v1 import block_observations


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
STATIC = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2/candidate_b2_stage_c_static_input_v2.npz"
REAL = LOOP / "temp_candidate_d2_real_v1/candidate_d2_real_output_v1.npz"
OUTPUT = LOOP / "temp_candidate_d2_modern_offset_attribution_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def concentration(values: np.ndarray) -> tuple[float, float, float]:
    absolute = np.abs(values)
    total = float(absolute.sum())
    if total == 0.0:
        return 0.0, 0.0, 0.0
    shares = np.sort(absolute / total)[::-1]
    return float(shares[0]), float(shares[:10].sum()), float(1.0 / np.sum(shares**2))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)

    with np.load(STATIC, allow_pickle=False) as package, np.load(REAL, allow_pickle=False) as real:
        years = np.asarray(package["years"], dtype=np.int32)
        if np.any((years >= 1921) & (years <= 1939)):
            raise RuntimeError("sealed-year-present")
        state_residual = np.asarray(package["pooled_state_residual_library"], dtype=float)
        operator_mean = np.asarray(package["proxy6_balanced_operator_forced_mean"], dtype=float)
        operator_residual = np.asarray(package["proxy6_pooled_operator_residual_library"], dtype=float)
        variance = np.asarray(package["proxy6_residual_variance"], dtype=float)
        weights = np.asarray(package["hma_weights"], dtype=float)
        weights /= weights.sum()
        observations, read_audit, proxy_source = b2_reader.read_allowed_proxy_observations(6, package, years)
        corrected, intercepts, counts = estimate_intercepts(years, observations, operator_mean, minimum_count=30)
        blocks = make_blocks(years, 20)
        block_obs, block_operator, _ = block_observations(corrected, operator_mean, blocks)
        stored_centers = np.asarray(real["block_centers"], dtype=float)
        stored_hma_increment = np.asarray(real["block_increment"], dtype=float) @ weights
        record_indices = np.asarray(package["proxy6_record_indices_0based"], dtype=int)
        locations = np.asarray(package["proxy6_locations"], dtype=float)
        coefficients = np.asarray(package["proxy6_psm_b0_b1_b2"], dtype=float)
        ptypes = np.asarray(package["proxy6_ptype"]).astype(str)
        methods = np.asarray(package["proxy6_psm_method"]).astype(str)

    block_rows: list[dict[str, object]] = []
    proxy_rows: list[dict[str, object]] = []
    contribution_matrix = np.full((len(blocks), len(record_indices)), np.nan)
    innovation_matrix = np.full_like(contribution_matrix, np.nan)
    closure_errors: list[float] = []

    for block_index, (start, end, _) in enumerate(blocks):
        center = (start + end) / 2.0
        if abs(center - stored_centers[block_index]) > 1e-12:
            raise RuntimeError("block-center-mismatch")
        active = np.isfinite(block_obs[block_index])
        gain = core.kalman_gain(state_residual, operator_residual[:, active], variance[active])
        innovation = block_obs[block_index, active] - block_operator[block_index, active]
        hma_gain = weights @ gain
        contribution = innovation * hma_gain
        contribution_matrix[block_index, active] = contribution
        innovation_matrix[block_index, active] = innovation / np.sqrt(variance[active])
        closure = abs(float(contribution.sum()) - float(stored_hma_increment[block_index]))
        closure_errors.append(closure)
        top1, top10, effective = concentration(contribution)
        block_rows.append(
            {
                "start_year": start,
                "end_year": end,
                "center_year": center,
                "active_proxy_count": int(active.sum()),
                "hma_increment_degC": float(contribution.sum()),
                "negative_innovation_fraction": float(np.mean(innovation < 0.0)),
                "negative_contribution_fraction": float(np.mean(contribution < 0.0)),
                "median_standardized_innovation": float(np.median(innovation / np.sqrt(variance[active]))),
                "top1_absolute_contribution_share": top1,
                "top10_absolute_contribution_share": top10,
                "absolute_contribution_effective_count": effective,
                "closure_error_degC": closure,
            }
        )

    modern = np.isin(stored_centers, [1949.5, 1969.5])
    modern_mean = np.nanmean(contribution_matrix[modern], axis=0)
    modern_innovation = np.nanmean(innovation_matrix[modern], axis=0)
    order = np.argsort(modern_mean)
    for rank, proxy_index in enumerate(order, start=1):
        proxy_rows.append(
            {
                "negative_rank": rank,
                "proxy6_column_0based": int(proxy_index),
                "archive_record_index_0based": int(record_indices[proxy_index]),
                "latitude": float(locations[proxy_index, 0]),
                "longitude": float(locations[proxy_index, 1]),
                "psm_b0": float(coefficients[proxy_index, 0]),
                "psm_b1_temperature": float(coefficients[proxy_index, 1]),
                "psm_b2_precipitation": float(coefficients[proxy_index, 2]),
                "ptype": ptypes[proxy_index],
                "psm_method": methods[proxy_index],
                "mean_1940_1979_standardized_innovation": float(modern_innovation[proxy_index]),
                "mean_1940_1979_hma_contribution_degC": float(modern_mean[proxy_index]),
            }
        )

    modern_values = modern_mean[np.isfinite(modern_mean)]
    top1, top10, effective = concentration(modern_values)
    negative_total = float(modern_values[modern_values < 0.0].sum())
    positive_total = float(modern_values[modern_values > 0.0].sum())
    sorted_negative = np.sort(modern_values[modern_values < 0.0])
    target = 0.8 * abs(negative_total)
    count_80 = int(np.searchsorted(np.cumsum(np.abs(sorted_negative)), target) + 1) if target > 0 else 0

    write_csv(OUTPUT / "block_attribution.csv", block_rows)
    write_csv(OUTPUT / "proxy_modern_contribution_ranking.csv", proxy_rows)

    centers = np.asarray([row["center_year"] for row in block_rows], dtype=float)
    increments = np.asarray([row["hma_increment_degC"] for row in block_rows], dtype=float)
    negative_fraction = np.asarray([row["negative_contribution_fraction"] for row in block_rows], dtype=float)
    top10_share = np.asarray([row["top10_absolute_contribution_share"] for row in block_rows], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.0), constrained_layout=True)
    axes[0].plot(centers, increments, color="#1F5A94", marker="o", markersize=3)
    axes[0].axhline(0.0, color="#555555", linewidth=0.8)
    axes[0].set_ylabel("HMA update (degC)")
    axes[0].set_title("Candidate D2 exact proxy contribution attribution")
    axes[1].plot(centers, negative_fraction, color="#B5473A", marker="o", markersize=3, label="negative contribution fraction")
    axes[1].plot(centers, top10_share, color="#2C7A5A", marker="s", markersize=3, label="top-10 absolute share")
    axes[1].set_ylabel("Fraction")
    axes[1].legend(frameon=False)
    top = np.argsort(np.abs(modern_mean))[::-1][:20]
    colors = np.where(modern_mean[top] < 0.0, "#B5473A", "#2C7A5A")
    axes[2].bar(np.arange(len(top)), modern_mean[top], color=colors)
    axes[2].set_xticks(np.arange(len(top)), [str(record_indices[index]) for index in top], rotation=60, ha="right")
    axes[2].set_xlabel("Archive record index (0-based)")
    axes[2].set_ylabel("Mean 1940-1979 contribution (degC)")
    fig.savefig(OUTPUT / "candidate_d2_modern_offset_attribution.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment": OUTPUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "modern_blocks": [1949.5, 1969.5],
        "modern_mean_hma_increment_degC": float(modern_values.sum()),
        "modern_negative_contribution_fraction": float(np.mean(modern_values < 0.0)),
        "modern_negative_contribution_total_degC": negative_total,
        "modern_positive_contribution_total_degC": positive_total,
        "modern_top1_absolute_share": top1,
        "modern_top10_absolute_share": top10,
        "modern_absolute_contribution_effective_count": effective,
        "negative_proxy_count_for_80pct_negative_total": count_80,
        "maximum_exact_closure_error_degC": float(max(closure_errors)),
        "proxy_read_audit": read_audit,
        "sealed_1921_1939_values_used": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Candidate D2 modern-offset attribution\n\n"
        f"Mean 1940-1979 HMA increment: `{summary['modern_mean_hma_increment_degC']:.6f} degC`.\n\n"
        f"Negative contribution fraction: `{summary['modern_negative_contribution_fraction']:.3f}`.\n\n"
        f"Top-10 absolute contribution share: `{summary['modern_top10_absolute_share']:.3f}`; effective count: `{summary['modern_absolute_contribution_effective_count']:.1f}`.\n\n"
        f"Exact closure maximum: `{summary['maximum_exact_closure_error_degC']:.3e} degC`.\n",
        encoding="utf-8",
    )
    sources = {"static": STATIC, "real_d2": REAL, "driver": DRIVER, "proxy6_archive": proxy_source}
    manifest = {name: {"path": str(path), "sha256": sha256(path)} for name, path in sources.items()}
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
