"""Finalize the complete intermediate tables from the 20-year amplitude run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
INPUT = RESULTS / "temperature_fold_native_oenkf_amplitude_20year_cv_v1"
OUTPUT = RESULTS / "temperature_fold_native_oenkf_amplitude_20year_cv_v2"
BASELINE_MANIFEST = RESULTS / "temperature_fold_native_oenkf_increment_cv_v5_regem_R/source_hash_manifest.json"
PRIMARY_PRODUCTS = ("CRU", "GISTEMP", "HadCRUT5", "NOAA")
BLOCKS = ((1940, 1959), (1960, 1979), (1980, 2000))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    required = [
        INPUT / "amplitude_transfer_metrics.csv",
        INPUT / "regional_curves.csv",
        INPUT / "operator_closure.csv",
        INPUT / "active_proxy_counts.csv",
        INPUT / "heldout_oof_fields.npz",
        INPUT / "config.json",
        INPUT / "ROUND_CONTRACT.md",
        INPUT / "matlab_proxy6_r2024b.log",
        INPUT / "matlab_proxy9_r2024b.log",
        BASELINE_MANIFEST,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    metrics = pd.read_csv(INPUT / "amplitude_transfer_metrics.csv")
    curves = pd.read_csv(INPUT / "regional_curves.csv")
    closure = pd.read_csv(INPUT / "operator_closure.csv")
    expected_counts = {
        ("block", "low11"): 720,
        ("block", "annual"): 720,
        ("overall", "low11"): 240,
        ("overall", "high11"): 240,
    }
    actual_counts = metrics.groupby(["scope", "frequency"]).size().to_dict()
    if len(metrics) != 1920 or actual_counts != expected_counts:
        raise RuntimeError(f"intermediate-table-contract-failed:{len(metrics)}:{actual_counts}")
    if len(closure) != 2 or (closure.hxb500_replay_max_abs > 1e-9).any() or (closure.R_replay_max_abs > 1e-9).any():
        raise RuntimeError("operator-closure-failed")
    for network in (6, 9):
        log = (INPUT / f"matlab_proxy{network}_r2024b.log").read_text(encoding="utf-8")
        if "FOLD_NATIVE_PSM_20YEAR_V1_COMPLETE" not in log:
            raise RuntimeError(f"matlab-completion-marker-missing:{network}")

    baseline = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))["sources_before"]
    shared_keys = (
        "allowed_proxy6",
        "allowed_proxy9",
        "prior",
        "indices",
        "replay_proxy6",
        "replay_proxy9",
        "proxy_data6",
        "proxy_data9",
        "instrumental_netcdf",
        "hma_mask",
        "bearth",
        "gpcc",
        "nearest_latlon",
        "regrid2gcm",
    )
    shared_source_checks = {}
    for key in shared_keys:
        path = Path(baseline[key]["path"])
        current = sha256(path)
        expected = baseline[key]["sha256"]
        shared_source_checks[key] = {
            "path": str(path),
            "expected_sha256": expected,
            "current_sha256": current,
            "unchanged": current == expected,
        }
    sources_unchanged = all(item["unchanged"] for item in shared_source_checks.values())
    if not sources_unchanged:
        raise RuntimeError("shared-source-hash-closure-failed")

    primary = metrics[
        (metrics["scope"] == "overall")
        & (metrics["frequency"] == "low11")
        & (metrics["region"] == "hma")
        & (metrics["branch"] == "combined_proxy6_proxy9")
        & metrics["product"].isin(PRIMARY_PRODUCTS)
    ]
    hma_ar = float(primary.amplitude_ratio.median())
    hma_beta = float(primary.beta.median())
    hma_r = float(primary.correlation.median())
    hma_skill = float(primary.skill_vs_zero.median())

    block_primary = metrics[
        (metrics["scope"] == "block")
        & (metrics["frequency"] == "low11")
        & (metrics["region"] == "hma")
        & (metrics["branch"] == "combined_proxy6_proxy9")
        & metrics["product"].isin(PRIMARY_PRODUCTS)
    ]
    block_summary = block_primary.groupby(["fold", "period"]).agg(
        amplitude_ratio=("amplitude_ratio", "median"),
        beta=("beta", "median"),
        correlation=("correlation", "median"),
        skill_vs_zero=("skill_vs_zero", "median"),
    ).reset_index()
    passing_blocks = int(((block_summary.beta >= 0.67) & (block_summary.beta <= 1.33) & (block_summary.skill_vs_zero > 0)).sum())
    product_support = int(((primary.beta > 0) & (primary.skill_vs_zero > 0)).sum())

    regional = metrics[
        (metrics["scope"] == "overall")
        & (metrics["frequency"] == "low11")
        & (metrics["region"] != "hma")
        & (metrics["branch"] == "combined_proxy6_proxy9")
        & metrics["product"].isin(PRIMARY_PRODUCTS)
    ].groupby("region").agg(
        amplitude_ratio=("amplitude_ratio", "median"),
        beta=("beta", "median"),
        correlation=("correlation", "median"),
        skill_vs_zero=("skill_vs_zero", "median"),
    ).reset_index()
    positive_regions = int((regional.beta > 0).sum())
    regional_median_ar = float(regional.amplitude_ratio.median())

    network_beta = {}
    for network in (6, 9):
        rows = metrics[
            (metrics["scope"] == "overall")
            & (metrics["frequency"] == "low11")
            & (metrics["region"] == "hma")
            & (metrics["branch"] == f"proxy{network}")
            & metrics["product"].isin(PRIMARY_PRODUCTS)
        ]
        network_beta[network] = float(rows.beta.median())

    gates = {
        "hma_median_amplitude_ratio_0p8_to_1p2": bool(0.8 <= hma_ar <= 1.2),
        "hma_median_beta_0p8_to_1p2": bool(0.8 <= hma_beta <= 1.2),
        "hma_median_low11_correlation_at_least_0p50": bool(hma_r >= 0.50),
        "hma_median_low11_skill_at_least_0p10": bool(hma_skill >= 0.10),
        "at_least_two_of_three_blocks_pass": bool(passing_blocks >= 2),
        "at_least_three_of_four_primary_products_support": bool(product_support >= 3),
        "at_least_twelve_of_fifteen_regions_positive_beta": bool(positive_regions >= 12),
        "regional_median_amplitude_ratio_0p8_to_1p2": bool(0.8 <= regional_median_ar <= 1.2),
        "proxy6_proxy9_same_positive_direction": bool(network_beta[6] > 0 and network_beta[9] > 0),
    }
    passes = all(gates.values())
    if network_beta[6] * network_beta[9] <= 0:
        diagnosis = "proxy_network_dependency"
    elif hma_r >= 0.50 and (hma_ar < 0.8 or hma_beta < 0.8):
        diagnosis = "systematic_low_frequency_amplitude_attenuation"
    elif hma_r < 0.30 or hma_beta <= 0:
        diagnosis = "phase_or_psm_relation_failure"
    elif hma_r >= 0.50 and positive_regions < 12:
        diagnosis = "regional_spatial_covariance_failure"
    elif passes:
        diagnosis = "native_operator_transfers_low_frequency_amplitude"
    else:
        diagnosis = "mixed_or_product_dependent_failure"

    output_hashes = {path.name: sha256(path) for path in required[:-1]}
    OUTPUT.mkdir(parents=True)
    metrics.to_csv(OUTPUT / "amplitude_transfer_metrics.csv", index=False)
    curves.to_csv(OUTPUT / "regional_curves.csv", index=False)
    closure.to_csv(OUTPUT / "operator_closure.csv", index=False)
    block_summary.to_csv(OUTPUT / "hma_block_summary.csv", index=False)
    regional.to_csv(OUTPUT / "gtng15_summary.csv", index=False)

    summary = {
        "experiment_id": OUTPUT.name,
        "status": "complete",
        "decision": diagnosis,
        "passes_all_gates": passes,
        "hma_primary_product_medians": {
            "amplitude_ratio": hma_ar,
            "beta": hma_beta,
            "low11_correlation": hma_r,
            "low11_skill_vs_zero": hma_skill,
        },
        "passing_block_count": passing_blocks,
        "supporting_primary_product_count": product_support,
        "positive_gtng15_region_count": positive_regions,
        "regional_median_amplitude_ratio": regional_median_ar,
        "network_hma_median_beta": {str(key): value for key, value in network_beta.items()},
        "gates": gates,
        "operator_hxb_closure_max_abs": float(closure.hxb500_replay_max_abs.max()),
        "operator_R_closure_max_abs": float(closure.R_replay_max_abs.max()),
        "source_hashes_unchanged": sources_unchanged,
        "locked_1921_1939_accessed": False,
        "production_authorized": False,
        "oggm_authorized": False,
        "interpretation_boundary": "Conditional 1940-2000 mechanism pseudo-blind test; not independent final validation.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"shared_source_checks": shared_source_checks, "input_output_hashes": output_hashes, "source_hashes_unchanged": sources_unchanged}, indent=2),
        encoding="utf-8",
    )

    hma = curves[curves.region.eq("hma")]
    figure, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, constrained_layout=True)
    candidate = hma[(hma.kind == "reconstruction") & (hma.series == "combined_proxy6_proxy9")].sort_values("year")
    axes[0].plot(candidate.year, candidate.value_degC, color="#b2182b", lw=2.0, label="Native OEnKF proxy6/9")
    for product in PRIMARY_PRODUCTS:
        values = hma[(hma.kind == "observation") & (hma.series == product)].sort_values("year")
        axes[0].plot(values.year, values.value_degC, lw=1.0, alpha=0.65, label=product)
    for start, end in BLOCKS:
        selected = candidate.year.between(start, end)
        pred_low = candidate.loc[selected, "value_degC"].rolling(11, center=True, min_periods=11).mean()
        axes[1].plot(candidate.loc[selected, "year"], pred_low - pred_low.mean(), color="#b2182b", lw=2.2)
        for product in PRIMARY_PRODUCTS:
            values = hma[(hma.kind == "observation") & (hma.series == product) & hma.year.between(start, end)].sort_values("year")
            low = values.value_degC.rolling(11, center=True, min_periods=11).mean()
            axes[1].plot(values.year, low - low.mean(), lw=1.0, alpha=0.65)
    for axis in axes:
        for start, _ in BLOCKS[1:]:
            axis.axvline(start - 0.5, color="#888888", lw=0.7, ls="--")
        axis.axhline(0, color="#777777", lw=0.7)
        axis.grid(alpha=0.18)
        axis.set_ylabel("Temperature anomaly (degC)")
    axes[0].set_title("HMA three-fold native OEnKF amplitude transfer")
    axes[1].set_title("Within-block centered 11-year means")
    axes[1].set_xlabel("Year")
    axes[0].legend(frameon=False, ncol=5, fontsize=8)
    figure.savefig(OUTPUT / "temperature_native_oenkf_amplitude_20year_cv.png", dpi=220)
    plt.close(figure)

    gate_lines = "\n".join(f"- `{name}`: `{str(value).lower()}`" for name, value in gates.items())
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Native OEnKF 20-Year Amplitude Transfer\n\n"
        f"- Decision: `{diagnosis}`.\n"
        f"- HMA median amplitude ratio: `{hma_ar:.3f}`.\n"
        f"- HMA median beta: `{hma_beta:.3f}`.\n"
        f"- HMA median low-frequency correlation: `{hma_r:.3f}`.\n"
        f"- HMA median skill versus zero: `{hma_skill:.3%}`.\n"
        f"- Passing blocks: `{passing_blocks}/3`; supporting products: `{product_support}/4`; positive GTN-G regions: `{positive_regions}/15`.\n\n"
        "## Gates\n\n" + gate_lines + "\n\n"
        "This is a mechanism pseudo-blind result. It does not authorize production climate or OGGM.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
