"""Audit transferable 50-year temperature information in proxy9 records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 as round25


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
OUTPUT = RESULTS / "climate_reconstruction_temperature_proxy_low_frequency_information_audit_v1"
ARCHIVE = Path(r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all")
PROXY9 = ARCHIVE / "python_data/proxy_data_9.mat"
LEDGER = LOOP / (
    "round_110_unresolved_proxy_metadata_recovery_v4/"
    "proxy_evidence_master_ledger_metadata_recovered.csv"
)
CURVES = RESULTS / (
    "climate_reconstruction_temperature_hma_reference_anchored_low_frequency_v1/"
    "temperature_hma_reference_anchored_candidate_1000_2000.csv"
)
YEARS = np.arange(1500, 1900)
WINDOWS = [(start, start + 49) for start in range(1500, 1900, 50)]
FAMILIES = ["local_enkf", "lmr", "neukom", "phyda"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 5 or np.nanstd(left[valid]) == 0 or np.nanstd(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def standardize(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    result = np.full(values.shape, np.nan)
    if int(valid.sum()) < 2:
        return result
    spread = float(np.std(values[valid], ddof=1))
    if spread == 0:
        return result
    result[valid] = (values[valid] - float(np.mean(values[valid]))) / spread
    return result


def longitude_region(longitude: float) -> str:
    if longitude < 80:
        return "west_lt80"
    if longitude < 90:
        return "central_80_90"
    return "east_ge90"


def main() -> int:
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(OUTPUT / "summary.json")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "proxy9": PROXY9,
        "ledger": LEDGER,
        "reference_curves": CURVES,
        "facts": PROJECT / "LIA_PROJECT_FACTS.md",
        "contract": OUTPUT / "ROUND_CONTRACT.md",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: sha256(path) for name, path in paths.items()}

    ledger = pd.read_csv(LEDGER, low_memory=False)
    network = ledger[ledger["network"].eq("proxy9")].copy()
    network["network_record_1based"] = network["network_record_1based"].astype(int)
    network = network.sort_values("network_record_1based").reset_index(drop=True)
    if len(network) != 297 or not np.array_equal(
        network["network_record_1based"].to_numpy(), np.arange(1, 298)
    ):
        raise RuntimeError("Proxy9 ledger alignment failed")
    ready = network["operator_metadata_ready_v4"].astype(str).str.lower().eq("true")
    temperature = network["documented_network_role_v4"].eq("documented_temperature")
    selected = (ready & temperature).to_numpy()
    if int(selected.sum()) != 226:
        raise RuntimeError(f"Frozen temperature record count failed: {int(selected.sum())}")

    _, observations, locked_reads = round25.read_proxy_allowed(PROXY9, YEARS)
    if locked_reads != 0 or observations.shape != (297, 400):
        raise RuntimeError("Proxy read contract failed")
    selected_observations = observations[selected]
    selected_ledger = network.loc[selected].reset_index(drop=True)
    coefficients = selected_ledger["psm_temperature_coefficient"].to_numpy(float)
    if not np.all(np.isfinite(coefficients)) or np.any(coefficients == 0):
        raise RuntimeError("Temperature PSM orientation coefficient failed")
    oriented = selected_observations * np.sign(coefficients)[:, None]

    window_values = np.full((len(selected_ledger), len(WINDOWS)), np.nan)
    finite_counts = np.zeros_like(window_values, dtype=int)
    for window_index, (start, end) in enumerate(WINDOWS):
        chosen = (YEARS >= start) & (YEARS <= end)
        finite = np.isfinite(oriented[:, chosen])
        counts = finite.sum(axis=1)
        finite_counts[:, window_index] = counts
        valid_record = counts >= 35
        values = np.nansum(oriented[:, chosen], axis=1) / np.where(
            counts == 0, np.nan, counts
        )
        window_values[valid_record, window_index] = values[valid_record]
    stable = np.all(np.isfinite(window_values), axis=1)
    standardized = np.stack([standardize(values) for values in window_values])

    curves = pd.read_csv(CURVES)
    reference_windows: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        values = curves[f"reference_{family}_hma_degC"].to_numpy(float)
        curve_years = curves["year"].to_numpy(int)
        means = []
        for start, end in WINDOWS:
            chosen = (curve_years >= start) & (curve_years <= end)
            means.append(float(np.mean(values[chosen])))
        reference_windows[family] = standardize(np.asarray(means))

    record_rows: list[dict[str, object]] = []
    robust = np.zeros(len(selected_ledger), dtype=bool)
    correlations = np.full((len(selected_ledger), len(FAMILIES)), np.nan)
    for record_index, record in selected_ledger.iterrows():
        for family_index, family in enumerate(FAMILIES):
            correlations[record_index, family_index] = correlation(
                standardized[record_index], reference_windows[family]
            )
        median_correlation = float(np.nanmedian(correlations[record_index]))
        all_positive = bool(np.all(correlations[record_index] > 0))
        robust[record_index] = bool(stable[record_index] and all_positive and median_correlation >= 0.50)
        record_rows.append(
            {
                "record_id": record["record_id"],
                "network_record_1based": int(record["network_record_1based"]),
                "latitude": float(record["latitude"]),
                "longitude": float(record["longitude"]),
                "longitude_region": longitude_region(float(record["longitude"])),
                "archive_class": record["canonical_archive_class"],
                "source_family": record["source_family"],
                "psm_temperature_coefficient": float(record["psm_temperature_coefficient"]),
                "stable_all_8_windows": bool(stable[record_index]),
                "median_family_correlation": median_correlation,
                "all_four_family_correlations_positive": all_positive,
                "robust_low_frequency_record": bool(robust[record_index]),
                **{
                    f"correlation_{family}": correlations[record_index, family_index]
                    for family_index, family in enumerate(FAMILIES)
                },
                **{
                    f"finite_years_{start}_{end}": int(finite_counts[record_index, window_index])
                    for window_index, (start, end) in enumerate(WINDOWS)
                },
            }
        )
    record_table = pd.DataFrame(record_rows)
    record_table.to_csv(OUTPUT / "proxy_record_low_frequency_metrics.csv", index=False)

    stable_standardized = standardized[stable]
    if len(stable_standardized) == 0:
        raise RuntimeError("No stable low-frequency records")
    composite = np.nanmean(stable_standardized, axis=0)
    composite_correlations = {
        family: correlation(composite, values)
        for family, values in reference_windows.items()
    }
    loo_correlations: dict[str, list[float]] = {family: [] for family in FAMILIES}
    for remove_index in range(len(stable_standardized)):
        retained = np.delete(stable_standardized, remove_index, axis=0)
        current = np.nanmean(retained, axis=0)
        for family in FAMILIES:
            loo_correlations[family].append(
                correlation(current, reference_windows[family])
            )
    loo_medians = {
        family: float(np.nanmedian(values))
        for family, values in loo_correlations.items()
    }

    region_rows = []
    for region in ("west_lt80", "central_80_90", "east_ge90"):
        in_region = record_table["longitude_region"].eq(region).to_numpy()
        stable_region = stable & in_region
        region_composite = (
            np.nanmean(standardized[stable_region], axis=0)
            if int(stable_region.sum()) > 0
            else np.full(len(WINDOWS), np.nan)
        )
        region_rows.append(
            {
                "longitude_region": region,
                "selected_record_count": int(in_region.sum()),
                "stable_record_count": int(stable_region.sum()),
                "robust_record_count": int((robust & in_region).sum()),
                **{
                    f"composite_correlation_{family}": correlation(
                        region_composite, reference_windows[family]
                    )
                    for family in FAMILIES
                },
            }
        )
    region_table = pd.DataFrame(region_rows)
    region_table.to_csv(OUTPUT / "longitude_region_low_frequency_support.csv", index=False)

    composite_table = pd.DataFrame(
        {
            "start_year": [start for start, _ in WINDOWS],
            "end_year": [end for _, end in WINDOWS],
            "stable_proxy_composite_z": standardize(composite),
            **{
                f"reference_{family}_z": values
                for family, values in reference_windows.items()
            },
        }
    )
    composite_table.to_csv(OUTPUT / "network_composite_50year_curves.csv", index=False)

    gates = {
        "composite_correlation_at_least_0p50_every_family": bool(
            min(composite_correlations.values()) >= 0.50
        ),
        "at_least_25_robust_individual_records": bool(int(robust.sum()) >= 25),
        "at_least_5_robust_records_each_longitude_region": bool(
            region_table["robust_record_count"].min() >= 5
        ),
        "loo_median_composite_correlation_at_least_0p50_every_family": bool(
            min(loo_medians.values()) >= 0.50
        ),
    }
    supported = bool(all(gates.values()))

    order = np.argsort(np.nanmedian(correlations, axis=1))[::-1]
    shown = order[: min(80, len(order))]
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    centers = np.asarray([(start + end) / 2 for start, end in WINDOWS])
    axes[0].plot(centers, standardize(composite), color="#0072B2", marker="o", linewidth=2.4, label="Stable proxy composite")
    colors = {"local_enkf": "#009E73", "lmr": "#D55E00", "neukom": "#CC79A7", "phyda": "#E69F00"}
    for family in FAMILIES:
        axes[0].plot(centers, reference_windows[family], color=colors[family], marker=".", linewidth=1.4, label=family)
    axes[0].set(title="Proxy9 50-year composite versus reference families", ylabel="Standardized anomaly")
    axes[0].legend(frameon=False, ncol=5)
    image = axes[1].imshow(correlations[shown], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_xticks(np.arange(len(FAMILIES)), FAMILIES)
    axes[1].set_ylabel("Top records by median family correlation")
    axes[1].set_title("Individual record 50-year correlations")
    fig.colorbar(image, ax=axes[1], shrink=0.8, label="Correlation")
    axes[2].bar(region_table["longitude_region"], region_table["stable_record_count"], color="#777777", label="Stable")
    axes[2].bar(region_table["longitude_region"], region_table["robust_record_count"], color="#009E73", label="Robust")
    axes[2].axhline(5, color="#333333", linestyle="--", linewidth=1.0)
    axes[2].set(title="Low-frequency record support by longitude region", ylabel="Record count")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    figure_path = OUTPUT / "temperature_proxy_low_frequency_information_audit.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    summary = {
        "status": "complete",
        "selected_temperature_record_count": int(len(selected_ledger)),
        "stable_all_8_windows_record_count": int(stable.sum()),
        "robust_low_frequency_record_count": int(robust.sum()),
        "composite_correlations": composite_correlations,
        "leave_one_record_out_median_correlations": loo_medians,
        "robust_record_counts_by_region": {
            row.longitude_region: int(row.robust_record_count)
            for row in region_table.itertuples(index=False)
        },
        "archive_class_counts": {
            str(key): int(value)
            for key, value in selected_ledger["canonical_archive_class"].value_counts().items()
        },
        "gates": gates,
        "low_frequency_reconstruction_supported": supported,
        "locked_1901_1939_accessed": False,
        "decision": (
            "authorize_predeclared_low_frequency_proxy_candidate"
            if supported
            else "existing_proxy_network_lacks_frozen_low_frequency_support"
        ),
        "interpretation_limit": (
            "Reference families are mutually dependent to an unknown degree and proxy records are not independent. "
            "This audit tests directional information content, not a calibrated reconstruction."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Temperature Proxy Low-Frequency Information Audit\n\n"
        f"- Stable records: `{summary['stable_all_8_windows_record_count']}/226`.\n"
        f"- Robust records: `{summary['robust_low_frequency_record_count']}`.\n"
        f"- Composite correlations: `{summary['composite_correlations']}`.\n"
        f"- Robust records by region: `{summary['robust_record_counts_by_region']}`.\n"
        f"- All gates pass: `{str(supported).lower()}`.\n\n"
        f"Decision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    after = {name: sha256(path) for name, path in paths.items()}
    unchanged = before == after
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("Source file changed")
    print(json.dumps(summary, indent=2))
    print(figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
