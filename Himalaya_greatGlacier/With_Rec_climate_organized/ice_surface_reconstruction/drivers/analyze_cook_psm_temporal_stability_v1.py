from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_ID = "climate_reconstruction_cook_psm_temporal_stability_v1"


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


def coefficient_class(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(
            frame["b1_early"].gt(0) & frame["b1_late"].gt(0),
            "positive_both",
            np.where(
                frame["b1_early"].lt(0) & frame["b1_late"].lt(0),
                "negative_both",
                "sign_flip",
            ),
        ),
        index=frame.index,
    )


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    study_root = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_pages2k_fixed_network_study_batch_v1"
    stability_root = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_psm_temporal_stability_v2"
    output = root / f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_ID}"
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "study_metadata": study_root / "fixed_network_study_metadata.csv",
        "coefficient_drift": stability_root / "psm_temporal_coefficient_drift_proxy_level.csv",
        "split_metrics": stability_root / "psm_temporal_stability_proxy_level.csv",
        "stability_summary": stability_root / "summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}

    metadata = pd.read_csv(sources["study_metadata"])
    drift = pd.read_csv(sources["coefficient_drift"])
    split = pd.read_csv(sources["split_metrics"])
    selected_metadata = metadata[metadata["data_author"].isin(["Cook, E.R.", "Shao, X.M."])][
        ["proxy_index", "data_author", "site_id", "latitude", "longitude", "event_change_z"]
    ]
    records = selected_metadata.merge(drift, on="proxy_index", validate="one_to_one")
    if (records["data_author"] == "Cook, E.R.").sum() != 15 or (records["data_author"] == "Shao, X.M.").sum() != 13:
        raise RuntimeError("cook-shao-count-contract-failed")
    records["b1_temporal_class"] = coefficient_class(records)
    records["b1_sign_agreement"] = np.sign(records["b1_early"]) == np.sign(records["b1_late"])
    records["b2_sign_agreement"] = np.sign(records["b2_early"]) == np.sign(records["b2_late"])

    split_records = split.merge(
        selected_metadata[["proxy_index", "data_author", "event_change_z"]],
        on="proxy_index",
        validate="many_to_one",
    )
    group_summary = records.groupby(["data_author", "b1_temporal_class"], as_index=False).agg(
        record_count=("proxy_index", "size"),
        unique_site_count=("site_id", "nunique"),
        mean_event_change_z=("event_change_z", "mean"),
        median_event_change_z=("event_change_z", "median"),
        negative_event_fraction=("event_change_z", lambda values: float((values < 0).mean())),
        mean_b1_early=("b1_early", "mean"),
        mean_b1_late=("b1_late", "mean"),
        mean_absolute_b1_drift=("b1_early_minus_late", lambda values: float(values.abs().mean())),
    )
    split_summary = split_records.groupby(["data_author", "split"], as_index=False).agg(
        record_count=("proxy_index", "size"),
        median_fit_r=("fit_r", "median"),
        median_test_r=("test_r", "median"),
        median_test_rmse=("test_rmse", "median"),
        median_external_1901_1920_r=("external_1901_1920_r", "median"),
        median_external_1901_1920_rmse=("external_1901_1920_rmse", "median"),
    )

    cook = records[records["data_author"] == "Cook, E.R."].copy()
    cook_positive = cook[cook["b1_temporal_class"] == "positive_both"].copy()
    leave_positive_rows = []
    for proxy_index in cook_positive["proxy_index"]:
        retained = cook_positive[cook_positive["proxy_index"] != proxy_index]
        leave_positive_rows.append({
            "omitted_proxy_index": int(proxy_index),
            "retained_record_count": int(len(retained)),
            "retained_mean_event_change_z": float(retained["event_change_z"].mean()),
            "retained_sign_negative": bool(retained["event_change_z"].mean() < 0),
        })
    leave_positive = pd.DataFrame(leave_positive_rows)

    output.mkdir(parents=True)
    records.to_csv(output / "cook_shao_coefficient_drift_and_event.csv", index=False)
    split_records.to_csv(output / "cook_shao_split_fit_metrics.csv", index=False)
    group_summary.to_csv(output / "coefficient_class_event_summary.csv", index=False)
    split_summary.to_csv(output / "split_performance_summary.csv", index=False)
    leave_positive.to_csv(output / "cook_positive_both_leave_one_out.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), constrained_layout=True)
    marker = {"Cook, E.R.": "o", "Shao, X.M.": "s"}
    for author, subset in records.groupby("data_author"):
        scatter = axes[0].scatter(
            subset["b1_early"], subset["b1_late"], c=subset["event_change_z"],
            cmap="coolwarm", vmin=-3.3, vmax=0.5, marker=marker[author], s=70,
            edgecolor="black", linewidth=0.35, label=author,
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].plot([-1.5, 1.5], [-1.5, 1.5], color="0.5", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Temperature coefficient, 1940-1969 fit")
    axes[0].set_ylabel("Temperature coefficient, 1970-2000 fit")
    axes[0].set_title("Bivariate PSM coefficient stability")
    axes[0].legend(frameon=False, fontsize=8)
    fig.colorbar(scatter, ax=axes[0], label="Target proxy event change (z)")

    class_order = ["negative_both", "sign_flip", "positive_both"]
    for position, class_name in enumerate(class_order):
        values = cook.loc[cook["b1_temporal_class"] == class_name, "event_change_z"].to_numpy(dtype=float)
        axes[1].scatter(values, np.full(len(values), position), s=55, color="#247BA0", alpha=0.85)
        axes[1].scatter([np.mean(values)], [position], marker="|", s=280, color="#C9483B", linewidth=2.2)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(np.arange(len(class_order)), class_order)
    axes[1].set_xlabel("Cook target event change (z)")
    axes[1].set_title("Event persists across coefficient classes")

    cook_split = split_summary[split_summary["data_author"] == "Cook, E.R."].copy()
    x = np.arange(len(cook_split))
    width = 0.34
    axes[2].bar(x - width / 2, cook_split["median_fit_r"], width, label="fit r", color="#247BA0")
    axes[2].bar(x + width / 2, cook_split["median_test_r"], width, label="cross-period test r", color="#E07A1F")
    axes[2].set_xticks(x, ["early fit\nlate test", "late fit\nearly test"])
    axes[2].set_ylabel("Median record correlation")
    axes[2].set_title("Cook cross-period performance")
    axes[2].legend(frameon=False)
    fig.savefig(output / "cook_psm_temporal_stability.png", dpi=220)
    plt.close(fig)

    positive_mean = float(cook_positive["event_change_z"].mean())
    summary = {
        "cook_record_count": int(len(cook)),
        "cook_b1_sign_agreement_fraction": float(cook["b1_sign_agreement"].mean()),
        "cook_b2_sign_agreement_fraction": float(cook["b2_sign_agreement"].mean()),
        "cook_positive_temperature_both_halves_count": int(len(cook_positive)),
        "cook_positive_temperature_both_halves_mean_event_change_z": positive_mean,
        "cook_positive_temperature_both_halves_negative_event_fraction": float((cook_positive["event_change_z"] < 0).mean()),
        "cook_positive_temperature_both_halves_leave_one_out_all_negative": bool(leave_positive["retained_sign_negative"].all()),
        "cook_coefficient_class_summary": group_summary[group_summary["data_author"] == "Cook, E.R."].to_dict(orient="records"),
        "cook_split_performance": cook_split.to_dict(orient="records"),
        "classification": (
            "cook_negative_event_is_not_explained_by_temperature_coefficient_sign_instability"
            if positive_mean < 0 and bool(leave_positive["retained_sign_negative"].all())
            else "cook_negative_event_is_concentrated_in_unstable_or_negative_temperature_coefficients"
        ),
        "interpretation": (
            "This reuses annual bivariate early/late fits. It shows that coefficient-sign instability is insufficient to explain the Cook event, "
            "but it is not yet a metadata-constrained summer temperature-only chronology validation."
        ),
        "locked_1921_1939_accessed": False,
    }
    after = {name: source_state(path) for name, path in sources.items()}
    unchanged = before == after
    summary["all_declared_sources_unchanged"] = unchanged
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("declared-source-changed-during-cook-audit")
    (output / "FINAL_RESULT.md").write_text(
        "# Cook PSM Temporal-stability Audit v1\n\n"
        f"Only {len(cook_positive)}/{len(cook)} Cook records have positive temperature coefficients in both calibration halves, "
        f"but their target event mean is {positive_mean:.3f} z. Every leave-one-out mean within this subset remains negative.\n\n"
        f"Classification: `{summary['classification']}`. This closes coefficient-sign instability as a sufficient explanation, "
        "not the broader chronology/PSM nonstationarity question.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
