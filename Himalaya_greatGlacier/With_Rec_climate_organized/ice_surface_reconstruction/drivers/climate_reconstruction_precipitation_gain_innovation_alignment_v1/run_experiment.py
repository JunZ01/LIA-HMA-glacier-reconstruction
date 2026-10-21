"""Test whether gain-innovation identity alignment explains the decline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 20260813
PERMUTATIONS = 100000


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def permutation_distribution(
    gains: np.ndarray,
    innovations: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.empty(PERMUTATIONS, dtype=float)
    for index in range(PERMUTATIONS):
        result[index] = float(gains @ innovations[rng.permutation(len(innovations))])
    return result


def grouped_permutation_distribution(
    gains: np.ndarray,
    innovations: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    group_indices = [np.where(groups == label)[0] for label in np.unique(groups)]
    result = np.empty(PERMUTATIONS, dtype=float)
    permuted = innovations.copy()
    for iteration in range(PERMUTATIONS):
        for indices in group_indices:
            permuted[indices] = innovations[indices][rng.permutation(len(indices))]
        result[iteration] = float(gains @ permuted)
    return result


def tail_metrics(observed: float, null: np.ndarray) -> dict[str, float]:
    return {
        "null_mean": float(null.mean()),
        "null_standard_deviation": float(null.std(ddof=0)),
        "null_q001": float(np.quantile(null, 0.001)),
        "null_q01": float(np.quantile(null, 0.01)),
        "null_q05": float(np.quantile(null, 0.05)),
        "null_median": float(np.median(null)),
        "null_q95": float(np.quantile(null, 0.95)),
        "lower_tail_probability": float((np.sum(null <= observed) + 1) / (len(null) + 1)),
        "two_sided_probability": float(
            min(1.0, 2.0 * (np.sum(null <= observed) + 1) / (len(null) + 1))
        ),
        "standardized_distance": float((observed - null.mean()) / null.std(ddof=0)),
    }


def main() -> int:
    root = project_root()
    source = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/"
        "proxy_trend_contribution_1901_1920.csv"
    )
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_gain_innovation_alignment_v1"
    )
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    if not source.exists():
        raise FileNotFoundError(source)
    output.mkdir(parents=True)
    source_hash_before = sha256(source)

    data = pd.read_csv(source).sort_values("proxy_index")
    if len(data) != 202:
        raise RuntimeError(f"unexpected-proxy-count:{len(data)}")
    gains = data["mean_hma_gain_weight"].to_numpy(dtype=float)
    innovations = data["innovation_trend_proxy_units_per_century"].to_numpy(dtype=float)
    observed = float(gains @ innovations)
    direct = float(data["contribution_trend_mm_day_per_century"].sum())
    if not np.isclose(observed, direct, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"source-factorization-failed:{observed}:{direct}")

    longitude_groups = pd.cut(
        data["longitude"],
        [0, 80, 90, 100, 200],
        labels=False,
        include_lowest=True,
    ).to_numpy(dtype=int)
    global_rng = np.random.default_rng(SEED)
    regional_rng = np.random.default_rng(SEED + 1)
    global_null = permutation_distribution(gains, innovations, global_rng)
    regional_null = grouped_permutation_distribution(
        gains, innovations, longitude_groups, regional_rng
    )

    global_metrics = tail_metrics(observed, global_null)
    regional_metrics = tail_metrics(observed, regional_null)
    null_table = pd.DataFrame(
        {
            "permutation": np.arange(1, PERMUTATIONS + 1),
            "global_identity_permutation": global_null,
            "within_longitude_band_permutation": regional_null,
        }
    )
    null_table.to_csv(output / "permutation_null_distributions.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    for axis, values, title, color in (
        (axes[0], global_null, "Global proxy-identity permutation", "#4c78a8"),
        (axes[1], regional_null, "Within-longitude-band permutation", "#59a14f"),
    ):
        axis.hist(values, bins=90, color=color, alpha=0.82, density=True)
        axis.axvline(observed, color="#b2182b", lw=2.2, label=f"observed {observed:.3f}")
        axis.axvline(float(values.mean()), color="#222222", lw=1.2, ls="--", label="null mean")
        axis.set_xlabel("HMA trend from gain x innovation (mm/day/century)")
        axis.set_ylabel("Density")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.savefig(output / "figure_gain_innovation_alignment.png", dpi=220)
    plt.close(figure)

    if global_metrics["lower_tail_probability"] < 0.01:
        global_diagnosis = "identity_specific_gain_innovation_alignment_is_material"
    else:
        global_diagnosis = "widespread_innovation_chronology_dominates_over_identity_alignment"
    if regional_metrics["lower_tail_probability"] < 0.01:
        regional_diagnosis = "alignment_remains_material_within_broad_regions"
    else:
        regional_diagnosis = "broad_regional_composition_explains_most_alignment"

    source_hash_after = sha256(source)
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_gain_innovation_alignment_v1",
        "date": "2026-08-13",
        "years": [1901, 1920],
        "locked_final_period_accessed": False,
        "source": str(source),
        "source_hash_before": source_hash_before,
        "source_hash_after": source_hash_after,
        "source_unchanged": source_hash_before == source_hash_after,
        "random_seed_global": SEED,
        "random_seed_regional": SEED + 1,
        "permutations_per_null": PERMUTATIONS,
        "observed_trend_mm_day_per_century": observed,
        "global_identity_permutation": global_metrics,
        "within_longitude_band_permutation": regional_metrics,
        "global_diagnosis": global_diagnosis,
        "regional_diagnosis": regional_diagnosis,
        "decision_boundary": (
            "This tests identity alignment under the frozen proxy6 experiment-1 gain. "
            "It does not validate proxy chronology or authorize covariance changes."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Precipitation Gain-Innovation Alignment V1",
                "",
                f"- Observed trend: `{observed:.6g} mm/day/century`",
                f"- Global-null lower-tail probability: `{global_metrics['lower_tail_probability']:.6g}`",
                f"- Regional-null lower-tail probability: `{regional_metrics['lower_tail_probability']:.6g}`",
                f"- Global diagnosis: `{global_diagnosis}`",
                f"- Regional diagnosis: `{regional_diagnosis}`",
                f"- Source unchanged: `{source_hash_before == source_hash_after}`",
                "",
                summary["decision_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
