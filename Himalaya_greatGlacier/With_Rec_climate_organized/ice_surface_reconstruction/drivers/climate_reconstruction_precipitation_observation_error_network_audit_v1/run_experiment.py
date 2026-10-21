"""Audit residual dependence discarded by the production diagonal R."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def effective_rank(eigenvalues: np.ndarray) -> float:
    positive = eigenvalues[eigenvalues > 0]
    probabilities = positive / positive.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def participation_ratio(eigenvalues: np.ndarray) -> float:
    positive = eigenvalues[eigenvalues > 0]
    return float(positive.sum() ** 2 / np.sum(positive**2))


def connected_components(adjacency: np.ndarray) -> list[list[int]]:
    seen = np.zeros(len(adjacency), dtype=bool)
    components = []
    for start in range(len(adjacency)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = np.where(adjacency[node])[0]
            for neighbor in neighbors:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(int(neighbor))
        components.append(component)
    return components


def main() -> int:
    root = project_root()
    attribution = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2"
    )
    metadata_path = attribution / "proxy6_metadata.csv"
    trend_path = attribution / "proxy_trend_contribution_1901_1920.csv"
    regem_path = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "enkf_reconstruction_source_recovery_audit_v2/"
        "regem_proxy6_exact_v1/regem_output.mat"
    )
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_observation_error_network_audit_v1"
    )
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    paths = {
        "metadata": metadata_path,
        "trend_attribution": trend_path,
        "regem_residuals": regem_path,
    }
    if missing := [str(path) for path in paths.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    output.mkdir(parents=True)
    hashes_before = {name: sha256(path) for name, path in paths.items()}

    metadata = pd.read_csv(metadata_path).sort_values("proxy_index")
    trends = pd.read_csv(trend_path).sort_values("proxy_index")
    residuals = np.asarray(
        scipy.io.loadmat(regem_path)["R0v_filled"], dtype=float
    )
    if residuals.shape != (202, 61) or not np.isfinite(residuals).all():
        raise RuntimeError(f"unexpected-residual-matrix:{residuals.shape}")

    correlation = np.corrcoef(residuals)
    covariance = np.cov(residuals, ddof=1)
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    off_diagonal = correlation[np.triu_indices_from(correlation, k=1)]

    proxy_table = metadata.merge(
        trends[
            [
                "proxy_index",
                "contribution_trend_mm_day_per_century",
                "innovation_trend_proxy_units_per_century",
                "mean_hma_gain_weight",
            ]
        ],
        on="proxy_index",
        validate="one_to_one",
    )

    threshold_rows = []
    component_rows = []
    for threshold in (0.3, 0.5, 0.7, 0.9):
        adjacency = np.abs(correlation) >= threshold
        np.fill_diagonal(adjacency, False)
        components = connected_components(adjacency)
        multi = [component for component in components if len(component) > 1]
        linked_nodes = sorted({node for component in multi for node in component})
        threshold_rows.append(
            {
                "absolute_correlation_threshold": threshold,
                "edge_count": int(np.triu(adjacency, k=1).sum()),
                "multi_proxy_component_count": len(multi),
                "largest_component_size": max((len(component) for component in multi), default=1),
                "linked_proxy_count": len(linked_nodes),
                "linked_proxy_fraction": len(linked_nodes) / 202.0,
            }
        )
        for component_index, component in enumerate(
            sorted(multi, key=len, reverse=True), start=1
        ):
            ids = [node + 1 for node in component]
            subset = proxy_table[proxy_table["proxy_index"].isin(ids)]
            component_rows.append(
                {
                    "absolute_correlation_threshold": threshold,
                    "component_index": component_index,
                    "component_size": len(component),
                    "proxy_indices": ",".join(str(value) for value in ids),
                    "site_count": int(
                        subset[["latitude", "longitude"]].drop_duplicates().shape[0]
                    ),
                    "signed_hma_trend_contribution_mm_day_per_century": float(
                        subset["contribution_trend_mm_day_per_century"].sum()
                    ),
                    "absolute_hma_trend_contribution_mm_day_per_century": float(
                        subset["contribution_trend_mm_day_per_century"].abs().sum()
                    ),
                    "median_psm_correlation": float(subset["psm_correlation"].median()),
                }
            )

    threshold_table = pd.DataFrame(threshold_rows)
    component_table = pd.DataFrame(component_rows)
    threshold_table.to_csv(output / "residual_correlation_graph_summary.csv", index=False)
    component_table.to_csv(output / "residual_correlation_components.csv", index=False)

    pair_rows = []
    for left in range(202):
        for right in range(left + 1, 202):
            value = float(correlation[left, right])
            if abs(value) < 0.5:
                continue
            left_row = proxy_table.iloc[left]
            right_row = proxy_table.iloc[right]
            pair_rows.append(
                {
                    "proxy_a": left + 1,
                    "proxy_b": right + 1,
                    "residual_correlation": value,
                    "same_exact_site": bool(
                        left_row["latitude"] == right_row["latitude"]
                        and left_row["longitude"] == right_row["longitude"]
                    ),
                    "latitude_a": float(left_row["latitude"]),
                    "longitude_a": float(left_row["longitude"]),
                    "latitude_b": float(right_row["latitude"]),
                    "longitude_b": float(right_row["longitude"]),
                    "trend_contribution_a": float(
                        left_row["contribution_trend_mm_day_per_century"]
                    ),
                    "trend_contribution_b": float(
                        right_row["contribution_trend_mm_day_per_century"]
                    ),
                }
            )
    pair_table = pd.DataFrame(pair_rows).sort_values(
        "residual_correlation", key=lambda values: values.abs(), ascending=False
    )
    pair_table.to_csv(output / "strong_residual_correlation_pairs.csv", index=False)

    cumulative = np.cumsum(np.clip(eigenvalues, 0, None)) / np.clip(
        eigenvalues, 0, None
    ).sum()
    modes_50 = int(np.searchsorted(cumulative, 0.5) + 1)
    modes_80 = int(np.searchsorted(cumulative, 0.8) + 1)
    modes_90 = int(np.searchsorted(cumulative, 0.9) + 1)
    modes_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    effective_rank_value = effective_rank(eigenvalues)
    participation_value = participation_ratio(eigenvalues)

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    image = axes[0].imshow(
        correlation,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
    )
    axes[0].set_xlabel("Proxy index")
    axes[0].set_ylabel("Proxy index")
    axes[0].set_title("RegEM residual correlation matrix")
    figure.colorbar(image, ax=axes[0], label="Residual correlation")
    axes[1].plot(np.arange(1, len(eigenvalues) + 1), cumulative, color="#4c78a8", lw=2)
    for level in (0.5, 0.8, 0.9, 0.95):
        axes[1].axhline(level, color="#777777", lw=0.7, ls="--")
    axes[1].set_xlim(1, 202)
    axes[1].set_ylim(0, 1.01)
    axes[1].set_xlabel("Number of residual-correlation modes")
    axes[1].set_ylabel("Cumulative eigenvalue fraction")
    axes[1].set_title("Residual-network dimensionality")
    axes[1].grid(alpha=0.2)
    figure.savefig(output / "figure_observation_error_network.png", dpi=220)
    plt.close(figure)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    abs_off = np.abs(off_diagonal)
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_observation_error_network_audit_v1",
        "date": "2026-08-13",
        "residual_matrix_shape": [202, 61],
        "production_R_rule": "diag(diag(covariance(residuals)))",
        "locked_final_period_accessed": False,
        "source_hashes_unchanged": hashes_before == hashes_after,
        "off_diagonal_absolute_correlation_median": float(np.median(abs_off)),
        "off_diagonal_absolute_correlation_q90": float(np.quantile(abs_off, 0.9)),
        "off_diagonal_absolute_correlation_q95": float(np.quantile(abs_off, 0.95)),
        "off_diagonal_absolute_correlation_maximum": float(abs_off.max()),
        "pair_fraction_abs_correlation_ge_0p3": float(np.mean(abs_off >= 0.3)),
        "pair_fraction_abs_correlation_ge_0p5": float(np.mean(abs_off >= 0.5)),
        "pair_fraction_abs_correlation_ge_0p7": float(np.mean(abs_off >= 0.7)),
        "correlation_matrix_numerical_rank": int(
            np.linalg.matrix_rank(correlation, tol=1e-10)
        ),
        "covariance_matrix_numerical_rank": int(
            np.linalg.matrix_rank(covariance, tol=1e-10)
        ),
        "correlation_effective_rank_entropy": effective_rank_value,
        "correlation_effective_rank_participation": participation_value,
        "modes_for_50_percent_variance": modes_50,
        "modes_for_80_percent_variance": modes_80,
        "modes_for_90_percent_variance": modes_90,
        "modes_for_95_percent_variance": modes_95,
        "largest_covariance_eigenvalue": float(covariance_eigenvalues[0]),
        "threshold_graphs": threshold_rows,
        "interpretation_boundary": (
            "Residual dependence indicates non-independent information, but this "
            "audit does not choose a full, block, shrinkage, or localized R."
        ),
    }
    (output / "source_manifest.json").write_text(
        json.dumps({"before": hashes_before, "after": hashes_after}, indent=2),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Precipitation Observation-Error Network Audit V1",
                "",
                f"- Median |residual correlation|: `{summary['off_diagonal_absolute_correlation_median']:.4f}`",
                f"- 95th percentile |correlation|: `{summary['off_diagonal_absolute_correlation_q95']:.4f}`",
                f"- Maximum |correlation|: `{summary['off_diagonal_absolute_correlation_maximum']:.4f}`",
                f"- Correlation numerical rank: `{summary['correlation_matrix_numerical_rank']}` of 202",
                f"- Entropy effective rank: `{effective_rank_value:.2f}`",
                f"- Modes for 90% variance: `{modes_90}`",
                f"- Source hashes unchanged: `{hashes_before == hashes_after}`",
                "",
                summary["interpretation_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
