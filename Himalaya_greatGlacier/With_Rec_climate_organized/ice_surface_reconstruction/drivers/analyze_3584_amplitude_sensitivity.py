"""Analyze length-amplitude sensitivity of saved system-3584 chains.

This is a read-only post-processing diagnostic. It never launches OGGM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GLEN_REFERENCE = 1.045220095930216e-23
SELECTED_FRAGMENT = "B_3584_glen_shape_refine_v1/glen_x0p75_dt_0p7"


def _parameter_block(path: Path) -> tuple[str, str]:
    value = path.as_posix()
    if "B_3584_glen_shape_refine_v1" in value:
        return "glen_a", "glen_a_multiplier"
    if "B_3584_temp_melt_shape_v1" in value:
        return "temp_melt", "temp_melt_c"
    if "B_3584_prcp_shape_v1" in value:
        return "prcp_fac", "prcp_fac_multiplier"
    if "B_curve_local_refine_v1" in value:
        return "melt_f_delta_t", "melt_f"
    return "other", "unknown"


def _relative_change(start: float, end: float) -> float:
    if not np.isfinite(start) or start <= 0 or not np.isfinite(end):
        return np.nan
    return (end - start) / start


def _nearest_row(frame: pd.DataFrame, year: int) -> pd.Series:
    return frame.iloc[(frame["year"] - year).abs().argmin()]


def _score_chain(summary_path: Path) -> dict:
    source_dir = summary_path.parent.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["lia_unique_id"]) != 3584:
        raise ValueError("not-system-3584")

    series = pd.read_csv(source_dir / "timeseries.csv")
    for column in ("year", "length_m", "volume_m3", "area_m2"):
        series[column] = pd.to_numeric(series[column], errors="coerce")
    spin = series[series["phase"].astype(str).eq("spinup")].sort_values("year")
    forward = series[series["phase"].astype(str).eq("forward")].sort_values("year")
    if spin.empty or forward.empty:
        raise ValueError(f"missing-phases:{source_dir}")

    targets = summary["targets"]
    lia_length = float(targets["lia_length_m"])
    rgi_length = float(targets["rgi_length_m"])
    target_amplitude = abs(lia_length - rgi_length)

    peak = forward.loc[forward["length_m"].idxmax()]
    minimum = forward.loc[forward["length_m"].idxmin()]
    start = forward.iloc[0]
    final = forward.iloc[-1]
    year_1900 = _nearest_row(forward, 1900)
    year_1950 = _nearest_row(forward, 1950)

    spin_end = spin.iloc[-1]
    spin_previous = _nearest_row(spin, int(spin_end["year"]) - 200)
    spin_stability = abs(_relative_change(
        float(spin_previous["volume_m3"]), float(spin_end["volume_m3"])
    ))

    peak_to_2000 = float(peak["length_m"] - final["length_m"])
    amplitude_ratio = peak_to_2000 / target_amplitude
    amplitude_relative_error = abs(peak_to_2000 - target_amplitude) / target_amplitude
    e_peak = abs(float(peak["length_m"]) - lia_length) / lia_length
    e_2000 = abs(float(final["length_m"]) - rgi_length) / rgi_length

    corridor_buffer = 0.20 * target_amplitude
    corridor_low = min(lia_length, rgi_length) - corridor_buffer
    corridor_high = max(lia_length, rgi_length) + corridor_buffer
    corridor_pass = bool(
        (forward["length_m"] >= corridor_low - 1e-6).all()
        and (forward["length_m"] <= corridor_high + 1e-6).all()
    )

    parameters = summary["parameters"]
    block, primary_parameter = _parameter_block(source_dir)
    parameter_values = {
        "melt_f": float(parameters["melt_f"]),
        "glen_a_multiplier": float(parameters["glen_a_pa_3_s_1"]) / GLEN_REFERENCE,
        "temp_melt_c": float(parameters["temp_melt_c"]),
        "prcp_fac_multiplier": float(parameters["prcp_fac_multiplier"]),
    }
    status_ok = str(summary.get("status", "unknown")) == "ok"
    reported_converged = bool(summary["spinup_control"]["converged"])
    core_pass = bool(
        status_ok
        and reported_converged
        and spin_stability <= 0.02
        and e_peak <= 0.10
        and e_2000 <= 0.10
    )

    return {
        "system_id": 3584,
        "source_dir": str(source_dir),
        "parameter_block": block,
        "primary_parameter": primary_parameter,
        "primary_value": parameter_values.get(primary_parameter, np.nan),
        "delta_t_c": float(parameters["delta_t_c"]),
        "melt_f": float(parameters["melt_f"]),
        "prcp_fac_multiplier": float(parameters["prcp_fac_multiplier"]),
        "glen_a_pa_3_s_1": float(parameters["glen_a_pa_3_s_1"]),
        "glen_a_multiplier": parameter_values["glen_a_multiplier"],
        "temp_melt_c": float(parameters["temp_melt_c"]),
        "status_ok": status_ok,
        "spinup_reported_converged": reported_converged,
        "spinup_actual_years": int(summary["spinup_control"]["actual_years"]),
        "spinup_volume_change_last_200": spin_stability,
        "lia_target_m": lia_length,
        "rgi_target_m": rgi_length,
        "target_amplitude_m": target_amplitude,
        "forward_start_length_m": float(start["length_m"]),
        "peak_year": int(round(float(peak["year"]))),
        "peak_length_m": float(peak["length_m"]),
        "length_2000_m": float(final["length_m"]),
        "minimum_year": int(round(float(minimum["year"]))),
        "minimum_length_m": float(minimum["length_m"]),
        "simulated_amplitude_peak_to_2000_m": peak_to_2000,
        "amplitude_ratio": amplitude_ratio,
        "amplitude_relative_error": amplitude_relative_error,
        "forward_length_range_m": float(peak["length_m"] - minimum["length_m"]),
        "e_peak_lia": e_peak,
        "e_2000_rgi": e_2000,
        "forward_volume_change_fraction": _relative_change(
            float(start["volume_m3"]), float(final["volume_m3"])
        ),
        "volume_change_1900_2000_fraction": _relative_change(
            float(year_1900["volume_m3"]), float(final["volume_m3"])
        ),
        "volume_change_1950_2000_fraction": _relative_change(
            float(year_1950["volume_m3"]), float(final["volume_m3"])
        ),
        "length_change_1900_2000_m": float(
            final["length_m"] - year_1900["length_m"]
        ),
        "core_historical_pass": core_pass,
        "auxiliary_corridor_pass": corridor_pass,
        "amplitude_within_20pct": bool(amplitude_relative_error <= 0.20),
        "selected_candidate": SELECTED_FRAGMENT in source_dir.as_posix(),
    }


def collect_metrics(roots: list[Path]) -> pd.DataFrame:
    paths: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for path in root.rglob("summary.json"):
            summary = json.loads(path.read_text(encoding="utf-8"))
            if int(summary.get("lia_unique_id", -1)) == 3584:
                paths[str(path.parent.resolve())] = path
    frame = pd.DataFrame(_score_chain(path) for path in paths.values())
    if len(frame) != 36:
        raise RuntimeError(f"expected 36 unique system-3584 chains, found {len(frame)}")
    if frame["source_dir"].duplicated().any():
        raise RuntimeError("duplicate source directories")
    return frame.sort_values(
        ["core_historical_pass", "amplitude_relative_error", "e_peak_lia", "e_2000_rgi"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def summarize_levels(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (block, parameter, value), group in frame.groupby(
        ["parameter_block", "primary_parameter", "primary_value"], dropna=False
    ):
        core = group[group["core_historical_pass"]]
        rows.append({
            "parameter_block": block,
            "primary_parameter": parameter,
            "primary_value": value,
            "chain_count": int(len(group)),
            "core_pass_count": int(group["core_historical_pass"].sum()),
            "auxiliary_corridor_pass_count": int(group["auxiliary_corridor_pass"].sum()),
            "median_amplitude_ratio_all": float(group["amplitude_ratio"].median()),
            "min_amplitude_ratio_all": float(group["amplitude_ratio"].min()),
            "max_amplitude_ratio_all": float(group["amplitude_ratio"].max()),
            "median_amplitude_ratio_core": (
                float(core["amplitude_ratio"].median()) if len(core) else np.nan
            ),
            "best_amplitude_relative_error_core": (
                float(core["amplitude_relative_error"].min()) if len(core) else np.nan
            ),
            "median_peak_error_core": (
                float(core["e_peak_lia"].median()) if len(core) else np.nan
            ),
            "median_2000_error_core": (
                float(core["e_2000_rgi"].median()) if len(core) else np.nan
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["parameter_block", "primary_value"], na_position="last"
    )


def summarize_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block, group in frame.groupby("parameter_block"):
        core = group[group["core_historical_pass"]]
        correlation = group[["primary_value", "amplitude_ratio"]].corr(
            method="spearman"
        ).iloc[0, 1]
        rows.append({
            "parameter_block": block,
            "chain_count": int(len(group)),
            "core_pass_count": int(group["core_historical_pass"].sum()),
            "auxiliary_corridor_pass_count": int(group["auxiliary_corridor_pass"].sum()),
            "amplitude_within_20pct_count": int(group["amplitude_within_20pct"].sum()),
            "core_amplitude_ratio_min": (
                float(core["amplitude_ratio"].min()) if len(core) else np.nan
            ),
            "core_amplitude_ratio_median": (
                float(core["amplitude_ratio"].median()) if len(core) else np.nan
            ),
            "core_amplitude_ratio_max": (
                float(core["amplitude_ratio"].max()) if len(core) else np.nan
            ),
            "marginal_spearman_primary_vs_amplitude": float(correlation),
        })
    return pd.DataFrame(rows).sort_values("parameter_block")


def _plot_parameter_blocks(frame: pd.DataFrame, output_path: Path) -> None:
    labels = {
        "melt_f_delta_t": ("melt_f", "melt_f"),
        "glen_a": ("glen_a multiplier", "glen_a_multiplier"),
        "temp_melt": ("temp_melt (degC)", "temp_melt_c"),
        "prcp_fac": ("prcp_fac multiplier", "prcp_fac_multiplier"),
    }
    colors = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    for axis, (block, (xlabel, column)) in zip(axes.flat, labels.items()):
        group = frame[frame["parameter_block"].eq(block)].copy()
        dt_min = float(frame["delta_t_c"].min())
        dt_span = max(float(frame["delta_t_c"].max()) - dt_min, 1e-9)
        for _, row in group.iterrows():
            color = colors((float(row["delta_t_c"]) - dt_min) / dt_span)
            marker = "*" if row["selected_candidate"] else (
                "o" if row["core_historical_pass"] else "x"
            )
            size = 150 if row["selected_candidate"] else 45
            axis.scatter(
                row[column], row["amplitude_ratio"], c=[color], marker=marker,
                s=size, linewidths=1.2, alpha=0.9,
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1.1)
        axis.axhspan(0.8, 1.2, color="0.7", alpha=0.15)
        axis.set_title(block.replace("_", " "))
        axis.set_xlabel(xlabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Simulated / target amplitude")
    axes[1, 0].set_ylabel("Simulated / target amplitude")
    fig.suptitle(
        "System 3584: saved-chain amplitude response by parameter block\n"
        "Circles: core historical pass; crosses: other chains; star: selected candidate"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_endpoint_tradeoff(frame: pd.DataFrame, output_path: Path) -> None:
    colors = {
        "melt_f_delta_t": "#1f77b4",
        "glen_a": "#2ca02c",
        "temp_melt": "#d62728",
        "prcp_fac": "#9467bd",
    }
    fig, axis = plt.subplots(figsize=(9, 7))
    for block, group in frame.groupby("parameter_block"):
        axis.scatter(
            100 * group["e_peak_lia"],
            100 * group["e_2000_rgi"],
            s=35 + 180 * np.clip(group["amplitude_ratio"], 0, 1.5),
            facecolors=[
                colors.get(block, "0.4") if passed else "none"
                for passed in group["core_historical_pass"]
            ],
            edgecolors=colors.get(block, "0.4"),
            linewidths=1.1,
            label=block.replace("_", " "),
            alpha=0.8,
        )
    selected = frame[frame["selected_candidate"]]
    axis.scatter(
        100 * selected["e_peak_lia"], 100 * selected["e_2000_rgi"],
        marker="*", s=260, color="black", label="selected candidate", zorder=5,
    )
    axis.axvline(10, color="0.3", linestyle="--", linewidth=1)
    axis.axhline(10, color="0.3", linestyle="--", linewidth=1)
    axis.set(
        xlabel="LIA peak error (%)",
        ylabel="Year-2000 RGI error (%)",
        title="System 3584: endpoint trade-off and length amplitude",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(roots: list[Path], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = collect_metrics(roots)
    levels = summarize_levels(frame)
    blocks = summarize_blocks(frame)
    frame.to_csv(output_dir / "amplitude_chain_metrics.csv", index=False)
    levels.to_csv(output_dir / "parameter_level_summary.csv", index=False)
    blocks.to_csv(output_dir / "parameter_block_summary.csv", index=False)
    _plot_parameter_blocks(frame, output_dir / "amplitude_by_parameter_block.png")
    _plot_endpoint_tradeoff(frame, output_dir / "endpoint_tradeoff_amplitude.png")

    core = frame[frame["core_historical_pass"]]
    selected = frame[frame["selected_candidate"]]
    if len(selected) != 1:
        raise RuntimeError(f"expected one selected candidate, found {len(selected)}")
    best_amplitude = core.loc[core["amplitude_relative_error"].idxmin()]
    summary = {
        "analysis": "system_3584_saved_chain_amplitude_sensitivity",
        "new_oggm_runs": 0,
        "chains_analyzed": int(len(frame)),
        "core_historical_passes": int(len(core)),
        "auxiliary_corridor_passes": int(frame["auxiliary_corridor_pass"].sum()),
        "target_amplitude_m": float(frame["target_amplitude_m"].iloc[0]),
        "core_amplitude_ratio_range": [
            float(core["amplitude_ratio"].min()),
            float(core["amplitude_ratio"].max()),
        ],
        "core_amplitude_ratio_median": float(core["amplitude_ratio"].median()),
        "core_amplitude_within_20pct_count": int(core["amplitude_within_20pct"].sum()),
        "best_existing_amplitude_match": {
            "source_dir": str(best_amplitude["source_dir"]),
            "amplitude_m": float(best_amplitude["simulated_amplitude_peak_to_2000_m"]),
            "amplitude_ratio": float(best_amplitude["amplitude_ratio"]),
            "peak_error_percent": 100 * float(best_amplitude["e_peak_lia"]),
            "rgi_2000_error_percent": 100 * float(best_amplitude["e_2000_rgi"]),
        },
        "selected_candidate": {
            "source_dir": str(selected.iloc[0]["source_dir"]),
            "amplitude_m": float(selected.iloc[0]["simulated_amplitude_peak_to_2000_m"]),
            "amplitude_ratio": float(selected.iloc[0]["amplitude_ratio"]),
            "peak_error_percent": 100 * float(selected.iloc[0]["e_peak_lia"]),
            "rgi_2000_error_percent": 100 * float(selected.iloc[0]["e_2000_rgi"]),
            "volume_change_1900_2000_percent": (
                100 * float(selected.iloc[0]["volume_change_1900_2000_fraction"])
            ),
            "length_change_1900_2000_m": float(
                selected.iloc[0]["length_change_1900_2000_m"]
            ),
        },
        "interpretation": (
            "Amplitude response remains far below the LIA-RGI target separation "
            "among all core historical passes. Parameter-block contrasts are "
            "diagnostic and confounded by paired delta_t recentering; they are not "
            "causal parameter estimates."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.roots, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
