"""Plot every saved B-window parameter group without running OGGM."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "layered_coordinate_search_v1"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "B_parameter_group_trajectory_plots_v1"


@dataclass(frozen=True)
class ExperimentBlock:
    name: str
    title: str
    source_dirs: tuple[str, ...]
    fixed_parameter: str | None


BLOCKS = (
    ExperimentBlock(
        "base_delta_t",
        "base delta_T profile",
        ("layer_B_adaptive_delta_profile_v1",),
        None,
    ),
    ExperimentBlock(
        "melt_delta_boundary",
        "melt_f - delta_T boundary",
        (
            "layer_B_melt_delta_boundary_refine_v1",
            "layer_B_melt_delta_boundary_refine_1044_round2_v1",
            "layer_B_melt_delta_boundary_refine_1545_round2_v1",
            "layer_B_melt_delta_boundary_refine_1545_round3_v1",
        ),
        "melt_f",
    ),
    ExperimentBlock(
        "prcp_fac",
        "prcp_fac block",
        ("layer_B_prcp_block_1545_v1",),
        "prcp_fac_multiplier",
    ),
    ExperimentBlock(
        "temp_melt",
        "temp_melt block",
        (
            "layer_B_temp_melt_block_1545_v1",
            "layer_B_temp_melt_block_1545_round2_v1",
        ),
        "temp_melt_c",
    ),
    ExperimentBlock(
        "glen_a",
        "glen_a block",
        ("layer_B_glen_block_1545_v1",),
        "glen_a_multiplier",
    ),
)


def _decode_number(token: str) -> float:
    return float(token.replace("m", "-").replace("p", "."))


def _glen_multiplier(chain_dir: Path) -> float:
    match = re.search(r"glen_x([^_]+)_dt_", chain_dir.name)
    if not match:
        raise ValueError(f"cannot parse glen multiplier from {chain_dir}")
    return _decode_number(match.group(1))


def _fixed_value(block: ExperimentBlock, summary: dict, chain_dir: Path) -> float | str:
    if block.fixed_parameter is None:
        return "all"
    if block.fixed_parameter == "glen_a_multiplier":
        return _glen_multiplier(chain_dir)
    return float(summary["parameters"][block.fixed_parameter])


def _load_chains(results_root: Path, block: ExperimentBlock) -> list[dict]:
    chains: list[dict] = []
    for source_name in block.source_dirs:
        source = results_root / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        for summary_path in sorted(source.rglob("summary.json")):
            chain_dir = summary_path.parent
            timeseries_path = chain_dir / "timeseries.csv"
            if not timeseries_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            # Non-converged and out-of-range chains reveal branch transitions.`n            # Exclude only files outside the B window.`n            if summary.get("layer") != "B":`n                continue
            series = pd.read_csv(timeseries_path)
            required = {"phase", "year", "length_m"}
            if not required.issubset(series.columns):
                raise ValueError(f"missing columns in {timeseries_path}")
            chains.append(
                {
                    "block": block,
                    "source_name": source_name,
                    "chain_dir": chain_dir,
                    "summary": summary,
                    "series": series,
                    "fixed_value": _fixed_value(block, summary, chain_dir),
                }
            )
    return chains


def _display_year(series: pd.DataFrame, forward_start: int) -> pd.Series:
    values = series["year"].astype(float).copy()
    spin = series["phase"].eq("spinup")
    if spin.any():
        spin_end = float(series.loc[spin, "year"].max())
        values.loc[spin] = forward_start - spin_end + values.loc[spin]
    return values


def _value_slug(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def _fixed_label(block: ExperimentBlock, value: float | str) -> str:
    if block.fixed_parameter is None:
        return "Stage-2 physical parameters fixed"
    if block.fixed_parameter == "melt_f":
        return f"same glacier / melt_f={float(value):.4f}"
    if block.fixed_parameter == "prcp_fac_multiplier":
        return f"same glacier / prcp_fac multiplier={float(value):.2f}"
    if block.fixed_parameter == "temp_melt_c":
        return f"same glacier / temp_melt={float(value):+.2f} degC"
    if block.fixed_parameter == "glen_a_multiplier":
        return f"same glacier / glen_a multiplier={float(value):.2f}"
    return str(value)


def _spread_labels(values: list[float], lower: float, upper: float) -> list[float]:
    if not values:
        return []
    span = max(upper - lower, 1.0)
    gap = span * 0.028
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    adjusted: list[tuple[int, float]] = []
    for index, value in indexed:
        position = max(value, adjusted[-1][1] + gap) if adjusted else value
        adjusted.append((index, position))
    overflow = adjusted[-1][1] - upper
    if overflow > 0:
        adjusted = [(index, position - overflow) for index, position in adjusted]
    result = [0.0] * len(values)
    for index, position in adjusted:
        result[index] = max(lower, min(upper, position))
    return result


def _plot_group(chains: list[dict], output_dir: Path) -> dict:
    block: ExperimentBlock = chains[0]["block"]
    summary0 = chains[0]["summary"]
    system_id = int(summary0["lia_unique_id"])
    forward_start = int(summary0["start_year"])
    fixed_value = chains[0]["fixed_value"]
    targets = summary0["targets"]
    lia_km = float(targets["lia_length_m"]) / 1000.0
    rgi_km = float(targets["rgi_length_m"]) / 1000.0
    chains = sorted(chains, key=lambda item: float(item["summary"]["parameters"]["delta_t_c"]))

    fig, axis = plt.subplots(figsize=(15, 8.5), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, max(len(chains), 2)))
    all_x: list[float] = []
    all_y: list[float] = [lia_km, rgi_km]
    endpoint_y: list[float] = []
    series_handles: list[Line2D] = []
    metric_lines: list[str] = []

    for color, item in zip(colors, chains):
        summary = item["summary"]
        series = item["series"]
        x = _display_year(series, forward_start)
        y = series["length_m"].astype(float) / 1000.0
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())
        dt = float(summary["parameters"]["delta_t_c"])
        converged = bool(summary["spinup_control"]["converged"])
        label = f"dT={dt:+.6g} K"
        for phase, linestyle, linewidth in (("spinup", "--", 1.5), ("forward", "-", 2.0)):
            mask = series["phase"].eq(phase)
            if mask.any():
                axis.plot(
                    x.loc[mask],
                    y.loc[mask],
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    marker="o" if phase == "forward" else None,
                    markersize=2.3,
                    alpha=0.95,
                )
        forward = series[series["phase"].eq("forward")]
        endpoint = float(forward.iloc[-1]["length_m"]) / 1000.0
        endpoint_y.append(endpoint)
        series_handles.append(Line2D([0], [0], color=color, lw=2.0, label=label))
        metrics = summary["metrics"]
        metric_lines.append(
            f"dT={dt:+.6g}: LIA {100 * float(metrics['e_lia']):.1f}%, "
            f"2000 {100 * float(metrics['e_2000']):.1f}%, "
            f"conv={'yes' if converged else 'no'}"
        )

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    y_pad = max((y_max - y_min) * 0.08, 0.4)
    axis.set_xlim(x_min, x_max + max((x_max - x_min) * 0.025, 20))
    axis.set_ylim(y_min - y_pad, y_max + y_pad)
    label_y = _spread_labels(endpoint_y, y_min - y_pad * 0.25, y_max + y_pad * 0.25)

    for color, value, position in zip(colors, endpoint_y, label_y):
        axis.scatter([2000], [value], color=color, s=30, zorder=5)
        axis.annotate(
            f"{value:.2f}",
            xy=(2000, value),
            xytext=(5, position - value),
            textcoords="offset points",
            color=color,
            va="center",
            fontsize=9,
        )

    axis.axhline(lia_km, color="black", linestyle="--", linewidth=1.5)
    axis.axhline(rgi_km, color="0.35", linestyle="-.", linewidth=1.3)
    axis.axvline(forward_start, color="0.35", linestyle=":", linewidth=1.3)
    axis.text(x_min + (x_max - x_min) * 0.01, lia_km + y_pad * 0.08, f"LIA target {lia_km:.2f} km", va="bottom")
    axis.text(x_min + (x_max - x_min) * 0.01, rgi_km - y_pad * 0.08, f"RGI target {rgi_km:.2f} km", va="top", color="0.25")
    axis.text(forward_start + (x_max - x_min) * 0.006, y_max + y_pad * 0.35, f"forward start {forward_start}", color="0.3")

    phase_legend = axis.legend(
        handles=(
            Line2D([0], [0], color="0.2", lw=1.7, linestyle="--", label=f"spin-up mapped to {forward_start}"),
            Line2D([0], [0], color="0.2", lw=2.0, linestyle="-", label=f"{forward_start}-2000 forward"),
        ),
        loc="upper left",
        frameon=False,
    )
    axis.add_artist(phase_legend)
    axis.legend(
        handles=series_handles,
        title=_fixed_label(block, fixed_value),
        loc="upper right",
        frameon=False,
        fontsize=8.5,
        title_fontsize=9,
    )
    axis.text(
        0.985,
        0.035,
        "\n".join(metric_lines),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
    )
    axis.set_title(f"System {system_id} | B window | {block.title}: length trajectories", fontsize=14)
    axis.set_xlabel("Display year")
    axis.set_ylabel("Length (km)")
    axis.grid(True, alpha=0.22)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{block.name}_system_{system_id}"
    if block.fixed_parameter is not None:
        filename += f"_{block.fixed_parameter}_{_value_slug(fixed_value)}"
    path = output_dir / f"{filename}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "experiment_group": block.name,
        "system_id": system_id,
        "fixed_parameter": block.fixed_parameter or "none",
        "fixed_value": fixed_value,
        "chain_count": len(chains),
        "delta_t_values": ";".join(
            f"{float(item['summary']['parameters']['delta_t_c']):.8g}" for item in chains
        ),
        "output_png": str(path.resolve()),
        "source_dirs": ";".join(sorted({item["source_name"] for item in chains})),
    }


def run(results_root: Path, output_dir: Path) -> pd.DataFrame:
    manifest: list[dict] = []
    for block in BLOCKS:
        chains = _load_chains(results_root, block)
        grouped: dict[tuple[int, float | str], list[dict]] = {}
        for item in chains:
            key = (int(item["summary"]["lia_unique_id"]), item["fixed_value"])
            grouped.setdefault(key, []).append(item)
        for key in sorted(grouped, key=lambda value: (value[0], str(value[1]))):
            manifest.append(_plot_group(grouped[key], output_dir))
    frame = pd.DataFrame(manifest)
    frame.to_csv(output_dir / "plot_manifest.csv", index=False)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = run(args.results_root, args.output_dir)
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
