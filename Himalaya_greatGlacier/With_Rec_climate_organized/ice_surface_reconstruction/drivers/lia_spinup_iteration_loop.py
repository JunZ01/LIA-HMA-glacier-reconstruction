"""Consolidate existing first-10 spin-up trials into one deterministic loop.

The loop does not rerun cases. It reuses every ``spinup_batch_summary.csv``
under the first-10 result directories, deduplicates trials, computes the
normalized late-volume trend, and reports which temperature points need to be
added before the next run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_trial_directory(path: Path) -> pd.DataFrame:
    summary_path = path / "spinup_batch_summary.csv"
    timeseries_path = path / "spinup_batch_timeseries.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(summary_path)
    if frame.empty:
        return frame
    frame["source_dir"] = str(path)
    if timeseries_path.exists():
        series = pd.read_csv(timeseries_path)
        if not series.empty:
            series["lia_unique_id"] = series["lia_unique_id"].astype(int)
            series["delta_t_c"] = series["delta_t_c"].astype(float)
            keys = ["lia_unique_id", "delta_t_c", "prcp_mode"]
            final = (
                series.sort_values("year")
                .groupby(keys, as_index=False)
                .tail(1)[keys + ["volume_m3"]]
                .rename(columns={"volume_m3": "final_volume_m3"})
            )
            frame = frame.merge(final, on=keys, how="left")
    return frame


def collect_trials(results_root: Path) -> pd.DataFrame:
    """Read and deduplicate all reusable first-10 trial summaries."""
    frames = []
    for directory in sorted(results_root.glob("lia_first10_*")):
        if directory.is_dir():
            frame = _read_trial_directory(directory)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no-first10-spinup-summaries:{results_root}")
    frame = pd.concat(frames, ignore_index=True)
    for column in ("lia_unique_id",):
        frame[column] = frame[column].astype(int)
    for column in ("delta_t_c", "length_relative_error", "last_100yr_volume_trend_m3_per_year"):
        frame[column] = frame[column].astype(float)
    frame["prcp_mode"] = frame["prcp_mode"].astype(str)
    frame["final_volume_m3"] = frame["final_volume_m3"].astype(float)
    frame["volume_trend_fraction_per_year"] = (
        frame["last_100yr_volume_trend_m3_per_year"] / frame["final_volume_m3"]
    )
    frame["volume_trend_fraction_per_century"] = (
        100.0 * frame["volume_trend_fraction_per_year"]
    )
    # Threshold columns are assigned by ``summarize_loop`` so CLI values are
    # applied consistently instead of being hidden in collection code.
    frame = frame.sort_values(
        ["lia_unique_id", "prcp_mode", "delta_t_c", "source_dir"]
    )
    frame = frame.drop_duplicates(
        ["lia_unique_id", "prcp_mode", "delta_t_c"], keep="last"
    )
    return frame.reset_index(drop=True)


def _sign_bracket(rows: pd.DataFrame, column: str) -> tuple[dict, dict] | None:
    ordered = rows.sort_values("delta_t_c").to_dict("records")
    for left, right in zip(ordered[:-1], ordered[1:]):
        a = float(left[column])
        b = float(right[column])
        if a == 0.0 or a * b <= 0.0:
            return left, right
    return None


def summarize_loop(
    trials: pd.DataFrame,
    *,
    length_threshold: float = 0.10,
    volume_trend_threshold_per_century: float = 0.02,
) -> pd.DataFrame:
    rows = []
    for (unique_id, prcp_mode), group in trials.groupby(
        ["lia_unique_id", "prcp_mode"], sort=True
    ):
        group = group.copy()
        group["geometry_pass"] = (
            group["length_relative_error"].abs() <= length_threshold
        )
        group["volume_pass"] = (
            group["volume_trend_fraction_per_century"].abs()
            <= volume_trend_threshold_per_century
        )
        geometry = group[group["geometry_pass"]].copy()
        accepted = geometry[geometry["volume_pass"]]
        if not accepted.empty:
            chosen = accepted.assign(
                abs_volume_trend=accepted[
                    "volume_trend_fraction_per_century"
                ].abs(),
                abs_length_error=accepted["length_relative_error"].abs(),
            ).sort_values(["abs_volume_trend", "abs_length_error"]).iloc[0]
            status = "accepted"
            next_dt = np.nan
            bracket_low = np.nan
            bracket_high = np.nan
        else:
            bracket = _sign_bracket(
                geometry, "volume_trend_fraction_per_century"
            )
            if bracket is not None:
                bracket_low = float(bracket[0]["delta_t_c"])
                bracket_high = float(bracket[1]["delta_t_c"])
                next_dt = 0.5 * (bracket_low + bracket_high)
                status = "needs-volume-bisection"
            elif not geometry.empty and group["volume_pass"].any():
                group = group.sort_values("delta_t_c").copy()
                group["state_pattern"] = (
                    2 * group["geometry_pass"].astype(int)
                    + group["volume_pass"].astype(int)
                )
                existing = group["delta_t_c"].to_numpy(float)
                pairs = []
                ordered = list(group.iterrows())
                for (_, left), (_, right) in zip(ordered[:-1], ordered[1:]):
                    if left["state_pattern"] == right["state_pattern"]:
                        continue
                    midpoint = 0.5 * (
                        float(left["delta_t_c"]) + float(right["delta_t_c"])
                    )
                    if np.min(np.abs(existing - midpoint)) < 1e-10:
                        continue
                    pairs.append((
                        abs(float(right["delta_t_c"]) - float(left["delta_t_c"])),
                        left,
                        right,
                    ))
                if pairs:
                    _, left, right = min(pairs, key=lambda item: item[0])
                    bracket_low = min(
                        float(left["delta_t_c"]),
                        float(right["delta_t_c"]),
                    )
                    bracket_high = max(
                        float(left["delta_t_c"]),
                        float(right["delta_t_c"]),
                    )
                    next_dt = 0.5 * (bracket_low + bracket_high)
                    status = "needs-joint-boundary-bisection"
                else:
                    bracket_low = np.nan
                    bracket_high = np.nan
                    next_dt = np.nan
                    status = "no-volume-bracket"
            elif geometry.empty:
                bracket_low = np.nan
                bracket_high = np.nan
                next_dt = np.nan
                status = "no-length-match"
            else:
                nearest = geometry.iloc[
                    geometry["volume_trend_fraction_per_century"].abs().argmin()
                ]
                bracket_low = np.nan
                bracket_high = np.nan
                next_dt = np.nan
                status = "no-volume-bracket"
            chosen = geometry.iloc[
                geometry["volume_trend_fraction_per_century"].abs().argmin()
            ] if not geometry.empty else group.iloc[0]
        rows.append({
            "lia_unique_id": int(unique_id),
            "prcp_mode": str(prcp_mode),
            "status": status,
            "selected_delta_t_c": float(chosen["delta_t_c"]),
            "selected_length_error_fraction": float(chosen["length_relative_error"]),
            "selected_volume_trend_fraction_per_century": float(
                chosen["volume_trend_fraction_per_century"]
            ),
            "length_threshold_fraction": float(length_threshold),
            "volume_trend_threshold_fraction_per_century": float(
                volume_trend_threshold_per_century
            ),
            "bracket_low_delta_t_c": bracket_low,
            "bracket_high_delta_t_c": bracket_high,
            "next_delta_t_c": next_dt,
            "trial_count": int(len(group)),
            "geometry_pass_count": int(len(geometry)),
            "accepted_count": int(len(accepted)),
        })
    return pd.DataFrame(rows)


def _plot_system(group: pd.DataFrame, status: dict, output_path: Path, threshold: float) -> None:
    fig, ax1 = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    x = group["delta_t_c"].to_numpy(float)
    length = 100.0 * group["length_relative_error"].to_numpy(float)
    volume = group["volume_trend_fraction_per_century"].to_numpy(float) * 100.0
    geometry = group["length_relative_error"].abs().le(
        float(status["length_threshold_fraction"])
    ).to_numpy(bool)
    ax1.scatter(x[~geometry], length[~geometry], color="#9ca3af", label="length fail")
    ax1.scatter(x[geometry], length[geometry], color="#2563eb", label="length pass")
    ax1.axhline(10.0, color="#2563eb", linestyle="--", linewidth=0.8)
    ax1.axhline(-10.0, color="#2563eb", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Delta T (C)")
    ax1.set_ylabel("500-year length error (%)")
    ax1.grid(alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(x, volume, color="#dc2626", marker="o", linewidth=1.0, label="volume trend")
    ax2.axhline(100.0 * threshold, color="#dc2626", linestyle="--", linewidth=0.8)
    ax2.axhline(-100.0 * threshold, color="#dc2626", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Last 100-year volume trend (% per century)")
    chosen = float(status["selected_delta_t_c"])
    ax1.axvline(chosen, color="#111827", linewidth=1.0, alpha=0.65)
    ax1.set_title(
        f"LIA {int(status['lia_unique_id'])} | {status['status']} | selected Delta T={chosen:g} C"
    )
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=8)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    trials = collect_trials(args.results_root)
    trials = trials[trials["prcp_mode"].eq(args.prcp_mode)].copy()
    if trials.empty:
        raise ValueError(f"no-trials-for-prcp-mode:{args.prcp_mode}")
    status = summarize_loop(
        trials,
        length_threshold=args.length_threshold,
        volume_trend_threshold_per_century=args.volume_trend_threshold_per_century,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.output_dir / "iteration_candidates.csv", index=False)
    status.to_csv(args.output_dir / "iteration_status.csv", index=False)
    for _, row in status.iterrows():
        group = trials[
            trials["lia_unique_id"].eq(int(row["lia_unique_id"]))
            & trials["prcp_mode"].eq(row["prcp_mode"])
        ]
        _plot_system(
            group,
            row.to_dict(),
            args.output_dir / f"lia_{int(row['lia_unique_id'])}_delta_t_loop.png",
            args.volume_trend_threshold_per_century,
        )
    overview = {
        "status": "completed",
        "prcp_mode": args.prcp_mode,
        "length_threshold_fraction": args.length_threshold,
        "volume_trend_threshold_fraction_per_century": args.volume_trend_threshold_per_century,
        "trial_count": int(len(trials)),
        "system_count": int(len(status)),
        "accepted_count": int(status["status"].eq("accepted").sum()),
        "status_csv": "iteration_status.csv",
        "candidate_csv": "iteration_candidates.csv",
    }
    (args.output_dir / "iteration_summary.json").write_text(
        json.dumps(overview, indent=2), encoding="utf-8"
    )
    return overview


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prcp-mode", default="off")
    parser.add_argument("--length-threshold", type=float, default=0.10)
    parser.add_argument("--volume-trend-threshold-per-century", type=float, default=0.02)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
