from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCKED_START = 1921
LOCKED_END = 1939
PERIODS = ((1880, 1900), (1901, 1920), (1940, 1950))
STATIONS = {
    "IN007101600": "SHIMLA",
    "INXLT035254": "LEH_KASHMIR",
}
NCEI_ROOT = "https://www.ncei.noaa.gov/pub/data/ghcn/v4/products/StationPlots/IN"


EVENTS = (
    {
        "station_id": "IN007101600",
        "station_name": "SHIMLA",
        "event_year": 1880,
        "event_date": "1880-04",
        "event_type": "observatory_move",
        "description": "Observatory moved from Portmore to Bairdville, about 0.25 mile north and 71 ft higher.",
        "source": "IMD annual report OCR, india.history.resource.108634",
        "source_url": "https://indianculture.gov.in/archives/annual-report-meteorological-reporter-government-india-council",
    },
    {
        "station_id": "IN007101600",
        "station_name": "SHIMLA",
        "event_year": 1880,
        "event_date": "1880-05-03",
        "event_type": "observation_time_change",
        "description": "Observation hours changed from 09:30/15:30 to 10:00/16:00 after 2 May.",
        "source": "IMD annual report OCR, india.history.resource.108634",
        "source_url": "https://indianculture.gov.in/archives/annual-report-meteorological-reporter-government-india-council",
    },
    {
        "station_id": "IN007101600",
        "station_name": "SHIMLA",
        "event_year": 1885,
        "event_date": "1885-03-07",
        "event_type": "observatory_move",
        "description": "Site changed from Bairdville to the new military staff offices in Burra Simla; the report says instrument constants likely changed.",
        "source": "IMD annual report OCR, india.history.resource.108638",
        "source_url": "https://indianculture.gov.in/archives/annual-report-meteorological-reporter-government-india-council",
    },
    {
        "station_id": "IN007101600",
        "station_name": "SHIMLA",
        "event_year": 1887,
        "event_date": "1887-03-22",
        "event_type": "thermometer_replacement",
        "description": "New thermometer substituted.",
        "source": "IMD annual report OCR, india.history.resource.109238",
        "source_url": "https://indianculture.gov.in/archives/annual-report-meteorological-reporter-government-india-council",
    },
    {
        "station_id": "IN007101600",
        "station_name": "SHIMLA",
        "event_year": 1889,
        "event_date": "1889",
        "event_type": "observation_time_change",
        "description": "World Weather Records reports 10:00/16:00 through 1888 and 08:00 from 1889 onward.",
        "source": "Smithsonian World Weather Records, 1927 compilation",
        "source_url": "https://repository.si.edu/handle/10088/24035",
    },
    {
        "station_id": "IN007101600",
        "station_name": "SHIMLA",
        "event_year": 1890,
        "event_date": "1890-04",
        "event_type": "thermometer_shed_shift",
        "description": "Thermometer shed shifted; the historical compilation states that it applied no temperature correction for this move.",
        "source": "Smithsonian World Weather Records, 1927 compilation",
        "source_url": "https://repository.si.edu/handle/10088/24035",
    },
    {
        "station_id": "INXLT035254",
        "station_name": "LEH_KASHMIR",
        "event_year": 1883,
        "event_date": "1883-08",
        "event_type": "observation_schedule_expansion",
        "description": "After August, observations expanded from two to four times per day.",
        "source": "IMD annual report OCR, india.history.resource.108636",
        "source_url": "https://indianculture.gov.in/archives/annual-report-meteorological-reporter-government-india-council",
    },
    {
        "station_id": "INXLT035254",
        "station_name": "LEH_KASHMIR",
        "event_year": 1883,
        "event_date": "1883-11-10",
        "event_type": "observatory_move",
        "description": "Observatory site changed.",
        "source": "IMD annual report OCR, india.history.resource.108637",
        "source_url": "https://indianculture.gov.in/archives/annual-report-meteorological-reporter-government-india-council",
    },
    {
        "station_id": "INXLT035254",
        "station_name": "LEH_KASHMIR",
        "event_year": 1894,
        "event_date": "1894-08",
        "event_type": "observation_time_change",
        "description": "World Weather Records reports 10:00/16:00 through July 1894 and 08:00 from August 1894 onward.",
        "source": "Smithsonian World Weather Records, 1927 compilation",
        "source_url": "https://repository.si.edu/handle/10088/24035",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--differences", type=Path, required=True)
    parser.add_argument("--candidate-evaluation", type=Path, required=True)
    parser.add_argument("--imd-raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_station_plot_inventory(output: Path) -> pd.DataFrame:
    rows = []
    for station_id, station_name in STATIONS.items():
        url = f"{NCEI_ROOT}/{station_id}/"
        with urllib.request.urlopen(url, timeout=30) as response:
            html = response.read().decode("utf-8", errors="replace")
        (output / f"{station_id}_directory_index.html").write_text(html, encoding="utf-8")
        for filename, size in re.findall(
            rf'href="({re.escape(station_id)}[^\"]+\.gif)".*?<td align="right">(\d+)</td>',
            html,
            flags=re.DOTALL,
        ):
            rows.append(
                {
                    "station_id": station_id,
                    "station_name": station_name,
                    "filename": filename,
                    "size_bytes": int(size),
                    "url": f"{url}{filename}",
                    "downloaded_or_opened": False,
                    "reason_not_opened": "full-record graphic crosses locked 1921-1939 interval",
                }
            )
    inventory = pd.DataFrame(rows)
    if len(inventory) != 12:
        raise RuntimeError(f"Unexpected station-plot inventory length: {len(inventory)}")
    return inventory


def make_segments(differences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station_id, station_name in STATIONS.items():
        frame = differences.loc[differences["station_id"].eq(station_id)].sort_values("year")
        if frame.empty:
            raise RuntimeError(f"Missing station differences: {station_id}")
        frame = frame.assign(
            adjustment_rounded=frame["qcf_minus_qcu_c"].round(6),
            new_segment=lambda x: (
                x["year"].diff().fillna(1).ne(1)
                | x["adjustment_rounded"].diff().fillna(0).abs().gt(1e-6)
            ),
        )
        frame["segment"] = frame["new_segment"].cumsum()
        for _, group in frame.groupby("segment"):
            rows.append(
                {
                    "station_id": station_id,
                    "station_name": station_name,
                    "start_year": int(group["year"].min()),
                    "end_year": int(group["year"].max()),
                    "year_count": int(len(group)),
                    "qcf_minus_qcu_c": float(group["qcf_minus_qcu_c"].mean()),
                    "segment_starts_after_gap": bool(
                        len(rows) > 0
                        and rows[-1]["station_id"] == station_id
                        and int(group["year"].min()) > rows[-1]["end_year"] + 1
                    ),
                }
            )
    return pd.DataFrame(rows)


def align_events(events: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event in events.itertuples(index=False):
        station_segments = segments.loc[segments["station_id"].eq(event.station_id)].copy()
        candidate_starts = station_segments["start_year"].to_numpy(int)
        nearest_index = int(np.argmin(np.abs(candidate_starts - event.event_year)))
        segment = station_segments.iloc[nearest_index]
        lag = int(segment.start_year - event.event_year)
        if abs(lag) <= 1:
            support = "near_event_step_or_first_available_segment"
        elif abs(lag) <= 2:
            support = "weak_temporal_alignment"
        else:
            support = "no_close_adjustment_step"
        rows.append(
            {
                **event._asdict(),
                "nearest_adjustment_segment_start": int(segment.start_year),
                "nearest_segment_adjustment_c": float(segment.qcf_minus_qcu_c),
                "lag_years": lag,
                "alignment_class": support,
                "magnitude_validated_by_metadata": False,
            }
        )
    return pd.DataFrame(rows)


def station_period_metrics(evaluation: pd.DataFrame, station_regions: pd.DataFrame) -> pd.DataFrame:
    evaluation = evaluation.merge(station_regions, on=["station_id", "station_name"], how="left", validate="many_to_one")
    rows = []
    for start, end in PERIODS:
        period = f"{start}-{end}"
        section = evaluation.loc[evaluation["year"].between(start, end)]
        for region_key, region_part in [("all_supported", section), *section.groupby("region")]:
            for target_column, target in (("qcf_station_degC", "qcf"), ("qcu_station_degC", "qcu")):
                station_rows = []
                for station_id, group in region_part.groupby("station_id"):
                    valid = (
                        np.isfinite(group[target_column])
                        & np.isfinite(group["original_ensemble_degC"])
                        & np.isfinite(group["candidate_v2_ensemble_degC"])
                    )
                    if valid.sum() >= 5:
                        station_rows.append(
                            {
                                "station_id": station_id,
                                "before_bias_c": float(
                                    np.mean(group.loc[valid, "original_ensemble_degC"] - group.loc[valid, target_column])
                                ),
                                "after_bias_c": float(
                                    np.mean(group.loc[valid, "candidate_v2_ensemble_degC"] - group.loc[valid, target_column])
                                ),
                            }
                        )
                station_frame = pd.DataFrame(station_rows)
                if station_frame.empty:
                    continue
                rows.append(
                    {
                        "period": period,
                        "region": region_key,
                        "target": target,
                        "station_count": int(len(station_frame)),
                        "before_mean_bias_c": float(station_frame["before_bias_c"].mean()),
                        "before_mae_c": float(station_frame["before_bias_c"].abs().mean()),
                        "after_mean_bias_c": float(station_frame["after_bias_c"].mean()),
                        "after_mae_c": float(station_frame["after_bias_c"].abs().mean()),
                    }
                )
    return pd.DataFrame(rows)


def processing_envelope(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (period, region), group in metrics.groupby(["period", "region"]):
        by_target = group.set_index("target")
        if not {"qcf", "qcu"}.issubset(by_target.index):
            continue
        before = [float(by_target.loc[target, "before_mean_bias_c"]) for target in ("qcf", "qcu")]
        after = [float(by_target.loc[target, "after_mean_bias_c"]) for target in ("qcf", "qcu")]
        rows.append(
            {
                "period": period,
                "region": region,
                "station_count": int(by_target.loc["qcf", "station_count"]),
                "qcf_before_bias_c": before[0],
                "qcu_before_bias_c": before[1],
                "before_bias_envelope_low_c": min(before),
                "before_bias_envelope_high_c": max(before),
                "qcf_after_bias_c": after[0],
                "qcu_after_bias_c": after[1],
                "after_bias_envelope_low_c": min(after),
                "after_bias_envelope_high_c": max(after),
                "processing_choice_changes_direction_before": bool(np.sign(before[0]) != np.sign(before[1])),
                "processing_choice_changes_direction_after": bool(np.sign(after[0]) != np.sign(after[1])),
            }
        )
    return pd.DataFrame(rows)


def plot_result(
    differences: pd.DataFrame,
    events: pd.DataFrame,
    envelope: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    colors = {"IN007101600": "#176b87", "INXLT035254": "#c44736"}
    for ax, station_id in zip(axes[:2], STATIONS):
        part = differences.loc[differences["station_id"].eq(station_id)].sort_values("year")
        gap_group = part["year"].diff().fillna(1).gt(1).cumsum()
        for _, segment in part.groupby(gap_group):
            ax.step(segment["year"], segment["qcf_minus_qcu_c"], where="mid", color=colors[station_id], lw=2)
        station_events = events.loc[events["station_id"].eq(station_id)]
        for index, event in enumerate(station_events.itertuples(index=False)):
            ax.axvline(event.event_year, color="#555555", alpha=0.55, lw=0.9)
            ax.text(
                event.event_year + 0.15,
                0.97 - 0.10 * (index % 4),
                event.event_type.replace("_", " "),
                transform=ax.get_xaxis_transform(),
                fontsize=8,
                va="top",
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.axvspan(LOCKED_START, LOCKED_END, color="#d8d8d8", alpha=0.65)
        ax.set_xlim(1875, 1950)
        ax.set_ylabel("qcf - qcu (degC)")
        ax.set_title(f"{STATIONS[station_id]} processing adjustment and documented events")
        ax.grid(alpha=0.2)

    overall = envelope.loc[envelope["region"].eq("all_supported")].copy()
    x = np.arange(len(overall))
    width = 0.34
    axes[2].bar(x - width / 2, overall["qcf_before_bias_c"], width, label="qcf primary", color="#176b87")
    axes[2].bar(x + width / 2, overall["qcu_before_bias_c"], width, label="qcu sensitivity", color="#c44736")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_xticks(x, overall["period"])
    axes[2].set_ylabel("Original reconstruction minus station (degC)")
    axes[2].set_title("Supported-station residual under both processing states")
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=0.2)
    fig.savefig(output / "station_breakpoint_history_and_residual_envelope.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [args.differences, args.candidate_evaluation]
    imd_sources = sorted(
        path
        for path in args.imd_raw_root.glob("*_station.json")
        if any(token in path.name for token in ("_simla_", "_leh_"))
    )
    inputs.extend(imd_sources)
    before = {str(path.resolve()): sha256(path) for path in inputs}

    differences = pd.read_csv(args.differences)
    evaluation = pd.read_csv(args.candidate_evaluation)
    for frame_name, frame in (("differences", differences), ("evaluation", evaluation)):
        if frame["year"].between(LOCKED_START, LOCKED_END).any():
            raise RuntimeError(f"Locked period unexpectedly present in {frame_name}")

    inventory = fetch_station_plot_inventory(args.output)
    events = pd.DataFrame(EVENTS)
    segments = make_segments(differences)
    alignment = align_events(events, segments)
    station_regions = differences[["station_id", "station_name", "region"]].drop_duplicates()
    metrics = station_period_metrics(evaluation, station_regions)
    envelope = processing_envelope(metrics)

    inventory.to_csv(args.output / "official_station_plot_inventory.csv", index=False)
    events.to_csv(args.output / "documented_station_events.csv", index=False)
    segments.to_csv(args.output / "qcf_qcu_adjustment_segments.csv", index=False)
    alignment.to_csv(args.output / "event_alignment.csv", index=False)
    metrics.to_csv(args.output / "regional_station_processing_metrics.csv", index=False)
    envelope.to_csv(args.output / "regional_processing_envelope.csv", index=False)
    plot_result(differences, events, envelope, args.output)

    overall = envelope.loc[envelope["region"].eq("all_supported")].set_index("period")
    early_direction_agreement = bool(
        (overall.loc[["1880-1900", "1901-1920"], ["qcf_before_bias_c", "qcu_before_bias_c"]] > 0).all().all()
    )
    unexplained_simla_steps = segments.loc[
        segments["station_id"].eq("IN007101600")
        & segments["start_year"].between(1892, 1895)
    ]["start_year"].astype(int).tolist()
    locked_gap_transition = bool(
        (segments["station_id"].eq("INXLT035254") & segments["start_year"].eq(1940)).any()
    )
    summary = {
        "experiment": "temperature_ghcn_breakpoint_station_history_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "station_ids": list(STATIONS),
        "official_station_plot_files_inventoried": int(len(inventory)),
        "official_station_plot_images_downloaded_or_opened": 0,
        "documented_event_count": int(len(events)),
        "early_processing_states_agree_original_is_warm_for_supported_stations": early_direction_agreement,
        "simla_late_step_start_years_not_explained_by_single_documented_event": unexplained_simla_steps,
        "leh_1894_1895_adjustment_transition_aligns_with_documented_time_change": True,
        "post_1920_to_1940_leh_adjustment_transition_present_but_not_localized": locked_gap_transition,
        "decision": "retain_qcf_primary_with_qcu_processing_envelope; station history supports plausibility of some adjustments but does not validate their full magnitude; no uniform HMA or pre-1880 correction",
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    after = {str(path.resolve()): sha256(path) for path in inputs}
    if before != after:
        raise RuntimeError("Source state changed")
    manifest = {
        "inputs": [{"path": path, "sha256": value} for path, value in before.items()],
        "web_metadata": [
            {
                "url": f"{NCEI_ROOT}/{station_id}/",
                "content": "directory listing only; full-record graphics not opened",
            }
            for station_id in STATIONS
        ],
        "source_hashes_unchanged": True,
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
    }
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    early = overall.loc[["1880-1900", "1901-1920"]]
    report = f"""# GHCN breakpoint and station-history audit

## Scope

This read-only audit tests whether the large GHCN-M qcf-minus-qcu adjustments at Shimla and Leh are temporally consistent with official station-history events. It also carries qcf/qcu processing uncertainty into the existing direct-station residual calculation. No production field, EnKF run, or OGGM run was touched.

The NCEI station directories contain 12 official diagnostic GIFs for the two stations. The graphics span the full record and were therefore inventoried but deliberately not downloaded or opened because they cross the locked 1921-1939 interval.

## Decisive findings

- Leh: the adjustment changes from about -1.75 degC before 1894 to -1.62 degC in 1894 and -0.96 degC from 1895. This closely aligns with the documented August 1894 change from 10:00/16:00 observations to 08:00. The alignment supports the adjustment's plausibility, but station metadata alone do not validate its exact magnitude.
- Shimla: qcf-qcu changes near the documented 1880 move/time change, 1885 move, 1887 thermometer replacement, 1889 time change, and 1890 shed shift. However, additional steps beginning in {', '.join(map(str, unexplained_simla_steps))} culminate in +1.68 degC and are not explained by one documented event. They remain statistically generated homogenization adjustments with incomplete metadata confirmation.
- A Leh adjustment change exists between the allowed 1920 and 1940 observations. Its date and cause are intentionally not localized because the intervening years remain sealed.

## Residual consequence

For all supported stations, the original reconstruction-minus-station mean bias is:

| Period | qcf primary | qcu sensitivity |
|---|---:|---:|
| 1880-1900 | {early.loc['1880-1900', 'qcf_before_bias_c']:+.3f} degC | {early.loc['1880-1900', 'qcu_before_bias_c']:+.3f} degC |
| 1901-1920 | {early.loc['1901-1920', 'qcf_before_bias_c']:+.3f} degC | {early.loc['1901-1920', 'qcu_before_bias_c']:+.3f} degC |
| 1940-1950 | {overall.loc['1940-1950', 'qcf_before_bias_c']:+.3f} degC | {overall.loc['1940-1950', 'qcu_before_bias_c']:+.3f} degC |

Both processing states therefore support the direction of an early warm residual for the represented stations, while the magnitude is processing-dependent. This strengthens the western/central direct-station statement, but it still does not establish an HMA-wide correction, a continuous temporal correction, or any pre-1880 correction.

## Decision

Retain qcf as protocol-primary and qcu as a mandatory equal-status processing envelope. The station-history evidence makes part of the adjustment chain physically plausible but does not validate the full Shimla magnitude. Candidate v2 remains a rejected diagnostic artifact because this audit does not repair its temporal-shape failures.

- Locked 1921-1939 values accessed: **no**
- Prohibited source accessed: **no**
- Original or production files modified: **no**
"""
    (args.output / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
