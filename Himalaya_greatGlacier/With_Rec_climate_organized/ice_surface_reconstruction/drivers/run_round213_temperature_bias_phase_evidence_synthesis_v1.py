from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "round_213_temperature_bias_phase_evidence_synthesis_v1"
R206 = LOOP / "round_206_setp_winter_temperature_locked_safe_comparison_v3"
R208 = LOOP / "round_208_setp_cru_tmin_annual_target_audit_v1"
R212 = LOOP / "round_212_wscp_temperature_locked_safe_comparison_v1"
CONTRACT = OUT / "ROUND_CONTRACT.md"
DRIVER = Path(__file__).resolve()

SOURCES = {
    "huang_annual": R206 / "annual_allowed_curves.csv",
    "huang_windows": R206 / "temperature_50year_windows_1500_1899.csv",
    "huang_trends": R206 / "diagnostic_interval_trends.csv",
    "huang_summary": R206 / "summary.json",
    "instrument_summary": R208 / "summary.json",
    "baiyu_annual": R212 / "annual_allowed_curves.csv",
    "baiyu_windows": R212 / "temperature_50year_windows_1500_1899.csv",
    "baiyu_trends": R212 / "diagnostic_interval_trends.csv",
    "baiyu_summary": R212 / "summary.json",
    "contract": CONTRACT,
    "driver": DRIVER,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def trend_row(frame: pd.DataFrame, start: int, end: int) -> pd.Series:
    selection = frame[(frame.start_year == start) & (frame.end_year == end)]
    if len(selection) != 1:
        raise RuntimeError(f"Missing unique trend row {start}-{end}")
    return selection.iloc[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in SOURCES.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: state(path) for name, path in SOURCES.items()}

    huang_annual = pd.read_csv(SOURCES["huang_annual"])
    huang_windows = pd.read_csv(SOURCES["huang_windows"])
    huang_trends = pd.read_csv(SOURCES["huang_trends"])
    huang_summary = json.loads(SOURCES["huang_summary"].read_text(encoding="utf-8"))
    instrument = json.loads(
        SOURCES["instrument_summary"].read_text(encoding="utf-8")
    )
    baiyu_annual = pd.read_csv(SOURCES["baiyu_annual"])
    baiyu_windows = pd.read_csv(SOURCES["baiyu_windows"])
    baiyu_trends = pd.read_csv(SOURCES["baiyu_trends"])
    baiyu_summary = json.loads(SOURCES["baiyu_summary"].read_text(encoding="utf-8"))

    for frame_name, frame in (
        ("huang", huang_annual),
        ("baiyu", baiyu_annual),
    ):
        years = set(frame.year.astype(int))
        leaked = years & set(range(1921, 1940))
        if leaked:
            raise RuntimeError(f"Locked years present in {frame_name}: {sorted(leaked)}")

    hmetrics = huang_summary["metrics"]
    bmetrics = baiyu_summary["metrics"]
    htrend = trend_row(huang_trends, 1901, 1920)
    btrend = trend_row(baiyu_trends, 1901, 1920)
    product_trends = instrument["annual_consensus"]["early_1902_1920"][
        "product_trends"
    ]
    station_support = instrument["cru_tmn"][
        "early_1902_1920_mean_station_supported_cell_month_fraction"
    ]

    evidence_rows = [
        {
            "record": "Huang 2019 Changdu winter Tmin",
            "season": "previous-November to current-February Tmin",
            "independence": "mostly independent; one probable/related chronology",
            "negative_50year_windows_1500_1899": hmetrics[
                "external_negative_window_count"
            ],
            "median_anomaly_1500_1899_degC": hmetrics[
                "median_external_winter_anomaly_1500_1899_degC"
            ],
            "trend_1901_1920_degC_per_century": float(
                htrend.external_winter_tmin_trend_degC_per_century
            ),
            "project_local_trend_1901_1920_degC_per_century": float(
                htrend.project_setp_annual_trend_degC_per_century
            ),
        },
        {
            "record": "Zhang/Li 2022 Baiyu June-July temperature",
            "season": "June-July mean temperature",
            "independence": "source-independent; sampled October 2017; nearest ledger group 111.355 km",
            "negative_50year_windows_1500_1899": bmetrics[
                "external_negative_window_count"
            ],
            "median_anomaly_1500_1899_degC": bmetrics[
                "median_external_anomaly_1500_1899_degC"
            ],
            "trend_1901_1920_degC_per_century": float(
                btrend.wscp_jj_reconstruction_anomaly_degC_trend_degC_per_century
            ),
            "project_local_trend_1901_1920_degC_per_century": float(
                btrend.project_local_box_annual_anomaly_degC_trend_degC_per_century
            ),
        },
    ]
    evidence = pd.DataFrame(evidence_rows)
    evidence.to_csv(OUT / "independent_temperature_evidence_summary.csv", index=False)

    uniform_support_count = int(
        np.sum(evidence.negative_50year_windows_1500_1899.to_numpy(int) >= 6)
    )
    proxy_negative_count = int(
        np.sum(evidence.trend_1901_1920_degC_per_century.to_numpy(float) < 0)
    )
    project_positive_count = int(
        np.sum(
            evidence.project_local_trend_1901_1920_degC_per_century.to_numpy(float)
            > 0
        )
    )
    grid_positive_count = int(np.sum(np.asarray(list(product_trends.values())) > 0))

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12.5, 8.2),
        gridspec_kw={"height_ratios": [1.0, 1.45], "hspace": 0.22},
    )
    axis = axes[0]
    centers = huang_windows.center_year.to_numpy(float)
    axis.plot(
        centers,
        huang_windows.external_winter_tmin_anomaly_degC,
        color="#222222",
        marker="o",
        linewidth=2.0,
        label="Changdu winter Tmin",
    )
    axis.plot(
        centers,
        huang_windows.project_setp_annual_anomaly_degC,
        color="#B33A3A",
        marker="o",
        linestyle="--",
        linewidth=1.6,
        label="Project at Changdu box",
    )
    axis.plot(
        centers,
        baiyu_windows.wscp_jj_reconstruction_anomaly_degC,
        color="#175B8C",
        marker="s",
        linewidth=2.0,
        label="Baiyu June-July",
    )
    axis.plot(
        centers,
        baiyu_windows.project_local_box_annual_anomaly_degC,
        color="#C89021",
        marker="s",
        linestyle="--",
        linewidth=1.6,
        label="Project at Baiyu box",
    )
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set_ylabel("50-year anomaly (degC; 1961-1990)")
    axis.set_title("Independent seasonal evidence does not support uniform 1500-1899 cooling")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, ncol=2, loc="lower left")

    axis = axes[1]
    h = huang_annual[(huang_annual.year >= 1870) & (huang_annual.year <= 1920)]
    b = baiyu_annual[(baiyu_annual.year >= 1870) & (baiyu_annual.year <= 1920)]
    axis.plot(
        h.year,
        h.external_winter_tmin_anomaly_degC,
        color="#222222",
        linewidth=2.0,
        label="Changdu winter Tmin",
    )
    axis.plot(
        h.year,
        h.project_setp_annual_anomaly_degC,
        color="#B33A3A",
        linewidth=1.5,
        linestyle="--",
        label="Project at Changdu box",
    )
    axis.plot(
        b.year,
        b.wscp_jj_reconstruction_anomaly_degC,
        color="#175B8C",
        linewidth=2.0,
        label="Baiyu June-July",
    )
    axis.plot(
        b.year,
        b.project_local_box_annual_anomaly_degC,
        color="#C89021",
        linewidth=1.5,
        linestyle="--",
        label="Project at Baiyu box",
    )
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set_xlim(1870, 1920)
    axis.set_xlabel("Year")
    axis.set_ylabel("Annual anomaly (degC; 1961-1990)")
    axis.set_title("Both seasonal proxies cool in 1901-1920 while project local curves warm")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, ncol=2, loc="lower left")
    figure.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.08)
    figure.savefig(OUT / "temperature_bias_vs_phase_synthesis.png", dpi=220)
    plt.close(figure)

    decision = (
        "uniform_warm_correction_rejected_early20_phase_conflict_strengthened_but_unresolved"
    )
    summary = {
        "round": 213,
        "decision": decision,
        "uniform_1500_1899": {
            "external_record_count": 2,
            "records_supporting_uniform_cooling": uniform_support_count,
            "huang_negative_windows": int(
                evidence.iloc[0].negative_50year_windows_1500_1899
            ),
            "baiyu_negative_windows": int(
                evidence.iloc[1].negative_50year_windows_1500_1899
            ),
            "interpretation": "no independent support for a uniform eastern-HMA cooling correction",
        },
        "early_1901_1920": {
            "seasonal_proxy_negative_count": proxy_negative_count,
            "project_local_positive_count": project_positive_count,
            "annual_grid_positive_count": grid_positive_count,
            "annual_grid_product_count": len(product_trends),
            "cru_station_supported_cell_month_fraction": station_support,
            "interpretation": "cross-season proxy-versus-grid conflict under sparse station support",
        },
        "correction_authorization": {
            "uniform_translation": False,
            "early_phase_correction": False,
            "production_rerun": False,
        },
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    after = {name: state(path) for name, path in SOURCES.items()}
    if before != after:
        raise RuntimeError("A derived source changed during synthesis")
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {"sources_before": before, "sources_after": after},
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = f"""# Round 213 Result

## Decision

`{decision}`

Neither independent seasonal record supports a uniform 1500-1899 cooling
correction. Changdu winter Tmin is negative in 2/8 windows and Baiyu
June-July temperature is negative in 2/8; both have positive historical
median anomalies relative to 1961-1990.

The records do agree on a separate issue: both cool over 1901-1920 while both
project-local annual curves warm. However, all five early annual gridded
products warm and exact CRU winter Tmin also warms, with only
`{station_support:.1%}` of early CRU box cell-months station-supported.

Therefore the uniform warm-bias hypothesis is rejected. The early phase
conflict is strengthened but remains unresolved; neither a translation nor a
1901-1920 correction is authorized.
"""
    (OUT / "FINAL_RESULT.md").write_text(result, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
