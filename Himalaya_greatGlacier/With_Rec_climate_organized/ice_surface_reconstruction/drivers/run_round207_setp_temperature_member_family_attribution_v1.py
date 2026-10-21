from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import h5py
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
OUT = LOOP / "round_207_setp_temperature_member_family_attribution_v1"
ROUND206 = LOOP / "round_206_setp_winter_temperature_locked_safe_comparison_v3"
ALLOWED_CURVES = ROUND206 / "annual_allowed_curves.csv"
LINEAGE = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v1"
    / "round_00_production_reproduction/member_lineage.csv"
)
REFERENCE_SOURCE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data"
    r"\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
HMA_MASK = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
)
CONTRACT = OUT / "ROUND_CONTRACT.md"
DRIVER = Path(__file__).resolve()
DRIVERS = DRIVER.parent
ROUND21_DRIVER = DRIVERS / "run_climate_reconstruction_loop_v2_round21_temperature_reference_field_v1.py"
ROUND23_DRIVER = DRIVERS / "run_climate_reconstruction_loop_v2_round23_corrected_reconstruction_reference_envelope_v1.py"

EXPECTED_REFERENCE_SOURCE_SHA256 = (
    "450850a6654e9080e9586df06d227205b564b2e91e5ba1f95b1610b723664d90"
)
BOX = {"lon_min": 94.5, "lon_max": 97.5, "lat_min": 29.5, "lat_max": 31.5}
PROJECT_MEMBERS = ("AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weighted_series(field: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(field) & mask[None]
    numerator = np.nansum(field * weights[None], axis=(1, 2))
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def finite_mean_stream(arrays: list[np.ndarray]) -> np.ndarray:
    total = np.zeros(arrays[0].shape, dtype=float)
    count = np.zeros(arrays[0].shape, dtype=np.uint8)
    for values in arrays:
        valid = np.isfinite(values)
        total[valid] += values[valid]
        count[valid] += 1
    output = np.full(total.shape, np.nan)
    valid = count > 0
    output[valid] = total[valid] / count[valid]
    return output


def smooth_segment(values: np.ndarray, years: np.ndarray, width: int = 31) -> np.ndarray:
    output = np.full(values.shape, np.nan)
    for start, end in ((1000, 1920), (1940, 2000)):
        selection = (years >= start) & (years <= end)
        output[selection] = (
            pd.Series(values[selection])
            .rolling(width, center=True, min_periods=width)
            .mean()
            .to_numpy()
        )
    return output


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3:
        return math.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def trend(values: np.ndarray, years: np.ndarray, start: int, end: int) -> float:
    selection = (years >= start) & (years <= end) & np.isfinite(values)
    if int(selection.sum()) != end - start + 1:
        raise RuntimeError(f"Incomplete trend interval {start}-{end}")
    return float(np.polyfit(years[selection], values[selection], 1)[0] * 100)


def window_values(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    output = []
    for start in range(1500, 1900, 50):
        selection = (years >= start) & (years <= start + 49)
        if int(np.isfinite(values[selection]).sum()) != 50:
            raise RuntimeError(f"Incomplete window {start}-{start + 49}")
        output.append(float(np.mean(values[selection])))
    return np.asarray(output)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        ALLOWED_CURVES,
        LINEAGE,
        REFERENCE_SOURCE,
        HMA_MASK,
        CONTRACT,
        ROUND21_DRIVER,
        ROUND23_DRIVER,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    round21 = load_module(ROUND21_DRIVER, "round21_locked_reader")
    round23 = load_module(ROUND23_DRIVER, "round23_locked_reader")
    lineage = pd.read_csv(LINEAGE)
    tas = lineage[lineage.variable == "tas"].copy()
    if len(tas) != 6 or set(tas.member) != set(PROJECT_MEMBERS):
        raise RuntimeError("Project member lineage changed")
    project_paths = {row.member: Path(row.source_tif) for row in tas.itertuples(index=False)}

    sources = {
        "allowed_curves": ALLOWED_CURVES,
        "lineage": LINEAGE,
        "reference_source": REFERENCE_SOURCE,
        "hma_mask": HMA_MASK,
        "contract": CONTRACT,
        "round21_driver": ROUND21_DRIVER,
        "round23_driver": ROUND23_DRIVER,
        **{f"project_{name}": path for name, path in project_paths.items()},
    }
    before = {name: state(path) for name, path in sources.items()}
    if before["reference_source"]["sha256"] != EXPECTED_REFERENCE_SOURCE_SHA256:
        raise RuntimeError("Round 21 source hash changed")
    for row in tas.itertuples(index=False):
        if before[f"project_{row.member}"]["sha256"] != row.source_tif_sha256:
            raise RuntimeError(f"Project member hash changed: {row.member}")

    allowed = pd.read_csv(ALLOWED_CURVES)
    if allowed.year.between(1921, 1939).any():
        raise RuntimeError("Round 206 allowed curves contain locked rows")
    years = np.arange(1000, 2001)
    external = np.full(len(years), np.nan)
    external_by_year = dict(
        zip(
            allowed.year.astype(int),
            allowed.external_winter_tmin_anomaly_degC.astype(float),
        )
    )
    for index, year in enumerate(years):
        if year in external_by_year:
            external[index] = external_by_year[year]

    with h5py.File(REFERENCE_SOURCE, "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int).ravel()
        latitudes = np.asarray(handle["lat"], dtype=float).ravel()
        longitudes = np.asarray(handle["lon"], dtype=float).ravel()
        if not np.array_equal(source_years, years):
            raise RuntimeError("Reference source year coordinate changed")
        box_mask = (
            (latitudes[:, None] >= BOX["lat_min"])
            & (latitudes[:, None] <= BOX["lat_max"])
            & (longitudes[None, :] >= BOX["lon_min"])
            & (longitudes[None, :] <= BOX["lon_max"])
        )
        if int(box_mask.sum()) != 24:
            raise RuntimeError(f"SETP box cell count changed: {int(box_mask.sum())}")

        reference_series: dict[str, np.ndarray] = {}
        local_fields = [
            round21.read_allowed_member(handle[round21.MEMBERS[name]], years)
            for name in ("HGAO", "O")
        ]
        reference_series["local_enkf"] = weighted_series(
            finite_mean_stream(local_fields), box_mask, latitudes
        )
        del local_fields

        reference_series["lmr"] = weighted_series(
            round21.read_allowed_member(handle[round21.MEMBERS["LMR"]], years),
            box_mask,
            latitudes,
        )

        neukom_fields = [
            round21.read_allowed_member(handle[round21.MEMBERS[name]], years)
            for name in ("Nk_AM", "Nk_CCA", "Nk_CPS", "Nk_DA", "Nk_GraphEM", "Nk_PCR")
        ]
        reference_series["neukom"] = weighted_series(
            finite_mean_stream(neukom_fields), box_mask, latitudes
        )
        del neukom_fields

        reference_series["phyda"] = weighted_series(
            round21.read_allowed_member(handle[round21.MEMBERS["PHYDA"]], years),
            box_mask,
            latitudes,
        )

    project_series: dict[str, np.ndarray] = {}
    orientations = set()
    for name in PROJECT_MEMBERS:
        field, orientation = round23.read_allowed_tif(
            project_paths[name], years, latitudes, longitudes
        )
        orientations.add(orientation)
        project_series[name] = weighted_series(field, box_mask, latitudes)
    if len(orientations) != 1:
        raise RuntimeError(f"Project orientations differ: {orientations}")

    locked = (years >= 1921) & (years <= 1939)
    for name, values in {**project_series, **reference_series}.items():
        if np.any(np.isfinite(values[locked])):
            raise RuntimeError(f"Locked values present: {name}")

    curves = pd.DataFrame({"year": years})
    curves["external_winter_tmin"] = external
    for name, values in project_series.items():
        curves[f"project_{name}"] = values
    for name, values in reference_series.items():
        curves[f"reference_{name}"] = values
    curves = curves.loc[~curves.year.between(1921, 1939)].copy()
    curves.to_csv(OUT / "member_family_annual_allowed_curves.csv", index=False)

    external_windows = window_values(external, years)
    external_smooth = smooth_segment(external, years)
    paleo = (years >= 1500) & (years <= 1899)
    rows = []
    all_series = {
        **{f"project_{name}": values for name, values in project_series.items()},
        **{f"reference_{name}": values for name, values in reference_series.items()},
    }
    for name, values in all_series.items():
        windows = window_values(values, years)
        smooth = smooth_segment(values, years)
        rows.append(
            {
                "series": name,
                "group": "project_member" if name.startswith("project_") else "reference_family",
                "window_correlation_external_1500_1899": correlation(windows, external_windows),
                "window_change_sign_agreement_external_of_7": int(
                    np.sum(np.sign(np.diff(windows)) == np.sign(np.diff(external_windows)))
                ),
                "smooth31_correlation_external_1500_1899": correlation(
                    smooth[paleo], external_smooth[paleo]
                ),
                "median_anomaly_1500_1899_degC": float(np.median(windows)),
                "trend_1850_1900_degC_per_century": trend(values, years, 1850, 1900),
                "trend_1901_1920_degC_per_century": trend(values, years, 1901, 1920),
                "trend_1940_2000_degC_per_century": trend(values, years, 1940, 2000),
            }
        )
    metrics = pd.DataFrame(rows)
    external_metrics = {
        "median_anomaly_1500_1899_degC": float(np.median(external_windows)),
        "trend_1850_1900_degC_per_century": trend(external, years, 1850, 1900),
        "trend_1901_1920_degC_per_century": trend(external, years, 1901, 1920),
        "trend_1940_2000_degC_per_century": trend(external, years, 1940, 2000),
    }
    metrics.to_csv(OUT / "member_family_attribution_metrics.csv", index=False)

    project_metrics = metrics[metrics.group == "project_member"]
    reference_metrics = metrics[metrics.group == "reference_family"]
    project_wrong = (
        np.sign(project_metrics.trend_1901_1920_degC_per_century.to_numpy(float))
        != np.sign(external_metrics["trend_1901_1920_degC_per_century"])
    )
    reference_wrong = (
        np.sign(reference_metrics.trend_1901_1920_degC_per_century.to_numpy(float))
        != np.sign(external_metrics["trend_1901_1920_degC_per_century"])
    )
    if external_metrics["trend_1901_1920_degC_per_century"] < 0 and int(project_wrong.sum()) == 6:
        attribution = "project_common_mode_wrong_sign"
    elif 0 < int(project_wrong.sum()) < 6:
        attribution = "project_group_specific_wrong_sign"
    else:
        attribution = "project_members_do_not_show_a_wrong_sign_pattern"

    colors = {"AO": "#2C6EAA", "HGAO": "#B33A3A", "O": "#C89021"}
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 8.4), sharex=True)
    axis = axes[0]
    axis.plot(years, external_smooth, color="#222222", linewidth=2.6, label="Huang winter Tmin")
    for name, values in project_series.items():
        method, proxy = name.split("_")
        axis.plot(
            years,
            smooth_segment(values, years),
            color=colors[method],
            linewidth=1.5,
            linestyle="-" if proxy == "6" else "--",
            label=name,
        )
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.axvspan(1921, 1939, color="#E5E5E5", alpha=0.8, linewidth=0)
    axis.set_ylabel("31-year anomaly (degC)")
    axis.set_title("Six frozen project members")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, ncol=4, loc="upper left")

    family_colors = {
        "local_enkf": "#7B4FA1",
        "lmr": "#147D78",
        "neukom": "#3B6F9F",
        "phyda": "#D06A28",
    }
    axis = axes[1]
    axis.plot(years, external_smooth, color="#222222", linewidth=2.6, label="Huang winter Tmin")
    for name, values in reference_series.items():
        axis.plot(
            years,
            smooth_segment(values, years),
            color=family_colors[name],
            linewidth=1.8,
            label=name,
        )
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.axvspan(1921, 1939, color="#E5E5E5", alpha=0.8, linewidth=0)
    axis.set_xlim(1490, 2005)
    axis.set_xlabel("Year")
    axis.set_ylabel("31-year anomaly (degC)")
    axis.set_title("Four reference families")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, ncol=5, loc="upper left")
    figure.suptitle("SETP temperature phase attribution, 1500-2000", fontsize=15)
    figure.text(
        0.99,
        0.01,
        "Gray gap: sealed 1921-1939. Solid/dashed project lines use proxy6/proxy9.",
        ha="right",
        fontsize=8.5,
        color="#555555",
    )
    figure.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.08, hspace=0.22)
    figure.savefig(OUT / "member_family_phase_attribution_1500_2000.png", dpi=220)
    plt.close(figure)

    after = {name: state(path) for name, path in sources.items()}
    if before != after:
        raise RuntimeError("A read-only source changed during Round 207")

    summary = {
        "round": 207,
        "decision": attribution,
        "external": external_metrics,
        "project_member_wrong_sign_count_1901_1920": int(project_wrong.sum()),
        "project_member_count": 6,
        "reference_family_wrong_sign_count_1901_1920": int(reference_wrong.sum()),
        "reference_family_count": 4,
        "best_project_window_shape_member": metrics.loc[
            project_metrics.window_correlation_external_1500_1899.idxmax(), "series"
        ],
        "best_reference_window_shape_family": metrics.loc[
            reference_metrics.window_correlation_external_1500_1899.idxmax(), "series"
        ],
        "interpretation_boundary": (
            "The external record is winter Tmin calibrated to CRU TS 4.01. "
            "This attribution diagnoses shared versus method-specific phase only; "
            "it does not authorize member weighting or an absolute correction."
        ),
        "source_states_unchanged": True,
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (OUT / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    result = f"""# Round 207 final result

## Decision

`{attribution}`

- External winter Tmin trend in 1901-1920:
  `{external_metrics['trend_1901_1920_degC_per_century']:+.3f} degC/century`.
- Project members with the opposite trend sign: `{int(project_wrong.sum())}/6`.
- Reference families with the opposite trend sign: `{int(reference_wrong.sum())}/4`.
- Best 1500-1899 50-year shape among project members:
  `{summary['best_project_window_shape_member']}`.
- Best 1500-1899 50-year shape among reference families:
  `{summary['best_reference_window_shape_family']}`.

This is a frozen read-only attribution. It does not authorize reweighting,
correction, EnKF, production reconstruction, or OGGM. No locked-period or
Jionglacuo value was accessed, and all source hashes are unchanged.
"""
    (OUT / "FINAL_RESULT.md").write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
