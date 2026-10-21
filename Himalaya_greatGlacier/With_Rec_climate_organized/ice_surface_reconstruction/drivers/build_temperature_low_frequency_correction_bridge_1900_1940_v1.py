"""Freeze a correction-only bridge across 1900-1940 without reading locked climate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from netCDF4 import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import audit_temperature_low_frequency_transition_1899_1940_v1 as transition


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
LOW = RESULTS / "climate_reconstruction_temperature_pmip_constrained_low_frequency_correction_v1"
LOW_NC = LOW / "temperature_equal_blend_constrained_low_frequency_candidate_1500_1899.nc"
TRANSITION_SUMMARY = RESULTS / (
    "climate_reconstruction_temperature_low_frequency_transition_1899_1940_audit_v1/summary.json"
)
REFERENCE = transition.REFERENCE
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
OUTPUT = RESULTS / "climate_reconstruction_temperature_low_frequency_correction_bridge_1900_1940_v1"
POST_YEARS = np.arange(1940, 1960)
BRIDGE_YEARS = np.arange(1900, 1941)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Correction-Only 1900-1940 Bridge Contract\n\n"
        "Use no 1901-1939 climate value. At each common-support cell, connect the accepted 1899 "
        "low-frequency correction and its mean 1895-1899 slope to the mean allowed reference-minus-"
        "background correction over 1940-1959 with zero terminal slope, using a cubic Hermite curve. "
        "Save only the correction field, never original or corrected locked-period climate. Require "
        "1899 and 1940 endpoint closure <=2e-6 degC, maximum bridge overshoot beyond the endpoint "
        "range <=0.10 degC, 99th-percentile absolute annual correction increment <=0.03 degC, and "
        "maximum increment <=0.10 degC. Passing freezes a bridge for later blind application and "
        "validation; it does not authorize production or OGGM forcing.\n",
        encoding="utf-8",
    )
    paths = {
        "low_nc": LOW_NC, "transition_summary": TRANSITION_SUMMARY,
        "reference": REFERENCE, "prior": transition.PRIOR,
        "indices": transition.INDICES, "proxy6": transition.PROXY6,
        "proxy9": transition.PROXY9, "driver": DRIVER, "contract": contract,
    }
    before = {name: sha256(path) for name, path in paths.items()}
    facts_snapshot = sha256(FACTS)
    transition_summary = json.loads(TRANSITION_SUMMARY.read_text(encoding="utf-8"))
    if not transition_summary["passes_bridge_feasibility_gates"]:
        raise RuntimeError("Transition audit did not authorize bridge design")

    with Dataset(LOW_NC) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        hma_mask = np.asarray(dataset.variables["hma_mask"][:], dtype=bool)
        common_grid = np.asarray(dataset.variables["common_support_mask"][:], dtype=bool)
        correction = np.asarray(dataset.variables["tas_low_frequency_correction"][:], dtype=float)
    common = common_grid[hma_mask]
    correction_points = correction[:, hma_mask]
    value_1899 = correction_points[years == 1899][0, common]
    slope_1895_1899 = (
        correction_points[years == 1899][0, common]
        - correction_points[years == 1895][0, common]
    ) / 4.0

    with Dataset(REFERENCE) as dataset:
        all_years = np.asarray(dataset.variables["year"][:], dtype=int)
        indices = np.asarray([int(np.flatnonzero(all_years == year)[0]) for year in transition.YEARS])
        reference = np.asarray(dataset.variables["tas_reference_family_balanced"][indices], dtype=float)[:, hma_mask]
    background = transition.recover_background(lat, lon, hma_mask)
    target_1940 = np.mean(
        (reference - background)[(transition.YEARS >= 1940) & (transition.YEARS <= 1959)][:, common],
        axis=0,
    )

    duration = 41.0
    u = (BRIDGE_YEARS - 1899) / duration
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    bridge_common = (
        h00[:, None] * value_1899[None]
        + h10[:, None] * duration * slope_1895_1899[None]
        + h01[:, None] * target_1940[None]
        + h11[:, None] * duration * 0.0
    )
    full_years = np.r_[years, BRIDGE_YEARS]
    full_points = np.full((len(full_years), int(hma_mask.sum())), np.nan, dtype=float)
    full_points[: len(years)] = correction_points
    full_points[len(years) :, common] = bridge_common

    endpoint_1899_closure = float(np.max(np.abs(full_points[full_years == 1899][0, common] - value_1899)))
    endpoint_1940_closure = float(np.max(np.abs(full_points[full_years == 1940][0, common] - target_1940)))
    lower = np.minimum(value_1899, target_1940)
    upper = np.maximum(value_1899, target_1940)
    overshoot = np.maximum(lower[None] - bridge_common, bridge_common - upper[None])
    max_overshoot = float(np.max(np.maximum(overshoot, 0)))
    increments = np.diff(np.vstack([value_1899[None], bridge_common]), axis=0)
    increment_p99 = float(np.percentile(np.abs(increments), 99))
    increment_max = float(np.max(np.abs(increments)))
    gates = {
        "endpoint_1899_closure_at_most_2e_6_degC": bool(endpoint_1899_closure <= 2e-6),
        "endpoint_1940_closure_at_most_2e_6_degC": bool(endpoint_1940_closure <= 2e-6),
        "maximum_overshoot_at_most_0p10_degC": bool(max_overshoot <= 0.10),
        "annual_increment_p99_at_most_0p03_degC": bool(increment_p99 <= 0.03),
        "annual_increment_max_at_most_0p10_degC": bool(increment_max <= 0.10),
    }
    pass_all = bool(all(gates.values()))

    output_nc = OUTPUT / "temperature_low_frequency_correction_only_1500_1940.nc"
    with Dataset(output_nc, "w", format="NETCDF4") as dataset:
        dataset.createDimension("year", len(full_years))
        dataset.createDimension("lat", len(lat))
        dataset.createDimension("lon", len(lon))
        dataset.createVariable("year", "i4", ("year",))[:] = full_years
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat
        dataset.createVariable("lon", "f8", ("lon",))[:] = lon
        dataset.createVariable("hma_mask", "i1", ("lat", "lon"))[:] = hma_mask.astype(np.int8)
        dataset.createVariable("common_support_mask", "i1", ("lat", "lon"))[:] = common_grid.astype(np.int8)
        grid = np.full((len(full_years), len(lat), len(lon)), np.nan, dtype=np.float32)
        grid[:, hma_mask] = full_points.astype(np.float32)
        variable = dataset.createVariable(
            "tas_low_frequency_correction_only", "f4", ("year", "lat", "lon"),
            zlib=True, complevel=4, chunksizes=(1, len(lat), len(lon)),
            fill_value=np.float32(np.nan),
        )
        variable[:] = grid
        variable.units = "degC correction"
        dataset.title = "Precommitted low-frequency correction-only bridge"
        dataset.status = "correction only; no locked-period climate values read or stored"
        dataset.bridge_method = "cubic Hermite: 1899 value/1895-1899 slope to 1940-1959 allowed mean/zero slope"
        dataset.production_authorized = "false"
        dataset.oggm_forcing_authorized = "false"

    weights = np.cos(np.deg2rad(np.meshgrid(lat, lon, indexing="ij")[0][hma_mask][common]))
    hma_bridge = np.sum(bridge_common * weights[None], axis=1) / np.sum(weights)
    fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    pre_hma = np.nansum(correction_points[:, common] * weights[None], axis=1) / np.sum(weights)
    axis.plot(years[years >= 1850], pre_hma[years >= 1850], color="#D55E00", linewidth=2.0, label="Accepted correction")
    axis.plot(BRIDGE_YEARS, hma_bridge, color="#0072B2", linewidth=2.2, label="Precommitted correction-only bridge")
    axis.scatter([1940], [float(np.sum(target_1940 * weights) / np.sum(weights))], color="#111111", s=35, label="1940-1959 allowed anchor")
    axis.axhline(0, color="#777777", linewidth=0.7)
    axis.set(title="HMA low-frequency correction bridge", xlabel="Year", ylabel="Correction (degC)")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    figure = OUTPUT / "temperature_low_frequency_correction_bridge_1900_1940.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in paths.items()}
    unchanged = before == after
    summary = {
        "status": "complete", "locked_1901_1939_climate_values_read": 0,
        "locked_1901_1939_corrected_climate_values_generated": 0,
        "numerical_sources_unchanged": unchanged, "facts_snapshot_sha256": facts_snapshot,
        "endpoint_1899_closure_max_degC": endpoint_1899_closure,
        "endpoint_1940_closure_max_degC": endpoint_1940_closure,
        "maximum_bridge_overshoot_degC": max_overshoot,
        "annual_correction_increment_p99_degC": increment_p99,
        "annual_correction_increment_max_degC": increment_max,
        "hma_bridge_1900_degC": float(hma_bridge[0]),
        "hma_bridge_1939_degC": float(hma_bridge[-2]),
        "hma_anchor_1940_degC": float(hma_bridge[-1]),
        "gates": gates, "passes_all_gates": pass_all,
        "decision": "freeze_correction_only_bridge_for_future_blind_application" if pass_all else "reject_correction_bridge",
        "production_authorized": False, "oggm_forcing_authorized": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "facts_snapshot_sha256": facts_snapshot}, indent=2), encoding="utf-8"
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Correction-Only Bridge\n\n"
        f"- 1900 HMA correction: `{summary['hma_bridge_1900_degC']:+.3f} degC`.\n"
        f"- 1939 HMA correction: `{summary['hma_bridge_1939_degC']:+.3f} degC`.\n"
        f"- 1940 anchor: `{summary['hma_anchor_1940_degC']:+.3f} degC`.\n"
        f"- Maximum overshoot: `{max_overshoot:.3e} degC`.\n"
        f"- Maximum annual increment: `{increment_max:.3e} degC`.\n"
        f"- All gates pass: `{str(pass_all).lower()}`.\n\n"
        "Only a correction field was frozen. No locked-period climate value was read or generated.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("A numerical source changed")
    print(json.dumps(summary, indent=2))
    print(output_nc)
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
