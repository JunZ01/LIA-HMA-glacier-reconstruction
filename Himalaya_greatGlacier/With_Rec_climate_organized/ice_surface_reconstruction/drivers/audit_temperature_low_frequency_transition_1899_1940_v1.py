"""Audit the allowed post-1940 anchor for a future low-frequency bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import preflight_temperature_pmip_equal_blend_spatial_field_v1 as base


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
LOW_FREQUENCY = RESULTS / "climate_reconstruction_temperature_pmip_constrained_low_frequency_correction_v1"
LOW_CURVES = LOW_FREQUENCY / "low_frequency_domain_curves.csv"
LOW_SUMMARY = LOW_FREQUENCY / "summary.json"
REFERENCE = base.REFERENCE
ASSIGNMENT = base.ASSIGNMENT
SUPPORT = base.SUPPORT
PRIOR = base.regional_base.SOURCE_RECOVERY / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy"
INDICES = base.regional_base.ARCHIVE / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat"
PROXY6 = base.regional_base.ARCHIVE / "python_data/proxy_data_6.mat"
PROXY9 = base.regional_base.ARCHIVE / "python_data/proxy_data_9.mat"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
OUTPUT = RESULTS / "climate_reconstruction_temperature_low_frequency_transition_1899_1940_audit_v1"
YEARS = np.arange(1940, 2001)
KEY_REGIONS = ["13-08", "14-03", "15-01", "15-02"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recover_background(lat: np.ndarray, lon: np.ndarray, mask: np.ndarray) -> np.ndarray:
    prior = np.load(PRIOR)
    with h5py.File(INDICES, "r") as handle:
        experiment_indices = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    adaptive = {}
    for network, proxy_path in [(6, PROXY6), (9, PROXY9)]:
        hxb, observations, locked_reads = base.round25.read_proxy_allowed(proxy_path, YEARS)
        if locked_reads != 0:
            raise RuntimeError("Locked proxy values read")
        adaptive[network], _ = base.round25.ao_background_mean(
            prior, experiment_indices, hxb, observations
        )
    fixed_one = np.mean(
        np.stack([np.mean(prior[:, pool], axis=1) for pool in experiment_indices]), axis=0
    )
    fixed = np.repeat(fixed_one[None, :], len(YEARS), axis=0)
    native_lat, native_lon = base.round25.native_coordinates()
    members = []
    for method, network in base.round25.MEMBERS.values():
        flat = fixed if method == "O" else adaptive[network]
        anomaly = base.round25.anomaly_field(flat, YEARS)
        members.append(base.round25.interpolate_fields(anomaly, native_lat, native_lon, lat, lon))
    return np.nanmean(np.stack(members), axis=0)[:, mask]


def domain_mean(values: np.ndarray, point_mask: np.ndarray, point_lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(point_lat[point_mask]))
    return np.sum(values[:, point_mask] * weights[None], axis=1) / np.sum(weights)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# 1899-to-1940 Transition Anchor Audit\n\n"
        "Read only 1940-2000 from the current background and family-balanced reference. Quantify the "
        "reference-minus-background correction needed in 1940-1959, 1961-1990, and 1991-2000, and "
        "compare it with the accepted 1899 constrained correction. A future bridge design may advance "
        "only if at least 12/15 regions preserve the 1899 correction sign into 1940-1959 and every "
        "key-region change is <=0.25 degC. This audit does not read or generate corrected climate "
        "values for 1901-1939 and does not authorize production or OGGM forcing.\n",
        encoding="utf-8",
    )
    paths = {
        "low_curves": LOW_CURVES, "low_summary": LOW_SUMMARY,
        "reference": REFERENCE, "assignment": ASSIGNMENT, "support": SUPPORT,
        "prior": PRIOR, "indices": INDICES, "proxy6": PROXY6, "proxy9": PROXY9,
        "driver": DRIVER, "contract": contract,
    }
    before = {name: sha256(path) for name, path in paths.items()}
    facts_snapshot = sha256(FACTS)
    low_summary = json.loads(LOW_SUMMARY.read_text(encoding="utf-8"))
    if not low_summary["passes_all_gates"]:
        raise RuntimeError("Low-frequency candidate was not retained")

    with Dataset(REFERENCE) as dataset:
        all_years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        hma_mask = np.asarray(dataset.variables["hma_mask"][:], dtype=bool)
        indices = np.asarray([int(np.flatnonzero(all_years == year)[0]) for year in YEARS])
        reference = np.asarray(dataset.variables["tas_reference_family_balanced"][indices], dtype=float)[:, hma_mask]
    background = recover_background(lat, lon, hma_mask)
    low_curves = pd.read_csv(LOW_CURVES)
    common = np.all(np.isfinite(reference), axis=0) & np.all(np.isfinite(background), axis=0)
    lat2, _ = np.meshgrid(lat, lon, indexing="ij")
    point_lat = lat2[hma_mask]
    assignment = pd.read_csv(ASSIGNMENT)
    support = pd.read_csv(SUPPORT).sort_values("o2region")
    domain_masks = {"hma": np.ones(int(hma_mask.sum()), dtype=bool)}
    domain_names = {"hma": "High Mountain Asia"}
    for region in support.itertuples(index=False):
        grid_mask = np.zeros_like(hma_mask)
        cells = assignment[assignment.o2region.eq(region.o2region)]
        grid_mask[cells.grid_row.to_numpy(int), cells.grid_col.to_numpy(int)] = True
        domain_masks[region.o2region] = grid_mask[hma_mask]
        domain_names[region.o2region] = region.full_name

    metric_rows = []
    curve_rows = []
    for domain, raw_mask in domain_masks.items():
        mask = raw_mask & common
        ref_series = domain_mean(reference, mask, point_lat)
        bg_series = domain_mean(background, mask, point_lat)
        needed = ref_series - bg_series
        correction_1899 = float(
            low_curves[(low_curves.domain.eq(domain)) & (low_curves.year.eq(1899))].correction_degC.iloc[0]
        )
        mean_1940_1959 = float(np.mean(needed[(YEARS >= 1940) & (YEARS <= 1959)]))
        mean_1961_1990 = float(np.mean(needed[(YEARS >= 1961) & (YEARS <= 1990)]))
        mean_1991_2000 = float(np.mean(needed[(YEARS >= 1991) & (YEARS <= 2000)]))
        metric_rows.append({
            "domain": domain, "domain_name": domain_names[domain],
            "correction_1899_degC": correction_1899,
            "needed_1940_1959_mean_degC": mean_1940_1959,
            "needed_1961_1990_mean_degC": mean_1961_1990,
            "needed_1991_2000_mean_degC": mean_1991_2000,
            "change_1899_to_1940_1959_degC": mean_1940_1959 - correction_1899,
            "same_sign_1899_and_1940_1959": bool(np.sign(correction_1899) == np.sign(mean_1940_1959)),
        })
        for index, year in enumerate(YEARS):
            curve_rows.append({
                "domain": domain, "domain_name": domain_names[domain], "year": int(year),
                "reference_degC": float(ref_series[index]),
                "background_degC": float(bg_series[index]),
                "needed_correction_degC": float(needed[index]),
            })
    metrics = pd.DataFrame(metric_rows)
    curves = pd.DataFrame(curve_rows)
    regional = metrics[metrics.domain.ne("hma")]
    hma = metrics[metrics.domain.eq("hma")].iloc[0]
    key = metrics[metrics.domain.isin(KEY_REGIONS)]
    gates = {
        "at_least_12_of_15_regions_preserve_correction_sign": bool(regional.same_sign_1899_and_1940_1959.sum() >= 12),
        "all_key_region_changes_at_most_0p25_degC": bool((key.change_1899_to_1940_1959_degC.abs() <= 0.25).all()),
    }
    pass_all = bool(all(gates.values()))
    metrics.to_csv(OUTPUT / "transition_anchor_domain_metrics.csv", index=False)
    curves.to_csv(OUTPUT / "transition_anchor_domain_curves_1940_2000.csv", index=False)

    plot_domains = ["hma", *KEY_REGIONS]
    fig, axes = plt.subplots(len(plot_domains), 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    for axis, domain in zip(axes, plot_domains, strict=True):
        rows = curves[curves.domain.eq(domain)].set_index("year")
        correction_1899 = float(metrics[metrics.domain.eq(domain)].correction_1899_degC.iloc[0])
        axis.plot(rows.index, rows.needed_correction_degC.rolling(11, center=True, min_periods=6).mean(), color="#0072B2", linewidth=1.8, label="Reference minus background (11y mean)")
        axis.scatter([1899], [correction_1899], color="#D55E00", s=30, label="Accepted 1899 correction")
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.set(title=domain_names[domain], ylabel="degC")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Year")
    figure = OUTPUT / "temperature_transition_anchor_1899_1940.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in paths.items()}
    unchanged = before == after
    summary = {
        "status": "complete", "locked_1901_1939_values_read": 0,
        "numerical_sources_unchanged": unchanged, "facts_snapshot_sha256": facts_snapshot,
        "hma_correction_1899_degC": float(hma.correction_1899_degC),
        "hma_needed_1940_1959_mean_degC": float(hma.needed_1940_1959_mean_degC),
        "hma_change_1899_to_1940_1959_degC": float(hma.change_1899_to_1940_1959_degC),
        "regional_same_sign_count": int(regional.same_sign_1899_and_1940_1959.sum()),
        "max_key_region_change_degC": float(key.change_1899_to_1940_1959_degC.abs().max()),
        "gates": gates, "passes_bridge_feasibility_gates": pass_all,
        "decision": "design_bridge_without_reading_locked_climate_values" if pass_all else "do_not_design_locked_period_bridge",
        "production_authorized": False, "oggm_forcing_authorized": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "facts_snapshot_sha256": facts_snapshot}, indent=2), encoding="utf-8"
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# 1899-to-1940 Transition Anchor Audit\n\n"
        f"- HMA correction in 1899: `{float(hma.correction_1899_degC):+.3f} degC`.\n"
        f"- Mean HMA correction needed in 1940-1959: `{float(hma.needed_1940_1959_mean_degC):+.3f} degC`.\n"
        f"- Change: `{float(hma.change_1899_to_1940_1959_degC):+.3f} degC`.\n"
        f"- Regions preserving sign: `{int(regional.same_sign_1899_and_1940_1959.sum())}/15`.\n"
        f"- Bridge feasibility gates pass: `{str(pass_all).lower()}`.\n\nDecision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("A numerical source changed")
    print(json.dumps(summary, indent=2))
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
