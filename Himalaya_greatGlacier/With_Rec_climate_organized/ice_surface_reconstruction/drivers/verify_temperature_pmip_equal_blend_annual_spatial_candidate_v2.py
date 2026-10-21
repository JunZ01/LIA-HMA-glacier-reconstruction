"""Independently verify the stored provisional annual temperature candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

import preflight_temperature_pmip_equal_blend_spatial_field_v1 as base


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
BUILD = RESULTS / "climate_reconstruction_temperature_pmip_equal_blend_annual_spatial_candidate_v1"
CANDIDATE = BUILD / "temperature_equal_blend_annual_spatial_candidate_1500_1899.nc"
BUILD_MANIFEST = BUILD / "source_manifest.json"
BUILD_SUMMARY = BUILD / "summary.json"
PREFLIGHT_FIELDS = RESULTS / (
    "climate_reconstruction_temperature_pmip_equal_blend_spatial_common_support_v1/"
    "equal_blend_common_support_50year_fields.npz"
)
REGIONAL_ANNUAL = RESULTS / (
    "climate_reconstruction_temperature_pmip_equal_blend_gtng15_v1/"
    "equal_blend_gtng15_annual_curves_1500_1899.csv"
)
ASSIGNMENT = base.ASSIGNMENT
SUPPORT = base.SUPPORT
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
OUTPUT = RESULTS / "climate_reconstruction_temperature_pmip_equal_blend_annual_spatial_candidate_verification_v2"
NUMERICAL_SOURCE_KEYS = [
    "reference", "catalog", "mode_sim", "assignment", "support", "regional_annual",
    "preflight_fields", "preflight_summary", "prior", "indices", "proxy6", "proxy9",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def domain_mean(values: np.ndarray, point_mask: np.ndarray, point_lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(point_lat[point_mask]))
    return np.sum(values[:, point_mask] * weights[None], axis=1) / np.sum(weights)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Annual Candidate Independent Verification Contract\n\n"
        "Read the stored v1 NetCDF only; do not repeat reconstruction. Verify that every numerical "
        "source still matches its pre-build fingerprint, years are exactly 1500-1899, the stored "
        "blend closes within 2e-6 degC (float32 tolerance), the eight 50-year fields close within "
        "2e-6 degC, GTN-G annual aggregates close within 2e-6 degC, and HMA annual aggregates differ "
        "from the accepted curves by no more than 0.005 degC. Record the current facts-file hash "
        "separately because it is a mutable project ledger, not a numerical temperature input. "
        "No 1901-1939 value is read. Passing retains only a provisional diagnostic candidate.\n",
        encoding="utf-8",
    )
    build_manifest = json.loads(BUILD_MANIFEST.read_text(encoding="utf-8"))
    build_summary = json.loads(BUILD_SUMMARY.read_text(encoding="utf-8"))
    expected = build_manifest["before"]
    numerical_paths = {
        "reference": base.REFERENCE, "catalog": base.CATALOG, "mode_sim": base.MODE_SIM,
        "assignment": ASSIGNMENT, "support": SUPPORT, "regional_annual": REGIONAL_ANNUAL,
        "preflight_fields": PREFLIGHT_FIELDS,
        "preflight_summary": RESULTS / "climate_reconstruction_temperature_pmip_equal_blend_spatial_common_support_v1/summary.json",
        "prior": base.regional_base.SOURCE_RECOVERY / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": base.regional_base.ARCHIVE / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy6": base.regional_base.ARCHIVE / "python_data/proxy_data_6.mat",
        "proxy9": base.regional_base.ARCHIVE / "python_data/proxy_data_9.mat",
    }
    before = {name: sha256(path) for name, path in numerical_paths.items()}
    numerical_sources_match_build = bool(all(before[name] == expected[name] for name in NUMERICAL_SOURCE_KEYS))
    facts_hash = sha256(FACTS)

    with Dataset(CANDIDATE) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        hma_mask = np.asarray(dataset.variables["hma_mask"][:], dtype=bool)
        common_grid = np.asarray(dataset.variables["common_support_mask"][:], dtype=bool)
        common = common_grid[hma_mask]
        fields = {
            "reference": np.asarray(dataset.variables["tas_reference_family_balanced"][:], dtype=float)[:, hma_mask],
            "background": np.asarray(dataset.variables["tas_background_six_member"][:], dtype=float)[:, hma_mask],
            "forced_mean": np.asarray(dataset.variables["tas_forced_equal_three_model_mean"][:], dtype=float)[:, hma_mask],
            "candidate": np.asarray(dataset.variables["tas_equal_half_candidate"][:], dtype=float)[:, hma_mask],
        }
    exact_years = bool(np.array_equal(years, np.arange(1500, 1900)))
    no_locked_years = bool(not np.any((years >= 1901) & (years <= 1939)))
    blend_closure = float(np.nanmax(np.abs(fields["candidate"] - 0.5 * fields["background"] - 0.5 * fields["forced_mean"])))

    preflight = np.load(PREFLIGHT_FIELDS)
    support_identical = bool(np.array_equal(common, np.asarray(preflight["common_support"], dtype=bool)))
    window_closure = 0.0
    for name, values in fields.items():
        windows = np.stack([
            np.nanmean(values[(years >= start) & (years <= end)], axis=0)
            for start, end in base.WINDOWS
        ])
        window_closure = max(
            window_closure,
            float(np.nanmax(np.abs(windows - np.asarray(preflight[name], dtype=float)))),
        )

    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    point_lat = lat2[hma_mask]
    assignment = pd.read_csv(ASSIGNMENT)
    support = pd.read_csv(SUPPORT).sort_values("o2region")
    domain_masks = {"hma": np.ones(int(hma_mask.sum()), dtype=bool)}
    for region in support.itertuples(index=False):
        grid_mask = np.zeros_like(hma_mask)
        cells = assignment[assignment.o2region.eq(region.o2region)]
        grid_mask[cells.grid_row.to_numpy(int), cells.grid_col.to_numpy(int)] = True
        domain_masks[region.o2region] = grid_mask[hma_mask]
    accepted = pd.read_csv(REGIONAL_ANNUAL)
    max_gtng_closure = 0.0
    max_hma_change = 0.0
    for domain, raw_mask in domain_masks.items():
        mask = raw_mask & common
        rows = accepted[accepted.domain.eq(domain)].sort_values("year")
        for name, column in [
            ("reference", "reference_four_family_degC"),
            ("background", "background_six_member_degC"),
            ("forced_mean", "forced_mean"),
            ("candidate", "equal_half_forced_mean"),
        ]:
            difference = domain_mean(fields[name], mask, point_lat) - rows[column].to_numpy(float)
            maximum = float(np.max(np.abs(difference)))
            if domain == "hma":
                max_hma_change = max(max_hma_change, maximum)
            else:
                max_gtng_closure = max(max_gtng_closure, maximum)

    gates = {
        "numerical_sources_match_prebuild_fingerprints": numerical_sources_match_build,
        "years_are_exactly_1500_1899": exact_years,
        "locked_years_absent": no_locked_years,
        "common_support_matches_preflight": support_identical,
        "stored_blend_closure_at_most_2e_6_degC": bool(blend_closure <= 2e-6),
        "eight_window_closure_at_most_2e_6_degC": bool(window_closure <= 2e-6),
        "gtng_annual_closure_at_most_2e_6_degC": bool(max_gtng_closure <= 2e-6),
        "hma_annual_change_at_most_0p005_degC": bool(max_hma_change <= 0.005),
        "v1_scientific_gates_passed": bool(build_summary["passes_all_gates"]),
    }
    after = {name: sha256(path) for name, path in numerical_paths.items()}
    numerical_sources_unchanged_during_verification = bool(before == after)
    gates["numerical_sources_unchanged_during_verification"] = numerical_sources_unchanged_during_verification
    pass_all = bool(all(gates.values()))
    summary = {
        "status": "complete", "candidate_sha256": sha256(CANDIDATE),
        "facts_snapshot_sha256": facts_hash,
        "locked_1901_1939_values_read": 0,
        "stored_blend_closure_max_degC": blend_closure,
        "eight_window_closure_max_degC": window_closure,
        "gtng_annual_closure_max_degC": max_gtng_closure,
        "hma_annual_change_max_degC": max_hma_change,
        "gates": gates, "passes_all_gates": pass_all,
        "decision": "verify_and_retain_provisional_annual_spatial_candidate" if pass_all else "reject_stored_annual_candidate",
        "production_authorized": False, "oggm_forcing_authorized": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "facts_snapshot_sha256": facts_hash}, indent=2), encoding="utf-8"
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Annual Candidate Independent Verification\n\n"
        f"- Candidate SHA256: `{summary['candidate_sha256']}`.\n"
        f"- Stored blend closure: `{blend_closure:.3e} degC`.\n"
        f"- Eight-window closure: `{window_closure:.3e} degC`.\n"
        f"- GTN-G annual closure: `{max_gtng_closure:.3e} degC`.\n"
        f"- HMA annual change: `{max_hma_change:.3e} degC`.\n"
        f"- All verification gates pass: `{str(pass_all).lower()}`.\n\n"
        "The artifact is retained only as a provisional diagnostic candidate. Production and OGGM use remain unauthorized.\n",
        encoding="utf-8",
    )
    if not pass_all:
        raise RuntimeError("Stored annual candidate failed independent verification")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
