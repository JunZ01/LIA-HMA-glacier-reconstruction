"""Preflight a literature-supported multimodel evolving-prior temperature method.

This script performs metadata and algebra checks only. It does not assimilate
proxies, read the sealed 1921-1939 interval, or modify production sources.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import h5py
import numpy as np
from netCDF4 import Dataset


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
OUTPUT = LOOP / "temperature_multimodel_evolving_prior_method_preflight_v1"
DRIVER = Path(__file__).resolve()

ARCHIVE = Path(r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all")
TEMPERATURE_PRIOR = RESULTS / (
    "enkf_reconstruction_source_recovery_audit_v2/"
    "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy"
)
PRECIPITATION_PRIOR = RESULTS / (
    "climate_reconstruction_precipitation_prior_recovery_preflight_v1/"
    "precipitation_prior_336x848_mm_day.mat"
)
INDICES = ARCHIVE / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat"
PROXY_PATHS = {
    6: ARCHIVE / "python_data/proxy_data_6.mat",
    9: ARCHIVE / "python_data/proxy_data_9.mat",
}
ACQUIRED_TEMPERATURE = RESULTS / (
    "climate_reconstruction_temperature_pmip_equal_blend_annual_spatial_candidate_v1/"
    "temperature_equal_blend_annual_spatial_candidate_1500_1899.nc"
)
PMIP_TEMPERATURE_CATALOG = RESULTS / (
    "climate_reconstruction_temperature_pmip_multimodel_prior_audit_v2/"
    "remote_source_catalog.json"
)
FACTS = ROOT / "LIA_PROJECT_FACTS.md"

ESGF_ENDPOINT = "https://esgf-node.llnl.gov/esg-search/search/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
    }


def decode_char(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1, order="F")
    return "".join(chr(int(value)) for value in values if int(value))


def proxy_inventory(path: Path) -> dict[str, object]:
    methods: Counter[str] = Counter()
    maximum_closure = 0.0
    all_component_lengths: set[int] = set()
    with h5py.File(path, "r") as archive:
        refs = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        for ref in refs:
            proxy = archive[ref]
            method = decode_char(proxy["psm_mthd"])
            methods[method] += 1
            b = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
            mt = np.asarray(proxy["M_t"], dtype=float).reshape(-1, order="F")
            mp = np.asarray(proxy["M_p"], dtype=float).reshape(-1, order="F")
            hxb = np.asarray(proxy["HXb"], dtype=float).reshape(-1, order="F")
            if b.size != 3 or not (mt.size == mp.size == hxb.size):
                raise RuntimeError("Unexpected archived bivariate PSM structure")
            rebuilt = b[0] + b[1] * mt + b[2] * mp
            maximum_closure = max(maximum_closure, float(np.max(np.abs(rebuilt - hxb))))
            all_component_lengths.add(int(hxb.size))
    return {
        "proxy_count": int(sum(methods.values())),
        "psm_method_counts": dict(methods),
        "all_psms_require_paired_temperature_and_precipitation": bool(
            set(methods).issubset({"bi", "bivariate"})
        ),
        "component_member_lengths": sorted(all_component_lengths),
        "maximum_hxb_rebuild_error": maximum_closure,
    }


def esgf_query(model: str, activity: str, experiment: str, variant: str) -> dict[str, object]:
    params = {
        "type": "File",
        "project": "CMIP6",
        "activity_id": activity,
        "source_id": model,
        "experiment_id": experiment,
        "variant_label": variant,
        "variable_id": "pr",
        "table_id": "Amon",
        "format": "application/solr+json",
        "limit": 3,
    }
    url = ESGF_ENDPOINT + "?" + urlencode(params)
    with urlopen(url, timeout=90) as response:
        payload = json.load(response)
    docs = payload["response"]["docs"]
    return {
        "query_url": url,
        "num_found_including_replicas": int(payload["response"]["numFound"]),
        "sample_ids": [str(item["id"]) for item in docs],
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(OUTPUT / "summary.json")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    sources = {
        "facts": FACTS,
        "temperature_prior": TEMPERATURE_PRIOR,
        "precipitation_prior": PRECIPITATION_PRIOR,
        "indices": INDICES,
        "proxy6": PROXY_PATHS[6],
        "proxy9": PROXY_PATHS[9],
        "acquired_temperature_fields": ACQUIRED_TEMPERATURE,
        "pmip_temperature_catalog": PMIP_TEMPERATURE_CATALOG,
        "driver": DRIVER,
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: state(path) for name, path in sources.items()}

    temperature_prior = np.load(TEMPERATURE_PRIOR, mmap_mode="r")
    with h5py.File(PRECIPITATION_PRIOR, "r") as archive:
        precipitation_shape = tuple(
            int(value) for value in archive["precipitation_prior_336x848_mm_day"].shape
        )
        precipitation_years = np.asarray(archive["member_years"], dtype=int).reshape(-1)
        precipitation_rng_seed = int(np.asarray(archive["rng_seed"]).reshape(-1)[0])
    with h5py.File(INDICES, "r") as archive:
        index_shape = tuple(int(value) for value in archive["all_Xb_indices"].shape)

    proxy = {str(network): proxy_inventory(path) for network, path in PROXY_PATHS.items()}
    maximum_hxb_closure = max(item["maximum_hxb_rebuild_error"] for item in proxy.values())

    with Dataset(ACQUIRED_TEMPERATURE) as dataset:
        acquired_variables = sorted(dataset.variables)
    acquired_has_precipitation = any(
        name.lower() in {"pr", "precip", "precipitation"} for name in acquired_variables
    )
    temperature_catalog = json.loads(PMIP_TEMPERATURE_CATALOG.read_text(encoding="utf-8"))
    catalog_models = sorted({str(item["model"]) for item in temperature_catalog})

    remote_precipitation = {
        "MPI-ESM1-2-LR_past2k": esgf_query(
            "MPI-ESM1-2-LR", "PMIP", "past2k", "r1i1p1f1"
        ),
        "MPI-ESM1-2-LR_historical": esgf_query(
            "MPI-ESM1-2-LR", "CMIP", "historical", "r1i1p1f1"
        ),
        "MIROC-ES2L_past1000": esgf_query(
            "MIROC-ES2L", "PMIP", "past1000", "r1i1p1f2"
        ),
        "MIROC-ES2L_historical": esgf_query(
            "MIROC-ES2L", "CMIP", "historical", "r1i1p1f2"
        ),
    }
    all_esgf_precipitation_available = all(
        item["num_found_including_replicas"] > 0 for item in remote_precipitation.values()
    )

    rows = [
        {
            "model_family": "CESM-LME010",
            "role": "legacy_model_prior_and_required_control",
            "temperature_status": "local_336x848_prior",
            "precipitation_status": "local_336x848_prior",
            "paired_member_status": "archived_M_t_M_p_HXb_pairing_closes",
            "production_ready": False,
            "remaining_action": "recover_time_evolving_forced_mean_and_rebuild_new_model_specific_HXb",
        },
        {
            "model_family": "ModE-Sim",
            "role": "independent_forced_AGCM_ensemble",
            "temperature_status": "local_annual_spatial_subset",
            "precipitation_status": "officially_documented_not_locally_acquired",
            "paired_member_status": "not_yet_built",
            "production_ready": False,
            "remaining_action": "acquire_paired_precipitation_for_identical_set_member_years",
        },
        {
            "model_family": "MPI-ESM1-2-LR",
            "role": "independent_PMIP4_coupled_model",
            "temperature_status": "local_annual_spatial_subset",
            "precipitation_status": "official_ESGF_files_confirmed",
            "paired_member_status": "not_yet_built",
            "production_ready": False,
            "remaining_action": "acquire_pr_for_exact_past2k_historical_member_and_regrid",
        },
        {
            "model_family": "MIROC-ES2L",
            "role": "independent_PMIP4_coupled_model",
            "temperature_status": "local_annual_spatial_subset",
            "precipitation_status": "official_ESGF_files_confirmed",
            "paired_member_status": "not_yet_built",
            "production_ready": False,
            "remaining_action": "acquire_pr_for_exact_past1000_historical_member_and_regrid",
        },
    ]
    write_csv(OUTPUT / "model_variable_feasibility_matrix.csv", rows)

    method_contract = """# Multimodel Evolving-Prior Method Contract

## Scientific purpose

Replace the unsupported post-hoc 50:50 temperature shift with a prior-sensitive
paleoclimate data-assimilation experiment. The legacy CESM-LME010 prior remains
the control and one of four model families; it is not assigned half of the final
answer.

## Primary method

Run four separate ordinary-EnKF reconstructions using CESM-LME010, ModE-Sim,
MPI-ESM1-2-LR, and MIROC-ES2L. For each model, build paired annual temperature
and precipitation state members from the identical realization and year. Use a
time-evolving model-specific forced mean plus a balanced, detrended anomaly
library from that model. Apply every archived bivariate PSM to the new paired
temperature and precipitation members; archived HXb values must not be reused.

For model m and year t:

    Xb_m(t,k) = mu_m(t) + Z_m(k)
    HXb_i,m(t,k) = b0_i + bT_i * T_i,m(t,k) + bP_i * P_i,m(t,k)
    Xa_m(t) = EnKF(Xb_m(t), HXb_m(t), y(t), R)

The primary multimodel estimate is the equal mean of the four independently
assimilated posterior means. Their spread is structural prior uncertainty.
Equal weights are fixed before validation; no skill-fitted model weights are
allowed. Leave-one-model-out results are mandatory.

## Secondary sensitivity

Construct one pooled multimodel prior using exactly the same number of paired
states from each model. This branch tests covariance pooling only and cannot
replace the primary separate-assimilation ensemble after results are seen.

## Frozen choices

- Ordinary EnKF first; AO/HGAO strategies are out of scope until the prior test passes.
- Archived proxy observations, PSM coefficients, RegEM R, state domain, and no-localization rule.
- Annual temperature and precipitation predictors sampled from the same model/member/year.
- Cellwise anomalies relative to each model's own 1961-1990 climatology for observation comparison.
- Model anomaly libraries are centered and detrended within model before balancing.
- Exactly equal model member counts; ModE-Sim's larger ensemble cannot dominate.
- 1880-1920 full-amplitude diagnostic is retained as a comparison curve, not injected into the prior.
- 1921-1939 remains absent and unread.

## Explicitly prohibited

- Adding a correction to the archived posterior while freezing its EnKF increment.
- Reusing CESM-derived HXb with non-CESM state members.
- Selecting model weights or transition years from validation performance.
- Treating model members as independent observational evidence.
- Production replacement, OGGM forcing, or EnKF execution without user authorization.
"""
    (OUTPUT / "METHOD_CONTRACT.md").write_text(method_contract, encoding="utf-8")

    validation_contract = """# Validation Contract

## Stage A: paired-prior closure

- Every model supplies paired annual tas/pr for identical member-years.
- Unit, calendar, grid, baseline, and finite-support checks pass.
- Rebuilt archived CESM HXb closes to at most 1e-12 before any new model is used.
- Each model contributes the same number of prior states.
- No 1921-1939 value is accessed.

## Stage B: biased-model pseudoproxy test

Withhold each model in turn as truth and build the prior from the other models.
The multimodel method must improve HMA RMSE over the CESM control in at least
three of four withheld truths, improve at least 10 of 15 GTN-G regions, and
degrade no withheld-truth HMA RMSE by more than 5 percent.

## Stage C: allowed real-data test

- 1500-1879: at least three of four reference families improve in 50-year RMSE;
  no family may degrade by more than 5 percent.
- 1880-1900 and 1901-1920: qcf direct-station absolute bias must not worsen and
  median station MAE must improve by at least 10 percent in each period.
- 1940-2000: blocked out-of-fold median regional skill must remain positive and
  no regional-product path may degrade by more than 5 percent.
- Direction, amplitude, regional pattern, and trend are reported separately.

Passing all stages authorizes only a proposal for one frozen full six-member
reconstruction. It does not authorize production replacement or OGGM.
"""
    (OUTPUT / "VALIDATION_CONTRACT.md").write_text(validation_contract, encoding="utf-8")

    gates = {
        "legacy_temperature_prior_shape_336x848": tuple(temperature_prior.shape) == (336, 848),
        "legacy_precipitation_prior_shape_336x848": precipitation_shape == (848, 336),
        "legacy_precipitation_member_years_1001_1848": np.array_equal(
            precipitation_years, np.arange(1001, 1849)
        ),
        "legacy_rng_seed_4236": precipitation_rng_seed == 4236,
        "archived_index_shape_50x500": index_shape in {(50, 500), (500, 50)},
        "all_temperature_psms_are_bivariate": all(
            item["all_psms_require_paired_temperature_and_precipitation"]
            for item in proxy.values()
        ),
        "archived_hxb_rebuild_error_at_most_1e_12": maximum_hxb_closure <= 1e-12,
        "acquired_external_candidate_is_temperature_only": not acquired_has_precipitation,
        "catalog_contains_mpi_and_miroc_temperature": catalog_models == [
            "MIROC-ES2L", "MPI-ESM1-2-LR"
        ],
        "esgf_mpi_miroc_past_and_historical_pr_available": all_esgf_precipitation_available,
    }
    method_feasible = bool(all(gates.values()))
    paired_external_prior_ready = False
    summary = {
        "status": "complete",
        "decision": "adopt_separate_assimilation_multimodel_evolving_prior_method_for_data_build_preflight",
        "method_feasible": method_feasible,
        "paired_external_prior_ready": paired_external_prior_ready,
        "temperature_prior_shape": list(temperature_prior.shape),
        "precipitation_prior_stored_shape": list(precipitation_shape),
        "proxy_networks": proxy,
        "maximum_archived_hxb_rebuild_error": maximum_hxb_closure,
        "acquired_external_temperature_variables": acquired_variables,
        "remote_precipitation_queries": remote_precipitation,
        "gates": gates,
        "next_action": "acquire_paired_pr_for_mode_sim_mpi_miroc_and_recover_cesm_time_evolving_mean_without_running_enkf",
        "locked_1921_1939_values_accessed": 0,
        "enkf_run": False,
        "oggm_run": False,
        "production_modified": False,
    }
    after = {name: state(path) for name, path in sources.items()}
    summary["all_declared_sources_unchanged"] = before == after
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8"
    )
    final = f"""# Multimodel Evolving-Prior Method Preflight

- Method feasible: `{str(method_feasible).lower()}`.
- Paired external prior ready: `false`.
- Temperature prior: `{temperature_prior.shape[0]} x {temperature_prior.shape[1]}`.
- Precipitation prior stored shape: `{precipitation_shape[0]} x {precipitation_shape[1]}`.
- Proxy6/Proxy9 counts: `{proxy['6']['proxy_count']}` / `{proxy['9']['proxy_count']}`.
- All archived temperature PSMs require paired temperature and precipitation: `true`.
- Maximum archived `HXb = b0 + bT*M_t + bP*M_p` error: `{maximum_hxb_closure:.3e}`.
- MPI and MIROC paired past/historical precipitation files exist in official ESGF: `{str(all_esgf_precipitation_available).lower()}`.
- Current external spatial artifact contains precipitation: `{str(acquired_has_precipitation).lower()}`.

Decision: use an ensemble of four separately assimilated model priors as the
primary supported method, with a balanced pooled multimodel prior only as a
secondary sensitivity. Do not use post-hoc 50:50 or frozen-increment correction.

The next non-EnKF step is paired precipitation acquisition and model-specific
state/HXb construction. No 1921-1939 value was accessed, and no production,
EnKF, or OGGM file was changed or run.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(final, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
