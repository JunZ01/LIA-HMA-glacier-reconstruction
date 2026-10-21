from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


DRIVER = Path(__file__).resolve()
DRIVER_DIR = DRIVER.parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))

import run_temperature_four_prior_strategy_c_replay_1901_1920_v1 as replay


ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
PRIOR = LOOP / (
    "temperature_balanced_pooled_candidate_b2_allowed_input_preflight_v3/"
    "candidate_b2_stage_c_prior_input_v3.npz"
)
STAGE_B = LOOP / (
    "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_preregistration_v2/"
    "stage_b_pseudoproxy_input_v1.npz"
)
BASE_CONTRACT = LOOP / (
    "temperature_balanced_pooled_candidate_b2_stage_c_preregistration_v1/"
    "CANDIDATE_B2_STAGE_C_CONTRACT.md"
)
CENTERING_CONTRACT = DRIVER_DIR / "temperature_candidate_b2_stage_c_observation_centering_v1_contract.md"
OUT = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v1"
PACKAGE = OUT / "candidate_b2_stage_c_static_input_v1.npz"
EXPECTED_PRIOR_SHA256 = "DE59F318E17BE328040897533D475467A6F26D667B2200F11998BFC7FB0C2EC4"
ALLOWED_YEARS = np.r_[np.arange(1500, 1921), np.arange(1940, 2001)].astype(np.int32)
POOL_SLICE = slice(499, 848)
NETWORKS = (6, 9)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def matlab_text(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1, order="F")
    return "".join(chr(int(value)) for value in values if int(value))


def proxy_static(network: int, stage: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    path = replay.prior_tools.ARCHIVE / f"python_data/proxy_data_{network}.mat"
    selected = np.asarray(stage[f"proxy{network}_record_indices_0based"], dtype=np.int32)
    centers = []
    locations = []
    coefficients = []
    ptypes = []
    methods = []
    with h5py.File(path, "r") as archive:
        references = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        for record_index in selected:
            proxy = archive[references[int(record_index)]]
            # Deliberately do not touch proxy['data']; this package precedes real-data authorization.
            hxb = np.asarray(proxy["HXb"], dtype=float).reshape(-1, order="F")
            if hxb.shape != (848,):
                raise RuntimeError(f"unexpected-HXb-shape:proxy{network}:record{record_index + 1}")
            pool = hxb[POOL_SLICE]
            if pool.shape != (349,) or not np.all(np.isfinite(pool)):
                raise RuntimeError(f"invalid-HXb-pool:proxy{network}:record{record_index + 1}")
            b = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
            location = np.asarray(proxy["location"], dtype=float).reshape(-1, order="F")
            if b.shape != (3,) or location.shape != (2,):
                raise RuntimeError(f"invalid-static-metadata:proxy{network}:record{record_index + 1}")
            centers.append(float(np.mean(pool)))
            locations.append(location)
            coefficients.append(b)
            ptypes.append(matlab_text(proxy["ptype"]))
            methods.append(matlab_text(proxy["psm_mthd"]))
    archived_locations = np.asarray(locations, dtype=float)
    frozen_locations = np.asarray(stage[f"proxy{network}_locations"], dtype=float)
    if not np.array_equal(archived_locations, frozen_locations):
        raise RuntimeError(f"proxy-location-mismatch:{network}")
    return {
        f"proxy{network}_record_indices_0based": selected,
        f"proxy{network}_locations": archived_locations,
        f"proxy{network}_psm_b0_b1_b2": np.asarray(coefficients, dtype=float),
        f"proxy{network}_archived_hxb_mean_1500_1848": np.asarray(centers, dtype=float),
        f"proxy{network}_ptype": np.asarray(ptypes, dtype="U64"),
        f"proxy{network}_psm_method": np.asarray(methods, dtype="U64"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    proxy_paths = {
        network: replay.prior_tools.ARCHIVE / f"python_data/proxy_data_{network}.mat"
        for network in NETWORKS
    }
    sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "execution_contract": ROOT / "LIA_EXECUTION_CONTRACT.md",
        "prior": PRIOR,
        "stage_b": STAGE_B,
        "base_contract": BASE_CONTRACT,
        "centering_contract": CENTERING_CONTRACT,
        "proxy6_archive": proxy_paths[6],
        "proxy9_archive": proxy_paths[9],
        "driver": DRIVER,
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    hashes_before = {name: sha256(path) for name, path in sources.items()}
    if hashes_before["prior"] != EXPECTED_PRIOR_SHA256:
        raise RuntimeError(f"prior-hash-mismatch:{hashes_before['prior']}")

    with np.load(PRIOR, allow_pickle=False) as prior, np.load(STAGE_B, allow_pickle=False) as stage:
        years = np.asarray(prior["years"], dtype=np.int32)
        if not np.array_equal(years, ALLOWED_YEARS):
            raise RuntimeError("allowed-year-axis")
        if np.any((years >= 1921) & (years <= 1939)):
            raise RuntimeError("sealed-years-present")
        arrays = {name: np.asarray(prior[name]) for name in prior.files}
        arrays.update(
            {
                "retained_state_indices_0based": np.asarray(stage["retained_state_indices"], dtype=np.int32),
                "retained_state_latitude": np.asarray(stage["retained_state_latitude"], dtype=float),
                "retained_state_longitude": np.asarray(stage["retained_state_longitude"], dtype=float),
                "gtng_region_codes": np.asarray(stage["gtng_region_codes"]).astype("U5"),
                "gtng_region_weights": np.asarray(stage["gtng_region_weights"], dtype=float),
                "hma_weights": np.asarray(stage["hma_weights"], dtype=float),
            }
        )
        for network in NETWORKS:
            arrays.update(proxy_static(network, stage))
            expected_count = 201 if network == 6 else 290
            if arrays[f"proxy{network}_record_indices_0based"].shape != (expected_count,):
                raise RuntimeError(f"proxy-count:{network}")
            if arrays[f"proxy{network}_archived_hxb_mean_1500_1848"].shape != (expected_count,):
                raise RuntimeError(f"proxy-center-count:{network}")

    np.savez_compressed(PACKAGE, **arrays)
    package_hash = sha256(PACKAGE)
    hashes_after = {name: sha256(path) for name, path in sources.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("source-hash-changed")

    gates = {
        "prior_hash_exact": True,
        "allowed_year_axis_exact": True,
        "sealed_year_absent": True,
        "proxy6_selected_count_201": True,
        "proxy9_selected_count_290": True,
        "operator_centers_finite": bool(
            np.all(np.isfinite(arrays["proxy6_archived_hxb_mean_1500_1848"]))
            and np.all(np.isfinite(arrays["proxy9_archived_hxb_mean_1500_1848"]))
        ),
        "real_proxy_data_dataset_read": False,
        "stage_c_enkf_run": False,
    }
    summary = {
        "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "static_package_pass": all(value for key, value in gates.items() if key != "real_proxy_data_dataset_read")
        and not gates["real_proxy_data_dataset_read"],
        "gates": gates,
        "observation_centering": "raw proxy value minus archived CESM HXb mean over 1500-1848",
        "datasets_read_from_proxy_records": ["HXb", "b", "location", "ptype", "psm_mthd"],
        "datasets_not_read_from_proxy_records": ["data", "age_old", "age_yng"],
        "package": str(PACKAGE),
        "package_sha256": package_hash,
        "prior_sha256": hashes_before["prior"],
        "real_proxy_observations_read": False,
        "stage_c_enkf_run": False,
        "sealed_1921_1939_values_used": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
        "production_modified": False,
        "oggm_run": False,
        "next_action": "run synthetic end-to-end Stage C closure and build authorization-gated launcher",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        name: {"path": str(path), "sha256": hashes_before[name]}
        for name, path in sources.items()
    }
    manifest["static_package"] = {"path": str(PACKAGE), "sha256": package_hash}
    manifest["proxy_record_datasets_read"] = summary["datasets_read_from_proxy_records"]
    manifest["real_proxy_data_dataset_read"] = False
    manifest["prohibited_source_accessed"] = False
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "AUTHORIZATION_REQUIRED.md").write_text(
        "# Stage C authorization boundary\n\n"
        "This static package contains no real proxy observations and performs no EnKF. "
        "The real-data driver must not run until the user explicitly authorizes "
        "`Candidate B2 Stage C real-proxy EnKF`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
