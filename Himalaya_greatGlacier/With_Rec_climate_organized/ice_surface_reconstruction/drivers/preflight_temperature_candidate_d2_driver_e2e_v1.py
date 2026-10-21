from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import run_temperature_candidate_d2_real_proxy_v1 as implementation
import run_temperature_candidate_d2_real_proxy_v2 as wrapper
import temperature_candidate_b2_stage_c_core_v1 as core


ROOT = Path(__file__).resolve().parents[4]
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUTPUT = LOOP / "temp_candidate_d2_driver_synthetic_e2e_v1"
STATIC = implementation.STATIC_PACKAGE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def synthetic_reader(
    network: int,
    package: np.lib.npyio.NpzFile,
    years: np.ndarray,
) -> tuple[np.ndarray, dict[str, int], Path]:
    if network != 6:
        raise RuntimeError("synthetic fixture permits proxy6 only")
    operator_mean = np.asarray(package["proxy6_balanced_operator_forced_mean"], dtype=float)
    operator_residual = np.asarray(package["proxy6_pooled_operator_residual_library"], dtype=float)
    proxy_pattern = operator_residual[217] - operator_residual.mean(axis=0)
    baseline = (years >= 1500) & (years <= 1848)
    signal_time = (years.astype(float) - 1674.0) / 220.0 + 0.24 * np.sin(2.0 * np.pi * (years - 1500.0) / 53.0)
    signal_time -= signal_time[baseline].mean()
    offsets = 1.6 * np.sin(np.arange(operator_mean.shape[1]) * 0.61)
    observations = operator_mean + signal_time[:, None] * proxy_pattern[None, :] + offsets[None, :]
    row = np.arange(len(years))[:, None]
    column = np.arange(operator_mean.shape[1])[None, :]
    mask = ((row + 5 * column) % 19) != 0
    mask[:, :5] = False
    mask[np.ix_(np.flatnonzero(baseline)[:20], np.arange(5))] = True
    observations[~mask] = np.nan
    audit = {
        "record_count": int(operator_mean.shape[1]),
        "age_metadata_cells_read": 0,
        "allowed_proxy_value_cells_read": 0,
        "sealed_proxy_value_cells_read": 0,
        "duplicate_allowed_year_assignments": 0,
        "synthetic_proxy_value_cells_generated": int(np.isfinite(observations).sum()),
    }
    return observations, audit, STATIC


def receipt(wrapper_hash: str | None = None, fingerprint: str | None = None, authorized: bool = True) -> dict[str, object]:
    return {
        "schema": implementation.AUTHORIZATION_SCHEMA,
        "authorized": authorized,
        "authorization_text": implementation.AUTHORIZATION_TEXT,
        "authorized_at": "SYNTHETIC_TEST_ONLY",
        "static_package_sha256": implementation.EXPECTED_STATIC_SHA256,
        "driver_sha256": core.sha256(implementation.DRIVER),
        "wrapper_sha256": wrapper_hash or core.sha256(wrapper.DRIVER),
        "launch_fingerprint_sha256": fingerprint or wrapper.EXPECTED_LAUNCH_FINGERPRINT,
        "synthetic_test_only": True,
    }


def array_hash(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        return core.canonical_array_hash({name: np.asarray(data[name]) for name in data.files})


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    receipt_path = OUTPUT / "synthetic_authorization_fixture.json"
    receipt_data = receipt()
    receipt_path.write_text(json.dumps(receipt_data, indent=2) + "\n", encoding="utf-8")

    wrapper.validate_wrapper_receipt(receipt_path)
    guard_results = {}
    for name, value in (
        ("wrong_wrapper_hash", receipt(wrapper_hash="0" * 64)),
        ("wrong_launch_fingerprint", receipt(fingerprint="F" * 64)),
        ("authorized_false", receipt(authorized=False)),
    ):
        path = OUTPUT / f"{name}.json"
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        try:
            wrapper.validate_wrapper_receipt(path)
        except PermissionError:
            guard_results[name] = True
        else:
            guard_results[name] = False

    original_authorization = implementation.validate_authorization
    original_reader = implementation.b2_reader.read_allowed_proxy_observations
    implementation.validate_authorization = lambda _: receipt_data
    implementation.b2_reader.read_allowed_proxy_observations = synthetic_reader
    try:
        first_output = OUTPUT / "driver_output_a"
        second_output = OUTPUT / "driver_output_b"
        implementation.main(receipt_path, first_output)
        implementation.main(receipt_path, second_output)
    finally:
        implementation.validate_authorization = original_authorization
        implementation.b2_reader.read_allowed_proxy_observations = original_reader

    first_npz = first_output / "candidate_d2_real_output_v1.npz"
    second_npz = second_output / "candidate_d2_real_output_v1.npz"
    first_summary = json.loads((first_output / "summary.json").read_text(encoding="utf-8"))
    with np.load(first_npz, allow_pickle=False) as result:
        years = np.asarray(result["years"], dtype=int)
        candidate = np.asarray(result["candidate_d2_temperature_anomaly"], dtype=float)
        background = np.asarray(result["balanced_background"], dtype=float)
        block_centers = np.asarray(result["block_centers"], dtype=float)
        identified = np.asarray(result["proxy6_identified_intercept"], dtype=bool)
        annual_increment = np.asarray(result["annual_interpolated_increment"], dtype=float)
    temporary_directories = [path.name for path in OUTPUT.glob(".temp_candidate_d2_real_v1_*")]

    gates = {
        "valid_wrapper_receipt_passes": True,
        "wrong_wrapper_hash_rejected": guard_results["wrong_wrapper_hash"],
        "wrong_launch_fingerprint_rejected": guard_results["wrong_launch_fingerprint"],
        "authorized_false_rejected": guard_results["authorized_false"],
        "two_runs_canonical_array_hash_equal": array_hash(first_npz) == array_hash(second_npz),
        "allowed_year_axis_exact": bool(np.array_equal(years, implementation.ALLOWED_YEARS)),
        "sealed_year_absent": not bool(np.any((years >= 1921) & (years <= 1939))),
        "candidate_all_finite": bool(np.all(np.isfinite(candidate))),
        "candidate_nontrivial": float(np.std(candidate - background)) > 1e-4,
        "block_count_24": len(block_centers) == 24,
        "five_underidentified_records_excluded": int((~identified).sum()) == 5,
        "annual_increment_all_finite": bool(np.all(np.isfinite(annual_increment))),
        "published_output_hash_matches": sha256(first_npz) == first_summary["output_sha256"],
        "no_real_proxy_values_read": first_summary["proxy_read_audit"]["allowed_proxy_value_cells_read"] == 0,
        "no_sealed_values_read": first_summary["proxy_read_audit"]["sealed_proxy_value_cells_read"] == 0,
        "no_temporary_publish_directory_left": len(temporary_directories) == 0,
    }
    passed = all(gates.values())
    summary = {
        "experiment": OUTPUT.name,
        "status": "pass" if passed else "fail",
        "synthetic_driver_e2e_pass": passed,
        "gates": gates,
        "canonical_array_hash": array_hash(first_npz),
        "driver_output_sha256_a": sha256(first_npz),
        "driver_output_sha256_b": sha256(second_npz),
        "guard_results": guard_results,
        "real_proxy_observations_read": False,
        "real_proxy_enkf_run": False,
        "sealed_1921_1939_values_used": False,
        "production_modified": False,
        "oggm_run": False,
        "next_action": "await explicit Candidate D2 real-proxy authorization" if passed else "repair driver before authorization",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Candidate D2 synthetic driver end-to-end preflight\n\n"
        f"Pass: `{passed}`. Canonical repeated-array hash: `{summary['canonical_array_hash']}`.\n",
        encoding="utf-8",
    )
    manifest = {
        "inputs": {
            str(STATIC): sha256(STATIC),
            str(implementation.DRIVER): sha256(implementation.DRIVER),
            str(wrapper.DRIVER): sha256(wrapper.DRIVER),
        },
        "outputs": {
            str(path.relative_to(OUTPUT)): sha256(path)
            for path in sorted(OUTPUT.rglob("*"))
            if path.is_file() and path.name != "source_manifest.json"
        },
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
