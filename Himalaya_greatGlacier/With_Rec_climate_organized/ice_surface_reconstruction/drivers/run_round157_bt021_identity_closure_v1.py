"""Safe allowed-period identity and scientific-role audit for NOAA BT021."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


YEARS = np.concatenate((np.arange(1000, 1921), np.arange(1940, 2001))).astype(int)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def read_bt021_allowed(path: Path) -> tuple[pd.DataFrame, int, int]:
    rows: list[dict[str, object]] = []
    locked_rows_skipped = 0
    outside_scope_rows_skipped = 0
    in_data = False
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if not in_data:
                if line == "age_CE\trsgi\tnumsamp":
                    in_data = True
                continue
            if not line or line.startswith("#"):
                continue
            first_separator = line.find("\t")
            if first_separator < 0:
                raise RuntimeError("unexpected-bt021-data-line")
            year = int(line[:first_separator])
            if 1921 <= year <= 1939:
                locked_rows_skipped += 1
                continue
            if not (1000 <= year <= 1920 or 1940 <= year <= 2000):
                outside_scope_rows_skipped += 1
                continue
            fields = line.split("\t")
            if len(fields) != 3:
                raise RuntimeError(f"unexpected-bt021-field-count:{year}:{len(fields)}")
            rows.append(
                {
                    "year": year,
                    "bt021_trsgi": float(fields[1]),
                    "bt021_sample_count": int(fields[2]),
                }
            )
    frame = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    if frame.year.duplicated().any():
        raise RuntimeError("duplicate-bt021-year")
    if ((frame.year >= 1921) & (frame.year <= 1939)).any():
        raise RuntimeError("locked-year-in-bt021-export")
    if locked_rows_skipped != 19:
        raise RuntimeError(f"unexpected-locked-row-count:{locked_rows_skipped}")
    return frame, locked_rows_skipped, outside_scope_rows_skipped


def align_series(frame: pd.DataFrame, value_column: str) -> np.ndarray:
    lookup = dict(zip(frame.year.astype(int), frame[value_column].astype(float)))
    return np.asarray([lookup.get(int(year), np.nan) for year in YEARS], dtype=float)


def main() -> int:
    root = project_root()
    driver_dir = Path(__file__).resolve().parent
    acquisition = (
        root
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_bt021_raw_chronology_acquisition_20260819_v1"
    )
    output = (
        root
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_bt021_identity_closure_20260819_v1"
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    contract = driver_dir / "round157_bt021_identity_closure_v1_contract.md"
    shutil.copyfile(contract, output / "ROUND_CONTRACT.md")
    round145 = load_module(
        "round157_identity_helpers",
        driver_dir / "run_round145_hengduan_haizi_sequence_identity_closure_v1.py",
    )
    base = load_module(
        "round157_proxy6_reader",
        driver_dir / "run_round134_netp_mjj_low_frequency_reconstruction_v2.py",
    )
    archive = load_module(
        "round157_archive_path", driver_dir / "audit_positive_b2_event_seasonal_psm_v1.py"
    )
    round25 = load_module(
        "round157_proxy9_reader",
        driver_dir
        / "run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1.py",
    )

    loop = (
        root
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    )
    manifest27 = loop / "round_27_temperature_innovation_gain_sign_decomposition_v1/source_manifest.json"
    proxy9_manifest = json.loads(manifest27.read_text(encoding="utf-8"))
    proxy9 = Path(proxy9_manifest["sources_before"]["proxy9"]["path"])
    paths = {
        "acquisition_manifest": acquisition / "acquisition_manifest.json",
        "bt021_noaa_template": acquisition / "raw_unopened/bt021-crn-noaa.txt",
        "bt021_noaa_crn": acquisition / "raw_unopened/bt021-noaa.crn",
        "bt021_tucson_crn": acquisition / "raw_unopened/bt021.crn",
        "bhutan2015_allowed": loop
        / "round_149_noaa_temperature_candidate_identity_v1/bhutan2015_allowed.csv",
        "proxy6": archive.archive_proxy_path(),
        "proxy6_metadata": root
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "proxy9": proxy9,
        "proxy9_metadata": root
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy_network_psm_audit_v2/proxy9_identity_and_psm_metadata.csv",
        "manifest27": manifest27,
        "contract": contract,
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: fingerprint(path) for name, path in paths.items()}

    acquisition_manifest = json.loads(paths["acquisition_manifest"].read_text(encoding="utf-8"))
    if acquisition_manifest.get("value_content_decoded") is not False:
        raise RuntimeError("acquisition-boundary-not-clean")
    expected_hashes = {
        item["name"]: item["sha256"] for item in acquisition_manifest["files"]
    }
    for key, filename in (
        ("bt021_noaa_template", "bt021-crn-noaa.txt"),
        ("bt021_noaa_crn", "bt021-noaa.crn"),
        ("bt021_tucson_crn", "bt021.crn"),
    ):
        if before[key]["sha256"] != expected_hashes[filename]:
            raise RuntimeError(f"acquisition-hash-conflict:{filename}")
    if before["proxy9"]["sha256"] != proxy9_manifest["sources_before"]["proxy9"]["sha256"]:
        raise RuntimeError("proxy9-hash-conflict")

    bt021, locked_rows, outside_rows = read_bt021_allowed(paths["bt021_noaa_template"])
    bt021.to_csv(output / "bt021_allowed.csv", index=False)
    bt021_values = align_series(bt021, "bt021_trsgi")

    proxy6_meta = pd.read_csv(paths["proxy6_metadata"])
    proxy6_ids = tuple(proxy6_meta.proxy_index.astype(int))
    proxy6_values = base.read_proxy_values(paths["proxy6"], proxy6_ids, YEARS)
    _, proxy9_values, proxy9_locked = round25.read_proxy_allowed(paths["proxy9"], YEARS)
    if proxy9_locked != 0:
        raise RuntimeError("proxy9-locked-read")
    proxy9_meta = pd.read_csv(paths["proxy9_metadata"])
    if proxy9_values.shape[0] != len(proxy9_meta):
        raise RuntimeError("proxy9-metadata-count-mismatch")

    comparisons: list[dict[str, object]] = []
    for network, matrix, metadata in (
        ("proxy6", proxy6_values.T, proxy6_meta),
        ("proxy9", proxy9_values, proxy9_meta),
    ):
        for index in range(matrix.shape[0]):
            item = round145.metrics(bt021_values, matrix[index])
            exact, near = round145.classify(item)
            meta = metadata.iloc[index]
            comparisons.append(
                {
                    "candidate": "bt021_trsgi",
                    "network": network,
                    "proxy_record_index_1based": index + 1,
                    "proxy_latitude": float(meta.latitude),
                    "proxy_longitude": float(meta.longitude),
                    **item,
                    "absolute_correlation": (
                        abs(item["correlation"])
                        if np.isfinite(item["correlation"])
                        else np.nan
                    ),
                    "exact_affine_alias": exact,
                    "near_exact_standardized_sequence": near,
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(
        output / "all_bt021_proxy_sequence_comparisons.csv", index=False
    )
    comparison_frame.sort_values(
        ["exact_affine_alias", "near_exact_standardized_sequence", "absolute_correlation"],
        ascending=[False, False, False],
    ).head(25).to_csv(output / "top_bt021_proxy_sequence_matches.csv", index=False)

    exact_alias_count = int(comparison_frame.exact_affine_alias.sum())
    near_alias_count = int(comparison_frame.near_exact_standardized_sequence.sum())
    bhutan = pd.read_csv(paths["bhutan2015_allowed"])
    if ((bhutan.year >= 1921) & (bhutan.year <= 1939)).any():
        raise RuntimeError("locked-year-in-bhutan-allowed")
    same_source_rows = []
    for column, label in (
        ("value_1", "bhutan2015_reconstructed_jja_temperature"),
        ("value_2", "bhutan2015_lowpass_reconstruction"),
    ):
        values = align_series(bhutan, column)
        item = round145.metrics(bt021_values, values)
        exact, near = round145.classify(item)
        same_source_rows.append(
            {
                "bt021_series": "bt021_trsgi",
                "published_series": label,
                **item,
                "exact_affine_derivation": exact,
                "near_exact_affine_derivation": near,
                "scientific_independence": False,
                "reason": "Bhutan 2015 explicitly derives from Ura-Dhur BT021 material",
            }
        )
    pd.DataFrame(same_source_rows).to_csv(
        output / "bt021_bhutan2015_same_source_relationships.csv", index=False
    )

    is_new = exact_alias_count == 0 and near_alias_count == 0
    decision = (
        "advance_bt021_as_new_east_fitting_candidate_to_modern_oof"
        if is_new
        else "quarantine_bt021_as_existing_network_alias"
    )
    roles = pd.DataFrame(
        [
            {
                "series": "bt021_trsgi",
                "role": "new_east_proxy_fitting_candidate" if is_new else "existing_proxy_alias",
                "may_enter_modern_oof": is_new,
                "may_be_independent_paleo_validation": False,
            },
            {
                "series": "bhutan2015_reconstructed_jja_temperature",
                "role": "same_source_derived_benchmark",
                "may_enter_modern_oof": False,
                "may_be_independent_paleo_validation": False,
            },
        ]
    )
    roles.to_csv(output / "bt021_scientific_roles.csv", index=False)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "experiment_id": "round157_bt021_identity_closure_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "allowed_first_year": int(bt021.year.min()),
        "allowed_last_year": int(bt021.year.max()),
        "allowed_value_count": int(len(bt021)),
        "locked_1921_1939_rows_skipped_before_value_conversion": locked_rows,
        "outside_scope_rows_skipped_before_value_conversion": outside_rows,
        "proxy6_record_count": int(proxy6_values.shape[1]),
        "proxy9_record_count": int(proxy9_values.shape[0]),
        "exact_affine_alias_count": exact_alias_count,
        "near_exact_standardized_sequence_count": near_alias_count,
        "same_source_relationships": same_source_rows,
        "decision": decision,
        "locked_1921_1939_values_accessed": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    (output / "FINAL_RESULT.md").write_text(
        "# BT021 Identity and Role Closure\n\n"
        f"- Allowed chronology rows: `{len(bt021)}` over `{bt021.year.min()}-{bt021.year.max()}` with the locked interval absent.\n"
        f"- Locked rows skipped before value conversion: `{locked_rows}`.\n"
        f"- Exact proxy6/proxy9 aliases: `{exact_alias_count}`.\n"
        f"- Near-exact proxy6/proxy9 sequences: `{near_alias_count}`.\n"
        f"- Decision: `{decision}`.\n"
        "- Bhutan 2015 remains same-source evidence and is not independent validation.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
