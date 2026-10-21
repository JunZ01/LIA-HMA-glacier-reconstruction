"""Boundary-safe identity audit for the Duan et al. (2019) TP reconstruction."""

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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    output = (
        root
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
        / "round_156_duan2019_temperature_identity_v1"
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    contract = driver / "round156_duan2019_temperature_identity_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    base = load_module(
        "round156_base",
        driver / "run_round149_noaa_temperature_candidate_identity_v1.py",
    )
    archive = load_module(
        "round156_archive",
        driver / "audit_positive_b2_event_seasonal_psm_v1.py",
    )
    round25 = load_module(
        "round156_round25",
        driver / "run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1.py",
    )

    acquisition = root / "validation_data/external_sources/hma_temperature_duan2019_acquisition_20260819_v1"
    manifest_path = acquisition / "acquisition_manifest.json"
    raw = acquisition / "raw_unopened/duan2019tibet-temp.txt"
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    manifest27 = loop / "round_27_temperature_innovation_gain_sign_decomposition_v1/source_manifest.json"
    proxy9 = Path(json.loads(manifest27.read_text(encoding="utf-8"))["sources_before"]["proxy9"]["path"])
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "acquisition_manifest": manifest_path,
        "raw_duan2019": raw,
        "proxy6": archive.archive_proxy_path(),
        "proxy6_metadata": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "proxy9": proxy9,
        "proxy9_metadata": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy_network_psm_audit_v2/proxy9_identity_and_psm_metadata.csv",
        "manifest27": manifest27,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest["files"][0]
    before = {name: fingerprint(path) for name, path in paths.items()}
    if before["raw_duan2019"]["sha256"].upper() != declared["sha256"].upper():
        raise RuntimeError("acquisition-hash-mismatch")
    if manifest.get("values_opened_at_acquisition"):
        raise RuntimeError("acquisition-boundary-mismatch")

    allowed_path = output / "duan2019_allowed.csv"
    parse_summary = base.parse_allowed(raw, allowed_path, output / "duan2019_header_metadata.txt")
    if parse_summary["locked_rows_skipped_before_value_decode"] != 19:
        raise RuntimeError(f"unexpected-locked-row-count:{parse_summary}")
    frame = pd.read_csv(allowed_path)
    if list(frame.columns) != ["year", "value_1", "value_2", "value_3"]:
        raise RuntimeError(f"unexpected-columns:{list(frame.columns)}")
    candidate = base.aligned(frame, "value_1")

    proxy6_meta = pd.read_csv(paths["proxy6_metadata"])
    proxy6_ids = tuple(proxy6_meta.proxy_index.astype(int))
    proxy6_values = load_module(
        "round156_reader",
        driver / "run_round134_netp_mjj_low_frequency_reconstruction_v2.py",
    ).read_proxy_values(paths["proxy6"], proxy6_ids, YEARS)
    _, proxy9_values, locked = round25.read_proxy_allowed(paths["proxy9"], YEARS)
    if locked != 0:
        raise RuntimeError("proxy9-locked-read")
    proxy9_meta = pd.read_csv(paths["proxy9_metadata"])
    if proxy9_values.shape[0] != len(proxy9_meta):
        raise RuntimeError("proxy9-metadata-count-mismatch")

    rows = []
    exact_total = 0
    near_total = 0
    for network, matrix, metadata in (
        ("proxy6", proxy6_values.T, proxy6_meta),
        ("proxy9", proxy9_values, proxy9_meta),
    ):
        for index in range(matrix.shape[0]):
            item = base.metrics(candidate, matrix[index])
            exact, near = base.classify(item)
            exact_total += int(exact)
            near_total += int(near)
            meta = metadata.iloc[index]
            rows.append(
                {
                    "candidate": "duan2019_primary_temperature",
                    "network": network,
                    "proxy_record_index_1based": index + 1,
                    "proxy_latitude": float(meta.latitude),
                    "proxy_longitude": float(meta.longitude),
                    **item,
                    "absolute_correlation": abs(item["correlation"])
                    if np.isfinite(item["correlation"])
                    else np.nan,
                    "exact_affine_alias": exact,
                    "near_exact_standardized_sequence": near,
                }
            )
    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(output / "all_candidate_proxy_sequence_comparisons.csv", index=False)
    comparisons.sort_values(
        ["exact_affine_alias", "near_exact_standardized_sequence", "absolute_correlation"],
        ascending=[False, False, False],
    ).head(30).to_csv(output / "top_candidate_proxy_sequence_matches.csv", index=False)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": "round_156_duan2019_temperature_identity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "parse_summary": parse_summary,
        "primary_column": "value_1_published_Temp",
        "uncertainty_columns_not_treated_as_candidates": ["value_2", "value_3"],
        "proxy6_record_count": int(proxy6_values.shape[1]),
        "proxy9_record_count": int(proxy9_values.shape[0]),
        "exact_affine_alias_count": exact_total,
        "near_exact_standardized_sequence_count": near_total,
        "scientific_role": "derived_seasonal_operator_or_trajectory_candidate_pending_material_overlap_audit",
        "decision": "advance_to_source_material_overlap_and_modern_seasonal_stability_audit",
        "locked_1921_1939_values_decoded": 0,
        "jionglacuo_external_validation_file_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 156 Duan 2019 Temperature Identity\n\n"
        f"- Exact affine aliases: `{exact_total}`.\n"
        f"- Near-exact standardized sequences: `{near_total}`.\n"
        "- Locked 1921-1939 values decoded: `0`.\n"
        f"- Decision: `{result['decision']}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
