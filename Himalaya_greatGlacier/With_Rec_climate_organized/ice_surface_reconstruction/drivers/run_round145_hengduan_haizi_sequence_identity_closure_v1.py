"""Exact sequence identity closure for the unsealed Hengduan and Haizi candidates."""

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
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def series(path: Path, year_column: str, value_column: str) -> np.ndarray:
    frame = pd.read_csv(path)
    years = frame[year_column].to_numpy(dtype=int)
    if ((years >= 1921) & (years <= 1939)).any():
        raise RuntimeError(f"locked-year-in-allowed-export:{path.name}")
    values = frame[value_column].to_numpy(dtype=float)
    lookup = dict(zip(years, values))
    return np.asarray([lookup.get(int(year), np.nan) for year in YEARS], dtype=float)


def metrics(source: np.ndarray, candidate: np.ndarray):
    valid = np.isfinite(source) & np.isfinite(candidate)
    count = int(valid.sum())
    result = {
        "overlap_count": count,
        "correlation": float("nan"),
        "affine_intercept": float("nan"),
        "affine_slope": float("nan"),
        "maximum_abs_residual": float("nan"),
        "normalized_rmse": float("nan"),
    }
    if count < 20:
        return result
    left = source[valid]
    right = candidate[valid]
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return result
    design = np.column_stack((np.ones(count), right))
    coefficients = np.linalg.lstsq(design, left, rcond=None)[0]
    residual = left - design @ coefficients
    result.update({
        "correlation": float(np.corrcoef(left, right)[0, 1]),
        "affine_intercept": float(coefficients[0]),
        "affine_slope": float(coefficients[1]),
        "maximum_abs_residual": float(np.max(np.abs(residual))),
        "normalized_rmse": float(np.sqrt(np.mean(residual**2)) / np.std(left, ddof=1)),
    })
    return result


def classify(item):
    exact = bool(
        item["overlap_count"] >= 20
        and np.isfinite(item["correlation"])
        and abs(item["correlation"]) >= 0.999999
        and item["maximum_abs_residual"] <= 1e-8
    )
    near = bool(
        item["overlap_count"] >= 20
        and np.isfinite(item["correlation"])
        and abs(item["correlation"]) >= 0.995
        and item["normalized_rmse"] <= 0.05
    )
    return exact, near


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_145_hengduan_haizi_sequence_identity_closure_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round145_hengduan_haizi_sequence_identity_closure_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    base = load_module("round145_base", driver / "run_round134_netp_mjj_low_frequency_reconstruction_v2.py")
    archive = load_module("round145_archive", driver / "audit_positive_b2_event_seasonal_psm_v1.py")
    round25 = load_module("round145_round25", driver / "run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1.py")
    r144 = loop / "round_144_temperature_candidate_boundary_preflight_v3"
    manifest27 = loop / "round_27_temperature_innovation_gain_sign_decomposition_v1/source_manifest.json"
    manifest = json.loads(manifest27.read_text(encoding="utf-8"))
    proxy9 = Path(manifest["sources_before"]["proxy9"]["path"])
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "r144_summary": r144 / "summary.json",
        "haizi": r144 / "haizi__sheet1__allowed.csv",
        "hengduan": r144 / "hengduan_info__tree_ring_width_chronology__allowed.csv",
        "hengduan_temperature": r144 / "hengduan_info__temperature_reconstruction__allowed.csv",
        "proxy6": archive.archive_proxy_path(),
        "proxy6_metadata": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "proxy9": proxy9,
        "proxy9_metadata": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy_network_psm_audit_v2/proxy9_identity_and_psm_metadata.csv",
        "manifest27": manifest27,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    r144_summary = json.loads(paths["r144_summary"].read_text(encoding="utf-8"))
    if r144_summary.get("locked_1921_1939_values_decoded") != 0 or r144_summary.get("jionglacuo_external_validation_file_opened"):
        raise RuntimeError("round144-boundary-failed")
    if fingerprint(proxy9)["sha256"] != manifest["sources_before"]["proxy9"]["sha256"]:
        raise RuntimeError("proxy9-hash-conflict")

    candidates = {
        "haizi_std": series(paths["haizi"], "year", "std"),
        "hengduan_ssfstb": series(paths["hengduan"], "Year", "SsfStb"),
    }
    internal = {
        "haizi_published_temperature": series(paths["haizi"], "year", "Temperature(\u2103)"),
        "hengduan_published_temperature": series(paths["hengduan_temperature"], "Year", "RT"),
    }
    proxy6_meta = pd.read_csv(paths["proxy6_metadata"])
    proxy6_ids = tuple(proxy6_meta.proxy_index.astype(int))
    proxy6_values = base.read_proxy_values(paths["proxy6"], proxy6_ids, YEARS)
    _, proxy9_values, locked = round25.read_proxy_allowed(paths["proxy9"], YEARS)
    if locked != 0:
        raise RuntimeError("proxy9-locked-read")
    proxy9_meta = pd.read_csv(paths["proxy9_metadata"])
    if proxy9_values.shape[0] != len(proxy9_meta):
        raise RuntimeError(f"proxy9-metadata-count:{proxy9_values.shape[0]}:{len(proxy9_meta)}")

    rows = []
    alias_counts = {}
    near_counts = {}
    for candidate_name, candidate_values in candidates.items():
        exact_total = 0
        near_total = 0
        for network, matrix, metadata in (
            ("proxy6", proxy6_values.T, proxy6_meta),
            ("proxy9", proxy9_values, proxy9_meta),
        ):
            for index in range(matrix.shape[0]):
                item = metrics(candidate_values, matrix[index])
                exact, near = classify(item)
                exact_total += int(exact)
                near_total += int(near)
                meta = metadata.iloc[index]
                rows.append({
                    "candidate": candidate_name,
                    "network": network,
                    "proxy_record_index_1based": index + 1,
                    "proxy_latitude": float(meta.latitude),
                    "proxy_longitude": float(meta.longitude),
                    **item,
                    "absolute_correlation": abs(item["correlation"]) if np.isfinite(item["correlation"]) else np.nan,
                    "exact_affine_alias": exact,
                    "near_exact_standardized_sequence": near,
                })
        alias_counts[candidate_name] = exact_total
        near_counts[candidate_name] = near_total
    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(output / "all_candidate_proxy_sequence_comparisons.csv", index=False)
    top = comparisons.sort_values(
        ["candidate", "exact_affine_alias", "near_exact_standardized_sequence", "absolute_correlation"],
        ascending=[True, False, False, False],
    ).groupby("candidate", as_index=False).head(20)
    top.to_csv(output / "top_candidate_proxy_sequence_matches.csv", index=False)

    internal_rows = []
    for candidate_name, temperature_name in (("haizi_std", "haizi_published_temperature"), ("hengduan_ssfstb", "hengduan_published_temperature")):
        item = metrics(candidates[candidate_name], internal[temperature_name])
        exact, near = classify(item)
        internal_rows.append({"candidate": candidate_name, "same_source_temperature": temperature_name, **item, "exact_affine_derivation": exact, "near_exact_affine_derivation": near})
    internal_frame = pd.DataFrame(internal_rows)
    internal_frame.to_csv(output / "same_source_chronology_temperature_relationships.csv", index=False)

    candidate_pair = metrics(candidates["haizi_std"], candidates["hengduan_ssfstb"])
    pair_exact, pair_near = classify(candidate_pair)
    pair = {"left": "haizi_std", "right": "hengduan_ssfstb", **candidate_pair, "exact_affine_alias": pair_exact, "near_exact_standardized_sequence": pair_near}
    pd.DataFrame([pair]).to_csv(output / "candidate_pair_sequence_comparison.csv", index=False)

    no_alias = all(value == 0 for value in alias_counts.values()) and not pair_exact
    roles = pd.DataFrame([
        {"series": "haizi_std", "role": "new_proxy_candidate" if alias_counts["haizi_std"] == 0 else "existing_proxy_alias", "may_fit_after_modern_oof": alias_counts["haizi_std"] == 0, "may_be_independent_validation": False},
        {"series": "haizi_published_temperature", "role": "same_source_derived_reconstruction_internal_only", "may_fit_after_modern_oof": False, "may_be_independent_validation": False},
        {"series": "hengduan_ssfstb", "role": "new_proxy_candidate" if alias_counts["hengduan_ssfstb"] == 0 else "existing_proxy_alias", "may_fit_after_modern_oof": alias_counts["hengduan_ssfstb"] == 0, "may_be_independent_validation": False},
        {"series": "hengduan_published_temperature", "role": "same_source_derived_reconstruction_internal_only", "may_fit_after_modern_oof": False, "may_be_independent_validation": False},
        {"series": "jionglacuo_temperature_reconstruction", "role": "sealed_external_validation", "may_fit_after_modern_oof": False, "may_be_independent_validation": True},
    ])
    roles.to_csv(output / "candidate_scientific_roles.csv", index=False)
    decision = "advance_hengduan_haizi_to_site_measurement_and_modern_oof" if no_alias else "quarantine_aliased_candidate_before_psm"

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "experiment_id": "round_145_hengduan_haizi_sequence_identity_closure_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "proxy6_record_count": int(proxy6_values.shape[1]),
        "proxy9_record_count": int(proxy9_values.shape[0]),
        "exact_affine_alias_counts": alias_counts,
        "near_exact_standardized_sequence_counts": near_counts,
        "candidate_pair": pair,
        "same_source_relationships": internal_frame.to_dict(orient="records"),
        "decision": decision,
        "locked_1921_1939_accessed": False,
        "jionglacuo_external_validation_file_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 145 Hengduan/Haizi Sequence Identity Closure\n\n"
        f"- Exact proxy6/proxy9 aliases: `{alias_counts}`.\n"
        f"- Near-exact proxy6/proxy9 sequences: `{near_counts}`.\n"
        f"- Candidate-to-candidate correlation: `{pair['correlation']:.3f}` over `{pair['overlap_count']}` years.\n"
        f"- Jionglacuo opened: `False`.\n"
        f"- Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
