"""Audit whether proxy6 and proxy9 support comparable exact EnKF replay.

This is a read-only archive audit.  It deliberately does not run OGGM, alter
original MAT files, or infer causal proxy contributions from member curves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.io


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mat_shape(value: object) -> list[int]:
    return list(np.asarray(value).shape)


def _proxy_summary(path: Path) -> dict[str, object]:
    required = {"HXb", "Rvec", "data", "age_old", "age_yng", "b", "location", "psm_mthd", "ptype"}
    with h5py.File(path, "r") as archive:
        data = archive["proxy_data"]
        references = np.asarray(data["full_proxy"]).reshape(-1, order="F")
        missing_fields: dict[str, int] = {name: 0 for name in required}
        hxb_shapes: set[tuple[int, ...]] = set()
        residual_shapes: set[tuple[int, ...]] = set()
        observed_residual_count = 0
        missing_residual_count = 0
        for reference in references:
            proxy = archive[reference]
            for name in required:
                if name not in proxy:
                    missing_fields[name] += 1
            if "HXb" in proxy:
                hxb_shapes.add(tuple(proxy["HXb"].shape))
            if "Rvec" in proxy:
                residual = np.asarray(proxy["Rvec"], dtype=float).reshape(-1, order="F")
                residual_shapes.add(tuple(proxy["Rvec"].shape))
                observed_residual_count += int(np.isfinite(residual).sum())
                missing_residual_count += int(np.isnan(residual).sum())
        return {
            "path": str(path),
            "sha256": _sha256(path),
            "top_level_keys": sorted(archive.keys()),
            "proxy_count": int(references.size),
            "all_required_fields_present": all(value == 0 for value in missing_fields.values()),
            "missing_field_proxy_counts": missing_fields,
            "hxb_shapes": [list(shape) for shape in sorted(hxb_shapes)],
            "residual_shapes": [list(shape) for shape in sorted(residual_shapes)],
            "observed_residual_count": observed_residual_count,
            "missing_residual_count": missing_residual_count,
        }


def _posterior_summary(path: Path) -> dict[str, object]:
    required = {"Xa_m_all", "Xa_sigma_all", "Xa_prct_all", "experiment_status"}
    with h5py.File(path, "r") as archive:
        missing = sorted(required.difference(archive.keys()))
        status = np.asarray(archive["experiment_status"]).reshape(-1, order="F")
        return {
            "path": str(path),
            "sha256": _sha256(path),
            "all_required_fields_present": not missing,
            "missing_fields": missing,
            "mean_shape": list(archive["Xa_m_all"].shape),
            "sigma_shape": list(archive["Xa_sigma_all"].shape),
            "percentile_shape": list(archive["Xa_prct_all"].shape),
            "experiment_status_shape": list(archive["experiment_status"].shape),
            "experiment_status_values": [str(value) for value in status.tolist()],
        }


def _indices_summary(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as archive:
        candidates = [key for key in archive.keys() if key != "#refs#"]
        if len(candidates) != 1:
            raise ValueError(f"unexpected-index-variables:{sorted(candidates)}")
        name = candidates[0]
        values = np.asarray(archive[name])
    flat = values.reshape(-1, order="F")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "variable": name,
        "shape": _mat_shape(values),
        "count": int(flat.size),
        "minimum": int(np.min(flat)),
        "maximum": int(np.max(flat)),
        "unique_count": int(np.unique(flat).size),
    }


def _existing_proxy6_replay(root: Path) -> dict[str, object]:
    replay = root / "replay_proxy6_enkf_exp1_matlab_v1"
    input_path = replay / "replay_input.mat"
    manifest_path = replay / "input_manifest.json"
    return {
        "replay_dir": str(replay),
        "input_exists": input_path.exists(),
        "manifest_exists": manifest_path.exists(),
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    archive_root = Path(config["archive_root"])
    result_root = root / config["result_root"]
    result_root.mkdir(parents=True, exist_ok=True)

    indices = archive_root / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat"
    proxy_summaries = {}
    posterior_summaries = {}
    for proxy_id in config["proxy_sets"]:
        key = f"proxy{proxy_id}"
        proxy_summaries[key] = _proxy_summary(archive_root / f"python_data/proxy_data_{proxy_id}.mat")
        posterior_summaries[key] = _posterior_summary(
            archive_root / f"fixed_proxy_experiments_69/tas/results/{key}/EnKF_results.mat"
        )

    proxy9 = proxy_summaries["proxy9"]
    posterior9 = posterior_summaries["proxy9"]
    structural_gate = bool(
        proxy9["all_required_fields_present"]
        and proxy9["hxb_shapes"] == [[848, 1]]
        and posterior9["all_required_fields_present"]
        and posterior9["mean_shape"] == [50, 1001, 336]
        and posterior9["sigma_shape"] == [50, 1001, 336]
        and posterior9["percentile_shape"] == [50, 1001, 3, 336]
    )
    residual_gate = int(proxy9["observed_residual_count"]) > 0
    regem_gap = {
        "proxy6_exact_regem_input_preserved": True,
        "proxy9_exact_regem_input_preserved": False,
        "meaning": "Proxy9 Rvec is archived, but a completed proxy9 RegEM residual matrix has not yet been materialized in this project.",
        "required_next_step": "Run the archived RegEM procedure for proxy9 in a new independent result directory; verify all originally observed residual entries are unchanged before EnKF replay.",
    }
    decision = (
        "proxy9-replay-prerequisites-pass-regem-step-pending"
        if structural_gate and residual_gate
        else "proxy9-exact-replay-blocked-by-archive-structure"
    )
    report = {
        "experiment_id": config["experiment_id"],
        "mode": config["mode"],
        "source_purity": "read-only sources; hashes recorded before no source writes",
        "indices": _indices_summary(indices),
        "proxy_archives": proxy_summaries,
        "posterior_archives": posterior_summaries,
        "existing_proxy6_exact_replay": _existing_proxy6_replay(root / config["source_recovery_root"]),
        "proxy9_replayability": {
            "structural_gate_pass": structural_gate,
            "observed_residual_gate_pass": residual_gate,
            "regem_provenance_gap": regem_gap,
            "decision": decision,
            "not_yet_permitted": "Do not calculate proxy9 K x innovation contributions or compare causal proxy rankings until the proxy9 replay closes against archived experiment 1.",
        },
    }
    (result_root / "provenance_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    markdown = f"""# Proxy Replay Provenance Audit

## Scope

Read-only audit for the 1901-1950 proxy-increment diagnosis. No OGGM run was
started, no source MAT file was modified, and `rec_ref` remains comparison-only.

## Result

`proxy9` structural replay gate: **{structural_gate}**.

- Proxy9 archive contains {proxy9['proxy_count']} proxies with archived `HXb`,
  observations, PSM fields, and residual vectors.
- Its fixed EnKF archive stores posterior mean/sigma/percentiles with shapes
  {posterior9['mean_shape']}, {posterior9['sigma_shape']}, and
  {posterior9['percentile_shape']}.
- The common index archive has shape {report['indices']['shape']}.

## Boundary

The currently materialized exact replay is only proxy6. Proxy9 has the raw
residual vector but not an already materialized RegEM-completed matrix in this
project. The next permitted action is a separate proxy9 RegEM reproduction,
followed by a strict observed-entry preservation check and archived-posterior
closure. Until then, proxy6/proxy9 member-mean differences are descriptive,
not per-proxy causal attribution.

Decision: `{decision}`.
"""
    (result_root / "FINAL_RESULT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report["proxy9_replayability"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
