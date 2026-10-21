from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io


EXPERIMENT_ID = "climate_reconstruction_precipitation_pakistan_group_weight_sensitivity_v1"
CALIBRATION_YEARS = np.arange(1941, 2001, dtype=int)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot-load-module:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest(path)}


def extract_calibration(proxy_path: Path, proxy_indices: np.ndarray) -> np.ndarray:
    result = np.full((len(CALIBRATION_YEARS), len(proxy_indices)), np.nan)
    with h5py.File(proxy_path, "r") as handle:
        refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for column, proxy_index in enumerate(proxy_indices):
            proxy = handle[refs[int(proxy_index) - 1]]
            years = np.asarray(proxy["age_yng"], dtype=int).reshape(-1, order="F")
            values = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            lookup = {int(year): float(value) for year, value in zip(years, values)}
            result[:, column] = [lookup.get(int(year), np.nan) for year in CALIBRATION_YEARS]
    return result


def main() -> int:
    root = project_root()
    drivers = Path(__file__).resolve().parent
    base = load_module(drivers / "audit_positive_b2_event_seasonal_psm_v1.py", "pakistan_weight_paths")
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "components": result_root / "climate_reconstruction_precipitation_significant_psm_operator_decomposition_v1/operator_components_input.mat",
        "metadata": result_root / "climate_reconstruction_pages2k_fixed_network_study_batch_v1/fixed_network_study_metadata.csv",
        "proxy_archive": Path(base.archive_proxy_path()),
        "replay": result_root / "climate_reconstruction_precipitation_enkf_exp1_replay_v1/replay_input.mat",
        "prior": result_root / "climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat",
        "baseline": result_root / "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/attribution_output.mat",
        "reference_hma": result_root / "climate_reconstruction_precipitation_common_mode_reference_phase_audit_v1/common_mode_reference_hma_series_1901_1920.csv",
        "positive_b2_ablation_summary": result_root / "climate_reconstruction_precipitation_positive_b2_group_ablation_delta_anchor_v1/summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}

    metadata = pd.read_csv(sources["metadata"])
    pakistan = metadata[metadata["data_author"].eq("Cook, E.R.") & (metadata["latitude"] >= 35.0)].copy()
    pakistan = pakistan.sort_values("proxy_index").reset_index(drop=True)
    proxy_indices = pakistan["proxy_index"].to_numpy(dtype=int)
    selected = scipy.io.loadmat(sources["components"], variable_names=["selected_proxy_indices"])["selected_proxy_indices"].reshape(-1).astype(int)
    if len(proxy_indices) != 14 or not np.isin(proxy_indices, selected).all():
        raise RuntimeError(f"pakistan-selection-contract-failed:{len(proxy_indices)}:{np.isin(proxy_indices, selected).sum()}")

    calibration = extract_calibration(sources["proxy_archive"], proxy_indices)
    if not np.isfinite(calibration).all():
        raise RuntimeError("pakistan-calibration-not-complete")
    standardized = (calibration - calibration.mean(axis=0)) / calibration.std(axis=0)
    correlation = np.corrcoef(standardized, rowvar=False)
    upper = correlation[np.triu_indices_from(correlation, k=1)]
    mean_correlation = float(upper.mean())
    design_effect = float(1.0 + (len(proxy_indices) - 1.0) * max(mean_correlation, 0.0))
    classical_effective_count = float(len(proxy_indices) / design_effect)
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
    spectral_effective_dimension = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
    if not (1.0 < design_effect < len(proxy_indices)):
        raise RuntimeError(f"invalid-design-effect:{design_effect}")

    output.mkdir(parents=True)
    scipy.io.savemat(
        output / "pakistan_group_input.mat",
        {
            "pakistan_proxy_indices": proxy_indices.reshape(-1, 1).astype(float),
            "design_effect": np.array([[design_effect]], dtype=float),
            "classical_effective_record_count": np.array([[classical_effective_count]], dtype=float),
            "spectral_effective_dimension": np.array([[spectral_effective_dimension]], dtype=float),
            "calibration_correlation": correlation,
        },
    )
    pakistan.assign(
        calibration_mean_pairwise_correlation=mean_correlation,
        group_design_effect=design_effect,
        group_classical_effective_record_count=classical_effective_count,
        group_spectral_effective_dimension=spectral_effective_dimension,
    ).to_csv(output / "pakistan_group_metadata.csv", index=False)
    pd.DataFrame(correlation, index=proxy_indices, columns=proxy_indices).to_csv(output / "pakistan_calibration_correlation.csv")

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "period": [1901, 1920],
        "calibration_period_for_weight": [1941, 2000],
        "pakistan_proxy_count": int(len(proxy_indices)),
        "pakistan_proxy_indices": proxy_indices.tolist(),
        "mean_pairwise_calibration_correlation": mean_correlation,
        "design_effect_R_multiplier": design_effect,
        "classical_effective_record_count": classical_effective_count,
        "spectral_effective_dimension": spectral_effective_dimension,
        "weight_interpretation": "Multiplying each Pakistan diagonal R variance by the design effect reduces aggregate equal-variance information from 14 records to the classical effective count. This is a sensitivity branch, not a validated production error model.",
        "locked_1921_1939_accessed": False,
    }
    after = {name: source_state(path) for name, path in sources.items()}
    summary["all_declared_sources_unchanged"] = before == after
    (output / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "prepare_source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-changed-during-pakistan-weight-preparation")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
