"""Prepare frozen operator-component inputs for the significant-PSM diagnosis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def archive_root() -> Path:
    archive_folder = "".join(chr(value) for value in (25991, 20214, 25972, 29702))
    paper_folder = "1 " + "".join(chr(value) for value in (35770, 25991))
    return Path(r"D:\Users\JUNZ\Desktop") / archive_folder / paper_folder / "work1" / "compared_result_all"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = project_root()
    attribution = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2"
    source_audit = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_proxy_common_mode_source_audit_v2"
    paths = {
        "replay_input": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_enkf_exp1_replay_v1/replay_input.mat",
        "prior": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat",
        "proxy6": archive_root() / "python_data/proxy_data_6.mat",
        "metadata": attribution / "proxy6_metadata.csv",
        "baseline_attribution": attribution / "attribution_output.mat",
        "source_audit": source_audit / "proxy_source_trend_table.csv",
    }
    if missing := [str(path) for path in paths.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_significant_psm_operator_decomposition_v1"
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    before = {name: sha256(path) for name, path in paths.items()}

    source_table = pd.read_csv(paths["source_audit"])
    metadata = pd.read_csv(paths["metadata"]).set_index("proxy_index")
    selected = np.sort(source_table.loc[source_table["psm_significance"].eq("p<=0.05"), "proxy_index"].astype(int).unique())
    if len(selected) != 65:
        raise RuntimeError(f"unexpected-significant-network-count:{len(selected)}")
    selected_metadata = metadata.loc[selected]
    if not (selected_metadata["proxy_type"].eq("tree").all() and selected_metadata["coefficient_count"].eq(3).all() and (selected_metadata["psm_pvalue"] <= 0.05).all()):
        raise RuntimeError("selected-proxy-definition-failed")

    replay = scipy.io.loadmat(paths["replay_input"])
    sample_indices = np.asarray(replay["sample_indices"], dtype=int).reshape(-1)
    baseline_hxb = np.asarray(replay["HXb0"], dtype=float)
    if baseline_hxb.shape != (202, 500) or sample_indices.shape != (500,):
        raise RuntimeError("unexpected-replay-input-shape")
    temperature = np.zeros((len(selected), len(sample_indices)))
    precipitation = np.zeros_like(temperature)
    full = np.zeros_like(temperature)
    errors = []
    with h5py.File(paths["proxy6"], "r") as handle:
        references = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for row, proxy_index in enumerate(selected):
            proxy = handle[references[proxy_index - 1]]
            b = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
            mt = np.asarray(proxy["M_t"], dtype=float).reshape(-1, order="F")[sample_indices - 1]
            mp = np.asarray(proxy["M_p"], dtype=float).reshape(-1, order="F")[sample_indices - 1]
            if b.shape != (3,) or not (np.isfinite(mt).all() and np.isfinite(mp).all()):
                raise RuntimeError(f"invalid-operator-components:{proxy_index}")
            temperature[row] = b[1] * (mt - mt.mean())
            precipitation[row] = b[2] * (mp - mp.mean())
            full[row] = temperature[row] + precipitation[row]
            errors.append(float(np.max(np.abs(baseline_hxb[proxy_index - 1].mean() + full[row] - baseline_hxb[proxy_index - 1]))))
    maximum_error = max(errors)
    if maximum_error > 1e-12:
        raise RuntimeError(f"operator-reconstruction-closure-failed:{maximum_error}")

    output.mkdir(parents=True)
    scipy.io.savemat(output / "operator_components_input.mat", {"selected_proxy_indices": selected.astype(float).reshape(-1, 1), "temperature_anomaly": temperature, "precipitation_anomaly": precipitation, "full_anomaly": full}, do_compression=False)
    after = {name: sha256(path) for name, path in paths.items()}
    summary = {"experiment_id": "climate_reconstruction_precipitation_significant_psm_operator_decomposition_v1", "date": "2026-08-14", "period": [1901, 1920], "locked_final_period_accessed": False, "selected_proxy_definition": "HMA-network bivariate tree PSM with archived p<=0.05", "selected_proxy_count": int(len(selected)), "selected_proxy_indices": selected.tolist(), "operator_reconstruction_max_abs_error": maximum_error, "source_hashes_unchanged": before == after}
    (output / "prepare_source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (output / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
