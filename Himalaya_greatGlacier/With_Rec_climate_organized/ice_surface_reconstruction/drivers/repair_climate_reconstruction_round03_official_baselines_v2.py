"""Create v2 Round 03 inputs with the audited official PSM baselines."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import scipy.io


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def main() -> int:
    project = root()
    output = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    p6_v1 = output / "proxy6_allowed_1940_2000.mat"
    p9_v1 = output / "proxy9_allowed_1940_2000.mat"
    p6_v2 = output / "proxy6_allowed_1940_2000_official_baseline_v2.mat"
    p9_v2 = output / "proxy9_allowed_1940_2000_official_baseline_v2.mat"
    if p6_v2.exists() or p9_v2.exists():
        raise FileExistsError("round03-v2-input-already-exists")
    replay = project / "validation_data/results/paleo_surface_validation_v0/enkf_reconstruction_source_recovery_audit_v2/replay_proxy6_enkf_exp1_matlab_v1/replay_input.mat"
    proxy9_baseline = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_calibration_window_stability_v1/calibration_windows_v4.mat"
    for path in (p6_v1, p9_v1, replay, proxy9_baseline):
        if not path.exists():
            raise FileNotFoundError(path)

    proxy6 = {key: value for key, value in scipy.io.loadmat(p6_v1).items() if not key.startswith("__")}
    official6 = scipy.io.loadmat(replay, variable_names=["sample_indices", "HXb0"])
    proxy6["closure_indices"] = np.asarray(official6["sample_indices"], dtype=float)
    proxy6["closure_hxb"] = np.asarray(official6["HXb0"], dtype=float)
    scipy.io.savemat(p6_v2, proxy6, do_compression=True)

    proxy9 = {key: value for key, value in scipy.io.loadmat(p9_v1).items() if not key.startswith("__")}
    with h5py.File(proxy9_baseline, "r") as handle:
        hxb = np.asarray(handle["hxb"], dtype=float).transpose(2, 1, 0)
        b = np.asarray(handle["b"], dtype=float).transpose(2, 1, 0)
    proxy9["closure_indices"] = np.arange(1, 849, dtype=float)[None, :]
    proxy9["closure_hxb"] = hxb[:, 2, :]
    proxy9["archived_b"] = b[:, 2, :]
    scipy.io.savemat(p9_v2, proxy9, do_compression=True)

    summary = {
        "status": "official_baseline_v2_inputs_created",
        "proxy6_closure": "replay_input.HXb0 on fixed 500 sample_indices",
        "proxy9_closure": "calibration_windows_v4 branch 3 on all 848 members",
        "locked_1921_1939_accessed": False,
        "outputs": [p6_v2.name, p9_v2.name],
    }
    (output / "official_baseline_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
