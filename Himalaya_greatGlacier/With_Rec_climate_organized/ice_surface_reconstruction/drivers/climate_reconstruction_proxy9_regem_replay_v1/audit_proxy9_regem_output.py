"""Verify that proxy9 RegEM completion preserves observed residual entries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    args = parser.parse_args()
    root = _project_root()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = root / config["result_root"]
    raw = np.asarray(scipy.io.loadmat(output / "regem_input.mat")["R0v"], dtype=float)
    completed_path = output / "regem_output.mat"
    if not completed_path.exists():
        raise FileNotFoundError(f"missing-regem-output:{completed_path}")
    completed = np.asarray(scipy.io.loadmat(completed_path)["R0v_filled"], dtype=float)
    if completed.shape != raw.shape:
        raise ValueError(f"shape-mismatch:{completed.shape}:{raw.shape}")
    observed = np.isfinite(raw)
    changed = np.abs(completed[observed] - raw[observed])
    gates = config["gates"]
    report = {
        "experiment_id": config["experiment_id"],
        "shape": list(completed.shape),
        "input_nan_count": int(np.isnan(raw).sum()),
        "output_nan_count": int(np.isnan(completed).sum()),
        "observed_max_abs_change": float(np.max(changed)) if changed.size else 0.0,
        "observed_entry_count": int(observed.sum()),
        "filled_entries": [
            {"row_1based": int(row + 1), "column_1based": int(column + 1), "value": float(completed[row, column])}
            for row, column in np.argwhere(~observed)
        ],
        "output_sha256": _sha256(completed_path),
    }
    passed = bool(
        report["output_nan_count"] == int(gates["completed_nan_count"])
        and report["observed_max_abs_change"] <= float(gates["observed_entry_max_abs_change"])
    )
    report["status"] = "pass" if passed else "fail"
    (output / "regem_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    if not passed:
        raise RuntimeError(json.dumps(report, ensure_ascii=True))
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
