"""Apply the declared archived-posterior closure gates to the proxy9 replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    args = parser.parse_args()
    root = _project_root()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = root / config["result_root"]
    result = scipy.io.loadmat(output / "replay_full.mat")
    metrics = {
        "mean_max_abs_difference_c": float(np.asarray(result["mean_global_max_abs_diff"]).squeeze()),
        "sigma_max_abs_difference_c": float(np.asarray(result["sigma_global_max_abs_diff"]).squeeze()),
        "percentile_max_abs_difference_c": float(np.asarray(result["prct_global_max_abs_diff"]).squeeze()),
        "runtime_seconds": float(np.asarray(result["runtime_seconds"]).squeeze()),
        "active_proxy_count_min": int(np.min(result["active_proxy_count"])),
        "active_proxy_count_max": int(np.max(result["active_proxy_count"])),
    }
    gates = config["gates"]
    passed = all(metrics[key] <= float(limit) for key, limit in gates.items())
    report = {"experiment_id": config["experiment_id"], "status": "pass" if passed else "fail", "metrics": metrics, "gates": gates, "next_stage": "permitted" if passed else "blocked"}
    (output / "replay_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    if not passed:
        raise RuntimeError(json.dumps(report, ensure_ascii=True))
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
