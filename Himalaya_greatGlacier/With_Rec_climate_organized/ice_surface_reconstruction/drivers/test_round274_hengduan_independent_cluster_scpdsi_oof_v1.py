from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


DRIVER = Path(__file__).with_name("run_round274_hengduan_independent_cluster_scpdsi_oof_v1.py")


def load_driver():
    spec = importlib.util.spec_from_file_location("round274_driver", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    driver = load_driver()
    rng = np.random.default_rng(274)
    signal = rng.normal(size=len(driver.YEARS))
    proxy = np.column_stack(
        (signal + rng.normal(0, 0.15, len(signal)), -signal + rng.normal(0, 0.15, len(signal)))
    )
    target = 0.8 * signal + rng.normal(0, 0.12, len(signal))
    prediction, baseline, folds = driver.blocked_oof(proxy, target)
    metrics = driver.pooled(target, prediction, baseline)
    assert metrics["correlation"] > 0.9
    assert metrics["skill_vs_training_fold_mean"] > 0.7
    assert len(folds) == 5
    assert all(row["fit_slope"] > 0 for row in folds)
    print("ROUND274_SYNTHETIC_OOF_TEST_OK")


if __name__ == "__main__":
    main()
