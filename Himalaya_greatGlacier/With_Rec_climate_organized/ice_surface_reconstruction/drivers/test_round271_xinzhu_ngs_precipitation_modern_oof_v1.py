from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


DRIVER = Path(__file__).with_name("run_round271_xinzhu_ngs_precipitation_modern_oof_v1.py")


def load_driver():
    spec = importlib.util.spec_from_file_location("round271_driver", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    driver = load_driver()
    monthly = np.arange(len(driver.SOURCE_YEARS) * 12, dtype=float)[:, None]
    seasonal = driver.aggregate_previous_november_current_february(monthly)
    assert seasonal.shape == (60, 1)
    assert seasonal[0, 0] == monthly[10, 0] + monthly[11, 0] + monthly[12, 0] + monthly[13, 0]

    rng = np.random.default_rng(271)
    predictor = rng.normal(size=len(driver.YEARS))
    target = 100.0 + 18.0 * predictor + rng.normal(0.0, 2.0, len(driver.YEARS))
    prediction, baseline, folds = driver.blocked_oof(predictor, target)
    pooled = driver.pooled_metrics(target, prediction, baseline)
    assert pooled["oof_correlation"] > 0.95
    assert pooled["oof_skill_vs_training_fold_mean"] > 0.8
    assert len(folds) == 5
    assert all(row["fit_slope"] > 0 for row in folds)
    print("ROUND271_SYNTHETIC_OOF_TEST_OK")


if __name__ == "__main__":
    main()
