from __future__ import annotations

import unittest

import numpy as np

from run_round267_nepa043_mam_temperature_modern_oof_v1 import YEARS, blocked_oof


class Round267SyntheticOofTests(unittest.TestCase):
    def test_negative_response_passes_direction_and_skill(self) -> None:
        rng = np.random.default_rng(267)
        proxy = rng.normal(size=len(YEARS))
        target = 1.5 - 0.8 * proxy + rng.normal(scale=0.08, size=len(YEARS))

        prediction, climatology, rows, summary = blocked_oof(proxy, target, "SYNTHETIC")

        self.assertEqual(len(rows), 5)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(np.isfinite(climatology).all())
        self.assertEqual(summary["oof_count"], 61)
        self.assertEqual(summary["negative_fit_slope_count"], 5)
        self.assertEqual(summary["positive_fold_correlation_count"], 5)
        self.assertGreater(summary["oof_correlation"], 0.9)
        self.assertGreater(summary["oof_skill_vs_training_fold_climatology"], 0.8)
        self.assertLess(summary["early_raw_proxy_target_correlation"], 0.0)
        self.assertLess(summary["late_raw_proxy_target_correlation"], 0.0)

    def test_positive_response_is_not_misclassified_as_negative(self) -> None:
        proxy = np.sin(np.arange(len(YEARS), dtype=float) / 3.0)
        target = 0.6 * proxy

        _, _, rows, summary = blocked_oof(proxy, target, "SYNTHETIC")

        self.assertEqual(len(rows), 5)
        self.assertEqual(summary["negative_fit_slope_count"], 0)
        self.assertTrue(all(float(row["fit_slope"]) > 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
