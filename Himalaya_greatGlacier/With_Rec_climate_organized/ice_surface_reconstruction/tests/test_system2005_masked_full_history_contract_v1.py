"""Direct target-window tests for the System-2005 full historical driver."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    run_system2005_masked_historical_branch_v2 as v2,
)
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    run_system2005_masked_historical_branch_v3 as v3,
)


class _Sequence:
    def numel(self) -> int:
        return 6372


class FullHistoryContractTests(unittest.TestCase):
    def _v3_result(self, checkpoint_count: int) -> dict:
        return {
            "checkpoint_metrics": {str(index): {} for index in range(1, checkpoint_count + 1)},
            "checks": {"all_annual_checkpoints_recorded": False},
            "historical_numerical_checks": {"all_annual_checkpoints_recorded": False},
            "actual_calendar_year": 2000,
            "completed_years": 530.0,
            "accepted_steps": 1,
            "rejected_steps": 0,
        }

    def _run_v3(self, checkpoint_count: int) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "out").mkdir()
            config = root / "config.json"
            config.write_text(
                json.dumps({"project_root": str(root), "output_root": "out", "target_years": 530}),
                encoding="utf-8",
            )
            with patch.object(v3.v2, "run", return_value=self._v3_result(checkpoint_count)):
                return v3.run(config)

    def test_exact_530_checkpoint_range_passes(self) -> None:
        result = self._run_v3(530)
        self.assertTrue(result["historical_numerical_checks"]["all_annual_checkpoints_recorded"])
        self.assertEqual(result["status"], "historical-preflight-branch-pass")

    def test_missing_final_checkpoint_fails(self) -> None:
        result = self._run_v3(529)
        self.assertFalse(result["historical_numerical_checks"]["all_annual_checkpoints_recorded"])
        self.assertEqual(result["status"], "historical-preflight-branch-technical-fail")

    def test_full_target_exposes_6360_months(self) -> None:
        captured: dict[str, int] = {}

        def fake_v1_run(config_path: Path, resume_manifest: Path | None = None) -> dict:
            bridge = object.__new__(v2.TorchMonthlyTIBridge)
            bridge.sequence_indices = _Sequence()
            captured["month_count"] = bridge.month_count
            return {"status": "synthetic"}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "out").mkdir()
            config = root / "config.json"
            config.write_text(
                json.dumps({"project_root": str(root), "output_root": "out"}), encoding="utf-8"
            )
            original_years = v2.engine.TARGET_YEARS
            v2.engine.TARGET_YEARS = 530
            try:
                with patch.object(v2.v1, "run", side_effect=fake_v1_run):
                    v2.run(config)
            finally:
                v2.engine.TARGET_YEARS = original_years
        self.assertEqual(captured["month_count"], 6360)


if __name__ == "__main__":
    unittest.main()
