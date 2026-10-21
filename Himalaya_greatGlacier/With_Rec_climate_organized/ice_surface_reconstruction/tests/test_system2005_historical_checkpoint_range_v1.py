"""Regression tests for the system-2005 historical checkpoint window."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    run_system2005_masked_historical_branch_v3 as historical_v3,
)


class HistoricalCheckpointRangeTest(unittest.TestCase):
    def _run_adjudication(self, target_years: int, checkpoint_years: range):
        result = {
            "checkpoint_metrics": {str(year): {} for year in checkpoint_years},
            "checks": {"all_annual_checkpoints_recorded": False},
            "historical_numerical_checks": {
                "all_annual_checkpoints_recorded": False,
                "unrelated_gate": True,
            },
            "actual_calendar_year": 1470 + target_years,
            "completed_years": target_years,
            "accepted_steps": 1,
            "rejected_steps": 0,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project_root": str(root),
                        "output_root": "output",
                        "target_years": target_years,
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(historical_v3.v2, "run", return_value=result), mock.patch.object(
                historical_v3.engine, "atomic_bytes"
            ):
                return historical_v3.run(config_path)

    def test_ten_year_target_accepts_exactly_ten_checkpoints(self):
        result = self._run_adjudication(10, range(1, 11))

        self.assertTrue(result["checks"]["all_annual_checkpoints_recorded"])
        self.assertEqual(result["status"], "historical-preflight-branch-pass")

    def test_ten_year_target_rejects_inherited_fifty_year_range(self):
        result = self._run_adjudication(10, range(1, 51))

        self.assertFalse(result["checks"]["all_annual_checkpoints_recorded"])
        self.assertEqual(result["status"], "historical-preflight-branch-technical-fail")

    def test_fifty_year_target_accepts_exactly_fifty_checkpoints(self):
        result = self._run_adjudication(50, range(1, 51))

        self.assertTrue(result["checks"]["all_annual_checkpoints_recorded"])
        self.assertEqual(result["status"], "historical-preflight-branch-pass")

    def test_fifty_year_target_rejects_only_ten_checkpoints(self):
        result = self._run_adjudication(50, range(1, 11))

        self.assertFalse(result["checks"]["all_annual_checkpoints_recorded"])
        self.assertEqual(result["status"], "historical-preflight-branch-technical-fail")


if __name__ == "__main__":
    unittest.main()
