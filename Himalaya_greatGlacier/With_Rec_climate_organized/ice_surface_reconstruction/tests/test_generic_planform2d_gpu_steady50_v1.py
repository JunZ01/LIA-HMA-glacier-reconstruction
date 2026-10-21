from __future__ import annotations

import unittest

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    run_system2005_masked_steady50_checkpointed_v1 as engine,
)


class GenericSteadyContractTests(unittest.TestCase):
    def test_system2005_legacy_climate_contract_stays_600_months(self) -> None:
        config = {"schema": "legacy", "system_id": 2005}
        self.assertEqual(engine.climate_month_contract(config, 600), (600, False))
        with self.assertRaisesRegex(RuntimeError, "climate-month-count-not-600"):
            engine.climate_month_contract(config, 360)

    def test_generic_parent_climate_contract_requires_explicit_repeat(self) -> None:
        config = {
            "schema": engine.GENERIC_STEADY_SCHEMA,
            "parent_climate_period_months": 360,
            "repeat_parent_climate": True,
        }
        self.assertEqual(engine.climate_month_contract(config, 360), (360, True))
        config["repeat_parent_climate"] = False
        with self.assertRaisesRegex(RuntimeError, "parent-climate-repeat-not-authorized"):
            engine.climate_month_contract(config, 360)

    def test_parent_month_mapping_repeats_1441_1470_without_clamping(self) -> None:
        expected = {0: 0, 11: 11, 359: 359, 360: 0, 599: 239}
        actual = {
            month: engine.source_month_index(month, 360, True) for month in expected
        }
        self.assertEqual(actual, expected)

    def test_checkpoint_ranges_are_target_specific_for_10_and_50_years(self) -> None:
        ten = engine.expected_checkpoint_years(10)
        fifty = engine.expected_checkpoint_years(50)
        self.assertEqual(ten, {str(year) for year in range(1, 11)})
        self.assertEqual(len(fifty), 50)
        self.assertLess(ten, fifty)
        self.assertIn("50", fifty)
        self.assertNotIn("50", ten)


if __name__ == "__main__":
    unittest.main()
