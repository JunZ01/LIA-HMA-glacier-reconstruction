"""On-disk regression tests for the system-7 and system-79 adapter fixtures."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.generic_planform2d_adapter_contract_v1 import (
    sha256,
)


PROJECT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"

CASES = {
    7: {
        "shape": [754, 508],
        "source": RESULTS / "core191_anchor_reroute_f19_v3/lia_7_RGI60-15.01141_system_grid_v1.nc",
        "source_sha256": "6D9FCD1CB9F451C5D7B800ECFBBD2038FFDEFDA73918FE24A252749487201097",
        "static": RESULTS / "lia_191_generic_planform2d_system7_static_adapter_fixture_v1",
        "climate": RESULTS / "lia_191_generic_planform2d_system7_historical_monthly_ti_fixture_v1",
    },
    79: {
        "shape": [591, 430],
        "source": RESULTS / "core191_anchor_reroute_f19_v3/lia_79_RGI60-15.01485_system_grid_v1.nc",
        "source_sha256": "9895B3A65F9B04DE2C65FC2999CA3FBA30C9E0DFABF23E5637D7CF8378E2DC78",
        "static": RESULTS / "lia_191_generic_planform2d_system79_static_adapter_fixture_v1",
        "climate": RESULTS / "lia_191_generic_planform2d_system79_historical_monthly_ti_fixture_v1",
    },
}


class GenericAdapterFixtureTests(unittest.TestCase):
    def test_static_identity_shape_hash_and_exact_thickness_copy(self) -> None:
        for system_id, case in CASES.items():
            with self.subTest(system_id=system_id):
                summary = json.loads((case["static"] / "static_summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["system_id"], system_id)
                self.assertEqual(summary["source_shape"], case["shape"])
                self.assertEqual(sha256(case["source"]), case["source_sha256"])
                self.assertTrue(summary["adapter_contract"]["source_system_id_matches"])
                self.assertFalse(summary["adapter_contract"]["initial_thickness_modified"])
                self.assertFalse(summary["adapter_contract"]["old_output_modified"])
                self.assertTrue(all(summary["gate_flags"].values()))

                with xr.open_dataset(case["source"]) as source:
                    raw = np.asarray(source["thickness_m"].isel(member=0), dtype=np.float64)
                    valid = np.asarray(source["valid_mask"].isel(member=0), dtype=bool)
                    lia = np.asarray(source["lia_mask"], dtype=bool)
                expected = np.zeros(raw.shape, dtype=np.float32)
                positive = valid & lia & np.isfinite(raw) & (raw > 1e-6)
                expected[positive] = np.maximum(raw[positive], 0.0).astype(np.float32)
                padding = int(summary["padding_cells_each_side"])
                with Dataset(case["static"] / "geology.nc", "r") as geology:
                    actual = np.asarray(
                        geology.variables["thk"][
                            padding : padding + raw.shape[0],
                            padding : padding + raw.shape[1],
                        ],
                        dtype=np.float32,
                    )
                self.assertTrue(np.array_equal(actual, expected))

    def test_monthly_ti_reference_samples_recompute_exactly(self) -> None:
        for system_id, case in CASES.items():
            with self.subTest(system_id=system_id):
                summary = json.loads((case["climate"] / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["system_id"], system_id)
                self.assertEqual(summary["sequence_month_count"], 6372)
                self.assertTrue(all(summary["checks"].values()))
                self.assertFalse(summary["dynamics_run"])
                self.assertFalse(summary["initial_thickness_modified"])

                package = np.load(case["climate"] / "monthly_ti_historical_1470_2000.npz")
                reference = np.load(case["climate"] / "reference_samples.npz")
                heights = reference["validation_heights_m"]
                positions = reference["selected_sequence_positions"]
                expected = reference["reference_monthly_mb_m_ice_s"]
                actual = []
                for position in positions:
                    index = int(package["sequence_indices"][int(position)])
                    temperature = float(package["temp"][index]) + float(package["grad"][index]) * (
                        heights - float(package["ref_hgt"])
                    )
                    melt_temperature = np.maximum(temperature - float(package["t_melt"]), 0.0)
                    solid_fraction = np.clip(
                        1.0
                        - (temperature - float(package["t_solid"]))
                        / (float(package["t_liq"]) - float(package["t_solid"])),
                        0.0,
                        1.0,
                    )
                    monthly_mass = float(package["prcp"][index]) * solid_fraction
                    monthly_mass -= float(package["monthly_melt_f"]) * melt_temperature
                    monthly_mass -= (
                        float(package["bias"]) * float(package["sec_in_month"]) / float(package["sec_in_year"])
                    )
                    actual.append(
                        monthly_mass / float(package["sec_in_month"]) / float(package["rho"])
                    )
                np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=1e-18)


if __name__ == "__main__":
    unittest.main()
