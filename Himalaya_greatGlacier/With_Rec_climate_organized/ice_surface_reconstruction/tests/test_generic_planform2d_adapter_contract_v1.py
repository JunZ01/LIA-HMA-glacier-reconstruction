"""Regression tests for the generic planform-2D adapter contract."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    generic_planform2d_adapter_contract_v1 as contract,
)
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    export_system2005_historical_monthly_ti_bridge_v1 as climate_export,
)
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    run_generic_masked_historical_summary_adapter_v1 as summary_adapter,
)


class GenericAdapterContractTests(unittest.TestCase):
    def test_system2005_legacy_schema_is_unchanged(self) -> None:
        geology, summary, generic = contract.static_schema_names({"system_id": 2005})
        self.assertFalse(generic)
        self.assertEqual(geology, "lia-191-igm-planform2d-system2005-geology-v1")
        self.assertEqual(summary, "lia-191-igm-planform2d-system2005-static-v1")
        climate, generic = contract.climate_schema_names(
            {"schema": "lia-191-system2005-historical-climate-export-config-v1"}
        )
        self.assertFalse(generic)
        self.assertEqual(climate, "lia-191-system2005-historical-monthly-ti-bridge-v1")

    def test_generic_schema_accepts_arbitrary_positive_system_id(self) -> None:
        geology, summary, generic = contract.static_schema_names(
            {"schema": contract.GENERIC_STATIC_CONFIG_SCHEMA, "system_id": 79}
        )
        self.assertTrue(generic)
        self.assertEqual(geology, contract.GENERIC_STATIC_GEOLOGY_SCHEMA)
        self.assertEqual(summary, contract.GENERIC_STATIC_SUMMARY_SCHEMA)

    def test_hashed_path_contract_records_exact_path_hash_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"immutable-source")
            expected = hashlib.sha256(source.read_bytes()).hexdigest().upper()
            result = contract.validate_hashed_paths(
                root, {"source": {"path": "source.bin", "sha256": expected}}
            )
        self.assertEqual(result["source"]["sha256"], expected)
        self.assertEqual(result["source"]["size_bytes"], 16)

    def test_hashed_path_contract_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source.bin").write_bytes(b"immutable-source")
            with self.assertRaisesRegex(RuntimeError, "source-hash-mismatch"):
                contract.validate_hashed_paths(
                    root, {"source": {"path": "source.bin", "sha256": "0" * 64}}
                )

    def test_real_gdir_is_cloned_into_an_isolated_runtime_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            baseline = source / "climate_historical.nc"
            baseline.write_bytes(b"frozen-climate")
            (source / "outlines.tar.gz").write_bytes(b"frozen-outline")
            (source / "glacier_grid.json").write_text("{}", encoding="utf-8")
            output = root / "output"
            output.mkdir()
            short_runtime = root / "runtime"
            runtime_base, runtime_rgi_id, recorded_source = climate_export.clone_runtime_gdir(
                {
                    "source_gdir": str(source),
                    "baseline_climate": str(baseline),
                    "main_rgi_id": "RGI60-15.06666",
                    "runtime_gdir_base": str(short_runtime),
                },
                output,
            )
            cloned = (
                runtime_base
                / "RGI60-15"
                / "RGI60-15.06"
                / "RGI60-15.06666"
            )
            self.assertEqual(runtime_rgi_id, "RGI60-15.06666")
            self.assertEqual(recorded_source, source)
            self.assertEqual(runtime_base, short_runtime)
            self.assertEqual((cloned / "climate_historical.nc").read_bytes(), b"frozen-climate")
            self.assertEqual((cloned / "outlines.tar.gz").read_bytes(), b"frozen-outline")
            self.assertFalse((cloned / "unrelated-large-file.bin").exists())
            self.assertEqual(baseline.read_bytes(), b"frozen-climate")

    def test_array_hash_covers_shape_and_bytes_without_mutation(self) -> None:
        source = np.arange(12, dtype=np.float64).reshape(3, 4)
        before = source.copy()
        digest = contract.array_sha256(source)
        self.assertTrue(np.array_equal(source, before))
        self.assertNotEqual(digest, contract.array_sha256(source.reshape(2, 6)))

    def test_legacy_6372_month_sample_positions_are_exact(self) -> None:
        self.assertEqual(
            contract.selected_sample_positions(6372).tolist(),
            [0, 1, 11, 12, 359, 360, 3180, 6359, 6360, 6371],
        )

    def test_checkpoint_ranges_use_registered_ten_and_fifty_year_targets(self) -> None:
        self.assertTrue(contract.checkpoint_range_complete(range(1, 11), 10))
        self.assertFalse(contract.checkpoint_range_complete(range(1, 51), 10))
        self.assertTrue(contract.checkpoint_range_complete(range(1, 51), 50))
        self.assertFalse(contract.checkpoint_range_complete(range(1, 11), 50))

    def test_summary_adapter_changes_no_input_and_runs_no_dynamics(self) -> None:
        source = {
            "checkpoint_metrics": {str(year): {} for year in range(1, 11)},
            "checks": {"all_annual_checkpoints_recorded": False},
            "historical_numerical_checks": {
                "all_annual_checkpoints_recorded": False,
                "finite_state": True,
            },
        }
        original = {
            "checkpoint_metrics": {str(year): {} for year in range(1, 11)},
            "checks": {"all_annual_checkpoints_recorded": False},
            "historical_numerical_checks": {
                "all_annual_checkpoints_recorded": False,
                "finite_state": True,
            },
        }
        result = summary_adapter.adjudicate(source, {"system_id": 7, "target_years": 10})
        self.assertEqual(source, original)
        self.assertEqual(result["system_id"], 7)
        self.assertEqual(result["status"], "historical-preflight-branch-pass")
        self.assertFalse(result["adapter_contract"]["dynamics_run_by_adapter"])
        self.assertFalse(result["adapter_contract"]["initial_thickness_modified"])


if __name__ == "__main__":
    unittest.main()
