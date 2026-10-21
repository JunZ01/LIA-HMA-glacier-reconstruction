"""Contract tests for the generic GPU scientific qualification preflight."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    run_generic_planform2d_gpu_qualification_preflight_v1 as driver,
)


class GenericGpuQualificationPreflightTests(unittest.TestCase):
    def test_checkpoint_round_trip_is_bitwise_exact(self) -> None:
        source = np.arange(30, dtype=np.float64).reshape(5, 6)
        with tempfile.TemporaryDirectory() as temporary:
            restored, metadata = driver.write_and_restore_checkpoint(
                Path(temporary),
                name="zero_midpoint",
                state=source,
                elapsed_seconds=123.5,
                proposed_dt_seconds=64.0,
                accepted_steps=11,
                rejected_steps=3,
                contract_hash="A" * 64,
            )
        self.assertTrue(np.array_equal(restored, source))
        self.assertEqual(metadata["accepted_steps"], 11)
        self.assertEqual(metadata["rejected_steps"], 3)

    def test_contract_freezes_two_minimal_phases_and_gates(self) -> None:
        config = {
            "schema": driver.SCHEMA,
            "system_id": 7,
            "run_role": "scientific-qualification-preflight",
            "geology": "geology.nc",
            "geology_sha256": "A" * 64,
            "source_thickness_sha256": "B" * 64,
            "geology_initial_thickness_sha256": "3" * 64,
            "climate_package": "climate.npz",
            "climate_package_sha256": "C" * 64,
            "static_summary": "static.json",
            "static_summary_sha256": "D" * 64,
            "upstream_evidence": "upstream.json",
            "upstream_evidence_sha256": "E" * 64,
            "upstream_config_snapshot": "config.json",
            "upstream_config_snapshot_sha256": "F" * 64,
            "upstream_status": "steady-50y-pass",
            "source_shape": [2, 3],
            "adapter_shape": [4, 5],
            "stage2_frozen": {key: 1.0 for key in driver.STAGE2_KEYS},
            "launch_route": "route.json",
            "launch_route_sha256": "1" * 64,
            "driver_sha256": "2" * 64,
        }
        payload = driver.contract_payload(config)
        self.assertEqual(payload["phases"]["zero_mb_years"], 0.1)
        self.assertEqual(payload["phases"]["climate_response_years"], 1.0)
        self.assertTrue(payload["phases"]["restart_each_phase_from_source_thickness"])
        self.assertTrue(payload["phases"]["midpoint_checkpoint_bitwise_restore"])
        self.assertEqual(payload["gates"]["zero_mb_mass_relative"], 1.0e-12)
        self.assertEqual(payload["gates"]["climate_mass_relative"], 1.0e-10)
        self.assertEqual(payload["solver"]["minimum_dt_seconds"], 0.001)
        self.assertEqual(payload["solver"]["maximum_dt_seconds"], 86400.0)

    def test_zero_mb_bridge_is_identity(self) -> None:
        state = torch.tensor([[0.0, 2.0], [3.0, 0.0]], dtype=torch.float64)
        restored, diagnostics = driver.ZeroMassBalanceBridge().apply_connected_ice_source(
            state,
            torch.zeros_like(state),
            month_index=0,
            dt_seconds=999.0,
        )
        self.assertTrue(torch.equal(restored, state))
        self.assertEqual(diagnostics["accepted_thickness_change_sum_m"], 0.0)
        self.assertTrue(diagnostics["remote_zero_unchanged"])


if __name__ == "__main__":
    unittest.main()
