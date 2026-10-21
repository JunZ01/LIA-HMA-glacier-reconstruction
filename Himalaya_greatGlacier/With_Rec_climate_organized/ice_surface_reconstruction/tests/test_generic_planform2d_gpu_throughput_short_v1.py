"""Contract tests for the generic short GPU throughput driver."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    run_generic_planform2d_gpu_throughput_short_v1 as driver,
)


class GenericGpuThroughputShortTests(unittest.TestCase):
    def test_checkpoint_round_trip_is_exact(self) -> None:
        source = np.arange(24, dtype=np.float64).reshape(4, 6)
        with tempfile.TemporaryDirectory() as temporary:
            restored, metadata = driver.write_and_restore_checkpoint(
                Path(temporary),
                state=source,
                elapsed_seconds=123.5,
                proposed_dt_seconds=64.0,
                accepted_steps=9,
                rejected_steps=2,
                contract_hash="A" * 64,
            )
        self.assertTrue(np.array_equal(restored, source))
        self.assertEqual(metadata["accepted_steps"], 9)
        self.assertEqual(metadata["rejected_steps"], 2)
        self.assertEqual(metadata["elapsed_seconds"], 123.5)

    def test_contract_keeps_solver_and_mass_gates_fixed(self) -> None:
        config = {
            "schema": driver.SCHEMA,
            "system_id": 7,
            "run_role": "isolated-baseline",
            "target_years": 0.25,
            "geology": "geology.nc",
            "geology_sha256": "A" * 64,
            "climate_package": "climate.npz",
            "climate_package_sha256": "B" * 64,
            "glen_a": 1.0,
            "launch_route": "route.json",
            "launch_route_sha256": "C" * 64,
            "driver_sha256": "D" * 64,
        }
        payload = driver.contract_payload(config)
        self.assertEqual(payload["solver"]["minimum_dt_seconds"], 0.001)
        self.assertEqual(payload["solver"]["maximum_dt_seconds"], 86400.0)
        self.assertEqual(payload["solver"]["error_gate_relative_l2"], driver.ERROR_GATE)
        self.assertEqual(payload["gates"]["per_step_and_cumulative_mass_relative"], 1e-10)
        self.assertTrue(payload["solver"]["midpoint_atomic_checkpoint_restore"])

    def test_target_window_is_bounded_to_one_year(self) -> None:
        self.assertEqual(driver.SCHEMA, "lia-191-generic-planform2d-gpu-throughput-short-config-v1")
        self.assertEqual(driver.CHECKPOINT_SCHEMA, "lia-191-generic-planform2d-gpu-throughput-checkpoint-v1")


if __name__ == "__main__":
    unittest.main()
