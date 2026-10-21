"""Combined synthetic verification for the phase-1 large-step kernels.

These tests do not read a glacier checkpoint or advance project model time.
They exercise small, analytically auditable control-volume systems only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import tempfile
import unittest

import numpy as np

from .large_step_coupling_v1 import CouplingConvergenceError, solve_coupled_large_step
from .large_step_transport_v1 import (
    ExistingMeshFluxAdapter,
    FixedVelocityRates,
    TransportTopology,
    diffusion_predict,
    implicit_conservative_step,
)


class _StaticFluxAdapter:
    def __init__(self, rates: FixedVelocityRates):
        self.topology = rates.topology
        self._rates = rates

    def rates(self, velocity):
        del velocity
        return self._rates


class Phase1KernelTests(unittest.TestCase):
    def test_internal_face_is_shared_and_conservative(self):
        topology = TransportTopology(
            dual_area=np.ones(2),
            internal_edges=np.array([[0, 1]]),
            boundary_donors=np.empty(0, dtype=np.int64),
        )
        rates = FixedVelocityRates(topology, np.array([1.0]), np.empty(0))
        result = implicit_conservative_step(
            np.array([1.0, 0.0]), np.zeros(2), 1.0, rates
        )
        np.testing.assert_allclose(result.thickness, [0.5, 0.5], atol=2e-15)
        self.assertLessEqual(result.mass_relative, 2e-15)
        self.assertLessEqual(result.local_residual_l1_relative, 2e-15)

    def test_boundary_outflow_is_in_the_volume_ledger(self):
        topology = TransportTopology(
            dual_area=np.ones(1),
            internal_edges=np.empty((0, 2), dtype=np.int64),
            boundary_donors=np.array([0]),
        )
        rates = FixedVelocityRates(topology, np.empty(0), np.array([2.0]))
        result = implicit_conservative_step(
            np.array([1.0]), np.zeros(1), 1.0, rates
        )
        self.assertAlmostEqual(result.thickness[0], 1.0 / 3.0, places=14)
        self.assertAlmostEqual(result.boundary_export_volume, 2.0 / 3.0, places=14)
        self.assertLessEqual(result.mass_relative, 2e-15)

    def test_ablation_uses_same_step_inflow_and_records_shortfall(self):
        topology = TransportTopology(
            dual_area=np.ones(2),
            internal_edges=np.array([[0, 1]]),
            boundary_donors=np.empty(0, dtype=np.int64),
        )
        rates = FixedVelocityRates(topology, np.array([1.0]), np.empty(0))
        result = implicit_conservative_step(
            np.array([1.0, 0.0]), np.array([0.0, -0.75]), 1.0, rates
        )
        np.testing.assert_allclose(result.thickness, [0.5, 0.0], atol=2e-15)
        np.testing.assert_allclose(result.actual_smb, [0.0, -0.5], atol=2e-15)
        np.testing.assert_allclose(
            result.unmet_ablation_volume, [0.0, 0.25], atol=2e-15
        )
        self.assertLessEqual(result.mass_relative, 2e-15)

    def test_deterministic_mixed_flow_ensemble_meets_all_kernel_gates(self):
        rng = np.random.default_rng(2387)
        topology = TransportTopology(
            dual_area=rng.uniform(0.5, 2.0, 9),
            internal_edges=np.column_stack([np.arange(8), np.arange(1, 9)]),
            boundary_donors=np.array([0, 8]),
        )
        for _ in range(40):
            rates = FixedVelocityRates(
                topology,
                rng.uniform(-2.0, 2.0, 8),
                rng.uniform(0.0, 0.3, 2),
            )
            result = implicit_conservative_step(
                rng.uniform(0.0, 3.0, 9),
                rng.uniform(-1.0, 0.5, 9),
                rng.uniform(0.05, 1.0),
                rates,
            )
            self.assertGreaterEqual(float(result.thickness.min()), 0.0)
            self.assertLessEqual(result.mass_relative, 1e-10)
            self.assertLessEqual(result.local_residual_l1_relative, 1e-12)
            self.assertTrue(np.all(result.actual_smb >= -1.0 - 1e-13))

    def test_diffusion_predictor_preserves_constant_surface(self):
        topology = TransportTopology(
            dual_area=np.ones(2),
            internal_edges=np.array([[0, 1]]),
            boundary_donors=np.empty(0, dtype=np.int64),
        )
        predicted = diffusion_predict(
            thickness_old=np.array([1.0, 0.0]),
            bed=np.array([0.0, 1.0]),
            smb_requested=np.zeros(2),
            dt_years=1.0,
            topology=topology,
            conductance=np.array([3.0]),
        )
        np.testing.assert_allclose(predicted, [1.0, 0.0], atol=2e-15)

    def test_existing_mesh_adapter_matches_gpu_flux_formula(self):
        arrays = {
            "geo_dual_area": np.ones(2),
            "geo_edges": np.array([[0, 1]], dtype=np.int64),
            "geo_outer_a": np.empty(0, dtype=np.int64),
            "geo_outer_b": np.empty(0, dtype=np.int64),
            "geo_a": np.array([0], dtype=np.int64),
            "geo_b": np.array([1], dtype=np.int64),
            "geo_third": np.array([0], dtype=np.int64),
            "geo_inverse": np.array([0], dtype=np.int64),
            "geo_normal": np.array([[1.0, 0.0]]),
            "geo_outer_normal": np.empty((0, 2)),
        }
        velocity = np.zeros((2, 2, 2))
        velocity[0, 0, 0] = 2.0  # depth mean is 1 m/yr at node zero
        adapter = ExistingMeshFluxAdapter(arrays)
        rates = adapter.rates(velocity.reshape(-1))
        self.assertAlmostEqual(rates.internal_signed[0], 7.0 / 12.0, places=15)

    def test_outer_bp_correction_accepts_only_joint_fixed_point(self):
        topology = TransportTopology(
            dual_area=np.ones(2),
            internal_edges=np.array([[0, 1]]),
            boundary_donors=np.empty(0, dtype=np.int64),
        )
        rates = FixedVelocityRates(topology, np.array([1.0]), np.empty(0))
        adapter = _StaticFluxAdapter(rates)
        calls = []

        def bp_solve(h, guess):
            calls.append(h.copy())
            return guess, {
                "relative_residual": 1e-9,
                "iterations": 1,
                "history": [{"cg_iterations": 2}],
            }

        result = solve_coupled_large_step(
            thickness_old=np.array([1.0, 0.0]),
            velocity_old=np.zeros(8),
            bed=np.zeros(2),
            smb_requested=np.zeros(2),
            dt_years=1.0,
            flux_adapter=adapter,
            bp_solve=bp_solve,
            use_diffusion_predictor=False,
            damping=1.0,
            maximum_bp_calls=8,
        )
        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(result.thickness, [0.5, 0.5], atol=2e-15)
        self.assertLessEqual(result.bp_relative, 1e-6)
        self.assertLessEqual(result.thickness_equation_relative, 1e-8)
        self.assertLessEqual(result.mass_relative, 1e-10)

    def test_diffusion_is_only_an_initial_guess(self):
        topology = TransportTopology(
            dual_area=np.ones(2),
            internal_edges=np.array([[0, 1]]),
            boundary_donors=np.empty(0, dtype=np.int64),
        )
        rates = FixedVelocityRates(topology, np.array([1.0]), np.empty(0))
        adapter = _StaticFluxAdapter(rates)

        def bp_solve(h, guess):
            del h
            return guess, {"relative_residual": 0.0, "iterations": 0, "history": []}

        result = solve_coupled_large_step(
            thickness_old=np.array([1.0, 0.0]),
            velocity_old=np.zeros(8),
            bed=np.zeros(2),
            smb_requested=np.zeros(2),
            dt_years=1.0,
            flux_adapter=adapter,
            bp_solve=bp_solve,
            use_diffusion_predictor=True,
            damping=1.0,
        )
        self.assertEqual(result.predictor, "implicit-local-diffusion")
        np.testing.assert_allclose(result.thickness, [0.5, 0.5], atol=2e-15)

    def test_frozen_point_seven_damping_cannot_meet_budget_on_constant_map(self):
        topology = TransportTopology(
            dual_area=np.ones(2),
            internal_edges=np.array([[0, 1]]),
            boundary_donors=np.empty(0, dtype=np.int64),
        )
        adapter = _StaticFluxAdapter(
            FixedVelocityRates(topology, np.array([1.0]), np.empty(0))
        )

        def bp_solve(h, guess):
            del h
            return guess, {"relative_residual": 0.0, "iterations": 0, "history": []}

        with self.assertRaises(CouplingConvergenceError) as caught:
            solve_coupled_large_step(
                np.array([1.0, 0.0]),
                np.zeros(8),
                np.zeros(2),
                np.zeros(2),
                1.0,
                adapter,
                bp_solve,
                use_diffusion_predictor=False,
                damping=0.7,
                maximum_bp_calls=8,
            )
        self.assertEqual(len(caught.exception.history), 8)
        self.assertGreater(
            caught.exception.history[-1]["fixed_point_relative"], 1e-8
        )

    def test_gpu_bridge_is_lazy_at_import(self):
        # Importing the registered worker-side adapter must not initialize CUDA
        # or load the CPU hierarchy helper in this synthetic verification.
        from .large_step_gpu_bridge_v1 import ProjectedAdaptiveBPSolver

        self.assertTrue(callable(ProjectedAdaptiveBPSolver))

    def test_isolated_transport_service_matches_reference_kernel(self):
        from .large_step_transport_client_v1 import IsolatedTransportService

        arrays = {
            "geo_dual_area": np.ones(2),
            "geo_edges": np.array([[0, 1]], dtype=np.int64),
            "geo_outer_a": np.empty(0, dtype=np.int64),
            "geo_outer_b": np.empty(0, dtype=np.int64),
            "geo_a": np.array([0], dtype=np.int64),
            "geo_b": np.array([1], dtype=np.int64),
            "geo_third": np.array([0], dtype=np.int64),
            "geo_inverse": np.array([0], dtype=np.int64),
            "geo_normal": np.array([[12.0 / 7.0, 0.0]]),
            "geo_outer_normal": np.empty((0, 2)),
        }
        velocity = np.zeros((2, 2, 2))
        velocity[0, 0, 0] = 2.0
        with tempfile.TemporaryDirectory() as folder:
            with IsolatedTransportService(arrays, folder) as service:
                answer = service.correct(
                    np.array([1.0, 0.0]),
                    np.array([0.5, 0.5]),
                    velocity.reshape(-1),
                    np.zeros(2),
                    1.0,
                )
        np.testing.assert_allclose(answer["thickness"], [0.5, 0.5], atol=2e-15)
        self.assertLessEqual(float(answer["mass_relative"]), 2e-15)
        self.assertLessEqual(float(answer["thickness_equation_relative"]), 2e-15)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Phase1KernelTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.json:
        root = Path(__file__).resolve().parent
        record = {
            "schema": "polim-large-step-phase1-synthetic-verification-v1",
            "status": "pass" if result.wasSuccessful() else "fail",
            "simulation_launched": False,
            "synthetic_tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "seconds": time.perf_counter() - started,
            "files": {
                name: _sha256(root / name)
                for name in [
                    "large_step_transport_v1.py",
                    "large_step_coupling_v1.py",
                    "large_step_gpu_bridge_v1.py",
                    "large_step_gpu_coupling_v1.py",
                    "large_step_transport_client_v1.py",
                    "large_step_transport_service_v1.py",
                    "large_step_phase1_verify_v1.py",
                ]
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
