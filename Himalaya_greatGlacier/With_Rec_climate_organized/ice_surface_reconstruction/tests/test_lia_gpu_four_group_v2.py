from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

DRIVERS = Path(__file__).resolve().parents[1] / "drivers"
sys.path.insert(0, str(DRIVERS))
from lia_gpu_four_group_search_v2 import Candidate, Search, DESIGN, load_search, ranking
from lia_gpu_matrix_io_v2 import Store, WriterLock, atomic_json, status
from lia_gpu_matrix_scoring_v2 import stability, score
from run_lia_gpu_four_group_v2 import Runtime


def record(c, group=1, stable=False, **extra):
    return {"candidate": asdict(c), "group": group,
            "spinup": {"status": "converged" if stable else "max-duration-nonconverged",
                       "domain_valid": True, "lia_error": abs(c.delta),
                       "volume_drift": .1, "length_drift_m": 400}, **extra}


class SearchTests(unittest.TestCase):
    def setUp(self):
        matrix = json.loads((DESIGN / "four_group_parameter_matrix.json").read_text())
        self.search = Search(matrix, {7: {"identity": "7"}, 79: {"identity": "79"}}, "test")

    def test_real_population_has_4011_unique_spinups(self):
        search = load_search("test")
        rows = search.candidates(1, [])
        self.assertEqual(len(rows), 4011)
        self.assertEqual(len({search.spin_key(c) for c in rows}), 4011)

    def test_spinup_identity_ignores_forward_but_not_physics(self):
        c = Candidate(7)
        self.assertEqual(self.search.spin_key(c), self.search.spin_key(replace(c, forward_end=.9)))
        for name, value in (("delta", .1), ("melt", 1.3), ("prcp", .8), ("glen", 2.)):
            self.assertNotEqual(self.search.spin_key(c), self.search.spin_key(replace(c, **{name: value})))
        self.assertNotEqual(self.search.forward_key(c, "state-a"), self.search.forward_key(c, "state-b"))

    def test_no_convergence_still_reaches_both_physics_groups(self):
        records = [record(c) for c in self.search.candidates(1, [])]
        self.assertEqual(self.search.candidates(2, records), [])
        group3 = self.search.candidates(3, records)
        self.assertEqual({c.system for c in group3}, {7, 79})
        self.assertEqual(len(group3), 180)
        self.assertEqual({c.forward_end for c in group3}, {0.})
        records.extend(record(c, 3) for c in group3)
        group4 = self.search.candidates(4, records)
        self.assertEqual({c.system for c in group4}, {7, 79})
        self.assertEqual(len(group4), 200)
        self.assertEqual(self.search.outcome(7, records, [1, 2, 3]), "search-incomplete")
        records.extend(record(c, 4) for c in group4)
        self.assertEqual(self.search.outcome(7, records, [1, 2, 3, 4]), "four-group-search-exhausted")

    def test_group1_never_needs_forward_metrics(self):
        rows = [record(Candidate(7, delta=.8), stable=True), record(Candidate(7, delta=.1), stable=True)]
        self.assertEqual(sorted(rows, key=ranking)[0]["candidate"]["delta"], .1)

    def test_group2_reuses_spinup_and_science_pass_freezes_system(self):
        rows = [record(Candidate(7), stable=True)]
        group2 = self.search.candidates(2, rows)
        self.assertEqual(len(group2), 11)
        self.assertEqual(len({self.search.spin_key(c) for c in group2}), 1)
        rows += [record(group2[0], 2, stable=True, forward={"provisional_pass": True})]
        self.assertEqual(self.search.candidates(3, rows), [])
        self.assertEqual(self.search.outcome(7, rows, [1, 2]), "provisional-pass-reproduction-required")

    def test_technical_failure_is_not_scientific_exhaustion(self):
        rows = [record(Candidate(7))]
        rows[0]["spinup"]["status"] = "technical-failure"
        self.assertEqual(self.search.candidates(3, rows), [])
        self.assertEqual(self.search.outcome(7, rows, [1, 2, 3, 4]), "technical-incomplete")

    def test_packing_is_order_independent(self):
        rows = self.search.candidates(1, [])
        self.assertEqual(self.search.pack(rows, {7: 200, 79: 100}, 8),
                         self.search.pack(list(reversed(rows)), {7: 200, 79: 100}, 8))


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = Store(self.root)
        self.key = "A" * 64

    def tearDown(self):
        self.temporary.cleanup()

    def test_partial_write_is_not_completion_and_commit_is_unique(self):
        target = self.store.directory("spin", self.key)
        target.mkdir(parents=True)
        (target / "orphan.npz").write_bytes(b"incomplete")
        self.assertIsNone(self.store.read("spin", self.key))
        self.store.commit("spin", self.key, {"status": "converged"}, {"thick": np.ones(3)})
        np.testing.assert_array_equal(self.store.arrays("spin", self.key)["thick"], np.ones(3))
        with self.assertRaises(FileExistsError):
            self.store.commit("spin", self.key, {})

    def test_corrupt_committed_state_is_rejected(self):
        result = self.store.commit("spin", self.key, {}, {"thick": np.ones(3)})
        artifact = self.store.directory("spin", self.key) / result["artifacts"][0]["name"]
        artifact.write_bytes(b"broken")
        with self.assertRaises(ValueError):
            self.store.read("spin", self.key)

    def test_checkpoint_history_and_lane_state_restore(self):
        arrays = {"thick": np.arange(7.), "history": np.arange(40.).reshape(8, 5), "years": np.arange(8)}
        self.store.checkpoint("spin", self.key, 330, {"counters": {"steps": 99}}, arrays)
        result, loaded = self.store.latest("spin", self.key)
        self.assertEqual(result["year"], 330)
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], loaded[name])

    def test_status_lock_never_raises(self):
        with patch("lia_gpu_matrix_io_v2.atomic_json", side_effect=PermissionError("locked")):
            self.assertFalse(status(self.root / "status.json", {"year": 1}))

    def test_one_writer_and_release(self):
        with WriterLock(self.root):
            with self.assertRaises(RuntimeError):
                with WriterLock(self.root):
                    pass
        with WriterLock(self.root):
            pass


class ScoringTests(unittest.TestCase):
    def test_same_phase_drift_uses_metres_and_three_percent(self):
        h = {year: [1000 + year*.1, 1e6, 1e8 * (1 + year*.00001)] for year in range(0, 331, 30)}
        self.assertTrue(stability(h, 300)["passed"])
        h[300][0] += 400
        self.assertFalse(stability(h, 300)["passed"])

    def test_constant_curve_fails_shape_even_with_endpoints(self):
        h = {y: [10000., 1e6, 1e8, 0., 0.] for y in range(1470, 2001)}
        value = score(h, 10000., 10000.)
        self.assertFalse(value["provisional_pass"])
        self.assertTrue(value["gates"]["lia"])
        self.assertTrue(value["gates"]["rgi"])
        self.assertFalse(value["gates"]["phases"])

    def test_missing_forward_year_not_silently_interpolated(self):
        with self.assertRaises(ValueError):
            score({1470: [1, 1, 1]}, 1, 1)


class OrchestrationTests(unittest.TestCase):
    def test_full_search_and_recovery_with_fake_backend(self):
        matrix = json.loads((DESIGN / "four_group_parameter_matrix.json").read_text())
        search = Search(matrix, {7: {}, 79: {}}, "synthetic-test-only")
        manifest = {"systems": [{"system": s, "nx": 8} for s in (7, 79)]}
        class FakeRuntime(Runtime):
            computations = 0
            def execute(self, candidates, kind):
                values = {}
                for c in candidates:
                    key = self.key(kind, c)
                    value = self.cache(kind, key)
                    if value is None:
                        self.computations += 1
                        value = ({"status": "converged" if c.system == 7 else "max-duration-nonconverged",
                                  "domain_valid": True, "lia_error": abs(c.delta), "volume_drift": .1,
                                  "length_drift_m": 400} if kind == "spin" else
                                 {"status": "completed", "provisional_pass": True})
                        value = self.persist(kind, key, value, {"thick": np.ones(8)})
                    values[c.key] = value
                return values
        with tempfile.TemporaryDirectory() as directory:
            runtime = FakeRuntime(search, manifest, {}, Path(directory), "unused")
            report = runtime.run()
            self.assertEqual(report["outcomes"]["7"], "provisional-pass-reproduction-required")
            self.assertEqual(report["outcomes"]["79"], "four-group-search-exhausted")
            self.assertEqual(len(report["provisional_passes"]), 110)
            restored = FakeRuntime(search, manifest, {}, Path(directory), "unused")
            self.assertEqual(restored.run()["outcomes"], report["outcomes"])
            self.assertEqual(restored.computations, 0)


if __name__ == "__main__":
    unittest.main()
