from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drivers"))
from lia_gpu_four_group_search_v2 import Candidate, Search, DESIGN
from run_lia_gpu_four_group_v2 import Runtime
from torch_flowline_1d_matrix_v2 import make_state, compact, snapshot
from torch_flowline_1d_batched_spinup_repeat_v1 import BatchedFlowlineStateSpinupRepeat
import json


class MatrixBackendTests(unittest.TestCase):
    def setUp(self):
        self.base = {s: {"base_temp_bias_degC": 0., "melt_f": 1., "prcp_fac": 1.,
                         "glen_a_pa_3_s_1": 0., "temp_melt_degC": -1.} for s in (7, 79)}
        self.bundle = {"system_bed_h": np.zeros((2, 12)), "system_widths_m": np.ones((2, 12))*100,
                       "system_thick": np.array([[10.] + [0.]*11]*2),
                       "system_valid_mask": np.array([[True]*8+[False]*4, [True]*12]),
                       "system_nx": np.array([8, 12]), "system_dx_meter": np.ones(2)*1000,
                       "system_climate_ref_hgt": np.zeros(2), "climate_years": np.arange(1441, 1471),
                       "forward_years": np.arange(1470, 2001)}
        for prefix, count in (("system_climate_", 30), ("forward_climate_", 531)):
            self.bundle[prefix+"temp"] = np.ones((2, count, 12))*.5
            self.bundle[prefix+"prcp"] = np.ones((2, count, 12))*100
            self.bundle[prefix+"gradient"] = np.zeros((2, count, 12))
        self.index = {7: 0, 79: 1}

    def state(self, cells, kind, year=None, **kw):
        return make_state(cells, self.base, self.bundle, self.index, torch.device("cpu"), kind, year=year, **kw)

    def test_trim_padding_per_batch(self):
        self.assertEqual(self.state([Candidate(7)], "spin").thick.shape, (1, 8))
        self.assertEqual(self.state([Candidate(7), Candidate(79)], "spin").thick.shape, (2, 12))

    def test_forward_excludes_equilibrium_bias_and_has_endpoint_ramp(self):
        a = Candidate(7, delta=-1., forward_end=.9)
        b = replace(a, delta=1.)
        state = self.state([a, b], "forward")
        torch.testing.assert_close(state.annual_mass_balance()[0], state.annual_mass_balance()[1], atol=0, rtol=0)
        middle = self.state([a], "forward", year=1735)
        expected = self.state([replace(a, forward_end=0.)], "forward", year=1735)
        expected.temp_bias.fill_(.45)
        torch.testing.assert_close(middle.annual_mass_balance(), expected.annual_mass_balance(), atol=0, rtol=0)
        end = self.state([a], "forward", year=2000)
        expected = self.state([replace(a, forward_end=0.)], "forward", year=2000)
        expected.temp_bias.fill_(.9)
        torch.testing.assert_close(end.annual_mass_balance(), expected.annual_mass_balance(), atol=0, rtol=0)

    def test_compaction_preserves_cached_mb_and_forward_parameters(self):
        state = self.state([Candidate(7, forward_end=-.9), Candidate(79, forward_end=.9)], "forward", year=1800)
        mb = state.annual_mass_balance()[1].clone()
        compact(state, [1])
        torch.testing.assert_close(state.annual_mass_balance()[0], mb, rtol=0, atol=0)
        self.assertEqual(state.forward_end.tolist(), [.9])

    def test_integer_year_restore_matches_short_synthetic_continuation(self):
        cells = [Candidate(7)]
        state = self.state(cells, "forward")
        state.run_until(1471.)
        _, h, totals, _, _ = snapshot(state)
        resumed = self.state(cells, "forward", year=1471, thick=h, totals=totals)
        state.run_until(1471.01)
        resumed.run_until(1471.01)
        torch.testing.assert_close(state.thick, resumed.thick, rtol=1e-10, atol=1e-6)

    def test_executor_checkpoint_recovery_with_synthetic_clock(self):
        matrix = json.loads((DESIGN / "four_group_parameter_matrix.json").read_text())
        search = Search(matrix, {7: self.base[7]}, "synthetic-clock-only")
        manifest = {"systems": [{"system": 7, "nx": 8, "target_lia": 1000., "target_rgi": 900.}]}
        candidate = Candidate(7)
        interrupted = {"enabled": True}
        def clock(state, year):
            if interrupted["enabled"] and year == 660:
                raise RuntimeError("injected interruption")
            difference = year - state.year
            state.thick[:, 0] += difference
            state.time_seconds = year * 31536000
            state.accepted_steps += int(difference * 300)
            state.minimum_actual_timestep_seconds = 108000.
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            with patch.object(BatchedFlowlineStateSpinupRepeat, "run_until", clock):
                first = Runtime(search, manifest, self.bundle, Path(a), torch.device("cpu"))
                with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                    first.execute([candidate], "spin")
                self.assertEqual(first.store.latest("spin", search.spin_key(candidate))[0]["year"], 630)
                interrupted["enabled"] = False
                resumed = Runtime(search, manifest, self.bundle, Path(a), torch.device("cpu"))
                continuous = Runtime(search, manifest, self.bundle, Path(b), torch.device("cpu"))
                result = resumed.execute([candidate], "spin")[candidate.key]
                reference = continuous.execute([candidate], "spin")[candidate.key]
                self.assertEqual(result["year"], 1500)
                self.assertEqual(result["counters"], reference["counters"])
                for name in ("thick", "totals", "years", "history"):
                    np.testing.assert_array_equal(resumed.store.arrays("spin", search.spin_key(candidate))[name],
                                                  continuous.store.arrays("spin", search.spin_key(candidate))[name])


if __name__ == "__main__":
    unittest.main()
