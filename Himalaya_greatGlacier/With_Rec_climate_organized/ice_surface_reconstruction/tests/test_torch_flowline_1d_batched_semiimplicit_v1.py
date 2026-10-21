from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "drivers" / "torch_flowline_1d_batched_semiimplicit_v1.py"
SPEC = importlib.util.spec_from_file_location("torch_flowline_1d_batched_semiimplicit_v1", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TorchFlowline1DTests(unittest.TestCase):
    def test_monthly_schedule_matches_fixed_policy(self):
        schedule = MODULE.monthly_step_schedule(1836.0, 1837.0)
        self.assertEqual(len(schedule), 300)
        self.assertEqual(min(schedule), 36_000.0)
        self.assertEqual(max(schedule), 108_000.0)
        self.assertEqual(sum(schedule), MODULE.SEC_IN_YEAR)

    def test_pcr_batched_ragged_matches_dense_torch(self):
        torch.manual_seed(42)
        batch, width = 4, 17
        nx = torch.tensor([1, 5, 9, 17])
        mask = torch.arange(width)[None, :] < nx[:, None]
        lower = -0.25 * torch.rand(batch, width, dtype=torch.float64)
        upper = -0.25 * torch.rand(batch, width, dtype=torch.float64)
        diagonal = 2.0 + 0.5 * torch.rand(batch, width, dtype=torch.float64)
        rhs = torch.randn(batch, width, dtype=torch.float64)
        lower[:, 0] = 0.0
        for lane, n_value in enumerate(nx):
            upper[lane, int(n_value) - 1] = 0.0
        got = MODULE.pcr_tridiagonal_solve(lower, diagonal, upper, rhs, mask)
        for lane, n_value in enumerate(nx):
            n = int(n_value)
            matrix = torch.diag(diagonal[lane, :n])
            if n > 1:
                matrix += torch.diag(upper[lane, : n - 1], diagonal=1)
                matrix += torch.diag(lower[lane, 1:n], diagonal=-1)
            expected = torch.linalg.solve(matrix, rhs[lane, :n])
            torch.testing.assert_close(got[lane, :n], expected, rtol=2e-13, atol=2e-13)
            self.assertEqual(int(torch.count_nonzero(got[lane, n:])), 0)

    def test_module_is_independent_of_2d_and_igm(self):
        text = MODULE_PATH.read_text(encoding="utf-8").lower()
        self.assertNotIn("planform_2d", text)
        self.assertNotIn("import igm", text)
        self.assertNotIn("from igm", text)

    def test_rectangular_width_section_and_padding_diagnostics(self):
        zeros_2d = torch.zeros((2, 4), dtype=torch.float64)
        years = torch.tensor([[1836], [1836]], dtype=torch.int64)
        state = MODULE.BatchedFlowlineState(
            bed_h=zeros_2d.clone(), widths_m=torch.tensor([[10., 20., 30., 1.], [4., 5., 1., 1.]], dtype=torch.float64),
            thick=torch.tensor([[2., 0., 3., 0.], [1., 2., 0., 0.]], dtype=torch.float64),
            valid_mask=torch.tensor([[True, True, True, False], [True, True, False, False]]),
            nx=torch.tensor([3, 2]), dx_meter=torch.tensor([50., 25.], dtype=torch.float64),
            glen_a=torch.ones(2, dtype=torch.float64) * 2.4e-24,
            melt_f=torch.ones(2, dtype=torch.float64), prcp_fac=torch.ones(2, dtype=torch.float64),
            temp_bias=torch.zeros(2, dtype=torch.float64), temp_melt=-torch.ones(2, dtype=torch.float64),
            climate_temp=torch.zeros((2, 1, 12), dtype=torch.float64),
            climate_prcp=torch.zeros((2, 1, 12), dtype=torch.float64),
            climate_gradient=torch.zeros((2, 1, 12), dtype=torch.float64),
            climate_ref_hgt=torch.zeros(2, dtype=torch.float64), climate_years=years,
        )
        diagnostics = state.diagnostics()
        torch.testing.assert_close(diagnostics["volume_m3"], torch.tensor([5500., 350.], dtype=torch.float64))
        torch.testing.assert_close(diagnostics["area_m2"], torch.tensor([2000., 225.], dtype=torch.float64))
        torch.testing.assert_close(diagnostics["length_m"], torch.tensor([100., 50.], dtype=torch.float64))
        self.assertEqual(int(torch.count_nonzero(state.thick[:, -1])), 0)


if __name__ == "__main__":
    unittest.main()
