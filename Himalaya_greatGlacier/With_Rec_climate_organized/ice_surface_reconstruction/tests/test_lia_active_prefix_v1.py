import copy
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'drivers'))
from torch_flowline_1d_active_prefix_v1 import make_optimized_state, cached_pcr, solve_width
from torch_flowline_1d_batched_semiimplicit_v1 import pcr_tridiagonal_solve
from torch_flowline_1d_matrix_v2 import make_state, compact, snapshot
from lia_gpu_four_group_search_v2 import Candidate


def synthetic(lanes=2, width=64, occupied=12, device='cpu', kind='spin', optimized=False):
    base = {7: dict(base_temp_bias_degC=0., melt_f=1., prcp_fac=1., glen_a_pa_3_s_1=2.4e-24, temp_melt_degC=-1.)}
    bed = np.linspace(3000, 0, width)[None, :]
    h = np.zeros((1, width))
    h[0, :occupied] = np.linspace(70., 1., occupied)
    bundle = dict(system_bed_h=bed, system_widths_m=np.full((1,width),100.), system_thick=h,
        system_valid_mask=np.ones((1,width),bool), system_nx=np.array([width]),
        system_dx_meter=np.array([50.]), system_climate_ref_hgt=np.array([0.]),
        climate_years=np.arange(1441,1471), forward_years=np.arange(1470,2001))
    for prefix, count in (('system_climate_',30), ('forward_climate_',531)):
        bundle[prefix+'temp'] = np.full((1,count,12), 3.)
        bundle[prefix+'prcp'] = np.zeros((1,count,12))
        bundle[prefix+'gradient'] = np.zeros((1,count,12))
    factory = make_optimized_state if optimized else make_state
    return factory([Candidate(7, forward_end=.3)]*lanes,base,bundle,{7:0},torch.device(device),kind)


class ActivePrefixTests(unittest.TestCase):
    device = os.environ.get('LIA_TEST_DEVICE','cpu')

    def pair(self, kind='spin'):
        return synthetic(device=self.device,kind=kind), synthetic(device=self.device,kind=kind,optimized=True)

    def assert_state(self, a, b):
        for name in ('thick','cumulative_climatic_mass_balance_m3','cumulative_mass_budget_residual_m3'):
            torch.testing.assert_close(getattr(a,name),getattr(b,name),atol=1e-6,rtol=1e-10,equal_nan=True)
        for name in ('accepted_steps','rejected_steps','time_seconds','minimum_actual_timestep_seconds'):
            self.assertEqual(getattr(a,name),getattr(b,name))
        for name in ('length_m','area_m2','volume_m3'):
            torch.testing.assert_close(a.diagnostics()[name],b.diagnostics()[name],atol=1e-6,rtol=1e-10)

    def test_pcr_cached_matches_masked_reference(self):
        for n in (3,17,64,65):
            a = torch.full((2,n),-.1,dtype=torch.float64,device=self.device)
            c = a.clone(); a[:,0]=0; c[:,-1]=0
            b = torch.ones_like(a)*2; d = torch.arange(n,device=self.device,dtype=torch.float64)[None,:].expand(2,-1)
            mask = torch.ones_like(a,dtype=torch.bool); mask[0,n//2:]=False; c[0,n//2-1]=0
            torch.testing.assert_close(cached_pcr(a,b,c,d,mask),pcr_tridiagonal_solve(a,b,c,d,mask),atol=0,rtol=0)

    def test_step_and_full_domain_budget(self):
        a,b = self.pair()
        a.step(108000); b.step(108000)
        self.assert_state(a,b)
        self.assertLess(b.last_solve_width,b.thick.shape[1])
        self.assertEqual(a.thick.shape,b.thick.shape)

    def test_repeated_steps_and_monthly_calendar(self):
        a,b = self.pair()
        a.run_until(.2); b.run_until(.2)
        self.assert_state(a,b)

    def test_remote_accumulation_kept(self):
        a,b = self.pair()
        for state in (a,b):
            mb = state.annual_mass_balance()
            mb[:,50]=1e-6
            state.step(108000)
        self.assertGreaterEqual(b.last_solve_width,52)
        self.assertGreater(float(b.thick[0,50].item()),0)
        self.assert_state(a,b)

    def test_front_can_advance(self):
        a,b = self.pair()
        for state in (a,b):
            state.annual_mass_balance().zero_()
            state.step(108000)
        self.assertGreater(float(b.thick[0,12].item()),0)
        a.step(108000); b.step(108000)
        self.assertGreaterEqual(b.last_solve_width,14)
        self.assert_state(a,b)

    def test_empty_ice(self):
        a,b = self.pair()
        for state in (a,b):
            state.thick.zero_(); state.step(36000)
        self.assert_state(a,b)
        self.assertEqual(b.last_solve_width,3)

    def test_nonfinite_not_hidden(self):
        state = self.pair()[1]
        state.thick[0,50]=float('nan')
        self.assertEqual(solve_width(state.thick,state.annual_mass_balance(),state.valid_mask),64)
        state.step(36000)
        self.assertFalse(bool(torch.isfinite(state.thick).all().item()))

    def test_compaction(self):
        a,b = self.pair()
        a.step(108000); b.step(108000)
        compact(a,[1]); compact(b,[1])
        a.step(36000); b.step(36000)
        self.assert_state(a,b)

    def test_forward_bias_and_annual_cache(self):
        a,b = self.pair('forward')
        for state in (a,b):
            state.time_seconds=265*31536000.
            state.run_until(1735.01)
        self.assert_state(a,b)

    def test_integer_boundary_checkpoint_restart(self):
        a = synthetic(device=self.device,optimized=True)
        a.time_seconds=31536000.
        _,h,totals,_,_=snapshot(a)
        b = synthetic(device=self.device,optimized=True)
        b.thick=torch.as_tensor(h.copy(),device=self.device)
        b.time_seconds=a.time_seconds
        b.cumulative_climatic_mass_balance_m3=torch.as_tensor(totals[:,0].copy(),device=self.device)
        b.cumulative_mass_budget_residual_m3=torch.as_tensor(totals[:,1].copy(),device=self.device)
        a.run_until(1.1); b.run_until(1.1)
        self.assert_state(a,b)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
