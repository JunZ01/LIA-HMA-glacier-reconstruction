from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'drivers'))
from analyze_lia_gpu_inactive_operator_v1 import active_prefix,assemble,solve,pcr_numpy


class AlgebraTests(unittest.TestCase):
    def test_remote_accumulation_is_not_discarded(self):
        h=np.zeros(20);h[:3]=1
        mb=np.zeros(20);mb[17]=1
        self.assertEqual(active_prefix(h,mb),19)

    def test_front_neighbor_is_retained(self):
        h=np.zeros(20);h[:5]=1
        self.assertEqual(active_prefix(h,-np.ones(20)),6)

    def test_zero_ice_ablation_clamps_to_zero(self):
        h=np.zeros(20);bed=np.arange(20,dtype=float)[::-1]
        coeff=assemble(h,bed,np.ones(20),50,1e-23,-np.ones(20),1)
        np.testing.assert_array_equal(np.maximum(solve(coeff),0),h)

    def test_reduced_operator_and_pcr(self):
        h=np.r_[np.array([20.,15.,10.,5.]),np.zeros(16)]
        mb=-np.ones(20)*1e-8
        coeff=assemble(h,np.linspace(3000,2000,20),np.ones(20)*100,50,1e-23,mb,108000)
        cut=active_prefix(h,mb)
        full=np.maximum(solve(coeff),0)
        reduced=np.zeros(20);reduced[:cut]=np.maximum(solve(tuple(x[:cut] for x in coeff)),0)
        np.testing.assert_array_equal(full,reduced)
        np.testing.assert_allclose(pcr_numpy(coeff),solve(coeff),rtol=1e-12,atol=1e-12)


if __name__=='__main__':
    unittest.main()
