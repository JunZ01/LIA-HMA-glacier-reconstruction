import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'drivers'))
from lia_gpu_four_group_search_v2 import file_hash
from lia_gpu_matrix_io_v2 import Store, atomic_json
from lia_gpu_prefix_recovery_store_v1 import RecoveryStore


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = Store(self.base / 'source')
        self.key = 'A' * 64
        self.other = 'B' * 64
        self.arrays = dict(thick=np.arange(10.), totals=np.array([1.,2.]),
                           years=np.array([0,330]), history=np.ones((2,5)))

    def tearDown(self):
        self.temp.cleanup()

    def inherited(self, mode):
        if mode == 'terminal':
            result = self.source.commit('spin', self.key, dict(status='converged'), self.arrays)
            directory = self.source.directory('spin', self.key)
            entry = dict(mode=mode, result=result)
        else:
            result = self.source.checkpoint('spin', self.key,330,dict(counters={}),self.arrays)
            directory = self.source.root / 'resume' / 'spin' / self.key[:16] / 'y0330' / 'checkpoint' / self.key[:16]
            entry = dict(mode=mode,year=330)
        paths = [directory/'result.json'] + [directory/a['name'] for a in result['artifacts']]
        entry['files'] = [dict(path=str(p.relative_to(self.source.root)),sha256=file_hash(p)) for p in paths]
        self.entries = {self.key:entry, self.other:dict(mode='new',files=[])}
        return RecoveryStore(self.base/'new',self.source.root,self.entries)

    def test_completed_is_read_only_and_cannot_recompute(self):
        store = self.inherited('terminal')
        self.assertEqual(store.read('spin',self.key)['status'],'converged')
        np.testing.assert_array_equal(store.arrays('spin',self.key)['thick'],self.arrays['thick'])
        with self.assertRaises(FileExistsError):
            store.commit('spin',self.key,dict(status='converged'),self.arrays)
        self.assertFalse((self.base/'new'/'spin').exists())

    def test_checkpoint_fallback_then_local_precedence(self):
        store = self.inherited('checkpoint')
        self.assertEqual(store.latest('spin',self.key)[0]['year'],330)
        store.checkpoint('spin',self.key,630,dict(counters={}),self.arrays)
        self.assertEqual(store.latest('spin',self.key)[0]['year'],630)
        self.assertEqual(self.source.latest('spin',self.key)[0]['year'],330)
        store.commit('spin',self.key,dict(status='converged'),self.arrays)
        self.assertIsNone(self.source.read('spin',self.key))

    def test_corruption_and_outside_scope_rejected(self):
        store = self.inherited('terminal')
        with self.assertRaises(ValueError):
            store.read('forward',self.key)
        with self.assertRaises(ValueError):
            store.read('spin','C'*64)
        p = self.source.root / self.entries[self.key]['files'][0]['path']
        atomic_json(p,dict(key=self.key,status='changed'))
        with self.assertRaises(ValueError):
            store.read('spin',self.key)

    def test_new_candidate_and_technical_failure_are_not_retried(self):
        store = self.inherited('terminal')
        self.assertIsNone(store.latest('spin',self.other))
        store.commit('spin',self.other,dict(status='technical-failure'))
        self.assertEqual(store.read('spin',self.other)['status'],'technical-failure')
        with self.assertRaises(FileExistsError):
            store.commit('spin',self.other,dict(status='converged'))


if __name__ == '__main__':
    unittest.main()
