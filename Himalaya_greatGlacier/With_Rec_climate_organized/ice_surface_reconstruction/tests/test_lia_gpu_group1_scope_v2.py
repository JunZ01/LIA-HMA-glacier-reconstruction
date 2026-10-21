"""Scope-only wrapper checks; no real model run."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'drivers'))
from launch_lia_gpu_group1_v2 import group_one
from lia_gpu_four_group_search_v2 import load_search


class ScopeTest(unittest.TestCase):
    def test_only_spinup_once_and_all_fixed_members(self):
        search = load_search('scope-test')
        candidates = search.candidates(1, [])
        runtime = Mock()
        runtime.execute.return_value = {c.key: {'status':'max-duration-nonconverged'} for c in candidates}
        rows = group_one(runtime, search)
        runtime.execute.assert_called_once_with(candidates, 'spin')
        runtime.run.assert_not_called()
        self.assertEqual(len(rows), 4011)
        self.assertEqual(len({r['candidate']['system'] for r in rows}), 191)
        self.assertTrue(all(r['group'] == 1 and r['forward'] == {} for r in rows))


if __name__ == '__main__':
    unittest.main()
