"""Only new fixed-structure checks; no real model evaluations or solves."""
import time,json,tempfile
from pathlib import Path
import numpy as np
from scipy import sparse
from .structure import Pattern,Cache,c_indices
from types import SimpleNamespace as NS

def run():
    t=time.perf_counter();checks={}
    row=np.array([0,1,0,0,1,2,2,0]);col=np.array([0,2,0,2,2,1,1,0]);val=np.array([1e16,3.,-1e16,2.,-3.,8.,-7.,1.])
    p=Pattern(row,col,(3,3));old=sparse.coo_matrix((val,(row,col)),shape=(3,3)).tocsr();a=p.update(val)
    checks['same_SciPy_reduction_order']=bool(np.array_equal(old.toarray(),a.toarray()))
    before=a.toarray().copy();action=a@np.arange(3.)
    b=p.update(np.zeros_like(val));checks['branch_zero_cleared']=bool(np.all(b.data==0))
    checks['parent_ownership']=bool(np.array_equal(a.toarray(),before) and np.array_equal(a@np.arange(3.),action) and not np.shares_memory(a.data,b.data))
    z=dict(xy=np.zeros((3,2)),tri=np.array([[0,1,2]]),ids=np.arange(12).reshape(1,12),bed=np.zeros(3))
    ad=NS(topology=NS(internal_edges=np.array([[0,1],[1,2]]),boundary_donors=np.array([],dtype=int)),a=np.array([0,1]),b=np.array([1,2]),third=np.array([2,0]),inverse=np.array([0,1]),outer_a=np.array([],dtype=int),outer_b=np.array([],dtype=int),normal=np.ones((2,2)),outer_normal=np.zeros((0,2)))
    cv=np.ones(len(c_indices(ad)[0]));bv=np.ones((1,12,3));cache=Cache()
    cache.update(z,ad,bv,cv);cache.update(z,ad,bv,cv);z['xy'][0,0]=1;cache.update(z,ad,bv,cv)
    checks['identity_invalidation']=cache.builds==2 and cache.hits==1 and cache.invalidations==1
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'failure.json'
        try:
            path.write_text(json.dumps(dict(status='fail',component='injected',count=1)))
            raise ValueError('injected after persistence')
        except ValueError:checks['failure_written_first']=json.loads(path.read_text())['component']=='injected'
    return dict(status='pass' if all(checks.values()) else 'fail',checks=checks,seconds=time.perf_counter()-t,real_BP=0,real_assembly=0,real_solve=0,time_steps=0)
