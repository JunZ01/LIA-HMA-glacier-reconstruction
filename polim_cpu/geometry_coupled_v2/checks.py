"""Only small algebra checks for the coupled velocity candidate repair."""
import time
from types import SimpleNamespace
import numpy as np
from .velocity_candidate import VelocityCandidate

def run():
    start=time.perf_counter();checks={}
    # Two thickness nodes, four free velocity components each. Nonzero F.
    Fu=np.diag(np.arange(1.,9.));FH=np.arange(16,dtype=float).reshape(8,2)/13-.3
    F=np.linspace(.1,.8,8);h=np.array([.2,1.]);u=np.linspace(.2,.9,8);dh=np.array([-1.,.3]);bed=np.zeros(2)
    c=np.linalg.solve(Fu,F);z=-np.linalg.solve(Fu,FH@dh);calls=[0]
    def solve(rhs,label):calls[0]+=1;return np.linalg.solve(Fu,rhs)
    lin=SimpleNamespace(h=h.copy(),u=u.copy(),a={'wet':np.ones(8,dtype=bool)},fh=lambda x:FH@x,solve=solve)
    helper=VelocityCandidate(lin,c,np.zeros(2,dtype=bool),bed);helper.capture(dh,z)
    alpha=.125;guess,actual,pred,info=helper.propose(h+alpha*dh,dh,alpha)
    expected=alpha*(z-c)
    checks['nonzero_F_sign_and_common_damping']=bool(np.allclose(pred,expected,atol=1e-15) and np.allclose(guess,u+expected,atol=1e-15))
    checks['final_J_direction_cache_reused']=bool(info['cache_hit'] and calls[0]==0)
    checks['undamped_F_would_be_detected']=bool(np.linalg.norm(pred-(alpha*z-c))>1e-2)
    alpha=.5;ht=np.maximum(h+alpha*dh,0.);guess,actual,pred,info=helper.propose(ht,dh,alpha)
    expected=-alpha*c-np.linalg.solve(Fu,FH@(ht-h))
    checks['actual_projected_displacement']=bool(np.allclose(pred,expected,atol=1e-15) and np.array_equal(actual,ht-h))
    checks['projected_extra_tangent_counted']=bool(not info['cache_hit'] and calls[0]==1)
    checks['newly_dry_velocity_explicitly_zero']=bool(np.array_equal(guess[:4],np.zeros(4)))
    checks['using_unprojected_direction_detected']=bool(np.linalg.norm(pred-alpha*(z-c))>1e-3)
    # Deliberately mutate only mock geometry, never real files.
    lin.h[0]+=.01
    try:helper.propose(ht,dh,alpha);stale=False
    except RuntimeError:stale=True
    checks['stale_point_cache_rejected']=stale
    lin.h[0]-=.01;helper=VelocityCandidate(lin,c,np.zeros(2,dtype=bool),bed)
    helper.capture(dh,z);helper.obstacle[0]=True
    try:helper.propose(ht,dh,alpha);stale=False
    except RuntimeError:stale=True
    checks['stale_active_set_rejected']=stale
    # Current dry-node direction restriction with a newly wet candidate.
    lin.h=np.array([0.,1.]);lin.u=u.copy();lin.u[:4]=0.;lin.a={'wet':np.array([False]*4+[True]*4)}
    seen=[]
    def fh(v):seen.append(v.copy());return FH@v
    lin.fh=fh;helper=VelocityCandidate(lin,c,np.zeros(2,dtype=bool),bed)
    helper.propose(np.array([.1,1.1]),np.array([.2,.2]),.5)
    checks['newly_wet_excluded_from_old_FH_branch']=bool(seen[-1][0]==0)
    return dict(status='pass' if all(checks.values()) else 'fail',checks=checks,seconds=time.perf_counter()-start,
        real_BP=0,real_tangent=0,old_synthetic_and_real_probe_rerun=False)
