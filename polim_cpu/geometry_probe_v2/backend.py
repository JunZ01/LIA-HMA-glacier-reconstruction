"""Future CPU backend. Imported ONLY behind the formal registration gate.

Reuse original BP solve/evaluate and pyamg; wrappers add counters and deadlines.
No fallback, additional BP, timestep or nonlinear tolerance adjustment.
"""
from time import perf_counter
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from polim_cpu.projected_amg_v1 import ProjectedMultigrid


class Backend:
    def __init__(self,z,work):
        from polim_cpu.v13r1 import bp
        self.bp=bp;self.work=work;self.A=z['A']
        self.mesh=dict(xy=z['xy'],triangles=z['tri'],bed=z['bed'])
        self.phase='BP'

    @contextmanager
    def measured(self):
        bp=self.bp;w=self.work
        prepare,evaluate,cg,amg=bp.prepare,bp.evaluate,bp.cg,bp.pyamg.smoothed_aggregation_solver
        def prep(*a,**kw):
            w.enter('geometry_build');t=perf_counter()
            try:return prepare(*a,**kw)
            finally:w.times['geometry_build']+=perf_counter()-t;w.progress()
        def ev(*a,**kw):
            w.enter('residual');t=perf_counter();linear=kw.get('tangent',a[2] if len(a)>2 else False)
            if linear:w.enter('matrix_build')
            try:return evaluate(*a,**kw)
            finally:w.times['residual_and_matrix']+=perf_counter()-t;w.progress()
        def pcg(*a,**kw):
            cb=kw.get('callback')
            def count(x):
                w.enter(self.phase+'_Krylov',200 if self.phase=='tangent' else None)
                if self.phase=='tangent' and w.counts['tangent_Krylov']%20==0:
                    w.npz(f'tangent-iteration-{w.counts["tangent_Krylov"]:03d}.npz',scaled_solution=x)
                if cb:cb(x)
            kw['callback']=count;return cg(*a,**kw)
        def build(*a,**kw):
            w.enter('AMG_build');t=perf_counter()
            try:return amg(*a,**kw)
            finally:w.times['AMG_build']+=perf_counter()-t;w.progress()
        bp.prepare,bp.evaluate,bp.cg,bp.pyamg.smoothed_aggregation_solver=prep,ev,pcg,build
        try:yield
        finally:bp.prepare,bp.evaluate,bp.cg,bp.pyamg.smoothed_aggregation_solver=prepare,evaluate,cg,amg

    def evaluate(self,h,u,tangent=False):
        a=self.bp.prepare({**self.mesh,'thickness':h})
        if np.any(u[~a['wet']]!=0):raise RuntimeError('Saved or returned dry velocity is nonzero; no implicit base repair')
        r,_,K,f=self.bp.evaluate(a,u,tangent,A=self.A)
        return r,K,f,a['wet']

    def full(self,h,u):
        self.work.enter('full_BP',2);t=perf_counter();self.phase='BP'
        try:
            def checkpoint(it,velocity,info):
                self.work.npz(f'BP-{self.work.counts["full_BP"]}-iteration-{it:03d}.npz',u=velocity.reshape(-1))
                self.work.json(f'BP-{self.work.counts["full_BP"]}-iteration-{it:03d}.json',info)
            velocity,info=self.bp.solve({**self.mesh,'thickness':h},initial=u.copy(),A=self.A,tolerance=1e-6,
                max_iterations=80,max_seconds=max(0.,self.work.seconds-(perf_counter()-self.work.start)),direct=False,checkpoint=checkpoint)
            return velocity[:,1:,:].reshape(-1),info
        finally:self.work.times['full_BP']+=perf_counter()-t;self.work.progress()

    def tangent(self,K,rhs,wet):
        self.work.enter('tangent',1);t=perf_counter();self.phase='tangent'
        try:
            diag=K.diagonal()
            if np.any(diag<=0):raise RuntimeError('Nonpositive Fu diagonal')
            d=diag**-.5;D=sparse.diags(d);scaled=(D@K@D).tocsr();modes=np.zeros((len(rhs),4))
            for j in range(4):modes[j::4,j]=1/d[j::4]
            modes/=np.linalg.norm(modes,axis=0)[None,:];np.random.seed(2387)
            ml=self.bp.pyamg.smoothed_aggregation_solver(scaled.tobsr(blocksize=(4,4)),B=modes,
                symmetry='symmetric',max_levels=12,max_coarse=160,
                presmoother=('block_gauss_seidel',{'sweep':'symmetric'}),postsmoother=('block_gauss_seidel',{'sweep':'symmetric'}))
            # The existing projection API acts on the true current wet subspace.
            base=SimpleNamespace(d=d,cycle=lambda x:ml.aspreconditioner()@x)
            projected=ProjectedMultigrid(base,SimpleNamespace(active=wet))
            pre=LinearOperator(K.shape,matvec=projected.cycle)
            y,status=self.bp.cg(scaled,d*rhs,M=pre,rtol=1e-10,atol=0,maxiter=200)
            du=d*y;true=float(np.linalg.norm(K@du-rhs)/max(np.linalg.norm(rhs),1e-300))
            if status or true>1e-8 or not np.isfinite(du).all():raise RuntimeError(f'Tangent failure status={status}, true={true}')
            return du,true
        finally:self.work.times['tangent']+=perf_counter()-t;self.work.progress()
