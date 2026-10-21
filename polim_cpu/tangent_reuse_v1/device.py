"""Retain v48 true Fu/FH and AMG; change tangent initial guess only."""
from polim_cpu.cn_cuda_v1.device import *
from polim_cpu.cn_cuda_v1.device import Engine as OriginalEngine
from .space import Space,solve_initial,VERSION as REUSE_VERSION

class Engine(OriginalEngine):
    def __init__(self,z,root,prepare_only=False,reuse=True):
        super().__init__(z,root,prepare_only);self.space=Space();self.reuse=reuse
    def bind(self,h,u,linear=False):
        result=super().bind(h,u,linear)
        if self.space.bind((self.state_id,self.bp.configuration,REUSE_VERSION)):
            self.m.counts['reuse_space_clear']+=1
        return result
    def full(self,h,u):
        self.space.bind(None)
        return super().full(h,u)
    def cold_preconditioner(self):
        # Registered replay comparison only, never used by annual propagation.
        if self.cg and self.cg.graph:self.cg.graph.reset()
        self.cg=None;self.mg=None;self.build_wet=None;self.rebuild_pending=False
    def tangent(self,rhs):
        if self.prepare_only:raise RuntimeError('Real tangent disabled in preparation')
        if self.step_calls['tangent']>=64:raise RuntimeError('64tangent budget')
        self.step_calls['tangent']+=1
        with self.m.span('tangent'):
            if bool((rhs[~self.bp.wet]!=0).any()):raise ValueError('Nonzero projected RHS')
            x0=torch.zeros_like(rhs)
            if self.reuse:
                with self.m.span('reuse_initial_guess'):x0,info=self.space.guess(rhs,self.bp.wet,self.bp.hvp)
            else:info=dict(reason='zero-baseline',rank=0,initial_relative=1. if bool(rhs.any()) else 0.)
            self.ensure_preconditioner()
            with self.m.span('tangent_Krylov_solve'):v,it,true=solve_initial(self.cg,rhs,x0,1e-10,1e-8,200,self.m,'tangent')
            if it>SOLVER['rebuild_iteration_trigger']:self.rebuild_pending=True
            v[~self.bp.wet]=0.
            with self.m.span('tangent_true_residual'):
                av=self.bp.hvp(v);den=float(rhs.norm());true=float((av-rhs).norm())/(den if den else 1.)
            self.m.events.append(dict(event='linear',kind='tangent',iterations=it,true_relative=true,original_rhs_norm=den,initial=info))
            if not true<=1e-8:raise RuntimeError('Original tangent true residual failed')
            if self.reuse:
                with self.m.span('reuse_space_maintenance'):out=self.space.add(v,av,self.bp.wet)
                self.m.events.append(dict(event='reuse-space',size=len(self.space.Z),update=out,point=self.state_id))
            return v
