"""One candidate: suppress redundant rebuilds at the exact same true Fu."""
from polim_cpu.cn_cuda_v1.device import Engine as BaseEngine,fh,tensor,torch,np,T,CG,ProjectedMultigrid
from .policy import SOLVER,VERSION

class Engine(BaseEngine):
    def __init__(self,z,root,prepare_only=False,strategy='baseline',replay=False):
        super().__init__(z,root,prepare_only);assert strategy in ['baseline','fixed-Fu']
        self.strategy=strategy;self.replay=replay;self.built_h=None;self.built_u=None
        import hashlib
        self.bp.configuration=hashlib.sha256((self.bp.configuration+VERSION).encode()).hexdigest()
    def full(self,h,u):
        if self.replay:raise RuntimeError('Replay contract forbids complete BP equilibrium')
        return super().full(h,u)
    def ensure_preconditioner(self):
        wet=self.bp.wet;changed=0 if self.build_wet is None else int((wet!=self.build_wet).sum())
        same=(self.built_h is not None and torch.equal(self.built_h.view(torch.int64),self.bp.h_copy.view(torch.int64))
            and torch.equal(self.built_u.view(torch.int64),self.bp.matrix_u.view(torch.int64)))
        pending=self.rebuild_pending
        if self.strategy=='fixed-Fu' and same and self.mg is not None and changed==0:self.rebuild_pending=False
        before=self.builds;super().ensure_preconditioner()
        if self.builds!=before:
            self.built_h=self.bp.h_copy.clone();self.built_u=self.bp.matrix_u.clone()
        elif pending:
            self.rebuild_pending=True # Defer deterioration trigger until a changed Fu, never lose it.
        self.m.events.append(dict(event='preconditioner-decision',strategy=self.strategy,rebuild=self.builds!=before,
            reason='cold' if before==0 else 'wet-change' if changed>SOLVER['rebuild_wet_fraction']*len(wet) else 'previous-CG-over96' if self.builds!=before else 'deferred-same-Fu' if pending and same and self.strategy=='fixed-Fu' else 'reuse',
            same_build_Fu=same,wet_changes=changed,point=self.state_id))

