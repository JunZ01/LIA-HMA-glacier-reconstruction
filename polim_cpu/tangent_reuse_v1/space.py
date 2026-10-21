"""Small exact-operator solution space. No numerical BP operator is changed."""
import torch
VERSION='same-Fu-v50-1'
CAPACITY=8
RANK_REL=1e-12
DEPENDENCE_REL=1e-10

class Space:
    def __init__(self):self.key=None;self.Z=[];self.B=[]
    def bind(self,key):
        changed=key!=self.key
        if changed:self.key=key;self.Z=[];self.B=[]
        return changed
    def guess(self,b,wet,action):
        zero=torch.zeros_like(b);norm=float(b.norm());reason='empty'
        if not self.Z:return zero,dict(reason=reason,rank=0,initial_relative=0. if norm==0 else 1.)
        try:
            B=torch.stack(self.B,1);Z=torch.stack(self.Z,1)
            U,s,Vh=torch.linalg.svd(B,full_matrices=False)
            if not bool(torch.isfinite(s).all()) or float(s[0])<=0:raise ValueError('invalid rank')
            keep=s>s[0]*RANK_REL;rank=int(keep.sum())
            if not rank:raise ValueError('zero rank')
            c=Vh[keep].T@((U[:,keep].T@b)/s[keep]);x=Z@c;x[~wet]=0.
            if not bool(torch.isfinite(x).all()):raise ValueError('nonfinite initial guess')
            residual=b-action(x);rn=float(residual.norm())
            if not bool(torch.isfinite(residual).all()) or rn>norm:return zero,dict(reason='worse-than-zero',rank=rank,initial_relative=rn/norm if norm else None)
            return x,dict(reason='SVD',rank=rank,initial_relative=rn/norm if norm else 0.)
        except (RuntimeError,ValueError) as exc:return zero,dict(reason='invalid-space:'+str(exc),rank=0,initial_relative=1. if norm else 0.)
    def add(self,x,ax,wet):
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(ax).all()):raise ValueError('Nonfinite validated solution')
        x=x.clone();ax=ax.clone();x[~wet]=0.;ax[~wet]=0.;original=float(ax.norm())
        if original==0:return 'zero-action'
        # Two-pass modified Gram-Schmidt in A-image space, applying identical
        # coefficients to Z. Only validated solutions enter this linear span.
        for _ in range(2):
            for z,b in zip(self.Z,self.B):
                c=b@ax;ax-=c*b;x-=c*z
        norm=float(ax.norm())
        if norm<=DEPENDENCE_REL*original:return 'dependent'
        if len(self.Z)==CAPACITY:self.Z.pop(0);self.B.pop(0)
        self.Z.append(x/norm);self.B.append(ax/norm);return 'added'

def solve_initial(cg,rhs,x0,rtol,true_tol,limit,meter,label):
    """Original PCG correction recurrence; stopping denominators use original b."""
    with meter.span('tangent_true_residual'):
        residual=rhs-cg.bp.hvp(x0);norm=rhs.norm();true=float(residual.norm()/norm) if float(norm) else float(residual.norm())
    if true<=true_tol:
        meter.counts['tangent_zero_CG']+=1
        return x0,0,true
    meter.counts['tangent_actual_CG_solves']+=1
    used=0;r=-residual
    if cg.use_graph and cg.graph is None:
        with meter.span('graph_capture'):cg.capture(r)
        used+=8;meter.counts[label+'_Krylov_setup']+=8
    cg.reset(r);den=(cg.mg.d*rhs).norm()
    while used+4<=limit:
        meter.tick();cg.graph.replay();used+=4;meter.counts[label+'_Krylov']+=4
        rel=float(cg.rr.norm()/den)
        if bool(cg.bad>0):raise RuntimeError('Nonpositive CG curvature')
        if rel<=rtol:
            x=x0+cg.mg.d*cg.y
            with meter.span('tangent_true_residual'):true=float((cg.bp.hvp(x)-rhs).norm()/norm)
            if true<=true_tol:return x,used,true
    x=x0+cg.mg.d*cg.y
    with meter.span('tangent_true_residual'):true=float((cg.bp.hvp(x)-rhs).norm()/norm)
    raise RuntimeError(f'tangent CG budget exhausted {used}/{limit}, true={true}')
